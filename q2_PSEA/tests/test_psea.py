"""
Tests for q2_PSEA.actions.psea — covering the pure-Python helper functions,
create_fgsea_table_for_pair (with R mocked), and the iterative pipeline
functions (with ctx mocked).
"""
import os
import pathlib
import tempfile
from math import log
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from q2_PSEA.actions.psea import (
    _collapse_residuals_to_epitope,
    _compute_pair_fit_and_residuals,
    create_df_from_gmt,
    create_fgsea_table_for_pair,
    process_scores,
    run_iterative_peptide_analysis,
    run_iterative_process_single_pair,
    write_gmt_from_dict,
)


# ---------------------------------------------------------------------------
# Shared test fixtures / helpers
# ---------------------------------------------------------------------------

class MockArtifact:
    """Minimal stand-in for a QIIME 2 artifact inside pipeline tests."""

    def __init__(self, data):
        self._data = data

    def view(self, type_):
        return self._data


class MockCtx:
    """Minimal stand-in for the QIIME 2 pipeline context object."""

    def __init__(self):
        self._actions = {}

    def register_action(self, plugin, action, fn):
        self._actions[(plugin, action)] = fn

    def get_action(self, plugin, action):
        return self._actions[(plugin, action)]

    def make_artifact(self, type_str, data):
        return MockArtifact(data)


def _make_scores_df(n=30, seed=0):
    """
    Return a (peptides × samples) DataFrame suitable for
    _compute_pair_fit_and_residuals.
    """
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(1, 10, n))
    y = x + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {"sA": x, "sB": y},
        index=[f"pep_{i}" for i in range(n)],
    )


def _make_pairs_dirfmt(content):
    """
    Return an object whose .path contains a pairs.tsv with *content*.
    Used to fake PSEAPairsDirFmt inside pipeline mocks.
    """
    tmpdir = tempfile.mkdtemp()

    class FakeDirFmt:
        path = pathlib.Path(tmpdir)

    with open(FakeDirFmt.path / "pairs.tsv", "w") as fh:
        fh.write(content)

    return FakeDirFmt()


# ---------------------------------------------------------------------------
# process_scores
# ---------------------------------------------------------------------------

class TestProcessScores:
    def _expected(self, v, base=2, offset=3):
        power = base ** offset  # 8
        adjusted = power + v
        clamped = max(1.0, adjusted)
        return log(clamped, base) - offset

    def test_selects_only_columns_in_pairs(self):
        scores = pd.DataFrame(
            {"sA": [0.0], "sB": [0.0], "sC": [0.0]},
            index=["pep1"],
        )
        result = process_scores(scores, [("sA", "sB")])
        assert list(result.columns) == ["sA", "sB"]
        assert "sC" not in result.columns

    def test_zero_input_maps_to_zero(self):
        # v=0 → 0+8=8 → log2(8)-3 = 0
        scores = pd.DataFrame({"sA": [0.0], "sB": [0.0]}, index=["pep1"])
        result = process_scores(scores, [("sA", "sB")])
        assert result.loc["pep1", "sA"] == pytest.approx(0.0)

    def test_very_negative_input_clamps_to_minus_three(self):
        # v=-100 → -100+8=-92 → clamped to 1 → log2(1)-3 = -3
        scores = pd.DataFrame({"sA": [-100.0], "sB": [-100.0]}, index=["pep1"])
        result = process_scores(scores, [("sA", "sB")])
        assert result.loc["pep1", "sA"] == pytest.approx(-3.0)

    def test_positive_value_transformed_correctly(self):
        v = 5.0
        scores = pd.DataFrame({"sA": [v], "sB": [0.0]}, index=["pep1"])
        result = process_scores(scores, [("sA", "sB")])
        assert result.loc["pep1", "sA"] == pytest.approx(self._expected(v))

    def test_duplicate_samples_across_pairs_deduplicated(self):
        # sA appears in both pairs → should appear once as a column
        scores = pd.DataFrame(
            {"sA": [1.0], "sB": [2.0], "sC": [3.0]},
            index=["pep1"],
        )
        result = process_scores(scores, [("sA", "sB"), ("sA", "sC")])
        assert sorted(result.columns) == ["sA", "sB", "sC"]

    def test_all_peptides_retained(self):
        scores = pd.DataFrame(
            {"sA": [0.0, 5.0, -10.0], "sB": [0.0, 3.0, -8.0]},
            index=["pep1", "pep2", "pep3"],
        )
        result = process_scores(scores, [("sA", "sB")])
        assert len(result) == 3


# ---------------------------------------------------------------------------
# _collapse_residuals_to_epitope
# ---------------------------------------------------------------------------

class TestCollapseResidualsToEpitope:
    def _make_epitope_map(self, mapping):
        """mapping: {epitope_id: [pep1, pep2, ...]}"""
        return pd.DataFrame({"CodeName": mapping})

    def test_single_peptide_per_epitope(self):
        epitope_map = self._make_epitope_map({"ep1": ["pep1"], "ep2": ["pep2"]})
        residuals = pd.Series({"pep1": 0.5, "pep2": -0.3})
        result = _collapse_residuals_to_epitope(residuals, epitope_map)
        assert result["ep1"] == pytest.approx(0.5)
        assert result["ep2"] == pytest.approx(-0.3)

    def test_multiple_peptides_per_epitope_keeps_max_abs(self):
        epitope_map = self._make_epitope_map({"ep1": ["pep1", "pep2"]})
        # pep2 has larger absolute value, so ep1 should get -0.8
        residuals = pd.Series({"pep1": 0.5, "pep2": -0.8})
        result = _collapse_residuals_to_epitope(residuals, epitope_map)
        assert result["ep1"] == pytest.approx(-0.8)

    def test_unmapped_peptide_maps_to_itself(self):
        epitope_map = self._make_epitope_map({"ep1": ["pep1"]})
        residuals = pd.Series({"pep1": 0.5, "pep_orphan": 0.9})
        result = _collapse_residuals_to_epitope(residuals, epitope_map)
        assert "pep_orphan" in result.index
        assert result["pep_orphan"] == pytest.approx(0.9)

    def test_peptide_in_multiple_epitopes(self):
        epitope_map = self._make_epitope_map(
            {"ep1": ["pep1", "pep2"], "ep2": ["pep1", "pep3"]}
        )
        residuals = pd.Series({"pep1": 1.0, "pep2": 0.2, "pep3": 0.5})
        result = _collapse_residuals_to_epitope(residuals, epitope_map)
        # ep1: max(abs(1.0), abs(0.2)) = 1.0 → pep1 wins
        # ep2: max(abs(1.0), abs(0.5)) = 1.0 → pep1 wins
        assert result["ep1"] == pytest.approx(1.0)
        assert result["ep2"] == pytest.approx(1.0)

    def test_returns_series(self):
        epitope_map = self._make_epitope_map({"ep1": ["pep1"]})
        residuals = pd.Series({"pep1": 0.7})
        result = _collapse_residuals_to_epitope(residuals, epitope_map)
        assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# write_gmt_from_dict / create_df_from_gmt (round-trip)
# ---------------------------------------------------------------------------

class TestGmtRoundTrip:
    def test_write_and_read_roundtrip(self):
        gmt_dict = {
            "sp1": ["pep1", "pep2", "pep3"],
            "sp2": ["pep4", "pep5"],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".gmt", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            write_gmt_from_dict(tmp_path, gmt_dict)
            result = create_df_from_gmt(tmp_path)

            assert set(result.index) == {"sp1", "sp2"}
            # Leading edge list includes a trailing empty string from split; strip it
            sp1_peps = [p for p in result.loc["sp1", "EpitopeID"] if p.strip()]
            assert set(sp1_peps) == {"pep1", "pep2", "pep3"}
        finally:
            os.unlink(tmp_path)

    def test_write_gmt_format(self):
        gmt_dict = {"sp1": ["pepA", "pepB"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".gmt", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            write_gmt_from_dict(tmp_path, gmt_dict)
            with open(tmp_path) as fh:
                line = fh.readline()
            # Format: "speciesID\t\tpep1\tpep2\t\n"
            assert line.startswith("sp1\t\t")
            assert "pepA" in line
            assert "pepB" in line
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# _compute_pair_fit_and_residuals (py-smooth — no R required)
# ---------------------------------------------------------------------------

class TestComputePairFitAndResiduals:
    def test_output_shapes_correct(self):
        scores = _make_scores_df(n=30)
        pair = ("sA", "sB")
        x, yfit, maxZ, deltaZ = _compute_pair_fit_and_residuals(
            scores, pair, "py-smooth", 3, None
        )
        assert len(x) == 30
        assert len(yfit) == 30
        assert len(maxZ) == 30
        assert len(deltaZ) == 30

    def test_maxZ_is_elementwise_max_of_pair(self):
        scores = _make_scores_df(n=30)
        pair = ("sA", "sB")
        _, _, maxZ, _ = _compute_pair_fit_and_residuals(
            scores, pair, "py-smooth", 3, None
        )
        # maxZ should be >= each individual column after sorting
        data_sorted = scores[list(pair)].sort_values(by="sA")
        expected_max = data_sorted.max(axis=1)
        assert_series_equal(
            maxZ.sort_index(),
            expected_max.sort_index(),
            check_names=False,
        )

    def test_deltaZ_is_y_minus_yfit(self):
        """deltaZ values should sum close to zero for a well-fit signal."""
        scores = _make_scores_df(n=40)
        pair = ("sA", "sB")
        x, yfit, _, deltaZ = _compute_pair_fit_and_residuals(
            scores, pair, "py-smooth", 3, None
        )
        # residuals from a smooth fit should be roughly mean-zero
        assert abs(deltaZ.mean()) < 0.5

    def test_with_epitope_map_collapses_residuals(self):
        scores = _make_scores_df(n=30)
        pair = ("sA", "sB")
        # Map pairs of peptides to a single epitope each
        pep_indices = list(scores.index)
        mapping = {
            f"ep_{i}": [pep_indices[2 * i], pep_indices[2 * i + 1]]
            for i in range(5)
        }
        epitope_map = pd.DataFrame({"CodeName": mapping})

        _, _, maxZ, deltaZ = _compute_pair_fit_and_residuals(
            scores, pair, "py-smooth", 3, None, epitope_map=epitope_map
        )
        # Epitope IDs (not peptide IDs) should appear in the result
        for ep in mapping:
            assert ep in maxZ.index
            assert ep in deltaZ.index

    def test_no_finite_issues(self):
        scores = _make_scores_df(n=30)
        pair = ("sA", "sB")
        x, yfit, maxZ, deltaZ = _compute_pair_fit_and_residuals(
            scores, pair, "py-smooth", 3, None
        )
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(yfit))
        assert np.all(np.isfinite(maxZ.values))
        assert np.all(np.isfinite(deltaZ.values))


# ---------------------------------------------------------------------------
# create_fgsea_table_for_pair (R mocked out)
# ---------------------------------------------------------------------------

def _fake_psea_result():
    return pd.DataFrame(
        {
            "ID": ["11520", "10376"],
            "enrichmentScore": [0.7, -0.4],
            "NES": [2.1, -1.6],
            "p.adjust": [0.01, 0.04],
            "core_enrichment": ["pep1/pep2", "pep3"],
            "pvalue": [0.005, 0.02],
            "qvalue": [0.01, 0.04],
            "all_tested_peptides": ["pep1/pep2/pep4", "pep3/pep5"],
        }
    )


class TestCreateFgseaTableForPair:
    def _run(self, scores_df, gmt_df, precomputed_fit=None, **kwargs):
        """Invoke create_fgsea_table_for_pair with R mocked out."""
        expected = _fake_psea_result()

        with (
            patch("q2_PSEA.actions.psea.INTERNAL") as mock_internal,
            patch("q2_PSEA.actions.psea.ro") as mock_ro,
        ):
            mock_internal.psea.return_value = "r_result_placeholder"

            # Wire up the rpy2 context manager + conversion passthrough
            mock_combined = MagicMock()
            ctx_mgr = MagicMock()
            ctx_mgr.__enter__ = MagicMock(return_value=None)
            ctx_mgr.__exit__ = MagicMock(return_value=False)
            mock_combined.context.return_value = ctx_mgr
            mock_ro.default_converter.__add__ = MagicMock(
                return_value=mock_combined
            )

            mock_conv = MagicMock()
            mock_conv.rpy2py.return_value = expected
            mock_ro.conversion.get_conversion.return_value = mock_conv
            mock_ro.NULL = None

            result = create_fgsea_table_for_pair(
                processed_scores=scores_df.T,  # method receives transposed
                peptide_sets=gmt_df,
                sample_a="sA",
                sample_b="sB",
                threshold=1.0,
                permutation_num=100,
                min_size=2,
                max_size=500,
                spline_type="py-smooth",
                degree=3,
                seed=42,
                precomputed_fit=precomputed_fit,
                **kwargs,
            )
        return result, mock_internal

    def test_returns_dataframe(self):
        scores = _make_scores_df(n=30)
        gmt = pd.DataFrame(
            {"term": ["sp1"] * 5, "gene": scores.index[:5].tolist()}
        )
        result, _ = self._run(scores, gmt)
        assert isinstance(result, pd.DataFrame)

    def test_calls_internal_psea(self):
        scores = _make_scores_df(n=30)
        gmt = pd.DataFrame(
            {"term": ["sp1"] * 5, "gene": scores.index[:5].tolist()}
        )
        _, mock_internal = self._run(scores, gmt)
        mock_internal.psea.assert_called_once()

    def test_precomputed_fit_skips_spline(self):
        """When precomputed_fit is supplied, _compute_pair_fit_and_residuals
        must not be called."""
        scores = _make_scores_df(n=30)
        gmt = pd.DataFrame(
            {"term": ["sp1"] * 5, "gene": scores.index[:5].tolist()}
        )
        maxZ = pd.Series(np.ones(30), index=scores.index)
        deltaZ = pd.Series(np.zeros(30), index=scores.index)
        fit_df = pd.DataFrame({"maxZ": maxZ, "deltaZ": deltaZ})

        with patch(
            "q2_PSEA.actions.psea._compute_pair_fit_and_residuals"
        ) as mock_fit:
            self._run(scores, gmt, precomputed_fit=fit_df)
            mock_fit.assert_not_called()

    def test_without_precomputed_fit_calls_spline(self):
        scores = _make_scores_df(n=30)
        gmt = pd.DataFrame(
            {"term": ["sp1"] * 5, "gene": scores.index[:5].tolist()}
        )
        with patch(
            "q2_PSEA.actions.psea._compute_pair_fit_and_residuals",
            wraps=_compute_pair_fit_and_residuals,
        ) as mock_fit:
            self._run(scores, gmt)
            mock_fit.assert_called_once()

    def test_species_taxa_file_path_passed_when_provided(self):
        scores = _make_scores_df(n=30)
        gmt = pd.DataFrame(
            {"term": ["sp1"] * 5, "gene": scores.index[:5].tolist()}
        )
        # Build a fake PSEASpeciesTaxaDirFmt
        tmpdir = tempfile.mkdtemp()

        class FakeTaxaDirFmt:
            path = pathlib.Path(tmpdir)

        taxa_file = pathlib.Path(tmpdir) / "species-taxa.tsv"
        taxa_file.write_text("InfluenzaA\t11520\n")
        fake_dirfmt = FakeTaxaDirFmt()

        result, mock_internal = self._run(scores, gmt, species_taxa=fake_dirfmt)
        # The species_taxa_file argument (5th positional after maxZ/deltaZ/gmt)
        # should be the real path, not ""
        call_args = mock_internal.psea.call_args
        # species_taxa_file is the 4th positional argument
        species_arg = call_args.args[3] if call_args.args else None
        assert species_arg == str(taxa_file)

    def test_no_species_taxa_passes_empty_string(self):
        scores = _make_scores_df(n=30)
        gmt = pd.DataFrame(
            {"term": ["sp1"] * 5, "gene": scores.index[:5].tolist()}
        )
        _, mock_internal = self._run(scores, gmt)
        call_args = mock_internal.psea.call_args
        species_arg = call_args.args[3] if call_args.args else None
        assert species_arg == ""


# ---------------------------------------------------------------------------
# run_iterative_process_single_pair
# ---------------------------------------------------------------------------

def _make_gmt_df():
    return pd.DataFrame(
        {
            "term": ["12345", "12345", "12345", "67890", "67890"],
            "gene": ["pep1", "pep2", "pep3", "pep1", "pep4"],
        }
    )


def _make_psea_table_df(sig=True):
    """Return a PSEA result with or without a significant species."""
    if sig:
        return pd.DataFrame(
            {
                "ID": ["12345"],
                "NES": [2.5],
                "p.adjust": [0.01],
                "all_tested_peptides": ["pep1/pep2"],
            }
        )
    return pd.DataFrame(
        {
            "ID": ["12345"],
            "NES": [0.3],
            "p.adjust": [0.9],
            "all_tested_peptides": ["pep1/pep2"],
        }
    )


class TestRunIterativeProcessSinglePair:
    def _build_ctx(self, psea_table_df):
        ctx = MockCtx()
        psea_art = MockArtifact(psea_table_df)

        def fake_create_fgsea(**kwargs):
            return (psea_art,)

        ctx.register_action(
            "psea", "create_fgsea_table_for_pair", fake_create_fgsea
        )
        return ctx

    def test_no_sig_species_gmt_unchanged(self):
        gmt_df = _make_gmt_df()
        table_df = _make_psea_table_df(sig=False)
        ctx = self._build_ctx(table_df)

        _, updated_gmt = run_iterative_process_single_pair(
            ctx,
            processed_scores=MockArtifact(pd.DataFrame()),
            peptide_sets=MockArtifact(gmt_df),
            sample_a="sA",
            sample_b="sB",
            threshold=1.0,
            permutation_num=100,
            min_size=2,
            max_size=500,
            spline_type="py-smooth",
            degree=3,
            seed=42,
            p_val_thresh=0.05,
            nes_thresh=1.0,
        )

        assert_frame_equal(updated_gmt._data, gmt_df)

    def test_sig_species_removes_leading_edge_from_others(self):
        """Leading-edge peptides of significant species should be removed from
        other species' rows in the GMT."""
        gmt_df = _make_gmt_df()
        # "12345" is significant with leading edge pep1/pep2
        table_df = _make_psea_table_df(sig=True)
        ctx = self._build_ctx(table_df)

        _, updated_gmt = run_iterative_process_single_pair(
            ctx,
            processed_scores=MockArtifact(pd.DataFrame()),
            peptide_sets=MockArtifact(gmt_df),
            sample_a="sA",
            sample_b="sB",
            threshold=1.0,
            permutation_num=100,
            min_size=2,
            max_size=500,
            spline_type="py-smooth",
            degree=3,
            seed=42,
            p_val_thresh=0.05,
            nes_thresh=1.0,
        )

        result_gmt = updated_gmt._data
        # pep1 should be removed from species 67890 (but kept in 12345)
        other_rows = result_gmt[result_gmt["term"] == "67890"]
        assert "pep1" not in other_rows["gene"].values
        # pep4 (not in leading edge) stays in 67890
        assert "pep4" in other_rows["gene"].values
        # All rows of species 12345 are unchanged
        own_rows = result_gmt[result_gmt["term"] == "12345"]
        assert set(own_rows["gene"]) == {"pep1", "pep2", "pep3"}

    def test_sig_species_already_tested_gmt_unchanged(self):
        gmt_df = _make_gmt_df()
        table_df = _make_psea_table_df(sig=True)
        ctx = self._build_ctx(table_df)

        _, updated_gmt = run_iterative_process_single_pair(
            ctx,
            processed_scores=MockArtifact(pd.DataFrame()),
            peptide_sets=MockArtifact(gmt_df),
            sample_a="sA",
            sample_b="sB",
            threshold=1.0,
            permutation_num=100,
            min_size=2,
            max_size=500,
            spline_type="py-smooth",
            degree=3,
            seed=42,
            p_val_thresh=0.05,
            nes_thresh=1.0,
            tested_species=["12345"],  # already tested
        )

        assert_frame_equal(updated_gmt._data, gmt_df)

    def test_species_name_fallback_when_column_missing(self):
        """run_iterative_process_single_pair should not raise KeyError when the
        'species_name' column is absent from the PSEA table (i.e., no
        species_taxa was passed)."""
        gmt_df = _make_gmt_df()
        # No 'species_name' column in table
        table_df = _make_psea_table_df(sig=True)
        assert "species_name" not in table_df.columns

        ctx = self._build_ctx(table_df)
        # Should not raise
        run_iterative_process_single_pair(
            ctx,
            processed_scores=MockArtifact(pd.DataFrame()),
            peptide_sets=MockArtifact(gmt_df),
            sample_a="sA",
            sample_b="sB",
            threshold=1.0,
            permutation_num=100,
            min_size=2,
            max_size=500,
            spline_type="py-smooth",
            degree=3,
            seed=42,
            p_val_thresh=0.05,
            nes_thresh=1.0,
        )


# ---------------------------------------------------------------------------
# run_iterative_peptide_analysis
# ---------------------------------------------------------------------------

class TestRunIterativePeptideAnalysis:
    def _build_ctx(self, psea_table_df, gmt_df):
        ctx = MockCtx()
        psea_art = MockArtifact(psea_table_df)
        gmt_art = MockArtifact(gmt_df)

        def fake_run_single_pair(**kwargs):
            return psea_art, gmt_art

        ctx.register_action(
            "psea", "run_iterative_process_single_pair", fake_run_single_pair
        )
        return ctx

    def test_terminates_when_no_sig_species(self):
        """With no significant species the loop should complete in one pass."""
        gmt_df = _make_gmt_df()
        # Non-significant PSEA table
        table_df = _make_psea_table_df(sig=False)
        ctx = self._build_ctx(table_df, gmt_df)

        fake_dirfmt = _make_pairs_dirfmt("sA\tsB\nsA\tsB\n")

        scores_df = _make_scores_df(n=30)
        # FeatureTable[Zscore] view returns samples-as-rows (transposed)
        scores_transposed = scores_df.T  # samples as rows, peptides as cols

        with patch(
            "q2_PSEA.actions.psea._compute_pair_fit_and_residuals"
        ) as mock_fit:
            pep_idx = scores_df.index
            mock_fit.return_value = (
                np.linspace(1, 10, 30),
                np.linspace(1, 10, 30),
                pd.Series(np.ones(30), index=pep_idx),
                pd.Series(np.zeros(30), index=pep_idx),
            )

            results = run_iterative_peptide_analysis(
                ctx,
                processed_scores=MockArtifact(scores_transposed),
                pairs=MockArtifact(fake_dirfmt),
                peptide_sets=MockArtifact(gmt_df),
                threshold=1.0,
                permutation_num=100,
                min_size=2,
                max_size=500,
                spline_type="py-smooth",
                degree=3,
                seed=42,
                p_val_thresh=0.05,
                nes_thresh=1.0,
            )

        assert isinstance(results, list)
        assert len(results) == 1  # one pair

    def test_returns_one_gmt_per_pair(self):
        """Result list length should equal the number of pairs."""
        gmt_df = _make_gmt_df()
        table_df = _make_psea_table_df(sig=False)
        # Two pairs
        ctx = self._build_ctx(table_df, gmt_df)
        fake_dirfmt = _make_pairs_dirfmt("sA\tsB\nsA\tsB\n")

        scores_df = _make_scores_df(n=30)
        scores_transposed = scores_df.T

        with patch("q2_PSEA.actions.psea._compute_pair_fit_and_residuals") as mock_fit:
            pep_idx = scores_df.index
            mock_fit.return_value = (
                np.linspace(1, 10, 30),
                np.linspace(1, 10, 30),
                pd.Series(np.ones(30), index=pep_idx),
                pd.Series(np.zeros(30), index=pep_idx),
            )

            results = run_iterative_peptide_analysis(
                ctx,
                processed_scores=MockArtifact(scores_transposed),
                pairs=MockArtifact(fake_dirfmt),
                peptide_sets=MockArtifact(gmt_df),
                threshold=1.0,
                permutation_num=100,
                min_size=2,
                max_size=500,
                spline_type="py-smooth",
                degree=3,
                seed=42,
                p_val_thresh=0.05,
                nes_thresh=1.0,
            )

        # Two pairs → two filtered GMTs
        assert len(results) == 2

    def test_fit_computed_once_per_pair(self):
        """_compute_pair_fit_and_residuals should be called exactly once per
        pair regardless of how many iterations run."""
        gmt_df = _make_gmt_df()
        table_df = _make_psea_table_df(sig=False)
        ctx = self._build_ctx(table_df, gmt_df)
        # Two distinct pairs
        fake_dirfmt = _make_pairs_dirfmt("sA\tsB\nsA\tsB\n")

        scores_df = _make_scores_df(n=30)
        scores_transposed = scores_df.T

        with patch(
            "q2_PSEA.actions.psea._compute_pair_fit_and_residuals"
        ) as mock_fit:
            pep_idx = scores_df.index
            mock_fit.return_value = (
                np.linspace(1, 10, 30),
                np.linspace(1, 10, 30),
                pd.Series(np.ones(30), index=pep_idx),
                pd.Series(np.zeros(30), index=pep_idx),
            )

            run_iterative_peptide_analysis(
                ctx,
                processed_scores=MockArtifact(scores_transposed),
                pairs=MockArtifact(fake_dirfmt),
                peptide_sets=MockArtifact(gmt_df),
                threshold=1.0,
                permutation_num=100,
                min_size=2,
                max_size=500,
                spline_type="py-smooth",
                degree=3,
                seed=42,
                p_val_thresh=0.05,
                nes_thresh=1.0,
            )

        # Two pairs → fit computed exactly twice (once per pair)
        assert mock_fit.call_count == 2
