import os
import pathlib
import tempfile
import unittest
from math import log
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import qiime2
from pandas.testing import assert_frame_equal, assert_series_equal
from qiime2.plugin.testing import TestPluginBase

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
# Pipeline test helpers — kept local, not shared via conftest
# ---------------------------------------------------------------------------

class _MockArtifact:
    """Minimal stand-in for a QIIME 2 artifact inside mocked pipeline tests."""

    def __init__(self, data):
        self._data = data

    def view(self, type_):
        return self._data


class _MockCtx:
    """Minimal stand-in for the QIIME 2 pipeline context object."""

    def __init__(self):
        self._actions = {}

    def register_action(self, plugin, action, fn):
        self._actions[(plugin, action)] = fn

    def get_action(self, plugin, action):
        return self._actions[(plugin, action)]

    def make_artifact(self, type_str, data):
        return _MockArtifact(data)


def _mock_ro_context(mock_ro, return_df):
    """Wire mock_ro so the rpy2 conversion block passes *return_df* through."""
    mock_combined = MagicMock()
    ctx_mgr = MagicMock()
    ctx_mgr.__enter__ = MagicMock(return_value=None)
    ctx_mgr.__exit__ = MagicMock(return_value=False)
    mock_combined.context.return_value = ctx_mgr
    mock_ro.default_converter.__add__ = MagicMock(return_value=mock_combined)
    mock_ro.NULL = None
    mock_conv = MagicMock()
    mock_conv.rpy2py.return_value = return_df
    mock_ro.conversion.get_conversion.return_value = mock_conv


# ---------------------------------------------------------------------------
# process_scores
# ---------------------------------------------------------------------------

class TestProcessScores(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.scores = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )

    def _expected(self, v, base=2, offset=3):
        return log(max(1.0, base ** offset + v), base) - offset

    def test_selects_only_columns_in_pairs(self):
        result = process_scores(self.scores, [("sA", "sB")])
        self.assertEqual(set(result.columns), {"sA", "sB"})
        self.assertNotIn("sC", result.columns)

    def test_zero_input_maps_to_zero(self):
        scores = pd.DataFrame({"sA": [0.0], "sB": [0.0]}, index=["p"])
        result = process_scores(scores, [("sA", "sB")])
        self.assertAlmostEqual(result.loc["p", "sA"], 0.0)

    def test_very_negative_input_clamps_to_minus_three(self):
        scores = pd.DataFrame({"sA": [-100.0], "sB": [-100.0]}, index=["p"])
        result = process_scores(scores, [("sA", "sB")])
        self.assertAlmostEqual(result.loc["p", "sA"], -3.0)

    def test_known_positive_value_transformed_correctly(self):
        v = 5.0
        scores = pd.DataFrame({"sA": [v], "sB": [0.0]}, index=["p"])
        result = process_scores(scores, [("sA", "sB")])
        self.assertAlmostEqual(result.loc["p", "sA"], self._expected(v))

    def test_duplicate_samples_across_pairs_deduplicated(self):
        result = process_scores(self.scores, [("sA", "sB"), ("sA", "sC")])
        self.assertEqual(sorted(result.columns), ["sA", "sB", "sC"])

    def test_all_peptides_retained(self):
        result = process_scores(self.scores, [("sA", "sB")])
        self.assertEqual(len(result), len(self.scores))


# ---------------------------------------------------------------------------
# _collapse_residuals_to_epitope
# ---------------------------------------------------------------------------

class TestCollapseResidualsToEpitope(TestPluginBase):
    package = "q2_PSEA.tests"

    def _emap(self, mapping):
        return pd.DataFrame({"CodeName": mapping})

    def test_single_peptide_per_epitope(self):
        emap = self._emap({"ep1": ["pep1"], "ep2": ["pep2"]})
        res = pd.Series({"pep1": 0.5, "pep2": -0.3})
        result = _collapse_residuals_to_epitope(res, emap)
        self.assertAlmostEqual(result["ep1"], 0.5)
        self.assertAlmostEqual(result["ep2"], -0.3)

    def test_multiple_peptides_keeps_max_abs(self):
        emap = self._emap({"ep1": ["pep1", "pep2"]})
        res = pd.Series({"pep1": 0.5, "pep2": -0.8})
        result = _collapse_residuals_to_epitope(res, emap)
        self.assertAlmostEqual(result["ep1"], -0.8)

    def test_unmapped_peptide_maps_to_itself(self):
        emap = self._emap({"ep1": ["pep1"]})
        res = pd.Series({"pep1": 0.5, "pep_orphan": 0.9})
        result = _collapse_residuals_to_epitope(res, emap)
        self.assertIn("pep_orphan", result.index)
        self.assertAlmostEqual(result["pep_orphan"], 0.9)

    def test_peptide_in_multiple_epitopes(self):
        emap = self._emap({"ep1": ["pep1", "pep2"], "ep2": ["pep1", "pep3"]})
        res = pd.Series({"pep1": 1.0, "pep2": 0.2, "pep3": 0.5})
        result = _collapse_residuals_to_epitope(res, emap)
        self.assertAlmostEqual(result["ep1"], 1.0)
        self.assertAlmostEqual(result["ep2"], 1.0)

    def test_returns_series(self):
        emap = self._emap({"ep1": ["pep1"]})
        result = _collapse_residuals_to_epitope(pd.Series({"pep1": 0.7}), emap)
        self.assertIsInstance(result, pd.Series)


# ---------------------------------------------------------------------------
# write_gmt_from_dict / create_df_from_gmt
# ---------------------------------------------------------------------------

class TestGmtRoundTrip(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_create_df_from_gmt_reads_fixture(self):
        result = create_df_from_gmt(self.get_data_path("gmt.gmt"))
        self.assertEqual(set(result.index), {"sp1", "sp2"})

    def test_create_df_reads_correct_peptides(self):
        result = create_df_from_gmt(self.get_data_path("gmt.gmt"))
        sp1_peps = [
            p.strip() for p in result.loc["sp1", "EpitopeID"] if p.strip()
        ]
        self.assertEqual(
            set(sp1_peps),
            {"pep_00", "pep_01", "pep_02", "pep_03"}
        )

    def test_write_then_read_roundtrip(self):
        gmt_dict = {"sp1": ["pep_00", "pep_01"], "sp2": ["pep_04"]}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".gmt", delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            write_gmt_from_dict(tmp_path, gmt_dict)
            result = create_df_from_gmt(tmp_path)
            sp1_peps = [p for p in result.loc["sp1", "EpitopeID"] if p.strip()]
            self.assertEqual(set(sp1_peps), {"pep_00", "pep_01"})
        finally:
            os.unlink(tmp_path)

    def test_write_gmt_line_format(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".gmt", delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            write_gmt_from_dict(tmp_path, {"sp1": ["pepA", "pepB"]})
            with open(tmp_path) as fh:
                line = fh.readline()
            self.assertTrue(line.startswith("sp1\t\t"))
            self.assertIn("pepA", line)
            self.assertIn("pepB", line)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# _compute_pair_fit_and_residuals (py-smooth — no R required)
# ---------------------------------------------------------------------------

class TestComputePairFitAndResiduals(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.scores = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        self.pair = ("sA", "sB")

    def test_output_shapes_correct(self):
        x, yfit, maxZ, deltaZ = _compute_pair_fit_and_residuals(
            self.scores, self.pair, "py-smooth", 3, None
        )
        n = len(self.scores)
        self.assertEqual(len(x), n)
        self.assertEqual(len(yfit), n)
        self.assertEqual(len(maxZ), n)
        self.assertEqual(len(deltaZ), n)

    def test_maxZ_equals_elementwise_max(self):
        _, _, maxZ, _ = _compute_pair_fit_and_residuals(
            self.scores, self.pair, "py-smooth", 3, None
        )
        data_sorted = self.scores[list(self.pair)].sort_values(by="sA")
        expected = data_sorted.max(axis=1)
        assert_series_equal(
            maxZ.sort_index(), expected.sort_index(), check_names=False
        )

    def test_residuals_roughly_mean_zero(self):
        _, _, _, deltaZ = _compute_pair_fit_and_residuals(
            self.scores, self.pair, "py-smooth", 3, None
        )
        self.assertLess(abs(float(deltaZ.mean())), 0.5)

    def test_no_nan_or_inf(self):
        x, yfit, maxZ, deltaZ = _compute_pair_fit_and_residuals(
            self.scores, self.pair, "py-smooth", 3, None
        )
        self.assertTrue(np.all(np.isfinite(x)))
        self.assertTrue(np.all(np.isfinite(yfit)))
        self.assertTrue(np.all(np.isfinite(maxZ.values)))
        self.assertTrue(np.all(np.isfinite(deltaZ.values)))

    def test_epitope_map_collapses_to_epitope_ids(self):
        peps = list(self.scores.index)
        mapping = {f"ep_{i}": [peps[2 * i], peps[2 * i + 1]] for i in range(5)}
        emap = pd.DataFrame({"CodeName": mapping})
        _, _, maxZ, deltaZ = _compute_pair_fit_and_residuals(
            self.scores, self.pair, "py-smooth", 3, None, epitope_map=emap
        )
        for ep in mapping:
            self.assertIn(ep, maxZ.index)
            self.assertIn(ep, deltaZ.index)


# ---------------------------------------------------------------------------
# create_fgsea_table_for_pair (R mocked)
# ---------------------------------------------------------------------------

def _fake_psea_result():
    return pd.DataFrame(
        {
            "ID": ["11520", "10376"],
            "enrichmentScore": [0.7, -0.4],
            "NES": [2.1, -1.6],
            "p.adjust": [0.01, 0.04],
            "core_enrichment": ["pep_00/pep_01", "pep_04"],
            "pvalue": [0.005, 0.02],
            "qvalue": [0.01, 0.04],
            "all_tested_peptides": ["pep_00/pep_01/pep_02", "pep_04/pep_05"],
        }
    )


class TestCreateFgseaTableForPair(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.scores = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        self.gmt = pd.read_csv(
            self.get_data_path("peptide-sets.tsv"), sep="\t"
        )

    def _call(self, precomputed_fit=None, **kwargs):
        """Invoke create_fgsea_table_for_pair with R fully mocked."""
        expected = _fake_psea_result()
        with (
            patch("q2_PSEA.actions.psea.INTERNAL") as mock_internal,
            patch("q2_PSEA.actions.psea.ro") as mock_ro,
        ):
            mock_internal.psea.return_value = "r_placeholder"
            _mock_ro_context(mock_ro, expected)
            result = create_fgsea_table_for_pair(
                processed_scores=self.scores.T,
                peptide_sets=self.gmt,
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
        result, _ = self._call()
        self.assertIsInstance(result, pd.DataFrame)

    def test_calls_internal_psea(self):
        _, mock_internal = self._call()
        mock_internal.psea.assert_called_once()

    def test_precomputed_fit_skips_spline(self):
        fit_df = pd.DataFrame(
            {
                "maxZ": pd.Series(np.ones(15), index=self.scores.index),
                "deltaZ": pd.Series(np.zeros(15), index=self.scores.index),
            }
        )
        with patch(
            "q2_PSEA.actions.psea._compute_pair_fit_and_residuals"
        ) as mock_fit:
            self._call(precomputed_fit=fit_df)
            mock_fit.assert_not_called()

    def test_no_precomputed_fit_calls_spline(self):
        with patch(
            "q2_PSEA.actions.psea._compute_pair_fit_and_residuals",
            wraps=_compute_pair_fit_and_residuals,
        ) as mock_fit:
            self._call()
            mock_fit.assert_called_once()

    def test_no_species_taxa_passes_empty_string_to_r(self):
        _, mock_internal = self._call()
        species_arg = mock_internal.psea.call_args.args[3]
        self.assertEqual(species_arg, "")

    def test_species_taxa_passes_file_path_to_r(self):
        taxa_md = qiime2.Metadata.load(self.get_data_path("species-taxa.tsv"))
        _, mock_internal = self._call(species_taxa=taxa_md)
        species_arg = mock_internal.psea.call_args.args[3]
        self.assertNotEqual(species_arg, "")


# ---------------------------------------------------------------------------
# run_iterative_process_single_pair
# ---------------------------------------------------------------------------

def _gmt_df():
    return pd.DataFrame(
        {
            "term": ["12345", "12345", "12345", "67890", "67890"],
            "gene": ["pep1", "pep2", "pep3", "pep1", "pep4"],
        }
    )


def _psea_table_df(sig=True):
    return pd.DataFrame(
        {
            "ID": ["12345"],
            "NES": [2.5 if sig else 0.3],
            "p.adjust": [0.01 if sig else 0.9],
            "all_tested_peptides": ["pep1/pep2"],
        }
    )


class TestRunIterativeProcessSinglePair(TestPluginBase):
    package = "q2_PSEA.tests"

    def _build_ctx(self, psea_table_df):
        ctx = _MockCtx()
        art = _MockArtifact(psea_table_df)
        ctx.register_action(
            "psea", "create_fgsea_table_for_pair", lambda **kw: (art,)
        )
        return ctx

    def _run(self, table_df, gmt_df, **extra):
        ctx = self._build_ctx(table_df)
        return run_iterative_process_single_pair(
            ctx,
            processed_scores=_MockArtifact(pd.DataFrame()),
            peptide_sets=_MockArtifact(gmt_df),
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
            **extra,
        )

    def test_no_sig_species_gmt_unchanged(self):
        gmt = _gmt_df()
        _, updated = self._run(_psea_table_df(sig=False), gmt)
        assert_frame_equal(updated._data, gmt)

    def test_sig_species_removes_leading_edge_from_other_species(self):
        _, updated = self._run(_psea_table_df(sig=True), _gmt_df())
        result = updated._data
        other = result[result["term"] == "67890"]
        # pep1 is in the leading edge of 12345 → removed from 67890
        self.assertNotIn("pep1", other["gene"].values)
        # pep4 is not in the leading edge → stays in 67890
        self.assertIn("pep4", other["gene"].values)
        # All of 12345's peptides are untouched
        own = result[result["term"] == "12345"]
        self.assertEqual(set(own["gene"]), {"pep1", "pep2", "pep3"})

    def test_already_tested_species_gmt_unchanged(self):
        gmt = _gmt_df()
        _, updated = self._run(
            _psea_table_df(sig=True), gmt, tested_species=["12345"]
        )
        assert_frame_equal(updated._data, gmt)

    def test_no_species_name_column_does_not_raise(self):
        """Without species_taxa the table has no 'species_name' column;
        the code must fall back to row['ID'] without KeyError."""
        table_df = _psea_table_df(sig=True)
        self.assertNotIn("species_name", table_df.columns)
        self._run(table_df, _gmt_df())  # must not raise


# ---------------------------------------------------------------------------
# run_iterative_peptide_analysis
# ---------------------------------------------------------------------------

class TestRunIterativePeptideAnalysis(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.scores = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        self.gmt = pd.read_csv(
            self.get_data_path("peptide-sets.tsv"), sep="\t"
        )

    def _make_pairs_dirfmt(self, pairs_content):
        """Write *pairs_content* as pairs.tsv in a temp dir and return a
        fake PSEAPairsDirFmt whose .path points there.  The function under
        test hard-codes 'pairs.tsv' as the filename, so we must name it that
        regardless of what content it holds."""
        import shutil
        tmpdir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmpdir)
        with open(os.path.join(tmpdir, "pairs.tsv"), "w") as fh:
            fh.write(pairs_content)

        class FakeDirFmt:
            path = pathlib.Path(tmpdir)

        return FakeDirFmt()

    def _build_ctx(self, table_df):
        ctx = _MockCtx()
        psea_art = _MockArtifact(table_df)
        gmt_art = _MockArtifact(self.gmt)
        ctx.register_action(
            "psea",
            "run_iterative_process_single_pair",
            lambda **kw: (psea_art, gmt_art),
        )
        return ctx

    def _mock_fit_return(self):
        n = len(self.scores)
        pep_idx = self.scores.index
        return (
            np.linspace(1, 8, n),
            np.linspace(1, 8, n),
            pd.Series(np.ones(n), index=pep_idx),
            pd.Series(np.zeros(n), index=pep_idx),
        )

    _ONE_PAIR = "sA\tsB\nsample1\tsample2\n"
    _TWO_PAIRS = "sA\tsB\nsample1\tsample2\nsample3\tsample4\n"

    def _run(self, ctx, pairs_content):
        fake_dirfmt = self._make_pairs_dirfmt(pairs_content)
        with patch(
            "q2_PSEA.actions.psea._compute_pair_fit_and_residuals"
        ) as mock_fit:
            mock_fit.return_value = self._mock_fit_return()
            result = run_iterative_peptide_analysis(
                ctx,
                processed_scores=_MockArtifact(self.scores.T),
                pairs=_MockArtifact(fake_dirfmt),
                peptide_sets=_MockArtifact(self.gmt),
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
        return result, mock_fit

    def test_single_pair_returns_one_gmt(self):
        ctx = self._build_ctx(_psea_table_df(sig=False))
        result, _ = self._run(ctx, self._ONE_PAIR)
        self.assertEqual(len(result), 1)

    def test_two_distinct_pairs_returns_two_gmts(self):
        ctx = self._build_ctx(_psea_table_df(sig=False))
        result, _ = self._run(ctx, self._TWO_PAIRS)
        self.assertEqual(len(result), 2)

    def test_fit_computed_once_per_distinct_pair(self):
        ctx = self._build_ctx(_psea_table_df(sig=False))
        _, mock_fit = self._run(ctx, self._TWO_PAIRS)
        self.assertEqual(mock_fit.call_count, 2)

    def test_returns_list(self):
        ctx = self._build_ctx(_psea_table_df(sig=False))
        result, _ = self._run(ctx, self._ONE_PAIR)
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
