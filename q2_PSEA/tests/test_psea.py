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
    count_antibody_events,
    create_df_from_gmt,
    create_fgsea_table_for_pair,
    process_scores,
    run_iterative_peptide_analysis,
    run_iterative_process_single_pair,
    write_gmt_from_dict,
    make_psea_table,
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

    def values(self):
        """Support Collection-like iteration without going through view()."""
        return self._data.values()


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
# Shared fake PSEA result
# ---------------------------------------------------------------------------

def _fake_psea_df():
    """Minimal PSEA result DataFrame — non-significant so AE counts stay 0."""
    return pd.DataFrame({
        "ID": ["sp1"],
        "enrichmentScore": [0.5],
        "NES": [0.3],
        "p.adjust": [0.9],
        "core_enrichment": ["pep_00"],
        "pvalue": [0.5],
        "qvalue": [0.9],
        "all_tested_peptides": ["pep_00/pep_01"],
    })


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
        # Function expects samples×features (QIIME 2 FeatureTable orientation)
        self.scores_q2 = self.scores.T

    def _call(self, **kwargs):
        return _compute_pair_fit_and_residuals(
            self.scores_q2, "sA", "sB", "py-smooth", 3, **kwargs
        )

    def test_returns_dataframe_with_required_columns(self):
        df = self._call()
        self.assertIsInstance(df, pd.DataFrame)
        for col in ("x", "yfit", "maxZ", "deltaZ"):
            self.assertIn(col, df.columns)

    def test_output_shapes_correct(self):
        df = self._call()
        n = len(self.scores)
        self.assertEqual(len(df["x"].dropna()), n)
        self.assertEqual(len(df["yfit"].dropna()), n)
        self.assertEqual(len(df["maxZ"].dropna()), n)
        self.assertEqual(len(df["deltaZ"].dropna()), n)

    def test_maxZ_equals_elementwise_max(self):
        df = self._call()
        maxZ = df["maxZ"].dropna()
        data_sorted = self.scores[["sA", "sB"]].sort_values(by="sA")
        expected = data_sorted.max(axis=1)
        assert_series_equal(
            maxZ.sort_index(), expected.sort_index(), check_names=False
        )

    def test_residuals_roughly_mean_zero(self):
        df = self._call()
        deltaZ = df["deltaZ"].dropna()
        self.assertLess(abs(float(deltaZ.mean())), 0.5)

    def test_no_nan_or_inf(self):
        df = self._call()
        self.assertTrue(np.all(np.isfinite(df["x"].dropna().to_numpy())))
        self.assertTrue(np.all(np.isfinite(df["yfit"].dropna().to_numpy())))
        self.assertTrue(np.all(np.isfinite(df["maxZ"].dropna().values)))
        self.assertTrue(np.all(np.isfinite(df["deltaZ"].dropna().values)))

    def test_epitope_map_collapses_to_epitope_ids(self):
        peps = list(self.scores.index)
        mapping = {f"ep_{i}": [peps[2 * i], peps[2 * i + 1]] for i in range(5)}
        emap = pd.DataFrame({"CodeName": mapping})
        df = self._call(epitope_map=emap)
        maxZ = df["maxZ"].dropna()
        deltaZ = df["deltaZ"].dropna()
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
        n = len(self.scores)
        idx = self.scores.index
        fit_df = pd.DataFrame({
            "x": np.linspace(0, 1, n),
            "yfit": np.linspace(0, 1, n),
            "maxZ": pd.Series(np.ones(n), index=idx),
            "deltaZ": pd.Series(np.zeros(n), index=idx),
        }, index=idx)
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

    def _make_spline_art(self):
        n = len(self.scores)
        idx = self.scores.index
        df = pd.DataFrame({
            "x": np.linspace(1, 8, n),
            "yfit": np.linspace(1, 8, n),
            "maxZ": np.ones(n),
            "deltaZ": np.zeros(n),
        }, index=idx)
        return _MockArtifact(df)

    def _build_ctx(self, table_df):
        ctx = _MockCtx()
        psea_art = _MockArtifact(table_df)
        gmt_art = _MockArtifact(self.gmt)
        spline_art = self._make_spline_art()
        ctx.register_action(
            "psea",
            "_compute_pair_fit_and_residuals",
            lambda **kw: (spline_art,),
        )
        ctx.register_action(
            "psea",
            "run_iterative_process_single_pair",
            lambda **kw: (psea_art, gmt_art),
        )
        return ctx

    _ONE_PAIR = "sA\tsB\nsample1\tsample2\n"
    _TWO_PAIRS = "sA\tsB\nsample1\tsample2\nsample3\tsample4\n"

    def _run(self, ctx, pairs_content):
        fake_dirfmt = self._make_pairs_dirfmt(pairs_content)
        return run_iterative_peptide_analysis(
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

    def test_single_pair_returns_one_gmt(self):
        ctx = self._build_ctx(_psea_table_df(sig=False))
        result = self._run(ctx, self._ONE_PAIR)
        self.assertEqual(len(result), 1)

    def test_two_distinct_pairs_returns_two_gmts(self):
        ctx = self._build_ctx(_psea_table_df(sig=False))
        result = self._run(ctx, self._TWO_PAIRS)
        self.assertEqual(len(result), 2)

    def test_fit_computed_once_per_distinct_pair(self):
        call_count = {"n": 0}

        def counting_fit(**kw):
            call_count["n"] += 1
            return (self._make_spline_art(),)

        ctx = self._build_ctx(_psea_table_df(sig=False))
        ctx.register_action(
            "psea", "_compute_pair_fit_and_residuals", counting_fit
        )
        self._run(ctx, self._TWO_PAIRS)
        self.assertEqual(call_count["n"], 2)

    def test_returns_list(self):
        ctx = self._build_ctx(_psea_table_df(sig=False))
        result = self._run(ctx, self._ONE_PAIR)
        self.assertIsInstance(result, list)


# ---------------------------------------------------------------------------
# count_antibody_events
# ---------------------------------------------------------------------------

def _ae_psea_tables(rows):
    """Build a fake psea_tables dict as count_antibody_events expects."""
    df = pd.DataFrame(rows)
    return {"pair1": df}


class TestCountAntibodyEvents(TestPluginBase):
    package = "q2_PSEA.tests"

    def _call(self, rows, **kwargs):
        tables = _ae_psea_tables(rows)
        defaults = dict(p_val_thresh=0.05, nes_thresh=1.0, taxa_access="ID")
        defaults.update(kwargs)
        return count_antibody_events(tables, **defaults)

    def test_returns_two_dataframes(self):
        pos, neg = self._call([
            {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
        ])
        self.assertIsInstance(pos, pd.DataFrame)
        self.assertIsInstance(neg, pd.DataFrame)

    def test_significant_positive_nes_counted(self):
        pos, _ = self._call([
            {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
        ])
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos.iloc[0]["Species"], "sp1")
        self.assertEqual(pos.iloc[0]["Events"], 1)

    def test_significant_negative_nes_counted(self):
        _, neg = self._call([
            {"ID": "sp2", "NES": -2.0, "p.adjust": 0.01},
        ])
        self.assertEqual(len(neg), 1)
        self.assertEqual(neg.iloc[0]["Species"], "sp2")
        self.assertEqual(neg.iloc[0]["Events"], 1)

    def test_non_significant_row_not_counted(self):
        pos, neg = self._call([
            {"ID": "sp1", "NES": 2.0, "p.adjust": 0.9},   # p-val too high
            {"ID": "sp2", "NES": 0.5, "p.adjust": 0.01},  # NES too low
        ])
        self.assertEqual(len(pos), 0)
        self.assertEqual(len(neg), 0)

    def test_multiple_pairs_accumulate_counts(self):
        tables = {
            "pairA": pd.DataFrame([
                {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
            ]),
            "pairB": pd.DataFrame([
                {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
            ]),
        }
        pos, _ = count_antibody_events(
            tables, p_val_thresh=0.05, nes_thresh=1.0, taxa_access="ID"
        )
        self.assertEqual(pos.iloc[0]["Events"], 2)

    def test_output_sorted_by_events_descending(self):
        tables = {
            "pairA": pd.DataFrame([
                {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
                {"ID": "sp2", "NES": 2.0, "p.adjust": 0.01},
            ]),
            "pairB": pd.DataFrame([
                {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
            ]),
        }
        pos, _ = count_antibody_events(
            tables, p_val_thresh=0.05, nes_thresh=1.0, taxa_access="ID"
        )
        self.assertEqual(pos.iloc[0]["Species"], "sp1")

    def test_output_columns(self):
        pos, neg = self._call([
            {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
        ])
        self.assertListEqual(list(pos.columns), ["Species", "Events"])
        self.assertListEqual(list(neg.columns), ["Species", "Events"])

    def test_taxa_access_species_name(self):
        pos, _ = self._call(
            [{"species_name": "InfluenzaA", "NES": 2.0, "p.adjust": 0.01}],
            taxa_access="species_name",
        )
        self.assertEqual(pos.iloc[0]["Species"], "InfluenzaA")


# ---------------------------------------------------------------------------
# TestMakePseaTableNonIterative
# ---------------------------------------------------------------------------

class TestMakePseaTableNonIterative(TestPluginBase):
    """Execution tests for make_psea_table with iterative_analysis=False.

    All QIIME 2 sub-actions are registered on a mock context so no R code
    runs.  The spline step inside the loop
    (``_compute_pair_fit_and_residuals``) is called directly with
    ``spline_type="py-smooth"`` and ``dof=1`` to stay pure-Python.
    """

    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()

        # scores-vis.tsv: index=peptide_id, cols=[sample1, sample2]
        raw_scores = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        # FeatureTable[Zscore] viewed as DataFrame has samples as rows,
        # peptides as columns — so transpose the on-disk representation.
        self.scores_q2 = raw_scores.T

        # pairs.tsv → one pair (sample1, sample2)
        self.pairs_one = pd.read_csv(
            self.get_data_path("pairs.tsv"), sep="\t"
        )
        # pairs-two.tsv → two pairs: (sample1, sample2) and (sample3, sample4)
        self.pairs_two = pd.read_csv(
            self.get_data_path("pairs-two.tsv"), sep="\t"
        )

        self.gmt = pd.read_csv(
            self.get_data_path("peptide-sets.tsv"), sep="\t"
        )

    # ------------------------------------------------------------------
    # Context factory
    # ------------------------------------------------------------------

    def _make_spline_art(self, n=15):
        """Return a mock FeatureData[Spline] artifact with n peptide rows."""
        idx = pd.Index([f"pep_{i:02d}" for i in range(n)])
        df = pd.DataFrame({
            "x": np.linspace(0.5, 2.0, n),
            "yfit": np.linspace(0.5, 2.0, n),
            "maxZ": np.ones(n),
            "deltaZ": np.zeros(n),
        }, index=idx)
        return _MockArtifact(df)

    def _build_ctx(self):
        ctx = _MockCtx()

        fake_psea = _MockArtifact(_fake_psea_df())
        spline_art = self._make_spline_art()
        pos_ae = _MockArtifact(
            pd.DataFrame({"Species": pd.Series([], dtype=str),
                          "Events": pd.Series([], dtype=int)})
        )
        neg_ae = _MockArtifact(
            pd.DataFrame({"Species": pd.Series([], dtype=str),
                          "Events": pd.Series([], dtype=int)})
        )

        ctx.register_action(
            "psea", "_compute_pair_fit_and_residuals",
            lambda **kw: (spline_art,),
        )
        ctx.register_action(
            "psea", "create_fgsea_table_for_pair",
            lambda **kw: (fake_psea,),
        )
        ctx.register_action(
            "psea", "count_antibody_events",
            lambda **kw: (pos_ae, neg_ae),
        )
        ctx.register_action(
            "psea", "zscatter",
            lambda **kw: (_MockArtifact(None),),
        )
        ctx.register_action(
            "psea", "volcano",
            lambda **kw: (_MockArtifact(None),),
        )
        ctx.register_action(
            "psea", "aeplots",
            lambda **kw: (_MockArtifact(None),),
        )

        return ctx

    # ------------------------------------------------------------------
    # Runner helper
    # ------------------------------------------------------------------

    def _run(self, pairs_df=None, scores_q2=None, ctx=None, **kwargs):
        if ctx is None:
            ctx = self._build_ctx()
        if pairs_df is None:
            pairs_df = self.pairs_one
        if scores_q2 is None:
            scores_q2 = self.scores_q2

        return make_psea_table(
            ctx,
            scores=_MockArtifact(scores_q2),
            pairs=_MockArtifact(pairs_df),
            peptide_sets=_MockArtifact(self.gmt),
            threshold=1.0,
            permutation_num=100,
            min_size=2,
            max_size=500,
            spline_type="py-smooth",
            degree=3,
            dof=1,
            seed=42,
            iterative_analysis=False,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Tests — output structure
    # ------------------------------------------------------------------

    def test_returns_four_outputs(self):
        result = self._run()
        self.assertEqual(len(result), 4)

    def test_psea_tables_is_dict(self):
        _, _, _, psea_tables = self._run()
        self.assertIsInstance(psea_tables, dict)

    def test_one_pair_produces_one_psea_table(self):
        _, _, _, psea_tables = self._run()
        self.assertEqual(len(psea_tables), 1)

    # ------------------------------------------------------------------
    # Tests — pairs-two.tsv (two-pair path)
    # ------------------------------------------------------------------

    def _scores_four_samples(self):
        """Extend scores_q2 to cover sample3/sample4 for pairs-two.tsv."""
        scores_4col = self.scores_q2.copy()
        scores_4col.loc["sample3"] = self.scores_q2.loc["sample1"] * 0.9
        scores_4col.loc["sample4"] = self.scores_q2.loc["sample2"] * 0.8
        return scores_4col

    def test_two_pairs_produces_two_psea_tables(self):
        """
        pairs-two.tsv drives a two-pair run; psea_tables must have 2 items.
        """
        _, _, _, psea_tables = self._run(
            pairs_df=self.pairs_two,
            scores_q2=self._scores_four_samples(),
        )
        self.assertEqual(len(psea_tables), 2)

    def test_create_fgsea_called_once_for_single_pair(self):
        call_count = {"n": 0}
        fake_psea = _MockArtifact(_fake_psea_df())

        def counting(**kw):
            call_count["n"] += 1
            return (fake_psea,)

        ctx = self._build_ctx()
        ctx.register_action("psea", "create_fgsea_table_for_pair", counting)
        self._run(ctx=ctx)
        self.assertEqual(call_count["n"], 1)

    def test_create_fgsea_called_twice_for_two_pairs(self):
        """pairs-two.tsv triggers create_fgsea_table_for_pair twice."""
        call_count = {"n": 0}
        fake_psea = _MockArtifact(_fake_psea_df())

        def counting(**kw):
            call_count["n"] += 1
            return (fake_psea,)

        ctx = self._build_ctx()
        ctx.register_action("psea", "create_fgsea_table_for_pair", counting)
        self._run(
            ctx=ctx,
            pairs_df=self.pairs_two,
            scores_q2=self._scores_four_samples(),
        )
        self.assertEqual(call_count["n"], 2)

    # ------------------------------------------------------------------
    # Tests — optional metadata inputs
    # ------------------------------------------------------------------

    def test_species_taxa_metadata_accepted(self):
        taxa_md = qiime2.Metadata.load(self.get_data_path("species-taxa.tsv"))
        result = self._run(species_taxa=taxa_md)
        self.assertIsNotNone(result)

    def test_species_colors_metadata_accepted(self):
        colors_md = qiime2.Metadata.load(
            self.get_data_path("species-colors.tsv")
        )
        result = self._run(species_colors=colors_md)
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# TestMakePseaTableIterative
# ---------------------------------------------------------------------------

class TestMakePseaTableIterative(TestPluginBase):
    """Execution tests for the iterative_analysis=True branch of
    make_psea_table.

    ``run_iterative_peptide_analysis`` is registered in the mock context and
    returns a Collection-like artifact whose ``.values()`` yields one filtered
    GMT artifact per pair.
    """

    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()

        raw_scores = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        self.scores_q2 = raw_scores.T

        self.pairs_one = pd.read_csv(
            self.get_data_path("pairs.tsv"), sep="\t"
        )

        self.gmt = pd.read_csv(
            self.get_data_path("peptide-sets.tsv"), sep="\t"
        )

    def _make_spline_art(self, n=15):
        idx = pd.Index([f"pep_{i:02d}" for i in range(n)])
        df = pd.DataFrame({
            "x": np.linspace(0.5, 2.0, n),
            "yfit": np.linspace(0.5, 2.0, n),
            "maxZ": np.ones(n),
            "deltaZ": np.zeros(n),
        }, index=idx)
        return _MockArtifact(df)

    def _build_ctx(self, on_run_iterative=None):
        """Build a mock context for the iterative analysis path.

        ``on_run_iterative`` is an optional override for the
        ``run_iterative_peptide_analysis`` callable so callers can inject
        tracking logic.
        """
        ctx = _MockCtx()

        fake_psea = _MockArtifact(_fake_psea_df())
        gmt_art = _MockArtifact(self.gmt)
        # Collection[GMT] artifact: .values() must return one GMT per pair.
        filtered_gmts_art = _MockArtifact({"0": gmt_art})
        spline_art = self._make_spline_art()

        pos_ae = _MockArtifact(
            pd.DataFrame({"Species": pd.Series([], dtype=str),
                          "Events": pd.Series([], dtype=int)})
        )
        neg_ae = _MockArtifact(
            pd.DataFrame({"Species": pd.Series([], dtype=str),
                          "Events": pd.Series([], dtype=int)})
        )

        if on_run_iterative is None:
            on_run_iterative = lambda **kw: (filtered_gmts_art,)  # noqa: E731

        ctx.register_action(
            "psea", "_compute_pair_fit_and_residuals",
            lambda **kw: (spline_art,),
        )
        ctx.register_action(
            "psea", "run_iterative_peptide_analysis", on_run_iterative
        )
        ctx.register_action(
            "psea", "create_fgsea_table_for_pair",
            lambda **kw: (fake_psea,),
        )
        ctx.register_action(
            "psea", "count_antibody_events",
            lambda **kw: (pos_ae, neg_ae),
        )
        ctx.register_action(
            "psea", "zscatter",
            lambda **kw: (_MockArtifact(None),),
        )
        ctx.register_action(
            "psea", "volcano",
            lambda **kw: (_MockArtifact(None),),
        )
        ctx.register_action(
            "psea", "aeplots",
            lambda **kw: (_MockArtifact(None),),
        )

        return ctx

    def _run(self, ctx=None, **kwargs):
        if ctx is None:
            ctx = self._build_ctx()

        return make_psea_table(
            ctx,
            scores=_MockArtifact(self.scores_q2),
            pairs=_MockArtifact(self.pairs_one),
            peptide_sets=_MockArtifact(self.gmt),
            threshold=1.0,
            permutation_num=100,
            min_size=2,
            max_size=500,
            spline_type="py-smooth",
            degree=3,
            dof=1,
            seed=42,
            iterative_analysis=True,
            **kwargs,
        )

    def test_iterative_calls_run_iterative_analysis(self):
        call_count = {"n": 0}
        gmt_art = _MockArtifact(self.gmt)
        filtered_gmts_art = _MockArtifact({"0": gmt_art})

        def tracking(**kw):
            call_count["n"] += 1
            return (filtered_gmts_art,)

        ctx = self._build_ctx(on_run_iterative=tracking)
        self._run(ctx=ctx)
        self.assertEqual(call_count["n"], 1)

    def test_iterative_returns_four_outputs(self):
        result = self._run()
        self.assertEqual(len(result), 4)

    def test_iterative_psea_tables_is_dict(self):
        _, _, _, psea_tables = self._run()
        self.assertIsInstance(psea_tables, dict)

    def test_iterative_one_pair_produces_one_psea_table(self):
        _, _, _, psea_tables = self._run()
        self.assertEqual(len(psea_tables), 1)


if __name__ == "__main__":
    unittest.main()
