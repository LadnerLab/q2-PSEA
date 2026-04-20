import unittest
from math import log
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import qiime2
from pandas.testing import assert_series_equal
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.actions.psea import (
    _compute_pair_fit_and_residuals,
    _update_gmt,
    count_antibody_events,
    create_fgsea_table_for_pair,
    process_scores,
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
        # scores.tsv is features×samples on disk; QIIME 2 FeatureTable[Zscore]
        # is viewed as samples×features, so transpose before passing.
        self.scores = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        self.scores_q2 = self.scores.T  # samples×features

    def _pairs(self, *pairs):
        """
        Build a PSEAPairs-style DataFrame from a sequence of (a, b) tuples.
        """
        return pd.DataFrame(
            [(a, b) for a, b in pairs], columns=["sampleA", "sampleB"]
        )

    def _expected(self, v, base=2, offset=3):
        return log(max(1.0, base ** offset + v), base) - offset

    def test_selects_only_columns_in_pairs(self):
        result = process_scores(self.scores_q2, self._pairs(("sA", "sB")))
        self.assertEqual(set(result.columns), {"sA", "sB"})
        self.assertNotIn("sC", result.columns)

    def test_zero_input_maps_to_zero(self):
        # samples×features: rows=samples, cols=features
        scores = pd.DataFrame({"p": [0.0, 0.0]}, index=["sA", "sB"])
        result = process_scores(scores, self._pairs(("sA", "sB")))
        self.assertAlmostEqual(result.loc["p", "sA"], 0.0)

    def test_very_negative_input_clamps_to_minus_three(self):
        scores = pd.DataFrame({"p": [-100.0, -100.0]}, index=["sA", "sB"])
        result = process_scores(scores, self._pairs(("sA", "sB")))
        self.assertAlmostEqual(result.loc["p", "sA"], -3.0)

    def test_known_positive_value_transformed_correctly(self):
        v = 5.0
        scores = pd.DataFrame({"p": [v, 0.0]}, index=["sA", "sB"])
        result = process_scores(scores, self._pairs(("sA", "sB")))
        self.assertAlmostEqual(result.loc["p", "sA"], self._expected(v))

    def test_duplicate_samples_across_pairs_deduplicated(self):
        result = process_scores(
            self.scores_q2, self._pairs(("sA", "sB"), ("sA", "sC"))
        )
        self.assertEqual(sorted(result.columns), ["sA", "sB", "sC"])

    def test_all_peptides_retained(self):
        result = process_scores(self.scores_q2, self._pairs(("sA", "sB")))
        self.assertEqual(len(result), len(self.scores))

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
# _update_gmt
# ---------------------------------------------------------------------------


class TestUpdateGmt(unittest.TestCase):
    """Unit tests for _update_gmt — no plugin registration required."""

    def _gmt(self):
        return pd.DataFrame({
            "term": ["sp1", "sp1", "sp2", "sp2"],
            "gene": ["pep_00", "pep_01", "pep_00", "pep_02"],
        })

    def _psea_table(self, sig=True):
        if sig:
            return pd.DataFrame({
                "ID": ["sp1", "sp2"],
                "NES": [2.0, 0.5],
                "p.adjust": [0.01, 0.9],
                "all_tested_peptides": ["pep_00/pep_01", "pep_00/pep_02"],
            })
        return pd.DataFrame({
            "ID": ["sp1", "sp2"],
            "NES": [0.5, 0.3],
            "p.adjust": [0.9, 0.8],
            "all_tested_peptides": ["pep_00/pep_01", "pep_00/pep_02"],
        })

    def _call(self, psea_table=None, peptide_sets=None, **kwargs):
        defaults = dict(
            p_val_thresh=0.05, nes_thresh=1.0, sample_a="sA", sample_b="sB"
        )
        defaults.update(kwargs)
        return _update_gmt(
            psea_table=psea_table if psea_table is not None
            else self._psea_table(),
            peptide_sets=peptide_sets if peptide_sets is not None
            else self._gmt(),
            **defaults,
        )

    def test_returns_dataframe(self):
        self.assertIsInstance(self._call(), pd.DataFrame)

    def test_no_significant_species_returns_unchanged_gmt(self):
        gmt = self._gmt()
        result = self._call(psea_table=self._psea_table(sig=False))
        self.assertEqual(len(result), len(gmt))

    def test_significant_species_removes_cross_reactive_peptides(self):
        """sp1 is significant; pep_00 shared with sp2 is removed from sp2."""
        result = self._call()
        sp2_rows = result[result["term"] == "sp2"]
        self.assertNotIn("pep_00", sp2_rows["gene"].values)

    def test_significant_species_own_rows_preserved(self):
        result = self._call()
        sp1_rows = result[result["term"] == "sp1"]
        self.assertEqual(len(sp1_rows), 2)

    def test_non_shared_peptides_preserved(self):
        """pep_02 is unique to sp2 and must not be removed."""
        result = self._call()
        sp2_rows = result[result["term"] == "sp2"]
        self.assertIn("pep_02", sp2_rows["gene"].values)

    def test_only_first_significant_species_processed(self):
        """
        Break after first significant row — only sp1's tested peps removed.
        """
        psea_table = pd.DataFrame({
            "ID": ["sp1", "sp2"],
            "NES": [2.0, 2.0],
            "p.adjust": [0.01, 0.02],
            "all_tested_peptides": ["pep_00/pep_01", "pep_00/pep_02"],
        })
        result = self._call(psea_table=psea_table)
        sp2_rows = result[result["term"] == "sp2"]
        self.assertNotIn("pep_00", sp2_rows["gene"].values)
        self.assertIn("pep_02", sp2_rows["gene"].values)


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

    def _default_fit(self):
        n = len(self.scores)
        idx = self.scores.index
        return pd.DataFrame({
            "x": np.linspace(0, 1, n),
            "yfit": np.linspace(0, 1, n),
            "maxZ": pd.Series(np.ones(n), index=idx),
            "deltaZ": pd.Series(np.zeros(n), index=idx),
        }, index=idx)

    def _call(self, precomputed_fit=None, **kwargs):
        """Invoke create_fgsea_table_for_pair with R fully mocked."""
        if precomputed_fit is None:
            precomputed_fit = self._default_fit()
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
        with patch(
            "q2_PSEA.actions.psea._compute_pair_fit_and_residuals"
        ) as mock_fit:
            self._call()
            mock_fit.assert_not_called()

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
        processed_scores_art = _MockArtifact(pd.DataFrame())
        pos_ae = _MockArtifact(
            pd.DataFrame({"Species": pd.Series([], dtype=str),
                          "Events": pd.Series([], dtype=int)})
        )
        neg_ae = _MockArtifact(
            pd.DataFrame({"Species": pd.Series([], dtype=str),
                          "Events": pd.Series([], dtype=int)})
        )

        ctx.register_action(
            "psea", "process_scores",
            lambda **kw: (processed_scores_art,),
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


if __name__ == "__main__":
    unittest.main()
