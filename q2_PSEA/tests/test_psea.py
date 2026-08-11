import unittest
from math import log

import numpy as np
import pandas as pd
import qiime2
from pandas.testing import assert_series_equal
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.actions.psea import (
    _compute_pair_fit_and_residuals,
    _map_residuals_and_zscores,
    _process_scores,
)


def _load_scores_art(path):
    """Import a features-as-rows TSV as FeatureTable[Zscore]."""
    raw = pd.read_csv(path, sep="\t", index_col=0)
    return qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)


def _load_pairs_art(path):
    df = pd.read_csv(path, sep="\t")
    return qiime2.Artifact.import_data("PSEAPairs", df)


def _load_gmt_art(path):
    df = pd.read_csv(path, sep="\t")
    return qiime2.Artifact.import_data("GMT", df)


# ---------------------------------------------------------------------------
# _filter_scores_to_pairs — integration via plugin method
# ---------------------------------------------------------------------------

class TestFilterScoresToPairsIntegration(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.scores_art = _load_scores_art(self.get_data_path("scores.tsv"))
        self.pairs_art = _load_pairs_art(self.get_data_path("pairs.tsv"))
        self.method = self.plugin.methods["_filter_scores_to_pairs"]

    def _run(self, scores=None, pairs=None):
        result, = self.method(
            scores=scores or self.scores_art,
            pairs=pairs or self.pairs_art,
        )
        return result.view(pd.DataFrame)

    def test_output_only_contains_pair_samples(self):
        result = self._run()
        # FeatureTable[Zscore] view: samples as index, features as columns
        self.assertSetEqual(set(result.index), {"sA", "sB"})

    def test_all_features_retained(self):
        raw = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        result = self._run()
        self.assertEqual(set(result.columns), set(raw.index))

    def test_two_pairs_selects_four_samples(self):
        pairs_two = _load_pairs_art(self.get_data_path("pairs-two.tsv"))
        result = self._run(pairs=pairs_two)
        self.assertSetEqual(set(result.index), {"sA", "sB", "sC", "sD"})


# ---------------------------------------------------------------------------
# _process_scores — integration via plugin method
# ---------------------------------------------------------------------------

class TestProcessScoresIntegration(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.scores_art = _load_scores_art(self.get_data_path("scores.tsv"))
        self.method = self.plugin.methods["_process_scores"]

    def _run(self, scores=None):
        result, = self.method(scores=scores or self.scores_art)
        return result.view(pd.DataFrame)

    def test_all_peptides_retained(self):
        raw = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        result = self._run()
        # FeatureTable[Zscore] view: samples as index, features as columns
        self.assertEqual(set(result.columns), set(raw.index))

    def test_zero_input_maps_to_zero(self):
        # raw z-score of 0 → 8+0=8; log2(8)-3 = 0
        raw = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        raw.iloc[:, :] = 0.0
        art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)
        result = self._run(scores=art)
        self.assertTrue(np.allclose(result.values, 0.0))

    def test_very_negative_input_clamps_at_negative_three(self):
        raw = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        raw.iloc[:, :] = -100.0
        art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)
        result = self._run(scores=art)
        self.assertTrue(np.allclose(result.values, -3.0))

    def test_known_value_transformed_correctly(self):
        raw = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        raw.iloc[:, :] = 5.0
        art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)
        result = self._run(scores=art)
        expected = log(13.0, 2) - 3  # log2(8+5) - 3
        self.assertTrue(np.allclose(result.values, expected))


# ---------------------------------------------------------------------------
# _compute_pair_fit_and_residuals — integration via plugin method
# ---------------------------------------------------------------------------

class TestComputePairFitIntegration(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        # scores-vis.tsv already has only sA and sB — no filtering needed
        raw = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        scores_art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)

        process = self.plugin.methods["_process_scores"]
        self.processed_art, = process(scores=scores_art)
        self.method = self.plugin.methods["_compute_pair_fit_and_residuals"]

    def _run(self, spline_type="py-smooth", **kwargs):
        result, = self.method(
            processed_zscores=self.processed_art,
            pair="sA~sB",
            spline_type=spline_type,
            degree=3,
            **kwargs,
        )
        return result

    def test_output_has_required_columns(self):
        df = self._run().view(pd.DataFrame)
        for col in ("x", "yfit", "maxZ", "deltaZ"):
            self.assertIn(col, df.columns)

    def test_x_and_yfit_have_one_value_per_peptide(self):
        raw = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        df = self._run().view(pd.DataFrame)
        n = len(raw)
        self.assertEqual(df["x"].dropna().shape[0], n)
        self.assertEqual(df["yfit"].dropna().shape[0], n)

    def test_maxz_and_deltaz_have_one_value_per_peptide(self):
        raw = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        df = self._run().view(pd.DataFrame)
        n = len(raw)
        self.assertEqual(df["maxZ"].dropna().shape[0], n)
        self.assertEqual(df["deltaZ"].dropna().shape[0], n)

    def test_no_nan_in_x_or_yfit(self):
        df = self._run().view(pd.DataFrame)
        self.assertFalse(df["x"].dropna().isna().any())
        self.assertFalse(df["yfit"].dropna().isna().any())

    def test_maxz_equals_elementwise_max_of_processed_scores(self):
        df = self._run().view(pd.DataFrame)
        processed_df = self.processed_art.view(pd.DataFrame)
        # processed_df: samples×features; select sA and sB rows,
        # transpose to features×samples, then take max across samples
        pair_df = processed_df.loc[["sA", "sB"], :].T
        expected_max = pair_df.max(axis=1)
        maxZ = df["maxZ"].dropna()
        assert_series_equal(
            maxZ.sort_index(), expected_max.sort_index(), check_names=False
        )


# ---------------------------------------------------------------------------
# _create_fgsea_table_for_pair — full R integration
# ---------------------------------------------------------------------------

class TestCreateFgseaTableForPairIntegration(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        # scores-vis.tsv already has only sA and sB — no filtering needed
        raw = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        scores_art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)
        self.gmt_art = _load_gmt_art(self.get_data_path("peptide-sets.tsv"))

        process = self.plugin.methods["_process_scores"]
        self.processed_art, = process(scores=scores_art)

        compute_fit = self.plugin.methods["_compute_pair_fit_and_residuals"]
        self.spline_art, = compute_fit(
            processed_zscores=self.processed_art,
            pair="sA~sB",
            spline_type="py-smooth",
            degree=3,
        )
        self.method = self.plugin.methods["_create_fgsea_table_for_pair"]

    def _run(self, **kwargs):
        result, = self.method(
            processed_zscores=self.processed_art,
            peptide_sets=self.gmt_art,
            precomputed_fit=self.spline_art,
            threshold=0.0,
            permutation_num=100,
            min_size=3,
            max_size=500,
            seed=42,
            **kwargs,
        )
        return result

    def test_output_has_expected_columns(self):
        df = self._run().view(pd.DataFrame)
        for col in ("ID", "enrichmentScore", "NES", "p.adjust",
                    "core_enrichment", "pvalue", "qvalue",
                    "all_tested_peptides"):
            self.assertIn(col, df.columns)

    def test_p_adjust_values_are_valid_probabilities(self):
        df = self._run().view(pd.DataFrame)
        self.assertTrue((df["p.adjust"] >= 0).all())
        self.assertTrue((df["p.adjust"] <= 1).all())

    def test_nes_values_are_finite(self):
        df = self._run().view(pd.DataFrame)
        self.assertTrue(np.all(np.isfinite(df["NES"].values)))

    def test_enrichment_score_between_minus_one_and_one(self):
        df = self._run().view(pd.DataFrame)
        self.assertTrue((df["enrichmentScore"].abs() <= 1.0).all())

    def test_species_taxa_metadata_adds_species_name_column(self):
        taxa = qiime2.Metadata.load(self.get_data_path("species-taxa.tsv"))
        df = self._run(species_taxa=taxa).view(pd.DataFrame)
        self.assertIn("species_name", df.columns)

    def test_deterministic_with_fixed_seed(self):
        df1 = self._run().view(pd.DataFrame)
        df2 = self._run().view(pd.DataFrame)
        pd.testing.assert_frame_equal(
            df1.sort_values("ID").reset_index(drop=True),
            df2.sort_values("ID").reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# count_antibody_events — integration via plugin method
# ---------------------------------------------------------------------------

class TestCountAntibodyEventsIntegration(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.method = self.plugin.methods["count_antibody_events"]

    def _make_psea_art(self, rows):
        df = pd.DataFrame(rows)
        return qiime2.Artifact.import_data("FeatureData[PSEAScores]", df)

    def _run(self, rows, taxa_access="ID", p_value=0.05, enrichment_score=1.0):
        art = self._make_psea_art(rows)
        pos, neg = self.method(
            psea_table=art,
            p_value=p_value,
            enrichment_score=enrichment_score,
            taxa_access=taxa_access,
        )
        return pos.view(pd.DataFrame), neg.view(pd.DataFrame)

    def test_significant_positive_nes_counted_in_pos(self):
        pos, neg = self._run([{"ID": "sp1", "NES": 2.0, "p.adjust": 0.01}])
        self.assertEqual(len(pos), 1)
        self.assertEqual(pos.iloc[0]["Species"], "sp1")
        self.assertEqual(pos.iloc[0]["Events"], 1)
        self.assertEqual(len(neg), 0)

    def test_significant_negative_nes_counted_in_neg(self):
        pos, neg = self._run([{"ID": "sp2", "NES": -2.0, "p.adjust": 0.01}])
        self.assertEqual(len(neg), 1)
        self.assertEqual(neg.iloc[0]["Species"], "sp2")
        self.assertEqual(neg.iloc[0]["Events"], 1)
        self.assertEqual(len(pos), 0)

    def test_non_significant_p_value_excluded(self):
        pos, neg = self._run([{"ID": "sp1", "NES": 2.0, "p.adjust": 0.9}])
        self.assertEqual(len(pos), 0)
        self.assertEqual(len(neg), 0)

    def test_nes_below_threshold_excluded(self):
        pos, neg = self._run([{"ID": "sp1", "NES": 0.3, "p.adjust": 0.01}])
        self.assertEqual(len(pos), 0)
        self.assertEqual(len(neg), 0)

    def test_multiple_rows_same_species_accumulate_events(self):
        art = self._make_psea_art([
            {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
            {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
        ])
        pos, _ = self.method(
            psea_table=art,
            p_value=0.05, enrichment_score=1.0, taxa_access="ID",
        )
        df = pos.view(pd.DataFrame)
        self.assertEqual(df.iloc[0]["Events"], 2)

    def test_output_sorted_descending_by_events(self):
        art = self._make_psea_art([
            {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
            {"ID": "sp2", "NES": 2.0, "p.adjust": 0.01},
            {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
        ])
        pos, _ = self.method(
            psea_table=art,
            p_value=0.05, enrichment_score=1.0, taxa_access="ID",
        )
        df = pos.view(pd.DataFrame)
        self.assertEqual(df.iloc[0]["Species"], "sp1")

    def test_empty_result_when_no_rows(self):
        art = self._make_psea_art([{"ID": "sp1", "NES": 0.1, "p.adjust": 0.9}])
        pos, neg = self.method(
            psea_table=art,
            p_value=0.05, enrichment_score=1.0, taxa_access="ID",
        )
        self.assertEqual(len(pos.view(pd.DataFrame)), 0)
        self.assertEqual(len(neg.view(pd.DataFrame)), 0)


# ---------------------------------------------------------------------------
# _process_scores direct function tests (unit-level, no plugin dispatch)
# ---------------------------------------------------------------------------

class TestProcessScoresDirect(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        raw = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)
        # view is samples×features
        self.scores_view = art.view(pd.DataFrame)

    def test_all_samples_retained(self):
        # Direct call receives samples×features and returns features×samples
        result = _process_scores(self.scores_view)
        self.assertSetEqual(set(result.columns), set(self.scores_view.index))

    def test_known_value_log_scaled(self):
        expected = log(13.0, 2) - 3  # log2(8+5) - 3
        raw = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        raw.iloc[:, :] = 5.0
        art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)
        result = _process_scores(art.view(pd.DataFrame))
        self.assertTrue(np.allclose(result.values, expected))


# ---------------------------------------------------------------------------
# _compute_pair_fit_and_residuals direct function tests
# ---------------------------------------------------------------------------

class TestComputePairFitDirect(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        raw = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)
        self.view = art.view(pd.DataFrame)

    def _call(self, **kwargs):
        return _compute_pair_fit_and_residuals(
            self.view, "sA~sB", "py-smooth", 3, **kwargs
        )

    def test_no_inf_in_output(self):
        df = self._call()
        for col in ("x", "yfit", "maxZ", "deltaZ"):
            vals = df[col].dropna().values
            self.assertTrue(
                np.all(np.isfinite(vals)), f"{col} has non-finite values"
            )

    def test_maxz_is_elementwise_max_of_pair(self):
        df = self._call()
        maxZ = df["maxZ"].dropna()
        data = self.view.loc[["sA", "sB"], :].T
        expected = data.max(axis=1)
        assert_series_equal(
            maxZ.sort_index(), expected.sort_index(), check_names=False
        )


# ---------------------------------------------------------------------------
# _map_residuals_and_zscores direct function tests
# ---------------------------------------------------------------------------

class TestMapResidualsAndZscoresDirect(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        raw = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)
        # samples×features, matching the FeatureTable[Zscore] view convention
        self.zscores = art.view(pd.DataFrame)
        self.spline = _compute_pair_fit_and_residuals(
            self.zscores, "sA~sB", "py-smooth", 3
        )
        # ep1 collapses pep_00+pep_01; the rest pass through unmapped, as
        # create_epitope_map would emit for peptides that were not collapsed.
        self.epitope_map = pd.DataFrame(
            {
                "CodeName": {
                    "ep1": ["pep_00", "pep_01"],
                    "pep_02": ["pep_02"],
                    "pep_03": ["pep_03"],
                    "pep_04": ["pep_04"],
                    "pep_05": ["pep_05"],
                    "pep_06": ["pep_06"],
                }
            }
        )

    def _call(self):
        return _map_residuals_and_zscores(
            self.spline.copy(), self.zscores.copy(), self.epitope_map.copy()
        )

    def test_collapsed_epitope_added_to_spline(self):
        mapped_spline, _ = self._call()
        self.assertIn("ep1", mapped_spline.index)

    def test_uncollapsed_peptides_still_present_in_spline(self):
        mapped_spline, _ = self._call()
        self.assertIn("pep_00", mapped_spline.index)
        self.assertIn("pep_01", mapped_spline.index)

    def test_epitope_deltaz_matches_max_abs_residual_peptide(self):
        mapped_spline, _ = self._call()
        winner = self.spline.loc[["pep_00", "pep_01"], "deltaZ"] \
            .abs().idxmax()
        self.assertEqual(
            mapped_spline.loc["ep1", "deltaZ"],
            self.spline.loc[winner, "deltaZ"],
        )

    def test_mapped_zscores_excludes_collapsed_peptides(self):
        _, mapped_zscores = self._call()
        self.assertNotIn("pep_00", mapped_zscores.index)
        self.assertNotIn("pep_01", mapped_zscores.index)
        self.assertIn("ep1", mapped_zscores.index)

    def test_mapped_zscores_keeps_unmapped_peptides(self):
        _, mapped_zscores = self._call()
        for pep in ("pep_02", "pep_03", "pep_04", "pep_05", "pep_06"):
            self.assertIn(pep, mapped_zscores.index)

    def test_mapped_zscores_columns_are_samples(self):
        _, mapped_zscores = self._call()
        self.assertEqual(set(mapped_zscores.columns), {"sA", "sB"})

    def test_epitope_zscore_matches_max_abs_residual_peptide(self):
        mapped_spline, mapped_zscores = self._call()
        winner = self.spline.loc[["pep_00", "pep_01"], "deltaZ"] \
            .abs().idxmax()
        expected = self.zscores.loc[:, winner]
        pd.testing.assert_series_equal(
            mapped_zscores.loc["ep1"], expected, check_names=False
        )


# ---------------------------------------------------------------------------
# _map_residuals_and_zscores — integration via plugin method
# ---------------------------------------------------------------------------

class TestMapResidualsAndZscoresIntegration(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        raw = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        scores_art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)

        process = self.plugin.methods["_process_scores"]
        self.processed_art, = process(scores=scores_art)

        compute_fit = self.plugin.methods["_compute_pair_fit_and_residuals"]
        self.spline_art, = compute_fit(
            processed_zscores=self.processed_art,
            pair="sA~sB",
            spline_type="py-smooth",
            degree=3,
        )

        # epitope.tsv only covers pep_00..pep_03, so build a small epitope
        # table that covers every peptide in scores-vis.tsv (pep_00..pep_06)
        # so no peptide gets dropped from the mapped Z-score output.
        epi = pd.DataFrame(
            {
                "SpeciesID": ["sp001"] * 2 + [f"sp00{i}" for i in range(2, 7)],
                "ClusterID": [f"C{i+1}" for i in range(7)],
                "EpitopeWindow": [f"W{i+1}" for i in range(7)],
                "Species": ["InfluenzaA"] * 2 + [f"Species{i}"
                                                 for i in range(2, 7)],
                "Subtype": ["H1N1", "H3N2", "S2", "S3", "S4", "S5", "S6"],
                "Category": ["Viral"] * 7,
            },
            index=pd.Index(
                [f"pep_0{i}" for i in range(7)], name="CodeName"
            ),
        )
        epi_art = qiime2.Artifact.import_data("FeatureData[Epitope]", epi)
        create_epitope_map = self.plugin.methods["create_epitope_map"]
        self.epitope_map_art, = create_epitope_map(
            epitope=epi_art, collapse="Viral"
        )

        self.method = self.plugin.methods["_map_residuals_and_zscores"]

    def _run(self):
        mapped_spline, mapped_zscores = self.method(
            spline=self.spline_art,
            zscores=self.processed_art,
            epitope_map=self.epitope_map_art,
        )
        return mapped_spline.view(pd.DataFrame), mapped_zscores.view(
            pd.DataFrame
        )

    def test_collapsed_epitope_present_in_mapped_spline(self):
        mapped_spline, _ = self._run()
        self.assertIn("sp001_C1_W1", mapped_spline.index)

    def test_collapsed_epitope_present_in_mapped_zscores(self):
        _, mapped_zscores = self._run()
        # FeatureTable[Zscore] view: samples as index, features as columns
        self.assertIn("sp001_C1_W1", mapped_zscores.columns)

    def test_uncollapsed_peptides_absent_from_mapped_zscores(self):
        _, mapped_zscores = self._run()
        self.assertNotIn("pep_00", mapped_zscores.columns)
        self.assertNotIn("pep_01", mapped_zscores.columns)


# ---------------------------------------------------------------------------
# _run_iterative_process_single_pair — integration via plugin method
# ---------------------------------------------------------------------------

class TestRunIterativeProcessSinglePairIntegration(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        raw = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        scores_art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)
        self.gmt_art = _load_gmt_art(self.get_data_path("peptide-sets.tsv"))
        self.gmt_df = pd.read_csv(
            self.get_data_path("peptide-sets.tsv"), sep="\t"
        )

        process = self.plugin.methods["_process_scores"]
        self.processed_art, = process(scores=scores_art)

        compute_fit = self.plugin.methods["_compute_pair_fit_and_residuals"]
        self.spline_art, = compute_fit(
            processed_zscores=self.processed_art,
            pair="sA~sB",
            spline_type="py-smooth",
            degree=3,
        )
        self.method = self.plugin.methods["_run_iterative_process_single_pair"]

    def _run(self, **kwargs):
        result, = self.method(
            processed_zscores=self.processed_art,
            peptide_sets=self.gmt_art,
            precomputed_fit=self.spline_art,
            threshold=0.0,
            permutation_num=100,
            min_size=3,
            max_size=500,
            p_value=0.05,
            enrichment_score=1.0,
            seed=42,
            **kwargs,
        )
        return result

    def test_output_is_valid_gmt(self):
        df = self._run().view(pd.DataFrame)
        self.assertIn("term", df.columns)
        self.assertIn("gene", df.columns)

    def test_output_row_count_at_most_input_row_count(self):
        df = self._run().view(pd.DataFrame)
        self.assertLessEqual(len(df), len(self.gmt_df))

    def test_output_terms_are_subset_of_input_terms(self):
        df = self._run().view(pd.DataFrame)
        self.assertTrue(set(df["term"]).issubset(set(self.gmt_df["term"])))

    def test_with_p_value_zero_output_equals_input(self):
        # p_value=0 means row["p.adjust"] < 0 is never satisfied,
        # so no species is ever called significant and the GMT is returned
        # unchanged after the first R call exits the while-loop.
        result, = self.method(
            processed_zscores=self.processed_art,
            peptide_sets=self.gmt_art,
            precomputed_fit=self.spline_art,
            threshold=0.0,
            permutation_num=100,
            min_size=3,
            max_size=500,
            p_value=0.0,
            enrichment_score=1.0,
            seed=42,
        )
        out_df = result.view(pd.DataFrame)
        pd.testing.assert_frame_equal(
            out_df.reset_index(drop=True),
            self.gmt_df.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# make_psea_table — pipeline integration
# ---------------------------------------------------------------------------

class TestMakePseaTableIntegration(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        # Use the full 7-peptide scores-vis.tsv and the full peptide-sets.tsv.
        # With multiple gene sets in the GMT, R's GSEA can compute ranked
        # enrichment even when deltaZ values are very small.  The epitope
        # metadata is extended to cover all 9 peptides in the GMT so that
        # _count_enriched_uncollapsed never hits a KeyError regardless of
        # which peptides appear in core_enrichment.
        raw = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        self.scores_art = qiime2.Artifact.import_data(
            "FeatureTable[Zscore]", raw
        )
        self.pairs_art = _load_pairs_art(self.get_data_path("pairs.tsv"))
        self.gmt_art = _load_gmt_art(self.get_data_path("peptide-sets.tsv"))
        # Extended epitope metadata covering all 9 peptides in peptide-sets.tsv
        pep_ids = [f"pep_{i:02d}" for i in range(9)]
        species = (
            ["sp001"] * 2 + ["sp002"] + ["sp003"]
            + ["sp004"] * 3 + ["sp005"] * 2
        )
        subtypes = (
            ["H1N1", "H3N2", "Yamagata", "K12"]
            + ["HSV1"] * 3 + ["HIV1"] * 2
        )
        categories = (
            ["Viral", "Viral", "Viral", "Bacterial"]
            + ["Viral"] * 3 + ["Viral"] * 2
        )
        epi_extended = pd.DataFrame(
            {
                "SpeciesID": species,
                "ClusterID": [f"C{i+1}" for i in range(9)],
                "EpitopeWindow": [f"W{i+1}" for i in range(9)],
                "Species": [
                    "InfluenzaA", "InfluenzaA", "InfluenzaB", "EColi",
                    "HerpesV", "HerpesV", "HerpesV", "HIV", "HIV",
                ],
                "Subtype": subtypes,
                "Category": categories,
            },
            index=pd.Index(pep_ids, name="CodeName"),
        )
        self.epi_art = qiime2.Artifact.import_data(
            "FeatureData[Epitope]", epi_extended
        )
        self.pipeline = self.plugin.pipelines["make_psea_table"]

    def test_raises_when_map_true_and_peptide_metadata_missing(self):
        # peptide_metadata is required any time use_epitope_mapping=True,
        # regardless of whether precomputed epitope_map/peptide_sets_map are
        # supplied; with use_epitope_mapping=False the pipeline just runs
        # unmapped and peptide_metadata is optional.
        with self.assertRaises(Exception):
            self.pipeline(
                scores=self.scores_art,
                pairs=self.pairs_art,
                peptide_sets=self.gmt_art,
                threshold=0.0,
                use_epitope_mapping=True,
                # peptide_metadata intentionally omitted
            )

    def test_raises_when_map_with_mapped_artifacts_no_peptide_metadata(
        self
    ):
        # Even when epitope_map/peptide_sets_map are both provided,
        # peptide_metadata is still required downstream (count_enriched's
        # collapsed analysis needs it), so this must still raise.
        epi_map_art, peptide_sets_map_art = self._make_mapped_artifacts()
        with self.assertRaises(Exception):
            self.pipeline(
                scores=self.scores_art,
                pairs=self.pairs_art,
                peptide_sets=self.gmt_art,
                threshold=0.0,
                use_epitope_mapping=True,
                epitope_map=epi_map_art,
                peptide_sets_map=peptide_sets_map_art,
                # peptide_metadata intentionally omitted
            )

    def test_raises_when_partial_map_artifacts_provided(self):
        # Providing epitope_map without peptide_sets_map is invalid.
        # Note: create_epitope_map uses the input name "epitope", not
        # "peptide_metadata".
        epi_map_art, = self.plugin.methods["create_epitope_map"](
            epitope=self.epi_art, collapse="Viral"
        )
        with self.assertRaises(Exception):
            self.pipeline(
                scores=self.scores_art,
                pairs=self.pairs_art,
                peptide_sets=self.gmt_art,
                threshold=0.0,
                use_epitope_mapping=True,
                epitope_map=epi_map_art,
                # peptide_sets_map omitted → partial → error
            )

    def _make_mapped_artifacts(self):
        epi_map_art, = self.plugin.methods["create_epitope_map"](
            epitope=self.epi_art, collapse="Viral"
        )
        peptide_sets_map_art, = self.plugin.methods["taxa_to_epitope"](
            peptide_metadata=self.epi_art,
            peptide_sets=self.gmt_art,
            collapse="Viral",
        )
        return epi_map_art, peptide_sets_map_art

    def test_raises_when_map_false_and_epitope_map_provided(self):
        epi_map_art, peptide_sets_map_art = self._make_mapped_artifacts()
        with self.assertRaises(Exception):
            self.pipeline(
                scores=self.scores_art,
                pairs=self.pairs_art,
                peptide_sets=self.gmt_art,
                threshold=0.0,
                use_epitope_mapping=False,
                epitope_map=epi_map_art,
                peptide_sets_map=peptide_sets_map_art,
            )

    def test_psea_tables_keyed_by_pair_name(self):
        _, _, _, psea_tables, _ = self.pipeline(
            scores=self.scores_art,
            pairs=self.pairs_art,
            peptide_sets=self.gmt_art,
            peptide_metadata=self.epi_art,
            threshold=0.0,
            permutation_num=100,
            min_size=3,
            max_size=500,
            p_value=1.0,
            enrichment_score=0.0,
            iterative_analysis=False,
            use_epitope_mapping=False,
            seed=42,
        )
        self.assertIn("sA~sB", psea_tables)


if __name__ == "__main__":
    unittest.main()
