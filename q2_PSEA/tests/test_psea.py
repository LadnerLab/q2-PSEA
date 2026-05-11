import unittest
from math import log

import numpy as np
import pandas as pd
import qiime2
from pandas.testing import assert_series_equal
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.actions.psea import (
    _compute_pair_fit_and_residuals,
    _filter_peptide_sets,
    _get_mapped_features,
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
            sample_a="sA",
            sample_b="sB",
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
# create_fgsea_table_for_pair — full R integration
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
            sample_a="sA",
            sample_b="sB",
            spline_type="py-smooth",
            degree=3,
        )
        self.method = self.plugin.methods["create_fgsea_table_for_pair"]

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
            psea_tables={"pair1": art},
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

    def test_multiple_pairs_accumulate_events(self):
        art1 = self._make_psea_art(
            [{"ID": "sp1", "NES": 2.0, "p.adjust": 0.01}]
        )
        art2 = self._make_psea_art(
            [{"ID": "sp1", "NES": 2.0, "p.adjust": 0.01}]
        )
        pos, _ = self.method(
            psea_tables={"pairA": art1, "pairB": art2},
            p_value=0.05, enrichment_score=1.0, taxa_access="ID",
        )
        df = pos.view(pd.DataFrame)
        self.assertEqual(df.iloc[0]["Events"], 2)

    def test_output_sorted_descending_by_events(self):
        art1 = self._make_psea_art([
            {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
            {"ID": "sp2", "NES": 2.0, "p.adjust": 0.01},
        ])
        art2 = self._make_psea_art([
            {"ID": "sp1", "NES": 2.0, "p.adjust": 0.01},
        ])
        pos, _ = self.method(
            psea_tables={"pA": art1, "pB": art2},
            p_value=0.05, enrichment_score=1.0, taxa_access="ID",
        )
        df = pos.view(pd.DataFrame)
        self.assertEqual(df.iloc[0]["Species"], "sp1")

    def test_empty_result_when_no_rows(self):
        art = self._make_psea_art([{"ID": "sp1", "NES": 0.1, "p.adjust": 0.9}])
        pos, neg = self.method(
            psea_tables={"p1": art},
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
            self.view, "sA", "sB", "py-smooth", 3, **kwargs
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

    def test_epitope_map_replaces_peptide_ids_in_maxz_and_deltaz(self):
        peps = list(self.view.columns)
        mapping = {f"ep_{i}": [peps[i * 2], peps[i * 2 + 1]]
                   for i in range(len(peps) // 2)}
        emap = pd.DataFrame({"CodeName": mapping})
        df = self._call(epitope_map=emap)
        for ep in mapping:
            self.assertIn(ep, df["maxZ"].dropna().index)
            self.assertIn(ep, df["deltaZ"].dropna().index)
        # Peptides that were mapped to epitopes should not appear directly
        mapped_peps = [p for peps_list in mapping.values() for p in peps_list]
        for pep in mapped_peps:
            self.assertNotIn(pep, df["maxZ"].dropna().index)
            self.assertNotIn(pep, df["deltaZ"].dropna().index)


# ---------------------------------------------------------------------------
# _get_mapped_features — unit tests
# ---------------------------------------------------------------------------

class TestGetMappedFeatures(TestPluginBase):
    package = "q2_PSEA.tests"

    # epitope_map: indexed by EpitopeID, 'CodeName' cell is a list of peptide
    # names that belong to that epitope (mirrors create_epitope_map output).
    # peptide_map: indexed by CodeName, 'EpitopeID' cell is a list of epitope
    # IDs that share that peptide (mirrors create_epitope_map output).

    def _make_maps(self, ep_to_peps, pep_to_eps):
        """Build minimal epitope_map and peptide_map fixtures.

        ep_to_peps : dict[str, list[str]]   epitope → peptides
        pep_to_eps : dict[str, list[str]]   peptide → epitopes
        """
        epitope_map = pd.DataFrame(
            {"CodeName": list(ep_to_peps.values())},
            index=pd.Index(list(ep_to_peps.keys()), name="EpitopeID"),
        )
        peptide_map = pd.DataFrame(
            {"EpitopeID": list(pep_to_eps.values())},
            index=pd.Index(list(pep_to_eps.keys()), name="CodeName"),
        )
        return epitope_map, peptide_map

    def test_single_feature_returns_sibling_epitopes(self):
        # ep1 maps to pep1; pep1 is also shared by ep2
        emap, pmap = self._make_maps(
            {"ep1": ["pep1"], "ep2": ["pep1"]},
            {"pep1": ["ep1", "ep2"]},
        )
        result = _get_mapped_features(emap, pmap, {"ep1"})
        self.assertEqual(result, {"ep1", "ep2"})

    def test_feature_with_no_shared_peptide_returns_only_itself(self):
        emap, pmap = self._make_maps(
            {"ep1": ["pep1"], "ep2": ["pep2"]},
            {"pep1": ["ep1"], "pep2": ["ep2"]},
        )
        result = _get_mapped_features(emap, pmap, {"ep1"})
        self.assertEqual(result, {"ep1"})

    def test_multiple_features_union_all_siblings(self):
        emap, pmap = self._make_maps(
            {"ep1": ["pep1"], "ep2": ["pep1"],
             "ep3": ["pep2"], "ep4": ["pep2"]},
            {"pep1": ["ep1", "ep2"], "pep2": ["ep3", "ep4"]},
        )
        result = _get_mapped_features(emap, pmap, {"ep1", "ep3"})
        self.assertEqual(result, {"ep1", "ep2", "ep3", "ep4"})

    def test_epitope_mapping_multiple_peptides(self):
        # ep1 maps to two peptides; each peptide brings in another epitope
        emap, pmap = self._make_maps(
            {"ep1": ["pep1", "pep2"], "ep2": ["pep1"], "ep3": ["pep2"]},
            {"pep1": ["ep1", "ep2"], "pep2": ["ep1", "ep3"]},
        )
        result = _get_mapped_features(emap, pmap, {"ep1"})
        self.assertEqual(result, {"ep1", "ep2", "ep3"})

    def test_empty_features_returns_empty_set(self):
        emap, pmap = self._make_maps(
            {"ep1": ["pep1"]},
            {"pep1": ["ep1"]},
        )
        result = _get_mapped_features(emap, pmap, set())
        self.assertEqual(result, set())


# ---------------------------------------------------------------------------
# _filter_peptide_sets — unit tests
# ---------------------------------------------------------------------------

class TestFilterPeptideSets(TestPluginBase):
    package = "q2_PSEA.tests"

    def _psea(self, rows):
        return pd.DataFrame(rows)

    def _gmt(self, rows):
        return pd.DataFrame(rows, columns=["term", "gene"])

    def test_significant_row_removes_its_peptides_from_other_species(self):
        psea = self._psea([
            {"ID": "sp1", "p.adjust": 0.01, "NES": 2.0,
             "all_tested_peptides": "pep1/pep2"},
        ])
        gmt = self._gmt([
            ("sp1", "pep1"), ("sp1", "pep2"),
            ("sp2", "pep1"), ("sp2", "pep3"),
        ])
        updated, tested, sig_found = _filter_peptide_sets(
            psea, gmt, set(), 0.05, 1.0, True
        )
        self.assertTrue(sig_found)
        self.assertIn("sp1", tested)
        sp2_genes = updated[updated["term"] == "sp2"]["gene"].tolist()
        self.assertNotIn("pep1", sp2_genes)
        self.assertIn("pep3", sp2_genes)

    def test_significant_species_own_rows_are_preserved(self):
        psea = self._psea([
            {"ID": "sp1", "p.adjust": 0.01, "NES": 2.0,
             "all_tested_peptides": "pep1"},
        ])
        gmt = self._gmt([("sp1", "pep1"), ("sp2", "pep1")])
        updated, _, _ = _filter_peptide_sets(
            psea, gmt, set(), 0.05, 1.0, True
        )
        sp1_genes = updated[updated["term"] == "sp1"]["gene"].tolist()
        self.assertIn("pep1", sp1_genes)

    def test_already_tested_species_is_skipped(self):
        psea = self._psea([
            {"ID": "sp1", "p.adjust": 0.01, "NES": 2.0,
             "all_tested_peptides": "pep1"},
            {"ID": "sp2", "p.adjust": 0.02, "NES": 2.0,
             "all_tested_peptides": "pep2"},
        ])
        gmt = self._gmt([("sp1", "pep1"), ("sp2", "pep2")])
        initial = {"sp1"}
        _, tested, sig_found = _filter_peptide_sets(
            psea, gmt, initial, 0.05, 1.0, True
        )
        # sp2 should be the newly added species (sp1 was already tested)
        self.assertTrue(sig_found)
        self.assertIn("sp2", tested)
        self.assertEqual(tested - {"sp1"}, {"sp2"})

    def test_no_significant_row_returns_sig_found_false(self):
        psea = self._psea([
            {"ID": "sp1", "p.adjust": 0.9, "NES": 2.0,
             "all_tested_peptides": "pep1"},
        ])
        gmt = self._gmt([("sp1", "pep1")])
        _, _, sig_found = _filter_peptide_sets(
            psea, gmt, set(), 0.05, 1.0, True
        )
        self.assertFalse(sig_found)

    def test_include_negative_enrichment_false_ignores_negative_nes(self):
        psea = self._psea([
            {"ID": "sp1", "p.adjust": 0.01, "NES": -2.0,
             "all_tested_peptides": "pep1"},
        ])
        gmt = self._gmt([("sp1", "pep1")])
        _, _, sig_found = _filter_peptide_sets(
            psea, gmt, set(), 0.05, 1.0, False
        )
        self.assertFalse(sig_found)

    def test_include_negative_enrichment_false_accepts_positive_nes(self):
        psea = self._psea([
            {"ID": "sp1", "p.adjust": 0.01, "NES": 2.0,
             "all_tested_peptides": "pep1"},
        ])
        gmt = self._gmt([("sp1", "pep1"), ("sp2", "pep2")])
        _, tested, sig_found = _filter_peptide_sets(
            psea, gmt, set(), 0.05, 1.0, False
        )
        self.assertTrue(sig_found)
        self.assertIn("sp1", tested)

    def test_lowest_p_adjust_row_is_chosen_first(self):
        # sp2 has lower p.adjust so it should be selected over sp1
        psea = self._psea([
            {"ID": "sp1", "p.adjust": 0.04, "NES": 2.0,
             "all_tested_peptides": "pep1"},
            {"ID": "sp2", "p.adjust": 0.01, "NES": 2.0,
             "all_tested_peptides": "pep2"},
        ])
        gmt = self._gmt([("sp1", "pep1"), ("sp2", "pep2")])
        _, tested, _ = _filter_peptide_sets(
            psea, gmt, set(), 0.05, 1.0, True
        )
        self.assertIn("sp2", tested)
        self.assertNotIn("sp1", tested)


if __name__ == "__main__":
    unittest.main()
