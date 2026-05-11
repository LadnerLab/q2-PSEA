import unittest

import numpy as np
import pandas as pd
from biom import load_table
from q2_types.feature_table import BIOMV210Format
from qiime2.plugin.testing import TestPluginBase

from qiime2 import Artifact
from q2_PSEA.actions.epitope import (
    _count_enriched_collapsed_helper,
    _count_enriched_uncollapsed,
    _count_enriched_uncollapsed_helper,
    _create_count_df,
    _create_EpitopeID_row,
    _filter_scores,
    _get_relative_enrichment_score,
    create_epitope_map,
    count_enriched,
    epitope_zscore,
    taxa_to_epitope,
)


class TestCreateEpitopeMap(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.epitope = pd.read_csv(
            self.get_data_path("epitope.tsv"), sep="\t", index_col=0
        )

    def test_viral_collapse_creates_combined_ids(self):
        result, _ = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertIn("sp001_C1_W1", result.index)
        self.assertIn("sp002_C2_W2", result.index)

    def test_non_collapsed_category_keeps_original_index(self):
        result, _ = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertIn("pep_03", result.index)

    def test_two_peptides_share_same_epitope(self):
        result, _ = create_epitope_map(self.epitope.copy(), collapse="Viral")
        codenames = result.loc["sp001_C1_W1", "CodeName"]
        self.assertEqual(set(codenames), {"pep_00", "pep_01"})

    def test_both_collapse_includes_bacterial(self):
        result, _ = create_epitope_map(self.epitope.copy(), collapse="Both")
        self.assertIn("sp003_C3_W3", result.index)

    def test_semicolon_entries_exploded_into_separate_rows(self):
        multi = pd.DataFrame(
            {
                "SpeciesID": ["sp001;sp002"],
                "ClusterID": ["C1;C2"],
                "EpitopeWindow": ["W1;W2"],
                "Species": ["InfluenzaA;InfluenzaB"],
                "Subtype": ["H1N1;Yamagata"],
                "Category": ["Viral"],
            },
            index=pd.Index(["pep_X"], name="CodeName"),
        )
        result, _ = create_epitope_map(multi, collapse="Viral")
        self.assertIn("sp001_C1_W1", result.index)
        self.assertIn("sp002_C2_W2", result.index)


class TestEpitopeZscore(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        # scores.tsv has features as rows; the view of FeatureTable[Zscore]
        # has samples as rows. epitope_zscore expects samples as rows.
        raw = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        art = Artifact.import_data("FeatureTable[Zscore]", raw)
        self.zscores = art.view(pd.DataFrame)
        self.epitope_map = pd.DataFrame(
            {
                "CodeName": {
                    "ep1": ["pep_00", "pep_01"],
                    "ep2": ["pep_02", "pep_03"],
                    "ep3": ["pep_04"],
                }
            }
        )

    def test_observations_match_epitope_ids(self):
        result = epitope_zscore(self.zscores.copy(), self.epitope_map)
        table = load_table(str(result))
        self.assertEqual(
            set(table.ids(axis="observation")), {"ep1", "ep2", "ep3"}
        )

    def test_samples_match_input_index(self):
        result = epitope_zscore(self.zscores.copy(), self.epitope_map)
        table = load_table(str(result))
        self.assertEqual(
            set(table.ids(axis="sample")), set(self.zscores.index)
        )

    def test_nan_values_filled_without_error(self):
        zscores_nan = self.zscores.copy().astype(float)
        zscores_nan.iloc[0, 0] = np.nan
        result = epitope_zscore(zscores_nan, self.epitope_map)
        self.assertIsInstance(result, BIOMV210Format)


class TestTaxaToEpitope(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        raw = pd.read_csv(
            self.get_data_path("epitope.tsv"), sep="\t", index_col=0
        )
        self.epitope_map_viral, _ = create_epitope_map(
            raw.copy(), collapse="Viral"
        )
        self.epitope_map_both, _ = create_epitope_map(
            raw.copy(), collapse="Both"
        )

    def test_term_contains_species_ids(self):
        result = taxa_to_epitope(self.epitope_map_viral.copy())
        self.assertIn("sp001", result["term"].values)

    def test_gene_contains_viral_epitope_ids(self):
        result = taxa_to_epitope(self.epitope_map_viral.copy())
        self.assertIn("sp001_C1_W1", result["gene"].values)

    def test_bacterial_gene_is_original_index_when_collapse_viral(self):
        result = taxa_to_epitope(self.epitope_map_viral.copy())
        self.assertIn("pep_03", result["gene"].values)

    def test_both_collapse_gives_bacterial_epitope_id(self):
        result = taxa_to_epitope(self.epitope_map_both.copy())
        self.assertIn("sp003_C3_W3", result["gene"].values)


class TestFilterScores(TestPluginBase):
    package = "q2_PSEA.tests"

    def _scores(self, rows):
        return {"pair1": pd.DataFrame(rows)}

    def test_filters_rows_above_p_value(self):
        scores = self._scores([
            {"p.adjust": 0.01, "enrichmentScore": 2.0},
            {"p.adjust": 0.5, "enrichmentScore": 2.0},
        ])
        result = _filter_scores(scores, 0.05, 1.0, True)
        self.assertEqual(len(result), 1)

    def test_concatenates_multiple_pair_tables(self):
        scores = {
            "pairA": pd.DataFrame(
                [{"p.adjust": 0.01, "enrichmentScore": 2.0}]
            ),
            "pairB": pd.DataFrame(
                [{"p.adjust": 0.01, "enrichmentScore": 2.0}]
            ),
        }
        result = _filter_scores(scores, 0.05, 1.0, True)
        self.assertEqual(len(result), 2)

    def test_all_filtered_returns_empty(self):
        scores = self._scores([{"p.adjust": 0.9, "enrichmentScore": 0.0}])
        result = _filter_scores(scores, 0.05, 1.0, True)
        self.assertEqual(len(result), 0)


class TestCountEnriched(TestPluginBase):
    package = "q2_PSEA.tests"

    def _empty(self):
        return {"epitope": {}, "subtype": {}}

    def _make_fixtures(self):
        epitope_map = pd.DataFrame(
            {"CodeName": [["pep1"]]},
            index=pd.Index(["ep1"]),
        )
        zscores = pd.DataFrame({"pep1": [1.0]}, index=["sA"])
        mapped_zscores = pd.DataFrame({"ep1": [1.0]}, index=["sA"])
        return (
            epitope_map,
            zscores,
            zscores.copy(),
            mapped_zscores,
            mapped_zscores.copy(),
        )

    def test_initializes_epitope_key(self):
        counts = self._empty()
        emap, zsc, pzsc, mzsc, mpzsc = self._make_fixtures()
        _count_enriched_collapsed_helper(
            counts, zsc, emap, pzsc, mzsc, mpzsc,
            "sp001", "InfluenzaA", "ep1", "H1N1"
        )
        self.assertEqual(counts["epitope"]["sp001"]["InfluenzaA"]["ep1"], 1)

    def test_second_call_increments_count(self):
        counts = self._empty()
        emap, zsc, pzsc, mzsc, mpzsc = self._make_fixtures()
        kwargs = dict(
            zscores=zsc, epitope_map=emap, processed_zscores=pzsc,
            mapped_zscores=mzsc, mapped_processed_zscores=mpzsc,
            species_id="sp001", species_name="InfluenzaA",
            epitope="ep1", subtype="H1N1",
        )
        _count_enriched_collapsed_helper(counts, **kwargs)
        _count_enriched_collapsed_helper(counts, **kwargs)
        self.assertEqual(counts["epitope"]["sp001"]["InfluenzaA"]["ep1"], 2)

    def test_tracks_subtype_separately(self):
        counts = self._empty()
        emap, zsc, pzsc, mzsc, mpzsc = self._make_fixtures()
        _count_enriched_collapsed_helper(
            counts, zsc, emap, pzsc, mzsc, mpzsc,
            "sp001", "InfluenzaA", "ep1", "H1N1"
        )
        self.assertIn("H1N1", counts["subtype"]["sp001"]["InfluenzaA"])
        self.assertEqual(
            counts["subtype"]["sp001"]["InfluenzaA"]["H1N1"]["Epitope Counts"],
            1,
        )


class TestEnrichedSubtypes(TestPluginBase):
    package = "q2_PSEA.tests"

    def _scores(self, rows):
        return {"pair1": pd.DataFrame(rows)}

    def _subtypes(self):
        # Epitope IDs that contain "Peptide" trigger the collapsed counting
        # path in _count_enriched_collapsed
        return pd.DataFrame(
            {
                "CodeName": [["pep_00", "pep_01"], ["pep_02"]],
                "Subtype": [["H1N1", "H3N2"], ["Yamagata"]],
                "Category": ["Viral", "Viral"],
            },
            index=pd.Index(
                ["sp001_Peptide_W1", "sp002_Peptide_W2"], name="EpitopeID"
            ),
        )

    def test_returns_dict_with_two_default_keys(self):
        scores = self._scores([{
            "p.adjust": 0.01,
            "enrichmentScore": 2.0,
            "core_enrichment": "sp001_Peptide_W1",
            "species_name": "InfluenzaA",
        }])
        zscores = pd.DataFrame(
            {"pep_00": [1.0], "pep_01": [1.5]}, index=["sA"]
        )
        mapped_zscores = pd.DataFrame(
            {"sp001_Peptide_W1": [1.0]}, index=["sA"]
        )
        result = count_enriched(
            scores,
            zscores,
            zscores,
            epitope_map=self._subtypes(),
            mapped_zscores=mapped_zscores,
            mapped_processed_zscores=mapped_zscores,
        )
        self.assertEqual(set(result.keys()), {"epitope", "subtype"})


# ---------------------------------------------------------------------------
# _create_EpitopeID_row — unit tests
# ---------------------------------------------------------------------------

class TestCreateEpitopeIDRow(TestPluginBase):
    package = "q2_PSEA.tests"

    def _row(self, species="InfluenzaA", subtype="H1N1", species_id="sp001",
             cluster="C1", window="W1", category="Viral", pep="pep_00"):
        return pd.DataFrame(
            [[species, subtype, species_id, cluster, window, category]],
            columns=["Species", "Subtype", "SpeciesID", "ClusterID",
                     "EpitopeWindow", "Category"],
            index=pd.Index([pep], name="CodeName"),
        )

    def test_viral_category_with_viral_collapse_gets_combined_id(self):
        result = _create_EpitopeID_row(self._row(), "Viral")
        self.assertIn("sp001_C1_W1", result["EpitopeID"].values)

    def test_bacterial_category_with_viral_collapse_keeps_original_name(self):
        result = _create_EpitopeID_row(
            self._row(category="Bacterial", pep="pep_bact"), "Viral"
        )
        self.assertIn("pep_bact", result["EpitopeID"].values)

    def test_both_collapse_gives_combined_id_for_bacterial(self):
        result = _create_EpitopeID_row(
            self._row(species_id="sp003", cluster="C3", window="W3",
                      category="Bacterial", pep="pep_03"),
            "Both",
        )
        self.assertIn("sp003_C3_W3", result["EpitopeID"].values)

    def test_semicolon_entries_explode_into_multiple_rows(self):
        df = pd.DataFrame(
            [["InfluenzaA;InfluenzaB", "H1N1;Yamagata",
              "sp001;sp002", "C1;C2", "W1;W2", "Viral"]],
            columns=["Species", "Subtype", "SpeciesID", "ClusterID",
                     "EpitopeWindow", "Category"],
            index=pd.Index(["pep_multi"], name="CodeName"),
        )
        result = _create_EpitopeID_row(df, "Viral")
        self.assertIn("sp001_C1_W1", result["EpitopeID"].values)
        self.assertIn("sp002_C2_W2", result["EpitopeID"].values)

    def test_nan_subtype_filled_with_subtypena(self):
        # Two rows so the column dtype stays object (not float64) even with NaN
        df = pd.DataFrame(
            [
                ["InfluenzaA", "H1N1", "sp001", "C1", "W1", "Viral"],
                ["InfluenzaB", np.nan, "sp002", "C2", "W2", "Viral"],
            ],
            columns=["Species", "Subtype", "SpeciesID", "ClusterID",
                     "EpitopeWindow", "Category"],
            index=pd.Index(["pep_01", "pep_02"], name="CodeName"),
        )
        result = _create_EpitopeID_row(df, "Viral")
        self.assertFalse(result["Subtype"].isna().any())
        self.assertIn("subtypeNA", result["Subtype"].values)

    def test_empty_subtype_string_replaced_with_subtypena(self):
        df = pd.DataFrame(
            [["InfluenzaA", "", "sp001", "C1", "W1", "Viral"]],
            columns=["Species", "Subtype", "SpeciesID", "ClusterID",
                     "EpitopeWindow", "Category"],
            index=pd.Index(["pep_empty"], name="CodeName"),
        )
        result = _create_EpitopeID_row(df, "Viral")
        self.assertIn("subtypeNA", result["Subtype"].values)


# ---------------------------------------------------------------------------
# _count_enriched_uncollapsed_helper — unit tests
# ---------------------------------------------------------------------------

class TestCountEnrichedUncollapsedHelper(TestPluginBase):
    package = "q2_PSEA.tests"

    def _empty_counts(self):
        return {"peptide": {}, "subtype": {}}

    def _fixtures(self):
        epitope = pd.DataFrame(
            {"Subtype": ["H1N1"]},
            index=pd.Index(["pep1"], name="CodeName"),
        )
        zscores = pd.DataFrame({"pep1": [1.0, 2.0]}, index=["sA", "sB"])
        return epitope, zscores, zscores.copy()

    def test_first_call_initializes_all_nested_keys(self):
        counts = self._empty_counts()
        epitope, zsc, pzsc = self._fixtures()
        _count_enriched_uncollapsed_helper(
            counts, epitope, zsc, pzsc, "pep1", "sp1", "InfluenzaA"
        )
        self.assertIn("sp1", counts["peptide"])
        self.assertIn("InfluenzaA", counts["peptide"]["sp1"])
        self.assertEqual(counts["peptide"]["sp1"]["InfluenzaA"]["pep1"], 1)

    def test_second_call_increments_peptide_count(self):
        counts = self._empty_counts()
        epitope, zsc, pzsc = self._fixtures()
        _count_enriched_uncollapsed_helper(
            counts, epitope, zsc, pzsc, "pep1", "sp1", "InfluenzaA"
        )
        _count_enriched_uncollapsed_helper(
            counts, epitope, zsc, pzsc, "pep1", "sp1", "InfluenzaA"
        )
        self.assertEqual(counts["peptide"]["sp1"]["InfluenzaA"]["pep1"], 2)

    def test_subtype_dict_has_correct_initial_structure(self):
        counts = self._empty_counts()
        epitope, zsc, pzsc = self._fixtures()
        _count_enriched_uncollapsed_helper(
            counts, epitope, zsc, pzsc, "pep1", "sp1", "InfluenzaA"
        )
        entry = counts["subtype"]["sp1"]["InfluenzaA"]["H1N1"]
        self.assertEqual(entry["Epitope Counts"], 1)
        self.assertIn("Relative Enrichment Score", entry)
        self.assertIn("Relative Processed Enrichment Score", entry)

    def test_relative_enrichment_score_is_sum_of_zscores_column(self):
        counts = self._empty_counts()
        epitope, zsc, pzsc = self._fixtures()
        _count_enriched_uncollapsed_helper(
            counts, epitope, zsc, pzsc, "pep1", "sp1", "InfluenzaA"
        )
        # sum of zscores["pep1"] = 1.0 + 2.0 = 3.0
        score = counts["subtype"]["sp1"]["InfluenzaA"]["H1N1"][
            "Relative Enrichment Score"
        ]
        self.assertAlmostEqual(score, 3.0)

    def test_second_call_increments_subtype_epitope_count(self):
        counts = self._empty_counts()
        epitope, zsc, pzsc = self._fixtures()
        _count_enriched_uncollapsed_helper(
            counts, epitope, zsc, pzsc, "pep1", "sp1", "InfluenzaA"
        )
        _count_enriched_uncollapsed_helper(
            counts, epitope, zsc, pzsc, "pep1", "sp1", "InfluenzaA"
        )
        count = \
            counts["subtype"]["sp1"]["InfluenzaA"]["H1N1"]["Epitope Counts"]
        self.assertEqual(count, 2)


# ---------------------------------------------------------------------------
# _count_enriched_uncollapsed — unit tests
# ---------------------------------------------------------------------------

class TestCountEnrichedUncollapsed(TestPluginBase):
    package = "q2_PSEA.tests"

    def _fixtures(self):
        epitope = pd.DataFrame(
            {"Subtype": ["H1N1", "H3N2"]},
            index=pd.Index(["pep1", "pep2"], name="CodeName"),
        )
        zscores = pd.DataFrame(
            {"pep1": [1.0], "pep2": [2.0]}, index=["sA"]
        )
        return epitope, zscores, zscores.copy()

    def _filtered(self, enrichment, species_name=None):
        row = {"core_enrichment": enrichment}
        if species_name is not None:
            row["species_name"] = species_name
        return pd.DataFrame([row])

    def test_returns_dict_with_peptide_and_subtype_keys(self):
        epitope, zsc, pzsc = self._fixtures()
        result = _count_enriched_uncollapsed(
            epitope, zsc, pzsc, self._filtered("pep1")
        )
        self.assertIn("peptide", result)
        self.assertIn("subtype", result)

    def test_result_values_are_dataframes(self):
        epitope, zsc, pzsc = self._fixtures()
        result = _count_enriched_uncollapsed(
            epitope, zsc, pzsc, self._filtered("pep1")
        )
        self.assertIsInstance(result["peptide"], pd.DataFrame)
        self.assertIsInstance(result["subtype"], pd.DataFrame)

    def test_multiindex_has_three_levels(self):
        epitope, zsc, pzsc = self._fixtures()
        result = _count_enriched_uncollapsed(
            epitope, zsc, pzsc, self._filtered("pep1")
        )
        self.assertEqual(result["peptide"].index.nlevels, 3)
        self.assertEqual(result["subtype"].index.nlevels, 3)

    def test_species_name_column_used_when_present(self):
        epitope, zsc, pzsc = self._fixtures()
        result = _count_enriched_uncollapsed(
            epitope, zsc, pzsc,
            self._filtered("pep1", species_name="InfluenzaA"),
        )
        names = result["peptide"].index.get_level_values(1)
        self.assertIn("InfluenzaA", names)

    def test_slash_delimited_enrichment_counts_each_peptide(self):
        epitope, zsc, pzsc = self._fixtures()
        result = _count_enriched_uncollapsed(
            epitope, zsc, pzsc, self._filtered("pep1/pep2")
        )
        peptide_level = result["peptide"].index.get_level_values(2)
        self.assertIn("pep1", peptide_level)
        self.assertIn("pep2", peptide_level)


# ---------------------------------------------------------------------------
# _get_relative_enrichment_score — unit tests
# ---------------------------------------------------------------------------

class TestGetRelativeEnrichmentScore(TestPluginBase):
    package = "q2_PSEA.tests"

    def _emap(self, codenames):
        return pd.DataFrame(
            {"CodeName": [codenames]},
            index=pd.Index(["ep1"]),
        )

    def test_single_sample_single_codename(self):
        emap = self._emap(["pep1"])
        zscores = pd.DataFrame({"pep1": [2.0]}, index=["sA"])
        mapped_zscores = pd.DataFrame({"ep1": [4.0]}, index=["sA"])
        result = _get_relative_enrichment_score(
            zscores, mapped_zscores, "ep1", emap
        )
        self.assertAlmostEqual(result, 0.5)  # 2.0 / 4.0

    def test_multiple_samples_sums_all_normalized_values(self):
        emap = self._emap(["pep1"])
        zscores = pd.DataFrame({"pep1": [1.0, 3.0]}, index=["sA", "sB"])
        mapped_zscores = pd.DataFrame({"ep1": [2.0, 6.0]}, index=["sA", "sB"])
        result = _get_relative_enrichment_score(
            zscores, mapped_zscores, "ep1", emap
        )
        self.assertAlmostEqual(result, 1.0)  # 1/2 + 3/6

    def test_multiple_codenames_sum_contributions_from_each(self):
        emap = self._emap(["pep1", "pep2"])
        zscores = pd.DataFrame({"pep1": [2.0], "pep2": [1.0]}, index=["sA"])
        mapped_zscores = pd.DataFrame({"ep1": [4.0]}, index=["sA"])
        result = _get_relative_enrichment_score(
            zscores, mapped_zscores, "ep1", emap
        )
        self.assertAlmostEqual(result, 0.75)  # 2/4 + 1/4

    def test_negative_zscores_divide_correctly(self):
        emap = self._emap(["pep1"])
        zscores = pd.DataFrame({"pep1": [-1.0]}, index=["sA"])
        mapped_zscores = pd.DataFrame({"ep1": [-2.0]}, index=["sA"])
        result = _get_relative_enrichment_score(
            zscores, mapped_zscores, "ep1", emap
        )
        self.assertAlmostEqual(result, 0.5)  # -1.0 / -2.0


# ---------------------------------------------------------------------------
# _create_count_df — unit tests
# ---------------------------------------------------------------------------

class TestCreateCountDf(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_single_entry_creates_three_level_multiindex(self):
        df = _create_count_df("peptide", {"sp1": {"InfluenzaA": {"pep1": 1}}})
        self.assertEqual(df.index.nlevels, 3)

    def test_index_level_names_use_capitalized_key_plus_s(self):
        df = _create_count_df("peptide", {"sp1": {"InfluenzaA": {"pep1": 1}}})
        self.assertEqual(df.index.names[0], "Species ID Called")
        self.assertEqual(df.index.names[1], "Species Called")
        self.assertEqual(df.index.names[2], "Peptides")

    def test_subtype_key_produces_subtypes_level_name(self):
        df = _create_count_df("subtype", {"sp1": {"InfluenzaA": {"H1N1": 2}}})
        self.assertEqual(df.index.names[2], "Subtypes")

    def test_count_value_appears_in_dataframe(self):
        df = _create_count_df("peptide", {"sp1": {"InfluenzaA": {"pep1": 7}}})
        self.assertEqual(df.loc[("sp1", "InfluenzaA", "pep1"), 0], 7)

    def test_multiple_innermost_entries_produce_multiple_rows(self):
        df = _create_count_df(
            "peptide", {"sp1": {"InfluenzaA": {"pep1": 1, "pep2": 3}}}
        )
        self.assertEqual(len(df), 2)

    def test_empty_dict_returns_empty_dataframe_with_correct_index_names(self):
        df = _create_count_df("epitope", {})
        self.assertEqual(len(df), 0)
        self.assertEqual(df.index.names[2], "Epitopes")


if __name__ == "__main__":
    unittest.main()
