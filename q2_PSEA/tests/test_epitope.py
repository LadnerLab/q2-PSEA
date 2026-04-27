import unittest

import numpy as np
import pandas as pd
from biom import load_table
from q2_types.feature_table import BIOMV210Format
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.actions.epitope import (
    _count_enriched,
    _filter_scores,
    _find_split_value,
    _get_keys,
    create_epitope_map,
    enriched_subtypes,
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

    def test_returns_dataframe(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertIsInstance(result, pd.DataFrame)

    def test_index_name_is_epitope_id(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertEqual(result.index.name, "EpitopeID")

    def test_viral_collapse_creates_combined_ids(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertIn("sp001_C1_W1", result.index)
        self.assertIn("sp002_C2_W2", result.index)

    def test_non_collapsed_category_keeps_original_index(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertIn("pep_03", result.index)

    def test_codenames_aggregated_as_list(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertIsInstance(result.loc["sp001_C1_W1", "CodeName"], list)

    def test_two_peptides_share_same_epitope(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        codenames = result.loc["sp001_C1_W1", "CodeName"]
        self.assertEqual(set(codenames), {"pep_00", "pep_01"})

    def test_both_collapse_includes_bacterial(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Both")
        self.assertIn("sp003_C3_W3", result.index)

    def test_category_conflict_raises_value_error(self):
        bad = pd.DataFrame(
            {
                "SpeciesID": ["sp001", "sp001"],
                "ClusterID": ["C1", "C1"],
                "EpitopeWindow": ["W1", "W1"],
                "Species": ["InfluenzaA", "EColi"],
                "Subtype": ["H1N1", "K12"],
                "Category": ["Viral", "Bacterial"],
            },
            index=pd.Index(["pep_A", "pep_B"], name="CodeName"),
        )
        with self.assertRaises(ValueError):
            create_epitope_map(bad, collapse="Both")

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
        result = create_epitope_map(multi, collapse="Viral")
        self.assertIn("sp001_C1_W1", result.index)
        self.assertIn("sp002_C2_W2", result.index)


class TestEpitopeZscore(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        # scores.tsv has features as rows; the view of FeatureTable[Zscore]
        # has samples as rows. epitope_zscore expects samples as rows.
        import qiime2
        raw = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)
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

    def test_returns_biomv210_format(self):
        result = epitope_zscore(self.zscores.copy(), self.epitope_map)
        self.assertIsInstance(result, BIOMV210Format)

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
        self.epitope = pd.read_csv(
            self.get_data_path("epitope.tsv"), sep="\t", index_col=0
        )

    def test_returns_dataframe(self):
        result = taxa_to_epitope(self.epitope.copy(), collapse="Viral")
        self.assertIsInstance(result, pd.DataFrame)

    def test_columns_are_term_and_gene(self):
        result = taxa_to_epitope(self.epitope.copy(), collapse="Viral")
        self.assertListEqual(list(result.columns), ["term", "gene"])

    def test_term_contains_species_ids(self):
        result = taxa_to_epitope(self.epitope.copy(), collapse="Viral")
        self.assertIn("sp001", result["term"].values)

    def test_gene_contains_viral_epitope_ids(self):
        result = taxa_to_epitope(self.epitope.copy(), collapse="Viral")
        self.assertIn("sp001_C1_W1", result["gene"].values)

    def test_bacterial_gene_is_original_index_when_collapse_viral(self):
        result = taxa_to_epitope(self.epitope.copy(), collapse="Viral")
        self.assertIn("pep_03", result["gene"].values)

    def test_both_collapse_gives_bacterial_epitope_id(self):
        result = taxa_to_epitope(self.epitope.copy(), collapse="Both")
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


class TestGetKeys(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_returns_product_of_split_values_and_default_keys(self):
        subtypes = pd.DataFrame({"Category": ["Viral", "Bacterial"]})
        result = _get_keys(subtypes, "Category", ["species", "subspecies"])
        self.assertIn("Viral-species", result)
        self.assertIn("Bacterial-subspecies", result)
        self.assertEqual(len(result), 4)

    def test_raises_key_error_for_missing_column(self):
        subtypes = pd.DataFrame({"Category": ["Viral"]})
        with self.assertRaises(KeyError):
            _get_keys(subtypes, "NotAColumn", ["species"])


class TestFindSplitValue(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_returns_empty_string_when_no_split_column(self):
        hit = pd.Series({"Category": "Viral"})
        self.assertEqual(_find_split_value(hit, None, 0), "")

    def test_returns_scalar_value_with_trailing_dash(self):
        hit = pd.Series({"Category": "Viral"})
        self.assertEqual(_find_split_value(hit, "Category", 0), "Viral-")

    def test_returns_indexed_list_element_with_trailing_dash(self):
        hit = pd.Series({"Category": ["Viral", "Bacterial"]})
        self.assertEqual(_find_split_value(hit, "Category", 1), "Bacterial-")


class TestCountEnriched(TestPluginBase):
    package = "q2_PSEA.tests"

    def _empty(self):
        return {"species": {}, "subspecies": {}, "species-epitope": {}}

    def test_initializes_species_key(self):
        counts = self._empty()
        _count_enriched(counts, "InfluenzaA", "InfluenzaA:H1N1", "ep1", "")
        self.assertEqual(counts["species"]["InfluenzaA"], 1)

    def test_second_call_increments_count(self):
        counts = self._empty()
        _count_enriched(counts, "InfluenzaA", "InfluenzaA:H1N1", "ep1", "")
        _count_enriched(counts, "InfluenzaA", "InfluenzaA:H1N1", "ep1", "")
        self.assertEqual(counts["species"]["InfluenzaA"], 2)

    def test_split_value_prefix_applied(self):
        counts = {
            "Viral-species": {},
            "Viral-subspecies": {},
            "Viral-species-epitope": {},
        }
        _count_enriched(
            counts, "InfluenzaA", "InfluenzaA:H1N1", "ep1", "Viral-"
        )
        self.assertIn("Viral-InfluenzaA", counts["Viral-species"])


class TestEnrichedSubtypes(TestPluginBase):
    package = "q2_PSEA.tests"

    def _scores(self, rows):
        return {"pair1": pd.DataFrame(rows)}

    def _subtypes(self):
        return pd.DataFrame(
            {
                "CodeName": [["pep_00", "pep_01"], ["pep_02"]],
                "Subtype": [["H1N1", "H3N2"], ["Yamagata"]],
                "Category": ["Viral", "Viral"],
            },
            index=pd.Index(["sp001_C1_W1", "sp002_C2_W2"], name="EpitopeID"),
        )

    def test_returns_dict_with_three_default_keys(self):
        scores = self._scores([{
            "p.adjust": 0.9,
            "enrichmentScore": 0.0,
            "core_enrichment": "sp001_C1_W1",
            "species_name": "InfluenzaA",
        }])
        result = enriched_subtypes(scores, self._subtypes())
        self.assertEqual(
            set(result.keys()), {"species", "subspecies", "species-epitope"}
        )

    def test_no_significant_rows_returns_empty_dataframes(self):
        scores = self._scores([{
            "p.adjust": 0.9,
            "enrichmentScore": 0.0,
            "core_enrichment": "sp001_C1_W1",
            "species_name": "InfluenzaA",
        }])
        result = enriched_subtypes(scores, self._subtypes())
        for val in result.values():
            self.assertEqual(len(val), 0)

    def test_split_column_multiplies_keys(self):
        scores = self._scores([{
            "p.adjust": 0.9,
            "enrichmentScore": 0.0,
            "core_enrichment": "sp001_C1_W1",
            "species_name": "InfluenzaA",
        }])
        result = enriched_subtypes(
            scores, self._subtypes(), split_column="Category"
        )
        self.assertIn("Viral-species", result)
        self.assertIn("Viral-subspecies", result)

    def test_invalid_split_column_raises_key_error(self):
        scores = self._scores([{
            "p.adjust": 0.01,
            "enrichmentScore": 2.0,
            "core_enrichment": "sp001_C1_W1",
            "species_name": "InfluenzaA",
        }])
        with self.assertRaises(KeyError):
            enriched_subtypes(
                scores, self._subtypes(), split_column="NotAColumn"
            )


if __name__ == "__main__":
    unittest.main()
