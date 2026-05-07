import unittest

import numpy as np
import pandas as pd
from biom import load_table
from q2_types.feature_table import BIOMV210Format
from qiime2.plugin.testing import TestPluginBase

from qiime2 import Artifact
from q2_PSEA.actions.epitope import (
    _count_enriched_collapsed_helper,
    _filter_scores,
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
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertIn("sp001_C1_W1", result.index)
        self.assertIn("sp002_C2_W2", result.index)

    def test_non_collapsed_category_keeps_original_index(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertIn("pep_03", result.index)

    def test_two_peptides_share_same_epitope(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        codenames = result.loc["sp001_C1_W1", "CodeName"]
        self.assertEqual(set(codenames), {"pep_00", "pep_01"})

    def test_both_collapse_includes_bacterial(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Both")
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
        result = create_epitope_map(multi, collapse="Viral")
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
        self.epitope_map_viral = create_epitope_map(
            raw.copy(), collapse="Viral"
        )
        self.epitope_map_both = create_epitope_map(raw.copy(), collapse="Both")

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

    def test_initializes_epitope_key(self):
        counts = self._empty()
        _count_enriched_collapsed_helper(counts, "sp001", "InfluenzaA", "ep1", "H1N1")
        self.assertEqual(counts["epitope"]["sp001"]["InfluenzaA"]["ep1"], 1)

    def test_second_call_increments_count(self):
        counts = self._empty()
        _count_enriched_collapsed_helper(counts, "sp001", "InfluenzaA", "ep1", "H1N1")
        _count_enriched_collapsed_helper(counts, "sp001", "InfluenzaA", "ep1", "H1N1")
        self.assertEqual(counts["epitope"]["sp001"]["InfluenzaA"]["ep1"], 2)

    def test_tracks_subtype_separately(self):
        counts = self._empty()
        _count_enriched_collapsed_helper(counts, "sp001", "InfluenzaA", "ep1", "H1N1")
        self.assertEqual(counts["subtype"]["sp001"]["InfluenzaA"]["H1N1"], 1)


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

    def test_returns_dict_with_two_default_keys(self):
        scores = self._scores([{
            "p.adjust": 0.01,
            "enrichmentScore": 2.0,
            "core_enrichment": "sp001_C1_W1",
            "species_name": "InfluenzaA",
        }])
        result = count_enriched(scores, self._subtypes())
        self.assertEqual(set(result.keys()), {"epitope", "subtype"})


if __name__ == "__main__":
    unittest.main()
