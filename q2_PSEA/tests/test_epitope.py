import unittest

import numpy as np
import pandas as pd
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.actions.epitope import (
    _create_EpitopeID_row,
    count_enriched,
    create_epitope_map,
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


class TestTaxaToEpitope(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.peptide_metadata = pd.read_csv(
            self.get_data_path("epitope.tsv"), sep="\t", index_col=0
        )

    def _peptide_sets(self):
        # GMT-shaped: term/gene, gene values are peptide CodeNames present
        # in peptide_metadata's index (epitope.tsv covers pep_00..pep_03).
        return pd.DataFrame(
            {
                "term": ["sp001", "sp001", "sp002", "sp003"],
                "gene": ["pep_00", "pep_01", "pep_02", "pep_03"],
            }
        )

    def test_term_contains_species_ids(self):
        result = taxa_to_epitope(
            self.peptide_metadata, self._peptide_sets(), collapse="Viral"
        )
        self.assertIn("sp001", result["term"].values)

    def test_gene_contains_viral_epitope_ids(self):
        result = taxa_to_epitope(
            self.peptide_metadata, self._peptide_sets(), collapse="Viral"
        )
        self.assertIn("sp001_C1_W1", result["gene"].values)

    def test_bacterial_gene_is_original_index_when_collapse_viral(self):
        result = taxa_to_epitope(
            self.peptide_metadata, self._peptide_sets(), collapse="Viral"
        )
        self.assertIn("pep_03", result["gene"].values)

    def test_both_collapse_gives_bacterial_epitope_id(self):
        result = taxa_to_epitope(
            self.peptide_metadata, self._peptide_sets(), collapse="Both"
        )
        self.assertIn("sp003_C3_W3", result["gene"].values)


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

    def test_nan_cluster_id_filled_with_clusterNA(self):
        df = pd.DataFrame(
            [
                ["InfluenzaA", "H1N1", "sp001", np.nan, "W1", "Viral"],
                ["InfluenzaB", "H3N2", "sp002", "C2", "W2", "Viral"],
            ],
            columns=["Species", "Subtype", "SpeciesID", "ClusterID",
                     "EpitopeWindow", "Category"],
            index=pd.Index(["pep_00", "pep_01"], name="CodeName"),
        )
        result = _create_EpitopeID_row(df, "Viral")
        self.assertIn("clusterNA", result["ClusterID"].values)

    def test_nan_epitope_window_filled_with_peptide_na(self):
        df = pd.DataFrame(
            [
                ["InfluenzaA", "H1N1", "sp001", "C1", np.nan, "Viral"],
                ["InfluenzaB", "H3N2", "sp002", "C2", "W2", "Viral"],
            ],
            columns=["Species", "Subtype", "SpeciesID", "ClusterID",
                     "EpitopeWindow", "Category"],
            index=pd.Index(["pep_00", "pep_01"], name="CodeName"),
        )
        result = _create_EpitopeID_row(df, "Viral")
        self.assertIn("Peptide_NA", result["EpitopeWindow"].values)


class TestCountEnriched(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.residuals = pd.DataFrame(
            {"deltaZ": {"pep_00": 2.0, "pep_01": 1.0}}
        )
        self.peptide_metadata = pd.DataFrame(
            {
                "SpeciesID": {"pep_00": "sp001", "pep_01": "sp001"},
                "Species": {"pep_00": "InfluenzaA", "pep_01": "InfluenzaA"},
                "Subtype": {"pep_00": "H1N1", "pep_01": "H1N1"},
            }
        )
        self.epitope_map = pd.DataFrame(
            {"CodeName": [["pep_00", "pep_01"]]},
            index=pd.Index(["ep1"], name="EpitopeID"),
        )
        self.psea_table = pd.DataFrame(
            {
                "p.adjust": [0.01],
                "core_enrichment": ["ep1"],
                "NES": [2.0],
                "ID": ["sp001"],
            }
        )

    def test_returns_empty_dataframe_when_epitope_map_is_none(self):
        result = count_enriched(
            self.psea_table,
            self.residuals,
            peptide_metadata=self.peptide_metadata,
            epitope_map=None,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_returns_empty_dataframe_when_peptide_metadata_is_none(self):
        result = count_enriched(
            self.psea_table,
            self.residuals,
            peptide_metadata=None,
            epitope_map=self.epitope_map,
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)

    def test_filters_by_p_value(self):
        residuals = pd.DataFrame({"deltaZ": {"pep_00": 2.0, "pep_02": 1.5}})
        peptide_metadata = pd.DataFrame(
            {
                "SpeciesID": {"pep_00": "sp001", "pep_02": "sp002"},
                "Species": {"pep_00": "InfluenzaA", "pep_02": "EBV"},
                "Subtype": {"pep_00": "H1N1", "pep_02": "EBV1"},
            }
        )
        epitope_map = pd.DataFrame(
            {"CodeName": [["pep_00"], ["pep_02"]]},
            index=pd.Index(["ep1", "ep2"], name="EpitopeID"),
        )
        psea_table = pd.DataFrame(
            {"p.adjust": [0.01, 0.1], "core_enrichment": ["ep1", "ep2"]}
        )
        result = count_enriched(
            psea_table, residuals, peptide_metadata, epitope_map, p_value=0.05
        )
        self.assertIn(
            "sp001", result.index.get_level_values("Species ID Called")
        )
        self.assertNotIn(
            "sp002", result.index.get_level_values("Species ID Called")
        )

    def test_filters_peptides_by_residual_threshold(self):
        result = count_enriched(
            self.psea_table,
            self.residuals,
            self.peptide_metadata,
            self.epitope_map,
            residual_threshold=1.5,
        )
        self.assertEqual(result.iloc[0]["Peptide Counts"], 1)

    def test_normalizes_and_sums_residuals_within_epitope(self):
        result = count_enriched(
            self.psea_table,
            self.residuals,
            self.peptide_metadata,
            self.epitope_map,
            residual_threshold=1.0,
        )
        self.assertEqual(result.iloc[0]["Peptide Counts"], 1)
        self.assertAlmostEqual(
            result.iloc[0]["Relative Enrichment Score"], 1.5
        )

    def test_output_has_correct_multiindex_names(self):
        result = count_enriched(
            self.psea_table,
            self.residuals,
            self.peptide_metadata,
            self.epitope_map,
        )
        self.assertEqual(
            list(result.index.names),
            ["Species ID Called", "Species Called", "Subtypes"],
        )

    def test_nan_subtype_becomes_subtypena(self):
        peptide_metadata = pd.DataFrame(
            {
                "SpeciesID": {"pep_00": "sp001", "pep_01": "sp001"},
                "Species": {"pep_00": "InfluenzaA", "pep_01": "InfluenzaA"},
                "Subtype": {"pep_00": np.nan, "pep_01": "H1N1"},
            }
        )
        epitope_map = pd.DataFrame(
            {"CodeName": [["pep_00"]]},
            index=pd.Index(["ep1"], name="EpitopeID"),
        )
        result = count_enriched(
            self.psea_table,
            self.residuals,
            peptide_metadata,
            epitope_map,
        )
        self.assertIn("subtypeNA", result.index.get_level_values("Subtypes"))


if __name__ == "__main__":
    unittest.main()
