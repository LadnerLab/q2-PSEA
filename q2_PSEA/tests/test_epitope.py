import unittest

import numpy as np
import pandas as pd
from biom import load_table
from q2_types.feature_table import BIOMV210Format
from qiime2.plugin.testing import TestPluginBase

from qiime2 import Artifact
from q2_PSEA.actions.epitope import (
    _create_EpitopeID_row,
    create_epitope_map,
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


if __name__ == "__main__":
    unittest.main()
