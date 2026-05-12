import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.utils import (
    filter_peptide_sets,
    _get_mapped_features,
    collapse_residuals_to_epitope,
    remove_peptides,
)

# ---------------------------------------------------------------------------
# filter_peptide_sets — unit tests
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
        updated, tested, sig_found = filter_peptide_sets(
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
        updated, _, _ = filter_peptide_sets(
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
        _, tested, sig_found = filter_peptide_sets(
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
        _, _, sig_found = filter_peptide_sets(
            psea, gmt, set(), 0.05, 1.0, True
        )
        self.assertFalse(sig_found)

    def test_include_negative_enrichment_false_ignores_negative_nes(self):
        psea = self._psea([
            {"ID": "sp1", "p.adjust": 0.01, "NES": -2.0,
             "all_tested_peptides": "pep1"},
        ])
        gmt = self._gmt([("sp1", "pep1")])
        _, _, sig_found = filter_peptide_sets(
            psea, gmt, set(), 0.05, 1.0, False
        )
        self.assertFalse(sig_found)

    def test_include_negative_enrichment_false_accepts_positive_nes(self):
        psea = self._psea([
            {"ID": "sp1", "p.adjust": 0.01, "NES": 2.0,
             "all_tested_peptides": "pep1"},
        ])
        gmt = self._gmt([("sp1", "pep1"), ("sp2", "pep2")])
        _, tested, sig_found = filter_peptide_sets(
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
        _, tested, _ = filter_peptide_sets(
            psea, gmt, set(), 0.05, 1.0, True
        )
        self.assertIn("sp2", tested)
        self.assertNotIn("sp1", tested)


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


if __name__ == "__main__":
    unittest.main()
