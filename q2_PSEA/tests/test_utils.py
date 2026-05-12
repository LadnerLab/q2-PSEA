import unittest

import pandas as pd
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


# ---------------------------------------------------------------------------
# remove_peptides — unit tests
# ---------------------------------------------------------------------------

class TestRemovePeptides(TestPluginBase):
    package = "q2_PSEA.tests"

    def _scores(self, peptides):
        return pd.DataFrame(
            {p: [1.0] for p in peptides}
        ).T  # peptides as index (rows), one sample column

    def _gmt(self, genes):
        return pd.DataFrame({"term": ["sp1"] * len(genes), "gene": genes})

    def test_peptides_not_in_gmt_are_removed(self):
        scores = self._scores(["pep1", "pep2", "pep3"])
        gmt = self._gmt(["pep1", "pep2"])
        out, _ = remove_peptides(scores, gmt)
        self.assertIn("pep1", out.index)
        self.assertIn("pep2", out.index)
        self.assertNotIn("pep3", out.index)

    def test_all_peptides_retained_when_all_in_gmt(self):
        scores = self._scores(["pep1", "pep2"])
        gmt = self._gmt(["pep1", "pep2"])
        out, _ = remove_peptides(scores, gmt)
        self.assertEqual(set(out.index), {"pep1", "pep2"})

    def test_peptide_sets_returned_unchanged(self):
        scores = self._scores(["pep1", "pep2", "pep3"])
        gmt = self._gmt(["pep1", "pep2"])
        _, out_gmt = remove_peptides(scores, gmt)
        pd.testing.assert_frame_equal(out_gmt, gmt)

    def test_extra_genes_in_gmt_not_present_in_scores_are_ignored(self):
        # pep3_extra is in the GMT but not in scores — nothing to drop from
        # scores
        scores = self._scores(["pep1", "pep2"])
        gmt = self._gmt(["pep1", "pep2", "pep3_extra"])
        out, _ = remove_peptides(scores, gmt)
        self.assertEqual(set(out.index), {"pep1", "pep2"})


# ---------------------------------------------------------------------------
# collapse_residuals_to_epitope — unit tests
# ---------------------------------------------------------------------------

class TestCollapseResidualsToEpitope(TestPluginBase):
    package = "q2_PSEA.tests"

    def _emap(self, mapping):
        return pd.DataFrame(
            {"CodeName": list(mapping.values())},
            index=pd.Index(list(mapping.keys()), name="EpitopeID"),
        )

    def test_single_peptide_assigned_to_its_epitope(self):
        emap = self._emap({"ep1": ["pep1"]})
        result = collapse_residuals_to_epitope({"pep1": 0.5}, emap)
        self.assertAlmostEqual(result["ep1"], 0.5)

    def test_largest_absolute_residual_wins_positive(self):
        emap = self._emap({"ep1": ["pep1", "pep2"]})
        result = collapse_residuals_to_epitope(
            {"pep1": 0.5, "pep2": 0.8}, emap
        )
        self.assertAlmostEqual(result["ep1"], 0.8)

    def test_negative_residual_wins_when_larger_absolute_value(self):
        # abs(-0.9) > abs(0.5), so -0.9 should be kept
        emap = self._emap({"ep1": ["pep1", "pep2"]})
        result = collapse_residuals_to_epitope(
            {"pep1": -0.9, "pep2": 0.5}, emap
        )
        self.assertAlmostEqual(result["ep1"], -0.9)

    def test_unmapped_peptide_uses_itself_as_key(self):
        emap = self._emap({"ep1": ["pep1"]})
        result = collapse_residuals_to_epitope(
            {"pep1": 0.3, "unmapped_pep": 0.7}, emap
        )
        self.assertAlmostEqual(result["unmapped_pep"], 0.7)

    def test_mapped_and_unmapped_both_present_in_result(self):
        emap = self._emap({"ep1": ["pep1"]})
        result = collapse_residuals_to_epitope(
            {"pep1": 0.3, "unmapped_pep": 0.7}, emap
        )
        self.assertAlmostEqual(result["ep1"], 0.3)
        self.assertAlmostEqual(result["unmapped_pep"], 0.7)

    def test_multiple_distinct_epitopes_each_get_their_residual(self):
        emap = self._emap({"ep1": ["pep1"], "ep2": ["pep2"]})
        result = collapse_residuals_to_epitope(
            {"pep1": 0.5, "pep2": 0.3}, emap
        )
        self.assertAlmostEqual(result["ep1"], 0.5)
        self.assertAlmostEqual(result["ep2"], 0.3)

    def test_returns_pandas_series(self):
        emap = self._emap({"ep1": ["pep1"]})
        result = collapse_residuals_to_epitope({"pep1": 0.5}, emap)
        self.assertIsInstance(result, pd.Series)


if __name__ == "__main__":
    unittest.main()
