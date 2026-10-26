import unittest

import pandas as pd
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.utils import (
    filter_peptide_sets,
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

    def test_with_epitope_gene_ids_removes_direct_match(self):
        # gene values may be epitope IDs (e.g. "ep1") rather than peptide
        # CodeNames when the GMT has already been collapsed; filtering works
        # the same way regardless.
        psea = self._psea([
            {"ID": "sp1", "p.adjust": 0.01, "NES": 2.0,
             "all_tested_peptides": "ep1"},
        ])
        gmt = self._gmt([
            ("sp1", "ep1"), ("sp2", "ep1"), ("sp2", "ep2"),
        ])
        updated, _, _ = filter_peptide_sets(
            psea, gmt, set(), 0.05, 1.0, True,
        )
        sp2_genes = updated[updated["term"] == "sp2"]["gene"].tolist()
        self.assertNotIn("ep1", sp2_genes)

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


if __name__ == "__main__":
    unittest.main()
