import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.utils import remove_peptides, remove_peptides_in_df_format


class TestRemovePeptidesInDfFormat(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.scores = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        self.gmt = pd.read_csv(
            self.get_data_path("peptide-sets.tsv"), sep="\t"
        )

    def test_removes_peptides_absent_from_gmt(self):
        # scores has pep_00..pep_14; gmt covers pep_00..pep_08 via sp1..sp3
        filtered, _ = remove_peptides_in_df_format(self.scores, self.gmt)
        expected_peps = set(self.gmt["gene"].unique())
        self.assertEqual(set(filtered.index), expected_peps)

    def test_peptides_in_gmt_are_retained(self):
        filtered, _ = remove_peptides_in_df_format(self.scores, self.gmt)
        for pep in self.gmt["gene"].unique():
            self.assertIn(pep, filtered.index)

    def test_peptides_absent_from_gmt_are_dropped(self):
        filtered, _ = remove_peptides_in_df_format(self.scores, self.gmt)
        gmt_peps = set(self.gmt["gene"])
        for pep in self.scores.index:
            if pep not in gmt_peps:
                self.assertNotIn(pep, filtered.index)

    def test_returns_original_gmt_unchanged(self):
        _, returned_gmt = remove_peptides_in_df_format(self.scores, self.gmt)
        assert_frame_equal(returned_gmt, self.gmt)

    def test_no_overlap_returns_empty_filtered(self):
        gmt_no_overlap = pd.DataFrame(
            {"term": ["sp_x"], "gene": ["pep_nonexistent"]}
        )
        filtered, _ = remove_peptides_in_df_format(self.scores, gmt_no_overlap)
        self.assertEqual(len(filtered), 0)

    def test_values_preserved_for_retained_peptides(self):
        filtered, _ = remove_peptides_in_df_format(self.scores, self.gmt)
        self.assertAlmostEqual(filtered.loc["pep_00", "sA"], 0.5)
        self.assertAlmostEqual(filtered.loc["pep_00", "sB"], 0.8)


class TestRemovePeptidesDispatch(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.scores = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        self.gmt = pd.read_csv(
            self.get_data_path("peptide-sets.tsv"), sep="\t"
        )

    def test_dispatches_df_format_when_given_dataframe(self):
        filtered, _ = remove_peptides(self.scores, self.gmt)
        self.assertIsInstance(filtered, pd.DataFrame)
        self.assertEqual(set(filtered.index), set(self.gmt["gene"].unique()))

    def test_unsupported_string_format_raises(self):
        with self.assertRaises(AssertionError):
            remove_peptides(self.scores, "file.unknown")


if __name__ == "__main__":
    unittest.main()
