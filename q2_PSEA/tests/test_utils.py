import unittest

import pandas as pd
from pandas.testing import assert_frame_equal
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.utils import (
    collapse_residuals_to_epitope,
    remove_peptides,
    remove_peptides_in_df_format,
)


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
        # scores has pep_00..pep_14; gmt covers pep_00..pep_08
        filtered, _ = remove_peptides_in_df_format(self.scores, self.gmt)
        self.assertEqual(set(filtered.index), set(self.gmt["gene"].unique()))

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
        _, returned = remove_peptides_in_df_format(self.scores, self.gmt)
        assert_frame_equal(returned, self.gmt)

    def test_no_overlap_returns_empty_filtered(self):
        gmt_none = pd.DataFrame({"term": ["sp_x"], "gene": ["pep_nonexistent"]})
        filtered, _ = remove_peptides_in_df_format(self.scores, gmt_none)
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

    def test_dispatches_df_format_for_dataframe_input(self):
        filtered, _ = remove_peptides(self.scores, self.gmt)
        self.assertIsInstance(filtered, pd.DataFrame)
        self.assertEqual(set(filtered.index), set(self.gmt["gene"].unique()))

    def test_unsupported_string_extension_raises(self):
        with self.assertRaises(AssertionError):
            remove_peptides(self.scores, "file.xyz")


class TestCollapseResidualsToEpitope(TestPluginBase):
    package = "q2_PSEA.tests"

    def _emap(self, mapping):
        return pd.DataFrame({"CodeName": mapping})

    def test_single_peptide_per_epitope(self):
        emap = self._emap({"ep1": ["pep1"], "ep2": ["pep2"]})
        res = pd.Series({"pep1": 0.5, "pep2": -0.3})
        result = collapse_residuals_to_epitope(res, emap)
        self.assertAlmostEqual(result["ep1"], 0.5)
        self.assertAlmostEqual(result["ep2"], -0.3)

    def test_multiple_peptides_keeps_max_abs_residual(self):
        emap = self._emap({"ep1": ["pep1", "pep2"]})
        res = pd.Series({"pep1": 0.5, "pep2": -0.8})
        result = collapse_residuals_to_epitope(res, emap)
        self.assertAlmostEqual(result["ep1"], -0.8)

    def test_unmapped_peptide_maps_to_itself(self):
        emap = self._emap({"ep1": ["pep1"]})
        res = pd.Series({"pep1": 0.5, "pep_orphan": 0.9})
        result = collapse_residuals_to_epitope(res, emap)
        self.assertIn("pep_orphan", result.index)
        self.assertAlmostEqual(result["pep_orphan"], 0.9)

    def test_peptide_in_multiple_epitopes_carries_max_abs(self):
        emap = self._emap({"ep1": ["pep1", "pep2"], "ep2": ["pep1", "pep3"]})
        res = pd.Series({"pep1": 1.0, "pep2": 0.2, "pep3": 0.5})
        result = collapse_residuals_to_epitope(res, emap)
        self.assertAlmostEqual(result["ep1"], 1.0)
        self.assertAlmostEqual(result["ep2"], 1.0)

    def test_returns_series(self):
        emap = self._emap({"ep1": ["pep1"]})
        result = collapse_residuals_to_epitope(pd.Series({"pep1": 0.7}), emap)
        self.assertIsInstance(result, pd.Series)


if __name__ == "__main__":
    unittest.main()
