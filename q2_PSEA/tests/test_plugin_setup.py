import unittest

from qiime2.plugin.testing import TestPluginBase


class TestMethodsRegistered(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_process_scores_registered(self):
        self.assertIn("_process_scores", self.plugin.methods)

    def test_compute_pair_fit_registered(self):
        self.assertIn("_compute_pair_fit_and_residuals", self.plugin.methods)

    def test_count_antibody_events_registered(self):
        self.assertIn("count_antibody_events", self.plugin.methods)

    def test_create_fgsea_table_registered(self):
        self.assertIn("_create_fgsea_table_for_pair", self.plugin.methods)

    def test_create_epitope_map_registered(self):
        self.assertIn("create_epitope_map", self.plugin.methods)

    def test_map_residuals_and_zscores_registered(self):
        self.assertIn("_map_residuals_and_zscores", self.plugin.methods)


class TestPipelinesRegistered(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_run_iterative_process_registered(self):
        self.assertIn(
            "_run_iterative_process_single_pair", self.plugin.methods
        )

    def test_make_psea_table_registered(self):
        self.assertIn("make_psea_table", self.plugin.pipelines)


class TestCreateFgseaSignature(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.sig = \
            self.plugin.methods["_create_fgsea_table_for_pair"].signature

    def test_processed_scores_in_inputs(self):
        self.assertIn("processed_zscores", self.sig.inputs)

    def test_peptide_sets_in_inputs(self):
        self.assertIn("peptide_sets", self.sig.inputs)

    def test_species_taxa_is_parameter_not_input(self):
        self.assertIn("species_taxa", self.sig.parameters)
        self.assertNotIn("species_taxa", self.sig.inputs)

    def test_required_parameters_present(self):
        for name in ("threshold", "permutation_num", "min_size", "max_size",
                     "seed"):
            self.assertIn(name, self.sig.parameters)

    def test_output_is_psea_scores(self):
        self.assertIn("psea_table", self.sig.outputs)


if __name__ == "__main__":
    unittest.main()
