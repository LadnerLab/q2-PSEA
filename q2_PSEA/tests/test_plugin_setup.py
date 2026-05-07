import unittest

from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.types import PSEAPairs, PSEAAECounts


class TestSemanticTypes(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_psea_pairs_name(self):
        self.assertEqual(str(PSEAPairs), "PSEAPairs")

    def test_psea_ae_counts_name(self):
        self.assertEqual(str(PSEAAECounts), "PSEAAECounts")

    def test_psea_pairs_registered(self):
        registered = {str(t) for t in self.plugin.type_fragments}
        self.assertIn("PSEAPairs", registered)

    def test_psea_ae_counts_registered(self):
        registered = {str(t) for t in self.plugin.type_fragments}
        self.assertIn("PSEAAECounts", registered)


class TestFormatsRegistered(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_pairs_tsv_format_registered(self):
        self.assertIn("PSEAPairsTSVFormat", self.plugin.formats)

    def test_pairs_dir_format_registered(self):
        self.assertIn("PSEAPairsDirFmt", self.plugin.formats)

    def test_ae_counts_tsv_format_registered(self):
        self.assertIn("PSEAAECountsTSVFormat", self.plugin.formats)

    def test_ae_counts_dir_format_registered(self):
        self.assertIn("PSEAAECountsDirFmt", self.plugin.formats)

    def test_spline_tsv_format_registered(self):
        self.assertIn("SplineTSVFormat", self.plugin.formats)

    def test_spline_dir_format_registered(self):
        self.assertIn("SplineDirFmt", self.plugin.formats)


class TestMethodsRegistered(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_process_scores_registered(self):
        self.assertIn("_process_scores", self.plugin.methods)

    def test_compute_pair_fit_registered(self):
        self.assertIn("_compute_pair_fit_and_residuals", self.plugin.methods)

    def test_count_antibody_events_registered(self):
        self.assertIn("count_antibody_events", self.plugin.methods)

    def test_create_fgsea_table_registered(self):
        self.assertIn("create_fgsea_table_for_pair", self.plugin.methods)

    def test_create_epitope_map_registered(self):
        self.assertIn("create_epitope_map", self.plugin.methods)

    def test_epitope_zscore_registered(self):
        self.assertIn("epitope_zscore", self.plugin.methods)


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
        self.sig = self.plugin.methods["create_fgsea_table_for_pair"].signature

    def test_processed_scores_in_inputs(self):
        self.assertIn("processed_scores", self.sig.inputs)

    def test_peptide_sets_in_inputs(self):
        self.assertIn("peptide_sets", self.sig.inputs)

    def test_species_taxa_is_parameter_not_input(self):
        self.assertIn("species_taxa", self.sig.parameters)
        self.assertNotIn("species_taxa", self.sig.inputs)

    def test_required_parameters_present(self):
        for name in ("sample_a", "sample_b", "threshold",
                     "permutation_num", "min_size", "max_size", "seed"):
            self.assertIn(name, self.sig.parameters)

    def test_output_is_psea_scores(self):
        self.assertIn("psea_table", self.sig.outputs)


if __name__ == "__main__":
    unittest.main()
