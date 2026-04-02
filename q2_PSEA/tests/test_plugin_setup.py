import unittest

from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.actions.splines import SPLINE_TYPES
from q2_PSEA.types import PSEAPairs, PSEASpeciesColors, PSEASpeciesTaxa


class TestSemanticTypes(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_psea_pairs_name(self):
        self.assertEqual(str(PSEAPairs), "PSEAPairs")

    def test_psea_species_taxa_name(self):
        self.assertEqual(str(PSEASpeciesTaxa), "PSEASpeciesTaxa")

    def test_psea_species_colors_name(self):
        self.assertEqual(str(PSEASpeciesColors), "PSEASpeciesColors")

    def test_all_three_types_registered(self):
        registered = {str(t) for t in self.plugin.type_fragments}
        self.assertIn("PSEAPairs", registered)
        self.assertIn("PSEASpeciesTaxa", registered)
        self.assertIn("PSEASpeciesColors", registered)


class TestFormats(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_pairs_tsv_format_registered(self):
        self.assertIn("PSEAPairsTSVFormat", self.plugin.formats)

    def test_species_taxa_tsv_format_registered(self):
        self.assertIn("PSEASpeciesTaxaTSVFormat", self.plugin.formats)

    def test_species_colors_tsv_format_registered(self):
        self.assertIn("PSEASpeciesColorsTSVFormat", self.plugin.formats)

    def test_pairs_dir_format_registered(self):
        self.assertIn("PSEAPairsDirFmt", self.plugin.formats)

    def test_species_taxa_dir_format_registered(self):
        self.assertIn("PSEASpeciesTaxaDirFmt", self.plugin.formats)

    def test_species_colors_dir_format_registered(self):
        self.assertIn("PSEASpeciesColorsDirFmt", self.plugin.formats)


class TestMethodRegistration(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.action = self.plugin.methods["create_fgsea_table_for_pair"]

    def test_create_fgsea_table_is_registered(self):
        self.assertIn("create_fgsea_table_for_pair", self.plugin.methods)

    def test_inputs_include_processed_scores(self):
        self.assertIn("processed_scores", self.action.signature.inputs)

    def test_inputs_include_peptide_sets(self):
        self.assertIn("peptide_sets", self.action.signature.inputs)

    def test_required_parameters_present(self):
        params = self.action.signature.parameters
        for name in ("sample_a", "sample_b", "threshold", "permutation_num",
                     "spline_type", "degree", "seed"):
            self.assertIn(name, params)

    def test_output_is_psea_table(self):
        self.assertIn("psea_table", self.action.signature.outputs)

    def test_spline_type_choices_include_all_types(self):
        param_str = str(self.action.signature.parameters["spline_type"])
        for spline in SPLINE_TYPES:
            self.assertIn(spline, param_str)


class TestPipelineRegistration(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_run_single_pair_registered(self):
        self.assertIn("run_iterative_process_single_pair", self.plugin.pipelines)

    def test_run_iterative_analysis_registered(self):
        self.assertIn("run_iterative_peptide_analysis", self.plugin.pipelines)

    def test_make_psea_table_registered(self):
        self.assertIn("make_psea_table", self.plugin.pipelines)

    def test_make_psea_table_inputs(self):
        sig = self.plugin.pipelines["make_psea_table"].signature
        for name in ("scores", "pairs", "peptide_sets"):
            self.assertIn(name, sig.inputs)

    def test_make_psea_table_outputs(self):
        outputs = self.plugin.pipelines["make_psea_table"].signature.outputs
        for name in ("scatter_plot", "volcano_plot", "ae_plots", "psea_tables"):
            self.assertIn(name, outputs)

    def test_run_iterative_analysis_output(self):
        outputs = (
            self.plugin.pipelines["run_iterative_peptide_analysis"]
            .signature.outputs
        )
        self.assertIn("filtered_peptide_sets", outputs)

    def test_run_single_pair_outputs(self):
        outputs = (
            self.plugin.pipelines["run_iterative_process_single_pair"]
            .signature.outputs
        )
        self.assertIn("psea_table", outputs)
        self.assertIn("updated_peptide_sets", outputs)


if __name__ == "__main__":
    unittest.main()
