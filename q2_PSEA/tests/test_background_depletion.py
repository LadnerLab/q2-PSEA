import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from q2_PSEA.actions import psea


class TestBackgroundDepletion(unittest.TestCase):
    def setUp(self):
        self.gmt = pd.DataFrame(
            [('A', 'a'), ('A', 'shared'), ('B', 'shared'), ('B', 'b'), ('C', 'c')],
            columns=['term', 'gene'],
        )
        self.rows = pd.DataFrame([
            {'ID': 'A', 'NES': 2., 'p.adjust': .01, 'all_tested_peptides': 'a/shared'},
            {'ID': 'B', 'NES': 2., 'p.adjust': .02, 'all_tested_peptides': 'b'},
            {'ID': 'C', 'NES': .5, 'p.adjust': .8, 'all_tested_peptides': 'c'},
        ])
        self.kwargs = dict(
            processed_zscores=pd.DataFrame([[1, 2, 3, 4]],
                                          columns=['a', 'shared', 'b', 'c']),
            peptide_sets=self.gmt, threshold=1., permutation_num=10,
            min_size=1, max_size=100, p_value=.05, enrichment_score=1., seed=1,
        )

    def test_cumulative_removal_and_retained_calls(self):
        observed = []
        def run(**kwargs):
            genes = set(kwargs['peptide_sets'].gene)
            observed.append(genes)
            if 'a' in genes:
                return self.rows.copy()
            if 'b' in genes:
                return self.rows.iloc[1:].copy()
            return self.rows.iloc[2:].copy()
        with patch.object(psea, '_create_fgsea_table_for_pair', side_effect=run):
            result = psea._run_depleting_process_single_pair(**self.kwargs)
        self.assertEqual(observed, [{'a', 'shared', 'b', 'c'}, {'b', 'c'}, {'c'}])
        assert_frame_equal(result, self.rows)
        self.assertEqual(set(self.gmt.gene), {'a', 'shared', 'b', 'c'})

    def test_first_iteration_matches_original_and_mapped_input(self):
        mapped = self.gmt.assign(gene=lambda df: 'ep_' + df.gene)
        first_calls = []
        def run(**kwargs):
            first_calls.append(kwargs)
            return self.rows.iloc[0:0].copy()
        with patch.object(psea, '_create_fgsea_table_for_pair', side_effect=run):
            psea._run_iterative_process_single_pair(**self.kwargs, mapped_peptide_sets=mapped)
            result = psea._run_depleting_process_single_pair(**self.kwargs, mapped_peptide_sets=mapped)
        self.assertTrue(result.empty)
        self.assertEqual(first_calls[0].keys(), first_calls[1].keys())
        for key in first_calls[0]:
            a, b = first_calls[0][key], first_calls[1][key]
            if isinstance(a, pd.DataFrame):
                assert_frame_equal(a, b)
            else:
                self.assertEqual(a, b)

    def test_exhausted_background_retains_last_call(self):
        observed = []
        def run(**kwargs):
            genes = set(kwargs['peptide_sets'].gene)
            observed.append(genes)
            return self.rows.iloc[:1 if genes else 0].copy()
        kwargs = dict(self.kwargs, peptide_sets=self.gmt[self.gmt.term == 'A'])
        with patch.object(psea, '_create_fgsea_table_for_pair', side_effect=run):
            result = psea._run_depleting_process_single_pair(**kwargs)
        self.assertEqual(observed, [{'a', 'shared'}, set()])
        self.assertEqual(list(result.ID), ['A'])

    def test_flag_requires_iteration_for_both_pipelines(self):
        for pipeline in (psea.make_psea, psea.make_psea_tables):
            with self.assertRaisesRegex(ValueError, 'requires iterative_analysis'):
                pipeline(None, None, None, None, 1., iterative_analysis=False,
                         remove_claimed_peptides_from_background=True, seed=1)

    def test_debug_files_reuse_terminal_run_and_preserve_combined_calls(self):
        from pathlib import Path
        import tempfile
        seen = []
        def run(**kwargs):
            iteration = kwargs['debug_iteration']
            seen.append(iteration)
            folder = Path(kwargs['debug_gsea_input_table_path']) / 'x~y'
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f'{iteration}_ranked_gene_list.tsv').write_text(iteration)
            genes = set(kwargs['peptide_sets'].gene)
            if 'a' in genes:
                return self.rows.copy()
            if 'b' in genes:
                return self.rows.iloc[1:].copy()
            return self.rows.iloc[2:].copy()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(psea, '_create_fgsea_table_for_pair', side_effect=run):
                result = psea._run_depleting_process_single_pair(
                    **self.kwargs, debug_gsea_input_table_path=directory, pair='x~y')
            folder = Path(directory) / 'x~y'
            self.assertEqual(seen, ['001', '002', '003'])
            self.assertEqual((folder / 'final_ranked_gene_list.tsv').read_text(), '003')
            combined = pd.read_csv(folder / 'combined_psea_output.tsv', sep='\t')
            assert_frame_equal(combined, result)
            calls = pd.read_csv(folder / 'called_species_iterations.tsv', sep='\t')
            self.assertEqual(list(calls.ID), ['A', 'B'])
            self.assertEqual(list(calls.iteration), [1, 2])
