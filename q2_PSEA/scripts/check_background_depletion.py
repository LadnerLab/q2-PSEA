"""Run with Python containing pandas/numpy; no QIIME installation required."""
import ast
import importlib.util
from pathlib import Path
import random
import tempfile
import os
import pandas as pd

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('psea_utils', root / 'utils.py')
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
tree = ast.parse((root / 'actions/psea.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name == '_run_depleting_process_single_pair')
node.returns = None
for arg in node.args.args:
    arg.annotation = None
class CaptureHolder:
    @staticmethod
    def get_or_set(seed, fallback):
        return fallback() if seed is None else seed

scope = dict(pd=pd, utils=utils, CaptureHolder=CaptureHolder, random=random,
             MIN_32_BIT_INT=-2**31, MAX_32_BIT_INT=2**31-1, os=os)
exec(compile(ast.Module(body=[node], type_ignores=[]), '<depletion>', 'exec'), scope)
rows = [dict(ID='A', NES=2., **{'p.adjust': .01}, all_tested_peptides='a/shared'),
        dict(ID='B', NES=2., **{'p.adjust': .02}, all_tested_peptides='b'),
        dict(ID='C', NES=.5, **{'p.adjust': .8}, all_tested_peptides='c')]
gmt = pd.DataFrame([('A','a'), ('A','shared'), ('B','shared'), ('B','b'), ('C','c')],
                   columns=['term','gene'])
scores = pd.DataFrame([[1,2,3,4],[2,3,4,5]], columns=['a','shared','b','c'])
seen = []
def fake_gsea(**kwargs):
    genes = set(kwargs['peptide_sets'].gene)
    seen.append((kwargs['debug_iteration'], genes))
    if 'a' in genes:
        return pd.DataFrame(rows)
    if 'b' in genes:
        return pd.DataFrame(rows[1:])
    return pd.DataFrame(rows[2:])
scope['_create_fgsea_table_for_pair'] = fake_gsea
with tempfile.TemporaryDirectory() as temp:
    result = scope[node.name](scores, gmt, threshold=1, permutation_num=10,
                             min_size=1, max_size=100, p_value=.05,
                             enrichment_score=1, seed=1,
                             debug_gsea_input_table_path=temp, pair='x~y')
assert seen == [('001', {'a','shared','b','c'}), ('002', {'b','c'}),
                ('003', {'c'}), ('final', {'c'})], seen
assert list(result.ID) == ['A','B','C']
assert list(result.all_tested_peptides) == ['a/shared','b','c']
assert set(gmt.gene) == {'a','shared','b','c'}  # caller input unchanged
# Default removal continues to preserve the called species' memberships.
updated, _, _ = utils.filter_peptide_sets(pd.DataFrame(rows), gmt, set(), .05, 1, True)
assert set(updated.gene) == set(gmt.gene)
# Depletion can exhaust every membership; earlier calls remain in the result.
seen.clear()
def exhausting(**kwargs):
    genes = set(kwargs['peptide_sets'].gene)
    seen.append(genes)
    return pd.DataFrame([rows[0]] if genes else [], columns=pd.DataFrame(rows).columns)
scope['_create_fgsea_table_for_pair'] = exhausting
result = scope[node.name](scores, gmt[gmt.term == 'A'], threshold=1,
                         permutation_num=10, min_size=1, max_size=100,
                         p_value=.05, enrichment_score=1, seed=1)
assert seen == [{'a','shared'}, set()]
assert list(result.ID) == ['A']
print('Depletion, cumulative exclusion, retained calls, exhaustion and default behavior passed.')
