"""Verify debug export parity in the actual R implementation."""
import ast
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class TestGseaDebug(unittest.TestCase):
    @unittest.skipUnless(shutil.which('Rscript'), 'Rscript required')
    def test_exports_preserve_results_and_handle_empty_background(self):
        path = Path(__file__).resolve().parents[1] / 'actions/r_functions.py'
        tree = ast.parse(path.read_text())
        code = next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'r_functions'
                            for t in n.targets))
        checks = r'''
suppressPackageStartupMessages(library(clusterProfiler))
BiocParallel::register(BiocParallel::SerialParam())
genes <- paste0("p", 1:200)
x <- setNames(rep(3,200), genes)
y <- setNames(seq(2,-2,length.out=200), genes)
gmt <- data.frame(term=rep(c("A","B","C"),each=30),
                  gene=genes[c(1:30,86:115,171:200)])
plain <- psea(x,y,gmt,"",1,100,5,100,42)
rng_plain <- .Random.seed
debug <- psea(x,y,gmt,"",1,100,5,100,42,"final")
stopifnot(identical(plain,debug), identical(rng_plain,.Random.seed))
suffixes <- c("gsea_input.tsv","ranked_gene_list.tsv","term_to_gene.tsv",
              "parameters.tsv","gsea_output.tsv","missing.tsv",
              "diagnostics.tsv","diagnostics.pdf","raw_fgsea.tsv")
stopifnot(all(file.exists(paste0("final_",suffixes))))
ranked <- read.delim("final_ranked_gene_list.tsv")
stopifnot(nrow(ranked)==200, identical(ranked$rank,1:200))
input <- read.delim("final_gsea_input.tsv")
stopifnot(nrow(input)==90)
empty <- psea(setNames(numeric(),character()),setNames(numeric(),character()),
              gmt[FALSE,],"",1,100,5,100,42,"empty")
stopifnot(nrow(empty)==0, all(file.exists(paste0("empty_",suffixes))))
for (mode in c("std", "pos", "neg")) {
    directional <- psea(x,y,gmt,"",1,100,5,100,42,mode,mode)
    if (mode == "std") stopifnot(identical(directional, plain))
    raw <- read.delim(paste0(mode,"_raw_fgsea.tsv"))
    parameters <- read.delim(paste0(mode,"_parameters.tsv"))
    stopifnot(parameters$score_type == mode)
    stopifnot(identical(read.delim(paste0(mode,"_ranked_gene_list.tsv")), ranked))
    expected <- raw[match(directional$ID,raw$pathway),]
    stopifnot(isTRUE(all.equal(directional$NES, expected$NES, tolerance=1e-8)))
    if (mode == "pos") stopifnot(all(directional$enrichmentScore >= 0))
    if (mode == "neg") stopifnot(all(directional$enrichmentScore <= 0))
    if (mode != "std") stopifnot(identical(directional$core_enrichment,expected$leadingEdge))
}
results <- data.frame(ID=paste0("s",20:1), p.adjust=(20:1)/100,pvalue=(20:1)/200)
stopifnot(identical(select_top_diagnostic_terms(results),paste0("s",1:15)))
'''
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder) / 'check.R'
            script.write_text(code + checks)
            subprocess.run(['Rscript', str(script)], cwd=folder, check=True)
