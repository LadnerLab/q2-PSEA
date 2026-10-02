"""Dependency-light checks: python3 scripts/check_gsea_diagnostics.py."""
import ast
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
for path in root.rglob('*.py'):
    compile(path.read_text(), str(path), 'exec')
pipeline = ast.parse((root / 'actions/psea.py').read_text())
functions = {n.name: n for n in pipeline.body if isinstance(n, ast.FunctionDef)}
for name in ('make_psea_table', '_run_iterative_process_single_pair',
             '_create_fgsea_table_for_pair'):
    assert 'debug_gsea_input_table_path' in [a.arg for a in functions[name].args.args]
for name in ('make_psea_table', '_run_iterative_process_single_pair'):
    calls = [n for n in ast.walk(functions[name]) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == '_create_fgsea_table_for_pair']
    assert len(calls) == 1
    assert {'debug_gsea_input_table_path', 'debug_pair', 'debug_iteration'} <= {
        k.arg for k in calls[0].keywords}
tree = ast.parse((root / 'actions/r_functions.py').read_text())
r = next(ast.literal_eval(n.value) for n in tree.body
         if isinstance(n, ast.Assign)
         and any(isinstance(t, ast.Name) and t.id == 'r_functions' for t in n.targets))
checks = r'''
results <- data.frame(ID=paste0("s", 20:1), p.adjust=(20:1)/100,
                      pvalue=(20:1)/200)
stopifnot(identical(select_top_diagnostic_terms(results), paste0("s", 1:15)))
results$p.adjust[results$ID == "s1"] <- NA_real_
stopifnot(identical(select_top_diagnostic_terms(results), paste0("s", 2:16)),
          length(select_top_diagnostic_terms(results[FALSE, ])) == 0,
          length(select_top_diagnostic_terms(results[1:3, ])) == 3)
ties <- data.frame(ID=c("b", "a", "c"), p.adjust=c(.1,.1,.1),
                   pvalue=c(.02,.02,.01))
stopifnot(identical(select_top_diagnostic_terms(ties), c("c", "a", "b")))
# Verify that each selected species actually reaches the individual plot loop.
many <- data.frame(term=paste0("s", 1:20), gene=paste0("g", 1:20),
                   maxZ=2, deltaZ=20:1, in_gene_list=TRUE)
ranked <- setNames(many$deltaZ, many$gene)
results$p.adjust <- results$pvalue * 2
diag_many <- build_diagnostics(many, results, 1, 100, ranked)
selected <- select_top_diagnostic_terms(results)
plotted <- 0L
original_hist <- graphics::hist
hist_recorder <- function(x, ...) {
    plotted <<- plotted + 1L
    original_hist(x, ...)
}
# Capture panel titles using a local title-independent histogram wrapper.
plot_env <- new.env(parent=environment(plot_diagnostics))
plot_env$hist_recorder <- hist_recorder
plot_source <- paste(deparse(plot_diagnostics), collapse="\n")
plot_source <- sub("graphics::hist(x, ...)", "hist_recorder(x, ...)",
                   plot_source, fixed=TRUE)
plot_test <- eval(parse(text=plot_source), envir=plot_env)
plot_test(many, diag_many, ranked, "top15.pdf", selected)
stopifnot(plotted == 15L)

d <- data.frame(term=c("a","b"), gene=c("x","y"), maxZ=c(2,2),
                deltaZ=c(1,-1), in_gene_list=c(TRUE,TRUE))
o <- data.frame(ID="a", pvalue=0.1, p.adjust=0.2)
g <- c(x=1,y=-1,z=0.5)
diag <- build_diagnostics(d,o,1,10,g)
stopifnot(nrow(diag)==2, diag$genes_in_gene_list==1,
          diag$fraction_of_ranked_gene_list==1/3,
          diag$appears_in_gsea_output[diag$ID=="a"],
          !diag$appears_in_gsea_output[diag$ID=="b"])
plot_diagnostics(d,diag,g,"sparse.pdf",c("a"))
d$in_gene_list <- FALSE
diag <- build_diagnostics(d,o,1,10,numeric())
plot_diagnostics(d,diag,numeric(),"unranked.pdf",c("a"))
empty <- d[FALSE,]
plot_diagnostics(empty,build_diagnostics(empty,o,1,10,numeric()),
                 numeric(),"empty.pdf",character())
stopifnot(all(file.info(c("sparse.pdf","unranked.pdf","empty.pdf"))$size > 0))
'''
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / 'check.R'
    path.write_text(r + checks)
    subprocess.run(['Rscript', str(path)], cwd=tmp, check=True)
print('Python syntax, debug parameter wiring, and R diagnostic checks passed.')
