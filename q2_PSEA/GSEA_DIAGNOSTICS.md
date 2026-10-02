# Automatic GSEA diagnostics

Add this option to your existing `qiime psea make-psea-table` command:

```sh
--p-debug-gsea-input-table-path /absolute/path/to/new-gsea-debug-directory
```

The output directory must not already exist. Use a shared, writable absolute
path for execution on multiple HPC nodes. Diagnostics are optional; once this
option is provided, all tables and R plots are generated automatically inside
the existing R call. No separate Rscript command is required.

Each pair gets its own directory, with `001`, `002`, etc. prefixes for iterative
analyses and a `final` prefix for the final analysis (including non-iterative
runs). Pair labels are URL-escaped for filenames.

Each prefix produces:

- `_gsea_input.tsv`: term/gene membership, maxZ, deltaZ, inclusion flag,
  actual R rank and score, matching the debug version's table columns.
- `_ranked_gene_list.tsv`: the complete ranked vector actually passed to GSEA.
  This may include genes outside the remaining sets after residual filtering.
- `_term_to_gene.tsv`: the exact TERM2GENE table.
- `_parameters.tsv`: threshold, permutation count, size limits and seed.
- `_gsea_output.tsv`: the actual pipeline R result, including tested peptides.
- `_missing.tsv`: input terms absent from the GSEA result.
- `_diagnostics.tsv`: effective set sizes, sign balance, score statistics,
  size-limit checks, output presence and likely exclusion reasons.
- `_diagnostics.pdf`: ranked-list coverage versus sign balance/skewness,
  returned/missing boxplots, and rank-position and score-distribution panels.
  Individual plots include the top 15 returned species by adjusted p-value
  (ties resolved by raw p-value, then ID), followed by missing terms that pass
  size limits. This applies to every iteration and the final PDF. If fewer
  than 15 species have finite adjusted p-values, all available ones are plotted.
- `_raw_fgsea.tsv`: a separate fgseaMultilevel diagnostic result using the same
  ranked vector, parameters and seed. This adds another enrichment calculation
  per iteration; its results do not replace the pipeline result.

The existing `--p-debug-per-iteration-table-path` option remains available for
Python-side iteration result exports. Existing mapping behavior and separate
`make-psea-plots` workflow are preserved. Diagnostic PDFs handle empty sets and
panels without finite data.

Run dependency-light checks with:

```sh
python3 scripts/check_gsea_diagnostics.py
```

These check Python syntax, parameter forwarding, and R diagnostic tables/PDFs
for sparse, unranked and empty data. A complete pipeline test still requires
the q2-psea QIIME 2 environment with rpy2, clusterProfiler and fgsea.

## Optional depletion of the ranked background

Add `--p-remove-claimed-peptides-from-background` to `make-psea-table` to remove
claimed `all_tested_peptides` globally before the next iteration. The default
is false. This option requires iterative analysis. It applies to peptide IDs
or mapped epitope IDs, depending on the analysis mode.

The selected species is determined by the existing adjusted-p-value/NES rules.
Its tested IDs are removed from every GMT membership, including its own;
therefore those IDs are excluded by the score filtering step from the next
ranked list. Exclusions accumulate independently for each pair. Residuals and
the spline are not refitted. This continues to use all_tested_peptides, not
leading-edge-only removal.

Called species retain their result rows from the iteration when selected.
The output artifact combines those rows with the terminal results for uncalled
species. Their p-values and adjusted p-values correspond to their individual
iteration backgrounds and test families, not a single combined final test.

Numbered debug files describe each iteration. The `final_*` files describe
only the terminal, depleted analysis; they do not reintroduce previously
claimed peptides. `combined_psea_output.tsv` mirrors the combined result, and
`called_species_iterations.tsv` identifies the source iteration of each called
species. Consult its numbered PDF for each retained call. An exhausted
background produces empty result tables and an explanatory diagnostic PDF.
