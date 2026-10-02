# Standalone species diagnostic plots

`plot_gsea_species.py` requires Python 3.9+, pandas, numpy and matplotlib.
It does not run GSEA and does not require QIIME 2 or R.

```sh
python plot_gsea_species.py final_gsea_input.tsv \
  --species 138951 12345 --output chosen_species.pdf

python plot_gsea_species.py final_gsea_input.tsv \
  --results final_gsea_output.tsv --top 15 --output top15.pdf

python plot_gsea_species.py final_gsea_input.tsv \
  --output all_species.pdf --png-dir species_pngs
```

Omitting `--species` and `--top` plots every species in the input. Each PDF page
contains the ranked background curve with species members highlighted and a
histogram of that species' included residuals. The histogram uses 30 bins by
default (`--bins` changes this). PNGs are optional. Existing output files are
overwritten.

The required input columns are `term`, `gene`, `deltaZ`, and `in_gene_list`.
IDs are read as strings, preserving leading zeros. Shared genes count once in
the background and once per species; duplicate membership rows do not inflate
histograms. Species without ranked members receive an explanatory empty panel.

The matching `final_ranked_gene_list.tsv` is automatically loaded when adjacent
to `final_gsea_input.tsv` (likewise for iteration-prefixed files). Use
`--ranked-list /path/to/file.tsv` to specify it explicitly. This is recommended
for reproducing the exact R background, including genes absent from all input
memberships. It must have `gene`, `score`, and `rank` columns and match the same
pair and iteration.

With only the input TSV, the background is reconstructed from unique included
genes. Existing `gene_list_rank` values are preserved, including gaps; if that
column is absent, scores are sorted descending and assigned ranks. A warning
notes that unrepresented background genes cannot be reconstructed. This plot
may therefore differ from the original R PDF.

`--results` optionally labels species as returned or missing and is required
for `--top`. Top species are ordered by adjusted p-value, then raw p-value,
then ID. This selects the most significant available results without imposing
a significance cutoff. Unlike the integrated diagnostic PDFs, `--top` selects
only the top N and does not add missing species automatically.
