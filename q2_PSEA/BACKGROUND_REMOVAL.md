# Optional removal of claimed peptides from the ranked background

Add `--p-remove-claimed-peptides-from-background` to either:

- `qiime psea make-psea`
- `qiime psea make-psea-tables`

The option defaults to false and requires iterative analysis. Omitting it uses
the existing iterative workflow. Optional GSEA debug outputs are enabled with
`--p-debug-gsea-input-table-path`; see GSEA_DIAGNOSTICS.md.
The existing per-iteration table export remains available.

With the option enabled, the same significance rules select one species per
iteration. Its `all_tested_peptides` are removed from all GMT memberships,
including its own. The next GSEA run excludes these IDs from its ranked
background; exclusions accumulate per sample pair. For epitope-mapped analyses,
these IDs represent epitopes. This is not leading-edge-only removal.

Iteration 1 has the same input as the default workflow. Set the same explicit
`--p-seed` when comparing separate runs. Subsequent runs sort the remaining
residuals again; the spline and residual values are not refitted.

Called species retain their result row from the iteration when selected.
The output combines those rows with the terminal results for uncalled species.
Their adjusted p-values describe their individual iteration's test family,
not a newly adjusted combined family. Removal can exhaust the available
background; this is handled as an empty terminal result while preserving calls.

Tests: `tests/test_background_depletion.py` checks cumulative removal, retained
calls, mapped-input/first-iteration parity, exhaustion and invalid flag usage.
