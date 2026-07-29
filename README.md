# q2-PSEA

## Dependencies

## Installation
```sh
conda env create -n q2-psea-dev -f ./environment-files/q2-psea-qiime2-tiny-2026.7.yml
conda activate q2-psea-dev
Rscript install_r_packages.R
```

### Updating

## Tutorial
These are some basic instructions for using q2-PSEA for the current intended workflow. We will be wrapping this workflow in a Snakemake pipeline that will include pre and post processing steps that will allow for pure .tsvs to be passed in and returned.

- Import your zscores as `FeatureTable[Zscore]`
- Import your pairs as `PSEAPairs`
- Import your .gmt file as `GMT`

If you are using `species_taxa` or `species_color`, these are now QIIME 2 Metadata and as such need the following column headers:

- For `species_taxa` id and TaxID
- For `species_color` id and Color

Run `make_psea_table` with inputs/parameters as desired. If you would like to use the same seed as the old default pass `--p-seed 149`. If running in parallel pass the `--parallel` flag. Docs on parallelization in QIIME 2 may be found [here](https://use.qiime2.org/en/latest/references/parallel-configuration/). Just using the `--parallel` flag alone will naively create as many jobs as you have CPUs. It is not recommended to use the `--parallel` flag without additional configuration on HPC.

If you would like to use epitope_mapping then please pass the `--p-use-epitope-mapping` flag then either provide either mapped artifacts to `--i-epitope-map`, and `--i-peptide-sets-map` or provide peptide metadata to `--i-peptide-metadata` so the pipeline can create the mappings for you.

## More Information
[no_link]
