# q2-PSEA

## Dependencies

## Installation
```sh
conda env create -n q2-psea-dev -f ./environment-files/q2-psea-qiime2-tiny-2026.4.yml
conda activate q2-psea-dev
Rscript install_r_packages.R
```

### Updating

## Tutorial
These are some basic instructions for using q2-PSEA for the current intended workflow. We will be wrapping this workflow in a Snakemake pipeline that will include pre and post processing steps that will allow for pure .tsvs to be passed in and returned.

- Import your zscores as `FeatureTable[Zscore]`
- Import your pairs as `PSEAPairs`
- Import your .gmt file as `GMT`
- If mapping, import your IN2 metadata as `FeatureData[Epitope]`

If you are using `species_taxa` or `species_color`, these are now QIIME 2 Metdata and as such need the following column headers:

- For `species_taxa` id and TaxID
- For `species_color` id and Color

Run `make_psea_table` with inputs/parameters as desired.

To run `enriched_subtypes` first put your `FeatureData[Epitope]` through `create_epitope_map` then pass the output from `create_epitope_map` and the psea_tables you got from `make_psea_table` into `enriched_subtypes`

## More Information
[no_link]
