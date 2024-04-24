qiime psea make-psea-table --p-scores-file example/IM0031_PV2T_25nt_raw_2mm_i1mm_Z-HDI75.tsv \
--p-pairs-file example/pairs.tsv \
--p-peptide-sets-file example/input.gmt \
--p-species-taxa-file example/species_taxa.tsv \
--p-threshold 0.750000 \
--p-min-size 3 \
--p-max-size 5000 \
--p-permutation-num 10000 \
--p-table-dir psea-example-tables \
--output-dir psea-example-outdir
