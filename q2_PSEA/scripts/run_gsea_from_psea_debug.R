#!/usr/bin/env Rscript

usage <- paste(
    "Usage:",
    "Rscript run_gsea_from_psea_debug.R <gsea_input.tsv> <output.tsv>",
    "[--permutation-num N] [--min-size N] [--max-size N] [--seed N]",
    "[--missing-output missing.tsv]",
    sep = "\n  "
)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop(usage, call. = FALSE)
}

input_path <- args[[1]]
output_path <- args[[2]]

permutation_num <- 10000
min_size <- 15
max_size <- 2000
seed <- 1
missing_output <- NA_character_

i <- 3
while (i <= length(args)) {
    flag <- args[[i]]
    if (i == length(args)) {
        stop(paste("Missing value for", flag), call. = FALSE)
    }
    value <- args[[i + 1]]

    if (flag == "--permutation-num") {
        permutation_num <- as.integer(value)
    } else if (flag == "--min-size") {
        min_size <- as.integer(value)
    } else if (flag == "--max-size") {
        max_size <- as.integer(value)
    } else if (flag == "--seed") {
        seed <- as.integer(value)
    } else if (flag == "--missing-output") {
        missing_output <- value
    } else {
        stop(paste("Unknown argument:", flag, "\n", usage), call. = FALSE)
    }
    i <- i + 2
}

suppressPackageStartupMessages(library(clusterProfiler))

gsea_input <- read.delim(
    input_path,
    header = TRUE,
    stringsAsFactors = FALSE,
    check.names = FALSE
)

required_cols <- c("term", "gene", "maxZ", "deltaZ", "in_gene_list")
missing_cols <- setdiff(required_cols, colnames(gsea_input))
if (length(missing_cols) > 0) {
    stop(
        paste("Input table is missing columns:", paste(missing_cols, collapse = ", ")),
        call. = FALSE
    )
}

term_to_gene <- gsea_input[, c("term", "gene")]
term_to_gene <- term_to_gene[order(term_to_gene$gene), , drop = FALSE]

gene_rows <- gsea_input[gsea_input$in_gene_list %in% c(TRUE, "TRUE", "True", "true", 1), ]
gene_rows <- gene_rows[!duplicated(gene_rows$gene), ]
gene_list <- gene_rows$deltaZ
names(gene_list) <- gene_rows$gene
gene_list <- sort(gene_list, decreasing = TRUE)

set.seed(seed)
out <- GSEA(
    geneList = gene_list,
    TERM2GENE = term_to_gene,
    pvalueCutoff = 1,
    minGSSize = min_size,
    maxGSSize = max_size,
    eps = 1e-30,
    verbose = FALSE,
    nPermSimple = permutation_num,
    exponent = 1
)

if (nrow(out) == 0) {
    outtable <- data.frame(
        ID = numeric(),
        enrichmentScore = numeric(),
        NES = numeric(),
        p.adjust = numeric(),
        core_enrichment = character(),
        pvalue = numeric(),
        qvalue = numeric()
    )
} else {
    outtable <- attributes(out)[[1]][, c(
        "ID", "enrichmentScore", "NES", "p.adjust",
        "core_enrichment", "pvalue", "qvalue"
    )]
}

write.table(
    outtable,
    file = output_path,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
)

if (!is.na(missing_output)) {
    input_terms <- sort(unique(as.character(gsea_input$term)))
    output_terms <- sort(unique(as.character(outtable$ID)))
    missing_terms <- setdiff(input_terms, output_terms)
    missing_table <- data.frame(ID = missing_terms)
    write.table(
        missing_table,
        file = missing_output,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )
}
