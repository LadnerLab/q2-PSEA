#!/usr/bin/env Rscript

usage <- paste(
    "Usage:",
    "Rscript run_gsea_from_psea_debug.R <gsea_input.tsv> <output.tsv>",
    "[--permutation-num N] [--min-size N] [--max-size N] [--seed N]",
    "[--missing-output missing.tsv] [--diagnostics-output diagnostics.tsv]",
    "[--fgsea-output raw_fgsea.tsv] [--diagnostics-plot-output plots.pdf]",
    "[--plot-terms comma,separated,ids]",
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
diagnostics_output <- NA_character_
fgsea_output <- NA_character_
diagnostics_plot_output <- NA_character_
plot_terms <- c("138951")

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
    } else if (flag == "--diagnostics-output") {
        diagnostics_output <- value
    } else if (flag == "--fgsea-output") {
        fgsea_output <- value
    } else if (flag == "--diagnostics-plot-output") {
        diagnostics_plot_output <- value
    } else if (flag == "--plot-terms") {
        plot_terms <- trimws(strsplit(value, ",")[[1]])
    } else {
        stop(paste("Unknown argument:", flag, "\n", usage), call. = FALSE)
    }
    i <- i + 2
}

suppressPackageStartupMessages(library(clusterProfiler))
suppressPackageStartupMessages(library(fgsea))

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
pathways <- split(as.character(term_to_gene$gene), as.character(term_to_gene$term))

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

if (!is.na(fgsea_output)) {
    set.seed(seed)
    fgsea_table <- fgsea::fgseaMultilevel(
        pathways = pathways,
        stats = gene_list,
        minSize = min_size,
        maxSize = max_size,
        eps = 1e-30,
        nPermSimple = permutation_num,
        gseaParam = 1,
        scoreType = "std"
    )

    if ("leadingEdge" %in% colnames(fgsea_table)) {
        fgsea_table$leadingEdge <- vapply(
            fgsea_table$leadingEdge,
            paste,
            collapse = "/",
            FUN.VALUE = character(1)
        )
    }

    write.table(
        as.data.frame(fgsea_table),
        file = fgsea_output,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )
}

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

calc_skewness <- function(x) {
    x <- x[!is.na(x)]
    if (length(x) < 3 || sd(x) == 0) {
        return(NA_real_)
    }
    mean((x - mean(x))^3) / sd(x)^3
}

build_diagnostics <- function(gsea_input, outtable, min_size, max_size, gene_list) {
    output_terms <- sort(unique(as.character(outtable$ID)))
    total_genes_in_gene_list <- length(unique(names(gene_list)))

    diagnostics <- do.call(
        rbind,
        lapply(
            split(gsea_input, as.character(gsea_input$term)),
            function(term_rows) {
                in_gene_list <- term_rows[
                    term_rows$in_gene_list %in% c(TRUE, "TRUE", "True", "true", 1),
                    ,
                    drop = FALSE
                ]
                ranked_scores <- in_gene_list$deltaZ
                ranked_scores <- ranked_scores[!is.na(ranked_scores)]
                term_id <- as.character(term_rows$term[[1]])
                effective_size <- length(unique(as.character(in_gene_list$gene)))

                data.frame(
                    ID = term_id,
                    input_term_gene_rows = nrow(term_rows),
                    unique_input_genes = length(unique(as.character(term_rows$gene))),
                    genes_in_gene_list = effective_size,
                    fraction_of_ranked_gene_list = if (total_genes_in_gene_list > 0) {
                        effective_size / total_genes_in_gene_list
                    } else {
                        NA_real_
                    },
                    positive_deltaZ = sum(ranked_scores > 0),
                    negative_deltaZ = sum(ranked_scores < 0),
                    positive_fraction = if (length(ranked_scores) > 0) {
                        sum(ranked_scores > 0) / length(ranked_scores)
                    } else {
                        NA_real_
                    },
                    zero_deltaZ = sum(term_rows$deltaZ == 0, na.rm = TRUE),
                    na_deltaZ = sum(is.na(term_rows$deltaZ)),
                    mean_deltaZ = if (length(ranked_scores) > 0) mean(ranked_scores) else NA_real_,
                    median_deltaZ = if (length(ranked_scores) > 0) median(ranked_scores) else NA_real_,
                    skewness_deltaZ = calc_skewness(ranked_scores),
                    min_deltaZ = if (length(ranked_scores) > 0) min(ranked_scores) else NA_real_,
                    max_deltaZ = if (length(ranked_scores) > 0) max(ranked_scores) else NA_real_,
                    passes_min_size = effective_size >= min_size,
                    passes_max_size = effective_size <= max_size,
                    appears_in_gsea_output = term_id %in% output_terms,
                    output_has_na_pvalue = if (term_id %in% output_terms) {
                        any(is.na(outtable$pvalue[as.character(outtable$ID) == term_id]))
                    } else {
                        NA
                    },
                    output_has_na_p_adjust = if (term_id %in% output_terms) {
                        any(is.na(outtable$p.adjust[as.character(outtable$ID) == term_id]))
                    } else {
                        NA
                    },
                    likely_exclusion_reason = if (effective_size < min_size) {
                        "below_min_size_after_gene_list_filter"
                    } else if (effective_size > max_size) {
                        "above_max_size_after_gene_list_filter"
                    } else if (effective_size == 0) {
                        "no_genes_in_ranked_gene_list"
                    } else if (!(term_id %in% output_terms)) {
                        "not_returned_by_clusterProfiler_or_fgsea"
                    } else {
                        "returned_by_gsea"
                    },
                    stringsAsFactors = FALSE
                )
            }
        )
    )

    diagnostics <- diagnostics[order(diagnostics$appears_in_gsea_output, diagnostics$ID), ]
    diagnostics
}

plot_diagnostics <- function(gsea_input, diagnostics, gene_list, output_path, plot_terms) {
    pdf(output_path, width = 11, height = 8.5)
    on.exit(dev.off(), add = TRUE)

    status <- ifelse(diagnostics$appears_in_gsea_output, "Returned", "Missing")
    point_cols <- ifelse(diagnostics$appears_in_gsea_output, "#377eb8", "#e41a1c")
    suspicious <- diagnostics[
        !diagnostics$appears_in_gsea_output
            & diagnostics$passes_min_size
            & diagnostics$passes_max_size,
        ,
        drop = FALSE
    ]
    comparison <- diagnostics[
        diagnostics$ID %in% plot_terms,
        ,
        drop = FALSE
    ]
    highlighted <- unique(rbind(suspicious, comparison))

    par(mfrow = c(2, 2), mar = c(4.5, 4.5, 3, 1))
    plot(
        diagnostics$fraction_of_ranked_gene_list,
        diagnostics$positive_fraction,
        pch = 19,
        col = point_cols,
        xlab = "Fraction of ranked gene list",
        ylab = "Fraction positive deltaZ",
        main = "Ranked-List Fraction vs Sign Balance"
    )
    legend(
        "topright",
        legend = c("Returned", "Missing"),
        col = c("#377eb8", "#e41a1c"),
        pch = 19,
        bty = "n"
    )
    if (nrow(highlighted) > 0) {
        text(
            highlighted$fraction_of_ranked_gene_list,
            highlighted$positive_fraction,
            labels = highlighted$ID,
            pos = 4,
            cex = 0.75
        )
    }

    plot(
        diagnostics$fraction_of_ranked_gene_list,
        diagnostics$skewness_deltaZ,
        pch = 19,
        col = point_cols,
        xlab = "Fraction of ranked gene list",
        ylab = "deltaZ skewness",
        main = "Ranked-List Fraction vs Score Skewness"
    )
    abline(h = 0, lty = 2, col = "gray50")
    if (nrow(highlighted) > 0) {
        text(
            highlighted$fraction_of_ranked_gene_list,
            highlighted$skewness_deltaZ,
            labels = highlighted$ID,
            pos = 4,
            cex = 0.75
        )
    }

    boxplot(
        positive_fraction ~ status,
        data = diagnostics,
        col = c("#e41a1c", "#377eb8"),
        ylab = "Fraction positive deltaZ",
        main = "Sign Balance by Output Status"
    )

    boxplot(
        skewness_deltaZ ~ status,
        data = diagnostics,
        col = c("#e41a1c", "#377eb8"),
        ylab = "deltaZ skewness",
        main = "Score Skewness by Output Status"
    )
    abline(h = 0, lty = 2, col = "gray50")

    if (nrow(highlighted) == 0) {
        plot.new()
        title("No Highlighted Species Found")
        return(invisible(NULL))
    }

    ranked_names <- names(gene_list)
    ranked_scores <- as.numeric(gene_list)

    for (term_id in highlighted$ID) {
        term_diag <- highlighted[highlighted$ID == term_id, , drop = FALSE]
        term_status <- ifelse(term_diag$appears_in_gsea_output, "returned", "missing")
        term_col <- ifelse(term_diag$appears_in_gsea_output, "#377eb8", "#e41a1c")
        term_rows <- gsea_input[
            as.character(gsea_input$term) == term_id
                & gsea_input$in_gene_list %in% c(TRUE, "TRUE", "True", "true", 1),
            ,
            drop = FALSE
        ]
        positions <- match(unique(as.character(term_rows$gene)), ranked_names)
        positions <- sort(positions[!is.na(positions)])
        term_scores <- term_rows$deltaZ[match(ranked_names[positions], term_rows$gene)]
        term_scores <- term_scores[!is.na(term_scores)]

        par(mfrow = c(2, 1), mar = c(4.5, 4.5, 3, 1))
        plot(
            seq_along(ranked_scores),
            ranked_scores,
            type = "l",
            col = "gray55",
            xlab = "Rank in gene_list",
            ylab = "deltaZ",
            main = paste("Rank Positions for", term_status, "Species", term_id)
        )
        abline(h = 0, lty = 2, col = "gray70")
        if (length(positions) > 0) {
            rug(positions, col = term_col, ticksize = 0.08)
            points(
                positions,
                ranked_scores[positions],
                pch = 16,
                col = adjustcolor(term_col, alpha.f = 0.35),
                cex = 0.65
            )
        }

        hist(
            term_scores,
            breaks = 30,
            col = term_col,
            border = "white",
            xlab = "deltaZ for this species",
            main = paste(
                "Score Distribution:",
                term_id,
                "|",
                term_status,
                "| n =",
                length(term_scores),
                "| positive fraction =",
                round(term_diag$positive_fraction, 3)
            )
        )
        abline(v = 0, lty = 2, col = "gray30")
    }
}

diagnostics <- NULL
if (!is.na(diagnostics_output) || !is.na(diagnostics_plot_output)) {
    diagnostics <- build_diagnostics(
        gsea_input,
        outtable,
        min_size,
        max_size,
        gene_list
    )
}

if (!is.na(diagnostics_output)) {
    write.table(
        diagnostics,
        file = diagnostics_output,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )
}

if (!is.na(diagnostics_plot_output)) {
    plot_diagnostics(
        gsea_input,
        diagnostics,
        gene_list,
        diagnostics_plot_output,
        plot_terms
    )
}
