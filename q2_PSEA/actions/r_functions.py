from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

r_functions = """
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

    if (nrow(gsea_input) == 0) return(data.frame(ID = character()))

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

select_top_diagnostic_terms <- function(outtable, n = 15L) {
    eligible <- outtable[is.finite(outtable$p.adjust), , drop = FALSE]
    eligible <- eligible[order(eligible$p.adjust, eligible$pvalue,
                               as.character(eligible$ID), na.last = TRUE), , drop = FALSE]
    head(unique(as.character(eligible$ID)), n)
}

plot_diagnostics <- function(gsea_input, diagnostics, gene_list, output_path, plot_terms) {
    pdf(output_path, width = 11, height = 8.5)
    on.exit(dev.off(), add = TRUE)

    if (nrow(diagnostics) == 0) {
        plot.new()
        title("No peptide sets available for diagnostics")
        return(invisible(NULL))
    }
    plot <- function(x, y, ...) {
        if (!any(is.finite(x) & is.finite(y))) {
            plot.new()
            title("No finite data for this diagnostic")
        } else graphics::plot(x, y, ...)
    }
    boxplot <- function(formula, data, ...) {
        values <- model.frame(formula, data = data)[[1]]
        if (!any(is.finite(values))) {
            plot.new()
            title("No finite data for this diagnostic")
        } else graphics::boxplot(formula, data = data, ...)
    }
    hist <- function(x, ...) {
        if (!any(is.finite(x))) {
            plot.new()
            title("No ranked scores for this species")
        } else graphics::hist(x, ...)
    }
    status <- ifelse(diagnostics$appears_in_gsea_output, "Returned", "Missing")
    diagnostics$status <- status
    point_cols <- ifelse(diagnostics$appears_in_gsea_output, "#377eb8", "#e41a1c")
    suspicious <- diagnostics[
        !diagnostics$appears_in_gsea_output
            & diagnostics$passes_min_size
            & diagnostics$passes_max_size,
        ,
        drop = FALSE
    ]
    comparison_ids <- plot_terms[plot_terms %in% diagnostics$ID]
    comparison <- diagnostics[
        match(comparison_ids, diagnostics$ID), , drop = FALSE
    ]
    highlighted <- unique(rbind(comparison, suspicious))

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

psea <- function(
        maxZ,
        deltaZ,
        peptide_sets,
        species_file,
        threshold,
        # default number of permutations from hardcoded parameter in original R
        # code
        permutation_num = 10000,
        min_size,
        max_size,
        seed,
        debug_prefix = ""
) {
    library(clusterProfiler)
    peptide_sets <- peptide_sets[order(peptide_sets$gene), , drop=FALSE]
    gene_list <- sort(
        deltaZ[intersect(which(maxZ > threshold), which(deltaZ != 0))],
        decreasing=TRUE
    )

    term_to_gene <- peptide_sets[, c("term", "gene")]
    if (debug_prefix != "") {
        write_debug <- function(table, suffix) {
            write.table(table, paste0(debug_prefix, suffix), sep="\t",
                        quote=FALSE, row.names=FALSE)
        }
        gsea_input <- term_to_gene
        gsea_input$maxZ <- as.numeric(maxZ[as.character(gsea_input$gene)])
        gsea_input$deltaZ <- as.numeric(deltaZ[as.character(gsea_input$gene)])
        gsea_input$in_gene_list <- as.character(gsea_input$gene) %in% names(gene_list)
        gsea_input$gene_list_rank <- match(as.character(gsea_input$gene), names(gene_list))
        gsea_input$gene_list_score <- as.numeric(gene_list[gsea_input$gene_list_rank])
        write_debug(gsea_input, "_gsea_input.tsv")
        write_debug(data.frame(gene=names(gene_list), score=as.numeric(gene_list),
                               rank=seq_along(gene_list)), "_ranked_gene_list.tsv")
        write_debug(term_to_gene, "_term_to_gene.tsv")
        write_debug(data.frame(threshold=threshold, permutation_num=permutation_num,
                               min_size=min_size, max_size=max_size, seed=seed),
                    "_parameters.tsv")
    }
    set.seed(seed)
    out <- NULL
    if (length(gene_list) > 0 && nrow(term_to_gene) > 0) out=GSEA(
        geneList=gene_list,
        TERM2GENE=term_to_gene,
        pvalueCutoff=1,
        minGSSize=min_size,
        maxGSSize=max_size,
        eps=1e-30,
        verbose=FALSE,
        nPermSimple=permutation_num,
        exponent=1
    )

    if (is.null(out) || nrow(as.data.frame(out)) == 0) {
        outtable_pre <- data.frame(
            ID=character(), enrichmentScore=numeric(), NES=numeric(),
            p.adjust=numeric(), core_enrichment=character(),
            pvalue=numeric(), qvalue=numeric()
        )
    } else {
        outtable_pre <- as.data.frame(out)[, c(
            "ID", "enrichmentScore", "NES", "p.adjust",
            "core_enrichment", "pvalue", "qvalue"
        )]
    }

    all_peptides = unlist(lapply(
        lapply(
            attributes(out)$geneSets,
            function(X) intersect(X, names(which(maxZ > threshold)))
        ),
        function(X) paste(X,collapse="/")
    ))
    all_tested_peptides <- as.character(all_peptides[match(
        as.character(outtable_pre$ID), names(all_peptides)
    )])

    outtable <- cbind(outtable_pre, all_tested_peptides)

    if (species_file != "")
    {
        species <- read.csv(file=species_file, sep="\t", header=FALSE)
        species[, 2] <- trimws(as.character(species[, 2]))
        outtable[, "ID"] <- trimws(as.character(outtable[, "ID"]))
        species_name <- species[
            match(as.numeric(outtable[, "ID"]), as.numeric(species[, 2])), 1
        ]
        outtable <- cbind(outtable, species_name)
    }

    if (debug_prefix != "") {
        write_debug(outtable, "_gsea_output.tsv")
        write_debug(data.frame(ID=setdiff(as.character(unique(term_to_gene$term)),
                                          as.character(outtable$ID))), "_missing.tsv")
        diagnostics <- build_diagnostics(gsea_input, outtable, min_size, max_size, gene_list)
        write_debug(diagnostics, "_diagnostics.tsv")
        plot_diagnostics(gsea_input, diagnostics, gene_list,
                         paste0(debug_prefix, "_diagnostics.pdf"),
                         select_top_diagnostic_terms(outtable))
        # Preserve the analysis RNG state around the additional diagnostic run.
        rng_state <- .Random.seed
        on.exit(assign(".Random.seed", rng_state, envir=.GlobalEnv), add=TRUE)
        set.seed(seed)
        raw <- data.frame(pathway=character(), pval=numeric(), padj=numeric(),
                          ES=numeric(), NES=numeric(), size=integer(), leadingEdge=character())
        if (length(gene_list) > 0 && nrow(term_to_gene) > 0) raw <- fgsea::fgseaMultilevel(
            pathways=split(as.character(term_to_gene$gene), as.character(term_to_gene$term)),
            stats=gene_list, minSize=min_size, maxSize=max_size, eps=1e-30,
            nPermSimple=permutation_num, gseaParam=1, scoreType="std")
        if ("leadingEdge" %in% colnames(raw)) {
            raw$leadingEdge <- vapply(raw$leadingEdge, paste, collapse="/", FUN.VALUE=character(1))
        }
        write_debug(as.data.frame(raw), "_raw_fgsea.tsv")
    }

    return(outtable)
}
"""

INTERNAL = SignatureTranslatedAnonymousPackage(r_functions, "internal")
