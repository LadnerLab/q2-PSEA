from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

r_functions = """
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
        seed
) {
    library(clusterProfiler)
    peptide_sets <- peptide_sets[order(peptide_sets$gene), , drop=FALSE]
    gene_list <- sort(
        deltaZ[intersect(which(maxZ > threshold), which(deltaZ != 0))],
        decreasing=TRUE
    )

    term_to_gene <- peptide_sets[, c("term", "gene")]
    set.seed(seed)
    out=GSEA(
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
    outtable_pre <- attributes(out)[[1]][,c(
        "ID", "enrichmentScore", "NES", "p.adjust",
        "core_enrichment", "pvalue", "qvalue"
    )]

    all_peptides = unlist(lapply(
        lapply(
            attributes(out)$geneSets,
            function(X) intersect(X, names(which(maxZ > threshold)))
        ),
        function(X) paste(X,collapse="/")
    ))
    all_tested_peptides <- all_peptides [match(
        row.names(outtable_pre), names(all_peptides)
    )]

    outtable <- cbind(outtable_pre, all_tested_peptides)

    if (species_file != "")
    {
        species <- read.csv(file=species_file, sep="\t", header=FALSE)
        species_name <- species[match(outtable[, "ID"], species[, 2]), 1]
        outtable <- cbind(outtable, species_name)
    }

    return(outtable)
}
"""

INTERNAL = SignatureTranslatedAnonymousPackage(r_functions, "internal")
