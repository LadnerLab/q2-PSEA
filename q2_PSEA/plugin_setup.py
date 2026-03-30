#! /usr/bin/env python
import pandas as pd

from q2_types.feature_data import FeatureData
from q2_types.feature_table import FeatureTable
from q2_pepsirf.format_types import (
    PSEAScores,
    Epitope,
    MappedEpitope,
    GMT,
    Zscore,
)

import q2_PSEA
import q2_PSEA.actions.splines as splines

from q2_PSEA.actions.psea import (
    create_fgsea_table_for_pair,
    run_iterative_process_single_pair,
    run_iterative_peptide_analysis,
    make_psea_table,
)
from q2_PSEA.format_types import (
    PSEAPairsTSVFormat,
    PSEASpeciesTaxaTSVFormat,
    PSEASpeciesColorsTSVFormat,
    PSEAPairsDirFmt,
    PSEASpeciesTaxaDirFmt,
    PSEASpeciesColorsDirFmt,
)
from q2_PSEA.types import PSEAPairs, PSEASpeciesTaxa, PSEASpeciesColors
from qiime2.plugin import (
    Bool,
    Collection,
    Float,
    Int,
    List,
    Plugin,
    Str,
    Visualization,
    Choices,
)


# ---------------------------------------------------------------------------
# Plugin object
# ---------------------------------------------------------------------------

plugin = Plugin(
    "psea",
    version=q2_PSEA.__version__,
    website="https://github.com/LadnerLab/q2-PSEA.git",
    description="QIIME 2 plugin for Peptide Set Enrichment Analysis.",
)

# ---------------------------------------------------------------------------
# Register formats
# ---------------------------------------------------------------------------

plugin.register_formats(
    PSEAPairsTSVFormat,
    PSEASpeciesTaxaTSVFormat,
    PSEASpeciesColorsTSVFormat,
    PSEAPairsDirFmt,
    PSEASpeciesTaxaDirFmt,
    PSEASpeciesColorsDirFmt,
)

# ---------------------------------------------------------------------------
# Register semantic types
# ---------------------------------------------------------------------------

plugin.register_semantic_types(PSEAPairs, PSEASpeciesTaxa, PSEASpeciesColors)

# ---------------------------------------------------------------------------
# Map semantic types -> directory formats
# ---------------------------------------------------------------------------

plugin.register_semantic_type_to_format(PSEAPairs, PSEAPairsDirFmt)
plugin.register_semantic_type_to_format(PSEASpeciesTaxa, PSEASpeciesTaxaDirFmt)
plugin.register_semantic_type_to_format(
    PSEASpeciesColors, PSEASpeciesColorsDirFmt
)

# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------

@plugin.register_transformer
def _psea_pairs_tsv_to_df(ff: PSEAPairsTSVFormat) -> pd.DataFrame:
    return pd.read_csv(str(ff), sep="\t", header=0)


@plugin.register_transformer
def _df_to_psea_pairs_tsv(df: pd.DataFrame) -> PSEAPairsTSVFormat:
    result = PSEAPairsTSVFormat()
    df.to_csv(str(result), sep="\t", index=False)
    return result


@plugin.register_transformer
def _psea_species_taxa_tsv_to_str(ff: PSEASpeciesTaxaTSVFormat) -> str:
    return str(ff)


@plugin.register_transformer
def _psea_species_colors_tsv_to_str(ff: PSEASpeciesColorsTSVFormat) -> str:
    return str(ff)


# ---------------------------------------------------------------------------
# Register create_fgsea_table_for_pair as a method
# ---------------------------------------------------------------------------

plugin.methods.register_function(
    function=create_fgsea_table_for_pair,
    inputs={
        "processed_scores": FeatureTable[Zscore],
        "peptide_sets": GMT,
        "species_taxa": PSEASpeciesTaxa,
        "epitope_map": FeatureData[MappedEpitope],
        "mapped_processed_scores": FeatureTable[Zscore],
        "mapped_peptide_sets": GMT,
        "precomputed_fit": FeatureData[PSEAScores],
    },
    parameters={
        "sample_a": Str,
        "sample_b": Str,
        "threshold": Float,
        "permutation_num": Int,
        "min_size": Int,
        "max_size": Int,
        "spline_type": Str % Choices(splines.SPLINE_TYPES),
        "degree": Int,
        "seed": Int,
        "dof": Int,
    },
    parameter_descriptions={
        "sample_a": "Name of the first sample in the pair.",
        "sample_b": "Name of the second sample in the pair.",
        "threshold": (
            "Minimum Z-score a peptide must have to be included in GSEA."
        ),
        "permutation_num": (
            "Number of permutations. Minimum nominal p-value is ~1/perm."
        ),
        "min_size": (
            "Minimum number of peptides from a set that must appear in the"
            " data."
        ),
        "max_size": (
            "Maximum number of peptides from a set that can appear in the"
            " data."
        ),
        "spline_type": "Spline method used to fit the Z-score scatter.",
        "degree": (
            "Polynomial degree for spline fitting (affects 'cubic' only)."
        ),
        "seed": "Random seed for GSEA permutations.",
        "dof": (
            "Degrees of freedom for spline fitting (affects 'cubic' only)."
        ),
    },
    input_descriptions={
        "processed_scores": (
            "Log-scaled Z-score matrix (FeatureTable[Zscore])."
        ),
        "peptide_sets": "GMT peptide-set file mapping species to peptides.",
        "species_taxa": (
            "Optional TSV file mapping species names to taxonomy IDs."
        ),
        "epitope_map": (
            "Optional mapped-epitope table for epitope-level collapsing."
        ),
        "mapped_processed_scores": (
            "Optional epitope-level processed Z-score matrix."
        ),
        "mapped_peptide_sets": "Optional epitope-level GMT peptide sets.",
        "precomputed_fit": (
            "Optional precomputed maxZ/deltaZ from a prior call, stored as a"
            " two-column table (maxZ, deltaZ). When provided, spline fitting"
            " is skipped."
        ),
    },
    outputs=[("psea_table", FeatureData[PSEAScores])],
    output_descriptions={
        "psea_table": (
            "PSEA result table for this sample pair containing enrichment"
            " scores, p-values, and leading-edge peptides."
        ),
    },
    name="Create FGSEA Table for Pair",
    description=(
        "Compute a PSEA enrichment table for a single sample pair by fitting"
        " a spline to the Z-score scatter, computing residuals, and running"
        " GSEA via the clusterProfiler R package."
    ),
)

# ---------------------------------------------------------------------------
# Register run_iterative_process_single_pair as a pipeline
# ---------------------------------------------------------------------------

plugin.pipelines.register_function(
    function=run_iterative_process_single_pair,
    inputs={
        "processed_scores": FeatureTable[Zscore],
        "peptide_sets": GMT,
        "species_taxa": PSEASpeciesTaxa,
        "epitope_map": FeatureData[MappedEpitope],
        "mapped_processed_scores": FeatureTable[Zscore],
        "mapped_peptide_sets": GMT,
        "precomputed_fit": FeatureData[PSEAScores],
    },
    parameters={
        "sample_a": Str,
        "sample_b": Str,
        "threshold": Float,
        "permutation_num": Int,
        "min_size": Int,
        "max_size": Int,
        "spline_type": Str % Choices(splines.SPLINE_TYPES),
        "degree": Int,
        "seed": Int,
        "p_val_thresh": Float,
        "nes_thresh": Float,
        "tested_species": List[Str],
        "dof": Int,
    },
    parameter_descriptions={
        "sample_a": "Name of the first sample in the pair.",
        "sample_b": "Name of the second sample in the pair.",
        "threshold": "Minimum Z-score for GSEA inclusion.",
        "permutation_num": "Number of GSEA permutations.",
        "min_size": "Minimum peptide-set size.",
        "max_size": "Maximum peptide-set size.",
        "spline_type": "Spline method for Z-score fitting.",
        "degree": "Polynomial degree for spline fitting.",
        "seed": "Random seed for GSEA permutations.",
        "p_val_thresh": (
            "Adjusted p-value threshold for calling a species significant."
        ),
        "nes_thresh": (
            "Absolute NES threshold for calling a species significant."
        ),
        "tested_species": (
            "Species IDs already tested in prior iterations; used to avoid"
            " re-testing the same species."
        ),
        "dof": "Degrees of freedom for spline fitting.",
    },
    input_descriptions={
        "processed_scores": "Log-scaled Z-score matrix.",
        "peptide_sets": "Current (possibly filtered) GMT for this pair.",
        "species_taxa": "Optional species-to-taxon-ID mapping file.",
        "epitope_map": "Optional mapped-epitope table.",
        "mapped_processed_scores": "Optional epitope-level Z-score matrix.",
        "mapped_peptide_sets": "Optional epitope-level GMT.",
        "precomputed_fit": (
            "Optional precomputed maxZ/deltaZ from a prior call. When"
            " provided, spline fitting is skipped."
        ),
    },
    outputs=[
        ("psea_table", FeatureData[PSEAScores]),
        ("updated_peptide_sets", GMT),
    ],
    output_descriptions={
        "psea_table": (
            "PSEA result table for this iteration of this pair."
        ),
        "updated_peptide_sets": (
            "GMT with the leading-edge peptides of the most significant"
            " new species removed from all other species."
        ),
    },
    name="Run Iterative Process for Single Pair",
    description=(
        "One iteration of the iterative peptide-filtering procedure for a"
        " single sample pair. Calls the registered"
        " create_fgsea_table_for_pair method, identifies the top significant"
        " untested species, and removes its leading-edge peptides from all"
        " other species in the GMT."
    ),
)

# ---------------------------------------------------------------------------
# Register run_iterative_peptide_analysis as a pipeline
# ---------------------------------------------------------------------------

plugin.pipelines.register_function(
    function=run_iterative_peptide_analysis,
    inputs={
        "processed_scores": FeatureTable[Zscore],
        "pairs": PSEAPairs,
        "peptide_sets": GMT,
        "species_taxa": PSEASpeciesTaxa,
        "epitope_map": FeatureData[MappedEpitope],
        "mapped_processed_scores": FeatureTable[Zscore],
        "mapped_peptide_sets": GMT,
    },
    parameters={
        "threshold": Float,
        "permutation_num": Int,
        "min_size": Int,
        "max_size": Int,
        "spline_type": Str % Choices(splines.SPLINE_TYPES),
        "degree": Int,
        "seed": Int,
        "p_val_thresh": Float,
        "nes_thresh": Float,
        "dof": Int,
    },
    parameter_descriptions={
        "threshold": "Minimum Z-score for GSEA inclusion.",
        "permutation_num": "Number of GSEA permutations.",
        "min_size": "Minimum peptide-set size.",
        "max_size": "Maximum peptide-set size.",
        "spline_type": "Spline method for Z-score fitting.",
        "degree": "Polynomial degree for spline fitting.",
        "seed": "Random seed for GSEA permutations.",
        "p_val_thresh": "Adjusted p-value threshold for significance.",
        "nes_thresh": "Absolute NES threshold for significance.",
        "dof": "Degrees of freedom for spline fitting.",
    },
    input_descriptions={
        "processed_scores": "Log-scaled Z-score matrix for all samples.",
        "pairs": "TSV file listing sample pairs (one per row).",
        "peptide_sets": "Initial GMT peptide sets.",
        "species_taxa": "Optional species-to-taxon-ID mapping file.",
        "epitope_map": "Optional mapped-epitope table.",
        "mapped_processed_scores": "Optional epitope-level Z-score matrix.",
        "mapped_peptide_sets": "Optional epitope-level GMT.",
    },
    outputs=[("filtered_peptide_sets", Collection[GMT])],
    output_descriptions={
        "filtered_peptide_sets": (
            "One final filtered GMT artifact per pair, in the same order as"
            " the rows of the pairs file."
        ),
    },
    name="Run Iterative Peptide Analysis",
    description=(
        "Iteratively filter cross-reactive peptides across all sample pairs."
        " Calls run_iterative_process_single_pair for each pair in each"
        " iteration until no new significant species are found."
    ),
)

# ---------------------------------------------------------------------------
# Register make_psea_table as a pipeline
# ---------------------------------------------------------------------------

plugin.pipelines.register_function(
    function=make_psea_table,
    inputs={
        "scores": FeatureTable[Zscore],
        "pairs": PSEAPairs,
        "peptide_sets": GMT,
        "species_taxa": PSEASpeciesTaxa,
        "species_colors": PSEASpeciesColors,
        "epitope": FeatureData[Epitope],
    },
    parameters={
        "threshold": Float,
        "collapse": Str % Choices(["Bacterial", "Viral", "Both"]),
        "p_val_thresh": Float,
        "nes_thresh": Float,
        "min_size": Int,
        "max_size": Int,
        "permutation_num": Int,
        "spline_type": Str % Choices(splines.SPLINE_TYPES),
        "degree": Int,
        "dof": Int,
        "iterative_analysis": Bool,
        "seed": Int,
    },
    parameter_descriptions={
        "threshold": (
            "Minimum Z-score a peptide must have to be included in GSEA."
        ),
        "collapse": (
            "Category to collapse to epitope level. Only used when the"
            " epitope input is provided."
        ),
        "p_val_thresh": (
            "Adjusted p-value threshold for significance in volcano and"
            " scatter plots."
        ),
        "nes_thresh": "Absolute NES threshold for significance.",
        "min_size": "Minimum peptide-set size for GSEA.",
        "max_size": "Maximum peptide-set size for GSEA.",
        "permutation_num": (
            "Number of GSEA permutations. Minimum nominal p-value is"
            " ~1/perm."
        ),
        "spline_type": "Spline method used to fit the Z-score scatter.",
        "degree": (
            "Polynomial degree for spline fitting (affects 'cubic' only)."
        ),
        "dof": (
            "Degrees of freedom for spline fitting (affects 'cubic' only)."
        ),
        "iterative_analysis": (
            "If True, run the iterative peptide-filtering procedure to"
            " remove cross-reactive peptides before the final analysis."
            " Requires a GMT peptide_sets input."
        ),
        "seed": "Random seed for GSEA permutations.",
    },
    input_descriptions={
        "scores": (
            "Z-score matrix. Collapsed to epitope level if epitope is"
            " provided."
        ),
        "pairs": (
            "Tab-delimited file listing pairs of sample names (one pair per"
            " row, header required)."
        ),
        "peptide_sets": (
            "GMT file mapping species identifiers to the peptides linked to"
            " them. Collapsed to epitope level if epitope is provided."
        ),
        "species_taxa": (
            "Optional TSV file mapping species names to taxonomy IDs. When"
            " provided, enrichment results are annotated with species names."
        ),
        "species_colors": (
            "Optional TSV file mapping species names to HEX color codes used"
            " in output visualizations."
        ),
        "epitope": (
            "Optional epitope table. When provided, peptide-level residuals"
            " are collapsed to the epitope level before GSEA."
        ),
    },
    outputs=[
        ("scatter_plot", Visualization),
        ("volcano_plot", Visualization),
        ("ae_plots", Visualization),
        ("psea_tables", Collection[FeatureData[PSEAScores]]),
    ],
    output_descriptions={
        "scatter_plot": (
            "Z-score scatter plot with spline fit and highlighted leading-edge"
            " peptides for significant taxa."
        ),
        "volcano_plot": (
            "Volcano plot of normalized enrichment scores vs. adjusted"
            " p-values."
        ),
        "ae_plots": "Antibody-event summary bar plots.",
        "psea_tables": (
            "Per-pair PSEA result tables containing enrichment scores,"
            " p-values, and leading-edge peptides."
        ),
    },
    name="Make PSEA Table",
    description=(
        "QIIME 2 pipeline for Peptide Set Enrichment Analysis. Wraps R's"
        " clusterProfiler::GSEA to perform enrichment analysis on Z-score"
        " data, optionally collapsing to epitope level and/or running an"
        " iterative cross-reactivity filtering step."
    ),
)
