#! /usr/bin/env python
import pandas as pd

from q2_types.feature_data import FeatureData
from q2_types.feature_table import FeatureTable
from q2_pepsirf.format_types import (
    PSEAScores,
    Enriched,
    Epitope,
    MappedEpitope,
    GMT,
    Zscore,
)

import q2_PSEA
import q2_PSEA.actions.splines as splines

from q2_PSEA.actions.psea import (
    _compute_pair_fit_and_residuals,
    count_antibody_events,
    create_fgsea_table_for_pair,
    process_scores,
    run_iterative_process_single_pair,
    make_psea_table,
)
from q2_PSEA.actions.visualizers import volcano, zscatter, aeplots
from q2_PSEA.actions.epitope import (
    create_epitope_map,
    epitope_zscore,
    taxa_to_epitope,
    enriched_subtypes,
)
from q2_PSEA.format_types import (
    PSEAAECountsDirFmt,
    PSEAAECountsTSVFormat,
    PSEAPairsTSVFormat,
    PSEAPairsDirFmt,
    SplineDirFmt,
    SplineTSVFormat,
)
from q2_PSEA.types import PSEAAECounts, PSEAPairs, Spline
from qiime2.plugin import (
    Bool,
    Collection,
    Float,
    Int,
    List,
    Metadata,
    Plugin,
    Range,
    Str,
    Visualization,
    Choices,
    Properties,
    TypeMatch,
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
    PSEAAECountsDirFmt, PSEAAECountsTSVFormat,
    PSEAPairsTSVFormat, PSEAPairsDirFmt,
    SplineDirFmt, SplineTSVFormat,
)

# ---------------------------------------------------------------------------
# Register semantic types
# ---------------------------------------------------------------------------

plugin.register_semantic_types(PSEAAECounts, PSEAPairs, Spline)

# ---------------------------------------------------------------------------
# Map semantic types -> directory formats
# ---------------------------------------------------------------------------

plugin.register_semantic_type_to_format(PSEAAECounts, PSEAAECountsDirFmt)
plugin.register_semantic_type_to_format(PSEAPairs, PSEAPairsDirFmt)
plugin.register_semantic_type_to_format(
    FeatureData[Spline], SplineDirFmt
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
def _ae_counts_tsv_to_df(ff: PSEAAECountsTSVFormat) -> pd.DataFrame:
    return pd.read_csv(str(ff), sep="\t", header=0)


@plugin.register_transformer
def _df_to_ae_counts_tsv(df: pd.DataFrame) -> PSEAAECountsTSVFormat:
    result = PSEAAECountsTSVFormat()
    df.to_csv(str(result), sep="\t", index=False)
    return result


@plugin.register_transformer
def _spline_tsv_to_df(ff: SplineTSVFormat) -> pd.DataFrame:
    return pd.read_csv(str(ff), sep="\t", index_col=0)


@plugin.register_transformer
def _df_to_spline_tsv(df: pd.DataFrame) -> SplineTSVFormat:
    result = SplineTSVFormat()
    df.to_csv(str(result), sep="\t", index=True)
    return result


# ---------------------------------------------------------------------------
# Register process_scores as a method
# ---------------------------------------------------------------------------

PROCESS_SCORES_MATCH = TypeMatch([Zscore, Zscore % Properties("mapped")])

plugin.methods.register_function(
    function=process_scores,
    inputs={
        "scores": FeatureTable[PROCESS_SCORES_MATCH],
        "pairs": PSEAPairs,
    },
    parameters={},
    outputs=[("processed_scores", FeatureTable[PROCESS_SCORES_MATCH])],
    input_descriptions={
        "scores": "Z-score matrix (FeatureTable[Zscore]).",
        "pairs": (
            "Tab-delimited file listing sample pairs (one per row, header"
            " required)."
        ),
    },
    parameter_descriptions={},
    output_descriptions={
        "processed_scores": (
            "Log-scaled Z-score matrix containing only the samples referenced"
            " in the pairs file."
        ),
    },
    name="Process Scores",
    description=(
        "Selects the samples referenced in the pairs file from the Z-score"
        " matrix and applies log-scaling to produce the processed Z-score"
        " matrix used in PSEA."
    ),
)

# ---------------------------------------------------------------------------
# Register _compute_pair_fit_and_residuals as a method
# ---------------------------------------------------------------------------

plugin.methods.register_function(
    function=_compute_pair_fit_and_residuals,
    inputs={
        "processed_scores": FeatureTable[Zscore],
        "epitope_map": FeatureData[MappedEpitope],
    },
    parameters={
        "sample_a": Str,
        "sample_b": Str,
        "spline_type": Str % Choices(splines.SPLINE_TYPES),
        "degree": Int,
        "dof": Int,
    },
    outputs=[("spline_fit", FeatureData[Spline])],
    input_descriptions={
        "processed_scores": (
            "Log-scaled Z-score matrix (FeatureTable[Zscore])."
        ),
        "epitope_map": (
            "Optional mapped-epitope table. When provided, maxZ and deltaZ"
            " are collapsed from peptide to epitope level."
        ),
    },
    parameter_descriptions={
        "sample_a": "Name of the first sample in the pair.",
        "sample_b": "Name of the second sample in the pair.",
        "spline_type": "Spline method used to fit the Z-score scatter.",
        "degree": (
            "Polynomial degree for spline fitting (affects 'cubic' only)."
        ),
        "dof": (
            "Degrees of freedom for spline fitting (affects 'cubic' only)."
        ),
    },
    output_descriptions={
        "spline_fit": (
            "Spline fit and residuals for this sample pair, stored as a"
            " four-column table (x, yfit, maxZ, deltaZ)."
        ),
    },
    name="Compute Pair Spline Fit and Residuals",
    description=(
        "Fits a spline to the Z-score scatter plot for a single sample pair"
        " and computes per-peptide (or per-epitope) residuals. Returns a"
        " FeatureData[Spline] artifact containing the x coordinates, fitted"
        " y values, maxZ, and deltaZ."
    ),
)

# ---------------------------------------------------------------------------
# Register count_antibody_events as a method
# ---------------------------------------------------------------------------

plugin.methods.register_function(
    function=count_antibody_events,
    inputs={
        "psea_tables": Collection[FeatureData[PSEAScores]],
    },
    parameters={
        "p_val_thresh": Float,
        "nes_thresh": Float,
        "taxa_access": Str,
    },
    outputs=[
        ("pos_ae_counts", PSEAAECounts),
        ("neg_ae_counts", PSEAAECounts),
    ],
    input_descriptions={
        "psea_tables": (
            "Per-pair PSEA result tables produced by"
            " create_fgsea_table_for_pair."
        ),
    },
    parameter_descriptions={
        "p_val_thresh": (
            "Adjusted p-value threshold; taxa below this value are counted"
            " as significant events."
        ),
        "nes_thresh": (
            "Absolute NES threshold; taxa whose absolute NES exceeds this"
            " value are counted as significant events."
        ),
        "taxa_access": (
            "Column name in the PSEA tables used to identify taxa (e.g."
            " 'ID' or 'species_name')."
        ),
    },
    output_descriptions={
        "pos_ae_counts": (
            "Species-level counts of significant positive-NES antibody"
            " events across all pairs, sorted by event count descending."
        ),
        "neg_ae_counts": (
            "Species-level counts of significant negative-NES antibody"
            " events across all pairs, sorted by event count descending."
        ),
    },
    name="Count Antibody Events",
    description=(
        "Counts the number of sample pairs in which each taxon is"
        " significantly enriched (positive or negative NES) according to"
        " the supplied p-value and NES thresholds."
    ),
)

# ---------------------------------------------------------------------------
# Register create_fgsea_table_for_pair as a method
# ---------------------------------------------------------------------------


plugin.methods.register_function(
    function=create_fgsea_table_for_pair,
    inputs={
        "processed_scores": FeatureTable[Zscore],
        "peptide_sets": GMT,
        "precomputed_fit": FeatureData[Spline],
    },
    parameters={
        "sample_a": Str,
        "sample_b": Str,
        "threshold": Float,
        "permutation_num": Int,
        "min_size": Int,
        "max_size": Int,
        "seed": Int,
        "species_taxa": Metadata,
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
        "seed": "Random seed for GSEA permutations.",
        "species_taxa": (
            "Optional Metadata mapping species names (IDs) to taxonomy IDs."
            " When provided, enrichment results are annotated with species"
            " names."
        ),
    },
    input_descriptions={
        "processed_scores": (
            "Log-scaled Z-score matrix (FeatureTable[Zscore])."
        ),
        "peptide_sets": "GMT peptide-set file mapping species to peptides.",
        "precomputed_fit": (
            "Optional precomputed spline fit from a prior call"
            " (FeatureData[Spline] with x, yfit, maxZ, deltaZ columns)."
            " When provided, spline fitting is skipped."
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

plugin.methods.register_function(
    function=run_iterative_process_single_pair,
    inputs={
        "processed_scores": FeatureTable[Zscore],
        "peptide_sets": GMT,
        "precomputed_fit": FeatureData[Spline],
        "epitope_map": FeatureData[MappedEpitope],
        "mapped_peptide_sets": GMT % Properties("mapped"),
    },
    parameters={
        "sample_a": Str,
        "sample_b": Str,
        "threshold": Float,
        "permutation_num": Int,
        "min_size": Int,
        "max_size": Int,
        "seed": Int,
        "p_val_thresh": Float,
        "nes_thresh": Float,
        "species_taxa": Metadata,
    },
    parameter_descriptions={
        "sample_a": "Name of the first sample in the pair.",
        "sample_b": "Name of the second sample in the pair.",
        "threshold": "Minimum Z-score for GSEA inclusion.",
        "permutation_num": "Number of GSEA permutations.",
        "min_size": "Minimum peptide-set size.",
        "max_size": "Maximum peptide-set size.",
        "seed": "Random seed for GSEA permutations.",
        "p_val_thresh": (
            "Adjusted p-value threshold for calling a species significant."
        ),
        "nes_thresh": (
            "Absolute NES threshold for calling a species significant."
        ),
        "species_taxa": (
            "Optional Metadata mapping species names (IDs) to taxonomy IDs."
        ),
    },
    input_descriptions={
        "processed_scores": "Log-scaled Z-score matrix.",
        "peptide_sets": "Current (possibly filtered) GMT for this pair.",
        "precomputed_fit": (
            "Optional precomputed maxZ/deltaZ from a prior call. When"
            " provided, spline fitting is skipped."
        ),
        "epitope_map": (
            "Optional epitope map passed in if data is collapsed to epitope"
            " level"
        ),
        "mapped_peptide_sets": (
            "Optional mapped peptide sets passed in if data is collapsed to"
            " epitope level."
        )
    },
    outputs=[
        ("updated_peptide_sets", GMT),
    ],
    output_descriptions={
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
# Register make_psea_table as a pipeline
# ---------------------------------------------------------------------------

plugin.pipelines.register_function(
    function=make_psea_table,
    inputs={
        "scores": FeatureTable[Zscore],
        "pairs": PSEAPairs,
        "peptide_sets": GMT,
        "epitope": FeatureData[Epitope],
        "epitope_map": FeatureData[MappedEpitope],
        "mapped_zscores": FeatureTable[Zscore % Properties("mapped")],
        "mapped_gmt": GMT % Properties("mapped")
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
        "species_taxa": Metadata,
        "species_colors": Metadata,
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
        "species_taxa": (
            "Optional Metadata mapping species names (IDs) to taxonomy IDs."
            " When provided, enrichment results are annotated with species"
            " names."
        ),
        "species_colors": (
            "Optional Metadata mapping species names (IDs) to HEX color"
            " codes used in output visualizations."
        ),
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
        "epitope": (
            "Optional epitope table. When provided, peptide-level residuals"
            " are collapsed to the epitope level before GSEA."
        ),
        "epitope_map": (
            "Optional already collapsed epitope table. When provided, this"
            " table is used in GSEA."
            "NOTE: Must be passed with mapped_zscores and mapped_gmt."
        ),
        "mapped_zscores": (
            "Optional already collapsed zscores. When provided, these"
            " scores are used in GSEA but NOT for spline fitting."
            "NOTE: Must be passed with eptiope_map and mapped_gmt."
        ),
        "mapped_gmt": (
            "Optional already collapsed epitope peptide sets. When provided,"
            " these peptides are used in GSEA."
            "NOTE: Must be passed with epitope_map and mapped_zscores."
        )
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

# ---------------------------------------------------------------------------
# Register volcano visualizer
# ---------------------------------------------------------------------------

plugin.visualizers.register_function(
    function=volcano,
    inputs={
        "pairs": PSEAPairs,
        "psea_tables": Collection[FeatureData[PSEAScores]],
    },
    parameters={
        "x": List[Float],
        "y": List[Float],
        "taxa": List[Str],
        "xy_access": List[Str],
        "taxa_access": Str,
        "x_threshold": Float,
        "y_threshold": Float,
        "log": Bool,
        "xy_labels": List[Str],
        "colors_file": Metadata,
    },
    input_descriptions={
        "pairs": (
            "Tab-delimited file listing pairs of sample names (one pair per"
            " row, header required)."
        ),
        "psea_tables": (
            "Per-pair PSEA result tables. When provided, x, y, and taxa are"
            " read from these artifacts using xy_access and taxa_access."
        ),
    },
    parameter_descriptions={
        "x": "Coordinates along the x-axis at which to plot points.",
        "y": "Coordinates along the y-axis at which to plot points.",
        "taxa": (
            "List of identifiers positionally associated with the p-values"
            " and enrichment scores. Displayed on hover."
        ),
        "xy_access": (
            "Column names in the PSEA table for x and y values, respectively."
        ),
        "taxa_access": (
            "Column name in the PSEA table used to grab highlighting"
            " information."
        ),
        "x_threshold": (
            "Minimum absolute enrichment score for a taxon to be highlighted."
        ),
        "y_threshold": (
            "Maximum adjusted p-value for a taxon to be highlighted."
        ),
        "log": (
            "If True, plot -log10(y) in ascending order; otherwise plot y"
            " values in descending order."
        ),
        "xy_labels": "Axis labels for x and y, respectively.",
        "colors_file": (
            "Optional Metadata mapping species names (IDs) to HEX color codes."
        ),
    },
    name="Volcano Visualizer",
    description=(
        "Generates a volcano plot of enrichment scores vs. adjusted p-values."
        " Significant taxa are highlighted when identifiers are provided."
    ),
)

# ---------------------------------------------------------------------------
# Register zscatter visualizer
# ---------------------------------------------------------------------------

plugin.visualizers.register_function(
    function=zscatter,
    inputs={
        "zscores": FeatureTable[Zscore],
        "pairs": PSEAPairs,
        "psea_tables": Collection[FeatureData[PSEAScores]],
        "splines": Collection[FeatureData[Spline]],
    },
    input_descriptions={
        "zscores": "Matrix of Z scores.",
        "pairs": (
            "Tab-delimited file listing pairs of sample names (one pair per"
            " row, header required)."
        ),
        "psea_tables": (
            "Per-pair PSEA result tables used to highlight leading-edge"
            " peptides for significant taxa."
        ),
        "splines" : "Collection of splines per pair."
    },
    parameters={
        "p_val_access": Str,
        "le_peps_access": Str,
        "taxa_access": Str,
        "highlight_threshold": Float,
        "colors_file": Metadata,
    },
    parameter_descriptions={
        "p_val_access": (
            "Column name in psea_tables compared to 'highlight_threshold'"
            " for highlighting."
        ),
        "le_peps_access": "Column name with leading-edge peptides for"
        " tooltip.",
        "taxa_access": "Column name with taxa names for highlighting.",
        "highlight_threshold": (
            "Maximum p-value for a taxon to be highlighted."
        ),
        "colors_file": (
            "Optional Metadata mapping species names (IDs) to HEX color codes."
        ),
    },
    name="Z Score Scatter Visualization",
    description=(
        "Creates a heatmap scatter plot of Z scores for each sample pair."
        " An optional spline fit and significant leading-edge peptides can"
        " be overlaid."
    ),
)

# ---------------------------------------------------------------------------
# Register aeplots visualizer
# ---------------------------------------------------------------------------

plugin.visualizers.register_function(
    function=aeplots,
    inputs={
        "pos_ae_counts": PSEAAECounts,
        "neg_ae_counts": PSEAAECounts,
    },
    input_descriptions={
        "pos_ae_counts": (
            "Species-level counts of significant positive-NES antibody"
            " events, as produced by count_antibody_events."
        ),
        "neg_ae_counts": (
            "Species-level counts of significant negative-NES antibody"
            " events, as produced by count_antibody_events."
        ),
    },
    parameters={
        "xy_access": List[Str],
        "xy_labels": List[Str],
        "colors_file": Metadata,
    },
    parameter_descriptions={
        "xy_access": (
            "Column names for x (events) and y (species) values,"
            " respectively."
        ),
        "xy_labels": "Axis labels for x and y, respectively.",
        "colors_file": (
            "Optional Metadata mapping species names (IDs) to HEX color codes."
        ),
    },
    name="Antibody Events Plots Visualizer",
    description=(
        "Generates bar plots of species antibody-event counts for positive"
        " and negative NES results."
    ),
)

# ---------------------------------------------------------------------------
# Register create_epitope_map as a method
# ---------------------------------------------------------------------------

plugin.methods.register_function(
    function=create_epitope_map,
    inputs={'epitope': FeatureData[Epitope]},
    parameters={
        'collapse': Str % Choices(['Bacterial', 'Viral', 'Both'])
    },
    outputs=[
        ('epitope_map', FeatureData[MappedEpitope])
    ],
    input_descriptions={'epitope': 'FeatureTable containing at least '
                        'CodeName, SpeciesID, ClusterID, EpitopeWindow, '
                        'Species, and Subtype columns'},
    parameter_descriptions={},
    output_descriptions={'epitope_map': 'FeatureTable containing columns '
                         'described in action descriptions.'},
    name='create epitope map',
    description='Creates the fully defined epitope name '
                'species_clusterID_EpitopeWindow mapped to peptide code names '
                'and species/subtypes the epitope is associated with.',
)

# ---------------------------------------------------------------------------
# Register epitope_zscore as a method
# ---------------------------------------------------------------------------

plugin.methods.register_function(
    function=epitope_zscore,
    inputs={
        'zscores': FeatureTable[Zscore],
        'epitope_map': FeatureData[MappedEpitope],
    },
    parameters={},
    outputs=[
        ('epitope_zscore', FeatureTable[Zscore % Properties("mapped")]),
    ],
    input_descriptions={
        'zscores': 'FeatureTable containing the code names of peptides and '
                   'their per sample z scores',
        'epitope_map': 'FeatureTable containing epitopes and their associated '
                       'peptides and subtypes',
    },
    parameter_descriptions={},
    output_descriptions={
        'epitope_zscore': 'FeatureTable containing the epitopes and their per '
                          'sample z scores.',
    },
    name='zscore',
    description='Creates a map of epitopes to their max z-score within each '
                'sample. The maxes are taken by finding the per sample maxes '
                'among z scores of peptides associated with a given epitope.',
)

# ---------------------------------------------------------------------------
# Register taxa_to_epitope as a method
# ---------------------------------------------------------------------------

plugin.methods.register_function(
    function=taxa_to_epitope,
    inputs={
        'epitope': FeatureData[MappedEpitope],
    },
    parameters={},
    outputs=[
        ('epitope_gmt', GMT % Properties("mapped")),
    ],
    input_descriptions={
        'epitope': 'Feature table containing at least SpeciesID, ClusterID, '
                   'and EpitopeWindow columns',
    },
    parameter_descriptions={},
    output_descriptions={
        'epitope_gmt': 'GMT mapping SpeciesIDs to associated epitopes.',
    },
    name='taxa to epitope',
    description='Creates a GMT file mapping SpeciesIDs to their associated '
                'epitopes.',
)

# ---------------------------------------------------------------------------
# Register enriched_subtypes as a method
# ---------------------------------------------------------------------------

plugin.methods.register_function(
    function=enriched_subtypes,
    inputs={
        'scores': Collection[FeatureData[PSEAScores]],
        'subtypes': FeatureData[MappedEpitope],
    },
    parameters={
        'p_value': Float % Range(0, None),
        'enrichment_score': Float % Range(0, None),
        'include_negative_enrichment': Bool,
        'peptide_library': Str,
    },
    outputs=[
        ('enriched', Collection[FeatureData[Enriched]]),
    ],
    input_descriptions={
        'scores': 'PSEAScores of peptides/epitopes.',
        'subtypes': 'subtypes',
    },
    parameter_descriptions={},
    output_descriptions={
        'enriched': 'Enriched subtypes.',
    },
    name='enriched subtypes',
    description='Counts which subtypes have been enriched.',
)
