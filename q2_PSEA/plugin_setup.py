#! /usr/bin/env python
from q2_types.feature_data import FeatureData
from q2_types.feature_table import FeatureTable
from q2_pepsirf.format_types import (
    PSEAScores,
    Enriched,
    Epitope,
    MappedEpitope,
    GMT,
    Zscore,
    PSEAAECounts,
    PSEAPairs,
    Spline
)

import q2_PSEA
import q2_PSEA.actions.splines as splines

from q2_PSEA.actions.psea import (
    _compute_pair_fit_and_residuals,
    count_antibody_events,
    _create_fgsea_table_for_pair,
    _filter_scores_to_pairs,
    _process_scores,
    _run_iterative_process_single_pair,
    _split_scores,
    _map_residuals_and_zscores,
    make_psea_table,
)
from q2_PSEA.actions.visualizers import volcano, zscatter, aeplots
from q2_PSEA.actions.epitope import (
    create_epitope_map,
    taxa_to_epitope,
    count_enriched,
)
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
    TypeMap,
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
# Register _filter_scores_to_pairs as a method
# ---------------------------------------------------------------------------

SCORES_IN, SCORES_OUT = TypeMap({
    FeatureTable[Zscore % Properties("mapped")]:
        FeatureTable[Zscore % Properties("mapped", "processed")],
    FeatureTable[Zscore]: FeatureTable[Zscore % Properties("processed")]
})

plugin.methods.register_function(
    function=_filter_scores_to_pairs,
    inputs={
        "scores": SCORES_IN,
        "pairs": PSEAPairs,
    },
    parameters={},
    outputs=[("filtered_scores", SCORES_OUT)],
    input_descriptions={
        "scores": "Z-score matrix (FeatureTable[Zscore]).",
        "pairs": "Pairs to filter on. Keep only scores related to these pairs."
    },
    parameter_descriptions={},
    output_descriptions={
        "filtered_scores": (
            "Filtered Z-score matrix."
        ),
    },
    name="Process Scores",
    description=(
        "Z-scores in the given matrix filtered to only those related to given"
        " pairs."
    ),
)


# ---------------------------------------------------------------------------
# Register _process_scores as a method
# ---------------------------------------------------------------------------


plugin.methods.register_function(
    function=_process_scores,
    inputs={
        "scores": SCORES_IN,
    },
    parameters={},
    outputs=[("processed_zscores", SCORES_OUT)],
    input_descriptions={
        "scores": "Z-score matrix (FeatureTable[Zscore]).",
    },
    parameter_descriptions={},
    output_descriptions={
        "processed_zscores": (
            "Log-scaled Z-score matrix."
        ),
    },
    name="Process Scores",
    description=(
        "Log scales z-scores in the given matrix."
    ),
)

# ---------------------------------------------------------------------------
# Register _compute_pair_fit_and_residuals as a method
# ---------------------------------------------------------------------------

plugin.methods.register_function(
    function=_compute_pair_fit_and_residuals,
    inputs={
        "processed_zscores": FeatureTable[Zscore % Properties("processed")],
    },
    parameters={
        "pair": Str,
        "spline_type": Str % Choices(splines.SPLINE_TYPES),
        "fit_threshold": Float,
        "linear_through_origin": Bool,
        "degree": Int,
        "dof": Int,
    },
    outputs=[("spline_fit", FeatureData[Spline])],
    input_descriptions={
        "processed_zscores": (
            "Log-scaled Z-score matrix (FeatureTable[Zscore])."
        ),
    },
    parameter_descriptions={
        "pair": "Name of the pair.",
        "spline_type": "Spline method used to fit the Z-score scatter.",
        "fit_threshold": (
            "Optional threshold used only for linear spline fitting; only"
            " points where x and y are both greater than this threshold are"
            " used to fit the line."
        ),
        "linear_through_origin": (
            "If True and spline_type is linear, force the regression line"
            " through (0, 0)."
        ),
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
        "psea_table": FeatureData[PSEAScores],
    },
    parameters={
        "p_value": Float,
        "enrichment_score": Float,
        "taxa_access": Str,
    },
    outputs=[
        ("pos_ae_counts", PSEAAECounts),
        ("neg_ae_counts", PSEAAECounts),
    ],
    input_descriptions={
        "psea_table": (
            "PSEA result table produced by _create_fgsea_table_for_pair."
        ),
    },
    parameter_descriptions={
        "p_value": (
            "Adjusted p-value threshold; taxa below this value are counted"
            " as significant events."
        ),
        "enrichment_score": (
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
# Register _create_fgsea_table_for_pair as a method
# ---------------------------------------------------------------------------


plugin.methods.register_function(
    function=_create_fgsea_table_for_pair,
    inputs={
        "processed_zscores": FeatureTable[Zscore % Properties("processed")],
        "peptide_sets": GMT,
        "precomputed_fit": FeatureData[Spline],
    },
    parameters={
        "threshold": Float,
        "permutation_num": Int,
        "min_size": Int,
        "max_size": Int,
        "seed": Int,
        "species_taxa": Metadata,
        "residual_abs_thresh": Float,
        "residual_min_peptides": Int,
    },
    parameter_descriptions={
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
        "residual_abs_thresh": (
            "Optional absolute residual threshold for peptide-set filtering"
            " before GSEA."
        ),
        "residual_min_peptides": (
            "Minimum number of peptides required per species with absolute"
            " residual greater than residual_abs_thresh to keep that species."
        ),
    },
    input_descriptions={
        "processed_zscores": (
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
# Register _run_iterative_process_single_pair as a method
# ---------------------------------------------------------------------------

plugin.methods.register_function(
    function=_run_iterative_process_single_pair,
    inputs={
        "processed_zscores": FeatureTable[Zscore % Properties("processed")],
        "peptide_sets": GMT,
        "precomputed_fit": FeatureData[Spline],
        "mapped_peptide_sets": GMT % Properties("mapped"),
    },
    parameters={
        "threshold": Float,
        "permutation_num": Int,
        "min_size": Int,
        "max_size": Int,
        "seed": Int,
        "p_value": Float,
        "enrichment_score": Float,
        "include_negative_enrichment": Bool,
        "species_taxa": Metadata,
        "debug_per_iteration_table_path": Str,
        "pair": Str,
        "residual_abs_thresh": Float,
        "residual_min_peptides": Int,
    },
    parameter_descriptions={
        "threshold": "Minimum Z-score for GSEA inclusion.",
        "permutation_num": "Number of GSEA permutations.",
        "min_size": "Minimum peptide-set size.",
        "max_size": "Maximum peptide-set size.",
        "seed": "Random seed for GSEA permutations.",
        "p_value": (
            "Adjusted p-value threshold for calling a species significant."
        ),
        "enrichment_score": (
            "Absolute NES threshold for calling a species significant."
        ),
        "include_negative_enrichment": (
            "Whether or not to include negative enrichment."
        ),
        "species_taxa": (
            "Optional Metadata mapping species names (IDs) to taxonomy IDs."
        ),
        "debug_per_iteration_table_path": (
            "Path to write per pair and per iteration psea tables to as .tsvs."
            " Only to be used when debugging and meaningless if not doing"
            " iterative analysis."
        ),
        "pair": (
            "The name of the pair we are running iterative analysis on. Only"
            " needed when writing debug tables."
        ),
        "residual_abs_thresh": (
            "Optional absolute residual threshold for peptide-set filtering"
            " before each GSEA run."
        ),
        "residual_min_peptides": (
            "Minimum number of peptides required per species with absolute"
            " residual greater than residual_abs_thresh to keep that species."
        ),
    },
    input_descriptions={
        "processed_zscores": "Log-scaled Z-score matrix.",
        "peptide_sets": "Current (possibly filtered) GMT for this pair.",
        "precomputed_fit": (
            "Optional precomputed maxZ/deltaZ from a prior call. When"
            " provided, spline fitting is skipped."
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
        " _create_fgsea_table_for_pair method, identifies the top significant"
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
        "peptide_metadata": FeatureData[Epitope],
        "epitope_map": FeatureData[MappedEpitope],
        "peptide_sets_map": GMT % Properties("mapped")
    },
    parameters={
        "threshold": Float,
        "collapse": Str % Choices(["Bacterial", "Viral", "Both"]),
        "p_value": Float,
        "enrichment_score": Float,
        "include_negative_enrichment": Bool,
        "min_size": Int,
        "max_size": Int,
        "permutation_num": Int,
        "spline_type": Str % Choices(splines.SPLINE_TYPES),
        "fit_threshold": Float,
        "linear_through_origin": Bool,
        "degree": Int,
        "dof": Int,
        "iterative_analysis": Bool,
        "seed": Int,
        "species_taxa": Metadata,
        "species_colors": Metadata,
        "use_epitope_mapping": Bool,
        "residual_threshold": Float,
        "residual_abs_thresh": Float,
        "residual_min_peptides": Int,
        "debug_per_iteration_table_path": Str,
    },
    parameter_descriptions={
        "threshold": (
            "Minimum Z-score a peptide must have to be included in GSEA."
        ),
        "collapse": (
            "Category to collapse to epitope level. Only used when the"
            " epitope input is provided."
        ),
        "p_value": (
            "Adjusted p-value threshold for significance in volcano and"
            " scatter plots."
        ),
        "enrichment_score": "NES threshold for significance.",
        "include_negative_enrichment": (
            "Whether or not to include negative enrichment."
        ),
        "min_size": "Minimum peptide-set size for GSEA.",
        "max_size": "Maximum peptide-set size for GSEA.",
        "permutation_num": (
            "Number of GSEA permutations. Minimum nominal p-value is"
            " ~1/perm."
        ),
        "spline_type": "Spline method used to fit the Z-score scatter.",
        "fit_threshold": (
            "Optional threshold used only for linear spline fitting; only"
            " points where x and y are both greater than this threshold are"
            " used to fit the line."
        ),
        "linear_through_origin": (
            "If True and spline_type is linear, force the regression line"
            " through (0, 0)."
        ),
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
        "use_epitope_mapping": (
            "If true, the analysis will be run with data collapsed to epitope"
            " level. This requires you to either pass 'peptide_metadata' so"
            " the pipeline can do the collapsing, or both of epitope_map,"
            " and peptide_sets_map"
        ),
        "residual_threshold": (
            "The threshold above which a peptide residual must be in order to"
            " be counted in count_enriched."
        ),
        "residual_abs_thresh": (
            "Optional absolute residual threshold for peptide-set filtering"
            " before GSEA."
        ),
        "residual_min_peptides": (
            "Minimum number of peptides required per species with absolute"
            " residual greater than residual_abs_thresh to keep that species."
        ),
        "debug_per_iteration_table_path": (
            "Path to write per pair and per iteration psea tables to as .tsvs."
            " Only to be used when debugging and meaningless if not doing"
            " iterative analysis."
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
        "peptide_metadata": (
            "Peptide level metadata. Must be passed when doing epitope mapped"
            " analysis"
        ),
        "epitope_map": (
            "Optional already collapsed epitope table. When provided, this"
            " table is used in GSEA. Maps epitopes to peptides and species."
            "NOTE: Must be passed with peptide_sets_map."
        ),
        "peptide_sets_map": (
            "Optional already collapsed epitope peptide sets. When provided,"
            " these peptides are used in GSEA."
            "NOTE: Must be passed with epitope_map."
        )
    },
    outputs=[
        ("scatter_plots", Collection[Visualization]),
        ("volcano_plots", Collection[Visualization]),
        ("ae_plots", Visualization),
        ("psea_tables", Collection[FeatureData[PSEAScores]]),
        ("enrichment_tables", Collection[FeatureData[Enriched]])
    ],
    output_descriptions={
        "scatter_plots": (
            "per-pair z-score scatter plots with spline fit and highlighted"
            " leading-edge peptides for significant taxa."
        ),
        "volcano_plots": (
            "per-pair volcano plots of normalized enrichment scores vs."
            " adjusted p-values."
        ),
        "ae_plots": "Antibody-event summary bar plots.",
        "psea_tables": (
            "Per-pair PSEA result tables containing enrichment scores,"
            " p-values, and leading-edge peptides."
        ),
        "enrichment_tables": (
            "Tables showing counts of enriched epitopes per subspecies, and"
            " enriched subspecies per epitope."
        )
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
        "psea_table": FeatureData[PSEAScores],
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
        "psea_table": (
            "PSEA result table. When provided, x, y, and taxa are read from"
            " this artifact using xy_access and taxa_access."
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
        "zscores": FeatureTable[Zscore % Properties("processed")],
        "psea_table": FeatureData[PSEAScores],
        "spline": FeatureData[Spline],
    },
    input_descriptions={
        "zscores": "Matrix of Z scores.",
        "psea_table": (
            "PSEA result table used to highlight leading-edge peptides for"
            " significant taxa."
        ),
        "spline": "Spline fit for the plot."
    },
    parameters={
        "pair": Str,
        "p_val_access": Str,
        "le_peps_access": Str,
        "taxa_access": Str,
        "highlight_threshold": Float,
        "colors_file": Metadata,
    },
    parameter_descriptions={
        "pair": "String of format sample_a~sample_b.",
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
        "Creates a heatmap scatter plot of Z scores for the given sample pair."
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
        "pos_ae_counts": List[PSEAAECounts],
        "neg_ae_counts": List[PSEAAECounts],
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
    output_descriptions={
        'epitope_map': 'FeatureTable mapping epitopes to peptides and species',
    },
    name='create epitope map',
    description='Creates the fully defined epitope name '
                'species_clusterID_EpitopeWindow mapped to peptide code names '
                'and species/subtypes the epitope is associated with.',
)

# ---------------------------------------------------------------------------
# Register taxa_to_epitope as a method
# ---------------------------------------------------------------------------

plugin.methods.register_function(
    function=taxa_to_epitope,
    inputs={
        'peptide_metadata': FeatureData[Epitope],
        'peptide_sets': GMT,
    },
    parameters={
        'collapse': Str % Choices(['Bacterial', 'Viral', 'Both']),
    },
    outputs=[
        ('epitope_gmt', GMT % Properties("mapped")),
    ],
    input_descriptions={
        'peptide_metadata': 'Feature table containing at least SpeciesID, '
                            'ClusterID, and EpitopeWindow columns keyed on '
                            'SpeciesID column.',
        'peptide_sets': 'GMT file mapping species identifiers to the peptides '
                        'linked to them. Collapsed to epitope level if '
                        'epitope is provided.'
    },
    parameter_descriptions={
        'collapse': (
            'Category to collapse to epitope level. Only used when the'
            ' epitope input is provided.'
        ),
    },
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
    function=count_enriched,
    inputs={
        'psea_table': FeatureData[PSEAScores],
        'residuals': FeatureData[Spline],
        'peptide_metadata': FeatureData[Epitope],
        'epitope_map': FeatureData[MappedEpitope],
    },
    parameters={
        'p_value': Float % Range(0, None),
        'residual_threshold': Float % Range(0, None),
        'include_negative_enrichment': Bool,
    },
    outputs=[
        ('enriched', FeatureData[Enriched]),
    ],
    input_descriptions={
        'psea_table':
            'PSEAScores of peptides/epitopes. Collection maps sample1~sample2'
            ' to Artifact',
        'epitope_map': 'subtypes',
    },
    parameter_descriptions={},
    output_descriptions={
        'enriched': 'Enriched subtypes.',
    },
    name='enriched subtypes',
    description='Counts which subtypes have been enriched.',
)

plugin.methods.register_function(
    function=_split_scores,
    inputs={
        'scores': SCORES_IN,
        'pairs': PSEAPairs,
    },
    parameters={},
    outputs=[
        ('split_scores', Collection[SCORES_OUT])
    ],
    name='split scores',
    description='Splits scores into per-pair artifacts.'
)


plugin.methods.register_function(
    function=_map_residuals_and_zscores,
    inputs={
        'spline': FeatureData[Spline],
        'zscores': FeatureTable[Zscore % Properties('processed')],
        'epitope_map': FeatureData[MappedEpitope]
    },
    parameters={},
    outputs=[
        ('mapped_spline', FeatureData[Spline % Properties('mapped')]),
        ('mapped_zscores',
         FeatureTable[Zscore % Properties('processed', 'mapped')])
    ],
    name="map residuals and zscores",
    description="maps residuals and zscores"
)
