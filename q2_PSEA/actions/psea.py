import numpy as np
import os
import pandas as pd
import qiime2
import random
import rpy2.robjects as ro
import q2_PSEA.actions.splines as splines
import q2_PSEA.utils as utils
import warnings
import tempfile

from math import log, pow
from qiime2.plugin import CaptureHolder, IContext
from rachis.core.exceptions import RachisWarning
from rpy2.robjects import pandas2ri, numpy2ri
from q2_PSEA.actions.r_functions import INTERNAL

# Need a random signed 32 bit int for R
MIN_32_BIT_INT = -2 ** 31
MAX_32_BIT_INT = 2**31 - 1


def make_psea_table(
    ctx: IContext,
    scores: qiime2.Artifact,
    pairs: qiime2.Artifact,
    peptide_sets: qiime2.Artifact,
    threshold: float,
    peptide_metadata: qiime2.Artifact = None,
    epitope_map: qiime2.Artifact = None,
    peptide_sets_map: qiime2.Artifact = None,
    collapse: str = "Viral",
    p_value: float = 0.05,
    enrichment_score: float = 1,
    include_negative_enrichment: bool = True,
    min_size: int = 15,
    max_size: int = 2000,
    permutation_num: int = 10000,
    spline_type: str = "r-smooth",
    fit_threshold: float = None,
    linear_through_origin: bool = False,
    degree: int = 3,
    dof: int = None,
    iterative_analysis: bool = True,
    seed: CaptureHolder[int] = None,
    species_taxa: qiime2.Metadata = None,
    species_colors: qiime2.Metadata = None,
    use_epitope_mapping: bool = False,
    residual_threshold: float = .5,
    residual_abs_thresh: float = None,
    residual_min_peptides: int = 1,
    debug_per_iteration_table_path: str = None,
) -> tuple[
    qiime2.Visualization,
    qiime2.Visualization,
    qiime2.Visualization,
    dict[str, qiime2.Artifact],
    dict[str, qiime2.Artifact],
]:
    seed = CaptureHolder.get_or_set(
        seed, lambda: random.randint(MIN_32_BIT_INT, MAX_32_BIT_INT)
    )

    # ------------------------------------------------------------------
    # Validate debug value
    # ------------------------------------------------------------------
    if debug_per_iteration_table_path is not None:
        warnings.warn(
            f"You have set a value of {debug_per_iteration_table_path} for"
            " 'debug_iteration_table_path.' Please only set this value if you"
            " intend to use the per iteration .TSVs for debugging. Do not set"
            " for regular analysis.", RachisWarning
        )

        if not iterative_analysis:
            raise ValueError(
                "Please only provide a path for debug_per_iteration_table_path"
                " if doing iterative analysis."
            )

        if os.path.exists(debug_per_iteration_table_path):
            raise ValueError(
                f"Path {debug_per_iteration_table_path} already exists. Please"
                " provide a path for debug .tsvs that does not already exist."
            )

        os.mkdir(debug_per_iteration_table_path)

    # ------------------------------------------------------------------
    # Determine what kind of analysis was asked for
    # ------------------------------------------------------------------
    map_provided = all(
        param is not None for param in [epitope_map, peptide_sets_map]
    )

    if not use_epitope_mapping and map_provided:
        raise ValueError("You provided mapped artifacts but indicated you do"
                         " not want mapping.")

    if use_epitope_mapping and any(
                param is not None for param in [
                    epitope_map, peptide_sets_map
                ]
            ) and not map_provided:
        raise ValueError(
            "Please pass either both of 'epitope_map', 'peptide_sets_map' or"
            " neither of them when running mapped analysis. If you provide"
            " neither, this pipeline will do the mapping."
        )

    if use_epitope_mapping and peptide_metadata is None:
        raise ValueError(
            "Must provide peptide metadata if doing a mapped analysis."
        )

    _filter_scores_to_pairs = ctx.get_action("psea", "_filter_scores_to_pairs")
    _process_scores = ctx.get_action("psea", "_process_scores")
    _split_scores = ctx.get_action("psea", "_split_scores")
    _compute_pair_fit_and_residuals = ctx.get_action(
        "psea", "_compute_pair_fit_and_residuals"
    )
    _map_residuals_and_zscores = \
        ctx.get_action("psea", "_map_residuals_and_zscores")
    _run_iterative_process_single_pair = ctx.get_action(
        "psea", "_run_iterative_process_single_pair"
    )
    _create_fgsea_table_for_pair = ctx.get_action(
        "psea", "_create_fgsea_table_for_pair"
    )
    count_enriched = ctx.get_action("psea", "count_enriched")

    count_antibody_events = ctx.get_action("psea", "count_antibody_events")

    volcano = ctx.get_action("psea", "volcano")
    zscatter = ctx.get_action("psea", "zscatter")
    aeplots = ctx.get_action("psea", "aeplots")

    taxa_access = "species_name" if species_taxa is not None else "ID"

    # ------------------------------------------------------------------
    # Filter scores
    # ------------------------------------------------------------------
    filtered_zscores, = _filter_scores_to_pairs(scores, pairs)

    # ------------------------------------------------------------------
    # Parse pairs list
    # ------------------------------------------------------------------
    pairs_list = pairs.view(list)

    # ------------------------------------------------------------------
    # Handle epitope collapsing if needed
    # ------------------------------------------------------------------
    if use_epitope_mapping and not map_provided:
        create_epitope_map = ctx.get_action("psea", "create_epitope_map")
        create_epitope_gmt = ctx.get_action("psea", "taxa_to_epitope")

        epitope_map, = create_epitope_map(peptide_metadata, collapse)
        peptide_sets_map, = create_epitope_gmt(
            peptide_metadata, peptide_sets, collapse=collapse
        )

    # ------------------------------------------------------------------
    # Process (log-scale) scores
    # ------------------------------------------------------------------
    processed_zscores, = _process_scores(filtered_zscores)

    # ------------------------------------------------------------------
    # Split scores out by pair
    # ------------------------------------------------------------------
    # TODO: may be need for optimization here. This becomes a blocking
    # operation down below when we need to index into this inside of the loop.
    #
    # It does reduce memory usage in later processes, but it also spikes it
    # here, and it takes some time... maybe too long for a large number
    # of pairs
    split_processed_zscores, = _split_scores(processed_zscores, pairs)

    pair_splines = {}
    pair_pep_sets_dict = {}
    psea_tables = {}
    enrichment_tables = {}
    scatter_plots = {}
    volcano_plots = {}
    pos_ae_counts = {}
    neg_ae_counts = {}

    # NOTE: We can parallelize pairs. We cannot parallelize iterations
    for pair in pairs_list:
        # Compute spline fit once per pair; reuse it for both the scatter
        # plot data and as the precomputed_fit input to create_fgsea_table.
        pair_splines[pair], = _compute_pair_fit_and_residuals(
            processed_zscores=split_processed_zscores[pair],
            pair=pair,
            spline_type=spline_type,
            fit_threshold=fit_threshold,
            linear_through_origin=linear_through_origin,
            degree=degree,
            dof=dof,
        )

        # Collapse residuals and zscores if using mapping
        if use_epitope_mapping:
            pair_splines[pair], split_processed_zscores[pair] = \
                _map_residuals_and_zscores(
                    pair_splines[pair],
                    split_processed_zscores[pair],
                    epitope_map
                )

        # ------------------------------------------------------------------
        # Determine per-pair peptide sets (iterative or flat)
        # ------------------------------------------------------------------
        if iterative_analysis:
            pair_pep_sets_dict[pair], = _run_iterative_process_single_pair(
                processed_zscores=split_processed_zscores[pair],
                peptide_sets=peptide_sets,
                precomputed_fit=pair_splines[pair],
                mapped_peptide_sets=peptide_sets_map,
                threshold=threshold,
                permutation_num=permutation_num,
                min_size=min_size,
                max_size=max_size,
                p_value=p_value,
                seed=seed,
                enrichment_score=enrichment_score,
                include_negative_enrichment=include_negative_enrichment,
                species_taxa=species_taxa,
                debug_per_iteration_table_path=debug_per_iteration_table_path,
                pair=pair,
                residual_abs_thresh=residual_abs_thresh,
                residual_min_peptides=residual_min_peptides,
            )
        else:
            pair_pep_sets_dict[pair] = \
                peptide_sets_map if use_epitope_mapping else peptide_sets

        # ------------------------------------------------------------------
        # Final per-pair PSEA analysis
        # ------------------------------------------------------------------
        psea_tables[pair], = _create_fgsea_table_for_pair(
            processed_zscores=split_processed_zscores[pair],
            peptide_sets=pair_pep_sets_dict[pair],
            threshold=threshold,
            permutation_num=permutation_num,
            min_size=min_size,
            max_size=max_size,
            seed=seed,
            species_taxa=species_taxa,
            precomputed_fit=pair_splines[pair],
            residual_abs_thresh=residual_abs_thresh,
            residual_min_peptides=residual_min_peptides,
        )

        enrichment_tables[pair], = count_enriched(
            psea_table=psea_tables[pair],
            residuals=pair_splines[pair],
            peptide_metadata=peptide_metadata,
            epitope_map=epitope_map,
            p_value=p_value,
            residual_threshold=residual_threshold,
            include_negative_enrichment=include_negative_enrichment
        )

        scatter_plots[pair], = zscatter(
            zscores=split_processed_zscores[pair],
            pair=pair,
            spline=pair_splines[pair],
            p_val_access="p.adjust",
            le_peps_access="core_enrichment",
            taxa_access=taxa_access,
            psea_table=psea_tables[pair],
            highlight_threshold=p_value,
            colors_file=species_colors,
        )

        volcano_plots[pair], = volcano(
            psea_table=psea_tables[pair],
            xy_access=["NES", "p.adjust"],
            taxa_access=taxa_access,
            x_threshold=enrichment_score,
            y_threshold=p_value,
            xy_labels=["Enrichment score", "Adjusted p-values"],
            colors_file=species_colors,
        )

        pos_ae_counts[pair], neg_ae_counts[pair] = count_antibody_events(
            psea_table=psea_tables[pair],
            p_value=p_value,
            enrichment_score=enrichment_score,
            taxa_access=taxa_access,
        )

    ae_plot, = aeplots(
        pos_ae_counts=pos_ae_counts,
        neg_ae_counts=neg_ae_counts,
        xy_access=["Events", "Species"],
        xy_labels=["Number of AEs in cohort", "Species"],
        colors_file=species_colors,
    )

    # TODO: Create a wrapper that unzips these in manner discussed previously
    # i.e. output unzipped raw files along with manifests keeping track of them
    # Also, include imports in the wrapper so
    # 1. import
    # 2. run
    # 3. export
    return (scatter_plots, volcano_plots, ae_plot, psea_tables,
            enrichment_tables)


def _run_iterative_process_single_pair(
    processed_zscores: pd.DataFrame,
    peptide_sets: pd.DataFrame,
    threshold: float,
    permutation_num: int,
    min_size: int,
    max_size: int,
    p_value: float,
    enrichment_score: float,
    precomputed_fit: pd.DataFrame = None,
    mapped_peptide_sets: pd.DataFrame = None,
    seed: CaptureHolder[int] = None,
    include_negative_enrichment: bool = True,
    species_taxa: qiime2.Metadata = None,
    debug_per_iteration_table_path: str = None,
    pair: str = None,
    residual_abs_thresh: float = None,
    residual_min_peptides: int = 1,
) -> pd.DataFrame:
    """QIIME 2 pipeline: run one iteration of iterative peptide analysis for a
    single sample pair.

    Calls the registered ``_create_fgsea_table_for_pair`` method, finds the
    most significant species not yet in *tested_species*, removes its leading-
    edge peptides from all other species in the GMT, and returns the updated
    peptide sets alongside the PSEA table for this iteration.

    Returns
    -------
    updated_peptide_sets : GMT
    """
    seed = CaptureHolder.get_or_set(
        seed, lambda: random.randint(MIN_32_BIT_INT, MAX_32_BIT_INT)
    )
    updated_peptide_sets = (
        mapped_peptide_sets if mapped_peptide_sets is not None
        else peptide_sets
    )

    tested_species = set()
    iteration = 1
    sig_found = True

    if debug_per_iteration_table_path:
        if not pair:
            raise ValueError("Pair name must be provided when debugging.")

        os.mkdir(os.path.join(debug_per_iteration_table_path, pair))

    while (sig_found):
        # Called as a raw Python function not a QIIME 2 Method
        current_psea_table = _create_fgsea_table_for_pair(
            processed_zscores=processed_zscores,
            peptide_sets=updated_peptide_sets,
            threshold=threshold,
            permutation_num=permutation_num,
            min_size=min_size,
            max_size=max_size,
            seed=seed,
            species_taxa=species_taxa,
            precomputed_fit=precomputed_fit,
            residual_abs_thresh=residual_abs_thresh,
            residual_min_peptides=residual_min_peptides,
        )

        if debug_per_iteration_table_path:
            current_psea_table.to_csv(
                os.path.join(
                    debug_per_iteration_table_path, pair, f'{iteration}.tsv'
                ), sep='\t', index=False
            )

        updated_peptide_sets, tested_species, sig_found = \
            utils.filter_peptide_sets(
                current_psea_table,
                updated_peptide_sets,
                tested_species,
                p_value,
                enrichment_score,
                include_negative_enrichment,
            )

        iteration += 1

    return updated_peptide_sets


def _create_fgsea_table_for_pair(
    processed_zscores: pd.DataFrame,
    peptide_sets: pd.DataFrame,
    precomputed_fit: pd.DataFrame,
    threshold: float,
    permutation_num: int,
    min_size: int,
    max_size: int,
    seed: CaptureHolder[int] = None,
    species_taxa: qiime2.Metadata = None,
    residual_abs_thresh: float = None,
    residual_min_peptides: int = 1,
) -> pd.DataFrame:
    """QIIME 2 method: compute the fgsea PSEA table for a single sample pair.

    Parameters
    ----------
    processed_zscores : pd.DataFrame
        Log-scaled Z-score matrix (from FeatureTable[Zscore]).
    peptide_sets : pd.DataFrame
        GMT peptide-set table with columns 'term' and 'gene' (from GMT).
    species_taxa : PSEASpeciesTaxaDirFmt, optional
        Directory format containing species-taxa.tsv; passed as a file path
        to the underlying R function.

    Returns
    -------
    pd.DataFrame
        PSEA result table for this pair (stored as FeatureData[PSEAScores]).
    """
    seed = CaptureHolder.get_or_set(
        seed, lambda: random.randint(MIN_32_BIT_INT, MAX_32_BIT_INT)
    )
    processed_zscores = processed_zscores.transpose()
    maxZ_all = precomputed_fit["maxZ"].dropna()
    deltaZ_all = precomputed_fit["deltaZ"].dropna()

    filtered_zscores, peptide_sets_for_analysis = \
        utils.remove_peptides(processed_zscores, peptide_sets)

    idx = filtered_zscores.index
    maxZ = maxZ_all.reindex(idx)
    deltaZ = deltaZ_all.reindex(idx)
    peptide_sets_for_analysis = utils.filter_peptide_sets_by_residual(
        peptide_sets_for_analysis,
        deltaZ,
        residual_abs_thresh=residual_abs_thresh,
        residual_min_peptides=residual_min_peptides,
    )

    # This ought to ensure this file is accessible where this code is actually
    # being run if run cross node on HPC for instance
    with tempfile.TemporaryDirectory() as tmpdir:
        if species_taxa is not None:
            taxa_df = species_taxa.to_dataframe().reset_index()
            species_taxa_file = os.path.join(tmpdir, "species_taxa.tsv")
            taxa_df.to_csv(
                species_taxa_file, sep="\t", header=False, index=False
            )
        else:
            species_taxa_file = ""

        with (ro.default_converter + pandas2ri.converter).context():
            table = INTERNAL.psea(
                maxZ,
                deltaZ,
                peptide_sets_for_analysis,
                species_taxa_file,
                threshold,
                permutation_num,
                min_size,
                max_size,
                seed,
            )
            table = ro.conversion.get_conversion().rpy2py(table)

    return table


def count_antibody_events(
    psea_table: pd.DataFrame,
    p_value: float,
    enrichment_score: float,
    taxa_access: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """QIIME 2 method: count positive- and negative-NES antibody events.

    A taxon is counted as one event for a given pair when its adjusted p-value
    is below *p_value* and the absolute value of its NES exceeds
    *enrichment_score*. Positive and negative NES events are tallied
    separately.

    Returns
    -------
    pos_ae_counts : pd.DataFrame
        Two-column DataFrame (Species, Events) sorted by event count
        descending. Contains taxa with significant positive NES.
    neg_ae_counts : pd.DataFrame
        Same structure for taxa with significant negative NES.
    """
    pos_count = {}
    neg_count = {}

    for _, row in psea_table.iterrows():
        taxa = row[taxa_access]
        if (
            row["p.adjust"] < p_value
            and abs(row["NES"]) > enrichment_score
        ):
            if row["NES"] > 0:
                pos_count[taxa] = pos_count.get(taxa, 0) + 1
            elif row["NES"] < 0:
                neg_count[taxa] = neg_count.get(taxa, 0) + 1

    pos_count = dict(
        sorted(pos_count.items(), key=lambda item: item[1], reverse=True)
    )
    neg_count = dict(
        sorted(neg_count.items(), key=lambda item: item[1], reverse=True)
    )

    pos_ae_df = pd.DataFrame(
        list(pos_count.items()), columns=["Species", "Events"]
    )
    neg_ae_df = pd.DataFrame(
        list(neg_count.items()), columns=["Species", "Events"]
    )

    return pos_ae_df, neg_ae_df


def _compute_pair_fit_and_residuals(
    processed_zscores: pd.DataFrame,
    pair: str,
    spline_type: str,
    degree: int,
    fit_threshold: float = None,
    linear_through_origin: bool = False,
    dof: int = None,
) -> pd.DataFrame:
    """Fit a spline to the Z-score scatter for a single sample pair and
    compute per-peptide/epitope residuals.

    Returns a DataFrame with columns ``x``, ``yfit``, ``maxZ``, ``deltaZ``
    (``FeatureData[Spline]``).  When *epitope_map* is supplied, ``maxZ`` and
    ``deltaZ`` are collapsed to epitope IDs, so those rows will differ from
    the peptide rows that carry ``x`` and ``yfit``; the non-applicable cells
    are ``NaN``.
    """
    # FeatureTable[Zscore] arrives as samples × features; convert to
    # features × samples so we can index by sample name.
    processed_zscores = processed_zscores.transpose()
    dof = ro.NULL if dof is None else dof

    sample_a, sample_b = pair.split('~')
    pair_list = [sample_a, sample_b]
    data_sorted = processed_zscores.loc[:, pair_list].sort_values(by=sample_a)
    x = data_sorted.loc[:, sample_a].to_numpy()
    y = data_sorted.loc[:, sample_b].to_numpy()

    if spline_type == "py-smooth":
        yfit = splines.smooth_spline(x, y)
    elif spline_type == "py-LinearGAM":
        yfit = splines.smooth_gam(x, y)
    elif spline_type == "linear":
        if fit_threshold is not None:
            filtered_mask = (x > fit_threshold) & (y > fit_threshold)
            x_filtered = x[filtered_mask]
            y_filtered = y[filtered_mask]
        else:
            x_filtered, y_filtered = x, y

        if len(x_filtered) < 2 or len(y_filtered) < 2:
            raise ValueError(
                "Not enough points to fit linear spline after threshold"
                f" filter: len(x_filtered)={len(x_filtered)},"
                f" len(y_filtered)={len(y_filtered)},"
                f" fit_threshold={fit_threshold}, pair={pair}"
            )

        yfit = splines.linear_regression(
            x_filtered,
            y_filtered,
            x_pred=x,
            through_origin=linear_through_origin,
        )
    elif spline_type == "cubic":
        with numpy2ri.converter.context():
            yfit = splines.R_SPLINES.cubic_spline(x, y, degree, dof)
    elif spline_type == "natural-cubic":
        yfit = splines.natural_cubic_spline(x, y)
    else:
        with numpy2ri.converter.context():
            yfit = splines.R_SPLINES.smooth_spline(x, y)

    maxZ = np.apply_over_axes(np.max, data_sorted.loc[:, pair_list], 1)
    maxZ = pd.Series(
        [num for elem in maxZ for num in elem], index=data_sorted.index
    )
    deltaZ = pd.Series(y - yfit, index=data_sorted.index)

    spline_df = pd.concat([
        pd.DataFrame(
            {"x": x, "yfit": yfit},
            index=data_sorted.index,
        ),
        pd.DataFrame({"maxZ": maxZ, "deltaZ": deltaZ}),
    ], axis=1)
    spline_df.index.name = "feature-id"

    return spline_df


def _map_residuals_and_zscores(
    spline: pd.DataFrame,
    zscores: pd.DataFrame,
    epitope_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zscores = zscores.transpose()

    spline_name = spline.index.name
    mapped_spline_rows = []

    mapped_zscores_rows = []

    def map_helper(row):
        # Skip this if this row is a peptide not an epitope. This check shows
        # that because row['CodeName'] is all peptides, so if row.name matches
        # the first (and if it's a peptide only) entry in row['CodeName'] the
        # row must be an uncollapsed peptide
        if row.name != row['CodeName'][0]:
            # Get row with peptide with largest residual
            max_peptide_row = spline.loc[
                spline.loc[row['CodeName']]['deltaZ'].abs().idxmax()
            ]
            max_peptide_zscores = zscores.loc[max_peptide_row.name]

            max_peptide_row.name = row.name
            max_peptide_zscores.name = row.name

            mapped_spline_rows.append(max_peptide_row)
            mapped_zscores_rows.append(max_peptide_zscores)
        else:
            mapped_zscores_rows.append(zscores.loc[row.name])

    epitope_map.apply(map_helper, axis=1)

    # Add our mapped spline info onto the old ones because we need the
    # per-peptide spline info to stay around
    spline = pd.concat(
        [spline, pd.DataFrame(mapped_spline_rows)],
        ignore_index=False
    )
    spline.index.name = spline_name

    # We do not need the per peptide z scores anymore, so we overwrite them
    mapped_zscores = pd.DataFrame(mapped_zscores_rows, columns=zscores.columns)
    mapped_zscores.index.name = zscores.index.name

    return spline, mapped_zscores


def _filter_scores_to_pairs(
    scores: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Select Z-scores for the samples referenced in *pairs*.

    Parameters
    ----------
    scores : pd.DataFrame
        Z-score matrix from FeatureTable[Zscore] (samples × features).
    pairs : pd.DataFrame
        Two-column pairs table (from PSEAPairs).

    Returns
    -------
    pd.DataFrame
        Filtered Z-score matrix (features × samples) stored as
        FeatureTable[Zscore].
    """
    # FeatureTable[Zscore] arrives as samples × features; convert to
    # features × samples so we can index by sample name.
    scores = scores.transpose()

    reps_list = list(np.unique(pairs.values.flatten()))
    filtered_zscores = scores.loc[:, reps_list]

    return filtered_zscores


def _process_scores(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Log-scale Z-scores for the scores.

    Parameters
    ----------
    scores : pd.DataFrame
        Z-score matrix from FeatureTable[Zscore] (samples × features).

    Returns
    -------
    pd.DataFrame
        Processed Z-score matrix (features × samples) stored as
        FeatureTable[Zscore].
    """
    # FeatureTable[Zscore] arrives as samples × features; convert to
    # features × samples so we can index by sample name.
    scores = scores.transpose()

    base = 2
    offset = 3
    power = pow(base, offset)

    processed_zscores = scores.apply(lambda row: power + row, axis=0)
    processed_zscores = processed_zscores.apply(
        lambda row: row.apply(lambda val: 1 if val < 1 else val),
        axis=0,
    )
    return processed_zscores.apply(
        lambda row: row.apply(lambda val: log(val, base) - offset)
    )


def _split_scores(
    scores: pd.DataFrame,
    pairs: list
) -> pd.DataFrame:
    # FeatureTable[Zscore] arrives as samples × features; convert to
    # features × samples so we can index by sample name.
    scores = scores.transpose()

    split_scores = {}

    for pair in pairs:
        pair_list = pair.split('~')
        split_scores[pair] = scores.loc[:, pair_list]

    return split_scores
