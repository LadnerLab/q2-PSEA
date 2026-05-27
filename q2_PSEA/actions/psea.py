import numpy as np
import os
import pandas as pd
import qiime2
import random
import rpy2.robjects as ro
import q2_PSEA.actions.splines as splines
import q2_PSEA.utils as utils
import tempfile

from math import log, pow
from qiime2.plugin import CaptureHolder, IContext
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
    peptide_map: qiime2.Artifact = None,
    mapped_zscores: qiime2.Artifact = None,
    mapped_gmt: qiime2.Artifact = None,
    collapse: str = "Viral",
    p_value: float = 0.05,
    enrichment_score: float = 1,
    include_negative_enrichment: bool = True,
    min_size: int = 15,
    max_size: int = 2000,
    permutation_num: int = 10000,
    spline_type: str = "r-smooth",
    degree: int = 3,
    dof: int = None,
    iterative_analysis: bool = True,
    seed: CaptureHolder[int] = None,
    species_taxa: qiime2.Metadata = None,
    species_colors: qiime2.Metadata = None,
    map: bool = True,
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
    # Determine what kind of analysis was asked for
    # ------------------------------------------------------------------
    map_provided = all(
        param is not None for param in [
            epitope_map, mapped_zscores, mapped_gmt
        ]
    )

    if not map and map_provided:
        raise ValueError("You provided mapped artifacts but indicated you do"
                         " not want mapping.")

    if map and any(
                param is not None for param in [
                    epitope_map, mapped_zscores, mapped_gmt
                ]
            ) and not map_provided:
        raise ValueError(
            "Please pass either all of 'epitope_map', 'mapped_zscores',"
            " and 'mapped_gmt' or none of them when running mapped analysis."
            " If you provide none, this pipeline will do the mapping."
        )

    if map and map_provided and iterative_analysis and not peptide_map:
        raise ValueError(
            "If doing mapped iterative analysis, you must pass in a"
            " peptide_map, or not provide mapped Artifacts and allow this"
            " pipeline to do the mapping."
        )

    _filter_scores_to_pairs = ctx.get_action("psea", "_filter_scores_to_pairs")
    _process_scores = ctx.get_action("psea", "_process_scores")
    _compute_pair_fit_and_residuals = ctx.get_action(
        "psea", "_compute_pair_fit_and_residuals"
    )
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
    # TODO: I think this needs to be more dynamic. The only one of these that's
    # likely to need rerun for every analysis is zscore
    if map and not map_provided:
        create_epitope_map = ctx.get_action("psea", "create_epitope_map")
        create_epitope_zscore = ctx.get_action("psea", "epitope_zscore")
        create_epitope_gmt = ctx.get_action("psea", "taxa_to_epitope")

        epitope_map, peptide_map = create_epitope_map(
            peptide_metadata, collapse
        )
        mapped_zscores, = create_epitope_zscore(filtered_zscores, epitope_map)
        mapped_gmt, = create_epitope_gmt(epitope_map)

    # ------------------------------------------------------------------
    # Process (log-scale) scores
    # ------------------------------------------------------------------
    processed_zscores, = _process_scores(filtered_zscores)
    mapped_processed_zscores = None
    if map:
        mapped_processed_zscores, = _process_scores(mapped_zscores)

    pair_splines = {}
    pair_pep_sets_dict = {}
    psea_tables = {}
    enrichment_tables = {}

    # NOTE: We can parallelize pairs. We cannot parallelize iterations
    for pair in pairs_list:
        sample_a, sample_b = pair.split("~")

        # Compute spline fit once per pair; reuse it for both the scatter
        # plot data and as the precomputed_fit input to create_fgsea_table.
        pair_splines[pair], = _compute_pair_fit_and_residuals(
            processed_zscores=processed_zscores,
            sample_a=sample_a,
            sample_b=sample_b,
            spline_type=spline_type,
            degree=degree,
            epitope_map=epitope_map,
            dof=dof,
        )

        # ------------------------------------------------------------------
        # Determine per-pair peptide sets (iterative or flat)
        # ------------------------------------------------------------------
        if iterative_analysis:
            pair_pep_sets_dict[pair], = _run_iterative_process_single_pair(
                processed_zscores=mapped_processed_zscores if map else
                processed_zscores,
                peptide_sets=peptide_sets,
                precomputed_fit=pair_splines[pair],
                epitope_map=epitope_map,
                peptide_map=peptide_map,
                mapped_peptide_sets=mapped_gmt,
                threshold=threshold,
                permutation_num=permutation_num,
                min_size=min_size,
                max_size=max_size,
                p_value=p_value,
                seed=seed,
                enrichment_score=enrichment_score,
                include_negative_enrichment=include_negative_enrichment,
                species_taxa=species_taxa,
            )
        else:
            pair_pep_sets_dict[pair] = \
                mapped_gmt if map else peptide_sets

        # ------------------------------------------------------------------
        # Final per-pair PSEA analysis
        # ------------------------------------------------------------------
        psea_tables[pair], = _create_fgsea_table_for_pair(
            processed_zscores=mapped_processed_zscores if map else
            processed_zscores,
            peptide_sets=pair_pep_sets_dict[pair],
            threshold=threshold,
            permutation_num=permutation_num,
            min_size=min_size,
            max_size=max_size,
            seed=seed,
            species_taxa=species_taxa,
            precomputed_fit=pair_splines[pair],
        )

        # TODO: If mapped then call count_enriched per pair here
        # If not mapped... we need to do something. Shunt out some empty files
        # the classic
        enrichment_tables[pair], = count_enriched(
            psea_table=psea_tables[pair],
            residuals=pair_splines[pair],
            epitope_map=epitope_map,
            peptide_metadata=peptide_metadata,
            p_value=p_value,
            # TODO: This should probably be parameterized seperately to
            # make-psea-table
            residual_threshold=enrichment_score,
            include_negative_enrichment=include_negative_enrichment
        )

    # ------------------------------------------------------------------
    # Count antibody events and build visualizations
    # ------------------------------------------------------------------
    pos_ae_counts, neg_ae_counts = count_antibody_events(
        psea_tables=psea_tables,
        p_value=p_value,
        enrichment_score=enrichment_score,
        taxa_access=taxa_access,
    )

    scatter_plot, = zscatter(
        zscores=(
            mapped_processed_zscores if map else processed_zscores
        ),
        pairs=pairs,
        splines=pair_splines,
        p_val_access="p.adjust",
        le_peps_access="core_enrichment",
        taxa_access=taxa_access,
        psea_tables=psea_tables,
        highlight_threshold=p_value,
        colors_file=species_colors,
    )

    volcano_plot, = volcano(
        pairs=pairs,
        psea_tables=psea_tables,
        xy_access=["NES", "p.adjust"],
        taxa_access=taxa_access,
        x_threshold=enrichment_score,
        y_threshold=p_value,
        xy_labels=["Enrichment score", "Adjusted p-values"],
        colors_file=species_colors,
    )

    ae_plot, = aeplots(
        pos_ae_counts=pos_ae_counts,
        neg_ae_counts=neg_ae_counts,
        xy_access=["Events", "Species"],
        xy_labels=["Number of AEs in cohort", "Species"],
        colors_file=species_colors,
    )

    return scatter_plot, volcano_plot, ae_plot, psea_tables, enrichment_tables


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
    epitope_map: pd.DataFrame = None,
    peptide_map: pd.DataFrame = None,
    mapped_peptide_sets: pd.DataFrame = None,
    seed: CaptureHolder[int] = None,
    include_negative_enrichment: bool = True,
    species_taxa: qiime2.Metadata = None,
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
    while (sig_found):
        # Called as a raw Python function not a QIIME 2 Method
        psea_table = _create_fgsea_table_for_pair(
            processed_zscores=processed_zscores,
            peptide_sets=updated_peptide_sets,
            threshold=threshold,
            permutation_num=permutation_num,
            min_size=min_size,
            max_size=max_size,
            seed=seed,
            species_taxa=species_taxa,
            precomputed_fit=precomputed_fit,
        )

        updated_peptide_sets, tested_species, sig_found = \
            utils.filter_peptide_sets(
                psea_table,
                updated_peptide_sets,
                tested_species,
                p_value,
                enrichment_score,
                include_negative_enrichment,
                epitope_map=epitope_map,
                peptide_map=peptide_map
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
    psea_tables: pd.DataFrame,
    p_value: float,
    enrichment_score: float,
    taxa_access: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """QIIME 2 method: count positive- and negative-NES antibody events.

    Iterates over every PSEA table in *psea_tables*. A taxon is counted as one
    event for a given pair when its adjusted p-value is below *p_value*
    and the absolute value of its NES exceeds *enrichment_score*. Positive and
    negative NES events are tallied separately.

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
    zero_count = {}

    for _, table_df in psea_tables.items():
        for _, row in table_df.iterrows():
            taxa = row[taxa_access]
            if (
                row["p.adjust"] < p_value
                and abs(row["NES"]) > enrichment_score
            ):
                if row["NES"] > 0:
                    pos_count[taxa] = pos_count.get(taxa, 0) + 1
                elif row["NES"] < 0:
                    neg_count[taxa] = neg_count.get(taxa, 0) + 1
                else:
                    zero_count[taxa] = zero_count.get(taxa, 0) + 1

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
    sample_a: str,
    sample_b: str,
    spline_type: str,
    degree: int,
    epitope_map: pd.DataFrame = None,
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

    pair = [sample_a, sample_b]
    data_sorted = processed_zscores.loc[:, pair].sort_values(by=sample_a)
    x = data_sorted.loc[:, sample_a].to_numpy()
    y = data_sorted.loc[:, sample_b].to_numpy()

    if spline_type == "py-smooth":
        yfit = splines.smooth_spline(x, y)
    elif spline_type == "py-LinearGAM":
        yfit = splines.smooth_gam(x, y)
    elif spline_type == "cubic":
        with numpy2ri.converter.context():
            yfit = splines.R_SPLINES.cubic_spline(x, y, degree, dof)
    else:
        with numpy2ri.converter.context():
            yfit = splines.R_SPLINES.smooth_spline(x, y)

    maxZ = np.apply_over_axes(np.max, data_sorted.loc[:, pair], 1)
    maxZ = pd.Series(
        [num for elem in maxZ for num in elem], index=data_sorted.index
    )
    deltaZ = pd.Series(y - yfit, index=data_sorted.index)

    if epitope_map is not None:
        maxZ_out = utils.collapse_residuals_to_epitope(maxZ, epitope_map)
        deltaZ_out = utils.collapse_residuals_to_epitope(deltaZ, epitope_map)
    else:
        maxZ_out = maxZ
        deltaZ_out = deltaZ

    spline_df = pd.concat([
        pd.DataFrame(
            {"x": x, "yfit": yfit},
            index=data_sorted.index,
        ),
        pd.DataFrame({"maxZ": maxZ_out, "deltaZ": deltaZ_out}),
    ], axis=1)
    spline_df.index.name = "feature-id"

    return spline_df


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
