import numpy as np
import os
import pandas as pd
import qiime2
import rpy2.robjects as ro
import q2_PSEA.actions.splines as splines
import q2_PSEA.utils as utils
import tempfile

import time

from math import log, pow
from rpy2.robjects import pandas2ri, numpy2ri
from q2_PSEA.actions.r_functions import INTERNAL


def create_fgsea_table_for_pair(
    processed_scores: pd.DataFrame,
    peptide_sets: pd.DataFrame,
    precomputed_fit: pd.DataFrame,
    sample_a: str,
    sample_b: str,
    threshold: float,
    permutation_num: int,
    min_size: int,
    max_size: int,
    seed: int,
    species_taxa: qiime2.Metadata = None,
) -> pd.DataFrame:
    """QIIME 2 method: compute the fgsea PSEA table for a single sample pair.

    Parameters
    ----------
    processed_scores : pd.DataFrame
        Log-scaled Z-score matrix (from FeatureTable[Zscore]).
    peptide_sets : pd.DataFrame
        GMT peptide-set table with columns 'term' and 'gene' (from GMT).
    sample_a, sample_b : str
        Names of the two samples forming the pair.
    species_taxa : PSEASpeciesTaxaDirFmt, optional
        Directory format containing species-taxa.tsv; passed as a file path
        to the underlying R function.

    Returns
    -------
    pd.DataFrame
        PSEA result table for this pair (stored as FeatureData[PSEAScores]).
    """
    print(f"Working on pair ({sample_a}, {sample_b})...")

    processed_scores = processed_scores.transpose()

    maxZ_all = precomputed_fit["maxZ"].dropna()
    deltaZ_all = precomputed_fit["deltaZ"].dropna()

    filtered_scores, peptide_sets_for_analysis = \
        utils.remove_peptides(processed_scores, peptide_sets)

    idx = filtered_scores.index
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
    p_val_thresh: float,
    nes_thresh: float,
    taxa_access: str,
) -> (pd.DataFrame, pd.DataFrame):
    """QIIME 2 method: count positive- and negative-NES antibody events.

    Iterates over every PSEA table in *psea_tables*. A taxon is counted as one
    event for a given pair when its adjusted p-value is below *p_val_thresh*
    and the absolute value of its NES exceeds *nes_thresh*. Positive and
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
                row["p.adjust"] < p_val_thresh
                and abs(row["NES"]) > nes_thresh
            ):
                if row["NES"] > 0:
                    pos_count[taxa] = pos_count.get(taxa, 0) + 1
                elif row["NES"] < 0:
                    neg_count[taxa] = neg_count.get(taxa, 0) + 1
                else:
                    zero_count[taxa] = zero_count.get(taxa, 0) + 1

    if zero_count:
        print()
        for taxa, count in zero_count.items():
            event_word = "events" if count > 1 else "event"
            print(f"{count} {event_word} for {taxa}, which has an NES of 0")

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


def run_iterative_process_single_pair(
    processed_scores: pd.DataFrame,
    peptide_sets: pd.DataFrame,
    sample_a: str,
    sample_b: str,
    threshold: float,
    permutation_num: int,
    min_size: int,
    max_size: int,
    seed: int,
    p_val_thresh: float,
    nes_thresh: float,
    epitope_map: pd.DataFrame = None,
    mapped_peptide_sets: pd.DataFrame = None,
    precomputed_fit: pd.DataFrame = None,
    species_taxa: qiime2.Metadata = None,
) -> pd.DataFrame:
    """QIIME 2 pipeline: run one iteration of iterative peptide analysis for a
    single sample pair.

    Calls the registered ``create_fgsea_table_for_pair`` method, finds the
    most significant species not yet in *tested_species*, removes its leading-
    edge peptides from all other species in the GMT, and returns the updated
    peptide sets alongside the PSEA table for this iteration.

    Returns
    -------
    updated_peptide_sets : GMT
    """
    updated_peptide_sets = (
        mapped_peptide_sets if mapped_peptide_sets is not None
        else peptide_sets
    )

    tested_species = set()
    iteration = 1
    sig_found = True
    while (sig_found):
        print(f"\nIteration: {iteration} for pair: ({sample_a}, {sample_b})")

        # Called as a raw Python function not a QIIME 2 Method
        psea_table = create_fgsea_table_for_pair(
            processed_scores=processed_scores,
            peptide_sets=updated_peptide_sets,
            sample_a=sample_a,
            sample_b=sample_b,
            threshold=threshold,
            permutation_num=permutation_num,
            min_size=min_size,
            max_size=max_size,
            seed=seed,
            species_taxa=species_taxa,
            precomputed_fit=precomputed_fit,
        )

        updated_peptide_sets, tested_species, sig_found = _filter_peptide_sets(
            psea_table,
            updated_peptide_sets,
            tested_species,
            p_val_thresh,
            nes_thresh,
            sample_a,
            sample_b,
            epitope_map=epitope_map
        )

        iteration += 1

    return updated_peptide_sets


def _filter_peptide_sets(
            psea_table: pd.DataFrame,
            updated_peptide_sets: pd.DataFrame,
            tested_species: set,
            p_val_thresh: int,
            nes_thresh: int,
            sample_a: str,
            sample_b: str,
            epitope_map: pd.DataFrame = None,
        ) -> tuple[pd.DataFrame, set, bool]:
    psea_table = psea_table.sort_values(by=["p.adjust"], ascending=True)\

    sig_found = True
    for _, row in psea_table.iterrows():
        row_id = str(row["ID"])
        if (
            row["p.adjust"] < p_val_thresh
            and abs(row["NES"]) > nes_thresh
            and row_id not in tested_species
        ):
            print(
                f"Found {row.get('species_name', row['ID'])} in"
                f" ({sample_a}, {sample_b}) to be significant"
            )
            all_tested_peps = set(row["all_tested_peptides"].split("/"))

            # TODO: In principle this should work, but I'm not seeing it
            # filter anything
            #
            # Write unit test
            if epitope_map is not None:
                all_tested_peps = _get_mapped_peps(
                    epitope_map, all_tested_peps
                )

            mask = (
                (updated_peptide_sets["term"].astype(str) != row_id)
                & updated_peptide_sets["gene"].isin(all_tested_peps)
            )

            updated_peptide_sets = updated_peptide_sets[~mask]
            tested_species.add(row_id)
            break
    else:
        sig_found = False

    return updated_peptide_sets, tested_species, sig_found


# TODO: optimize this.
def _get_mapped_peps(epitope_df, all_tested_features):
    epitopes = set()
    for _, row in epitope_df.iterrows():
        for tested in all_tested_features:
            if tested in row['CodeName']:
                epitopes.add(row.name)

    return epitopes


def make_psea_table(
    ctx,
    scores,
    pairs,
    peptide_sets,
    threshold,
    species_taxa=None,
    species_colors=None,
    epitope=None,
    epitope_map=None,
    mapped_zscores=None,
    mapped_gmt=None,
    collapse="Viral",
    p_val_thresh=0.05,
    nes_thresh=1,
    min_size=15,
    max_size=2000,
    permutation_num=10000,
    spline_type="r-smooth",
    degree=3,
    dof=None,
    iterative_analysis=True,
    seed=149,
):
    start_time = time.perf_counter()

    if any([epitope_map, mapped_zscores, mapped_gmt]) \
            and not all([epitope_map, mapped_zscores, mapped_gmt]):
        raise ValueError(
            "Please pass either all of 'epitope_map',"" 'mapped_zscores',"
            " and 'mapped_gmt' or none of them"
        )

    process_scores_action = ctx.get_action("psea", "process_scores")
    compute_fit = ctx.get_action("psea", "_compute_pair_fit_and_residuals")
    run_iterative = ctx.get_action("psea", "run_iterative_process_single_pair")
    create_fgsea_table = ctx.get_action("psea", "create_fgsea_table_for_pair")

    count_ae = ctx.get_action("psea", "count_antibody_events")

    volcano = ctx.get_action("psea", "volcano")
    zscatter = ctx.get_action("psea", "zscatter")
    aeplots = ctx.get_action("psea", "aeplots")

    taxa_access = "species_name" if species_taxa is not None else "ID"

    # ------------------------------------------------------------------
    # Parse pairs list
    # ------------------------------------------------------------------
    pairs_list = pairs.view(list)

    # ------------------------------------------------------------------
    # Handle epitope collapsing
    # ------------------------------------------------------------------
    collapsed = False
    if epitope is not None and epitope_map is None:
        collapsed = True

        create_epitope_map = ctx.get_action("psea", "create_epitope_map")
        create_epitope_zscore = ctx.get_action("psea", "epitope_zscore")
        create_epitope_gmt = ctx.get_action("psea", "taxa_to_epitope")

        epitope_map, = create_epitope_map(epitope, collapse)
        mapped_zscores, = create_epitope_zscore(scores, epitope_map)
        mapped_gmt, = create_epitope_gmt(epitope_map)
    elif epitope_map is not None:
        collapsed = True

    # ------------------------------------------------------------------
    # Process (log-scale) scores
    # ------------------------------------------------------------------
    processed_scores, = process_scores_action(scores=scores, pairs=pairs)

    processed_mapped_scores = None
    if collapsed:
        processed_mapped_scores, = process_scores_action(
            scores=mapped_zscores, pairs=pairs
        )

    pair_splines = {}
    pair_pep_sets_dict = {}
    psea_tables = {}

    # NOTE: We can parallelize pairs. We cannot parallelize iterations
    for pair in pairs_list:
        sample_a, sample_b = pair.split("~")

        # Compute spline fit once per pair; reuse it for both the scatter
        # plot data and as the precomputed_fit input to create_fgsea_table.
        pair_splines[pair], = compute_fit(
            processed_scores=processed_scores,
            sample_a=sample_a,
            sample_b=sample_b,
            spline_type=spline_type,
            degree=degree,
            dof=dof,
            epitope_map=epitope_map,
        )

        # ------------------------------------------------------------------
        # Determine per-pair peptide sets (iterative or flat)
        # ------------------------------------------------------------------
        if iterative_analysis:
            pair_pep_sets_dict[pair], = run_iterative(
                processed_scores=processed_mapped_scores if collapsed else
                processed_scores,
                peptide_sets=peptide_sets,
                precomputed_fit=pair_splines[pair],
                epitope_map=epitope_map,
                mapped_peptide_sets=mapped_gmt,
                sample_a=sample_a,
                sample_b=sample_b,
                threshold=threshold,
                permutation_num=permutation_num,
                min_size=min_size,
                max_size=max_size,
                seed=seed,
                p_val_thresh=p_val_thresh,
                nes_thresh=nes_thresh,
                species_taxa=species_taxa,
            )
        else:
            pair_pep_sets_dict[pair] = \
                mapped_gmt if collapsed else peptide_sets

        # ------------------------------------------------------------------
        # Final per-pair PSEA analysis
        # ------------------------------------------------------------------
        psea_tables[pair], = create_fgsea_table(
            processed_scores=processed_mapped_scores if collapsed else
            processed_scores,
            peptide_sets=pair_pep_sets_dict[pair],
            sample_a=sample_a,
            sample_b=sample_b,
            threshold=threshold,
            permutation_num=permutation_num,
            min_size=min_size,
            max_size=max_size,
            seed=seed,
            species_taxa=species_taxa,
            precomputed_fit=pair_splines[pair],
        )

    # ------------------------------------------------------------------
    # Count antibody events and build visualizations
    # ------------------------------------------------------------------
    pos_ae_counts, neg_ae_counts = count_ae(
        psea_tables=psea_tables,
        p_val_thresh=p_val_thresh,
        nes_thresh=nes_thresh,
        taxa_access=taxa_access,
    )

    scatter_plot, = zscatter(
        zscores=(
            processed_mapped_scores if collapsed else processed_scores
        ),
        pairs=pairs,
        splines=pair_splines,
        p_val_access="p.adjust",
        le_peps_access="core_enrichment",
        taxa_access=taxa_access,
        psea_tables=psea_tables,
        highlight_threshold=p_val_thresh,
        colors_file=species_colors,
    )

    volcano_plot, = volcano(
        pairs=pairs,
        psea_tables=psea_tables,
        xy_access=["NES", "p.adjust"],
        taxa_access=taxa_access,
        x_threshold=nes_thresh,
        y_threshold=p_val_thresh,
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

    end_time = time.perf_counter()
    # TODO: This becomes meaningless when running in parallel
    print(f"\nFinished in {round(end_time - start_time, 2)} seconds")

    return scatter_plot, volcano_plot, ae_plot, psea_tables


def _compute_pair_fit_and_residuals(
    processed_scores: pd.DataFrame,
    sample_a: str,
    sample_b: str,
    spline_type: str,
    degree: int,
    dof: int = None,
    epitope_map: pd.DataFrame = None,
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
    processed_scores = processed_scores.transpose()
    dof = ro.NULL if dof is None else dof

    pair = [sample_a, sample_b]
    data_sorted = processed_scores.loc[:, pair].sort_values(by=sample_a)
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

    # TODO: Do we also need to map x and y?
    if epitope_map is not None:
        maxZ_out = utils.collapse_residuals_to_epitope(maxZ, epitope_map)
        deltaZ_out = utils.collapse_residuals_to_epitope(deltaZ, epitope_map)
    else:
        maxZ_out = maxZ
        deltaZ_out = deltaZ

    # TODO: Why does it do it like this?
    spline_df = pd.concat([
        pd.DataFrame(
            {"x": x, "yfit": yfit},
            index=data_sorted.index,
        ),
        pd.DataFrame({"maxZ": maxZ_out, "deltaZ": deltaZ_out}),
    ], axis=1)
    spline_df.index.name = "feature-id"

    return spline_df


def process_scores(
    scores: pd.DataFrame,
    pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Select and log-scale Z-scores for the samples referenced in *pairs*.

    Parameters
    ----------
    scores : pd.DataFrame
        Z-score matrix from FeatureTable[Zscore] (samples × features).
    pairs : pd.DataFrame
        Two-column pairs table (from PSEAPairs).

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
    reps_list = list(np.unique(pairs.values.flatten()))
    processed_scores = scores.loc[:, reps_list]

    processed_scores = processed_scores.apply(lambda row: power + row, axis=0)
    processed_scores = processed_scores.apply(
        lambda row: row.apply(lambda val: 1 if val < 1 else val),
        axis=0,
    )
    return processed_scores.apply(
        lambda row: row.apply(lambda val: log(val, base) - offset)
    )
