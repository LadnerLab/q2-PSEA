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
from q2_PSEA.format_types import PSEAPairsDirFmt


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
    epitope_map: pd.DataFrame = None,
    mapped_processed_scores: pd.DataFrame = None,
    mapped_peptide_sets: pd.DataFrame = None,
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
    epitope_map : pd.DataFrame, optional
        Mapped-epitope table (from FeatureData[MappedEpitope]).
    mapped_processed_scores : pd.DataFrame, optional
        Epitope-level processed scores (from FeatureTable[Zscore]).
    mapped_peptide_sets : pd.DataFrame, optional
        Epitope-level GMT table (from GMT).

    Returns
    -------
    pd.DataFrame
        PSEA result table for this pair (stored as FeatureData[PSEAScores]).
    """
    print(f"Working on pair ({sample_a}, {sample_b})...")

    processed_scores = processed_scores.transpose()
    if mapped_processed_scores is not None:
        mapped_processed_scores = mapped_processed_scores.transpose()

    maxZ_all = precomputed_fit["maxZ"].dropna()
    deltaZ_all = precomputed_fit["deltaZ"].dropna()

    if epitope_map is not None:
        filtered_scores, peptide_sets_for_analysis = \
            utils.remove_peptides(mapped_processed_scores, mapped_peptide_sets)
    else:
        filtered_scores, peptide_sets_for_analysis = \
            utils.remove_peptides(processed_scores, peptide_sets)

    idx = filtered_scores.index
    maxZ = maxZ_all.reindex(idx)
    deltaZ = deltaZ_all.reindex(idx)

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


def run_iterative_process_single_pair(
    ctx,
    processed_scores,
    peptide_sets,
    sample_a,
    sample_b,
    threshold,
    permutation_num,
    min_size,
    max_size,
    seed,
    p_val_thresh,
    nes_thresh,
    tested_species=None,
    species_taxa=None,
    epitope_map=None,
    mapped_processed_scores=None,
    mapped_peptide_sets=None,
    precomputed_fit=None,
):
    """QIIME 2 pipeline: run one iteration of iterative peptide analysis for a
    single sample pair.

    Calls the registered ``create_fgsea_table_for_pair`` method, finds the
    most significant species not yet in *tested_species*, removes its leading-
    edge peptides from all other species in the GMT, and returns the updated
    peptide sets alongside the PSEA table for this iteration.

    Returns
    -------
    psea_table : FeatureData[PSEAScores]
    updated_peptide_sets : GMT
    """
    create_fgsea_table = ctx.get_action("psea", "create_fgsea_table_for_pair")

    psea_table, = create_fgsea_table(
        processed_scores=processed_scores,
        peptide_sets=peptide_sets,
        sample_a=sample_a,
        sample_b=sample_b,
        threshold=threshold,
        permutation_num=permutation_num,
        min_size=min_size,
        max_size=max_size,
        seed=seed,
        species_taxa=species_taxa,
        epitope_map=epitope_map,
        mapped_processed_scores=mapped_processed_scores,
        mapped_peptide_sets=mapped_peptide_sets,
        precomputed_fit=precomputed_fit,
    )

    if tested_species is None:
        tested_species = []

    table_df = psea_table.view(pd.DataFrame)
    table_df_sorted = table_df.sort_values(by=["p.adjust"], ascending=True)

    gmt_df = peptide_sets.view(pd.DataFrame)

    for _, row in table_df_sorted.iterrows():
        row_id = str(row["ID"])
        if (
            row["p.adjust"] < p_val_thresh
            and abs(row["NES"]) > nes_thresh
            and row_id not in [str(s) for s in tested_species]
        ):
            print(
                f"Found {row.get('species_name', row['ID'])} in"
                f" ({sample_a}, {sample_b}) to be significant"
            )
            all_tested_peps = set(row["all_tested_peptides"].split("/"))
            mask = (
                (gmt_df["term"].astype(str) != row_id)
                & gmt_df["gene"].isin(all_tested_peps)
            )
            gmt_df = gmt_df[~mask].copy()
            break

    updated_peptide_sets = ctx.make_artifact("GMT", gmt_df)
    return psea_table, updated_peptide_sets


def run_iterative_peptide_analysis(
    ctx,
    processed_scores,
    pairs,
    peptide_sets,
    threshold,
    permutation_num,
    min_size,
    max_size,
    spline_type,
    degree,
    seed,
    p_val_thresh,
    nes_thresh,
    dof=None,
    species_taxa=None,
    epitope_map=None,
    mapped_processed_scores=None,
    mapped_peptide_sets=None,
):
    """QIIME 2 pipeline: iteratively filter cross-reactive peptides for every
    sample pair.

    Calls the registered ``run_iterative_process_single_pair`` pipeline once
    per pair per iteration until no new significant species are discovered.

    Returns
    -------
    filtered_peptide_sets : Collection[GMT]
        One final filtered GMT artifact per pair, in the same order as the
        rows of the pairs file.
    """
    run_single_pair = ctx.get_action(
        "psea", "run_iterative_process_single_pair"
    )

    pairs_dirfmt = pairs.view(PSEAPairsDirFmt)
    pairs_df = pd.read_csv(
        str(pairs_dirfmt.path / "pairs.tsv"), sep="\t", header=0
    )
    pairs_list = [
        (str(row.iloc[0]), str(row.iloc[1]))
        for _, row in pairs_df.iterrows()
    ]

    pair_gmt_dict = {pair: peptide_sets for pair in pairs_list}
    sig_species_found_dict = {pair: True for pair in pairs_list}
    tested_species_dict = {pair: [] for pair in pairs_list}

    # Compute spline fit and residuals once per pair before iterating.
    # Calling via ctx ensures provenance is captured and QIIME 2 handles
    # data transformation automatically.
    compute_fit = ctx.get_action("psea", "_compute_pair_fit_and_residuals")
    pair_fit_artifact = {}
    for pair in pairs_list:
        sample_a, sample_b = pair
        spline_art, = compute_fit(
            processed_scores=processed_scores,
            sample_a=sample_a,
            sample_b=sample_b,
            spline_type=spline_type,
            degree=degree,
            dof=dof,
            epitope_map=epitope_map,
        )
        pair_fit_artifact[pair] = spline_art

    iteration_num = 1

    while any(sig_species_found_dict.values()):
        print(f"\nIteration: {iteration_num}")

        for pair in pairs_list:
            if not sig_species_found_dict[pair]:
                continue

            sample_a, sample_b = pair

            iter_psea_table, updated_gmt = run_single_pair(
                processed_scores=processed_scores,
                peptide_sets=pair_gmt_dict[pair],
                sample_a=sample_a,
                sample_b=sample_b,
                threshold=threshold,
                permutation_num=permutation_num,
                min_size=min_size,
                max_size=max_size,
                seed=seed,
                p_val_thresh=p_val_thresh,
                nes_thresh=nes_thresh,
                tested_species=(
                    tested_species_dict[pair] if
                    tested_species_dict[pair] != [] else None
                ),
                species_taxa=species_taxa,
                epitope_map=epitope_map,
                mapped_processed_scores=mapped_processed_scores,
                mapped_peptide_sets=mapped_peptide_sets,
                precomputed_fit=pair_fit_artifact[pair],
            )

            table_df = iter_psea_table.view(pd.DataFrame)
            table_df_sorted = table_df.sort_values(
                by=["p.adjust"], ascending=True
            )

            sig_found = False
            for _, row in table_df_sorted.iterrows():
                row_id = str(row["ID"])
                if (
                    row["p.adjust"] < p_val_thresh
                    and abs(row["NES"]) > nes_thresh
                    and row_id
                    not in [str(s) for s in tested_species_dict[pair]]
                ):
                    sig_found = True
                    tested_species_dict[pair].append(row_id)
                    break

            sig_species_found_dict[pair] = sig_found
            pair_gmt_dict[pair] = updated_gmt

        iteration_num += 1

    print("\nEnd of Iterative Peptide Analysis\n")
    return list(pair_gmt_dict.values())


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


def make_psea_table(
    ctx,
    scores,
    pairs,
    peptide_sets,
    threshold,
    species_taxa=None,
    species_colors=None,
    epitope=None,
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

    volcano = ctx.get_action("psea", "volcano")
    zscatter = ctx.get_action("psea", "zscatter")
    aeplots = ctx.get_action("psea", "aeplots")
    create_fgsea_table = ctx.get_action("psea", "create_fgsea_table_for_pair")
    compute_fit = ctx.get_action("psea", "_compute_pair_fit_and_residuals")

    taxa_access = "species_name" if species_taxa is not None else "ID"

    # ------------------------------------------------------------------
    # Parse pairs list
    # ------------------------------------------------------------------
    pairs_df = pairs.view(pd.DataFrame)
    pairs_list = [
        (str(row.iloc[0]), str(row.iloc[1]))
        for _, row in pairs_df.iterrows()
    ]

    # ------------------------------------------------------------------
    # Handle epitope collapsing
    # ------------------------------------------------------------------
    mapped_epitope = None
    epitope_zscore = None
    epitope_gmt = None

    if epitope is not None:
        create_epitope_map = ctx.get_action("psea", "create_epitope_map")
        mapped_epitope, = create_epitope_map(epitope, collapse)

        create_epitope_zscore = ctx.get_action("psea", "epitope_zscore")
        epitope_zscore, = create_epitope_zscore(scores, mapped_epitope)

        create_epitope_gmt = ctx.get_action("psea", "taxa_to_epitope")
        epitope_gmt, = create_epitope_gmt(epitope, collapse)

    # ------------------------------------------------------------------
    # Process (log-scale) scores
    # ------------------------------------------------------------------
    scores_df = scores.view(pd.DataFrame)
    scores_df = scores_df.transpose()

    processed_scores_df = process_scores(scores_df, pairs_list)
    processed_scores_art = ctx.make_artifact(
        "FeatureTable[Zscore]", processed_scores_df
    )

    mapped_processed_scores_art = None
    if epitope is not None:
        epitope_zscore_df = epitope_zscore.view(pd.DataFrame).transpose()
        mapped_processed_scores_df = process_scores(
            epitope_zscore_df, pairs_list
        )
        mapped_processed_scores_art = ctx.make_artifact(
            "FeatureTable[Zscore]", mapped_processed_scores_df
        )

    # ------------------------------------------------------------------
    # Determine per-pair peptide sets (iterative or flat)
    # ------------------------------------------------------------------
    if iterative_analysis:
        run_iterative = ctx.get_action(
            "psea", "run_iterative_peptide_analysis"
        )
        filtered_gmts, = run_iterative(
            processed_scores=processed_scores_art,
            pairs=pairs,
            peptide_sets=peptide_sets,
            threshold=threshold,
            permutation_num=permutation_num,
            min_size=min_size,
            max_size=max_size,
            spline_type=spline_type,
            degree=degree,
            seed=seed,
            p_val_thresh=p_val_thresh,
            nes_thresh=nes_thresh,
            dof=dof,
            species_taxa=species_taxa,
            epitope_map=mapped_epitope,
            mapped_processed_scores=mapped_processed_scores_art,
            mapped_peptide_sets=epitope_gmt,
        )

        pair_pep_sets_dict = {
            pair: gmt for pair, gmt in zip(pairs_list, filtered_gmts.values())
        }
    else:
        pair_pep_sets_dict = {pair: peptide_sets for pair in pairs_list}

    # ------------------------------------------------------------------
    # Final per-pair PSEA analysis
    # ------------------------------------------------------------------
    pair_spline_dict = {"x": list(), "y": list(), "pair": list()}
    psea_tables = {}

    for pair in pairs_list:
        sample_a, sample_b = pair
        table_prefix = f"{sample_a}~{sample_b}"

        # Compute spline fit once per pair; reuse it for both the scatter
        # plot data and as the precomputed_fit input to create_fgsea_table.
        spline_art, = compute_fit(
            processed_scores=processed_scores_art,
            sample_a=sample_a,
            sample_b=sample_b,
            spline_type=spline_type,
            degree=degree,
            dof=dof,
            epitope_map=mapped_epitope,
        )

        spline_df = spline_art.view(pd.DataFrame)
        x = spline_df["x"].dropna().to_numpy()
        yfit = spline_df["yfit"].dropna().to_numpy()

        psea_table, = create_fgsea_table(
            processed_scores=processed_scores_art,
            peptide_sets=pair_pep_sets_dict[pair],
            sample_a=sample_a,
            sample_b=sample_b,
            threshold=threshold,
            permutation_num=permutation_num,
            min_size=min_size,
            max_size=max_size,
            seed=seed,
            species_taxa=species_taxa,
            epitope_map=mapped_epitope,
            mapped_processed_scores=mapped_processed_scores_art,
            mapped_peptide_sets=epitope_gmt,
            precomputed_fit=spline_art,
        )
        psea_tables[table_prefix] = psea_table

        pair_spline_dict["x"].extend(x.tolist())
        pair_spline_dict["y"].extend(yfit.tolist())
        pair_spline_dict["pair"].extend([table_prefix] * len(x))

    # ------------------------------------------------------------------
    # Count antibody events and build visualizations
    # ------------------------------------------------------------------
    count_ae = ctx.get_action("psea", "count_antibody_events")
    pos_ae_counts, neg_ae_counts = count_ae(
        psea_tables=psea_tables,
        p_val_thresh=p_val_thresh,
        nes_thresh=nes_thresh,
        taxa_access=taxa_access,
    )

    with tempfile.TemporaryDirectory() as spline_tempdir:
        spline_file = os.path.join(spline_tempdir, "spline_data.tsv")
        pd.DataFrame(pair_spline_dict).to_csv(
            spline_file, sep="\t", index=False
        )

        scatter_plot, = zscatter(
            zscores=(
                mapped_processed_scores_art if epitope is not None
                else processed_scores_art
            ),
            pairs=pairs,
            spline_file=spline_file,
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

    pair = (sample_a, sample_b)
    data_sorted = processed_scores.loc[:, list(pair)].sort_values(by=sample_a)
    x = data_sorted.loc[:, sample_a].to_numpy()
    y = data_sorted.loc[:, sample_b].to_numpy()

    dof_r = ro.NULL if dof is None else dof

    if spline_type == "py-smooth":
        yfit = splines.smooth_spline(x, y)
    elif spline_type == "py-LinearGAM":
        yfit = splines.smooth_gam(x, y)
    elif spline_type == "cubic":
        with numpy2ri.converter.context():
            yfit = splines.R_SPLINES.cubic_spline(x, y, degree, dof_r)
    else:
        with numpy2ri.converter.context():
            yfit = splines.R_SPLINES.smooth_spline(x, y)

    maxZ = np.apply_over_axes(np.max, data_sorted.loc[:, list(pair)], 1)
    maxZ = pd.Series(
        [num for elem in maxZ for num in elem], index=data_sorted.index
    )
    deltaZ = pd.Series(y - yfit, index=data_sorted.index)

    if epitope_map is not None:
        maxZ_out = _collapse_residuals_to_epitope(maxZ, epitope_map)
        deltaZ_out = _collapse_residuals_to_epitope(deltaZ, epitope_map)
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


# ---------------------------------------------------------------------------
# Internal helpers (not registered as QIIME 2 actions)
# ---------------------------------------------------------------------------


def process_scores(scores, pairs) -> pd.DataFrame:
    """Grabs replicates specified `pairs` from scores matrix and processes
    those remaining scores.
    Returns a Pandas DataFrame of processed Z scores.
    """
    base = 2
    offset = 3
    power = pow(base, offset)
    reps_list = []
    for pair in pairs:
        for rep in pair:
            reps_list.append(rep)
    reps_list = list(np.unique(reps_list))
    processed_scores = scores.loc[:, reps_list]

    processed_scores = processed_scores.apply(lambda row: power + row, axis=0)
    processed_scores = processed_scores.apply(
        lambda row: row.apply(lambda val: 1 if val < 1 else val),
        axis=0,
    )
    return processed_scores.apply(
        lambda row: row.apply(lambda val: log(val, base) - offset)
    )


def write_gmt_from_dict(outfile_name, gmt_dict) -> None:
    with open(outfile_name, "w") as gmt_file:
        for species in gmt_dict.keys():
            gmt_file.write(f"{species}\t\t")
            for peptide in gmt_dict[species]:
                gmt_file.write(f"{peptide}\t")
            gmt_file.write("\n")


def create_df_from_gmt(gmt_file_path):
    result = pd.DataFrame(columns=["EpitopeID"])
    with open(gmt_file_path) as fh:
        for line in fh.readlines():
            speciesID, epitopeID = line.split("\t\t")
            epitopeID = epitopeID.split("\t")
            result.loc[speciesID] = [epitopeID]
    result.index.name = "SpeciesID"
    return result


def _collapse_residuals_to_epitope(peptide_residuals, epitope_map):
    peptide_to_epitopes = {}
    for epitope, peptides in epitope_map["CodeName"].items():
        for peptide in peptides:
            if peptide not in peptide_to_epitopes:
                peptide_to_epitopes[peptide] = []
            peptide_to_epitopes[peptide].append(epitope)

    epitope_residuals = {}
    for peptide, residual in peptide_residuals.items():
        mapped_epitopes = peptide_to_epitopes.get(peptide)
        if not mapped_epitopes:
            mapped_epitopes = (peptide,)
        for epitope in mapped_epitopes:
            if epitope not in epitope_residuals:
                epitope_residuals[epitope] = residual
            elif abs(residual) > abs(epitope_residuals[epitope]):
                epitope_residuals[epitope] = residual

    return pd.Series(epitope_residuals)
