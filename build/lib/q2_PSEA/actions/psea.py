import numpy as np
import os
import pandas as pd
import rpy2.robjects as ro
import q2_PSEA.actions.splines as splines
import qiime2
import q2_PSEA.utils as utils
import tempfile

import time

from math import isnan, log, pow
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from q2_pepsirf.format_types import PepsirfContingencyTSVFormat
from q2_PSEA.actions.r_functions import INTERNAL


pandas2ri.activate()


def make_psea_table(
        ctx,
        scores_file,
        pairs_file,
        peptide_sets_file,
        threshold,
        p_val_thresh=0.05,
        es_thresh=0.4,
        species_taxa_file="",
        min_size=15,
        max_size=2000,
        permutation_num=10000,  # as per original PSEA code
        spline_type="r-smooth",
        degree=3,
        dof=None,
        table_dir="./psea_table_outdir",
        pepsirf_binary="pepsirf"
):    
    start_time = time.perf_couter()

    volcano = ctx.get_action("ps-plot", "volcano")
    zscatter = ctx.get_action("ps-plot", "zscatter")

    assert spline_type in splines.SPLINE_TYPES, \
        f"'{spline_type}' is not a valid spline method!"
    assert not os.path.exists(table_dir), \
        f"'{table_dir}' already exists! Please move or remove this directory."

    os.mkdir(table_dir)

    with open(pairs_file, "r") as fh:
        pairs = [
            tuple(line.replace("\n", "").split("\t"))
            for line in fh.readlines()
        ]
    scores = pd.read_csv(scores_file, sep="\t", index_col=0)
    processed_scores = process_scores(scores, pairs)

    scores_file_split = scores_file.rsplit("/", 1)
    if len(scores_file_split) > 1:
        processed_scores_file = f"transformed_{scores_file_split[1]}"
    else:
        processed_scores_file = f"transformed_{scores_file_split[0]}"

    with tempfile.TemporaryDirectory() as tempdir:
        processed_scores.to_csv(processed_scores_file, sep="\t")
        processed_scores, peptide_sets = utils.remove_peptides(
            processed_scores, peptide_sets_file
        )

        titles = []
        taxa_access = "species_name"
        used_pairs = []
        pair_spline_dict = { "x": list(), "y": list(), "pair": list() }

        if not dof:
            dof = ro.NULL
        if not species_taxa_file:
            taxa_access = "ID"

        for pair in pairs:
            table_prefix = f"{pair[0]}~{pair[1]}"
            print(f"Working on pair ({pair[0]}, {pair[1]})...")
            data_sorted = processed_scores.loc[:, pair].sort_values(by=pair[0])
            x = data_sorted.loc[:, pair[0]].to_numpy()
            y = data_sorted.loc[:, pair[1]].to_numpy()

            # TODO: optimize with a dictionary, if possible
            if spline_type == "py-smooth":
                yfit = splines.smooth_spline(x, y)
            elif spline_type == "cubic":
                yfit = splines.R_SPLINES.cubic_spline(x, y, degree, dof)
            else:
                yfit = splines.R_SPLINES.smooth_spline(x, y)

            maxZ = np.apply_over_axes(
                np.max,
                data_sorted.loc[:, pair],
                1
            )
            maxZ = pd.Series(
                data=[num for elem in maxZ for num in elem],
                index=data_sorted.index
            )
            deltaZ = pd.Series(
                data=y - yfit, index=data_sorted.index
            )
            pair_spline_dict["x"].extend(x.tolist())
            pair_spline_dict["y"].extend(yfit.tolist())
            pair_spline_dict["pair"].extend([table_prefix] * len(x))
            used_pairs.append(pair)

            table = INTERNAL.psea(
                maxZ,
                deltaZ,
                peptide_sets,
                species_taxa_file,
                threshold,
                permutation_num,
                min_size,
                max_size
            )
            with (ro.default_converter + pandas2ri.converter).context():
                table = ro.conversion.get_conversion().rpy2py(table)
            table.to_csv(
                f"{table_dir}/{table_prefix}_psea_table.tsv",
                sep="\t", index=False
            )

            taxa = table.loc[:, taxa_access].to_list()

            titles.append(table_prefix)

        pd.DataFrame(used_pairs).to_csv(
            f"{tempdir}/used_pairs.tsv", sep="\t",
            header=False, index=False
        )
        pd.DataFrame(pair_spline_dict).to_csv(
            f"{tempdir}/spline_data.tsv", sep="\t", index=False
        )

        processed_scores_art = ctx.make_artifact(
            type="FeatureTable[Zscore]",
            view=processed_scores_file,
            view_type=PepsirfContingencyTSVFormat
        )
    
        scatter_plot, = zscatter(
            zscores=processed_scores_art,
            pairs_file=f"{tempdir}/used_pairs.tsv",
            spline_file=f"{tempdir}/spline_data.tsv",
            p_val_access="p.adjust",
            le_peps_access="core_enrichment",
            taxa_access=taxa_access,
            highlight_data=table_dir,
            highlight_threshold=p_val_thresh
        )

        volcano_plot, = volcano(
            xy_dir=table_dir,
            xy_access=["NES", "p.adjust"],
            taxa_access=taxa_access,
            x_threshold=es_thresh,
            y_threshold=p_val_thresh,
            xy_labels=["Enrichment score", "Adjusted p-values"],
            titles=titles
        )

    end_time = time.perf_couter()

    print(f"Finished in {round(end_time-start_time, 2)} seconds")
    
    return scatter_plot, volcano_plot


def process_scores(scores, pairs) -> pd.DataFrame:
    """Grabs replicates specified `pairs` from scores matrix and processes
    those remaining scores

    Returns a Pandas DataFrame of processed Z scores
    """
    base = 2
    offset = 3
    power = pow(base, offset)
    # collect unique replicates from pairs
    reps_list = []
    for pair in pairs:
        for rep in pair:
            reps_list.append(rep)
    reps_list = list(np.unique(reps_list))
    # exclude unused replicates
    processed_scores = scores.loc[:, reps_list]

    processed_scores = processed_scores.apply(lambda row: power + row, axis=0)
    processed_scores = processed_scores.apply(
        lambda row: row.apply(lambda val: 1 if val < 1 else val),
        axis=0
    )
    return processed_scores.apply(
        lambda row: row.apply(lambda val: log(val, base) - offset)
    )