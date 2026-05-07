import pandas as pd
import rpy2.robjects as ro
import qiime2

from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr


cluster_profiler = importr("clusterProfiler")


def generate_metadata(replicates):
    """

    Parameters
    ----------

    Returns
    -------
    """
    base_reps = []

    replicates.sort()

    for replicate in replicates:
        base_seq_name = replicate.split("_")[2]
        base_reps.append(base_seq_name)

    meta_series = pd.Series(data=base_reps, index=replicates)
    meta_series.index.name = "sample-id"
    meta_series.name = "source"

    return qiime2.metadata.CategoricalMetadataColumn(meta_series)


def make_metadata(df, length):
    """Given a Pandas DataFrame, and the length of columns, returns Qiime2
    Metadata
    """
    indexes = []
    for i in range(length):
        indexes.append(f"{i}")
    df.index = indexes
    df.index.name = "sample-id"
    return qiime2.Metadata(df)


def save_taxa_leading_peps_file(
            taxa_peps_filepath,
            taxa,
            leading_peps
        ) -> None:
    """

    Parameters
    ----------

    Returns
    -------
    """
    with open(taxa_peps_filepath, "w") as fh:
        for i in range(len(taxa)):
            fh.write(
                taxa[i] + "\t" + leading_peps[i].replace("/", "\t") + "\n"
            )


def remove_peptides_in_csv_format(
        scores,
        peptide_sets_file
) -> (pd.DataFrame, pd.DataFrame):
    """Removes peptides not present in CSV formatted sets file from a matrix of
    Z scores

    Returns
    -------
    pd.DataFrame
        Contains remaining peptides which were found in the peptide sets file
    """
    peptide_sets = pd.read_csv(peptide_sets_file, sep=",")
    pep_list = scores.index.difference(peptide_sets.loc[:, "gene"])
    return scores.drop(index=pep_list), peptide_sets


def remove_peptides_in_gmt_format(scores, peptide_sets_file) -> pd.DataFrame:
    """Removes peptides not present in GMT formatted sets file from a matrix of
    Z scores

    Returns
    -------
    pd.DataFrame
        Contains remaining peptides which were found in the peptide sets file
    """
    read_gmtr = ro.r["read.gmt"]
    with (ro.default_converter + pandas2ri.converter).context():
        peptide_sets = read_gmtr(peptide_sets_file)
    pep_list = scores.index.difference(peptide_sets.loc[:, "gene"])
    return scores.drop(index=pep_list), peptide_sets


def remove_peptides_in_tsv_format(scores, peptide_sets_file) -> pd.DataFrame:
    """Removes peptide not present in TSV formatted sets file from a matrix of
    Z scores

    Returns
    -------
    pd.DataFrame
        Contains remaining peptides which were found in the peptide sets file
    """
    peptide_sets = pd.read_csv(peptide_sets_file, sep="\t")
    pep_list = scores.index.difference(peptide_sets.loc[:, "gene"])
    return scores.drop(index=pep_list), peptide_sets


def remove_peptides_in_df_format(scores, peptide_sets) -> pd.DataFrame:
    """Removes peptide not present in df formatted sets from a matrix of Z
    scores

    Returns
    -------
    pd.DataFrame
        Contains remaining peptides which were found in the peptide sets file
    """
    pep_list = scores.index.difference(peptide_sets.loc[:, "gene"])
    return scores.drop(index=pep_list), peptide_sets


REMOVE_PEPTIDES_SWITCH = {
    "csv": remove_peptides_in_csv_format,
    "gmt": remove_peptides_in_gmt_format,
    "tsv": remove_peptides_in_tsv_format,
    "df": remove_peptides_in_df_format,
}


def remove_peptides(scores, peptide_sets) -> (pd.DataFrame, pd.DataFrame):
    """Provides an interface to abstract support for TSV, CSV, and GMT file
    formats

    Notes
    -----
    * TSV and CSV file formats are basically the same but use tabs and commas,
      respectively

    Returns
    -------
    pd.DataFrame
        DataFrame from processing
    """
    if isinstance(peptide_sets, pd.DataFrame):
        format = "df"
    else:
        format = peptide_sets.split(".")[1]
    assert format in list(REMOVE_PEPTIDES_SWITCH), \
        f"'{format}' is not a supported format for the peptide sets file!"
    return REMOVE_PEPTIDES_SWITCH[format](scores, peptide_sets)


def collapse_residuals_to_epitope(peptide_residuals, epitope_map):
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
