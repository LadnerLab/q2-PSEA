import pandas as pd
import numpy as np


def remove_peptides(scores, peptide_sets) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Removes peptide not present in df formatted sets from a matrix of Z
    scores

    Returns
    -------
    pd.DataFrame
        Contains remaining peptides which were found in the peptide sets file
    """
    pep_list = scores.index.difference(peptide_sets.loc[:, "gene"])
    return scores.drop(index=pep_list), peptide_sets


def filter_peptide_sets(
            psea_table: pd.DataFrame,
            updated_peptide_sets: pd.DataFrame,
            tested_species: set,
            p_value: int,
            enrichment_score: int,
            include_negative_enrichment: bool,
        ) -> tuple[pd.DataFrame, set, bool]:
    psea_table = psea_table.sort_values(by=["p.adjust"], ascending=True)

    sig_found = True
    for _, row in psea_table.iterrows():
        row_id = str(row["ID"])
        if (
            row["p.adjust"] < p_value
            and (abs(row["NES"]) > enrichment_score
                 if include_negative_enrichment
                 else row["NES"] > enrichment_score)
            and row_id not in tested_species
        ):
            all_tested_features = set(row["all_tested_peptides"].split("/"))

            mask = (
                (updated_peptide_sets["term"].astype(str) != row_id)
                & updated_peptide_sets["gene"].isin(all_tested_features)
            )

            updated_peptide_sets = updated_peptide_sets[~mask]
            tested_species.add(row_id)
            break
    else:
        sig_found = False

    return updated_peptide_sets, tested_species, sig_found


def filter_peptide_sets_by_residual(
    peptide_sets: pd.DataFrame,
    deltaZ: pd.Series,
    residual_abs_thresh: float,
    residual_min_peptides: int = 1,
    species_col: str = "term",
    gene_col: str = "gene",
) -> pd.DataFrame:
    if residual_abs_thresh is None:
        return peptide_sets

    df = peptide_sets.copy()
    abs_residuals = deltaZ.abs()
    df["_passes_residual_threshold"] = (
        df[gene_col].map(abs_residuals).fillna(-np.inf)
        > residual_abs_thresh
    )

    keep_species = (
        df.groupby(species_col)["_passes_residual_threshold"]
        .sum()
        .loc[lambda counts: counts >= residual_min_peptides]
        .index
    )

    return df[df[species_col].isin(keep_species)].drop(
        columns="_passes_residual_threshold"
    )


def collate_ae_counts(ae_counts: list[pd.DataFrame]):
    return pd.concat(
        ae_counts, ignore_index=True
    ).groupby('Species', as_index=False)['Events'].sum() \
        .sort_values('Events', ascending=False)
