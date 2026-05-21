import pandas as pd


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
            epitope_map: pd.DataFrame = None,
            peptide_map: pd.DataFrame = None,
        ) -> tuple[pd.DataFrame, set, bool]:
    psea_table = psea_table.sort_values(by=["p.adjust"], ascending=True)\

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

            if peptide_map is not None:
                all_tested_features = _get_mapped_features(
                    epitope_map, peptide_map, all_tested_features
                )

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


def _get_mapped_features(epitope_map, peptide_map, all_tested_features):
    """
    This function is only run if we are using epitope mapping. An epitope maps
    to one peptide and one species; however, multiple epitopes from multiple
    species can map to the same peptide. We need to map epitopes we hit back
    to peptides so we can get all the epitopes that map to that peptide.

    Parameters
    ----------
    epitope_map : pd.DataFrame
        Maps epitopes to peptides.
    peptide_map : pd.DataFrame
        Maps peptides to epitopes.
    all_tested_features : set[str]
        A set of all features, epitopes or peptides that have been tested so
        far.

    Returns
    -------
    set[str]
        All features the peptide we tested map to
    """
    expanded = set()

    for tested_feature in all_tested_features:
        codenames = epitope_map.loc[tested_feature, 'CodeName']
        for codename in codenames:
            expanded = expanded.union(
                set(peptide_map.loc[codename, 'EpitopeID'])
            )

    return expanded


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

    peptide_residuals.update(epitope_residuals)
    return pd.Series(peptide_residuals)
