# ----------------------------------------------------------------------------
# Copyright (c) 2025-2025, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
import numpy as np
import pandas as pd


def create_epitope_map(
            epitope: pd.DataFrame, collapse: str = 'Viral'
        ) -> pd.DataFrame:
    epitope = _create_EpitopeID_row(epitope, collapse)
    epitope = epitope.reset_index()

    epitope_map = \
        epitope.groupby(
            'EpitopeID').agg(list).reset_index()
    epitope_map.set_index('EpitopeID', inplace=True)

    return epitope_map


def taxa_to_epitope(
            peptide_metadata: pd.DataFrame,
            peptide_sets: pd.DataFrame,
            collapse: bool = 'Viral'
        ) -> pd.DataFrame:
    def _get_epitope_id(gmt_row):
        peptide = gmt_row['gene']
        metadata_row = peptide_metadata.loc[peptide]
        if collapse == 'Both' or metadata_row['Category'] == collapse:
            species_id = metadata_row['SpeciesID'].split(';')[0]
            cluster_id = metadata_row['ClusterID'].split(';')[0]
            epitope_window = metadata_row['EpitopeWindow'].split(';')[0]

            return f"{species_id}_{cluster_id}_{epitope_window}"

        return metadata_row.name

    peptide_sets['gene'] = peptide_sets.apply(_get_epitope_id, axis=1)
    peptide_sets.drop_duplicates(inplace=True, ignore_index=True)
    return peptide_sets


def _create_EpitopeID_row(epitope, collapse):
    epitope['Species'] = epitope['Species'].str.split(';')
    epitope['Subtype'] = epitope['Subtype'].str.split(';')
    epitope['SpeciesID'] = epitope['SpeciesID'].str.split(';')
    epitope['ClusterID'] = epitope['ClusterID'].str.split(';')
    epitope['EpitopeWindow'] = epitope['EpitopeWindow'].str.split(';')

    epitope = epitope.explode(
        ['Species', 'Subtype', 'SpeciesID', 'ClusterID', 'EpitopeWindow']
    )

    epitope['Subtype'] = epitope['Subtype'].fillna('subtypeNA')
    # Happens because some subtype rows have an empty value along with valid
    # values indicating one or more missing
    epitope['Subtype'] = epitope['Subtype'].replace('', 'subtypeNA')
    epitope['ClusterID'] = epitope['ClusterID'].fillna('clusterNA')
    epitope['EpitopeWindow'] = epitope['EpitopeWindow'].fillna('Peptide_NA')

    epitope.drop_duplicates(inplace=True)

    def combine(row):
        if collapse == 'Both' or row['Category'] == collapse:
            return \
                f"{row['SpeciesID']}_{row['ClusterID']}_{row['EpitopeWindow']}"

        return row.name

    epitope['EpitopeID'] = epitope.apply(combine, axis=1)

    return epitope


# Take list collapsed epitopes
# Go to metadata and get associated peptides
# Get residuals for those peptides if above some threshold we count it
# Then normalize all residuals against the abs val of max residual for epitope
# then sum them across epitopes
# Output this one file per pair
def count_enriched(
            psea_table: pd.DataFrame,
            residuals: pd.DataFrame,
            peptide_metadata: pd.DataFrame = None,
            epitope_map: pd.DataFrame = None,
            p_value: float = .05,
            residual_threshold: float = 0.5,
            include_negative_enrichment: bool = True,
        ) -> pd.DataFrame:
    # Short circuit if we weren't collapsed
    if peptide_metadata is None or epitope_map is None:
        return pd.DataFrame()

    filtered_psea_table = psea_table.loc[psea_table['p.adjust'] <= p_value]
    counts = {}

    for core_enrichment in filtered_psea_table['core_enrichment']:
        # Conglomerate all peptides and subtypes here
        for enriched in core_enrichment.split('/'):
            peptides = epitope_map.loc[enriched]['CodeName']

            # NOTE: In theory maybe we can get rid of this loop and normalize
            # post facto; however, because we are normalizing within epitopes
            # but summing cross epitope this will get needlessly fiddly.
            peptide_to_residual = {}
            max_residual = 0
            for peptide in peptides:
                residual = abs(residuals.loc[peptide]['deltaZ']) if \
                    include_negative_enrichment else \
                    residuals.loc[peptide]['deltaZ']

                if residual > max_residual:
                    max_residual = residual

                peptide_to_residual[peptide] = residual

            # NOTE: If we really need to optimize, try excepting here would be
            # faster than if/else. Probably negligible though.
            for peptide, residual in peptide_to_residual.items():
                # Get all speciesIDs for this peptide
                speciesIDs = peptide_metadata.loc[peptide]['SpeciesID']
                if isinstance(speciesIDs, str):
                    speciesIDs = speciesIDs.split(';')
                else:
                    speciesIDs = (speciesIDs,)

                # Get all species names for this peptide
                species_names = peptide_metadata.loc[peptide]['Species']
                if isinstance(species_names, str):
                    species_names = species_names.split(';')
                else:
                    species_names = (species_names,)

                # Get all subtypes for this peptide
                subtypes = peptide_metadata.loc[peptide]['Subtype']
                if subtypes is np.nan:
                    subtypes = tuple(['subtypeNA'] * len(speciesIDs))
                else:
                    if isinstance(subtypes, str):
                        subtypes = subtypes.split(';')
                    else:
                        subtypes = (subtypes,)

                # Assert we have the same number of IDs, names, and subtypes
                assert len(speciesIDs) == len(species_names)
                assert len(species_names) == len(subtypes)

                # Normalize the residual before we start summing
                normalized_residual = residual / max_residual

                # Iterate over all unique subtypes
                for speciesID, species_name, subtype in zip(
                            speciesIDs, species_names, subtypes
                        ):
                    # Get the nested dicts for this speciesID and name
                    id_dict = counts.setdefault(speciesID, {})
                    name_dict = id_dict.setdefault(species_name, {})

                    # Now count the subtype
                    subtype_dict = name_dict.setdefault(subtype, {
                        # This is a set so we can take the length and get the
                        # number of unique peptides easily later
                        'Peptide Counts': set(),
                        'Relative Enrichment Score': 0
                    })

                    # Only count the peptide if its residual is above given
                    # threshold
                    if residual > residual_threshold:
                        subtype_dict['Peptide Counts'].add(peptide)

                    # Add the residual to the enrichment score regardless
                    subtype_dict[
                        'Relative Enrichment Score'
                    ] += normalized_residual

    counts = pd.DataFrame.from_dict(
        {
            (species_id, species_name, subtype): {
                'Peptide Counts': len(count['Peptide Counts']),
                'Relative Enrichment Score': count['Relative Enrichment Score']
            }
            for species_id, inner in counts.items()
            for species_name, inner_inner in inner.items()
            for subtype, count in inner_inner.items()
        }, orient='index'
    )
    counts.index = pd.MultiIndex.from_tuples(
        counts.index,
        names=(
            'Species ID Called', 'Species Called', 'Subtypes'
        )
    )

    return counts
