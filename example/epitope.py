# ----------------------------------------------------------------------------
# Copyright (c) 2025-2025, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from biom import Table

from q2_types.feature_table import BIOMV210Format


def create_epitope_map(epitope: pd.DataFrame, collapse:str='Viral') -> pd.DataFrame:
    epitope = _create_EpitopeID_row(epitope, collapse)
    epitope = epitope.reset_index()
    epitope = _create_SpeciesSubtype_row(epitope)

    epitope = \
        epitope.groupby(
            'EpitopeID').agg(list).reset_index()
    epitope['CodeName'] = \
        epitope['CodeName'].transform(lambda x: ';'.join(x))
    epitope['SpeciesSubtype'] = \
        epitope['SpeciesSubtype'].transform(lambda x: ';'.join(x))
    epitope.set_index('EpitopeID', inplace=True)

    def validate_categories(row):
        '''
        Ensure the categories column is actually valid. After aggregating some
        rows will have a list of multiple categories, these should just be the
        same category multiple times.

        1. Ensure that is the case
        2. Make it just a single value not a list
        '''
        

    return epitope


def epitope_zscore(
            zscores: pd.DataFrame,
            epitope_map: pd.DataFrame
        ) -> BIOMV210Format:
    zscores.fillna(value=0, axis=1, inplace=True)
    samples = list(zscores.index)
    observations = list(epitope_map.index)

    data = []
    for _, row in epitope_map.iterrows():
        max_z_scores_per_sample = []
        split_row = row['CodeName'].split(';')

        # Filter the scores dataframe to only include columns corresponding to
        # the peptides we're looking at
        z_scores = zscores.columns[zscores.columns.isin(split_row)]

        for _, row in zscores[z_scores.values].iterrows():
            max_z_scores_per_sample.append(max(row.values, key=abs))

        data.append(max_z_scores_per_sample)

    data = np.array(data)
    table = Table(data, observations, samples)

    result = BIOMV210Format()
    with result.open() as fh:
        table.to_hdf5(fh, generated_by="q2-pepsirf for pepsirf")

    return result


def taxa_to_epitope(
        epitope: pd.DataFrame, collapse: str='Viral') -> pd.DataFrame:
    epitope = _create_EpitopeID_row(epitope, collapse)

    mapped = epitope[['EpitopeID', 'SpeciesID']]
    mapped = mapped.groupby(['SpeciesID'])
    mapped = mapped['EpitopeID'].unique()
    mapped = mapped.reset_index()
    mapped.set_index('SpeciesID', inplace=True)

    return mapped


def _create_EpitopeID_row(epitope, collapse):
    epitope['SpeciesID'] = epitope['SpeciesID'].str.split(';')
    epitope['ClusterID'] = epitope['ClusterID'].fillna('clusterNA')
    epitope['ClusterID'] = epitope['ClusterID'].str.split(';')
    epitope['EpitopeWindow'] = epitope['EpitopeWindow'].fillna('Peptide_NA')
    epitope['EpitopeWindow'] = epitope['EpitopeWindow'].str.split(';')

    epitope = epitope.explode(['SpeciesID', 'ClusterID', 'EpitopeWindow'])
    epitope.drop_duplicates(inplace=True)

    def combine(row):
        if collapse == 'Both' or row['Category'] == collapse:
            return \
                f"{row['SpeciesID']}_{row['ClusterID']}_{row['EpitopeWindow']}"

        return row.name

    epitope['EpitopeID'] = epitope.apply(combine, axis=1)

    return epitope


def _create_SpeciesSubtype_row(epitope):
    epitope['Species'] = epitope['Species'].fillna('speciesNA')
    epitope['Subtype'] = epitope['Subtype'].fillna('subtypeNA')

    def combine(row):
        combined_row = ''
        species = row['Species'].split(';')
        subtype = row['Subtype'].split(';')
        zipped = zip(species, subtype)

        for species, subtype in zipped:
            combined_row += f'{species}:{subtype};'

        return combined_row[:-1]

    epitope['SpeciesSubtype'] = epitope.apply(combine, axis=1)

    return epitope


# Take those peptides and only collapse those to epitope and get subtypes (should be same workflow but on already filtered data)

# NOTE: At the end we are going to run tens out thousands of samples so we will want to make things efficient
def enriched_subtypes(
        scores: pd.DataFrame, subtypes: pd.DataFrame, p_value: float = .05,
        enrichment_score: float = 1, include_negative_enrichment: bool = True,
        split_column: str = None) -> pd.DataFrame:

    if split_column is not None:
        if split_column not in subtypes:
            raise KeyError(
                f"The requested split_column: '{split_column}' does not "
                f"exist. Valid columns are: {list(subtypes.columns)}"
            )

        split_values =

    # species-level peptide-level, species-level epitope-level, subspecies-level peptide-level
    # raise ValueError(subtypes[split_column])
    # Get categories

    scores = pd.concat(list(scores.values()))
    scores = scores.loc[scores['p.adjust'] <= p_value]

    if include_negative_enrichment:
        scores = scores.loc[abs(scores['enrichmentScore'] >= enrichment_score)]
    else:
        scores = scores.loc[scores['enrichmentScore'] >= enrichment_score]

    subtype_counts = {}
    for _, row in scores.iterrows():
        epitopes = row['core_enrichment'].split('/')

        for epitope in epitopes:
            possible_subtypes = \
                subtypes.loc[subtypes['CodeName'].str.contains(epitope)] \
                    ['SpeciesSubtype'].values[0].split(";")

            for possible_subtype in possible_subtypes:
                if possible_subtype not in subtype_counts:
                    subtype_counts[possible_subtype] = 0

                subtype_counts[possible_subtype] += 1

    # TODO: Sort this by values
    subtype_counts = pd.DataFrame({'Counts': subtype_counts.values()},
                                  index=subtype_counts.keys())
    return subtype_counts
