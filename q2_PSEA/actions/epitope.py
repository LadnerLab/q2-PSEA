# ----------------------------------------------------------------------------
# Copyright (c) 2025-2025, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from typing import Tuple

import numpy as np
import pandas as pd
from biom import Table

from q2_types.feature_table import BIOMV210Format


def create_epitope_map(
            epitope: pd.DataFrame, collapse: str = 'Viral'
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    epitope = _create_EpitopeID_row(epitope, collapse)
    epitope = epitope.reset_index()

    epitope_map = \
        epitope.groupby(
            'EpitopeID').agg(list).reset_index()
    epitope_map.set_index('EpitopeID', inplace=True)

    peptide_map = \
        epitope.groupby(
            'CodeName').agg(list).reset_index()
    peptide_map = peptide_map[['CodeName', 'EpitopeID']]
    peptide_map.set_index('CodeName', inplace=True)

    return epitope_map, peptide_map


def epitope_zscore(
            zscores: pd.DataFrame,
            epitope_map: pd.DataFrame
        ) -> BIOMV210Format:
    zscores.fillna(value=0, axis=1, inplace=True)
    samples = list(zscores.index)
    observations = list(epitope_map.index)

    data = []

    def get_max_z_scores_per_sample(row):
        max_z_scores_per_sample = []
        sample_zscores = zscores.columns[zscores.columns.isin(row['CodeName'])]

        zscores[sample_zscores.values].apply(
            lambda row: max_z_scores_per_sample.append(
                max(row.values, key=abs)
            ), axis=1
        )

        data.append(max_z_scores_per_sample)

    epitope_map.apply(get_max_z_scores_per_sample, axis=1)

    data = np.array(data)
    table = Table(data, observations, samples)

    result = BIOMV210Format()
    with result.open() as fh:
        table.to_hdf5(fh, generated_by="q2-pepsirf for pepsirf")

    return result


def taxa_to_epitope(epitope: pd.DataFrame) -> pd.DataFrame:
    epitope = epitope.reset_index()
    epitope = epitope[['SpeciesID', 'EpitopeID']]
    epitope = epitope.explode('SpeciesID')
    epitope.drop_duplicates(inplace=True)
    epitope = epitope.rename(
        columns={'SpeciesID': 'term', 'EpitopeID': 'gene'}
    )

    return epitope


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
            residual_threshold: float = 1.0,
            include_negative_enrichment: bool = True,
        ) -> pd.DataFrame:
    if epitope_map is None or peptide_metadata is None:
        return pd.DataFrame()

    filtered_psea_table = psea_table.loc[psea_table['p.adjust'] <= p_value]
    counts = {}

    for core_enrichment in filtered_psea_table['core_enrichment']:
        # Conglomerate all peptides and subtypes here
        for enriched in core_enrichment.split('/'):
            peptides = epitope_map.loc[enriched]['CodeName']

            filtered_map = {}
            for peptide in peptides:
                residual = abs(residuals.loc[peptide]['deltaZ'])
                if abs(residual) >= residual_threshold:
                    filtered_map[peptide] = residual

            if filtered_map == {}:
                continue

            # NOTE: We are normalizing the residuals within epitope then
            # summing them cross epitope, this can cause the score to be higher
            # than the number of peptides
            max_residual = max(filtered_map.values())
            for peptide, residual in filtered_map.items():
                residual = residual / max_residual

                # If there are multiple subtypes they will be ; seperated. The
                # speciesID and species_name cells will contain duplicated ;
                # seperated values. FOr ID and name we can just take the first
                # for subtypes we need to iterate
                speciesID = \
                    peptide_metadata.loc[peptide]['SpeciesID'].split(';')[0]
                species_name = \
                    peptide_metadata.loc[peptide]['Species'].split(';')[0]
                subtypes = peptide_metadata.loc[peptide]['Subtype']

                if subtypes is np.nan:
                    subtypes = ['subtypeNA']
                else:
                    subtypes = subtypes.split(';')

                id_dict = counts.setdefault(speciesID, {})
                name_dict = id_dict.setdefault(species_name, {})

                for subtype in subtypes:
                    subtype_dict = name_dict.setdefault(subtype, {
                        'Peptide Counts': 0,
                        'Relative Enrichment Score': 0
                    })

                    subtype_dict['Peptide Counts'] += 1
                    subtype_dict['Relative Enrichment Score'] += residual

    counts = pd.DataFrame.from_dict(
        {
            (species_id, species_name, epitope_subtype): count
            for species_id, inner in counts.items()
            for species_name, inner_inner in inner.items()
            for epitope_subtype, count in inner_inner.items()
        }, orient='index'
    )
    counts.index = pd.MultiIndex.from_tuples(
        counts.index,
        names=(
            'Species ID Called', 'Species Called', 'Subtypes'
        )
    )

    return counts
