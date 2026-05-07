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
    # This matches the spec .gmt files are read into in q2-PSEA
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


def count_enriched(
            psea_tables: pd.DataFrame,
            zscores: pd.DataFrame,
            processed_zscores: pd.DataFrame,
            peptide_metadata: pd.DataFrame = None,
            epitope_map: pd.DataFrame = None,
            mapped_zscores: pd.DataFrame = None,
            mapped_processed_zscores: pd.DataFrame = None,
            p_value: float = .05,
            enrichment_score: float = 1,
            include_negative_enrichment: bool = True,
        ) -> pd.DataFrame:
    if (peptide_metadata is not None and epitope_map is not None) or \
            (peptide_metadata is None and epitope_map is None):
        raise ValueError(
            "Please pass one and only one of eptiope and epitope_map"
        )

    if epitope_map is not None and not (
            mapped_zscores is not None and
            mapped_processed_zscores is not None):
        raise ValueError("If passing epitope_map you must also pass"
                         " mapped_zscores and mapped_processed_zscores")

    filtered_scores = _filter_scores(
        psea_tables, p_value, enrichment_score, include_negative_enrichment
    )

    if epitope_map is not None:
        counts = _count_enriched_collapsed(
            epitope_map, zscores, processed_zscores, mapped_zscores,
            mapped_processed_zscores, filtered_scores
        )
    else:
        counts = _count_enriched_uncollapsed(
            peptide_metadata, zscores, processed_zscores, filtered_scores
        )

    return counts


def _filter_scores(scores, p_value, enrichment_score,
                   include_negative_enrichment):
    scores = pd.concat(list(scores.values()))
    scores = scores.loc[scores['p.adjust'] <= p_value]

    if include_negative_enrichment:
        return scores.loc[abs(scores['enrichmentScore']) >= enrichment_score]

    return scores.loc[scores['enrichmentScore'] >= enrichment_score]


def _count_enriched_uncollapsed(
            epitope, zscores, processed_zscores, filtered_scores
        ):
    counts = {
        'peptide': {},
        'subtype': {},
    }

    def _count(row):
        enriched_elements = row['core_enrichment'].split('/')
        species_id = row.name

        species_name = row.name
        if 'species_name' in row:
            species_name = row['species_name']

        for enriched in enriched_elements:
            _count_enriched_uncollapsed_helper(
                counts, epitope, zscores, processed_zscores, enriched,
                species_id, species_name
            )

    filtered_scores.apply(_count, axis=1)
    for key, value in counts.items():
        df = _create_count_df(key, value)

        if key == 'subtype':
            df.columns = [
                'Peptide Counts',
                'Relative Enrichment Score',
                'Processed Relative Enrichment Score'
            ]
        else:
            df.columns = ['Subtype Counts']
        counts[key] = df

    return counts


def _count_enriched_uncollapsed_helper(
            counts, epitope, zscores, processed_zscores, enriched, species_id,
            species_name
        ):
    subtype = epitope.loc[enriched]['Subtype']

    if species_id not in counts['peptide']:
        counts['peptide'][species_id] = {}

    if species_id not in counts['subtype']:
        counts['subtype'][species_id] = {}

    if species_name not in counts['peptide'][species_id]:
        counts['peptide'][species_id][species_name] = {}

    if species_name not in counts['subtype'][species_id]:
        counts['subtype'][species_id][species_name] = {}

    if enriched not in counts['peptide'][species_id][species_name]:
        counts['peptide'][species_id][species_name][enriched] = 1
    else:
        counts['peptide'][species_id][species_name][enriched] += 1

    if subtype not in counts['subtype'][species_id][species_name]:
        counts['subtype'][species_id][species_name][subtype] = {
            'Epitope Counts': 1,
            'Relative Enrichment Score': sum(zscores[enriched]),
            'Relative Processed Enrichment Score':
                sum(processed_zscores[enriched]),
        }
    else:
        counts['subtype'][species_id][species_name][subtype][
            'Epitope Counts'
        ] += 1


def _count_enriched_collapsed(
            epitope_map, zscores, processed_zscores, mapped_zscores,
            mapped_processed_zscores, filtered_scores
        ):
    counts = {
        'epitope': {},
        'subtype': {},
    }

    def _count(row):
        enriched_elements = row['core_enrichment'].split('/')
        species_id = row.name

        species_name = row.name
        if 'species_name' in row:
            species_name = row['species_name']

        for enriched in enriched_elements:
            if 'Peptide' in enriched:
                # Here we are collapsed which means we are looking at an
                # epitope
                hit = epitope_map.loc[enriched]
                for subtype in hit['Subtype']:
                    _count_enriched_collapsed_helper(
                        counts, zscores, epitope_map, processed_zscores,
                        mapped_zscores, mapped_processed_zscores, species_id,
                        species_name, enriched, subtype
                    )
            else:
                # Here we are uncollapsed which means we are looking at an
                # individual peptide
                hits = epitope_map.loc[epitope_map['CodeName'].apply(
                    lambda peptides: enriched in peptides
                )]

                def _count_uncollapsed(hit):
                    for subtype in hit['Subtype']:
                        _count_enriched_collapsed_helper(
                            counts, zscores, epitope_map, processed_zscores,
                            mapped_zscores, mapped_processed_zscores,
                            species_id, species_name, enriched, subtype
                        )

                hits.apply(_count_uncollapsed, axis=1)

    filtered_scores.apply(_count, axis=1)
    for key, value in counts.items():
        df = _create_count_df(key, value)

        if key == 'subtype':
            df.columns = [
                'Epitope Counts',
                'Relative Enrichment Score',
                'Processed Relative Enrichment Score'
            ]
        else:
            df.columns = ['Subtype Counts']
        counts[key] = df

    return counts


def _count_enriched_collapsed_helper(
            counts, zscores, epitope_map, processed_zscores, mapped_zscores,
            mapped_processed_zscores, species_id, species_name, epitope,
            subtype
        ):
    if species_id not in counts['epitope']:
        counts['epitope'][species_id] = {}

    if species_id not in counts['subtype']:
        counts['subtype'][species_id] = {}

    if species_name not in counts['epitope'][species_id]:
        counts['epitope'][species_id][species_name] = {}

    if species_name not in counts['subtype'][species_id]:
        counts['subtype'][species_id][species_name] = {}

    if epitope not in counts['epitope'][species_id][species_name]:
        counts['epitope'][species_id][species_name][epitope] = 1
    else:
        counts['epitope'][species_id][species_name][epitope] += 1

    if subtype not in counts['subtype'][species_id][species_name]:
        counts['subtype'][species_id][species_name][subtype] = {
            'Epitope Counts': 1,
            'Relative Enrichment Score': _get_relative_enrichment_score(
                zscores,
                mapped_zscores,
                epitope,
                epitope_map
            ),
            'Relative Processed Enrichment Score':
                _get_relative_enrichment_score(
                    processed_zscores,
                    mapped_processed_zscores,
                    epitope,
                    epitope_map
                ),
        }
    else:
        counts['subtype'][species_id][species_name][subtype][
            'Epitope Counts'
        ] += 1


def _get_relative_enrichment_score(
            zscores,
            mapped_zscores,
            feature,
            epitope_map
        ):
    max_z_scores = mapped_zscores[feature]
    # The zscore matrix is keyed on peptides, but the subtypes here map 1 to 1
    # to peptides
    subtypes = epitope_map.loc[feature]['CodeName']

    normalized_zscores = []

    for subtype in subtypes:
        z_scores = zscores[subtype]

        for max_z_score, zscore in zip(max_z_scores, z_scores):
            normalized_zscores.append(zscore / max_z_score)

    relative_enrichment_score = sum(max_z_scores)
    return relative_enrichment_score


def _create_count_df(key, value):
    df = pd.DataFrame.from_dict(
        {
            (species_id, species_name, epitope_subtype): count
            for species_id, inner in value.items()
            for species_name, inner_inner in inner.items()
            for epitope_subtype, count in inner_inner.items()
        }, orient='index'
    )
    df.index = pd.MultiIndex.from_tuples(
        df.index,
        names=(
            'Species ID Called', 'Species Called', key.capitalize() + 's'
        )
    )

    return df
