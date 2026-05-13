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


def count_enriched(
            psea_tables: pd.DataFrame,
            zscores: pd.DataFrame,
            processed_zscores: pd.DataFrame,
            peptide_metadata: pd.DataFrame,
            epitope_map: pd.DataFrame = None,
            mapped_zscores: pd.DataFrame = None,
            mapped_processed_zscores: pd.DataFrame = None,
            p_value: float = .05,
            enrichment_score: float = 1,
            include_negative_enrichment: bool = True,
        ) -> pd.DataFrame:
    samples = [pair.split('~')[1] for pair in psea_tables.keys()]

    if epitope_map is not None and not (
            mapped_zscores is not None and
            mapped_processed_zscores is not None):
        raise ValueError("If passing epitope_map you must also pass"
                         " mapped_zscores and mapped_processed_zscores")

    filtered_scores = _filter_scores(
        psea_tables, p_value, enrichment_score, include_negative_enrichment
    )

    counts = _count_enriched_uncollapsed(
        peptide_metadata, zscores, processed_zscores, filtered_scores, samples,
        epitope_map=epitope_map
    )

    if epitope_map is not None:
        counts.update(_count_enriched_collapsed(
            epitope_map, zscores, processed_zscores, mapped_zscores,
            mapped_processed_zscores, filtered_scores
        ))

    return counts


def _filter_scores(scores, p_value, enrichment_score,
                   include_negative_enrichment):
    scores = pd.concat(list(scores.values()))
    scores = scores.loc[scores['p.adjust'] <= p_value]

    if include_negative_enrichment:
        return scores.loc[abs(scores['enrichmentScore']) >= enrichment_score]

    return scores.loc[scores['enrichmentScore'] >= enrichment_score]


# TODO: Uncollapsed here is not split on
# TODO: Sum the enrichment scores based only on the second timepoint in a pair
def _count_enriched_uncollapsed(
            epitope, zscores, processed_zscores, filtered_scores, samples,
            epitope_map=None
        ):
    counts = {
        'uncollapsed_peptide': {},
        'uncollapsed_subtype': {},
    }

    def _count(row):
        enriched_elements = row['core_enrichment'].split('/')
        if epitope_map is not None:
            peptides = []
            for enriched in enriched_elements:
                peptides.extend(epitope_map.loc[enriched]['CodeName'])
            enriched_elements = peptides

        species_id = row.name

        species_name = row.name
        if 'species_name' in row:
            species_name = row['species_name']

        for enriched in enriched_elements:
            subtypes = epitope.loc[enriched]['Subtype'].split(';')
            for subtype in subtypes:
                _count_enriched_uncollapsed_helper(
                    counts, subtype, zscores, processed_zscores, enriched,
                    species_id, species_name, samples
                )

    filtered_scores.apply(_count, axis=1)
    for key, value in counts.items():
        if key == 'uncollapsed_subtype':
            _normalize_and_sum(value)
        df = _create_count_df(key, value)

        if key == 'uncollapsed_subtype':
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
            counts, subtype, zscores, processed_zscores, enriched, species_id,
            species_name, samples
        ):
    if species_id not in counts['uncollapsed_peptide']:
        counts['uncollapsed_peptide'][species_id] = {}

    if species_id not in counts['uncollapsed_subtype']:
        counts['uncollapsed_subtype'][species_id] = {}

    if species_name not in counts['uncollapsed_peptide'][species_id]:
        counts['uncollapsed_peptide'][species_id][species_name] = {}

    if species_name not in counts['uncollapsed_subtype'][species_id]:
        counts['uncollapsed_subtype'][species_id][species_name] = {}

    if enriched not in counts['uncollapsed_peptide'][species_id][species_name]:
        counts['uncollapsed_peptide'][species_id][species_name][enriched] = 1
    else:
        counts['uncollapsed_peptide'][species_id][species_name][enriched] += 1

    enrichment_scores = []
    processed_enrichment_scores = []

    for sample in samples:
        enrichment_scores.append(zscores[enriched][sample])
        processed_enrichment_scores.append(processed_zscores[enriched][sample])

    if subtype not in counts['uncollapsed_subtype'][species_id][species_name]:
        counts['uncollapsed_subtype'][species_id][species_name][subtype] = {
            'Epitope Counts': 1,
            'Relative Enrichment Score': enrichment_scores,
            'Relative Processed Enrichment Score': processed_enrichment_scores
        }
    else:
        counts['uncollapsed_subtype'][species_id][species_name][subtype][
            'Epitope Counts'
        ] += 1
        counts['uncollapsed_subtype'][species_id][species_name][subtype][
            'Relative Enrichment Score'
        ].extend(enrichment_scores)
        counts['uncollapsed_subtype'][species_id][species_name][subtype][
            'Relative Processed Enrichment Score'
        ].extend(processed_enrichment_scores)


def _count_enriched_collapsed(
            epitope_map, zscores, processed_zscores, mapped_zscores,
            mapped_processed_zscores, filtered_scores
        ):
    counts = {
        'collapsed_epitope': {},
        'collapsed_subtype': {},
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
                        counts, species_id, species_name, enriched, subtype
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
                            counts, species_id, species_name, enriched, subtype
                        )

                hits.apply(_count_uncollapsed, axis=1)

    filtered_scores.apply(_count, axis=1)
    for key, value in counts.items():
        df = _create_count_df(key, value)

        if key == 'collapsed_subtype':
            df.columns = ['Epitope Counts']
        else:
            df.columns = ['Subtype Counts']
        counts[key] = df

    return counts


def _count_enriched_collapsed_helper(
            counts, species_id, species_name, epitope, subtype
        ):
    if species_id not in counts['collapsed_epitope']:
        counts['collapsed_epitope'][species_id] = {}

    if species_id not in counts['collapsed_subtype']:
        counts['collapsed_subtype'][species_id] = {}

    if species_name not in counts['collapsed_epitope'][species_id]:
        counts['collapsed_epitope'][species_id][species_name] = {}

    if species_name not in counts['collapsed_subtype'][species_id]:
        counts['collapsed_subtype'][species_id][species_name] = {}

    if epitope not in counts['collapsed_epitope'][species_id][species_name]:
        counts['collapsed_epitope'][species_id][species_name][epitope] = 1
    else:
        counts['collapsed_epitope'][species_id][species_name][epitope] += 1

    if subtype not in counts['collapsed_subtype'][species_id][species_name]:
        counts['collapsed_subtype'][species_id][species_name][subtype] = 1
    else:
        counts['collapsed_subtype'][species_id][species_name][subtype] += 1


def _normalize_and_sum(counts):
    for _, inner in counts.items():
        for _, inner_inner in inner.items():
            for _, final in inner_inner.items():
                res = final['Relative Enrichment Score']
                res_proc = final['Relative Processed Enrichment Score']

                _max = max(res)
                _max_processed = max(res_proc)

                final['Relative Enrichment Score'] = \
                    sum([score / _max for score in res])
                final['Relative Processed Enrichment Score'] = \
                    sum([score / _max_processed for score in res_proc])


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
