# ----------------------------------------------------------------------------
# Copyright (c) 2025-2025, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd
from biom import Table

from q2_types.feature_table import BIOMV210Format


def create_epitope_map(
            epitope: pd.DataFrame, collapse: str = 'Viral'
        ) -> pd.DataFrame:
    epitope = _create_EpitopeID_row(epitope, collapse)
    epitope = epitope.reset_index()

    epitope = \
        epitope.groupby(
            'EpitopeID').agg(list).reset_index()
    epitope.set_index('EpitopeID', inplace=True)

    # def validate_categories(row):
    #     '''
    #     Ensure the categories column is actually valid. After aggregating some
    #     rows will have a list of multiple categories, these should just be the
    #     same category multiple times.

    #     1. Ensure that is the case
    #     2. Make it just a single value not a list
    #     '''
    #     category_set = set(row['Category'])
    #     if len(category_set) > 1:
    #         raise ValueError(
    #             'Collapsed epitope mapped some subtypes to one category and '
    #             f'some to another. Offending row is: {row}'
    #         )

    #     return row['Category'][0]

    # epitope['Category'] = epitope.apply(validate_categories, axis=1)

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

        # Filter the scores dataframe to only include columns corresponding to
        # the peptides we're looking at
        z_scores = zscores.columns[zscores.columns.isin(row['CodeName'])]

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
            epitope: pd.DataFrame, collapse: str = 'Viral'
        ) -> pd.DataFrame:
    mapped = _create_EpitopeID_row(epitope, collapse)
    mapped = mapped.reset_index()
    mapped = mapped[['SpeciesID', 'EpitopeID']]
    # This matches the spec .gmt files are read into in q2-PSEA
    mapped = mapped.rename(columns={'SpeciesID': 'term', 'EpitopeID': 'gene'})

    return mapped


def _create_EpitopeID_row(epitope, collapse):
    epitope['Species'] = epitope['Species'].str.split(';')
    epitope['Subtype'] = epitope['Subtype'].str.split(';')
    epitope['SpeciesID'] = epitope['SpeciesID'].str.split(';')
    epitope['ClusterID'] = epitope['ClusterID'].str.split(';')
    epitope['EpitopeWindow'] = epitope['EpitopeWindow'].str.split(';')

    # TODO: This is likely quite inefficient, do in a better way
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


# TODO: Need to fix the formatting so the indices aren't duplicated in output
# files, This will need to happen in transformation in both directions. df to
# tsv we need to dedup. tsv to DataFrame need to redup so pd can read it
def enriched_subtypes(
            scores: pd.DataFrame,
            subtypes: pd.DataFrame,
            p_value: float = .05,
            enrichment_score: float = 1,
            include_negative_enrichment: bool = True,
            peptide_library: str = 'IN2'
        ) -> pd.DataFrame:
    filtered_scores = _filter_scores(
        scores, p_value, enrichment_score, include_negative_enrichment
    )

    counts = {
        'epitope': {},
        'subtype': {},
    }

    def _count(row):
        enriched_elements = row['core_enrichment'].split('/')
        species_id = row.name
        species_name = row['species_name']

        for enriched in enriched_elements:
            if enriched.startswith(peptide_library):
                # Here we are uncollapsed which means we are looking at an
                # individual peptide
                hits = subtypes.loc[subtypes['CodeName'].apply(
                    lambda peptides: enriched in peptides
                )]

                def _count_uncollapsed(hit):
                    for subtype in hit['Subtype']:
                        _count_enriched(
                            counts, species_id, species_name, enriched, subtype
                        )

                hits.apply(_count_uncollapsed, axis=1)
            else:
                # Here we are collapsed which means we are looking at an
                # epitope
                hit = subtypes.loc[enriched]
                for subtype in hit['Subtype']:
                    _count_enriched(
                        counts, species_id, species_name, enriched, subtype
                    )

    filtered_scores.apply(_count, axis=1)
    for key, value in counts.items():
        df = pd.DataFrame.from_dict(
            {(species_id, species_name, epitope_subtype): count
              for species_id, inner in value.items()
              for species_name, inner_inner in inner.items()
              for epitope_subtype, count in inner_inner.items()
            },
            orient='index'
        )
        df.index = pd.MultiIndex.from_tuples(
            df.index,
            names=(
                'Species ID Called', 'Species Called', key.capitalize() + 's'
            )
        )

        df.columns = ['Counts']
        counts[key] = df

    return counts


def _filter_scores(scores, p_value, enrichment_score,
                   include_negative_enrichment):
    scores = pd.concat(list(scores.values()))
    scores = scores.loc[scores['p.adjust'] <= p_value]

    if include_negative_enrichment:
        return scores.loc[abs(scores['enrichmentScore']) >= enrichment_score]

    return scores.loc[scores['enrichmentScore'] >= enrichment_score]


def _count_enriched(counts, species_id, species_name, epitope, subtype):
    # On these first two levels, epitope and subtype are the same
    if species_id not in counts['epitope']:
        counts['epitope'][species_id] = {}
        counts['subtype'][species_id] = {}

    if species_name not in counts['epitope'][species_id]:
        counts['epitope'][species_id][species_name] = {}
        counts['subtype'][species_id][species_name] = {}

    if epitope not in counts['epitope'][species_id][species_name]:
        counts['epitope'][species_id][species_name][epitope] = 1
    else:
        counts['epitope'][species_id][species_name][epitope] += 1

    if subtype not in counts['subtype'][species_id][species_name]:
        counts['subtype'][species_id][species_name][subtype] = 1
    else:
        counts['subtype'][species_id][species_name][subtype] += 1
