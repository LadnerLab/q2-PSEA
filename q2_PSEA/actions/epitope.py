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

    return epitope


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
        # TODO: This blows up if the psea tables were created without
        # species_to_taxa
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

        # TODO: This explodes if nothing passed the filter
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
        counts['subtype'][species_id][species_name][subtype] = 1
    else:
        counts['subtype'][species_id][species_name][subtype] += 1
