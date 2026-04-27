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

    def validate_categories(row):
        '''
        Ensure the categories column is actually valid. After aggregating some
        rows will have a list of multiple categories, these should just be the
        same category multiple times.

        1. Ensure that is the case
        2. Make it just a single value not a list
        '''
        category_set = set(row['Category'])
        if len(category_set) > 1:
            raise ValueError(
                'Collapsed epitope mapped some subtypes to one category and '
                f'some to another. Offending row is: {row}'
            )

        return row['Category'][0]

    epitope['Category'] = epitope.apply(validate_categories, axis=1)

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


def enriched_subtypes(
            scores: pd.DataFrame, subtypes: pd.DataFrame, p_value: float = .05,
            enrichment_score: float = 1,
            include_negative_enrichment: bool = True,
            peptide_library: str = 'IN2'
        ) -> pd.DataFrame:
    default_keys = ['species', 'subspecies', 'species-epitope', ]

    keys = _get_keys(subtypes, "Category", default_keys)

    filtered_scores = _filter_scores(
        scores, p_value, enrichment_score, include_negative_enrichment
    )

    counts = {key: {} for key in keys}

    def _count(row):
        # Get core_enrichement from psea_out
        enriched_elements = row['core_enrichment'].split('/')
        species = row['species_name']

        for enriched in enriched_elements:
            # Determine if we are looking at something that has been collapsed
            # to epitope or not
            if enriched.startswith(peptide_library):
                # uncollapsed
                peptide = enriched
                hits = subtypes.loc[subtypes['CodeName'].apply(
                            lambda peptides: peptide in peptides
                        )]

                def _count_uncollapsed(hit):
                    subtype = hit['Subtype']
                    species_subtype = f'{species}:{subtype}'
                    # NOTE: If this is uncollapsed the epitope will just be the
                    # peptide (if this is a bacterium and bacteria was not
                    # collapsed for instance)
                    epitope = 'uncollapsed'

                    # Since this is uncollapsed, we know we are looking at one
                    # individual peptide. There won't be a list of subtypes and
                    # all that here, so index is 0
                    found_category = _find_split_value(
                        hit, "Category", index=0
                    )
                    _count_enriched(
                        counts, species, species_subtype, epitope,
                        found_category
                    )
                hits.apply(_count_uncollapsed, axis=1)
            else:
                # collapsed
                epitope = enriched
                hit = subtypes.loc[epitope]
                for index, subtype in enumerate(hit['Subtype']):
                    species_subtype = f'{species}:{subtype}'

                    found_category = _find_split_value(
                        hit, "Category", index
                    )
                    _count_enriched(
                        counts, species, species_subtype, epitope,
                        found_category
                    )

    filtered_scores.apply(_count, axis=1)
    for key, value in counts.items():
        _sorted = dict(
            sorted(value.items(), key=lambda item: item[1], reverse=True)
        )
        counts[key] = pd.DataFrame(
            {'Counts': _sorted.values()}, index=_sorted.keys()
        )

    return counts


def _get_keys(subtypes, split_column, default_keys):
    if split_column not in subtypes:
        raise KeyError(
            f"The requested split_column: '{split_column}' does not "
            f"exist. Valid columns are: {list(subtypes.columns)}"
        )

    # If we a split_column our keys need to be the product of the possible
    # split column values and the existing default keys.
    split_values = subtypes[split_column].unique()
    new_keys = []
    for value in split_values:
        for key in default_keys:
            new_keys.append(f'{value}-{key}')

    return new_keys


def _filter_scores(scores, p_value, enrichment_score,
                   include_negative_enrichment):
    scores = pd.concat(list(scores.values()))
    scores = scores.loc[scores['p.adjust'] <= p_value]

    if include_negative_enrichment:
        return scores.loc[abs(scores['enrichmentScore'] >= enrichment_score)]

    return scores.loc[scores['enrichmentScore'] >= enrichment_score]


def _find_split_value(hit, split_column, index):
    found_split_value = ''
    if split_column is not None:
        found_split_value = hit[split_column]
        if isinstance(found_split_value, list):
            found_split_value = found_split_value[index]
        found_split_value += '-'

    return found_split_value


def _count_enriched(counts, species, species_subtype, epitope,
                    found_category):
    # Track species and split value if relevant
    split_species = f'{found_category}species'
    found_split_species = f'{found_category}{species}'

    if found_split_species not in counts[split_species]:
        counts[split_species][found_split_species] = 0
    counts[split_species][found_split_species] += 1

    # Track subspecies and split value if relevant
    split_subtype = f'{found_category}subspecies'
    found_split_subtype = f'{found_category}{species_subtype}'

    if found_split_subtype not in counts[split_subtype]:
        counts[split_subtype][found_split_subtype] = 0
    counts[split_subtype][found_split_subtype] += 1

    # Track species and epitope including split value if relevant
    split_species_epitope = f'{found_category}species-epitope'
    found_split_species_epitope = f'{found_category}{species}-{epitope}'

    if found_split_species_epitope not in counts[split_species_epitope]:
        counts[split_species_epitope][found_split_species_epitope] = 0
    counts[split_species_epitope][found_split_species_epitope] += 1
