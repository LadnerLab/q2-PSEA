# ----------------------------------------------------------------------------
# Copyright (c) 2025-2025, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import unittest

import numpy as np
import pandas as pd
from biom import load_table
from q2_types.feature_table import BIOMV210Format
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.actions.epitope import (
    _count_enriched,
    _filter_scores,
    _find_split_value,
    _get_keys,
    create_epitope_map,
    enriched_subtypes,
    epitope_zscore,
    taxa_to_epitope,
)


# ---------------------------------------------------------------------------
# create_epitope_map
# ---------------------------------------------------------------------------

class TestCreateEpitopeMap(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.epitope = pd.read_csv(
            self.get_data_path("epitope.tsv"), sep="\t", index_col=0
        )

    def test_returns_dataframe(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertIsInstance(result, pd.DataFrame)

    def test_index_name_is_epitope_id(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertEqual(result.index.name, "EpitopeID")

    def test_viral_collapse_creates_combined_ids(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertIn("sp001_C1_W1", result.index)
        self.assertIn("sp002_C2_W2", result.index)

    def test_non_collapsed_category_keeps_original_index(self):
        # pep_03 is Bacterial; with collapse='Viral' it is not collapsed, so
        # its EpitopeID stays as the original row name.
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertIn("pep_03", result.index)

    def test_codenames_aggregated_as_list(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        self.assertIsInstance(result.loc["sp001_C1_W1", "CodeName"], list)

    def test_two_peptides_share_same_epitope(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Viral")
        codenames = result.loc["sp001_C1_W1", "CodeName"]
        self.assertEqual(set(codenames), {"pep_00", "pep_01"})

    def test_both_collapse_includes_bacterial(self):
        result = create_epitope_map(self.epitope.copy(), collapse="Both")
        self.assertIn("sp003_C3_W3", result.index)

    def test_category_conflict_raises_value_error(self):
        # Two rows map to the same EpitopeID but carry different Category
        # values.
        bad = pd.DataFrame(
            {
                "SpeciesID": ["sp001", "sp001"],
                "ClusterID": ["C1", "C1"],
                "EpitopeWindow": ["W1", "W1"],
                "Species": ["InfluenzaA", "EColi"],
                "Subtype": ["H1N1", "K12"],
                "Category": ["Viral", "Bacterial"],
            },
            index=pd.Index(["pep_A", "pep_B"], name="CodeName"),
        )
        with self.assertRaises(ValueError):
            create_epitope_map(bad, collapse="Both")

    def test_semicolon_separated_entries_exploded(self):
        # A single row with two species joined by ';' should yield two
        # EpitopeIDs.
        multi = pd.DataFrame(
            {
                "SpeciesID": ["sp001;sp002"],
                "ClusterID": ["C1;C2"],
                "EpitopeWindow": ["W1;W2"],
                "Species": ["InfluenzaA;InfluenzaB"],
                "Subtype": ["H1N1;Yamagata"],
                "Category": ["Viral"],
            },
            index=pd.Index(["pep_X"], name="CodeName"),
        )
        result = create_epitope_map(multi, collapse="Viral")
        self.assertIn("sp001_C1_W1", result.index)
        self.assertIn("sp002_C2_W2", result.index)


# ---------------------------------------------------------------------------
# epitope_zscore
# ---------------------------------------------------------------------------

class TestEpitopeZscore(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        scores = pd.read_csv(
            self.get_data_path("scores.tsv"), sep="\t", index_col=0
        )
        # epitope_zscore expects samples as rows, peptides as columns
        self.zscores = scores.T
        self.epitope_map = pd.DataFrame(
            {
                "CodeName": {
                    "ep1": ["pep_00", "pep_01"],
                    "ep2": ["pep_02", "pep_03"],
                    "ep3": ["pep_04"],
                }
            }
        )

    def test_returns_biomv210_format(self):
        result = epitope_zscore(self.zscores.copy(), self.epitope_map)
        self.assertIsInstance(result, BIOMV210Format)

    def test_observations_match_epitope_ids(self):
        result = epitope_zscore(self.zscores.copy(), self.epitope_map)
        table = load_table(str(result))
        self.assertEqual(
            set(table.ids(axis="observation")), {"ep1", "ep2", "ep3"}
        )

    def test_samples_match_input_index(self):
        result = epitope_zscore(self.zscores.copy(), self.epitope_map)
        table = load_table(str(result))
        self.assertEqual(
            set(table.ids(axis="sample")), set(self.zscores.index)
        )

    def test_max_abs_zscore_selected_per_sample(self):
        # ep1 covers pep_00 (sA=0.5) and pep_01 (sA=1.0) → max abs for sA = 1.0
        result = epitope_zscore(self.zscores.copy(), self.epitope_map)
        table = load_table(str(result))
        val = table.get_value_by_ids("ep1", "sA")
        self.assertAlmostEqual(abs(val), 1.0)

    def test_nan_values_filled_to_zero(self):
        zscores_nan = self.zscores.copy().astype(float)
        zscores_nan.iloc[0, 0] = np.nan
        # Must not raise; NaN is replaced with 0 before computation.
        result = epitope_zscore(zscores_nan, self.epitope_map)
        self.assertIsInstance(result, BIOMV210Format)


# ---------------------------------------------------------------------------
# taxa_to_epitope
# ---------------------------------------------------------------------------

class TestTaxaToEpitope(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.epitope = pd.read_csv(
            self.get_data_path("epitope.tsv"), sep="\t", index_col=0
        )

    def test_returns_dataframe(self):
        result = taxa_to_epitope(self.epitope.copy(), collapse="Viral")
        self.assertIsInstance(result, pd.DataFrame)

    def test_columns_are_term_and_gene(self):
        result = taxa_to_epitope(self.epitope.copy(), collapse="Viral")
        self.assertListEqual(list(result.columns), ["term", "gene"])

    def test_term_contains_species_ids(self):
        result = taxa_to_epitope(self.epitope.copy(), collapse="Viral")
        self.assertIn("sp001", result["term"].values)

    def test_gene_contains_viral_epitope_ids(self):
        result = taxa_to_epitope(self.epitope.copy(), collapse="Viral")
        self.assertIn("sp001_C1_W1", result["gene"].values)

    def test_bacterial_gene_is_original_index_when_collapse_viral(self):
        # pep_03 (Bacterial) is not collapsed → EpitopeID = original row name
        result = taxa_to_epitope(self.epitope.copy(), collapse="Viral")
        self.assertIn("pep_03", result["gene"].values)

    def test_both_collapse_gives_bacterial_epitope_id(self):
        result = taxa_to_epitope(self.epitope.copy(), collapse="Both")
        self.assertIn("sp003_C3_W3", result["gene"].values)


# ---------------------------------------------------------------------------
# _filter_scores
# ---------------------------------------------------------------------------

class TestFilterScores(TestPluginBase):
    package = "q2_PSEA.tests"

    def _scores(self, rows):
        return {"pair1": pd.DataFrame(rows)}

    def test_filters_out_rows_above_p_value(self):
        scores = self._scores([
            {"p.adjust": 0.01, "enrichmentScore": 2.0},
            {"p.adjust": 0.5, "enrichmentScore": 2.0},
        ])
        result = _filter_scores(scores, 0.05, 1.0, True)
        self.assertEqual(len(result), 1)

    def test_negative_enrichment_excluded_regardless_of_flag(self):
        # _filter_scores contains a known quirk: abs() is applied to the
        # boolean result of (enrichmentScore >= threshold), not to the score
        # itself.  This means negative scores are always excluded by the
        # threshold check, even when include_negative_enrichment=True.
        scores = self._scores([
            {"p.adjust": 0.01, "enrichmentScore": -2.0},
        ])
        result = _filter_scores(scores, 0.05, 1.0, True)
        self.assertEqual(len(result), 0)

    def test_excludes_negative_enrichment_when_flag_false(self):
        scores = self._scores([
            {"p.adjust": 0.01, "enrichmentScore": -2.0},
        ])
        result = _filter_scores(scores, 0.05, 1.0, False)
        self.assertEqual(len(result), 0)

    def test_concatenates_multiple_pair_dicts(self):
        scores = {
            "pairA": pd.DataFrame(
                [{"p.adjust": 0.01, "enrichmentScore": 2.0}]
            ),
            "pairB": pd.DataFrame(
                [{"p.adjust": 0.01, "enrichmentScore": 2.0}]
            ),
        }
        result = _filter_scores(scores, 0.05, 1.0, True)
        self.assertEqual(len(result), 2)

    def test_all_rows_filtered_returns_empty(self):
        scores = self._scores([
            {"p.adjust": 0.9, "enrichmentScore": 0.0},
        ])
        result = _filter_scores(scores, 0.05, 1.0, True)
        self.assertEqual(len(result), 0)


# ---------------------------------------------------------------------------
# _get_keys
# ---------------------------------------------------------------------------

class TestGetKeys(TestPluginBase):
    package = "q2_PSEA.tests"

    def _subtypes(self, values):
        return pd.DataFrame({"Category": values})

    def test_returns_product_of_split_values_and_default_keys(self):
        subtypes = self._subtypes(["Viral", "Bacterial"])
        result = _get_keys(subtypes, "Category", ["species", "subspecies"])
        self.assertIn("Viral-species", result)
        self.assertIn("Bacterial-subspecies", result)
        self.assertEqual(len(result), 4)  # 2 values × 2 keys

    def test_raises_key_error_for_missing_column(self):
        subtypes = self._subtypes(["Viral"])
        with self.assertRaises(KeyError):
            _get_keys(subtypes, "NotAColumn", ["species"])


# ---------------------------------------------------------------------------
# _find_split_value
# ---------------------------------------------------------------------------

class TestFindSplitValue(TestPluginBase):
    package = "q2_PSEA.tests"

    def _hit(self, **kwargs):
        return pd.Series(kwargs)

    def test_returns_empty_string_when_no_split_column(self):
        hit = self._hit(Category="Viral")
        self.assertEqual(_find_split_value(hit, None, 0), "")

    def test_returns_scalar_value_with_trailing_dash(self):
        hit = self._hit(Category="Viral")
        self.assertEqual(_find_split_value(hit, "Category", 0), "Viral-")

    def test_returns_indexed_list_element_with_trailing_dash(self):
        hit = self._hit(Category=["Viral", "Bacterial"])
        self.assertEqual(_find_split_value(hit, "Category", 1), "Bacterial-")


# ---------------------------------------------------------------------------
# _count_enriched
# ---------------------------------------------------------------------------

class TestCountEnriched(TestPluginBase):
    package = "q2_PSEA.tests"

    def _empty(self):
        return {
            "species": {},
            "subspecies": {},
            "species-epitope": {},
        }

    def test_initializes_species_key(self):
        counts = self._empty()
        _count_enriched(counts, "InfluenzaA", "InfluenzaA:H1N1", "ep1", "")
        self.assertEqual(counts["species"]["InfluenzaA"], 1)

    def test_initializes_subspecies_key(self):
        counts = self._empty()
        _count_enriched(counts, "InfluenzaA", "InfluenzaA:H1N1", "ep1", "")
        self.assertEqual(counts["subspecies"]["InfluenzaA:H1N1"], 1)

    def test_initializes_species_epitope_key(self):
        counts = self._empty()
        _count_enriched(counts, "InfluenzaA", "InfluenzaA:H1N1", "ep1", "")
        self.assertEqual(counts["species-epitope"]["InfluenzaA-ep1"], 1)

    def test_second_call_increments_count(self):
        counts = self._empty()
        _count_enriched(counts, "InfluenzaA", "InfluenzaA:H1N1", "ep1", "")
        _count_enriched(counts, "InfluenzaA", "InfluenzaA:H1N1", "ep1", "")
        self.assertEqual(counts["species"]["InfluenzaA"], 2)

    def test_split_value_prefix_applied_to_all_keys(self):
        counts = {
            "Viral-species": {},
            "Viral-subspecies": {},
            "Viral-species-epitope": {},
        }
        _count_enriched(
            counts, "InfluenzaA", "InfluenzaA:H1N1", "ep1", "Viral-"
        )
        self.assertIn("Viral-InfluenzaA", counts["Viral-species"])
        self.assertIn("Viral-InfluenzaA:H1N1", counts["Viral-subspecies"])
        self.assertIn("Viral-InfluenzaA-ep1", counts["Viral-species-epitope"])


# ---------------------------------------------------------------------------
# enriched_subtypes
# ---------------------------------------------------------------------------

class TestEnrichedSubtypes(TestPluginBase):
    package = "q2_PSEA.tests"

    def _scores(self, rows):
        return {"pair1": pd.DataFrame(rows)}

    def _subtypes(self):
        return pd.DataFrame(
            {
                "CodeName": [["pep_00", "pep_01"], ["pep_02"]],
                "Subtype": [["H1N1", "H3N2"], ["Yamagata"]],
                "Category": ["Viral", "Viral"],
            },
            index=pd.Index(
                ["sp001_C1_W1", "sp002_C2_W2"], name="EpitopeID"
            ),
        )

    def test_returns_dict_with_three_default_keys(self):
        scores = self._scores([
            {
                "p.adjust": 0.9,
                "enrichmentScore": 0.0,
                "core_enrichment": "sp001_C1_W1",
                "species_name": "InfluenzaA",
            }
        ])
        result = enriched_subtypes(scores, self._subtypes())
        self.assertIsInstance(result, dict)
        self.assertEqual(
            set(result.keys()), {"species", "subspecies", "species-epitope"}
        )

    def test_all_values_are_dataframes(self):
        scores = self._scores([
            {
                "p.adjust": 0.01,
                "enrichmentScore": 2.0,
                "core_enrichment": "sp001_C1_W1",
                "species_name": "InfluenzaA",
            }
        ])
        for val in enriched_subtypes(scores, self._subtypes()).values():
            self.assertIsInstance(val, pd.DataFrame)

    def test_collapsed_epitope_counts_species(self):
        scores = self._scores([
            {
                "p.adjust": 0.01,
                "enrichmentScore": 2.0,
                "core_enrichment": "sp001_C1_W1",
                "species_name": "InfluenzaA",
            }
        ])
        result = enriched_subtypes(scores, self._subtypes())
        self.assertIn("InfluenzaA", result["species"].index)

    def test_no_significant_rows_returns_empty_dataframes(self):
        scores = self._scores([
            {
                "p.adjust": 0.9,
                "enrichmentScore": 0.0,
                "core_enrichment": "sp001_C1_W1",
                "species_name": "InfluenzaA",
            }
        ])
        result = enriched_subtypes(scores, self._subtypes())
        for val in result.values():
            self.assertEqual(len(val), 0)

    def test_split_column_multiplies_keys(self):
        scores = self._scores([
            {
                "p.adjust": 0.9,
                "enrichmentScore": 0.0,
                "core_enrichment": "sp001_C1_W1",
                "species_name": "InfluenzaA",
            }
        ])
        result = enriched_subtypes(
            scores, self._subtypes(), split_column="Category"
        )
        # Unique Category = ['Viral'] → keys become Viral-<default_key>
        self.assertIn("Viral-species", result)
        self.assertIn("Viral-subspecies", result)
        self.assertIn("Viral-species-epitope", result)

    def test_invalid_split_column_raises_key_error(self):
        scores = self._scores([
            {
                "p.adjust": 0.01,
                "enrichmentScore": 2.0,
                "core_enrichment": "sp001_C1_W1",
                "species_name": "InfluenzaA",
            }
        ])
        with self.assertRaises(KeyError):
            enriched_subtypes(
                scores, self._subtypes(), split_column="NotAColumn"
            )

    def test_uncollapsed_peptide_path_uses_peptide_library_prefix(self):
        # An enriched element that starts with the peptide_library string is
        # treated as uncollapsed and looked up via CodeName membership.
        subtypes = pd.DataFrame(
            {
                "CodeName": [["IN2_001", "IN2_002"]],
                "Subtype": ["H1N1"],
                "Category": ["Viral"],
            },
            index=pd.Index(["sp001_C1_W1"], name="EpitopeID"),
        )
        scores = self._scores([
            {
                "p.adjust": 0.01,
                "enrichmentScore": 2.0,
                "core_enrichment": "IN2_001",
                "species_name": "InfluenzaA",
            }
        ])
        # Should not raise; the uncollapsed peptide is found via CodeName
        # lookup.
        result = enriched_subtypes(
            scores, subtypes, peptide_library="IN2"
        )
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
