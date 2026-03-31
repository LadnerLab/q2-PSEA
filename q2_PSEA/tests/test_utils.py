"""
Tests for q2_PSEA.utils helper functions.

Only the DataFrame-path (format="df") is tested here because all other paths
(csv, gmt, tsv) require external files; they are exercised through integration
tests.
"""
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from q2_PSEA.utils import remove_peptides, remove_peptides_in_df_format


def _make_scores(peptides, samples=("sA", "sB")):
    """Return a scores DataFrame with *peptides* as index."""
    data = {s: list(range(len(peptides))) for s in samples}
    return pd.DataFrame(data, index=peptides)


def _make_gmt(term_gene_pairs):
    """Return a GMT-style DataFrame from a list of (term, gene) tuples."""
    terms, genes = zip(*term_gene_pairs)
    return pd.DataFrame({"term": list(terms), "gene": list(genes)})


# ---------------------------------------------------------------------------
# remove_peptides_in_df_format
# ---------------------------------------------------------------------------

class TestRemovePeptidesInDfFormat:
    def test_removes_peptides_not_in_gmt(self):
        scores = _make_scores(["pep1", "pep2", "pep3"])
        gmt = _make_gmt([("sp1", "pep1"), ("sp1", "pep2")])

        filtered, returned_gmt = remove_peptides_in_df_format(scores, gmt)

        assert list(filtered.index) == ["pep1", "pep2"]
        assert_frame_equal(returned_gmt, gmt)

    def test_all_peptides_in_gmt_retained(self):
        scores = _make_scores(["pep1", "pep2"])
        gmt = _make_gmt([("sp1", "pep1"), ("sp2", "pep2")])

        filtered, _ = remove_peptides_in_df_format(scores, gmt)

        assert set(filtered.index) == {"pep1", "pep2"}

    def test_no_overlap_yields_empty(self):
        scores = _make_scores(["pep1", "pep2"])
        gmt = _make_gmt([("sp1", "pep3"), ("sp1", "pep4")])

        filtered, _ = remove_peptides_in_df_format(scores, gmt)

        assert len(filtered) == 0

    def test_values_are_preserved(self):
        scores = pd.DataFrame(
            {"sA": [10.0, 20.0, 30.0]},
            index=["pep1", "pep2", "pep3"],
        )
        gmt = _make_gmt([("sp1", "pep1"), ("sp1", "pep3")])

        filtered, _ = remove_peptides_in_df_format(scores, gmt)

        assert filtered.loc["pep1", "sA"] == 10.0
        assert filtered.loc["pep3", "sA"] == 30.0


# ---------------------------------------------------------------------------
# remove_peptides dispatch (df format)
# ---------------------------------------------------------------------------

class TestRemovePeptides:
    def test_dispatches_to_df_format_when_given_dataframe(self):
        scores = _make_scores(["pep1", "pep2", "pep3"])
        gmt = _make_gmt([("sp1", "pep1"), ("sp1", "pep2")])

        filtered, _ = remove_peptides(scores, gmt)

        assert list(filtered.index) == ["pep1", "pep2"]

    def test_unsupported_string_format_raises(self):
        scores = _make_scores(["pep1"])
        with pytest.raises(AssertionError):
            remove_peptides(scores, "file.unknown")
