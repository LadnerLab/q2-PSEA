"""
Tests for PSEAPairsTSVFormat, PSEASpeciesTaxaTSVFormat, and
PSEASpeciesColorsTSVFormat validation logic.
"""
import pytest
from qiime2.plugin import ValidationError

from q2_PSEA.format_types import (
    PSEAPairsTSVFormat,
    PSEASpeciesTaxaTSVFormat,
    PSEASpeciesColorsTSVFormat,
)


def _make_fmt(fmt_cls, content):
    """Create a format instance and write *content* into it."""
    fmt = fmt_cls()
    with fmt.open() as fh:
        fh.write(content)
    return fmt


# ---------------------------------------------------------------------------
# PSEAPairsTSVFormat
# ---------------------------------------------------------------------------

class TestPSEAPairsTSVFormat:
    def test_valid_single_pair(self):
        fmt = _make_fmt(PSEAPairsTSVFormat, "sA\tsB\nS1\tS2\n")
        fmt.validate()  # must not raise

    def test_valid_multiple_pairs(self):
        fmt = _make_fmt(
            PSEAPairsTSVFormat,
            "sA\tsB\nS1\tS2\nS3\tS4\nS5\tS6\n",
        )
        fmt.validate()

    def test_empty_file_raises(self):
        fmt = _make_fmt(PSEAPairsTSVFormat, "")
        with pytest.raises(ValidationError, match="header"):
            fmt.validate()

    def test_header_only_raises(self):
        fmt = _make_fmt(PSEAPairsTSVFormat, "sA\tsB\n")
        with pytest.raises(ValidationError, match="one pair"):
            fmt.validate()

    def test_single_column_data_row_raises(self):
        fmt = _make_fmt(PSEAPairsTSVFormat, "sA\tsB\nS1\n")
        with pytest.raises(ValidationError):
            fmt.validate()

    def test_empty_first_column_raises(self):
        fmt = _make_fmt(PSEAPairsTSVFormat, "sA\tsB\n\tS2\n")
        with pytest.raises(ValidationError):
            fmt.validate()

    def test_empty_second_column_raises(self):
        fmt = _make_fmt(PSEAPairsTSVFormat, "sA\tsB\nS1\t\n")
        with pytest.raises(ValidationError):
            fmt.validate()


# ---------------------------------------------------------------------------
# PSEASpeciesTaxaTSVFormat
# ---------------------------------------------------------------------------

class TestPSEASpeciesTaxaTSVFormat:
    def test_valid(self):
        fmt = _make_fmt(PSEASpeciesTaxaTSVFormat, "InfluenzaA\t11520\n")
        fmt.validate()

    def test_valid_multiple_rows(self):
        fmt = _make_fmt(
            PSEASpeciesTaxaTSVFormat,
            "InfluenzaA\t11520\nEBV\t10376\nHIV-1\t11676\n",
        )
        fmt.validate()

    def test_empty_file_raises(self):
        fmt = _make_fmt(PSEASpeciesTaxaTSVFormat, "")
        with pytest.raises(ValidationError, match="empty"):
            fmt.validate()

    def test_single_column_raises(self):
        fmt = _make_fmt(PSEASpeciesTaxaTSVFormat, "InfluenzaA\n")
        with pytest.raises(ValidationError):
            fmt.validate()

    def test_empty_second_column_raises(self):
        fmt = _make_fmt(PSEASpeciesTaxaTSVFormat, "InfluenzaA\t\n")
        with pytest.raises(ValidationError):
            fmt.validate()


# ---------------------------------------------------------------------------
# PSEASpeciesColorsTSVFormat
# ---------------------------------------------------------------------------

class TestPSEASpeciesColorsTSVFormat:
    def test_valid_with_hash(self):
        fmt = _make_fmt(PSEASpeciesColorsTSVFormat, "InfluenzaA\t#FF0000\n")
        fmt.validate()

    def test_valid_without_hash(self):
        fmt = _make_fmt(PSEASpeciesColorsTSVFormat, "InfluenzaA\tFF0000\n")
        fmt.validate()

    def test_valid_lowercase_hex(self):
        fmt = _make_fmt(PSEASpeciesColorsTSVFormat, "InfluenzaA\t#ff0000\n")
        fmt.validate()

    def test_valid_mixed_case_hex(self):
        fmt = _make_fmt(PSEASpeciesColorsTSVFormat, "InfluenzaA\t#aAbBcC\n")
        fmt.validate()

    def test_valid_multiple_rows(self):
        fmt = _make_fmt(
            PSEASpeciesColorsTSVFormat,
            "InfluenzaA\t#FF0000\nEBV\t#00FF00\nHIV-1\t0000FF\n",
        )
        fmt.validate()

    def test_empty_file_raises(self):
        fmt = _make_fmt(PSEASpeciesColorsTSVFormat, "")
        with pytest.raises(ValidationError, match="empty"):
            fmt.validate()

    def test_invalid_hex_text_raises(self):
        fmt = _make_fmt(PSEASpeciesColorsTSVFormat, "InfluenzaA\tnotacolor\n")
        with pytest.raises(ValidationError, match="HEX"):
            fmt.validate()

    def test_hex_too_short_raises(self):
        fmt = _make_fmt(PSEASpeciesColorsTSVFormat, "InfluenzaA\t#FF00\n")
        with pytest.raises(ValidationError, match="HEX"):
            fmt.validate()

    def test_hex_too_long_raises(self):
        fmt = _make_fmt(PSEASpeciesColorsTSVFormat, "InfluenzaA\t#FF0000AA\n")
        with pytest.raises(ValidationError, match="HEX"):
            fmt.validate()

    def test_single_column_raises(self):
        fmt = _make_fmt(PSEASpeciesColorsTSVFormat, "InfluenzaA\n")
        with pytest.raises(ValidationError):
            fmt.validate()
