import unittest

from qiime2.plugin import ValidationError
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.format_types import (
    PSEAPairsTSVFormat,
    PSEASpeciesColorsTSVFormat,
    PSEASpeciesTaxaTSVFormat,
)


def _write(fmt_cls, content):
    """Create a new format instance and write *content* into it."""
    fmt = fmt_cls()
    with fmt.open() as fh:
        fh.write(content)
    return fmt


class TestPSEAPairsTSVFormat(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_valid_fixture(self):
        fmt = PSEAPairsTSVFormat(self.get_data_path("pairs.tsv"), mode="r")
        fmt.validate()

    def test_valid_multiple_pairs(self):
        fmt = PSEAPairsTSVFormat(self.get_data_path("pairs-two.tsv"), mode="r")
        fmt.validate()

    def test_empty_file_raises(self):
        fmt = _write(PSEAPairsTSVFormat, "")
        with self.assertRaisesRegex(ValidationError, "header"):
            fmt.validate()

    def test_header_only_raises(self):
        fmt = _write(PSEAPairsTSVFormat, "sA\tsB\n")
        with self.assertRaisesRegex(ValidationError, "one pair"):
            fmt.validate()

    def test_single_column_data_row_raises(self):
        fmt = _write(PSEAPairsTSVFormat, "sA\tsB\nS1\n")
        with self.assertRaises(ValidationError):
            fmt.validate()

    def test_empty_first_column_raises(self):
        fmt = _write(PSEAPairsTSVFormat, "sA\tsB\n\tS2\n")
        with self.assertRaises(ValidationError):
            fmt.validate()

    def test_empty_second_column_raises(self):
        fmt = _write(PSEAPairsTSVFormat, "sA\tsB\nS1\t\n")
        with self.assertRaises(ValidationError):
            fmt.validate()


class TestPSEASpeciesTaxaTSVFormat(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_valid_fixture(self):
        fmt = PSEASpeciesTaxaTSVFormat(
            self.get_data_path("species-taxa.tsv"), mode="r"
        )
        fmt.validate()

    def test_empty_file_raises(self):
        fmt = _write(PSEASpeciesTaxaTSVFormat, "")
        with self.assertRaisesRegex(ValidationError, "empty"):
            fmt.validate()

    def test_single_column_raises(self):
        fmt = _write(PSEASpeciesTaxaTSVFormat, "InfluenzaA\n")
        with self.assertRaises(ValidationError):
            fmt.validate()

    def test_empty_second_column_raises(self):
        fmt = _write(PSEASpeciesTaxaTSVFormat, "InfluenzaA\t\n")
        with self.assertRaises(ValidationError):
            fmt.validate()


class TestPSEASpeciesColorsTSVFormat(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_valid_fixture_with_hash(self):
        fmt = PSEASpeciesColorsTSVFormat(
            self.get_data_path("species-colors.tsv"), mode="r"
        )
        fmt.validate()

    def test_valid_without_hash(self):
        fmt = _write(PSEASpeciesColorsTSVFormat, "InfluenzaA\tFF0000\n")
        fmt.validate()

    def test_valid_lowercase_hex(self):
        fmt = _write(PSEASpeciesColorsTSVFormat, "InfluenzaA\t#ff0000\n")
        fmt.validate()

    def test_empty_file_raises(self):
        fmt = _write(PSEASpeciesColorsTSVFormat, "")
        with self.assertRaisesRegex(ValidationError, "empty"):
            fmt.validate()

    def test_invalid_hex_raises(self):
        fmt = _write(PSEASpeciesColorsTSVFormat, "InfluenzaA\tnotacolor\n")
        with self.assertRaisesRegex(ValidationError, "HEX"):
            fmt.validate()

    def test_hex_too_short_raises(self):
        fmt = _write(PSEASpeciesColorsTSVFormat, "InfluenzaA\t#FF00\n")
        with self.assertRaisesRegex(ValidationError, "HEX"):
            fmt.validate()

    def test_hex_too_long_raises(self):
        fmt = _write(PSEASpeciesColorsTSVFormat, "InfluenzaA\t#FF0000AA\n")
        with self.assertRaisesRegex(ValidationError, "HEX"):
            fmt.validate()

    def test_single_column_raises(self):
        fmt = _write(PSEASpeciesColorsTSVFormat, "InfluenzaA\n")
        with self.assertRaises(ValidationError):
            fmt.validate()


if __name__ == "__main__":
    unittest.main()
