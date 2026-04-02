import unittest

from qiime2.plugin import ValidationError
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.format_types import PSEAPairsTSVFormat


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


if __name__ == "__main__":
    unittest.main()
