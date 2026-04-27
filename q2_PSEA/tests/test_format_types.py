import unittest

from qiime2.plugin import ValidationError
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.format_types import (
    PSEAPairsTSVFormat, PSEAAECountsTSVFormat, SplineTSVFormat
)


def _write(fmt_cls, content):
    fmt = fmt_cls()
    with fmt.open() as fh:
        fh.write(content)
    return fmt


class TestPSEAPairsTSVFormat(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_valid_single_pair_fixture(self):
        fmt = PSEAPairsTSVFormat(self.get_data_path("pairs.tsv"), mode="r")
        fmt.validate()

    def test_valid_two_pair_fixture(self):
        fmt = PSEAPairsTSVFormat(self.get_data_path("pairs-two.tsv"), mode="r")
        fmt.validate()

    def test_empty_file_raises_validation_error(self):
        fmt = _write(PSEAPairsTSVFormat, "")
        with self.assertRaisesRegex(ValidationError, "header"):
            fmt.validate()

    def test_header_only_raises_validation_error(self):
        fmt = _write(PSEAPairsTSVFormat, "sample_a\tsample_b\n")
        with self.assertRaisesRegex(ValidationError, "one pair"):
            fmt.validate()

    def test_single_column_data_row_raises(self):
        fmt = _write(PSEAPairsTSVFormat, "a\tb\nS1\n")
        with self.assertRaises(ValidationError):
            fmt.validate()

    def test_empty_first_column_raises(self):
        fmt = _write(PSEAPairsTSVFormat, "a\tb\n\tS2\n")
        with self.assertRaises(ValidationError):
            fmt.validate()

    def test_empty_second_column_raises(self):
        fmt = _write(PSEAPairsTSVFormat, "a\tb\nS1\t\n")
        with self.assertRaises(ValidationError):
            fmt.validate()

    def test_valid_multiple_data_rows(self):
        fmt = _write(PSEAPairsTSVFormat, "a\tb\nS1\tS2\nS3\tS4\n")
        fmt.validate()


class TestPSEAAECountsTSVFormat(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_valid_pos_ae_fixture(self):
        fmt = PSEAAECountsTSVFormat(self.get_data_path("pos-ae.tsv"), mode="r")
        fmt.validate()

    def test_valid_neg_ae_fixture(self):
        fmt = PSEAAECountsTSVFormat(self.get_data_path("neg-ae.tsv"), mode="r")
        fmt.validate()

    def test_wrong_header_raises(self):
        fmt = _write(PSEAAECountsTSVFormat, "Name\tCount\nInfluenzaA\t3\n")
        with self.assertRaises(ValidationError):
            fmt.validate()

    def test_empty_file_raises(self):
        fmt = _write(PSEAAECountsTSVFormat, "")
        with self.assertRaises(ValidationError):
            fmt.validate()

    def test_correct_header_passes(self):
        fmt = _write(PSEAAECountsTSVFormat, "Species\tEvents\nsp1\t1\n")
        fmt.validate()


class TestSplineTSVFormat(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_valid_spline_content(self):
        content = \
            "feature-id\tx\tyfit\tmaxZ\tdeltaZ\npep_00\t0.1\t0.1\t0.5\t0.0\n"
        fmt = _write(SplineTSVFormat, content)
        fmt.validate()

    def test_missing_required_column_raises(self):
        content = "feature-id\tx\tyfit\tmaxZ\n"
        fmt = _write(SplineTSVFormat, content)
        with self.assertRaises(ValidationError):
            fmt.validate()

    def test_all_required_columns_needed(self):
        for missing in ("x", "yfit", "maxZ", "deltaZ"):
            cols = {"x", "yfit", "maxZ", "deltaZ"} - {missing}
            header = "feature-id\t" + "\t".join(sorted(cols)) + "\n"
            fmt = _write(SplineTSVFormat, header)
            with self.assertRaises(ValidationError):
                fmt.validate()

    def test_empty_file_raises(self):
        fmt = _write(SplineTSVFormat, "")
        with self.assertRaises(ValidationError):
            fmt.validate()


if __name__ == "__main__":
    unittest.main()
