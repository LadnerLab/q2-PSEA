import re

from qiime2.plugin import ValidationError, model


class PSEAPairsTSVFormat(model.TextFileFormat):
    def _validate_(self, level):
        with self.open() as fh:
            header = fh.readline().strip()
            if not header:
                raise ValidationError("Pairs file must have a header line.")
            seen_data = False
            for line_no, line in enumerate(fh, start=2):
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 2 or not cols[0] or not cols[1]:
                    raise ValidationError(
                        f"Expected two non-empty tab-delimited values on"
                        f" line {line_no} of pairs file."
                    )
                seen_data = True
            if not seen_data:
                raise ValidationError(
                    "Pairs file must contain at least one pair row."
                )


class PSEASpeciesTaxaTSVFormat(model.TextFileFormat):
    def _validate_(self, level):
        with self.open() as fh:
            found_line = False
            for line_no, line in enumerate(fh, start=1):
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 2 or not cols[0] or not cols[1]:
                    raise ValidationError(
                        f"Species taxa line {line_no} must have two"
                        f" non-empty tab-delimited columns."
                    )
                found_line = True
            if not found_line:
                raise ValidationError("Species taxa file is empty.")


class PSEASpeciesColorsTSVFormat(model.TextFileFormat):
    _HEX_PATTERN = re.compile(r"^#?[0-9a-fA-F]{6}$")

    def _validate_(self, level):
        with self.open() as fh:
            found_line = False
            for line_no, line in enumerate(fh, start=1):
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 2 or not cols[0] or not cols[1]:
                    raise ValidationError(
                        f"Species colors line {line_no} must have two"
                        f" non-empty tab-delimited columns."
                    )
                if not self._HEX_PATTERN.match(cols[1]):
                    raise ValidationError(
                        f"Species colors line {line_no} has an invalid"
                        f" HEX color value: '{cols[1]}'."
                    )
                found_line = True
            if not found_line:
                raise ValidationError("Species colors file is empty.")


PSEAPairsDirFmt = model.SingleFileDirectoryFormat(
    "PSEAPairsDirFmt", "pairs.tsv", PSEAPairsTSVFormat
)
PSEASpeciesTaxaDirFmt = model.SingleFileDirectoryFormat(
    "PSEASpeciesTaxaDirFmt", "species-taxa.tsv", PSEASpeciesTaxaTSVFormat
)
PSEASpeciesColorsDirFmt = model.SingleFileDirectoryFormat(
    "PSEASpeciesColorsDirFmt", "species-colors.tsv", PSEASpeciesColorsTSVFormat
)


__all__ = [
    "PSEAPairsTSVFormat",
    "PSEASpeciesTaxaTSVFormat",
    "PSEASpeciesColorsTSVFormat",
    "PSEAPairsDirFmt",
    "PSEASpeciesTaxaDirFmt",
    "PSEASpeciesColorsDirFmt",
]
