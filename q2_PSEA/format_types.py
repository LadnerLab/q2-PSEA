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


PSEAPairsDirFmt = model.SingleFileDirectoryFormat(
    "PSEAPairsDirFmt", "pairs.tsv", PSEAPairsTSVFormat
)


class PSEAAECountsTSVFormat(model.TextFileFormat):
    def _validate_(self, level):
        with self.open() as fh:
            header = fh.readline().strip().split("\t")
            if header != ["Species", "Events"]:
                raise ValidationError(
                    "AE counts file must have a header of 'Species\\tEvents'."
                )


PSEAAECountsDirFmt = model.SingleFileDirectoryFormat(
    "PSEAAECountsDirFmt", "ae_counts.tsv", PSEAAECountsTSVFormat
)


class SplineTSVFormat(model.TextFileFormat):
    _REQUIRED_COLUMNS = {"x", "yfit", "maxZ", "deltaZ"}

    def _validate_(self, level):
        with self.open() as fh:
            header_line = fh.readline().strip()
            if not header_line:
                raise ValidationError("Spline file must have a header line.")
            cols = set(header_line.split("\t")[1:])  # first field is index
            missing = self._REQUIRED_COLUMNS - cols
            if missing:
                raise ValidationError(
                    f"Spline file is missing required columns: {missing}."
                )


SplineDirFmt = model.SingleFileDirectoryFormat(
    "SplineDirFmt", "spline.tsv", SplineTSVFormat
)


__all__ = [
    "PSEAAECountsDirFmt",
    "PSEAAECountsTSVFormat",
    "PSEAPairsDirFmt",
    "PSEAPairsTSVFormat",
    "SplineDirFmt",
    "SplineTSVFormat",
]
