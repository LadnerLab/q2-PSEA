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


__all__ = [
    "PSEAAECountsDirFmt",
    "PSEAAECountsTSVFormat",
    "PSEAPairsDirFmt",
    "PSEAPairsTSVFormat",
]
