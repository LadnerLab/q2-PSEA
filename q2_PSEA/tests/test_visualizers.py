import os
import pathlib
import tempfile
import unittest

import pandas as pd
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.actions.visualizers import aeplots, volcano, zscatter


# ---------------------------------------------------------------------------
# Helpers shared across visualizer test classes
# ---------------------------------------------------------------------------

class _FakePairsDirFmt:
    """Minimal stand-in for PSEAPairsDirFmt in visualizer tests.

    All three visualizers read ``pairs.path / "pairs.tsv"`` to discover the
    ordered list of sample pairs.  The test data directory already contains a
    ``pairs.tsv`` with header row and one data row ``sample1\\tsample2``,
    producing the pair string ``"sample1~sample2"``.
    """

    def __init__(self, path):
        self.path = pathlib.Path(path)


def _data_dir(test_instance, filename="pairs.tsv"):
    """Return the tests/data directory as a Path via get_data_path."""
    return pathlib.Path(test_instance.get_data_path(filename)).parent


def _volcano_psea_table():
    """One significant row (InfluenzaA) and one non-significant row (EBV)."""
    return pd.DataFrame({
        "NES": [2.1, -0.3],
        "p.adjust": [0.01, 0.9],
        "species_name": ["InfluenzaA", "EBV"],
    })


def _zscatter_psea_table():
    """All rows non-significant — no leading-edge peptide lookups occur."""
    return pd.DataFrame({
        "p.adjust": [0.9, 0.8],
        "core_enrichment": ["pep_00", "pep_01"],
        "species_name": ["InfluenzaA", "EBV"],
    })


# ---------------------------------------------------------------------------
# volcano
# ---------------------------------------------------------------------------

class TestVolcano(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.pairs = _FakePairsDirFmt(_data_dir(self))
        # psea_tables dict: key must match the pair string from pairs.tsv
        self.psea_tables = {"sample1~sample2": _volcano_psea_table()}

    def _call(self, output_dir, **kwargs):
        return volcano(output_dir=output_dir, pairs=self.pairs, **kwargs)

    # ------------------------------------------------------------------
    # Core output
    # ------------------------------------------------------------------

    def test_creates_index_html_from_xy(self):
        with tempfile.TemporaryDirectory() as output_dir:
            self._call(
                output_dir,
                x=[1.0, -0.5],
                y=[0.01, 0.8],
                xy_labels=["NES", "p.adjust"],
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    def test_creates_index_html_from_psea_tables(self):
        with tempfile.TemporaryDirectory() as output_dir:
            self._call(
                output_dir,
                psea_tables=self.psea_tables,
                xy_access=["NES", "p.adjust"],
                xy_labels=["NES", "p.adjust"],
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    # ------------------------------------------------------------------
    # Highlight layer (taxa_access)
    # ------------------------------------------------------------------

    def test_taxa_access_adds_highlight_layer(self):
        """Significant row triggers a highlight layer; index.html must exist."""
        with tempfile.TemporaryDirectory() as output_dir:
            self._call(
                output_dir,
                psea_tables=self.psea_tables,
                xy_access=["NES", "p.adjust"],
                xy_labels=["NES", "p.adjust"],
                taxa_access="species_name",
                x_threshold=0.4,
                y_threshold=0.05,
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    def test_no_taxa_access_skips_highlight_layer(self):
        """No taxa_access → no highlight chart built; must still produce HTML."""
        with tempfile.TemporaryDirectory() as output_dir:
            self._call(
                output_dir,
                psea_tables=self.psea_tables,
                xy_access=["NES", "p.adjust"],
                xy_labels=["NES", "p.adjust"],
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    # ------------------------------------------------------------------
    # log=False
    # ------------------------------------------------------------------

    def test_log_false_does_not_raise(self):
        with tempfile.TemporaryDirectory() as output_dir:
            self._call(
                output_dir,
                x=[1.0, -0.5],
                y=[0.01, 0.8],
                xy_labels=["NES", "p.adjust"],
                log=False,
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    # ------------------------------------------------------------------
    # colors_file
    # ------------------------------------------------------------------

    def test_colors_file_does_not_raise(self):
        """Custom color mapping for significant taxa must not raise."""
        with tempfile.TemporaryDirectory() as output_dir:
            self._call(
                output_dir,
                psea_tables=self.psea_tables,
                xy_access=["NES", "p.adjust"],
                xy_labels=["NES", "p.adjust"],
                taxa_access="species_name",
                x_threshold=0.4,
                y_threshold=0.05,
                colors_file=self.get_data_path("species-colors.tsv"),
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    # ------------------------------------------------------------------
    # vis_outputs_dir
    # ------------------------------------------------------------------

    def test_vis_outputs_dir_creates_per_pair_html(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with tempfile.TemporaryDirectory() as vis_dir:
                self._call(
                    output_dir,
                    x=[1.0, -0.5],
                    y=[0.01, 0.8],
                    xy_labels=["NES", "p.adjust"],
                    vis_outputs_dir=vis_dir,
                )
                per_pair_html = os.path.join(
                    vis_dir, "volcano_plots", "sample1~sample2_volcano.html"
                )
                self.assertTrue(os.path.exists(per_pair_html))

    def test_vis_outputs_dir_creates_volcano_plots_subdir(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with tempfile.TemporaryDirectory() as vis_dir:
                self._call(
                    output_dir,
                    x=[1.0, -0.5],
                    y=[0.01, 0.8],
                    xy_labels=["NES", "p.adjust"],
                    vis_outputs_dir=vis_dir,
                )
                self.assertTrue(
                    os.path.isdir(os.path.join(vis_dir, "volcano_plots"))
                )


# ---------------------------------------------------------------------------
# zscatter
# ---------------------------------------------------------------------------

class TestZscatter(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        # scores-vis.tsv has sample1 and sample2 columns matching pairs.tsv data
        self.zscores = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        self.pairs = _FakePairsDirFmt(_data_dir(self, "scores-vis.tsv"))

    def _call(self, output_dir, **kwargs):
        return zscatter(
            output_dir=output_dir,
            zscores=self.zscores,
            pairs=self.pairs,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core output
    # ------------------------------------------------------------------

    def test_creates_index_html(self):
        with tempfile.TemporaryDirectory() as output_dir:
            self._call(output_dir)
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    # ------------------------------------------------------------------
    # Required-arg assertions when psea_tables is provided
    # ------------------------------------------------------------------

    def test_psea_tables_without_p_val_access_raises(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(AssertionError):
                self._call(
                    output_dir,
                    psea_tables={"sample1~sample2": _zscatter_psea_table()},
                    le_peps_access="core_enrichment",
                    taxa_access="species_name",
                )

    def test_psea_tables_without_le_peps_access_raises(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(AssertionError):
                self._call(
                    output_dir,
                    psea_tables={"sample1~sample2": _zscatter_psea_table()},
                    p_val_access="p.adjust",
                    taxa_access="species_name",
                )

    def test_psea_tables_without_taxa_access_raises(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(AssertionError):
                self._call(
                    output_dir,
                    psea_tables={"sample1~sample2": _zscatter_psea_table()},
                    p_val_access="p.adjust",
                    le_peps_access="core_enrichment",
                )

    # ------------------------------------------------------------------
    # psea_tables with non-significant rows (no peptide lookups)
    # ------------------------------------------------------------------

    def test_psea_tables_nonsig_rows_creates_index_html(self):
        """With all p.adjust above threshold, no peptide lookups occur."""
        with tempfile.TemporaryDirectory() as output_dir:
            self._call(
                output_dir,
                psea_tables={"sample1~sample2": _zscatter_psea_table()},
                p_val_access="p.adjust",
                le_peps_access="core_enrichment",
                taxa_access="species_name",
                highlight_threshold=0.05,
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    # ------------------------------------------------------------------
    # spline_file
    # ------------------------------------------------------------------

    def test_spline_file_creates_index_html(self):
        with tempfile.TemporaryDirectory() as output_dir:
            self._call(
                output_dir,
                spline_file=self.get_data_path("spline-data.tsv"),
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    # ------------------------------------------------------------------
    # vis_outputs_dir
    # ------------------------------------------------------------------

    def test_vis_outputs_dir_creates_scatter_subdir(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with tempfile.TemporaryDirectory() as vis_dir:
                self._call(output_dir, vis_outputs_dir=vis_dir)
                self.assertTrue(
                    os.path.isdir(os.path.join(vis_dir, "scatter_plots"))
                )

    def test_vis_outputs_dir_creates_per_pair_html(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with tempfile.TemporaryDirectory() as vis_dir:
                self._call(output_dir, vis_outputs_dir=vis_dir)
                per_pair_html = os.path.join(
                    vis_dir, "scatter_plots", "sample1~sample2_scatter.html"
                )
                self.assertTrue(os.path.exists(per_pair_html))


# ---------------------------------------------------------------------------
# aeplots
# ---------------------------------------------------------------------------

class TestAeplots(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.pos_ae = self.get_data_path("pos-ae.tsv")
        self.neg_ae = self.get_data_path("neg-ae.tsv")

    def _call(self, output_dir, **kwargs):
        return aeplots(
            output_dir=output_dir,
            pos_nes_ae_file=self.pos_ae,
            neg_nes_ae_file=self.neg_ae,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core output
    # ------------------------------------------------------------------

    def test_creates_index_html(self):
        with tempfile.TemporaryDirectory() as output_dir:
            self._call(output_dir)
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    # ------------------------------------------------------------------
    # colors_file
    # ------------------------------------------------------------------

    def test_colors_file_does_not_raise(self):
        """Species in pos/neg AE files that match colors file get custom colors."""
        with tempfile.TemporaryDirectory() as output_dir:
            self._call(
                output_dir,
                colors_file=self.get_data_path("species-colors.tsv"),
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    # ------------------------------------------------------------------
    # vis_outputs_dir
    # ------------------------------------------------------------------

    def test_vis_outputs_dir_saves_aeplots_html(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with tempfile.TemporaryDirectory() as vis_dir:
                self._call(output_dir, vis_outputs_dir=vis_dir)
                self.assertTrue(
                    os.path.exists(os.path.join(vis_dir, "aeplots.html"))
                )


if __name__ == "__main__":
    unittest.main()
