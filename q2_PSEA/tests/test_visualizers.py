import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import qiime2
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.actions.visualizers import aeplots, volcano, zscatter


def _pairs_df(rows=None):
    if rows is None:
        rows = [("sA", "sB")]
    return pd.DataFrame(rows, columns=["sample_a", "sample_b"])


def _pairs_df_titled(rows=None):
    if rows is None:
        rows = [("sA", "sB", "Title A vs B")]
    return pd.DataFrame(rows, columns=["sample_a", "sample_b", "title"])


def _ae_df(rows=None):
    if rows is None:
        rows = [("InfluenzaA", 3), ("EBV", 2)]
    return pd.DataFrame(rows, columns=["Species", "Events"])


def _zscores_df():
    rng = np.random.default_rng(42)
    peptides = [f"pep_{i:02d}" for i in range(20)]
    return pd.DataFrame(
        rng.standard_normal((2, 20)),
        index=["sA", "sB"],
        columns=peptides,
    )


def _psea_table_df():
    return pd.DataFrame({
        "NES": [2.1, -1.5, 0.3],
        "p.adjust": [0.01, 0.03, 0.8],
        "core_enrichment": ["pep_00/pep_01", "pep_02/pep_03", "pep_04"],
        "ID": ["InfluenzaA", "EBV", "CoV"],
    })


# ---------------------------------------------------------------------------
# aeplots
# ---------------------------------------------------------------------------

class TestAeplots(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.pos_df = pd.read_csv(
            self.get_data_path("pos-ae.tsv"), sep="\t"
        )
        self.neg_df = pd.read_csv(
            self.get_data_path("neg-ae.tsv"), sep="\t"
        )

    def _run(self, **kwargs):
        with tempfile.TemporaryDirectory() as output_dir:
            aeplots(
                output_dir,
                pos_ae_counts=kwargs.get("pos_ae_counts", self.pos_df),
                neg_ae_counts=kwargs.get("neg_ae_counts", self.neg_df),
                **{k: v for k, v in kwargs.items()
                   if k not in ("pos_ae_counts", "neg_ae_counts")},
            )
            return os.path.exists(os.path.join(output_dir, "index.html"))

    def test_via_plugin_returns_visualization(self):
        pos_art = qiime2.Artifact.import_data("PSEAAECounts", self.pos_df)
        neg_art = qiime2.Artifact.import_data("PSEAAECounts", self.neg_df)
        viz, = self.plugin.visualizers["aeplots"](
            pos_ae_counts=[pos_art],
            neg_ae_counts=[neg_art],
        )
        self.assertEqual(str(viz.type), "Visualization")

    def test_with_colors_file(self):
        colors = qiime2.Metadata.load(self.get_data_path("species-colors.tsv"))
        pos_art = qiime2.Artifact.import_data("PSEAAECounts", self.pos_df)
        neg_art = qiime2.Artifact.import_data("PSEAAECounts", self.neg_df)
        viz, = self.plugin.visualizers["aeplots"](
            pos_ae_counts=[pos_art],
            neg_ae_counts=[neg_art],
            colors_file=colors,
        )
        self.assertEqual(str(viz.type), "Visualization")

    def test_custom_xy_access(self):
        pos = pd.DataFrame([("InfluenzaA", 3)], columns=["Species", "Events"])
        neg = pd.DataFrame([("EBV", 1)], columns=["Species", "Events"])
        with tempfile.TemporaryDirectory() as output_dir:
            aeplots(
                output_dir,
                pos_ae_counts=[pos],
                neg_ae_counts=[neg],
                xy_access=["Events", "Species"],
                xy_labels=["Count", "Organism"],
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    def test_single_species_produces_output(self):
        pos = pd.DataFrame([("InfluenzaA", 1)], columns=["Species", "Events"])
        neg = pd.DataFrame([], columns=["Species", "Events"])
        with tempfile.TemporaryDirectory() as output_dir:
            aeplots(output_dir, pos_ae_counts=[pos], neg_ae_counts=[neg])
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )


# ---------------------------------------------------------------------------
# volcano
# ---------------------------------------------------------------------------

class TestVolcano(TestPluginBase):
    package = "q2_PSEA.tests"

    def _pairs_art(self, rows=None):
        return qiime2.Artifact.import_data("PSEAPairs", _pairs_df(rows))

    def test_creates_index_html_with_explicit_xy(self):
        with tempfile.TemporaryDirectory() as output_dir:
            volcano(
                output_dir,
                x=[0.5, -0.6, 0.1],
                y=[0.01, 0.04, 0.9],
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    def test_via_plugin_returns_visualization(self):
        viz, = self.plugin.visualizers["volcano"](
            x=[0.5, -0.6, 0.1],
            y=[0.01, 0.04, 0.9],
        )
        self.assertEqual(str(viz.type), "Visualization")

    def test_log_false_produces_output(self):
        with tempfile.TemporaryDirectory() as output_dir:
            volcano(
                output_dir,
                x=[0.5, -0.6],
                y=[0.01, 0.04],
                log=False,
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    def test_with_taxa_highlights_significant_points(self):
        with tempfile.TemporaryDirectory() as output_dir:
            volcano(
                output_dir,
                x=[0.6, -0.8, 0.1],
                y=[0.01, 0.02, 0.9],
                taxa=["InfluenzaA", "EBV", "CoV"],
                x_threshold=0.4,
                y_threshold=0.05,
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    def test_with_colors_file(self):
        colors = qiime2.Metadata.load(self.get_data_path("species-colors.tsv"))
        viz, = self.plugin.visualizers["volcano"](
            x=[0.6, -0.8],
            y=[0.01, 0.02],
            taxa=["InfluenzaA", "EBV"],
            colors_file=colors,
        )
        self.assertEqual(str(viz.type), "Visualization")

    def test_with_psea_table(self):
        psea_art = qiime2.Artifact.import_data(
            "FeatureData[PSEAScores]", _psea_table_df()
        )
        viz, = self.plugin.visualizers["volcano"](
            psea_table=psea_art,
            xy_access=["NES", "p.adjust"],
            taxa_access="ID",
        )
        self.assertEqual(str(viz.type), "Visualization")

    def test_single_point_produces_output(self):
        with tempfile.TemporaryDirectory() as output_dir:
            volcano(
                output_dir,
                x=[0.5],
                y=[0.01],
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    def test_custom_axis_labels(self):
        with tempfile.TemporaryDirectory() as output_dir:
            volcano(
                output_dir,
                x=[0.5, -0.6],
                y=[0.01, 0.04],
                xy_labels=["Enrichment Score", "-log10(p)"],
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )


# ---------------------------------------------------------------------------
# zscatter
# ---------------------------------------------------------------------------

class TestZscatter(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        self.zscores_df = _zscores_df()
        self.pairs_df = _pairs_df()

    def _zscores_art(self):
        # scores-vis.tsv already has only sA and sB — no filtering needed
        raw = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        scores_art = qiime2.Artifact.import_data("FeatureTable[Zscore]", raw)
        process = self.plugin.methods["_process_scores"]
        processed_art, = process(scores=scores_art)
        return processed_art

    def test_creates_index_html(self):
        with tempfile.TemporaryDirectory() as output_dir:
            zscatter(
                output_dir,
                zscores=self.zscores_df,
                pair="sA~sB",
            )
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )

    def test_via_plugin_returns_visualization(self):
        processed_art = self._zscores_art()
        viz, = self.plugin.visualizers["zscatter"](
            zscores=processed_art,
            pair="sA~sB",
        )
        self.assertEqual(str(viz.type), "Visualization")

    def test_with_psea_tables_highlights_leading_edge(self):
        raw = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        pep_ids = raw.index.tolist()
        psea_df = pd.DataFrame({
            "NES": [2.1],
            "p.adjust": [0.01],
            "core_enrichment": [f"{pep_ids[0]}/{pep_ids[1]}"],
            "ID": ["InfluenzaA"],
        })
        processed_art = self._zscores_art()
        psea_art = qiime2.Artifact.import_data(
            "FeatureData[PSEAScores]", psea_df
        )
        viz, = self.plugin.visualizers["zscatter"](
            zscores=processed_art,
            pair="sA~sB",
            psea_table=psea_art,
            p_val_access="p.adjust",
            le_peps_access="core_enrichment",
            taxa_access="ID",
            highlight_threshold=0.05,
        )
        self.assertEqual(str(viz.type), "Visualization")

    def test_with_colors_file(self):
        raw = pd.read_csv(
            self.get_data_path("scores-vis.tsv"), sep="\t", index_col=0
        )
        pep_ids = raw.index.tolist()
        psea_df = pd.DataFrame({
            "NES": [2.1],
            "p.adjust": [0.01],
            "core_enrichment": [f"{pep_ids[0]}/{pep_ids[1]}"],
            "ID": ["InfluenzaA"],
        })
        processed_art = self._zscores_art()
        psea_art = qiime2.Artifact.import_data(
            "FeatureData[PSEAScores]", psea_df
        )
        colors = qiime2.Metadata.load(self.get_data_path("species-colors.tsv"))
        viz, = self.plugin.visualizers["zscatter"](
            zscores=processed_art,
            pair="sA~sB",
            psea_table=psea_art,
            p_val_access="p.adjust",
            le_peps_access="core_enrichment",
            taxa_access="ID",
            colors_file=colors,
        )
        self.assertEqual(str(viz.type), "Visualization")

    def test_psea_table_without_access_params_raises(self):
        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(AssertionError):
                zscatter(
                    output_dir,
                    zscores=self.zscores_df,
                    pair="sA~sB",
                    psea_table=_psea_table_df(),
                )

    def test_non_default_pair_from_larger_zscore_matrix(self):
        rng = np.random.default_rng(7)
        zscores_four = pd.DataFrame(
            rng.standard_normal((4, 20)),
            index=["sA", "sB", "sC", "sD"],
            columns=[f"pep_{i:02d}" for i in range(20)],
        )
        with tempfile.TemporaryDirectory() as output_dir:
            zscatter(output_dir, zscores=zscores_four, pair="sC~sD")
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "index.html"))
            )


if __name__ == "__main__":
    unittest.main()
