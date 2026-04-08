import os
import pathlib
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.actions.visualizers import aeplots, volcano, zscatter


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

class _FakePairsFormat:
    """
    Minimal stand-in for PSEAPairsDirFmt; exposes .path like the real one.
    """

    def __init__(self, directory):
        self.path = pathlib.Path(directory)


class _FakeZscoresFmt:
    """Minimal stand-in for PepsirfContingencyTSVFormat."""

    def __init__(self, df):
        self._df = df

    def view(self, type_):
        return self._df


def _write_psea_table(directory, pair_str, p_adjust=0.01):
    """Write a minimal PSEA result TSV to *directory* for *pair_str*."""
    df = pd.DataFrame({
        "ID": ["sp1", "sp2"],
        "NES": [2.1, -1.3],
        "p.adjust": [p_adjust, 0.03],
        "species_name": ["InfluenzaA", "EBV"],
        "core_enrichment": ["pep_00/pep_01", "pep_04"],
    })
    path = os.path.join(directory, f"{pair_str}_psea_table.tsv")
    df.to_csv(path, sep="\t", index=False)
    return path


def _ae_df(species_events):
    rows = [{"Species": sp, "Events": ev} for sp, ev in species_events.items()]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plugin registration tests
# ---------------------------------------------------------------------------

class TestVisualizerRegistration(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_volcano_registered_as_visualizer(self):
        self.assertIn("volcano", self.plugin.visualizers)

    def test_zscatter_registered_as_visualizer(self):
        self.assertIn("zscatter", self.plugin.visualizers)

    def test_aeplots_registered_as_visualizer(self):
        self.assertIn("aeplots", self.plugin.visualizers)

    def test_volcano_pairs_is_input_not_parameter(self):
        sig = self.plugin.visualizers["volcano"].signature
        self.assertIn("pairs", sig.inputs)
        self.assertNotIn("pairs", sig.parameters)

    def test_volcano_no_pairs_file_parameter(self):
        sig = self.plugin.visualizers["volcano"].signature
        self.assertNotIn("pairs_file", sig.parameters)

    def test_zscatter_pairs_is_input_not_parameter(self):
        sig = self.plugin.visualizers["zscatter"].signature
        self.assertIn("pairs", sig.inputs)
        self.assertNotIn("pairs", sig.parameters)

    def test_zscatter_zscores_is_input(self):
        sig = self.plugin.visualizers["zscatter"].signature
        self.assertIn("zscores", sig.inputs)

    def test_zscatter_no_pairs_file_parameter(self):
        sig = self.plugin.visualizers["zscatter"].signature
        self.assertNotIn("pairs_file", sig.parameters)

    def test_aeplots_has_no_artifact_inputs(self):
        sig = self.plugin.visualizers["aeplots"].signature
        self.assertEqual(len(sig.inputs), 0)

    def test_aeplots_file_params_are_parameters(self):
        sig = self.plugin.visualizers["aeplots"].signature
        self.assertIn("pos_nes_ae_file", sig.parameters)
        self.assertIn("neg_nes_ae_file", sig.parameters)


# ---------------------------------------------------------------------------
# volcano functional tests
# ---------------------------------------------------------------------------

class TestVolcano(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        # _FakePairsFormat whose .path / "pairs.tsv" resolves to the fixture
        data_dir = os.path.dirname(self.get_data_path("pairs.tsv"))
        self.fake_pairs = _FakePairsFormat(data_dir)

    def test_reads_pairs_from_artifact_path(self):
        """
        pairs.path / "pairs.tsv" is opened; a raw string path is NOT used.
        """
        opened_paths = []
        real_open = open

        def spy_open(path, *args, **kwargs):
            opened_paths.append(str(path))
            return real_open(path, *args, **kwargs)

        with tempfile.TemporaryDirectory() as xy_dir, \
             patch("q2_PSEA.actions.visualizers.alt"), \
             patch("builtins.open", side_effect=spy_open):
            _write_psea_table(xy_dir, "sample1~sample2")
            volcano(
                output_dir="/fake",
                pairs=self.fake_pairs,
                psea_tables=xy_dir,
                xy_access=["NES", "p.adjust"],
            )

        expected = str(self.fake_pairs.path / "pairs.tsv")
        self.assertIn(expected, opened_paths)

    def test_index_html_path_passed_to_save(self):
        with tempfile.TemporaryDirectory() as xy_dir, \
             patch("q2_PSEA.actions.visualizers.alt") as mock_alt:
            _write_psea_table(xy_dir, "sample1~sample2")
            volcano(
                output_dir="/fake_out",
                pairs=self.fake_pairs,
                psea_tables=xy_dir,
                xy_access=["NES", "p.adjust"],
            )

        # final_chart = alt.vconcat(...); final_chart.save(<path>)
        save_path = mock_alt.vconcat.return_value.save.call_args[0][0]
        self.assertEqual(save_path, os.path.join("/fake_out", "index.html"))

    def test_files_without_tilde_are_skipped(self):
        """Files in xy_dir that lack '~' must not contribute chart data."""
        with tempfile.TemporaryDirectory() as xy_dir, \
             patch("q2_PSEA.actions.visualizers.alt") as mock_alt:
            # a decoy file with no tilde — should be silently ignored
            pd.DataFrame({"NES": [9.9], "p.adjust": [0.001]}).to_csv(
                os.path.join(xy_dir, "no_tilde_decoy.tsv"), sep="\t"
            )
            # one valid pair file
            _write_psea_table(xy_dir, "sample1~sample2")
            volcano(
                output_dir="/fake",
                pairs=self.fake_pairs,
                psea_tables=xy_dir,
                xy_access=["NES", "p.adjust"],
            )

        # Chart construction must have been attempted
        self.assertTrue(mock_alt.Chart.called)

    def test_x_y_lists_accepted_without_xy_dir(self):
        """Supplying x/y lists directly instead of xy_dir must work."""
        with patch("q2_PSEA.actions.visualizers.alt"):
            # must not raise
            volcano(
                output_dir="/fake",
                pairs=self.fake_pairs,
                x=[2.1, -1.3],
                y=[0.01, 0.04],
                xy_labels=["NES", "Adjusted p-value"],
            )

    def test_vis_outputs_dir_triggers_per_pair_saves(self):
        with tempfile.TemporaryDirectory() as xy_dir, \
             tempfile.TemporaryDirectory() as vis_dir, \
             patch("q2_PSEA.actions.visualizers.alt") as mock_alt, \
             patch("os.mkdir"):
            _write_psea_table(xy_dir, "sample1~sample2")
            volcano(
                output_dir="/fake",
                pairs=self.fake_pairs,
                psea_tables=xy_dir,
                xy_access=["NES", "p.adjust"],
                vis_outputs_dir=vis_dir,
            )

        # save should be called at least twice: index.html + per-pair file(s)
        total_saves = sum(
            1 for c in mock_alt.mock_calls if ".save(" in str(c)
        )
        self.assertGreaterEqual(total_saves, 2)


# ---------------------------------------------------------------------------
# zscatter functional tests
# ---------------------------------------------------------------------------

class TestZscatter(TestPluginBase):
    package = "q2_PSEA.tests"

    def setUp(self):
        super().setUp()
        # pairs.tsv has "sample1" / "sample2" as pair sample names.
        # zscatter calls zscores.view(pd.DataFrame) then .transpose(), and
        # afterwards indexes columns by sample name.  So .view() must return
        # a DataFrame with samples as ROWS (pre-transpose) so that after the
        # transpose samples become columns.
        pep_ids = [f"pep_{i:02d}" for i in range(15)]
        self.scores = pd.DataFrame(
            {
                "sample1": [i * 0.5 for i in range(15)],
                "sample2": [i * 0.5 + 0.3 for i in range(15)],
            },
            index=pep_ids,
        )
        # The format is expected to give samples-as-rows before the transpose
        data_dir = os.path.dirname(self.get_data_path("pairs.tsv"))
        self.fake_pairs = _FakePairsFormat(data_dir)
        self.fake_zscores = _FakeZscoresFmt(self.scores.T)

    def test_reads_pairs_from_artifact_path(self):
        opened_paths = []
        real_open = open

        def spy_open(path, *args, **kwargs):
            opened_paths.append(str(path))
            return real_open(path, *args, **kwargs)

        with patch("q2_PSEA.actions.visualizers.alt"), \
             patch("builtins.open", side_effect=spy_open):
            zscatter(
                output_dir="/fake",
                zscores=self.fake_zscores,
                pairs=self.fake_pairs,
            )

        expected = str(self.fake_pairs.path / "pairs.tsv")
        self.assertIn(expected, opened_paths)

    def test_zscores_view_called_with_dataframe_type(self):
        view_args = []
        orig_view = self.fake_zscores.view

        def spy_view(type_):
            view_args.append(type_)
            return orig_view(type_)

        self.fake_zscores.view = spy_view
        with patch("q2_PSEA.actions.visualizers.alt"):
            zscatter(
                output_dir="/fake",
                zscores=self.fake_zscores,
                pairs=self.fake_pairs,
            )

        self.assertEqual(len(view_args), 1)
        self.assertIs(view_args[0], pd.DataFrame)

    def test_index_html_path_passed_to_save(self):
        with patch("q2_PSEA.actions.visualizers.alt") as mock_alt:
            zscatter(
                output_dir="/fake_out",
                zscores=self.fake_zscores,
                pairs=self.fake_pairs,
            )

        save_path = mock_alt.vconcat.return_value.save.call_args[0][0]
        self.assertEqual(save_path, os.path.join("/fake_out", "index.html"))

    def test_highlight_data_requires_p_val_access(self):
        with self.assertRaises(AssertionError):
            with patch("q2_PSEA.actions.visualizers.alt"):
                zscatter(
                    output_dir="/fake",
                    zscores=self.fake_zscores,
                    pairs=self.fake_pairs,
                    psea_tables="/some/dir",
                    # p_val_access omitted intentionally
                )

    def test_highlight_data_requires_le_peps_access(self):
        with self.assertRaises(AssertionError):
            with patch("q2_PSEA.actions.visualizers.alt"):
                zscatter(
                    output_dir="/fake",
                    zscores=self.fake_zscores,
                    pairs=self.fake_pairs,
                    psea_tables="/some/dir",
                    p_val_access="p.adjust",
                    # le_peps_access omitted intentionally
                )

    def test_highlight_data_requires_taxa_access(self):
        with self.assertRaises(AssertionError):
            with patch("q2_PSEA.actions.visualizers.alt"):
                zscatter(
                    output_dir="/fake",
                    zscores=self.fake_zscores,
                    pairs=self.fake_pairs,
                    psea_tables="/some/dir",
                    p_val_access="p.adjust",
                    le_peps_access="core_enrichment",
                    # taxa_access omitted intentionally
                )

    def test_spline_file_is_read_when_provided(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tsv", delete=False
        ) as f:
            f.write("x\ty\tpair\n0.5\t0.6\tsample1~sample2\n")
            spline_path = f.name

        read_calls = []
        orig_read_csv = pd.read_csv

        def spy_read_csv(path, *a, **kw):
            read_calls.append(str(path))
            return orig_read_csv(path, *a, **kw)

        try:
            with patch("q2_PSEA.actions.visualizers.alt"), \
                 patch("q2_PSEA.actions.visualizers.pd.read_csv",
                       side_effect=spy_read_csv):
                zscatter(
                    output_dir="/fake",
                    zscores=self.fake_zscores,
                    pairs=self.fake_pairs,
                    spline_file=spline_path,
                )
        finally:
            os.unlink(spline_path)

        self.assertIn(spline_path, read_calls)

    def test_highlight_with_sig_taxa_extends_chart(self):
        """When highlight_data is a directory with a matching file and
        there are significant rows, a highlight chart layer is added."""
        with tempfile.TemporaryDirectory() as highlight_dir, \
             patch("q2_PSEA.actions.visualizers.alt") as mock_alt:
            _write_psea_table(highlight_dir, "sample1~sample2", p_adjust=0.001)
            zscatter(
                output_dir="/fake",
                zscores=self.fake_zscores,
                pairs=self.fake_pairs,
                psea_tables=highlight_dir,
                p_val_access="p.adjust",
                le_peps_access="core_enrichment",
                taxa_access="species_name",
                highlight_threshold=0.05,
            )

        # alt.layer should have been called for the highlight overlay
        self.assertTrue(mock_alt.layer.called)


# ---------------------------------------------------------------------------
# aeplots functional tests
# ---------------------------------------------------------------------------

class TestAeplots(TestPluginBase):
    package = "q2_PSEA.tests"

    def _write_ae_files(self, directory, pos_data, neg_data):
        pos_path = os.path.join(directory, "pos.tsv")
        neg_path = os.path.join(directory, "neg.tsv")
        _ae_df(pos_data).to_csv(pos_path, sep="\t", index=False)
        _ae_df(neg_data).to_csv(neg_path, sep="\t", index=False)
        return pos_path, neg_path

    def test_index_html_path_passed_to_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pos, neg = self._write_ae_files(
                tmpdir, {"InfluenzaA": 5}, {"EBV": 2}
            )
            with patch("q2_PSEA.actions.visualizers.alt") as mock_alt:
                aeplots(
                    output_dir="/fake_out",
                    pos_nes_ae_file=pos,
                    neg_nes_ae_file=neg,
                )

        # (bar_chart + text).facet().resolve_scale().save(path)
        bar_chart = mock_alt.Chart.return_value.mark_bar.return_value\
            .encode.return_value
        final_chart = bar_chart.__add__.return_value\
            .facet.return_value.resolve_scale.return_value
        final_chart.save.assert_called_with(
            os.path.join("/fake_out", "index.html")
        )

    def test_reads_both_ae_files(self):
        read_calls = []
        orig_read_csv = pd.read_csv

        def spy_read_csv(path, *a, **kw):
            read_calls.append(str(path))
            return orig_read_csv(path, *a, **kw)

        with tempfile.TemporaryDirectory() as tmpdir:
            pos, neg = self._write_ae_files(
                tmpdir, {"InfluenzaA": 5}, {"EBV": 2}
            )
            with patch("q2_PSEA.actions.visualizers.alt"), \
                 patch("q2_PSEA.actions.visualizers.pd.read_csv",
                       side_effect=spy_read_csv):
                aeplots(
                    output_dir="/fake",
                    pos_nes_ae_file=pos,
                    neg_nes_ae_file=neg,
                )

        self.assertIn(pos, read_calls)
        self.assertIn(neg, read_calls)

    def test_nes_column_added_to_both_dataframes(self):
        """pos rows get NES="Positive", neg rows get NES="Negative"."""
        captured_df = []

        with tempfile.TemporaryDirectory() as tmpdir:
            pos, neg = self._write_ae_files(
                tmpdir, {"InfluenzaA": 5}, {"EBV": 2}
            )
            with patch("q2_PSEA.actions.visualizers.alt") as mock_alt:
                # capture the DataFrame passed to alt.Chart(ae_df)
                def capture_chart(df, *a, **kw):
                    captured_df.append(df)
                    return MagicMock()

                mock_alt.Chart.side_effect = capture_chart
                aeplots(
                    output_dir="/fake",
                    pos_nes_ae_file=pos,
                    neg_nes_ae_file=neg,
                )

        self.assertEqual(len(captured_df), 1)
        ae_df = captured_df[0]
        self.assertIn("NES", ae_df.columns)
        self.assertIn("Positive", ae_df["NES"].values)
        self.assertIn("Negative", ae_df["NES"].values)

    def test_vis_outputs_dir_triggers_second_save(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pos, neg = self._write_ae_files(
                tmpdir, {"InfluenzaA": 5}, {"EBV": 2}
            )
            with patch("q2_PSEA.actions.visualizers.alt") as mock_alt:
                aeplots(
                    output_dir="/fake",
                    pos_nes_ae_file=pos,
                    neg_nes_ae_file=neg,
                    vis_outputs_dir=tmpdir,
                )

        bar_chart = mock_alt.Chart.return_value.mark_bar.return_value\
            .encode.return_value
        final_chart = bar_chart.__add__.return_value\
            .facet.return_value.resolve_scale.return_value
        # save is called twice: index.html and aeplots.html
        self.assertEqual(final_chart.save.call_count, 2)
        second_call_path = final_chart.save.call_args_list[1][0][0]
        self.assertTrue(second_call_path.endswith("aeplots.html"))


if __name__ == "__main__":
    unittest.main()
