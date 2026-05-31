import importlib
import unittest
from unittest.mock import patch

import numpy as np


class AlignStackPipelineSmokeTests(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("align_stack")

    def test_run_pipeline_single_component_branch(self):
        event_context = (
            10.0,
            35.0,
            -117.0,
            None,
            "/tmp",
            {"1": (35.0, -117.0)},
            object(),
            {},
        )
        stream_ref_context = (
            object(),
            "Z",
            "1",
            object(),
            1.0,
            2.0,
            None,
            None,
            None,
            1.5,
            1,
        )
        alignment_tuple = (
            10,
            20.0,
            2,
            1,
            5,
            {"1": 0.0},
            np.array([0.1]),
            np.array([0.1]),
            {"1"},
            {"1": 0.9},
            1,
            {"1"},
            {"1": np.array([0.1])},
            np.array([0.1]),
            [("1", 0.9, np.array([0.1]))],
            [],
            {"1": 0.0},
            {"1": np.array([0.1])},
            np.array([0.0]),
            np.array([True]),
            np.array([0.1]),
        )

        with patch.object(self.mod, "events", ["E1"]), patch.object(self.mod, "all_channels", False), patch.object(
            self.mod,
            "get_component_selection",
            return_value=(["DPZ"], False, ["Z"]),
        ), patch.object(
            self.mod,
            "load_event_context_and_waveforms",
            return_value=event_context,
        ), patch.object(
            self.mod,
            "prepare_stream_reference_context",
            return_value=stream_ref_context,
        ), patch.object(
            self.mod,
            "preprocess_traces_bandpass",
            return_value=None,
        ) as mock_preprocess, patch.object(
            self.mod,
            "run_alignment_and_unpack",
            return_value=alignment_tuple,
        ), patch.object(
            self.mod,
            "start_plot_timing",
            return_value=(1.0, 2.0),
        ), patch.object(
            self.mod,
            "plot_stage_stacks",
            return_value=None,
        ) as mock_plot_stage, patch.object(
            self.mod,
            "plot_record_section_and_stack",
            return_value="FIG",
        ), patch.object(
            self.mod,
            "plot_single_component_products",
            return_value=None,
        ) as mock_single, patch.object(
            self.mod,
            "finalize_single_component_plotting",
            return_value=None,
        ) as mock_finalize, patch.object(
            self.mod,
            "store_three_component_data",
            return_value=None,
        ) as mock_store:
            self.mod.run_pipeline()

        mock_preprocess.assert_called_once()
        mock_plot_stage.assert_called_once()
        mock_single.assert_called_once()
        mock_finalize.assert_called_once()
        mock_store.assert_not_called()

    def test_run_pipeline_no_events_short_circuits(self):
        with patch.object(self.mod, "events", []), patch.object(self.mod, "all_channels", False), patch.object(
            self.mod,
            "get_component_selection",
            return_value=("DPZ".split(), False, "Z".split()),
        ), patch.object(
            self.mod,
            "load_event_context_and_waveforms",
            return_value=None,
        ) as mock_load, patch.object(
            self.mod,
            "prepare_stream_reference_context",
            return_value=None,
        ) as mock_prepare, patch.object(
            self.mod,
            "run_alignment_and_unpack",
            return_value=(),
        ) as mock_align, patch.object(
            self.mod,
            "plot_record_section_and_stack",
            return_value=None,
        ) as mock_plot_record:
            self.mod.run_pipeline()

        mock_load.assert_not_called()
        mock_prepare.assert_not_called()
        mock_align.assert_not_called()
        mock_plot_record.assert_not_called()

    def test_run_pipeline_skips_when_event_context_missing(self):
        with patch.object(self.mod, "events", ["E1"]), patch.object(self.mod, "all_channels", False), patch.object(
            self.mod,
            "get_component_selection",
            return_value=("DPZ".split(), False, "Z".split()),
        ), patch.object(
            self.mod,
            "load_event_context_and_waveforms",
            return_value=None,
        ) as mock_load, patch.object(
            self.mod,
            "prepare_stream_reference_context",
            return_value=None,
        ) as mock_prepare, patch.object(
            self.mod,
            "preprocess_traces_bandpass",
            return_value=None,
        ) as mock_preprocess:
            self.mod.run_pipeline()

        mock_load.assert_called_once()
        mock_prepare.assert_not_called()
        mock_preprocess.assert_not_called()

    def test_run_pipeline_skips_when_stream_reference_context_missing(self):
        event_context = (
            10.0,
            35.0,
            -117.0,
            None,
            "/tmp",
            {"1": (35.0, -117.0)},
            object(),
            {},
        )

        with patch.object(self.mod, "events", ["E1"]), patch.object(self.mod, "all_channels", False), patch.object(
            self.mod,
            "get_component_selection",
            return_value=("DPZ".split(), False, "Z".split()),
        ), patch.object(
            self.mod,
            "load_event_context_and_waveforms",
            return_value=event_context,
        ) as mock_load, patch.object(
            self.mod,
            "prepare_stream_reference_context",
            return_value=None,
        ) as mock_prepare, patch.object(
            self.mod,
            "preprocess_traces_bandpass",
            return_value=None,
        ) as mock_preprocess:
            self.mod.run_pipeline()

        mock_load.assert_called_once()
        mock_prepare.assert_called_once()
        mock_preprocess.assert_not_called()

    def test_run_pipeline_three_component_combined_branch(self):
        event_context = (
            10.0,
            35.0,
            -117.0,
            None,
            "/tmp",
            {"1": (35.0, -117.0)},
            object(),
            {},
        )
        stream_ref_context = (
            object(),
            "Z",
            "1",
            object(),
            1.0,
            2.0,
            None,
            None,
            None,
            1.5,
            1,
        )
        alignment_tuple = (
            10,
            20.0,
            2,
            1,
            5,
            {"1": 0.0},
            np.array([0.1]),
            np.array([0.1]),
            {"1"},
            {"1": 0.9},
            1,
            {"1"},
            {"1": np.array([0.1])},
            np.array([0.1]),
            [("1", 0.9, np.array([0.1]))],
            [],
            {"1": 0.0},
            {"1": np.array([0.1])},
            np.array([0.0]),
            np.array([True]),
            np.array([0.1]),
        )

        def fake_store(**kwargs):
            comp_key = kwargs["sel_comp"] if kwargs["channel"] != "DPZ" else "DPZ"
            kwargs["all_component_data"][comp_key] = {"ok": True}

        with patch.object(self.mod, "events", ["E1"]), patch.object(self.mod, "all_channels", True), patch.object(
            self.mod,
            "get_component_selection",
            return_value=("DPZ DPN DPE".split(), True, "Z R T".split()),
        ), patch.object(
            self.mod,
            "load_event_context_and_waveforms",
            return_value=event_context,
        ), patch.object(
            self.mod,
            "prepare_stream_reference_context",
            return_value=stream_ref_context,
        ), patch.object(
            self.mod,
            "preprocess_traces_bandpass",
            return_value=None,
        ), patch.object(
            self.mod,
            "run_alignment_and_unpack",
            return_value=alignment_tuple,
        ), patch.object(
            self.mod,
            "start_plot_timing",
            return_value=(1.0, 2.0),
        ), patch.object(
            self.mod,
            "plot_stage_stacks",
            return_value=None,
        ) as mock_plot_stage, patch.object(
            self.mod,
            "plot_record_section_and_stack",
            return_value="FIG",
        ), patch.object(
            self.mod,
            "store_three_component_data",
            side_effect=fake_store,
        ) as mock_store, patch.object(
            self.mod,
            "setup_three_component_record_figure",
            return_value=("FIG3", "GS"),
        ) as mock_setup, patch.object(
            self.mod,
            "get_three_component_plot_context",
            return_value=(
                ["DPZ", "R", "T"],
                ["Z", "R", "T"],
                "E1",
                "S",
                -10.0,
                20.0,
                np.array([0.0]),
                np.array([True]),
                20.0,
                None,
            ),
        ), patch.object(
            self.mod,
            "render_and_collect_three_component_stacks",
            return_value=({"DPZ": np.array([0.1]), "R": np.array([0.1]), "T": np.array([0.1])}, np.array([0.0]), np.array([True])),
        ), patch.object(
            self.mod,
            "persist_three_component_outputs",
            return_value="/tmp",
        ) as mock_persist, patch.object(
            self.mod,
            "plot_three_component_summary_products",
            return_value=None,
        ) as mock_summary, patch.object(
            self.mod,
            "finalize_three_component_plotting",
            return_value=None,
        ) as mock_finalize3, patch.object(
            self.mod,
            "plot_single_component_products",
            return_value=None,
        ) as mock_single:
            self.mod.run_pipeline()

        self.assertEqual(mock_store.call_count, 3)
        mock_plot_stage.assert_not_called()
        mock_setup.assert_called_once()
        mock_persist.assert_called_once()
        mock_summary.assert_called_once()
        mock_finalize3.assert_called_once()
        mock_single.assert_not_called()

    def test_run_pipeline_three_component_skips_combined_when_incomplete(self):
        event_context = (
            10.0,
            35.0,
            -117.0,
            None,
            "/tmp",
            {"1": (35.0, -117.0)},
            object(),
            {},
        )
        stream_ref_context = (
            object(),
            "Z",
            "1",
            object(),
            1.0,
            2.0,
            None,
            None,
            None,
            1.5,
            1,
        )
        alignment_tuple = (
            10,
            20.0,
            2,
            1,
            5,
            {"1": 0.0},
            np.array([0.1]),
            np.array([0.1]),
            {"1"},
            {"1": 0.9},
            1,
            {"1"},
            {"1": np.array([0.1])},
            np.array([0.1]),
            [("1", 0.9, np.array([0.1]))],
            [],
            {"1": 0.0},
            {"1": np.array([0.1])},
            np.array([0.0]),
            np.array([True]),
            np.array([0.1]),
        )

        def fake_store_incomplete(**kwargs):
            if kwargs["channel"] == "DPZ":
                kwargs["all_component_data"]["DPZ"] = {"ok": True}
            elif kwargs["sel_comp"] == "R":
                kwargs["all_component_data"]["R"] = {"ok": True}
            # Deliberately skip adding T to keep payload incomplete.

        with patch.object(self.mod, "events", ["E1"]), patch.object(self.mod, "all_channels", True), patch.object(
            self.mod,
            "get_component_selection",
            return_value=("DPZ DPN DPE".split(), True, "Z R T".split()),
        ), patch.object(
            self.mod,
            "load_event_context_and_waveforms",
            return_value=event_context,
        ), patch.object(
            self.mod,
            "prepare_stream_reference_context",
            return_value=stream_ref_context,
        ), patch.object(
            self.mod,
            "preprocess_traces_bandpass",
            return_value=None,
        ), patch.object(
            self.mod,
            "run_alignment_and_unpack",
            return_value=alignment_tuple,
        ), patch.object(
            self.mod,
            "start_plot_timing",
            return_value=(1.0, 2.0),
        ), patch.object(
            self.mod,
            "plot_stage_stacks",
            return_value=None,
        ) as mock_plot_stage, patch.object(
            self.mod,
            "plot_record_section_and_stack",
            return_value="FIG",
        ), patch.object(
            self.mod,
            "store_three_component_data",
            side_effect=fake_store_incomplete,
        ) as mock_store, patch.object(
            self.mod,
            "setup_three_component_record_figure",
            return_value=("FIG3", "GS"),
        ) as mock_setup, patch.object(
            self.mod,
            "get_three_component_plot_context",
            return_value=(
                ["DPZ", "R", "T"],
                ["Z", "R", "T"],
                "E1",
                "S",
                -10.0,
                20.0,
                np.array([0.0]),
                np.array([True]),
                20.0,
                None,
            ),
        ), patch.object(
            self.mod,
            "render_and_collect_three_component_stacks",
            return_value=({"DPZ": np.array([0.1]), "R": np.array([0.1]), "T": np.array([0.1])}, np.array([0.0]), np.array([True])),
        ), patch.object(
            self.mod,
            "persist_three_component_outputs",
            return_value="/tmp",
        ) as mock_persist, patch.object(
            self.mod,
            "plot_three_component_summary_products",
            return_value=None,
        ) as mock_summary, patch.object(
            self.mod,
            "finalize_three_component_plotting",
            return_value=None,
        ) as mock_finalize3:
            self.mod.run_pipeline()

        self.assertEqual(mock_store.call_count, 3)
        mock_plot_stage.assert_not_called()
        mock_setup.assert_not_called()
        mock_persist.assert_not_called()
        mock_summary.assert_not_called()
        mock_finalize3.assert_not_called()


if __name__ == "__main__":
    unittest.main()