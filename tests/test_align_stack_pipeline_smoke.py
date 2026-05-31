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


if __name__ == "__main__":
    unittest.main()