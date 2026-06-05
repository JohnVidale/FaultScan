import importlib
import unittest
from unittest.mock import patch
from unittest.mock import Mock


class AlignStackSmokeTests(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("align_stack")

    def test_start_plot_timing_returns_two_floats(self):
        wall, cpu = self.mod.start_plot_timing()
        self.assertIsInstance(wall, float)
        self.assertIsInstance(cpu, float)

    def test_prepare_reference_and_phase_timing_returns_none_without_reference(self):
        with patch.object(self.mod, "select_reference_trace", return_value=(None, None)):
            out = self.mod.prepare_reference_and_phase_timing(
                st_comp=[],
                name2ll={},
                raw_limits_by_station={},
                event_depth=10.0,
                origin=None,
                align_phase_name="P",
            )
        self.assertIsNone(out)

    def test_prepare_reference_and_phase_timing_returns_none_without_phase(self):
        fake_ref = object()
        with patch.object(self.mod, "select_reference_trace", return_value=("STA", fake_ref)), patch.object(
            self.mod,
            "print_reference_summary",
            return_value=None,
        ), patch.object(
            self.mod,
            "compute_phase_travel_times",
            return_value=(1.0, 2.0, None, None, None),
        ):
            out = self.mod.prepare_reference_and_phase_timing(
                st_comp=[fake_ref],
                name2ll={"STA": (0.0, 0.0)},
                raw_limits_by_station={},
                event_depth=10.0,
                origin=None,
                align_phase_name="S",
            )
        self.assertIsNone(out)

    def test_prepare_reference_and_phase_timing_success(self):
        fake_ref = object()
        with patch.object(self.mod, "select_reference_trace", return_value=("STA", fake_ref)), patch.object(
            self.mod,
            "print_reference_summary",
            return_value=None,
        ), patch.object(
            self.mod,
            "compute_phase_travel_times",
            return_value=(1.0, 2.0, 3.0, 4.0, 5.0),
        ):
            out = self.mod.prepare_reference_and_phase_timing(
                st_comp=[fake_ref],
                name2ll={"STA": (0.0, 0.0)},
                raw_limits_by_station={},
                event_depth=10.0,
                origin=None,
                align_phase_name="S",
            )

        self.assertIsNotNone(out)
        self.assertEqual(len(out), 8)
        self.assertEqual(out[0], "STA")
        self.assertEqual(out[1], fake_ref)
        self.assertEqual(out[2], 1.0)
        self.assertEqual(out[3], 2.0)
        self.assertEqual(out[7], 5.0)

    def test_prepare_reference_and_phase_timing_returns_none_without_station_id(self):
        fake_ref = object()
        with patch.object(self.mod, "select_reference_trace", return_value=(None, fake_ref)):
            out = self.mod.prepare_reference_and_phase_timing(
                st_comp=[fake_ref],
                name2ll={"STA": (0.0, 0.0)},
                raw_limits_by_station={},
                event_depth=10.0,
                origin=None,
                align_phase_name="P",
            )
        self.assertIsNone(out)

    def test_prepare_stream_reference_context_returns_none_when_no_traces(self):
        fake_stream = []
        ref_phase_timing = ("STA", object(), 1.0, 2.0, None, None, 1.5, 1.5)
        with patch.object(self.mod, "select_component_stream", return_value=(fake_stream, "Z")), patch.object(
            self.mod,
            "prepare_reference_and_phase_timing",
            return_value=ref_phase_timing,
        ):
            out = self.mod.prepare_stream_reference_context(
                st_window=object(),
                sel_comp="Z",
                channel="DPZ",
                name2ll={"STA": (0.0, 0.0)},
                eve_lat=0.0,
                eve_lon=0.0,
                raw_limits_by_station={},
                event_depth=10.0,
                origin=None,
                align_phase_name="S",
            )
        self.assertIsNone(out)

    def test_prepare_stream_reference_context_returns_none_when_no_ref_phase_timing(self):
        fake_stream = [object(), object()]
        with patch.object(self.mod, "select_component_stream", return_value=(fake_stream, "Z")), patch.object(
            self.mod,
            "prepare_reference_and_phase_timing",
            return_value=None,
        ):
            out = self.mod.prepare_stream_reference_context(
                st_window=object(),
                sel_comp="Z",
                channel="DPZ",
                name2ll={"STA": (0.0, 0.0)},
                eve_lat=0.0,
                eve_lon=0.0,
                raw_limits_by_station={},
                event_depth=10.0,
                origin=None,
                align_phase_name="S",
            )
        self.assertIsNone(out)

    def test_prepare_stream_reference_context_success_shape(self):
        fake_stream = [object(), object()]
        fake_ref = object()
        ref_phase_timing = ("STA", fake_ref, 1.0, 2.0, None, None, 1.5, 1.5)
        with patch.object(self.mod, "select_component_stream", return_value=(fake_stream, "Z")), patch.object(
            self.mod,
            "prepare_reference_and_phase_timing",
            return_value=ref_phase_timing,
        ):
            out = self.mod.prepare_stream_reference_context(
                st_window=object(),
                sel_comp="Z",
                channel="DPZ",
                name2ll={"STA": (0.0, 0.0)},
                eve_lat=0.0,
                eve_lon=0.0,
                raw_limits_by_station={},
                event_depth=10.0,
                origin=None,
                align_phase_name="S",
            )

        self.assertIsNotNone(out)
        self.assertEqual(len(out), 11)
        self.assertEqual(out[0], fake_stream)
        self.assertEqual(out[1], "Z")
        self.assertEqual(out[2], "STA")
        self.assertEqual(out[3], fake_ref)

    def test_load_event_context_and_waveforms_returns_none_when_no_waveforms(self):
        with patch.object(
            self.mod,
            "load_event_metadata",
            return_value=(10.0, 35.0, -117.0, None),
        ), patch.object(
            self.mod,
            "make_event_output_dir",
            return_value="/tmp/out",
        ), patch.object(
            self.mod,
            "load_station_lookup",
            return_value={"STA": (35.0, -117.0)},
        ), patch.object(
            self.mod,
            "read_waveforms_for_event",
            return_value=(None, None),
        ):
            out = self.mod.load_event_context_and_waveforms(
                eve_id="E1",
                channel="DPZ",
                process_as_three_comp=False,
                horizontal_window_cache={},
                horizontal_raw_limits_cache={},
            )
        self.assertIsNone(out)

    def test_load_event_context_and_waveforms_success_shape(self):
        fake_stream = object()
        fake_limits = {"STA": (0.0, 1.0)}
        with patch.object(
            self.mod,
            "load_event_metadata",
            return_value=(10.0, 35.0, -117.0, None),
        ), patch.object(
            self.mod,
            "make_event_output_dir",
            return_value="/tmp/out",
        ), patch.object(
            self.mod,
            "load_station_lookup",
            return_value={"STA": (35.0, -117.0)},
        ), patch.object(
            self.mod,
            "read_waveforms_for_event",
            return_value=(fake_stream, fake_limits),
        ):
            out = self.mod.load_event_context_and_waveforms(
                eve_id="E1",
                channel="DPZ",
                process_as_three_comp=False,
                horizontal_window_cache={},
                horizontal_raw_limits_cache={},
            )

        self.assertIsNotNone(out)
        self.assertEqual(len(out), 8)
        self.assertEqual(out[0], 10.0)
        self.assertEqual(out[6], fake_stream)
        self.assertEqual(out[7], fake_limits)

    def test_select_component_stream_rotates_for_r_or_t(self):
        fake_stream = object()
        with patch.object(
            self.mod,
            "rotate_horizontals_to_component",
            return_value=(fake_stream, "R"),
        ) as mock_rotate:
            out_stream, out_plot = self.mod.select_component_stream(
                st_window=object(),
                sel_comp="R",
                channel="DPN",
                name2ll={"STA": (0.0, 0.0)},
                eve_lat=0.0,
                eve_lon=0.0,
            )

        mock_rotate.assert_called_once()
        self.assertIs(out_stream, fake_stream)
        self.assertEqual(out_plot, "R")

    def test_select_component_stream_uses_direct_channel_for_z(self):
        st_window = Mock()
        selected_stream = object()
        st_window.select.return_value = selected_stream

        out_stream, out_plot = self.mod.select_component_stream(
            st_window=st_window,
            sel_comp="Z",
            channel="DPZ",
            name2ll={"STA": (0.0, 0.0)},
            eve_lat=0.0,
            eve_lon=0.0,
        )

        st_window.select.assert_called_once_with(channel="DPZ")
        self.assertIs(out_stream, selected_stream)
        self.assertEqual(out_plot, "Z")

    def test_run_alignment_and_unpack_tuple_order(self):
        sentinel = {
            "npts": 1,
            "sample_rate": 2,
            "move_limit_samples": 3,
            "win_start": 4,
            "win_end": 5,
            "calc_shifts": 6,
            "aligned_stack": 7,
            "selected_aligned_stack": 8,
            "selected_ids": 9,
            "station_corr": 10,
            "n_pass_window": 11,
            "pass_window_ids": 12,
            "snippet_by_station": 13,
            "ref_window": 14,
            "selected_rows": 15,
            "rejected_rows": 16,
            "station_shifts": 17,
            "aligned_traces_by_station": 18,
            "t_abs": 19,
            "mask": 20,
            "stack_vec": 21,
        }
        expected = tuple(sentinel[k] for k in (
            "npts",
            "sample_rate",
            "move_limit_samples",
            "win_start",
            "win_end",
            "calc_shifts",
            "aligned_stack",
            "selected_aligned_stack",
            "selected_ids",
            "station_corr",
            "n_pass_window",
            "pass_window_ids",
            "snippet_by_station",
            "ref_window",
            "selected_rows",
            "rejected_rows",
            "station_shifts",
            "aligned_traces_by_station",
            "t_abs",
            "mask",
            "stack_vec",
        ))

        with patch.object(self.mod, "compute_alignment_products", return_value=sentinel):
            out = self.mod.run_alignment_and_unpack(
                st_comp=object(),
                ref_trace=object(),
                ref_station_id="1",
                name2ll={"1": (0.0, 0.0)},
                eve_lat=0.0,
                eve_lon=0.0,
                event_depth=10.0,
                align_phase_name="P",
                t_ref=1.0,
            )

        self.assertEqual(out, expected)


if __name__ == "__main__":
    unittest.main()