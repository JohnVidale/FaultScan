import importlib
import unittest
from unittest.mock import patch


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


if __name__ == "__main__":
    unittest.main()