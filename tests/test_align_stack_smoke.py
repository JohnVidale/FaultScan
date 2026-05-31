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


if __name__ == "__main__":
    unittest.main()