import unittest
from unittest.mock import patch

import numpy as np

import align_utils


class AlignUtilsUnitTests(unittest.TestCase):
    def test_resolve_component_key_variants(self):
        self.assertEqual(align_utils.resolve_component_key("DPZ", "R"), "DPZ")
        self.assertEqual(align_utils.resolve_component_key("DPN", "R"), "R")
        self.assertEqual(align_utils.resolve_component_key("DPE", "T"), "T")
        self.assertEqual(align_utils.resolve_component_key("DPX", "Z"), "DPX")

    def test_add_stage_timing_accumulates(self):
        state = align_utils.TimingState()
        with patch.object(align_utils.time, "perf_counter", side_effect=[10.5, 11.2]), patch.object(
            align_utils.time,
            "process_time",
            side_effect=[5.1, 5.6],
        ):
            align_utils.add_stage_timing(state, "stageA", wall_start=10.0, cpu_start=5.0)
            align_utils.add_stage_timing(state, "stageA", wall_start=10.0, cpu_start=5.0)

        self.assertAlmostEqual(state.stage_wall_times["stageA"], 1.7, places=6)
        self.assertAlmostEqual(state.stage_cpu_times["stageA"], 0.7, places=6)
        self.assertEqual(state.stage_counts["stageA"], 2)

    def test_build_alignment_products_payload_contains_expected_keys(self):
        payload = align_utils.build_alignment_products_payload(
            npts=100,
            sample_rate=20.0,
            move_limit_samples=3,
            win_start=10,
            win_end=20,
            calc_shifts={"1": 0.1},
            aligned_stack=np.array([1.0, 2.0]),
            selected_aligned_stack=np.array([1.0]),
            selected_ids={"1"},
            station_corr={"1": 0.9},
            n_pass_window=1,
            pass_window_ids={"1"},
            snippet_by_station={"1": np.array([1.0])},
            ref_window=np.array([1.0, 1.0]),
            selected_rows=[("1", 0.9, np.array([1.0]))],
            rejected_rows=[],
            station_shifts={"1": 0.0},
            aligned_traces_by_station={"1": np.array([1.0])},
            t_abs=np.array([0.0, 1.0]),
            mask=np.array([True, True]),
            stack_vec=np.array([0.1, 0.2]),
        )

        for key in ("npts", "sample_rate", "calc_shifts", "selected_ids", "stack_vec"):
            self.assertIn(key, payload)
        self.assertEqual(payload["npts"], 100)
        self.assertEqual(payload["sample_rate"], 20.0)

    def test_build_component_output_payload_copies_arrays(self):
        selected_trace = np.array([1.0, 2.0])
        stack_vec = np.array([3.0, 4.0])
        t_abs = np.array([0.0, 1.0])
        mask = np.array([True, False])
        ref_window = np.array([0.5, 0.6])
        snippet = np.array([7.0, 8.0])
        aligned = np.array([9.0, 10.0])

        payload = align_utils.build_component_output_payload(
            record_fig=None,
            selected_rows=[("1", 0.8, selected_trace)],
            rejected_rows=[],
            stack_vec=stack_vec,
            t_abs=t_abs,
            mask=mask,
            sample_rate=20.0,
            win_start=1,
            win_end=2,
            move_limit_sec_value=0.1,
            move_limit_samples=2,
            npts=2,
            start_t=-1.0,
            end_t=1.0,
            eve_id="E1",
            align_phase_name="P",
            origin=None,
            station_shifts={"1": 0.0},
            station_corr={"1": 0.9},
            calc_shifts={"1": 0.0},
            n_pass_window=1,
            pass_window_ids={"1"},
            snippet_by_station={"1": snippet},
            ref_window=ref_window,
            p_traveltime=1.0,
            s_traveltime=2.0,
            name2ll={"1": (10.0, 20.0)},
            selected_ids={"1"},
            aligned_traces_by_station={"1": aligned},
            t_ref=0.0,
        )

        stack_vec[0] = -999.0
        selected_trace[0] = -999.0
        snippet[0] = -999.0
        aligned[0] = -999.0

        self.assertNotEqual(payload["stack_vec"][0], -999.0)
        self.assertNotEqual(payload["all_rows"][0][2][0], -999.0)
        self.assertNotEqual(payload["snippet_by_station"]["1"][0], -999.0)
        self.assertNotEqual(payload["aligned_traces_by_station"]["1"][0], -999.0)


if __name__ == "__main__":
    unittest.main()