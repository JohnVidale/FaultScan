import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock

import pandas as pd
import numpy as np


class AlignStackSmokeTests(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("align_stack")

    def test_event_output_directory_uses_shared_stack_output_root(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            self.mod,
            "path_prefix",
            f"{tmp}/",
        ), patch.object(
            self.mod,
            "RUN_OUTPUT_DIR",
            None,
        ):
            event_dir = self.mod.get_run_event_output_dir("E1")

        self.assertEqual(event_dir, Path(tmp) / "stack_output" / "E1")

    def test_aligned_component_offset_plot_uses_only_skip_zero_events(self):
        time = np.array([0.0, 0.1, 0.2])
        mask = np.array([True, True, True])
        stack = np.array([0.1, 1.0, 0.1])
        metadata = {
            "start_time": 0.0,
            "win_start": 0,
            "win_end": 2,
            "sample_rate": 10.0,
            "move_limit_sec": 0.05,
            "npts": 3,
            "t_ref": 0.1,
        }
        series = [
            (event_id, time, mask, stack, metadata)
            for event_id in ("E0", "E1", "E2")
        ]
        catalog = pd.DataFrame(
            {"evid": ["E0", "E1", "E2"], "skip": [0, 1, 2]}
        )
        captured_event_ids = []

        def fake_alignment(input_series, *_args, **_kwargs):
            captured_event_ids.extend(item[0] for item in input_series)
            return pd.DataFrame(
                {
                    "event_id": [item[0] for item in input_series],
                    "shift_left_to_align_waveform_seconds": [0.0]
                    * len(input_series),
                }
            )

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            self.mod, "catalog_local", catalog
        ), patch.object(
            self.mod, "compute_event_stack_alignment_shifts", side_effect=fake_alignment
        ), patch.object(
            self.mod, "event_time_shift_for_plot", return_value=0.0
        ), patch.object(
            self.mod, "catalog_time_shift_column", return_value="time shift"
        ):
            outputs = self.mod.plot_all_events_component_offsets_aligned(
                component_stacks={"R": series},
                run_output_dir=Path(temporary),
                align_phase_name="S",
                reference_event="E0",
                max_shift_sec=0.2,
            )

        self.assertEqual(captured_event_ids, ["E0"])
        self.assertEqual(len(outputs), 1)

    def test_aligned_component_offset_plot_uses_all_events_without_catalog(self):
        time = np.array([0.0, 0.1, 0.2])
        mask = np.array([True, True, True])
        stack = np.array([0.1, 1.0, 0.1])
        metadata = {
            "start_time": 0.0,
            "win_start": 0,
            "win_end": 2,
            "sample_rate": 10.0,
            "move_limit_sec": 0.05,
            "npts": 3,
            "t_ref": 0.1,
        }
        series = [
            (event_id, time, mask, stack, metadata)
            for event_id in ("E0", "E1")
        ]
        captured_event_ids = []

        def fake_alignment(input_series, *_args, **_kwargs):
            captured_event_ids.extend(item[0] for item in input_series)
            return pd.DataFrame(
                {
                    "event_id": [item[0] for item in input_series],
                    "shift_left_to_align_waveform_seconds": [0.0]
                    * len(input_series),
                }
            )

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            self.mod, "catalog_local", None
        ), patch.object(
            self.mod, "compute_event_stack_alignment_shifts", side_effect=fake_alignment
        ), patch.object(
            self.mod, "event_time_shift_for_plot", return_value=0.0
        ), patch.object(
            self.mod, "catalog_time_shift_column", return_value="time shift"
        ):
            outputs = self.mod.plot_all_events_component_offsets_aligned(
                component_stacks={"R": series},
                run_output_dir=Path(temporary),
                align_phase_name="S",
                reference_event="E0",
                max_shift_sec=0.2,
            )

        self.assertEqual(captured_event_ids, ["E0", "E1"])
        self.assertEqual(len(outputs), 1)

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
            "get_run_event_output_dir",
            return_value="/tmp/out",
        ), patch.object(
            self.mod,
            "apply_event_location_override",
            side_effect=lambda event_depth, eve_lat, eve_lon: (event_depth, eve_lat, eve_lon),
        ), patch.object(
            self.mod,
            "apply_event_origin_time_shift",
            side_effect=lambda eve_id, origin: origin,
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
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                self.mod,
                "load_event_metadata",
                return_value=(10.0, 35.0, -117.0, None),
            ), patch.object(
                self.mod,
                "get_run_event_output_dir",
                return_value="/tmp/out",
            ), patch.object(
                self.mod,
                "apply_event_location_override",
                side_effect=lambda event_depth, eve_lat, eve_lon: (event_depth, eve_lat, eve_lon),
            ), patch.object(
                self.mod,
                "apply_event_origin_time_shift",
                side_effect=lambda eve_id, origin: origin,
            ), patch.object(
                self.mod,
                "load_station_lookup",
                return_value={"STA": (35.0, -117.0)},
            ), patch.object(
                self.mod,
                "get_event_data_path",
                return_value=Path(tmp),
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
            "n_rejected_correlation": 12,
            "n_rejected_trace_peak_to_pre_p": 13,
            "n_rejected_any": 14,
            "pass_window_ids": 15,
            "snippet_by_station": 16,
            "ref_window": 17,
            "selected_rows": 18,
            "rejected_rows": 19,
            "station_shifts": 20,
            "aligned_traces_by_station": 21,
            "t_abs": 22,
            "mask": 23,
            "stack_vec": 24,
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
            "n_rejected_correlation",
            "n_rejected_trace_peak_to_pre_p",
            "n_rejected_any",
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

    def test_write_screening_failure_counts_writes_one_row_per_event_component(self):
        rows = [
            {
                "event_id": "E1",
                "component": "Z",
                "total_traces": 10,
                "accepted_traces": 6,
                "failed_any_threshold": 4,
                "failed_correlation_threshold": 3,
                "failed_noise_ratio_threshold": 2,
                "correlation_threshold_min": 0.6,
                "noise_ratio_threshold_min": 10.0,
            },
            {
                "event_id": "E1",
                "component": "R",
                "total_traces": 9,
                "accepted_traces": 7,
                "failed_any_threshold": 2,
                "failed_correlation_threshold": 1,
                "failed_noise_ratio_threshold": 1,
                "correlation_threshold_min": 0.6,
                "noise_ratio_threshold_min": 10.0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out_file = self.mod.write_screening_failure_counts(rows, Path(tmp))
            written = pd.read_excel(
                out_file,
                sheet_name="Screening counts",
                dtype={"event_id": str},
            )

        self.assertEqual(out_file.name, "screening_failure_counts.xlsx")
        self.assertEqual(len(written), 2)
        self.assertEqual(written.iloc[0]["event_id"], "E1")
        self.assertEqual(written.iloc[0]["failed_correlation_threshold"], 3)
        self.assertEqual(written.iloc[0]["failed_noise_ratio_threshold"], 2)

    def test_event_has_zero_catalog_skip_accepts_only_numeric_zero(self):
        catalog = pd.DataFrame(
            {
                "evid": ["ZERO_INT", "ZERO_FLOAT", "ZERO_TEXT", "ONE", "TWO", "BAD"],
                "skip": [0, 0.0, "0", 1, 2, "not-a-number"],
            }
        )
        with patch.object(self.mod, "catalog_local", catalog):
            for event_id in ("ZERO_INT", "ZERO_FLOAT", "ZERO_TEXT"):
                with self.subTest(event_id=event_id):
                    self.assertTrue(self.mod.event_has_zero_catalog_skip(event_id))
            for event_id in ("ONE", "TWO", "BAD", "MISSING"):
                with self.subTest(event_id=event_id):
                    self.assertFalse(self.mod.event_has_zero_catalog_skip(event_id))

    def test_apply_event_location_override_uses_configured_values(self):
        with patch.object(self.mod, "use_json_event_location", True), patch.object(
            self.mod,
            "event_lat_override",
            33.48,
        ), patch.object(
            self.mod,
            "event_lon_override",
            -116.513,
        ), patch.object(self.mod, "event_depth_override", 9.3):
            out = self.mod.apply_event_location_override(
                event_depth=1.0,
                eve_lat=2.0,
                eve_lon=3.0,
            )

        self.assertEqual(out, (9.3, 33.48, -116.513))

    def test_apply_event_location_override_uses_catalog_when_disabled(self):
        with patch.object(self.mod, "use_json_event_location", False):
            out = self.mod.apply_event_location_override(
                event_depth=1.0,
                eve_lat=2.0,
                eve_lon=3.0,
            )

        self.assertEqual(out, (1.0, 2.0, 3.0))

    def test_apply_event_origin_time_shift_uses_catalog_shift(self):
        catalog = pd.DataFrame({"evid": ["CI_TEST"], "time shift": [0.024]})
        origin = self.mod.UTCDateTime("2022-09-30T11:56:20.82Z")

        with patch.object(self.mod, "use_event_static_correction", True), patch.object(
            self.mod, "catalog_local", catalog
        ):
            adjusted = self.mod.apply_event_origin_time_shift("CI_TEST", origin)

        self.assertAlmostEqual(adjusted - origin, 0.024)

    def test_apply_event_origin_time_shift_uses_shared_column_for_all_components(self):
        catalog = pd.DataFrame({"evid": ["CI_TEST"], "time shift": [-0.031]})
        origin = self.mod.UTCDateTime("2022-09-30T11:56:20.82Z")

        with patch.object(self.mod, "use_event_static_correction", True), patch.object(
            self.mod, "component", "T"
        ), patch.object(
            self.mod, "catalog_local", catalog
        ):
            adjusted = self.mod.apply_event_origin_time_shift("CI_TEST", origin)

        self.assertAlmostEqual(adjusted - origin, -0.031)

    def test_apply_event_origin_time_shift_is_independent_of_location_override(self):
        origin = self.mod.UTCDateTime("2022-09-30T11:56:20.82Z")
        catalog = pd.DataFrame({"evid": ["CI_TEST"], "time shift": [0.015]})

        with patch.object(self.mod, "use_event_static_correction", True), patch.object(
            self.mod, "use_json_event_location", False
        ), patch.object(
            self.mod, "catalog_local", catalog
        ):
            adjusted = self.mod.apply_event_origin_time_shift("CI_TEST", origin)

        self.assertAlmostEqual(adjusted - origin, 0.015)

    def test_apply_event_origin_time_shift_uses_catalog_origin_when_static_disabled(self):
        catalog = pd.DataFrame({"evid": ["CI_TEST"], "time shift": [0.024]})
        origin = self.mod.UTCDateTime("2022-09-30T11:56:20.82Z")

        with patch.object(self.mod, "use_event_static_correction", False), patch.object(
            self.mod, "catalog_local", catalog
        ):
            adjusted = self.mod.apply_event_origin_time_shift("CI_TEST", origin)

        self.assertEqual(adjusted, origin)

    def test_imposed_station_shifts_are_relative_to_reference_station(self):
        with tempfile.TemporaryDirectory() as tmp:
            station_file = Path(tmp) / "stations.xlsx"
            pd.DataFrame(
                {
                    "station": ["00001", "00002"],
                    "sta_statics_R": [0.012, 0.035],
                }
            ).to_excel(station_file, index=False)
            reference = self.mod.Trace(data=np.ones(5))
            reference.stats.station = "00001"
            other = self.mod.Trace(data=np.ones(5))
            other.stats.station = "00002"
            stream = self.mod.Stream([reference, other])

            with patch.object(self.mod, "station_static_mode", "tabulated"), patch.object(
                self.mod, "align_phase", "S"
            ), patch.object(
                self.mod, "station_static_file", station_file
            ), patch.object(self.mod, "station_static_column", "sta_statics_R"), patch.object(
                self.mod, "_station_static_cache", None
            ):
                shifts = self.mod.imposed_station_shifts_for_stream(stream, "00001")

        self.assertEqual(shifts["00001"], 0.0)
        self.assertAlmostEqual(shifts["00002"], 0.023)

    def test_imposed_station_shifts_are_disabled_for_p_alignment(self):
        reference = self.mod.Trace(data=np.ones(5))
        reference.stats.station = "00001"
        stream = self.mod.Stream([reference])

        with patch.object(self.mod, "station_static_mode", "tabulated"), patch.object(
            self.mod, "align_phase", "P"
        ):
            shifts = self.mod.imposed_station_shifts_for_stream(stream, "00001")

        self.assertIsNone(shifts)

    def test_cross_correlation_station_residuals_allow_p_alignment(self):
        with patch.object(self.mod, "station_static_mode", "cross_correlation"), patch.object(
            self.mod, "align_phase", "P"
        ):
            self.assertTrue(self.mod.measure_station_residuals_enabled())

        with patch.object(self.mod, "station_static_mode", "none"), patch.object(
            self.mod, "align_phase", "P"
        ):
            self.assertFalse(self.mod.measure_station_residuals_enabled())

    def test_tabulated_mode_uses_configured_transverse_static_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            station_file = Path(tmp) / "stations.xlsx"
            pd.DataFrame(
                {
                    "station": ["00001", "00002"],
                    "sta_statics_R": [0.012, 0.035],
                    "sta_statics_T": [0.021, 0.052],
                }
            ).to_excel(station_file, index=False)
            reference = self.mod.Trace(data=np.ones(5))
            reference.stats.station = "00001"
            other = self.mod.Trace(data=np.ones(5))
            other.stats.station = "00002"
            stream = self.mod.Stream([reference, other])

            with patch.object(self.mod, "station_static_mode", "tabulated"), patch.object(
                self.mod, "align_phase", "S"
            ), patch.object(self.mod, "station_static_file", station_file), patch.object(
                self.mod, "station_static_column", "sta_statics_T"
            ), patch.object(self.mod, "_station_static_cache", None):
                shifts = self.mod.imposed_station_shifts_for_stream(stream, "00001")

        self.assertEqual(shifts["00001"], 0.0)
        self.assertAlmostEqual(shifts["00002"], 0.031)

    def test_resolve_station_static_mode_accepts_all_explicit_modes(self):
        for mode in ("none", "tabulated", "cross_correlation"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    self.mod.resolve_station_static_mode(
                        {"station_static_mode": mode},
                        "cross_correlation",
                    ),
                    mode,
                )

    def test_resolve_station_static_mode_maps_removed_tabulate_t_alias(self):
        self.assertEqual(
            self.mod.resolve_station_static_mode(
                {"station_static_mode": "tabulate_T"},
                "cross_correlation",
            ),
            "tabulated",
        )

    def test_resolve_station_static_mode_rejects_invalid_mode(self):
        with self.assertRaisesRegex(ValueError, "station_static_mode"):
            self.mod.resolve_station_static_mode(
                {"station_static_mode": "automatic"},
                "cross_correlation",
            )

    def test_resolve_station_static_mode_maps_legacy_boolean(self):
        self.assertEqual(
            self.mod.resolve_station_static_mode(
                {"use_station_static_correction": True},
                "none",
            ),
            "tabulated",
        )
        self.assertEqual(
            self.mod.resolve_station_static_mode(
                {"use_station_static_correction": False},
                "none",
            ),
            "cross_correlation",
        )

    def test_explicit_station_static_mode_takes_precedence_over_legacy_boolean(self):
        self.assertEqual(
            self.mod.resolve_station_static_mode(
                {
                    "station_static_mode": "none",
                    "use_station_static_correction": True,
                },
                "cross_correlation",
            ),
            "none",
        )

    def test_compute_event_stack_alignment_shifts_finds_delayed_stack(self):
        meta = {
            "sample_rate": 10.0,
            "win_start": 4,
            "win_end": 7,
            "t_ref": 5.0,
        }
        reference = np.array([0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0], dtype=float)
        delayed = np.array([0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0], dtype=float)
        series = [
            ("REF", np.arange(12) / 10.0, np.ones(12, dtype=bool), reference, meta),
            ("LATE", np.arange(12) / 10.0, np.ones(12, dtype=bool), delayed, meta),
        ]

        shifts = self.mod.compute_event_stack_alignment_shifts(
            series=series,
            reference_event="REF",
            max_shift_sec=0.2,
        ).set_index("event_id")

        self.assertEqual(shifts.loc["REF", "shift_left_to_align_waveform_seconds"], 0.0)
        self.assertAlmostEqual(shifts.loc["LATE", "shift_left_to_align_waveform_seconds"], 0.2)
        self.assertAlmostEqual(shifts.loc["LATE", "waveform_correlation"], 1.0)

    def test_compute_event_stack_alignment_shifts_skips_xcorr_when_disabled(self):
        meta = {
            "sample_rate": 10.0,
            "win_start": 4,
            "win_end": 8,
            "t_ref": 0.6,
        }
        reference = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        delayed = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0])
        series = [
            ("REF", np.arange(12) / 10.0, np.ones(12, dtype=bool), reference, meta),
            ("LATE", np.arange(12) / 10.0, np.ones(12, dtype=bool), delayed, meta),
        ]

        shifts = self.mod.compute_event_stack_alignment_shifts(
            series=series,
            reference_event="REF",
            max_shift_sec=0.2,
            measure_xcorr_residual=False,
        ).set_index("event_id")

        self.assertEqual(shifts.loc["LATE", "shift_left_to_align_waveform_seconds"], 0.0)
        self.assertEqual(shifts.loc["LATE", "xcorr_residual_lag_seconds"], 0.0)
        self.assertFalse(shifts.loc["LATE", "xcorr_residual_measured"])
        self.assertTrue(np.isnan(shifts.loc["LATE", "waveform_correlation"]))

    def test_write_radial_s_wave_time_shifts_saves_residual_excel(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(self.mod, "path_prefix", f"{tmp}/"), patch.object(
                self.mod, "component", "R"
            ):
                out = self.mod.write_component_phase_time_shifts(
                    save_dir=Path(tmp) / "event_plots",
                    eve_id="EV1",
                    plot_comp="R",
                    align_phase_name="S",
                    station_shifts={
                        "2": {"lag_samples": 5, "lag_seconds": 0.05},
                        "1": {"lag_samples": 2, "lag_seconds": 0.02},
                    },
                    calc_shifts={"1": 0.01, "2": -0.02},
                    station_corr={"1": 0.9, "2": 0.7},
                    pass_window_ids={"1"},
                    sample_rate=100.0,
                    min_freq_hz=3.0,
                    max_freq_hz=10.0,
                )

            self.assertIsNotNone(out)
            self.assertEqual(out.parent.name, "Statics")
            self.assertEqual(out.parent.parent.name, "stack_output")
            self.assertEqual(out.name, "EV1_R_S_3-10Hz_shiftR_xcorr_statics.xlsx")
            self.assertEqual(out.suffix, ".xlsx")
            df = pd.read_excel(out)

        self.assertEqual(list(df["station"].astype(str)), ["1", "2"])
        self.assertEqual(set(df["catalog_shift_component"]), {"R"})
        self.assertAlmostEqual(df.loc[0, "shift_relative_to_predicted_seconds"], 0.01)
        self.assertAlmostEqual(df.loc[1, "shift_relative_to_predicted_seconds"], 0.07)
        self.assertTrue(bool(df.loc[0, "passed_window_correlation"]))
        self.assertFalse(bool(df.loc[1, "passed_window_correlation"]))

    def test_write_component_phase_time_shifts_records_actual_shared_catalog_column(self):
        for plot_component in ("T", "Z"):
            with self.subTest(component=plot_component), tempfile.TemporaryDirectory() as tmp:
                with patch.object(self.mod, "path_prefix", f"{tmp}/"), patch.object(
                    self.mod, "component", "T"
                ):
                    out = self.mod.write_component_phase_time_shifts(
                        save_dir=Path(tmp) / "event_plots",
                        eve_id="EV1",
                        plot_comp=plot_component,
                        align_phase_name="S",
                        station_shifts={"1": {"lag_samples": 2, "lag_seconds": 0.02}},
                        calc_shifts={"1": 0.01},
                        station_corr={"1": 0.9},
                        pass_window_ids={"1"},
                        sample_rate=100.0,
                        min_freq_hz=3.0,
                        max_freq_hz=10.0,
                    )

                self.assertIsNotNone(out)
                self.assertEqual(out.name, f"EV1_{plot_component}_S_3-10Hz_shiftR_xcorr_statics.xlsx")
                df = pd.read_excel(out)
                self.assertEqual(df.loc[0, "component"], plot_component)
                self.assertEqual(df.loc[0, "catalog_shift_component"], "R")
                self.assertEqual(df.loc[0, "catalog_time_shift_column"], "time shift")
                self.assertEqual(df.loc[0, "phase"], "S")


if __name__ == "__main__":
    unittest.main()
