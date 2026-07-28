import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import plot_event_rt_snippets as rt_plot


class FakeTrace:
    def __init__(self, station: str, values, origin_times):
        self.data = np.asarray(values, dtype=float)
        self.stats = SimpleNamespace(
            station=station,
            npts=len(self.data),
        )
        self._origin_times = np.asarray(origin_times, dtype=float)

    def times(self, reftime):
        return self._origin_times.copy()


class PlotEventRtSnippetsTests(unittest.TestCase):
    def test_default_bandpass_is_one_to_four_hz(self):
        self.assertEqual(rt_plot.MIN_FREQ, 1.0)
        self.assertEqual(rt_plot.MAX_FREQ, 4.0)

    def test_no_time_shift_option_disables_station_statics(self):
        parser = rt_plot.build_argument_parser()

        self.assertEqual(
            rt_plot.resolve_station_static_components(parser.parse_args([])),
            frozenset({"R", "T"}),
        )
        self.assertEqual(
            rt_plot.resolve_station_static_components(
                parser.parse_args(["--no-time-shift"])
            ),
            frozenset(),
        )

    def test_station_static_options_are_independent_by_component(self):
        parser = rt_plot.build_argument_parser()

        self.assertEqual(
            rt_plot.resolve_station_static_components(
                parser.parse_args(["--no-statics-t"])
            ),
            frozenset({"R"}),
        )
        self.assertEqual(
            rt_plot.resolve_station_static_components(
                parser.parse_args(["--no-time-shift", "--shift-r"])
            ),
            frozenset({"R"}),
        )

    def test_no_z_option_disables_z_component(self):
        parser = rt_plot.build_argument_parser()

        self.assertEqual(
            parser.parse_args([]).include_z,
            rt_plot.USE_Z_COMPONENT,
        )
        self.assertFalse(parser.parse_args(["--no-z"]).include_z)

    def test_loads_correlation_window_from_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_file = Path(temporary) / "rp_input.json"
            config_file.write_text(
                json.dumps(
                    {
                        "win_pre": 0.05,
                        "win_post": 0.15,
                        "move_limit_sec": 0.05,
                    }
                ),
                encoding="utf-8",
            )

            values = rt_plot.load_correlation_window_config(config_file)

        self.assertEqual(values, (0.05, 0.15, 0.05))
        self.assertEqual(
            rt_plot.correlation_window_bounds(*values),
            (-0.05, 0.15, -0.10, 0.20),
        )

    def test_loads_station_statics_by_padded_station_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            station_file = Path(temporary) / "stations.xlsx"
            pd.DataFrame(
                {
                    "station": [1, 36],
                    "station static s": [0.024, -0.031],
                }
            ).to_excel(station_file, index=False)

            statics = rt_plot.load_station_statics(
                station_file,
                "station static s",
            )

        self.assertEqual(statics, {"00001": 0.024, "00036": -0.031})

    def test_trace_times_shift_predicted_s_to_zero(self):
        trace = FakeTrace("1", [1.0, 2.0, 1.0], [4.0, 5.0, 6.0])

        shifted = rt_plot.trace_times_relative_to_predicted_s(
            trace,
            origin=object(),
            s_arrival=5.0,
        )

        np.testing.assert_allclose(shifted, [-1.0, 0.0, 1.0])
        statically_shifted = rt_plot.trace_times_relative_to_predicted_s(
            trace,
            origin=object(),
            s_arrival=5.0,
            station_static=0.12,
        )
        np.testing.assert_allclose(
            statically_shifted,
            [-1.12, -0.12, 0.88],
        )

    def test_plot_uses_s_relative_window_and_marks_common_pick(self):
        radial = FakeTrace("1", [1.0, 2.0, 1.0], [4.0, 5.0, 6.0])
        transverse = FakeTrace("1", [-1.0, -2.0, -1.0], [4.0, 5.0, 6.0])
        vertical = FakeTrace("1", [0.5, 1.0, 0.5], [4.0, 5.0, 6.0])

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            rt_plot.plt,
            "close",
        ):
            output_files = rt_plot.plot_rt_frames(
                event_id="E1",
                origin=object(),
                trace_pairs=[(radial, transverse, vertical, 5.0)],
                start_time=-3.0,
                end_time=7.0,
                min_freq=1.0,
                max_freq=4.0,
                traces_per_frame=20,
                output_dir=Path(temporary),
                win_pre=0.05,
                win_post=0.15,
                move_limit_sec=0.05,
                station_statics={"00001": 0.12},
                static_components={"R"},
            )
            axis = rt_plot.plt.gcf().axes[0]

            trace_lines = [
                line
                for line in axis.lines
                if len(np.atleast_1d(line.get_xdata())) == 3
            ]
            np.testing.assert_allclose(
                trace_lines[0].get_xdata(),
                [-1.12, -0.12, 0.88],
            )
            np.testing.assert_allclose(trace_lines[1].get_xdata(), [-1.0, 0.0, 1.0])
            np.testing.assert_allclose(trace_lines[2].get_xdata(), [-1.0, 0.0, 1.0])
            self.assertEqual(trace_lines[2].get_color(), "0.45")
            marker_positions = [
                float(np.atleast_1d(line.get_xdata())[0])
                for line in axis.lines
                if len(np.atleast_1d(line.get_xdata())) == 2
                and np.allclose(
                    np.atleast_1d(line.get_xdata()),
                    np.atleast_1d(line.get_xdata())[0],
                )
            ]
            for expected in (-0.10, -0.05, 0.15, 0.20):
                self.assertTrue(
                    any(np.isclose(position, expected) for position in marker_positions)
                )
            self.assertEqual(axis.get_xlim(), (-3.0, 7.0))
            self.assertEqual(
                axis.get_xlabel(),
                "Time relative to TauP-predicted S (s)",
            )
            pick_segments = axis.collections[0].get_segments()
            self.assertTrue(
                all(
                    np.allclose(segment[:, 0], rt_plot.COMMON_S_PICK_TIME)
                    for segment in pick_segments
                )
            )
            shift_segments = axis.collections[1].get_segments()
            self.assertTrue(
                all(
                    np.allclose(segment[:, 0], 0.12)
                    for segment in shift_segments
                )
            )
            self.assertTrue(output_files[0].exists())

    def test_plot_omits_static_marker_and_label_when_disabled(self):
        radial = FakeTrace("1", [1.0, 2.0, 1.0], [4.0, 5.0, 6.0])
        transverse = FakeTrace("1", [-1.0, -2.0, -1.0], [4.0, 5.0, 6.0])
        vertical = FakeTrace("1", [0.5, 1.0, 0.5], [4.0, 5.0, 6.0])

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            rt_plot.plt,
            "close",
        ):
            rt_plot.plot_rt_frames(
                event_id="E1",
                origin=object(),
                trace_pairs=[(radial, transverse, vertical, 5.0)],
                start_time=-3.0,
                end_time=7.0,
                min_freq=3.0,
                max_freq=10.0,
                traces_per_frame=20,
                output_dir=Path(temporary),
                win_pre=0.05,
                win_post=0.15,
                move_limit_sec=0.05,
                station_statics=None,
            )
            axis = rt_plot.plt.gcf().axes[0]

            self.assertEqual(len(axis.collections), 1)
            legend_labels = [
                text.get_text() for text in axis.get_legend().get_texts()
            ]
            self.assertNotIn(
                "Station static from stations.xlsx",
                legend_labels,
            )

    def test_plot_omits_z_trace_label_and_filename_when_disabled(self):
        radial = FakeTrace("1", [1.0, 2.0, 1.0], [4.0, 5.0, 6.0])
        transverse = FakeTrace("1", [-1.0, -2.0, -1.0], [4.0, 5.0, 6.0])

        with tempfile.TemporaryDirectory() as temporary, patch.object(
            rt_plot.plt,
            "close",
        ):
            output_files = rt_plot.plot_rt_frames(
                event_id="E1",
                origin=object(),
                trace_pairs=[(radial, transverse, None, 5.0)],
                start_time=-3.0,
                end_time=7.0,
                min_freq=3.0,
                max_freq=10.0,
                traces_per_frame=20,
                output_dir=Path(temporary),
                win_pre=0.05,
                win_post=0.15,
                move_limit_sec=0.05,
                station_statics=None,
            )
            axis = rt_plot.plt.gcf().axes[0]

            trace_lines = [
                line
                for line in axis.lines
                if len(np.atleast_1d(line.get_xdata())) == 3
            ]
            self.assertEqual(len(trace_lines), 2)
            legend_labels = [
                text.get_text() for text in axis.get_legend().get_texts()
            ]
            self.assertNotIn("Z", legend_labels)
            self.assertIn("R/T snippets", axis.get_title())
            self.assertIn("_RT_snippets_", output_files[0].name)

    def test_main_preserves_none_z_entries_in_rt_only_mode(self):
        radial = FakeTrace("1", [1.0, 2.0, 1.0], [4.0, 5.0, 6.0])
        transverse = FakeTrace("1", [-1.0, -2.0, -1.0], [4.0, 5.0, 6.0])
        unexpected_vertical = FakeTrace(
            "1",
            [0.5, 1.0, 0.5],
            [4.0, 5.0, 6.0],
        )
        trace_pairs = [(radial, transverse, unexpected_vertical, 5.0)]

        with tempfile.TemporaryDirectory() as temporary, patch(
            "sys.argv",
            [
                "plot_event_rt_snippets.py",
                "--no-z",
                "--no-time-shift",
                "--output-dir",
                temporary,
            ],
        ), patch.object(
            rt_plot,
            "load_correlation_window_config",
            return_value=(0.05, 0.15, 0.05),
        ), patch.object(
            rt_plot,
            "collect_rt_traces",
            return_value=(object(), trace_pairs),
        ), patch.object(
            rt_plot,
            "preprocess_traces_bandpass",
        ) as preprocess, patch.object(
            rt_plot,
            "plot_rt_frames",
            return_value=[],
        ) as plot_frames:
            rt_plot.main()

        self.assertEqual(preprocess.call_count, 2)
        plotted_pairs = plot_frames.call_args.args[2]
        self.assertEqual(len(plotted_pairs), 1)
        self.assertIsNone(plotted_pairs[0][2])
        self.assertEqual(plot_frames.call_args.args[13], set())


if __name__ == "__main__":
    unittest.main()
