"""Filter and plot overlaid radial/transverse snippets for one event."""

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib.pyplot as plt
import numpy as np
from obspy.taup import TauPyModel

from align_utils import (
    TimingState,
    load_event_metadata,
    load_station_lookup,
    preprocess_traces_bandpass,
    read_waveforms_for_event,
    rotate_horizontals_to_component,
)


PATH_PREFIX = Path("/Users/jvidale/Documents/Research/FaultScanR")
DEFAULT_EVENT = "CI_40353864"
DISPLAY_AMPLITUDE = 0.456  # 20% larger peak-to-peak display than the prior 0.38 scale.


def common_trace_peak(trace_pairs: list[tuple]) -> float:
    """Return one peak amplitude shared by all R and T traces in the event."""
    peaks = [
        float(np.max(np.abs(np.asarray(trace.data, dtype=float))))
        for radial, transverse, _s_arrival in trace_pairs
        for trace in (radial, transverse)
        if trace.stats.npts > 0
    ]
    return max(peaks, default=0.0)


def predicted_s_arrival_seconds(model: TauPyModel, event_depth: float, distance_deg: float) -> float | None:
    """Return the first direct S-arrival time for one station."""
    arrivals = model.get_travel_times(
        source_depth_in_km=event_depth,
        distance_in_degree=distance_deg,
        phase_list=["S", "s"],
    )
    for arrival in arrivals:
        if arrival.name.upper() == "S":
            return float(arrival.time)
    return None


def collect_rt_traces(
    event_id: str,
    snippets_root: Path,
    info_root: Path,
    start_time: float,
    end_time: float,
    sampling_hz: int,
):
    """Read horizontal snippets and return matched, distance-sorted R/T trace pairs."""
    event_depth, event_lat, event_lon, origin = load_event_metadata(event_id, info_root)
    stations = load_station_lookup(info_root)
    timing = TimingState()
    horizontal_stream, _raw_limits = read_waveforms_for_event(
        eve_id=event_id,
        channel="DP1",
        process_as_three_comp_mode=True,
        horizontal_window_cache={},
        horizontal_raw_limits_cache={},
        name2ll=stations,
        eve_lat=event_lat,
        eve_lon=event_lon,
        origin=origin,
        data_path=snippets_root / event_id,
        input_is_snippets=True,
        sampling_hz=sampling_hz,
        start_time=start_time,
        end_time=end_time,
        timing_state=timing,
    )
    if horizontal_stream is None:
        raise RuntimeError(f"No horizontal snippets found for {event_id}")

    radial, _ = rotate_horizontals_to_component(
        horizontal_stream.copy(), "R", stations, event_lat, event_lon, timing
    )
    transverse, _ = rotate_horizontals_to_component(
        horizontal_stream.copy(), "T", stations, event_lat, event_lon, timing
    )
    transverse_by_station = {str(trace.stats.station): trace for trace in transverse}
    model = TauPyModel(model="iasp91")
    pairs = [
        (
            trace,
            transverse_by_station[str(trace.stats.station)],
            predicted_s_arrival_seconds(model, event_depth, float(trace.stats.dist_deg)),
        )
        for trace in radial
        if str(trace.stats.station) in transverse_by_station
    ]
    pairs.sort(key=lambda pair: float(pair[0].stats.dist_km))
    if not pairs:
        raise RuntimeError(f"No matched R/T station pairs found for {event_id}")
    return origin, pairs


def plot_rt_frames(
    event_id: str,
    origin,
    trace_pairs: list[tuple],
    start_time: float,
    end_time: float,
    min_freq: float,
    max_freq: float,
    traces_per_frame: int,
    output_dir: Path,
) -> list[Path]:
    """Save offset R/T overlays, with one station pair per vertical row."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []
    amplitude_peak = common_trace_peak(trace_pairs)
    if amplitude_peak == 0.0:
        raise ValueError("All R/T traces have zero amplitude")
    for frame_index, first in enumerate(range(0, len(trace_pairs), traces_per_frame), start=1):
        subset = trace_pairs[first : first + traces_per_frame]
        fig, ax = plt.subplots(figsize=(12, max(6.0, 0.35 * len(subset) + 1.6)))
        offsets = np.arange(len(subset) - 1, -1, -1, dtype=float)

        for row, (radial, transverse, s_arrival) in enumerate(subset):
            r_time = radial.times(reftime=origin)
            t_time = transverse.times(reftime=origin)
            radial_data = np.asarray(radial.data, dtype=float) / amplitude_peak
            transverse_data = np.asarray(transverse.data, dtype=float) / amplitude_peak
            ax.plot(
                r_time,
                DISPLAY_AMPLITUDE * radial_data + offsets[row],
                color="tab:blue",
                lw=0.8,
            )
            ax.plot(
                t_time,
                DISPLAY_AMPLITUDE * transverse_data + offsets[row],
                color="tab:orange",
                lw=0.8,
            )
            if s_arrival is not None and start_time <= s_arrival <= end_time:
                ax.vlines(
                    s_arrival,
                    offsets[row] - 0.38,
                    offsets[row] + 0.38,
                    color="tab:green",
                    lw=1.0,
                    zorder=4,
                )

        ax.set_xlim(start_time, end_time)
        ax.set_ylim(-0.65, len(subset) - 0.35)
        ax.set_yticks(offsets)
        ax.set_yticklabels([str(pair[0].stats.station).zfill(5) for pair in subset], fontsize=8)
        ax.set_xlabel("Time since origin (s)")
        ax.set_ylabel("Station (near to far)")
        ax.set_title(
            f"{event_id}: filtered R/T snippets, {min_freq:g}-{max_freq:g} Hz "
            f"(stations {first + 1}-{first + len(subset)})",
            fontweight="bold",
        )
        ax.plot([], [], color="tab:blue", lw=1.4, label="R")
        ax.plot([], [], color="tab:orange", lw=1.4, label="T")
        ax.plot([], [], color="tab:green", lw=1.4, label="Predicted S")
        ax.legend(loc="upper right")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()

        output_file = output_dir / (
            f"{event_id}_RT_snippets_{start_time:g}-{end_time:g}s_"
            f"{min_freq:g}-{max_freq:g}Hz_frame{frame_index:02d}.png"
        )
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        output_files.append(output_file)
    return output_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot overlaid R/T event snippets in offset frames.")
    parser.add_argument("--event", default=DEFAULT_EVENT)
    parser.add_argument("--start-time", type=float, default=2.0)
    parser.add_argument("--end-time", type=float, default=12.0)
    parser.add_argument("--min-freq", type=float, default=1.0)
    parser.add_argument("--max-freq", type=float, default=4.0)
    parser.add_argument("--sampling-hz", type=int, default=250)
    parser.add_argument("--traces-per-frame", type=int, default=20)
    parser.add_argument("--snippets-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.end_time <= args.start_time:
        raise ValueError("end-time must be greater than start-time")
    if args.min_freq <= 0.0 or args.max_freq <= args.min_freq:
        raise ValueError("frequency range must satisfy 0 < min-freq < max-freq")

    snippets_root = args.snippets_root or (PATH_PREFIX / f"Sgrams/Snippets_{args.sampling_hz}Hz")
    output_dir = args.output_dir or (
        PATH_PREFIX
        / "output"
        / f"{args.event}_RT_snippets_{args.start_time:g}-{args.end_time:g}s_{args.min_freq:g}-{args.max_freq:g}Hz"
    )
    origin, trace_pairs = collect_rt_traces(
        args.event,
        snippets_root,
        PATH_PREFIX / "event_sta_info",
        args.start_time,
        args.end_time,
        args.sampling_hz,
    )
    radial = [pair[0] for pair in trace_pairs]
    transverse = [pair[1] for pair in trace_pairs]
    timing = TimingState()
    preprocess_traces_bandpass(radial, args.min_freq, args.max_freq, timing)
    preprocess_traces_bandpass(transverse, args.min_freq, args.max_freq, timing)
    output_files = plot_rt_frames(
        args.event,
        origin,
        [
            (radial[index], transverse[index], trace_pairs[index][2])
            for index in range(len(trace_pairs))
        ],
        args.start_time,
        args.end_time,
        args.min_freq,
        args.max_freq,
        args.traces_per_frame,
        output_dir,
    )
    print(f"Wrote {len(output_files)} R/T overlay frames to {output_dir}")


if __name__ == "__main__":
    main()
