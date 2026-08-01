"""Filter and plot overlaid radial/transverse snippets for one event."""

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy.geodetics import locations2degrees
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
RP_INPUT_FILE = Path(__file__).resolve().with_name("rp_input.json")
DEFAULT_EVENT = "CI_40353864"
START_TIME = -1.0  # Seconds relative to each station's TauP-predicted phase arrival.
END_TIME = 2.0
APPLY_STATION_STATICS_R = False
APPLY_STATION_STATICS_T = False
APPLY_STATION_STATICS_Z = False
USE_Z_COMPONENT = False
COMMON_PHASE_PICK_TIME = 0.0
DEFAULT_DISPLAY_AMPLITUDE = 0.5
NORMALIZE_TO_LARGEST_COMPONENT = True


def correlation_window_peak(
    trace,
    phase_relative_time: np.ndarray,
    win_pre: float,
    win_post: float,
) -> float:
    """Return a trace's finite peak absolute amplitude in the correlation window."""
    data = np.asarray(trace.data, dtype=float)
    in_window = (
        (phase_relative_time >= COMMON_PHASE_PICK_TIME - win_pre)
        & (phase_relative_time <= COMMON_PHASE_PICK_TIME + win_post)
        & np.isfinite(data)
    )
    if not np.any(in_window):
        return 0.0
    return float(np.max(np.abs(data[in_window])))


def normalize_to_correlation_window_peak(
    trace,
    phase_relative_time: np.ndarray,
    win_pre: float,
    win_post: float,
    normalization_peak: float | None = None,
) -> np.ndarray:
    """Normalize a trace by its own or an explicitly shared window peak."""
    data = np.asarray(trace.data, dtype=float)
    peak = (
        correlation_window_peak(trace, phase_relative_time, win_pre, win_post)
        if normalization_peak is None
        else normalization_peak
    )
    return data / peak if peak > 0.0 else np.zeros_like(data)


def predicted_phase_arrival_seconds(
    model: TauPyModel,
    event_depth: float,
    distance_deg: float,
    phase: str,
) -> float | None:
    """Return the first direct configured-phase arrival time for one station."""
    phase = phase.upper()
    if phase not in {"P", "S"}:
        raise ValueError(f"Unsupported alignment phase {phase!r}; choose P or S")
    arrivals = model.get_travel_times(
        source_depth_in_km=event_depth,
        distance_in_degree=distance_deg,
        phase_list=[phase, phase.lower()],
    )
    for arrival in arrivals:
        if arrival.name.upper() == phase:
            return float(arrival.time)
    return None


def trace_times_relative_to_predicted_phase(
    trace,
    origin,
    phase_arrival: float,
    station_static: float = 0.0,
) -> np.ndarray:
    """Return phase-relative times after applying the station-static correction."""
    return (
        np.asarray(trace.times(reftime=origin), dtype=float)
        - float(phase_arrival)
        + COMMON_PHASE_PICK_TIME
        - float(station_static)
    )


def load_correlation_window_config(config_file: Path) -> tuple[float, float, float]:
    """Load correlation-window widths and shift limit from rp_input.json."""
    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = ("win_pre", "win_post", "move_limit_sec")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(
            f"{config_file} is missing correlation-window fields: {missing}"
        )
    win_pre, win_post, move_limit_sec = (
        float(config["win_pre"]),
        float(config["win_post"]),
        float(config["move_limit_sec"]),
    )
    if win_pre < 0.0 or win_post < 0.0 or move_limit_sec < 0.0:
        raise ValueError(
            "win_pre, win_post, and move_limit_sec must be nonnegative"
        )
    return win_pre, win_post, move_limit_sec


def load_bandpass_config(config_file: Path) -> tuple[float, float]:
    """Load the active bandpass limits from rp_input.json."""
    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = ("min_freq", "max_freq")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{config_file} is missing bandpass fields: {missing}")
    min_freq, max_freq = float(config["min_freq"]), float(config["max_freq"])
    if min_freq <= 0.0 or max_freq <= min_freq:
        raise ValueError("Bandpass must satisfy 0 < min_freq < max_freq")
    return min_freq, max_freq


def load_align_phase_config(config_file: Path) -> str:
    """Load the active P or S alignment phase from rp_input.json."""
    with config_file.open("r", encoding="utf-8") as handle:
        phase = str(json.load(handle).get("align_phase", "S")).upper()
    if phase not in {"P", "S"}:
        raise ValueError(f"align_phase must be P or S; got {phase!r}")
    return phase


def load_station_static_settings(config_file: Path) -> tuple[Path, str]:
    """Return the station-static workbook and column configured in rp_input.json."""
    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    station_file = Path(
        config.get(
            "station_static_file",
            PATH_PREFIX / "event_sta_info" / "stations.xlsx",
        )
    )
    if not station_file.is_absolute():
        station_file = PATH_PREFIX / station_file
    station_column = str(
        config.get("station_static_column", "station static s")
    )
    return station_file, station_column


def load_station_statics(
    station_file: Path,
    station_column: str,
) -> dict[str, float]:
    """Load finite station-specific statics keyed by five-digit station ID."""
    stations = pd.read_excel(station_file, dtype={"station": str})
    required = {"station", station_column}
    missing = required - set(stations.columns)
    if missing:
        raise ValueError(
            f"{station_file} is missing required columns: {sorted(missing)}"
        )
    stations = stations[["station", station_column]].copy()
    stations["station"] = stations["station"].astype(str).str.zfill(5)
    stations[station_column] = pd.to_numeric(
        stations[station_column],
        errors="coerce",
    )
    stations = stations.dropna(subset=[station_column])
    if stations["station"].duplicated().any():
        raise ValueError(f"{station_file} contains duplicate station IDs")
    return dict(
        zip(
            stations["station"],
            stations[station_column].astype(float),
            strict=True,
        )
    )


def correlation_window_bounds(
    win_pre: float,
    win_post: float,
    move_limit_sec: float,
) -> tuple[float, float, float, float]:
    """Return correlation and lag-search bounds on the phase-relative axis."""
    window_start = COMMON_PHASE_PICK_TIME - win_pre
    window_end = COMMON_PHASE_PICK_TIME + win_post
    return (
        window_start,
        window_end,
        window_start - move_limit_sec,
        window_end + move_limit_sec,
    )


def collect_rt_traces(
    event_id: str,
    snippets_root: Path,
    info_root: Path,
    start_time: float,
    end_time: float,
    sampling_hz: int,
    align_phase: str,
    include_z: bool = True,
):
    """Read enough data for the phase-relative window and matched component traces."""
    event_depth, event_lat, event_lon, origin = load_event_metadata(event_id, info_root)
    stations = load_station_lookup(info_root)
    model = TauPyModel(model="iasp91")
    phase_arrival_by_station = {}
    for station_id, (station_lat, station_lon) in stations.items():
        distance_deg = float(locations2degrees(
            event_lat,
            event_lon,
            station_lat,
            station_lon,
        ))
        phase_arrival = predicted_phase_arrival_seconds(
            model,
            event_depth,
            distance_deg,
            align_phase,
        )
        if phase_arrival is not None:
            phase_arrival_by_station[str(station_id)] = phase_arrival
    if not phase_arrival_by_station:
        raise RuntimeError(f"No TauP {align_phase} arrivals found for {event_id}")

    input_start_time = min(phase_arrival_by_station.values()) + start_time - COMMON_PHASE_PICK_TIME
    input_end_time = max(phase_arrival_by_station.values()) + end_time - COMMON_PHASE_PICK_TIME
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
        start_time=input_start_time,
        end_time=input_end_time,
        timing_state=timing,
    )
    if horizontal_stream is None:
        raise RuntimeError(f"No horizontal snippets found for {event_id}")
    vertical_stream = None
    if include_z:
        vertical_stream, _vertical_raw_limits = read_waveforms_for_event(
            eve_id=event_id,
            channel="DPZ",
            process_as_three_comp_mode=False,
            horizontal_window_cache={},
            horizontal_raw_limits_cache={},
            name2ll=stations,
            eve_lat=event_lat,
            eve_lon=event_lon,
            origin=origin,
            data_path=snippets_root / event_id,
            input_is_snippets=True,
            sampling_hz=sampling_hz,
            start_time=input_start_time,
            end_time=input_end_time,
            timing_state=timing,
        )
        if vertical_stream is None:
            raise RuntimeError(f"No vertical snippets found for {event_id}")

    radial, _ = rotate_horizontals_to_component(
        horizontal_stream.copy(), "R", stations, event_lat, event_lon, timing
    )
    transverse, _ = rotate_horizontals_to_component(
        horizontal_stream.copy(), "T", stations, event_lat, event_lon, timing
    )
    transverse_by_station = {str(trace.stats.station): trace for trace in transverse}
    vertical_by_station = (
        {str(trace.stats.station): trace for trace in vertical_stream}
        if vertical_stream is not None
        else {}
    )
    pairs = [
        (
            trace,
            transverse_by_station[str(trace.stats.station)],
            (
                vertical_by_station[str(trace.stats.station)]
                if include_z
                else None
            ),
            phase_arrival_by_station[str(trace.stats.station)],
        )
        for trace in radial
        if (
            str(trace.stats.station) in transverse_by_station
            and (
                not include_z
                or str(trace.stats.station) in vertical_by_station
            )
            and str(trace.stats.station) in phase_arrival_by_station
        )
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
    align_phase: str,
    display_amplitude: float,
    win_pre: float,
    win_post: float,
    move_limit_sec: float,
    station_statics: dict[str, float] | None,
    static_components: frozenset[str] | set[str] | None = None,
    normalize_to_largest_component: bool = False,
) -> list[Path]:
    """Save TauP-phase-aligned Z/R/T overlays, with one station group per row."""
    if station_statics is None:
        active_static_components = frozenset()
    elif static_components is None:
        active_static_components = frozenset({"R", "T", "Z"})
    else:
        active_static_components = frozenset(static_components)
    invalid_components = active_static_components - {"R", "T", "Z"}
    if invalid_components:
        raise ValueError(
            "Unknown station-static components: "
            f"{sorted(invalid_components)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []
    if active_static_components and station_statics is not None:
        station_ids = {
            str(radial.stats.station).zfill(5)
            for radial, _transverse, _vertical, _phase_arrival in trace_pairs
        }
        missing_statics = sorted(station_ids - set(station_statics.keys()))
        if missing_statics:
            raise ValueError(
                f"Station statics are missing for {len(missing_statics)} plotted "
                f"stations, including {', '.join(missing_statics[:10])}"
            )
    for frame_index, first in enumerate(range(0, len(trace_pairs), traces_per_frame), start=1):
        subset = trace_pairs[first : first + traces_per_frame]
        fig, ax = plt.subplots(figsize=(12, max(6.0, 0.35 * len(subset) + 1.6)))
        offsets = np.arange(len(subset) - 1, -1, -1, dtype=float)
        (
            window_start,
            window_end,
            search_start,
            search_end,
        ) = correlation_window_bounds(win_pre, win_post, move_limit_sec)
        ax.axvspan(
            window_start,
            window_end,
            color="gold",
            alpha=0.14,
            label="Correlation window",
            zorder=0,
        )
        ax.axvline(window_start, color="goldenrod", lw=1.2, alpha=0.9)
        ax.axvline(window_end, color="goldenrod", lw=1.2, alpha=0.9)
        ax.axvline(
            search_start,
            color="tab:blue",
            lw=1.2,
            alpha=0.9,
            linestyle=":",
            label="Shift-search limits",
        )
        ax.axvline(
            search_end,
            color="tab:blue",
            lw=1.2,
            alpha=0.9,
            linestyle=":",
        )

        for row, (radial, transverse, vertical, phase_arrival) in enumerate(subset):
            station_id = str(radial.stats.station).zfill(5)
            station_static = (
                float(station_statics[station_id])
                if active_static_components and station_statics is not None
                else 0.0
            )
            r_time = trace_times_relative_to_predicted_phase(
                radial,
                origin,
                phase_arrival,
                station_static if "R" in active_static_components else 0.0,
            )
            t_time = trace_times_relative_to_predicted_phase(
                transverse,
                origin,
                phase_arrival,
                station_static if "T" in active_static_components else 0.0,
            )
            z_time = None
            if vertical is not None:
                z_time = trace_times_relative_to_predicted_phase(
                    vertical,
                    origin,
                    phase_arrival,
                    station_static if "Z" in active_static_components else 0.0,
                )
            shared_peak = None
            if normalize_to_largest_component:
                component_peaks = [
                    correlation_window_peak(radial, r_time, win_pre, win_post),
                    correlation_window_peak(transverse, t_time, win_pre, win_post),
                ]
                if vertical is not None and z_time is not None:
                    component_peaks.append(
                        correlation_window_peak(vertical, z_time, win_pre, win_post)
                    )
                shared_peak = max(component_peaks)
            radial_data = normalize_to_correlation_window_peak(
                radial,
                r_time,
                win_pre,
                win_post,
                shared_peak,
            )
            transverse_data = normalize_to_correlation_window_peak(
                transverse,
                t_time,
                win_pre,
                win_post,
                shared_peak,
            )
            ax.plot(
                r_time,
                display_amplitude * radial_data + offsets[row],
                color="tab:blue",
                lw=0.8,
            )
            ax.plot(
                t_time,
                display_amplitude * transverse_data + offsets[row],
                color="tab:orange",
                lw=0.8,
            )
            if vertical is not None and z_time is not None:
                vertical_data = normalize_to_correlation_window_peak(
                    vertical,
                    z_time,
                    win_pre,
                    win_post,
                    shared_peak,
                )
                ax.plot(
                    z_time,
                    display_amplitude * vertical_data + offsets[row],
                    color="0.45",
                    lw=0.8,
                )
            if start_time <= COMMON_PHASE_PICK_TIME <= end_time:
                ax.vlines(
                    COMMON_PHASE_PICK_TIME,
                    offsets[row] - 0.38,
                    offsets[row] + 0.38,
                    color="tab:green",
                    lw=1.0,
                    zorder=4,
                )
            if (
                active_static_components
                and start_time <= station_static <= end_time
            ):
                ax.vlines(
                    station_static,
                    offsets[row] - 0.30,
                    offsets[row] + 0.30,
                    color="tab:purple",
                    lw=1.2,
                    linestyle="--",
                    zorder=5,
                )
                ax.text(
                    station_static,
                    offsets[row] + 0.32,
                    f"{station_static:+.3f}",
                    color="tab:purple",
                    fontsize=5,
                    ha="center",
                    va="bottom",
                )

        ax.set_xlim(start_time, end_time)
        ax.set_ylim(-0.65, len(subset) - 0.35)
        ax.set_yticks(offsets)
        ax.set_yticklabels([str(pair[0].stats.station).zfill(5) for pair in subset], fontsize=8)
        ax.set_xlabel(f"Time relative to TauP-predicted {align_phase} (s)")
        ax.set_ylabel("Station (near to far)")
        has_z = any(pair[2] is not None for pair in subset)
        component_label = "Z/R/T" if has_z else "R/T"
        filename_component_label = "ZRT" if has_z else "RT"
        ax.set_title(
            f"{event_id}: TauP-{align_phase}-aligned {component_label} snippets, "
            f"{min_freq:g}-{max_freq:g} Hz "
            f"({'largest-component' if normalize_to_largest_component else 'per-trace'} "
            f"window-peak normalized; scale={display_amplitude:g}; "
            f"stations {first + 1}-{first + len(subset)})",
            fontweight="bold",
        )
        ax.plot([], [], color="tab:blue", lw=1.4, label="R")
        ax.plot([], [], color="tab:orange", lw=1.4, label="T")
        if has_z:
            ax.plot([], [], color="0.45", lw=1.4, label="Z")
        ax.plot([], [], color="tab:green", lw=1.4, label=f"Predicted {align_phase}")
        if active_static_components:
            shifted_component_label = "/".join(
                component
                for component in ("R", "T", "Z")
                if component in active_static_components
            )
            ax.plot(
                [],
                [],
                color="tab:purple",
                lw=1.4,
                linestyle="--",
                label=(
                    "Station static from stations.xlsx "
                    f"(applied to {shifted_component_label})"
                ),
            )
        ax.legend(loc="upper right")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()

        output_file = output_dir / (
            f"{event_id}_{filename_component_label}_snippets_{align_phase}_aligned_"
            f"{start_time:g}-{end_time:g}s_"
            f"{min_freq:g}-{max_freq:g}Hz_frame{frame_index:02d}.png"
        )
        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        output_files.append(output_file)
    return output_files


def resolve_station_static_components(
    args: argparse.Namespace,
) -> frozenset[str]:
    """Resolve global and per-component station-static command-line options."""
    defaults = {
        "R": APPLY_STATION_STATICS_R,
        "T": APPLY_STATION_STATICS_T,
        "Z": APPLY_STATION_STATICS_Z,
    }
    active = set()
    for component in ("R", "T", "Z"):
        component_setting = getattr(
            args,
            f"apply_station_statics_{component.lower()}",
        )
        if component_setting is not None:
            enabled = component_setting
        elif args.use_station_statics is not None:
            enabled = args.use_station_statics
        else:
            enabled = defaults[component]
        if enabled:
            active.add(component)
    return frozenset(active)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build command-line arguments for the standalone plotter."""
    parser = argparse.ArgumentParser(
        description="Plot TauP-aligned overlaid Z/R/T event snippets in offset frames."
    )
    parser.add_argument("--event", default=DEFAULT_EVENT)
    parser.add_argument(
        "--start-time",
        type=float,
        default=START_TIME,
        help=f"Window start relative to the predicted phase (default: {START_TIME:g} s).",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=END_TIME,
        help=f"Window end relative to the predicted phase (default: {END_TIME:g} s).",
    )
    parser.add_argument(
        "--min-freq",
        type=float,
        default=None,
        help="Override the min_freq value in rp_input.json.",
    )
    parser.add_argument(
        "--max-freq",
        type=float,
        default=None,
        help="Override the max_freq value in rp_input.json.",
    )
    parser.add_argument("--sampling-hz", type=int, default=250)
    parser.add_argument(
        "--display-amplitude",
        type=float,
        default=DEFAULT_DISPLAY_AMPLITUDE,
        help=(
            "Vertical multiplier after each trace is normalized to its "
            "correlation-window peak."
        ),
    )
    normalization_group = parser.add_mutually_exclusive_group()
    normalization_group.add_argument(
        "--normalize-to-largest-component",
        dest="normalize_to_largest_component",
        action="store_true",
        help=(
            "For each station, normalize all displayed components by the "
            "largest correlation-window peak among them."
        ),
    )
    normalization_group.add_argument(
        "--normalize-separately",
        dest="normalize_to_largest_component",
        action="store_false",
        help="Normalize each displayed component by its own correlation-window peak.",
    )
    parser.set_defaults(
        normalize_to_largest_component=NORMALIZE_TO_LARGEST_COMPONENT
    )
    parser.add_argument("--traces-per-frame", type=int, default=20)
    parser.add_argument("--snippets-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    static_group = parser.add_mutually_exclusive_group()
    static_group.add_argument(
        "--use-time-shift",
        "--use-station-statics",
        dest="use_station_statics",
        action="store_true",
        help="Apply station statics to every component unless individually overridden.",
    )
    static_group.add_argument(
        "--no-time-shift",
        "--no-station-statics",
        dest="use_station_statics",
        action="store_false",
        help="Disable station statics for every component unless individually overridden.",
    )
    parser.set_defaults(use_station_statics=None)
    for component in ("r", "t", "z"):
        component_group = parser.add_mutually_exclusive_group()
        component_group.add_argument(
            f"--apply-statics-{component}",
            f"--shift-{component}",
            dest=f"apply_station_statics_{component}",
            action="store_true",
            help=f"Apply the stations.xlsx static to {component.upper()}.",
        )
        component_group.add_argument(
            f"--no-statics-{component}",
            f"--no-shift-{component}",
            dest=f"apply_station_statics_{component}",
            action="store_false",
            help=f"Do not apply the stations.xlsx static to {component.upper()}.",
        )
        parser.set_defaults(
            **{f"apply_station_statics_{component}": None}
        )
    z_group = parser.add_mutually_exclusive_group()
    z_group.add_argument(
        "--include-z",
        dest="include_z",
        action="store_true",
        help="Read and plot the Z component.",
    )
    z_group.add_argument(
        "--no-z",
        dest="include_z",
        action="store_false",
        help="Do not read or plot the Z component.",
    )
    parser.set_defaults(include_z=USE_Z_COMPONENT)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    config_min_freq, config_max_freq = load_bandpass_config(RP_INPUT_FILE)
    align_phase = load_align_phase_config(RP_INPUT_FILE)
    min_freq = config_min_freq if args.min_freq is None else args.min_freq
    max_freq = config_max_freq if args.max_freq is None else args.max_freq
    static_components = set(resolve_station_static_components(args))
    if not args.include_z:
        static_components.discard("Z")
    static_component_label = (
        "/".join(
            component
            for component in ("R", "T", "Z")
            if component in static_components
        )
        if static_components
        else "off"
    )
    print(
        "Plot options: "
        f"station statics applied to={static_component_label}, "
        f"Z component={'on' if args.include_z else 'off'}"
    )
    if args.end_time <= args.start_time:
        raise ValueError("end-time must be greater than start-time")
    if not args.start_time <= COMMON_PHASE_PICK_TIME <= args.end_time:
        raise ValueError(
            "start-time and end-time must include the common predicted phase pick "
            f"at {COMMON_PHASE_PICK_TIME:g} s"
        )
    if min_freq <= 0.0 or max_freq <= min_freq:
        raise ValueError("frequency range must satisfy 0 < min-freq < max-freq")
    if args.display_amplitude <= 0.0:
        raise ValueError("display-amplitude must be positive")

    win_pre, win_post, move_limit_sec = load_correlation_window_config(
        RP_INPUT_FILE
    )
    station_statics = None
    if static_components:
        station_static_file, station_static_column = load_station_static_settings(
            RP_INPUT_FILE
        )
        station_statics = load_station_statics(
            station_static_file,
            station_static_column,
        )
    snippets_root = args.snippets_root or (PATH_PREFIX / f"Sgrams/Snippets_{args.sampling_hz}Hz")
    filename_component_label = "ZRT" if args.include_z else "RT"
    output_dir = args.output_dir or (
        PATH_PREFIX
        / "output"
        / (
            f"{args.event}_{filename_component_label}_snippets_{align_phase}_aligned_"
            f"{args.start_time:g}-{args.end_time:g}s_"
            f"{min_freq:g}-{max_freq:g}Hz"
        )
    )
    origin, trace_pairs = collect_rt_traces(
        args.event,
        snippets_root,
        PATH_PREFIX / "event_sta_info",
        args.start_time,
        args.end_time,
        args.sampling_hz,
        align_phase,
        args.include_z,
    )
    if not args.include_z:
        trace_pairs = [
            (radial_trace, transverse_trace, None, phase_arrival)
            for radial_trace, transverse_trace, _vertical_trace, phase_arrival
            in trace_pairs
        ]
    radial = [pair[0] for pair in trace_pairs]
    transverse = [pair[1] for pair in trace_pairs]
    vertical = [pair[2] for pair in trace_pairs if pair[2] is not None]
    timing = TimingState()
    preprocess_traces_bandpass(radial, min_freq, max_freq, timing)
    preprocess_traces_bandpass(transverse, min_freq, max_freq, timing)
    if vertical:
        preprocess_traces_bandpass(vertical, min_freq, max_freq, timing)
    output_files = plot_rt_frames(
        args.event,
        origin,
        trace_pairs,
        args.start_time,
        args.end_time,
        min_freq,
        max_freq,
        args.traces_per_frame,
        output_dir,
        align_phase,
        args.display_amplitude,
        win_pre,
        win_post,
        move_limit_sec,
        station_statics,
        static_components,
        args.normalize_to_largest_component,
    )
    component_label = "Z/R/T" if args.include_z else "R/T"
    print(
        f"Wrote {len(output_files)} {component_label} overlay frames to "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
