from __future__ import annotations

from datetime import timedelta, timezone
from dataclasses import dataclass, field
import time
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import read, Stream
from obspy.geodetics import gps2dist_azimuth
from obspy.geodetics import degrees2kilometers, locations2degrees
from obspy import UTCDateTime


@dataclass
class TimingState:
    start_cpu_time: float = field(default_factory=time.process_time)
    start_wall_time: float = field(default_factory=time.perf_counter)
    timing_reported: bool = False
    stage_wall_times: dict[str, float] = field(default_factory=dict)
    stage_cpu_times: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)


def add_stage_timing(state: TimingState, stage_name: str, wall_start: float, cpu_start: float) -> None:
    """Accumulate elapsed wall/cpu time for a named processing stage."""
    wall_dt = time.perf_counter() - wall_start
    cpu_dt = time.process_time() - cpu_start
    state.stage_wall_times[stage_name] = state.stage_wall_times.get(stage_name, 0.0) + wall_dt
    state.stage_cpu_times[stage_name] = state.stage_cpu_times.get(stage_name, 0.0) + cpu_dt
    state.stage_counts[stage_name] = state.stage_counts.get(stage_name, 0) + 1


def report_stage_timing(state: TimingState) -> None:
    """Print stage-level timing summary sorted by wall time."""
    if not state.stage_wall_times:
        return
    total_wall = sum(state.stage_wall_times.values())
    print("\033[36mStage timing breakdown (wall/cpu):\033[0m")
    for name, wall_sec in sorted(state.stage_wall_times.items(), key=lambda kv: kv[1], reverse=True):
        cpu_sec = state.stage_cpu_times.get(name, 0.0)
        calls = state.stage_counts.get(name, 0)
        frac = (100.0 * wall_sec / total_wall) if total_wall > 0 else 0.0
        print(
            f"  {name:<28} wall={wall_sec:7.2f}s  cpu={cpu_sec:7.2f}s  "
            f"calls={calls:3d}  ({frac:5.1f}%)"
        )


def report_timing_once(state: TimingState) -> None:
    """Report cpu and wall time before showing plots."""
    if state.timing_reported:
        return
    cpu_sec = time.process_time() - state.start_cpu_time
    wall_sec = time.perf_counter() - state.start_wall_time
    print(f"\033[31mTiming: cpu={cpu_sec:.2f}s  wall={wall_sec:.2f}s\033[0m")
    report_stage_timing(state)
    state.timing_reported = True


def resolve_component_key(channel: str, sel_comp: str) -> str:
    """Resolve storage key used in three-component aggregate output."""
    if channel == "DPZ":
        return "DPZ"
    if sel_comp == "R":
        return "R"
    if sel_comp == "T":
        return "T"
    return channel


def build_component_output_payload(
    record_fig,
    selected_rows: list,
    rejected_rows: list,
    stack_vec: np.ndarray,
    t_abs: np.ndarray,
    mask: np.ndarray,
    sample_rate: float,
    win_start: int,
    win_end: int,
    move_limit_sec_value: float,
    move_limit_samples: int,
    npts: int,
    start_t: float,
    end_t: float,
    eve_id: str,
    align_phase_name: str,
    origin,
    station_shifts: dict,
    station_corr: dict,
    calc_shifts: dict,
    n_pass_window: int,
    pass_window_ids: set,
    snippet_by_station: dict,
    ref_window: np.ndarray,
    p_traveltime,
    s_traveltime,
    name2ll: dict,
    selected_ids: set,
    aligned_traces_by_station: dict,
    t_ref,
):
    """Create deep-copied component payload used by three-component plotting."""
    payload = {
        "fig": record_fig,
        "all_rows": [(r[0], r[1], r[2].copy()) for r in (selected_rows + rejected_rows)],
        "stack_vec": stack_vec.copy(),
        "t_abs": t_abs.copy(),
        "mask": mask.copy(),
        "sample_rate": sample_rate,
        "win_start": win_start,
        "win_end": win_end,
        "move_limit_sec": move_limit_sec_value,
        "move_limit_samples": move_limit_samples,
        "npts": npts,
        "start_time": start_t,
        "end_time": end_t,
        "eve_id": eve_id,
        "align_phase": align_phase_name,
        "origin": origin,
        "station_shifts": station_shifts.copy(),
        "station_corr": station_corr.copy(),
        "calc_shifts": calc_shifts.copy(),
        "n_pass_window": int(n_pass_window),
        "pass_window_ids": sorted(list(pass_window_ids), key=lambda s: int(s)),
        "snippet_by_station": {k: v.copy() for k, v in snippet_by_station.items()},
        "ref_window": ref_window.copy(),
        "p_traveltime": None if p_traveltime is None else float(p_traveltime),
        "s_traveltime": None if s_traveltime is None else float(s_traveltime),
        "station_ll": {k: (float(v[0]), float(v[1])) for k, v in name2ll.items()},
        # Stations that passed Stage-2 screening for this component
        "selected_ids": sorted(list(selected_ids), key=lambda s: int(s)),
        "aligned_traces_by_station": {k: v.copy() for k, v in aligned_traces_by_station.items()},
        "t_ref": t_ref,
    }
    return payload


def build_alignment_products_payload(
    npts: int,
    sample_rate: float,
    move_limit_samples: int,
    win_start: int,
    win_end: int,
    calc_shifts: dict,
    aligned_stack: np.ndarray,
    selected_aligned_stack: np.ndarray,
    selected_ids: set,
    station_corr: dict,
    n_pass_window: int,
    pass_window_ids: set,
    snippet_by_station: dict,
    ref_window: np.ndarray,
    selected_rows: list,
    rejected_rows: list,
    station_shifts: dict,
    aligned_traces_by_station: dict,
    t_abs: np.ndarray,
    mask: np.ndarray,
    stack_vec: np.ndarray,
):
    """Assemble the alignment products payload used by downstream plotting/output."""
    return {
        "npts": npts,
        "sample_rate": sample_rate,
        "move_limit_samples": move_limit_samples,
        "win_start": win_start,
        "win_end": win_end,
        "calc_shifts": calc_shifts,
        "aligned_stack": aligned_stack,
        "selected_aligned_stack": selected_aligned_stack,
        "selected_ids": selected_ids,
        "station_corr": station_corr,
        "n_pass_window": n_pass_window,
        "pass_window_ids": pass_window_ids,
        "snippet_by_station": snippet_by_station,
        "ref_window": ref_window,
        "selected_rows": selected_rows,
        "rejected_rows": rejected_rows,
        "station_shifts": station_shifts,
        "aligned_traces_by_station": aligned_traces_by_station,
        "t_abs": t_abs,
        "mask": mask,
        "stack_vec": stack_vec,
    }


def make_event_output_dir(base_prefix: str, eve_id: str) -> Path:
    """Create and return output directory for one event."""
    save_path = Path(base_prefix + "output")
    save_dir = save_path / eve_id
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def load_event_metadata(eve_id: str, info_dir: Path):
    """Load event row and return key metadata for one event id."""
    eve_info = pd.read_csv(info_dir / "catalog_20220930_8events.csv")
    row = eve_info.loc[eve_info["evid"] == eve_id].iloc[0]
    event_depth = float(row["depth"])
    eve_lat = float(row["latitude"])
    eve_lon = float(row["longitude"])
    origin = UTCDateTime(str(row["origin_time"]))
    return event_depth, eve_lat, eve_lon, origin


def load_station_lookup(info_dir: Path):
    """Read station coordinates and return station->(lat, lon) lookup."""
    station_file = info_dir / "stations.txt"
    sta_info = np.genfromtxt(
        station_file,
        dtype=[("name", "U10"), ("lat", "f8"), ("lon", "f8")],
        usecols=(0, 1, 2),
        comments="#",
    )
    sta_name = np.array([s.decode() if hasattr(s, "decode") else s for s in sta_info["name"]])
    sta_lat = sta_info["lat"]
    sta_lon = sta_info["lon"]
    return {sta_name[i]: (sta_lat[i], sta_lon[i]) for i in range(len(sta_name))}


def read_waveforms_for_event(
    eve_id: str,
    channel: str,
    process_as_three_comp_mode: bool,
    horizontal_window_cache: dict,
    horizontal_raw_limits_cache: dict,
    name2ll: dict,
    eve_lat: float,
    eve_lon: float,
    origin,
    data_path: Path,
    sps_rate: str,
    start_time: float,
    end_time: float,
    verbose: bool,
    timing_state: TimingState,
):
    """Read (or reuse) windowed traces for an event/channel and return stream + raw limits."""
    use_horizontal_cache = (
        process_as_three_comp_mode
        and channel == "DP2"
        and eve_id in horizontal_window_cache
    )

    if use_horizontal_cache:
        st_window = horizontal_window_cache[eve_id].copy()
        raw_limits_by_station = horizontal_raw_limits_cache.get(eve_id, {}).copy()
        print("Reusing horizontal read cache for transverse component.")
        return st_window, raw_limits_by_station

    st_all = Stream()
    stanum = 0
    raw_limits_by_station = {}

    _read_wall_start = time.perf_counter()
    _read_cpu_start = time.process_time()
    for sta, (slat, slon) in name2ll.items():
        code_num = int(sta)
        code_str = f"{code_num:05d}"

        if channel in ["DP1", "DP2"]:
            chan_list_this_sta = ["DP1", "DP2"]
        else:
            chan_list_this_sta = [channel]

        dist_deg = locations2degrees(eve_lat, eve_lon, slat, slon)
        dist_km = degrees2kilometers(dist_deg)

        first_chan_for_sta = True

        for ch_read in chan_list_this_sta:
            fpath = (
                data_path
                / code_str
                / f"{ch_read}.D"
                / f"7V.{code_str}.00.{ch_read}.D.2022.273.{sps_rate}.mseed"
            )
            if verbose:
                if stanum % 20 == 0:
                    print(f"Reading station {sta} channel {ch_read} from {fpath}")
            if not fpath.exists():
                if verbose:
                    print("No such file")
                continue

            st_win = read(
                str(fpath),
                starttime=origin + start_time,
                endtime=origin + end_time,
            )
            if len(st_win) == 0:
                continue
            tr = st_win[0]
            if sta not in raw_limits_by_station:
                raw_limits_by_station[sta] = (tr.stats.starttime, tr.stats.endtime)

            tr.stats.dist_km = dist_km
            tr.stats.dist_deg = dist_deg
            tr.stats.relatime = tr.times(reftime=origin)
            tr.stats.station = sta
            st_all.append(tr)

            if first_chan_for_sta:
                stanum += 1
                if stanum % 100 == 0:
                    print(f"Event {eve_id}: processed {stanum} stations...")
                first_chan_for_sta = False
    add_stage_timing(timing_state, "waveform_read_slice", _read_wall_start, _read_cpu_start)

    st_all.sort(keys=["dist_km"])
    if not st_all:
        print(f"No traces found for event {eve_id}, skip.")
        return None, None

    st_window = Stream()
    kept = 0
    for tr in st_all:
        if tr.stats.endtime >= origin + start_time and tr.stats.starttime <= origin + end_time:
            tr_i = tr.copy()
            if tr_i is not None and tr_i.stats.npts > 0:
                tr_i.stats.station = tr.stats.station
                st_window.append(tr_i)
                kept += 1

    if kept == 0:
        print("  No data in plot window (start_time to end_time).")
        return None, None

    if process_as_three_comp_mode and channel == "DP1":
        horizontal_window_cache[eve_id] = st_window.copy()
        horizontal_raw_limits_cache[eve_id] = raw_limits_by_station.copy()

    return st_window, raw_limits_by_station


def select_reference_trace(st_comp, name2ll: dict):
    """Pick reference station closest to array center and return (id, trace)."""
    st_comp.sort(keys=["dist_km"])
    if len(st_comp) == 0:
        return None, None

    station_ids = sorted({str(tr.stats.station) for tr in st_comp})
    center_lats = [name2ll[s][0] for s in station_ids if s in name2ll]
    center_lons = [name2ll[s][1] for s in station_ids if s in name2ll]

    ref_station_id = None
    if len(center_lats) > 0 and len(center_lons) > 0:
        center_lat = float(np.mean(center_lats))
        center_lon = float(np.mean(center_lons))
        best_dist_m = None
        for sid in station_ids:
            if sid not in name2ll:
                continue
            slat, slon = name2ll[sid]
            dist_m, _, _ = gps2dist_azimuth(center_lat, center_lon, slat, slon)
            if best_dist_m is None or dist_m < best_dist_m:
                best_dist_m = dist_m
                ref_station_id = sid

    if ref_station_id is None:
        ref_station_id = str(st_comp[0].stats.station)

    ref_trace = st_comp.select(station=ref_station_id)
    if len(ref_trace) == 0:
        ref_trace = [st_comp[0]]
        ref_station_id = str(ref_trace[0].stats.station)
    return ref_station_id, ref_trace[0]


def print_reference_summary(ref_station_id: str, ref_trace, raw_limits_by_station: dict):
    """Print reference station and data-window summary."""
    ref_trace_dur = float(ref_trace.stats.npts) / float(ref_trace.stats.sampling_rate)
    print(
        f"    Reference station (auto): {ref_station_id} (closest to array center)  "
        f"dist_km={ref_trace.stats.dist_km:.2f}  dur_s={ref_trace_dur:.2f}"
    )
    print(f"    Epicentral distance ~= {ref_trace.stats.dist_km:.1f} km")

    ref_start_dt = ensure_utc_datetime(ref_trace.stats.starttime.datetime)
    ref_end_dt = ensure_utc_datetime(ref_trace.stats.endtime.datetime)
    print(
        "\033[32m    Reference seismogram UTC window: "
        f"{ref_start_dt.isoformat()} to {ref_end_dt.isoformat()}\033[0m"
    )
    if ref_station_id in raw_limits_by_station:
        raw_start, raw_end = raw_limits_by_station[ref_station_id]
        raw_start_dt = ensure_utc_datetime(raw_start.datetime)
        raw_end_dt = ensure_utc_datetime(raw_end.datetime)
        print(
            "\033[32m    Reference read UTC window: "
            f"{raw_start_dt.isoformat()} to {raw_end_dt.isoformat()}\033[0m"
        )


def compute_phase_travel_times(model_obj, event_depth: float, ref_trace, origin_time, align_phase_name: str):
    """Compute P/S travel times at reference station and selected alignment phase time."""
    ref_deg = float(ref_trace.stats.dist_deg)
    tts = model_obj.get_travel_times(
        source_depth_in_km=event_depth,
        distance_in_degree=ref_deg,
        phase_list=["p", "P", "s", "S"],
    )

    p_traveltime = None
    s_traveltime = None
    p_arrival_time = None
    s_arrival_time = None

    for tt in reversed(tts):
        if tt.phase.name.upper() == "P":
            p_traveltime = float(tt.time)
            p_arrival_time = origin_time + p_traveltime
        if tt.phase.name.upper() == "S":
            s_traveltime = float(tt.time)
            s_arrival_time = origin_time + s_traveltime

    if align_phase_name == "P" and p_traveltime is not None:
        phase_traveltime = float(p_traveltime)
    elif align_phase_name == "S" and s_traveltime is not None:
        phase_traveltime = float(s_traveltime)
    else:
        phase_traveltime = None

    return p_traveltime, s_traveltime, p_arrival_time, s_arrival_time, phase_traveltime


def compute_taup_station_shifts(
    model_obj,
    st_comp,
    event_depth: float,
    align_phase_name: str,
    t_ref,
    timing_state: TimingState,
) -> dict:
    """Compute TauP time shifts per station relative to reference arrival time."""
    calc_shifts = {}
    if t_ref is None:
        return calc_shifts

    phase_key = align_phase_name.upper()
    _taup_wall_start = time.perf_counter()
    _taup_cpu_start = time.process_time()
    for tr in st_comp:
        station_id = str(tr.stats.station)
        dist_deg = float(tr.stats.dist_deg)

        tts_sta = model_obj.get_travel_times(
            source_depth_in_km=event_depth,
            distance_in_degree=dist_deg,
            phase_list=[phase_key.lower(), phase_key.upper()],
        )
        t_sta = None
        for tt in reversed(tts_sta):
            if tt.phase.name.upper() == phase_key:
                t_sta = float(tt.time)
                break

        if t_sta is not None:
            calc_shifts[station_id] = t_sta - t_ref
    add_stage_timing(timing_state, "taup_station_shifts", _taup_wall_start, _taup_cpu_start)
    return calc_shifts


def compute_alignment_setup(
    st_comp,
    ref_trace,
    move_limit_sec: float,
    start_time: float,
    win_pre: float,
    win_post: float,
    t_ref,
):
    """Compute shared setup values used across alignment stages."""
    npts = min(tr.stats.npts for tr in st_comp)
    sample_rate = float(st_comp[0].stats.sampling_rate)
    ref = ref_trace.data[:npts]
    move_limit_samples = int(round(move_limit_sec * sample_rate))

    t0 = float(t_ref) if t_ref is not None else 0.0
    center_time = t0
    win_start = int(max(0, sample_rate * ((center_time - start_time) - win_pre)))
    win_end = int(min(npts, sample_rate * ((center_time - start_time) + win_post)))

    return {
        "npts": npts,
        "sample_rate": sample_rate,
        "ref": ref,
        "move_limit_samples": move_limit_samples,
        "win_start": win_start,
        "win_end": win_end,
    }


def normalize_traces_in_window(st_comp, win_start: int, win_end: int) -> None:
    """Normalize each trace by max amplitude within the correlation window."""
    for tr in st_comp:
        win = tr.data[win_start:win_end]
        mx = np.max(np.abs(win)) if win.size > 0 else 0.0
        if mx > 0:
            tr.data = tr.data / mx


def compute_stage1_aligned_stack(
    st_comp,
    ref: np.ndarray,
    npts: int,
    sample_rate: float,
    win_start: int,
    win_end: int,
    move_limit_samples: int,
    calc_shifts: dict,
    timing_state: TimingState,
) -> np.ndarray:
    """Stage 1: align traces to reference and return normalized stack."""
    aligned_stack = np.zeros(npts)
    _stage1_wall_start = time.perf_counter()
    _stage1_cpu_start = time.process_time()
    for tr in st_comp:
        d = tr.data[:npts]
        lag0 = 0
        rolled = shift_left_zeropad(d, lag0)

        station_id = str(tr.stats.station)
        if station_id in calc_shifts:
            expected_shift_samples = int(round(calc_shifts[station_id] * sample_rate))
            rolled_expected = shift_left_zeropad(rolled, expected_shift_samples)
            lag1 = expected_shift_samples + compute_lag(
                ref, rolled_expected, win_start, win_end, move_limit_samples
            )
        else:
            lag1 = lag0 + compute_lag(ref, rolled, win_start, win_end, move_limit_samples)

        aligned_stack += shift_left_zeropad(d, lag1)
    add_stage_timing(timing_state, "align_stage1", _stage1_wall_start, _stage1_cpu_start)

    win = aligned_stack[win_start:win_end]
    mx = np.max(np.abs(win)) if win.size > 0 else 0.0
    if mx > 0:
        aligned_stack = aligned_stack / mx
    return aligned_stack


def compute_stage2_screened_stack(
    st_comp,
    aligned_stack: np.ndarray,
    npts: int,
    sample_rate: float,
    win_start: int,
    win_end: int,
    move_limit_samples: int,
    calc_shifts: dict,
    r_window_min: float,
    timing_state: TimingState,
):
    """Stage 2: align to Stage-1 stack, score window correlation, and keep passing traces."""
    selected_aligned_stack = np.zeros(npts)
    selected_ids = set()
    station_corr = {}
    n_pass_window = 0
    pass_window_ids = set()
    snippet_by_station = {}
    ref_window = aligned_stack[win_start:win_end]

    _stage2_wall_start = time.perf_counter()
    _stage2_cpu_start = time.process_time()
    for tr in st_comp:
        d = tr.data[:npts]
        station_id = str(tr.stats.station)

        lag0 = 0
        rolled = shift_left_zeropad(d, lag0)

        if station_id in calc_shifts:
            expected_shift_samples = int(round(calc_shifts[station_id] * sample_rate))
            rolled_expected = shift_left_zeropad(rolled, expected_shift_samples)
            lag2 = expected_shift_samples + compute_lag(
                aligned_stack,
                rolled_expected,
                win_start,
                win_end,
                move_limit_samples,
            )
        else:
            lag2 = lag0 + compute_lag(
                aligned_stack, rolled, win_start, win_end, move_limit_samples
            )

        aligned_data = shift_left_zeropad(d, lag2)
        aligned_window = aligned_data[win_start:win_end]
        snippet_by_station[station_id] = aligned_window.copy()

        if np.linalg.norm(aligned_window) == 0 or np.linalg.norm(ref_window) == 0:
            r_window = 0.0
        else:
            r_window = float(
                np.dot(aligned_window, ref_window)
                / (np.linalg.norm(aligned_window) * np.linalg.norm(ref_window))
            )

        if r_window >= r_window_min:
            n_pass_window += 1
            pass_window_ids.add(station_id)

        station_corr[station_id] = r_window

        if r_window >= r_window_min:
            selected_aligned_stack += aligned_data
            selected_ids.add(station_id)
        else:
            print(f"    Rejected {station_id}: r_win={r_window:.2f}")
    add_stage_timing(timing_state, "align_stage2_screen", _stage2_wall_start, _stage2_cpu_start)

    win = selected_aligned_stack[win_start:win_end]
    mx = np.max(np.abs(win)) if win.size > 0 else 0.0
    if mx > 0:
        selected_aligned_stack = selected_aligned_stack / mx

    return {
        "selected_aligned_stack": selected_aligned_stack,
        "selected_ids": selected_ids,
        "station_corr": station_corr,
        "n_pass_window": n_pass_window,
        "pass_window_ids": pass_window_ids,
        "snippet_by_station": snippet_by_station,
        "ref_window": ref_window,
    }


def compute_stage3_finalized_rows(
    st_comp,
    ref: np.ndarray,
    ref_station_id: str,
    selected_ids: set,
    calc_shifts: dict,
    npts: int,
    sample_rate: float,
    win_start: int,
    win_end: int,
    move_limit_samples: int,
    name2ll: dict,
    eve_lat: float,
    eve_lon: float,
    timing_state: TimingState,
):
    """Stage 3: align traces on reference timebase and build selected/rejected rows."""
    selected_rows = []  # (dist_km, station_id, y_aligned_norm)
    rejected_rows = []
    aligned_bank = []
    aligned_bank_all = []
    station_shifts = {}
    aligned_traces_by_station = {}

    _stage3_wall_start = time.perf_counter()
    _stage3_cpu_start = time.process_time()
    for tr in st_comp:
        x = tr.data[:npts]
        station_id = str(tr.stats.station)

        if station_id == ref_station_id:
            lag3 = 0
            y = x.copy()
        else:
            if station_id in calc_shifts:
                expected_shift_samples = int(round(calc_shifts[station_id] * sample_rate))
                x_expected = shift_left_zeropad(x, expected_shift_samples)
                lag_delta = compute_lag(
                    ref, x_expected, win_start, win_end, move_limit_samples
                )
                lag3 = expected_shift_samples + lag_delta
            else:
                lag3 = compute_lag(ref, x, win_start, win_end, move_limit_samples)
            y = shift_left_zeropad(x, lag3)

        station_shifts[station_id] = {
            "lag_samples": lag3,
            "lag_seconds": lag3 / sample_rate,
        }

        win = y[win_start:win_end]
        my = np.max(np.abs(win)) if win.size > 0 else 1.0
        if my > 0:
            y = y / my

        aligned_traces_by_station[station_id] = y.copy()
        aligned_bank_all.append(y)

        slat, slon = name2ll[station_id]
        dist_m, _, _ = gps2dist_azimuth(eve_lat, eve_lon, slat, slon)
        dist_km = dist_m / 1000.0

        if station_id in selected_ids:
            selected_rows.append((dist_km, station_id, y))
            aligned_bank.append(y)
        else:
            rejected_rows.append((dist_km, station_id, y))
    add_stage_timing(timing_state, "align_stage3_finalize", _stage3_wall_start, _stage3_cpu_start)

    selected_rows.sort(key=lambda t: t[0])
    rejected_rows.sort(key=lambda t: t[0])
    print(
        f"    Final lag3 traces: selected={len(selected_rows)}, "
        f"rejected={len(rejected_rows)}, total={len(selected_rows) + len(rejected_rows)}"
    )

    return {
        "selected_rows": selected_rows,
        "rejected_rows": rejected_rows,
        "aligned_bank": aligned_bank,
        "aligned_bank_all": aligned_bank_all,
        "station_shifts": station_shifts,
        "aligned_traces_by_station": aligned_traces_by_station,
    }


def compute_time_axis_and_stack(
    start_time: float,
    end_time: float,
    npts: int,
    sample_rate: float,
    aligned_bank_all: list,
    win_start: int,
    win_end: int,
):
    """Build time axis/mask and normalized final stack from aligned traces."""
    t_abs = start_time + (np.arange(npts) / sample_rate)
    mask = (t_abs >= start_time) & (t_abs <= end_time)

    if len(aligned_bank_all) > 0:
        stack_vec = np.mean(np.vstack(aligned_bank_all), axis=0)
        win = stack_vec[win_start:win_end]
        ms = np.max(np.abs(win)) if win.size > 0 else 1.0
        if ms > 0:
            stack_vec = stack_vec / ms
    else:
        stack_vec = np.zeros_like(t_abs)

    return {
        "t_abs": t_abs,
        "mask": mask,
        "stack_vec": stack_vec,
    }


def compute_lag(
    ref: np.ndarray,
    d: np.ndarray,
    win_start: int,
    win_end: int,
    move_limit_samples: int,
) -> int:
    """Compute integer lag (samples) by maximizing correlation within a short window."""
    ref_window = ref[win_start:win_end]
    d_window = d[win_start - move_limit_samples : win_end + move_limit_samples]
    corr = np.correlate(d_window, ref_window, mode="valid")
    return int(np.argmax(corr) - move_limit_samples)


def shift_left_zeropad(x: np.ndarray, n: int) -> np.ndarray:
    """Shift 1D array left by n samples with zero padding (no wrap-around)."""
    x = np.asarray(x)
    y = np.zeros_like(x)

    if n == 0:
        y[:] = x
        return y

    if n > 0:
        if n >= x.size:
            return y
        y[:-n] = x[n:]
        return y

    n = -n
    if n >= x.size:
        return y
    y[n:] = x[:-n]
    return y


def ensure_utc_datetime(dt_obj):
    """Return a timezone-aware UTC datetime for printing/labeling."""
    if dt_obj.tzinfo is None:
        return dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj.astimezone(timezone.utc)


def correlation_time_bounds(start_t, win_start_samp, win_end_samp, samp_rate, move_sec, npts):
    """Compute correlation window and search bounds in seconds since origin."""
    t_win_start = start_t + (win_start_samp / samp_rate)
    t_win_end = start_t + (win_end_samp / samp_rate)
    t_explore_start = max(start_t, t_win_start - move_sec)
    t_explore_end = min(start_t + (npts / samp_rate), t_win_end + move_sec)
    return t_win_start, t_win_end, t_explore_start, t_explore_end


def draw_correlation_markers(ax, start_t, win_start_samp, win_end_samp, samp_rate, move_sec, npts):
    """Draw yellow (window) and green (search) vertical bounds on one axis."""
    t_win_start, t_win_end, t_explore_start, t_explore_end = correlation_time_bounds(
        start_t, win_start_samp, win_end_samp, samp_rate, move_sec, npts
    )
    ax.axvline(x=t_win_start, color="y", lw=2, alpha=0.9, zorder=7)
    ax.axvline(x=t_win_end, color="y", lw=2, alpha=0.9, zorder=7)
    ax.axvline(x=t_explore_start, color="g", lw=2, alpha=0.9, zorder=7)
    ax.axvline(x=t_explore_end, color="g", lw=2, alpha=0.9, zorder=7)


def set_figure_title(fig, title: str) -> None:
    """Set a descriptive window title if the backend supports it."""
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass


def get_component_selection(all_channels_mode: bool, comp: str):
    """Return (channels, process_as_three_comp, selected_components)."""
    if all_channels_mode:
        return ["DPZ", "DP1", "DP2"], True, ["Z", "R", "T"]
    if comp == "Z":
        return ["DPZ"], False, ["Z"]
    if comp in ("R", "T"):
        # Single-component R/T still reads both horizontals for rotation.
        return ["DP1"], False, [comp]
    raise ValueError("component must be 'Z', 'R', or 'T'")


def add_catalog_event_lines(ax, origin_time, catalog_df, tmin, tmax) -> None:
    """Draw vertical lines for each catalog event time on a time-since-origin axis."""
    if origin_time is None or catalog_df is None:
        return
    if "origin_time" not in catalog_df.columns:
        print("[WARN] Catalog missing 'origin_time' column; no event lines drawn.")
        return

    color_map = {0: "red", 1: "black", 2: "green"}
    for _, row in catalog_df.iterrows():
        try:
            evt_time = UTCDateTime(str(row["origin_time"]))
        except Exception:
            continue
        dt = float(evt_time - origin_time)
        if dt < tmin or dt > tmax:
            continue
        skip_val = row.get("skip", 0)
        try:
            skip_int = int(skip_val)
        except Exception:
            skip_int = 0
        color = color_map.get(skip_int, "red")
        ax.axvline(x=dt, color=color, lw=1.1, alpha=0.8, zorder=6)


def add_utc_time_axis(ax, origin_time, tick_tz=timezone.utc, label_size: int = 10) -> None:
    """Add a bottom UTC axis that mirrors the primary x-axis ticks."""
    if origin_time is None:
        return
    origin_dt_utc = ensure_utc_datetime(origin_time.datetime)
    ax_time = ax.twiny()
    ax_time.set_xlim(ax.get_xlim())
    ax_time.xaxis.set_label_position("bottom")
    ax_time.xaxis.set_ticks_position("bottom")
    ax_time.spines["bottom"].set_position(("outward", 36))
    ax_time.spines["top"].set_visible(False)

    ticks = ax.get_xticks()
    labels = [
        (origin_dt_utc + timedelta(seconds=float(t))).astimezone(tick_tz).strftime("%H:%M:%S")
        for t in ticks
    ]
    ax_time.set_xticks(ticks)
    ax_time.set_xticklabels(labels)
    date_str = origin_dt_utc.date().isoformat()
    ax_time.set_xlabel(f"UTC time ({date_str})", fontsize=label_size)
