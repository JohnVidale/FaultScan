import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

from obspy import read, UTCDateTime, Stream
from obspy.geodetics import degrees2kilometers, locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel
from obspy.signal.rotate import rotate_ne_rt

min_freq, max_freq            = 0.5, 10.0 # Bandpass filter (Hz)
start_time, end_time          = 0.0, 10.0 # Plotting time window (seconds since origin)
short_win_pre, short_win_post = 0.8,  0.8 # Short-window correlation parameters (seconds)
r_window_min, r_whole_min     = 0.8,  0.5 # Minimum correlation coefficients for trace selection
move_limit_sec                = 0.3       # Maximum allowed shift (seconds) searched in compute_lag

# Run modes
all_channels = True  # Changed to True to process all channels
all_events   = False # Process all events in the event list [missing file]
event       = "CI_40353544" # Single run selection (used when the corresponding "all_*" is False)
component   = "T"       # Component selection: 'Z', 'R', or 'T' (user-facing choice)
align_phase = "S"       # Alignment phase 'P' or 'S'

# Paths
info_root = Path("/Users/vidale/Documents/Research/Mingze_SJF/20220930_events_cut/event_sta_info")
data_path = Path("/Users/vidale/Documents/Research/Mingze_SJF/20220930_events_cut/07_1hour_20220930")

# plotting / stacking options
stack_all_traces_for_final  = True  # Final stack mode: True = use ALL traces after final lag3; False = use only selected traces
rt_stats_use_all_stations   = True  # R–T comparison mode: True = use ALL stations with both R and T; False = use only stations selected in BOTH R and T
stack_only_zero_rt_diff     = False # Stack mode for R/T in the combined plot: True = stack ONLY stations with (R shift - T shift) == 0
show_diff_rt_plot           = True  # Show R T differences in the combined plot
show_rt_filter_compare_plot = True  # Compare stacks with/without R–T screening (overlay: black=before, red=after)
save_rt_filter_station_list = True  # Save / reuse R–T screening result (stations with (R shift - T shift) == 0)

# When running Z-only mode (all_channels=False & component='Z'), read the saved station list and
# build a filtered stack + compare against the unfiltered stack.
use_saved_rt_filter_for_z_stack = True

# Travel-time model
model = TauPyModel(model="iasp91")

# ===================== Helper functions =====================
def compute_lag(ref: np.ndarray, d: np.ndarray, win_start: int, win_end: int) -> int:
        """Compute integer lag (samples) by maximizing correlation within a short window.

        This matches the original implementation:
            - ref_window = ref[win_start:win_end]
        - d_window   = d[win_start-move_limit_samples : win_end+move_limit_samples]
            - corr = np.correlate(d_window, ref_window, mode='valid')
        - lag = argmax(corr) - move_limit_samples

        Returns:
                Best lag (integer samples). Positive lag advances the target waveform.
        """
        ref_window = ref[win_start:win_end]
        d_window = d[win_start - move_limit_samples : win_end + move_limit_samples]
        corr = np.correlate(d_window, ref_window, mode="valid")
        return int(np.argmax(corr) - move_limit_samples)


def shift_left_zeropad(x: np.ndarray, n: int) -> np.ndarray:
    """Shift 1D array left by n samples with zero padding (no wrap-around).

    Equivalent to np.roll(x, -n) but WITHOUT circular wrap.
      - n > 0: advance in time
      - n < 0: delay
    """
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

    # n < 0
    n = -n
    if n >= x.size:
        return y
    y[n:] = x[:-n]
    return y


# ===================== Event list =====================
if all_events:
    events_file = info_root / "20230111events.txt"
    with open(events_file, "r") as f:
        events = [line.strip() for line in f if line.strip()]
else:
    events = [event]


# ===================== Channel / component selection =====================
# User-facing components: Z, R, T
if all_channels:
    channels = ["DPZ", "DP1", "DP2"]
    process_as_three_comp = True
    sel_comp_list = ["Z", "R", "T"]
else:
    process_as_three_comp = False
    if component == "Z":
        channels = ["DPZ"]
        sel_comp_list = ["Z"]
    elif component in ("R", "T"):
        # Read both horizontals internally by iterating over R and T
        channels = ["DP1", "DP2"]
        sel_comp_list = [component]
    else:
        raise ValueError("component must be 'Z', 'R', or 'T'")


# ===================== Main loop =====================
# Storage for three-component mode
if process_as_three_comp:
    all_component_data = {}

for idx, channel in enumerate(channels):
    sel_comp = sel_comp_list[idx]
    
    print(f"Processing channel: {channel}")

    for eve_id in events:
        print(f"==========Processing event {eve_id}===========")

        # ---- Read event info ----
        eve_info = pd.read_csv(info_root / "catalog_20220930_8events.csv")
        row = eve_info.loc[eve_info["evid"] == eve_id].iloc[0]

        event_depth = float(row["depth"])
        eve_lat, eve_lon = float(row["latitude"]), float(row["longitude"])
        origin = UTCDateTime(str(row["origin_time"]))

        # ---- Read station list ----
        station_file = info_root / "stations.txt"
        sta_info = np.genfromtxt(
            station_file,
            dtype=[("name", "U10"), ("lat", "f8"), ("lon", "f8")],
            usecols=(0, 1, 2),
            comments="#",
        )
        sta_name = np.array(
            [s.decode() if hasattr(s, "decode") else s for s in sta_info["name"]]
        )
        sta_lat = sta_info["lat"]
        sta_lon = sta_info["lon"]
        name2ll = {sta_name[i]: (sta_lat[i], sta_lon[i]) for i in range(len(sta_name))}

        # ---- Read waveforms for all stations ----
        st_all = Stream()
        stanum = 0

        for sta, (slat, slon) in name2ll.items():
            code_num = int(sta)
            code_str = f"{code_num:05d}"

            # If requested channel is horizontal, read both DP1 and DP2 (needed for rotation).
            if channel in ["DP1", "DP2"]:
                chan_list_this_sta = ["DP1", "DP2"]
            else:
                chan_list_this_sta = [channel]

            # Epicentral distance
            dist_deg = locations2degrees(eve_lat, eve_lon, slat, slon)
            dist_km = degrees2kilometers(dist_deg)

            first_chan_for_sta = True

            for ch_read in chan_list_this_sta:
                fpath = (
                    data_path
                    / code_str
                    / f"{ch_read}.D"
                    / f"7V.{code_str}.00.{ch_read}.D.2022.273.down50.mseed"
                )
                if stanum % 20 == 0:
                    print(f"Reading station {sta} channel {ch_read} from {fpath}")
                if not fpath.exists():
                    print("No such file")
                    continue

                # Read the component and slice [origin, origin + end_time]
                tr = read(str(fpath))[0]
                tr = tr.slice(starttime=origin, endtime=origin + end_time)
                if (tr is None) or (tr.stats.npts == 0):
                    continue

                # Store metadata for later plotting/selection
                tr.stats.dist_km = dist_km
                tr.stats.dist_deg = dist_deg
                tr.stats.relatime = tr.times(reftime=origin)
                tr.stats.station = sta
                st_all.append(tr)

                # Count stations (once per station)
                if first_chan_for_sta:
                    stanum += 1
                    if stanum % 20 == 0:
                        print(f"Event {eve_id}: processed {stanum} stations...")
                    first_chan_for_sta = False

        # ---- Sort by distance ----
        st_all.sort(keys=["dist_km"])
        if not st_all:
            print(f"No traces found for event {eve_id}, skip.")
            continue

        ref_deg = float(st_all[0].stats.dist_deg)
        print(f"    Epicentral distance ≈ {st_all[0].stats.dist_km:.1f} km")

        # ---- Theoretical travel times (nearest station) ----
        tts = model.get_travel_times(
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
                p_arrival_time = origin + p_traveltime
            if tt.phase.name.upper() == "S":
                s_traveltime = float(tt.time)
                s_arrival_time = origin + s_traveltime

        if align_phase == "P" and p_traveltime is not None:
            phase_traveltime = float(p_traveltime)  # seconds since origin
        elif align_phase == "S" and s_traveltime is not None:
            phase_traveltime = float(s_traveltime)
        else:
            print("    No valid phase for alignment. Skip array.")
            continue

        # ---- Keep traces that cover [origin + start_time, origin + end_time] ----
        st_window = Stream()
        kept = 0

        for tr in st_all:
            # Keep the entire [0, end_time] record as long as it covers the plot window
            if tr.stats.endtime >= origin + start_time and tr.stats.starttime <= origin + end_time:
                tr_i = tr.copy()
                if tr_i is not None and tr_i.stats.npts > 0:
                    tr_i.stats.station = tr.stats.station
                    st_window.append(tr_i)
                    kept += 1

        if kept == 0:
            print("  No data in plot window (start_time to end_time).")
            continue

        # Label to show on the figure
        plot_comp = sel_comp

        # ---- Rotate horizontal components to R/T ----
        if sel_comp in ("R", "T"):
            print("Rotating horizontal components (N/E) to R/T ...")

            # In this dataset: DP1 is treated as N-like; DP2 is treated as E-like
            stN = st_window.select(channel="DP1")
            stE = st_window.select(channel="DP2")
            rotated_traces = []

            for trN in stN:
                sid = str(trN.stats.station)
                stE_match = stE.select(station=sid)
                if len(stE_match) == 0:
                    continue
                trE = stE_match[0]

                # Back-azimuth (station -> event), plus instrument orientation correction (11°)
                slat, slon = name2ll[sid]
                _, _, baz_geo = gps2dist_azimuth(eve_lat, eve_lon, slat, slon)
                baz = baz_geo - 11.0

                # Synchronize length and rotate
                npts_rot = min(trN.stats.npts, trE.stats.npts)
                n = trN.data[:npts_rot]
                e = trE.data[:npts_rot]
                r, t = rotate_ne_rt(n, e, baz)

                if sel_comp == "R":
                    trR = trN.copy()
                    trR.data = r
                    trR.stats.channel = trN.stats.channel[:-1] + "R"
                    rotated_traces.append(trR)
                    plot_comp = "R"
                elif sel_comp == "T":
                    trT = trN.copy()
                    trT.data = t
                    trT.stats.channel = trN.stats.channel[:-1] + "T"
                    rotated_traces.append(trT)
                    plot_comp = "T"

            st_comp = Stream(traces=rotated_traces)
        else:
            st_comp = st_window.select(channel=channel)

        # ---- Auto-select reference station: nearest trace by epicentral distance ----
        st_comp.sort(keys=["dist_km"])
        if len(st_comp) == 0:
            continue

        ref_station_id = str(st_comp[0].stats.station)
        print(
            f"    Reference station (auto): {ref_station_id} (nearest)  "
            f"dist_km={st_comp[0].stats.dist_km:.2f}"
        )

        num_traces = len(st_comp)
        print(f"    {num_traces} traces on {plot_comp}")
        if num_traces == 0:
            continue

        # ---- Preprocess traces (detrend/taper/filter) and normalize per trace ----
        for tr in st_comp:
            tr.detrend(type="demean")
            tr.taper(max_percentage=0.05, type="cosine")
            tr.filter(
                "bandpass",
                freqmin=min_freq,
                freqmax=max_freq,
                corners=4,
                zerophase=True,
            )
            mx = np.max(np.abs(tr.data))
            if mx > 0:
                tr.data = tr.data / mx

        # ---- Common length / sampling rate ----
        npts = min(tr.stats.npts for tr in st_comp)
        sample_rate = float(st_comp[0].stats.sampling_rate)
        ref = st_comp[0].data[:npts]
        move_limit_samples = int(round(move_limit_sec * sample_rate))
        print(f"    sample_rate = {sample_rate:.1f} Hz")

        # ---- Correlation window indices ----
        t0 = float(phase_traveltime) if phase_traveltime is not None else 0.0

        # Correlation window centered at (theoretical arrival)
        center_time = t0
        win_start = int(max(0, sample_rate * (center_time - short_win_pre)))
        win_end = int(min(npts, sample_rate * (center_time + short_win_post)))

        # ---- Predicted time offsets (per-station) ----
        phase_time_cache = {ref_deg: t0}

        def get_phase_time(dist_deg: float):
            if dist_deg in phase_time_cache:
                return phase_time_cache[dist_deg]
            tts_sta = model.get_travel_times(
                source_depth_in_km=event_depth,
                distance_in_degree=dist_deg,
                phase_list=["p", "P", "s", "S"],
            )
            t_val = None
            for tt in reversed(tts_sta):
                if align_phase.upper() == "P" and tt.phase.name.upper() == "P":
                    t_val = float(tt.time)
                if align_phase.upper() == "S" and tt.phase.name.upper() == "S":
                    t_val = float(tt.time)
            phase_time_cache[dist_deg] = t_val
            return t_val

        def get_offset_samples(dist_deg: float) -> int:
            t_sta = get_phase_time(dist_deg)
            if t_sta is None or t0 is None:
                return 0
            return int(round((t_sta - t0) * sample_rate))

        # ===================== Stage 1: align to reference -> aligned_stack =====================
        aligned_stack = np.zeros(npts)
        for tr in st_comp:
            d = tr.data[:npts]
            offset_samples = get_offset_samples(float(tr.stats.dist_deg))
            lag0 = 0
            rolled = shift_left_zeropad(d, lag0)
            lag1 = lag0 + compute_lag(ref, rolled, win_start, win_end, offset_samples)
            aligned_stack += shift_left_zeropad(d, lag1)

        mx = np.max(np.abs(aligned_stack))
        if mx > 0:
            aligned_stack = aligned_stack / mx

        # ===================== Stage 2: align to aligned_stack -> select traces =====================
        selected_aligned_stack = np.zeros(npts)
        selected_ids = set()
        station_corr = {}

        for tr in st_comp:
            d = tr.data[:npts]
            station_id = str(tr.stats.station)
            offset_samples = get_offset_samples(float(tr.stats.dist_deg))

            lag0 = 0
            rolled = shift_left_zeropad(d, lag0)
            lag2 = lag0 + compute_lag(aligned_stack, rolled, win_start, win_end, offset_samples)

            aligned_data = shift_left_zeropad(d, lag2)
            ref_window = aligned_stack[win_start:win_end]
            aligned_window = aligned_data[win_start:win_end]

            if np.linalg.norm(aligned_window) == 0 or np.linalg.norm(ref_window) == 0:
                r_window = 0.0
            else:
                r_window = float(
                    np.dot(aligned_window, ref_window)
                    / (np.linalg.norm(aligned_window) * np.linalg.norm(ref_window))
                )

            if np.linalg.norm(aligned_data) == 0 or np.linalg.norm(aligned_stack) == 0:
                r_whole = 0.0
            else:
                r_whole = float(
                    np.dot(aligned_data, aligned_stack)
                    / (np.linalg.norm(aligned_data) * np.linalg.norm(aligned_stack))
                )

            station_corr[station_id] = r_whole

            if r_window >= r_window_min and r_whole >= r_whole_min:
                selected_aligned_stack += aligned_data
                selected_ids.add(station_id)
            else:
                print(f"    Rejected {station_id}: r_win={r_window:.2f}, r_all={r_whole:.2f}")

        mx = np.max(np.abs(selected_aligned_stack))
        if mx > 0:
            selected_aligned_stack = selected_aligned_stack / mx

        # ===================== Final alignment for plotting (lag3 to reference) =====================
        selected_rows = []  # (dist_km, station_id, y_aligned_norm)
        rejected_rows = []

        aligned_bank = []
        aligned_bank_all = []
        station_shifts = {}  # Track shifts for each station
        aligned_traces_by_station = {}  # station_id -> final aligned & normalized trace (for later stacking)

        for tr in st_comp:
            x = tr.data[:npts]
            station_id = str(tr.stats.station)

            # Final alignment: align every trace to the reference station so the x-axis remains
            # the reference station's time since origin.
            if station_id == ref_station_id:
                lag3 = 0
                y = x.copy()
            else:
                offset_samples = get_offset_samples(float(tr.stats.dist_deg))
                lag3 = compute_lag(ref, x, win_start, win_end, offset_samples)
                y = shift_left_zeropad(x, lag3)

            # Store shift (in samples and seconds)
            station_shifts[station_id] = {
                'lag_samples': lag3,
                'lag_seconds': lag3 / sample_rate
            }

            # Per-trace normalization for plotting/stacking
            my = np.max(np.abs(y)) or 1.0
            y = y / my

            # Store for later stacking by station
            aligned_traces_by_station[station_id] = y.copy()

            aligned_bank_all.append(y)

            # Epicentral distance (km)
            slat, slon = name2ll[station_id]
            dist_m, _, _ = gps2dist_azimuth(eve_lat, eve_lon, slat, slon)
            dist_km = dist_m / 1000.0

            if station_id in selected_ids:
                selected_rows.append((dist_km, station_id, y))
                aligned_bank.append(y)
            else:
                rejected_rows.append((dist_km, station_id, y))

        selected_rows.sort(key=lambda t: t[0])
        rejected_rows.sort(key=lambda t: t[0])
        print(
            f"    Final lag3 traces: selected={len(selected_rows)}, "
            f"rejected={len(rejected_rows)}, total={len(selected_rows) + len(rejected_rows)}"
        )

        # ---- Time axis (seconds since origin) ----
        t_abs = np.arange(npts) / sample_rate
        mask = (t_abs >= start_time) & (t_abs <= end_time)

        # ---- Final stack (normalized) ----
        bank = aligned_bank_all if stack_all_traces_for_final else aligned_bank
        if len(bank) > 0:
            stack_vec = np.mean(np.vstack(bank), axis=0)
            ms = np.max(np.abs(stack_vec)) or 1.0
            stack_vec = stack_vec / ms
        else:
            stack_vec = np.zeros_like(t_abs)

        # ---- Plot: record section (top) + stack (bottom) ----
        fig, (ax, ax2) = plt.subplots(
            2,
            1,
            figsize=(10, 9),
            sharex=False,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        all_rows = selected_rows + rejected_rows
        all_rows.sort(key=lambda t: t[0])

        t_masked = t_abs[mask]
        if len(all_rows) > 0 and np.any(mask):
            A = np.vstack([row[2][mask] for row in all_rows])
            dvec = np.array([row[0] for row in all_rows], dtype=float)

            # y-edges for irregular station spacing
            if len(dvec) == 1:
                y_edges = np.array([dvec[0] - 0.5, dvec[0] + 0.5])
            else:
                mids = 0.5 * (dvec[1:] + dvec[:-1])
                y_edges = np.empty(len(dvec) + 1)
                y_edges[1:-1] = mids
                y_edges[0] = dvec[0] - (mids[0] - dvec[0])
                y_edges[-1] = dvec[-1] + (dvec[-1] - mids[-1])

            # t-edges for pcolormesh
            if len(t_masked) == 1:
                t_edges = np.array(
                    [t_masked[0] - 0.5 / sample_rate, t_masked[0] + 0.5 / sample_rate]
                )
            else:
                tmids = 0.5 * (t_masked[1:] + t_masked[:-1])
                t_edges = np.empty(len(t_masked) + 1)
                t_edges[1:-1] = tmids
                t_edges[0] = t_masked[0] - (tmids[0] - t_masked[0])
                t_edges[-1] = t_masked[-1] + (t_masked[-1] - tmids[-1])

            ax.pcolormesh(
                t_edges,
                y_edges,
                A,
                cmap="gray",
                shading="auto",
                vmin=-1.0,
                vmax=1.0,
            )

        ax.set_xlim(start_time, end_time)
        ax.set_xlabel("Time since origin (s)")
        ax.set_ylabel("Epicentral distance (km)")
        ax.set_title(f"Aligned {align_phase} waveforms Event {eve_id} comp = {plot_comp}")
        ax.grid(alpha=0.2)

        # Theoretical arrival time (nearest station) as a vertical reference line
        try:
            if align_phase.upper() == "P" and p_traveltime is not None:
                t_ref = float(p_traveltime)
            elif align_phase.upper() == "S" and s_traveltime is not None:
                t_ref = float(s_traveltime)
            else:
                t_ref = None

            if t_ref is not None:
                for axi in (ax, ax2):
                    axi.axvline(x=t_ref, color="k", lw=3, alpha=0.5, zorder=6)
        except Exception as e:
            print(f"    [WARN] Failed to draw vertical reference arrival for {align_phase}: {e}")

        # Cross-correlation window bounds as vertical yellow lines
        try:
            t_win_start = win_start / sample_rate
            t_win_end = win_end / sample_rate
            for axi in (ax, ax2):
                axi.axvline(x=t_win_start, color="y", lw=2, alpha=0.9, zorder=7)
                axi.axvline(x=t_win_end, color="y", lw=2, alpha=0.9, zorder=7)
        except Exception as e:
            print(f"    [WARN] Failed to draw correlation window bounds: {e}")

        # Cross-correlation search limits (green): yellow window expanded by move_limit_sec
        try:
            t_explore_start = t_win_start - move_limit_sec
            t_explore_end = t_win_end + move_limit_sec
            t_explore_start = max(0.0, t_explore_start)
            t_explore_end = min(npts / sample_rate, t_explore_end)
            for axi in (ax, ax2):
                axi.axvline(x=t_explore_start, color="g", lw=2, alpha=0.9, zorder=7)
                axi.axvline(x=t_explore_end, color="g", lw=2, alpha=0.9, zorder=7)
        except Exception as e:
            print(f"    [WARN] Failed to draw correlation search limits: {e}")

        # Legend for window bounds
        try:
            legend_handles = [
                Line2D([0], [0], color='y', lw=2, label='Correlation window'),
                Line2D([0], [0], color='g', lw=2, label='Correlation search (±move_limit_sec)'),
            ]
            ax.legend(
                handles=legend_handles,
                loc='upper left',
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=9,
            )
        except Exception as e:
            print(f"    [WARN] Failed to add legend: {e}")

        # Bottom panel: normalized stack
        ax2.plot(t_abs[mask], stack_vec[mask], color="C3", lw=1.5)
        ax2.axhline(0.0, color="k", lw=0.6)
        ax2.set_xlim(start_time, end_time)
        ax2.set_xlabel("Time since origin (s)")
        ax2.set_ylabel("Stack (norm.)")
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_title(
            "Final stack uses ALL traces (no screening)"
            if stack_all_traces_for_final
            else "Final stack uses SELECTED traces only"
        )
        ax2.grid(alpha=0.2)

        plt.tight_layout()

        # Store data for three-component plotting or show individual plot
        if process_as_three_comp:
            # Determine component name
            if channel == "DPZ":
                comp_key = "DPZ"
            elif sel_comp == "R":
                comp_key = "R"
            elif sel_comp == "T":
                comp_key = "T"
            else:
                comp_key = channel
            
            # Store data
            all_component_data[comp_key] = {
                'fig': fig,
                'all_rows': [(r[0], r[1], r[2].copy()) for r in (selected_rows + rejected_rows)],
                'stack_vec': stack_vec.copy(),
                't_abs': t_abs.copy(),
                'mask': mask.copy(),
                'sample_rate': sample_rate,
                'win_start': win_start,
                'win_end': win_end,
                'move_limit_sec': move_limit_sec,
                'move_limit_samples': move_limit_samples,
                'npts': npts,
                'start_time': start_time,
                'end_time': end_time,
                'eve_id': eve_id,
                'align_phase': align_phase,
                'station_shifts': station_shifts.copy(),
                'station_corr': station_corr.copy(),
                # Stations that passed Stage-2 screening for this component
                'selected_ids': sorted(list(selected_ids), key=lambda s: int(s)),
                'aligned_traces_by_station': {k: v.copy() for k, v in aligned_traces_by_station.items()},
            }
            # Store t_ref
            try:
                if align_phase.upper() == "P" and p_traveltime is not None:
                    all_component_data[comp_key]['t_ref'] = float(p_traveltime)
                elif align_phase.upper() == "S" and s_traveltime is not None:
                    all_component_data[comp_key]['t_ref'] = float(s_traveltime)
                else:
                    all_component_data[comp_key]['t_ref'] = None
            except:
                all_component_data[comp_key]['t_ref'] = None
            
            plt.close(fig)  # Close individual figure
        else:
            # Show individual plot in non-three-component mode
            # Save figure (same location/pattern as the original script)
            save_path = Path(
                "/Users/vidale/Documents/Research/Mingze_SJF/output"
            )
            save_dir = save_path / eve_id
            #save_dir.mkdir(parents=True, exist_ok=True)
            save_file = save_dir / f"{eve_id}_{plot_comp}_{align_phase}.png"
            #plt.savefig(save_file, dpi=300, bbox_inches="tight")

            # ===================== Z-only: stack compare using saved R–T screening list =====================
            if (component == 'Z') and show_rt_filter_compare_plot and use_saved_rt_filter_for_z_stack:
                rt_txt = save_dir / f"{eve_id}_rt_zero_diff_stations_S.txt"
                if not rt_txt.exists():
                    print(f"[WARN] Saved R–T station list not found: {rt_txt} (run all_channels=True first)")
                else:
                    try:
                        with open(rt_txt, 'r') as f:
                            keep_sta = [line.strip() for line in f if line.strip() and (not line.startswith('#'))]
                        keep_sta = [s for s in keep_sta if s in aligned_traces_by_station]

                        if len(keep_sta) == 0:
                            print("[WARN] R–T station list loaded but none found in this Z run for stacking.")
                        else:
                            bank_z = [aligned_traces_by_station[s] for s in keep_sta]
                            stack_z_filt = np.mean(np.vstack(bank_z), axis=0)
                            ms = np.max(np.abs(stack_z_filt)) or 1.0
                            stack_z_filt = stack_z_filt / ms

                            fig_zcmp, ax_zcmp = plt.subplots(1, 1, figsize=(10, 3.8))
                            ax_zcmp.plot(t_abs[mask], stack_vec[mask], color='k', lw=2, label='No R–T screening')
                            ax_zcmp.plot(t_abs[mask], stack_z_filt[mask], color='r', lw=2, label='With R–T screening (saved list)')
                            ax_zcmp.axhline(0.0, color='k', lw=0.6, alpha=0.6)
                            ax_zcmp.set_xlim(start_time, end_time)
                            ax_zcmp.set_ylim(-1.1, 1.1)
                            ax_zcmp.grid(alpha=0.2)
                            ax_zcmp.set_xlabel('Time since origin (s)')
                            ax_zcmp.set_ylabel('Stack (norm.)')
                            ax_zcmp.set_title(f"Event {eve_id} Z-stack compare | N_keep={len(keep_sta)}")
                            ax_zcmp.legend(loc='upper right', fontsize=9)
                            plt.tight_layout()

                            zcmp_file = save_dir / f"{eve_id}_Zstack_compare_savedRT_{align_phase}.png"
                            #plt.savefig(zcmp_file, dpi=300, bbox_inches='tight')
                            print(f"✓ Z-only stack comparison plot saved to: {zcmp_file}")
                            # plt.show()  # defer until end
                    except Exception as e:
                        print(f"[WARN] Failed to load/apply saved R–T station list for Z-only stacking: {e}")

            # Show the main Z figure (deferred until end)
            # plt.show()



# ===================== Three-component combined plotting =====================
if process_as_three_comp and len(all_component_data) == 3:
    print(f"\\n{'='*70}")
    print(f"Creating combined three-component plot...")
    print(f"{'='*70}\\n")
    
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.25)
    
    comp_order = ['DPZ', 'R', 'T']
    comp_titles = ['Vertical (Z)', 'Radial (R)', 'Transverse (T)']
    
    # Get common parameters
    first_data = all_component_data[comp_order[0]]
    eve_id = first_data['eve_id']
    align_phase = first_data['align_phase']
    start_time = first_data['start_time']
    end_time = first_data['end_time']

    # Pre-compute stations with zero R–T shift difference (for optional stacking)
    zero_rt_diff_stations = None
    if stack_only_zero_rt_diff and ('R' in all_component_data) and ('T' in all_component_data):
        r_shifts = all_component_data['R']['station_shifts']
        t_shifts = all_component_data['T']['station_shifts']
        common = set(r_shifts.keys()) & set(t_shifts.keys())

        # Optionally restrict to stations selected in BOTH R and T (same logic as RT stats)
        if not rt_stats_use_all_stations:
            r_sel = set(all_component_data['R'].get('selected_ids', []))
            t_sel = set(all_component_data['T'].get('selected_ids', []))
            common = common & r_sel & t_sel

        zero_rt_diff_stations = {
            sta for sta in common
            if np.isclose(
                float(r_shifts[sta]['lag_seconds']) - float(t_shifts[sta]['lag_seconds']),
                0.0,
                atol=1e-12,
            )
        }
        print(f"Zero R–T shift-difference stations: {len(zero_rt_diff_stations)}")
    
    for idx, comp_name in enumerate(comp_order):
        if comp_name not in all_component_data:
            print(f"Warning: {comp_name} data not found")
            continue

        data = all_component_data[comp_name]
        all_rows = data['all_rows']
        stack_vec = data['stack_vec']
        t_abs = data['t_abs']
        mask = data['mask']
        sample_rate = data['sample_rate']
        win_start = data['win_start']
        win_end = data['win_end']
        move_limit_sec = data['move_limit_sec']
        npts = data['npts']
        t_ref = data.get('t_ref')

        # Optionally rebuild the R/T stack using only stations with zero R–T shift difference
        if stack_only_zero_rt_diff and (zero_rt_diff_stations is not None) and (comp_name in ('R', 'T')):
            tr_map = data.get('aligned_traces_by_station', {})
            bank = [tr_map[sta] for sta in sorted(zero_rt_diff_stations, key=lambda s: int(s)) if sta in tr_map]
            if len(bank) > 0:
                stack_vec = np.mean(np.vstack(bank), axis=0)
                ms = np.max(np.abs(stack_vec)) or 1.0
                stack_vec = stack_vec / ms

        # Top panel: record section
        ax = fig.add_subplot(gs[0, idx])

        all_rows.sort(key=lambda t: t[0])
        t_masked = t_abs[mask]

        if len(all_rows) > 0 and np.any(mask):
            A = np.vstack([row[2][mask] for row in all_rows])
            dvec = np.array([row[0] for row in all_rows], dtype=float)

            # y-edges
            if len(dvec) == 1:
                y_edges = np.array([dvec[0] - 0.5, dvec[0] + 0.5])
            else:
                mids = 0.5 * (dvec[1:] + dvec[:-1])
                y_edges = np.empty(len(dvec) + 1)
                y_edges[1:-1] = mids
                y_edges[0] = dvec[0] - (mids[0] - dvec[0])
                y_edges[-1] = dvec[-1] + (dvec[-1] - mids[-1])

            # t-edges
            if len(t_masked) == 1:
                t_edges = np.array([t_masked[0] - 0.5 / sample_rate,
                                   t_masked[0] + 0.5 / sample_rate])
            else:
                tmids = 0.5 * (t_masked[1:] + t_masked[:-1])
                t_edges = np.empty(len(t_masked) + 1)
                t_edges[1:-1] = tmids
                t_edges[0] = t_masked[0] - (tmids[0] - t_masked[0])
                t_edges[-1] = t_masked[-1] + (t_masked[-1] - tmids[-1])

            ax.pcolormesh(t_edges, y_edges, A, cmap='gray',
                         shading='auto', vmin=-1.0, vmax=1.0)

        ax.set_xlim(start_time, end_time)
        if idx == 0:
            ax.set_ylabel('Epicentral distance (km)', fontsize=11)
        ax.set_title(f'{comp_titles[idx]}', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.2)

        # Vertical reference line
        if t_ref is not None:
            ax.axvline(x=t_ref, color='r', lw=2, alpha=0.6, linestyle='--', zorder=6)
        # Cross-correlation window bounds
        try:
            t_win_start = win_start / sample_rate
            t_win_end = win_end / sample_rate
            ax.axvline(x=t_win_start, color='y', lw=2, alpha=0.9, zorder=7)
            ax.axvline(x=t_win_end, color='y', lw=2, alpha=0.9, zorder=7)
        except Exception as e:
            print(f"[WARN] Failed to draw correlation window bounds (top {comp_name}): {e}")

        # Cross-correlation search limits (green): yellow window expanded by move_limit_sec
        try:
            t_explore_start = t_win_start - move_limit_sec
            t_explore_end = t_win_end + move_limit_sec
            t_explore_start = max(0.0, t_explore_start)
            t_explore_end = min(npts / sample_rate, t_explore_end)
            ax.axvline(x=t_explore_start, color='g', lw=2, alpha=0.9, zorder=7)
            ax.axvline(x=t_explore_end, color='g', lw=2, alpha=0.9, zorder=7)
        except Exception as e:
            print(f"[WARN] Failed to draw correlation search limits (top {comp_name}): {e}")

        # Legend for window bounds (only once, top-left panel)
        if idx == 0:
            try:
                legend_handles = [
                    Line2D([0], [0], color='y', lw=2, label='Correlation window'),
                    Line2D([0], [0], color='g', lw=2, label='Correlation search (±move_limit_sec)'),
                ]
                ax.legend(
                    handles=legend_handles,
                    loc='upper left',
                    bbox_to_anchor=(1.02, 1.0),
                    borderaxespad=0.0,
                    fontsize=9,
                )
            except Exception as e:
                print(f"[WARN] Failed to add legend (top {comp_name}): {e}")

        # Bottom panel: stack
        ax2 = fig.add_subplot(gs[1, idx])
        ax2.plot(t_abs[mask], stack_vec[mask], color='C3', lw=2)
        ax2.axhline(0.0, color='k', lw=0.6)
        ax2.set_xlim(start_time, end_time)
        ax2.set_xlabel('Time since origin (s)', fontsize=11)
        if idx == 0:
            ax2.set_ylabel('Stack (norm.)', fontsize=11)
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(alpha=0.2)

        if stack_only_zero_rt_diff and (zero_rt_diff_stations is not None) and (comp_name in ('R', 'T')):
            ax2.set_title(f"Stack: zero R–T diff stations (N={len(zero_rt_diff_stations)})", fontsize=10)

        if t_ref is not None:
            ax2.axvline(x=t_ref, color='r', lw=2, alpha=0.6, linestyle='--', zorder=6)
        # Cross-correlation window bounds
        try:
            t_win_start = win_start / sample_rate
            t_win_end = win_end / sample_rate
            ax2.axvline(x=t_win_start, color='y', lw=2, alpha=0.9, zorder=7)
            ax2.axvline(x=t_win_end, color='y', lw=2, alpha=0.9, zorder=7)
        except Exception as e:
            print(f"[WARN] Failed to draw correlation window bounds (bottom {comp_name}): {e}")

        # Cross-correlation search limits (green): yellow window expanded by move_limit_sec
        try:
            t_explore_start = t_win_start - move_limit_sec
            t_explore_end = t_win_end + move_limit_sec
            t_explore_start = max(0.0, t_explore_start)
            t_explore_end = min(npts / sample_rate, t_explore_end)
            ax2.axvline(x=t_explore_start, color='g', lw=2, alpha=0.9, zorder=7)
            ax2.axvline(x=t_explore_end, color='g', lw=2, alpha=0.9, zorder=7)
        except Exception as e:
            print(f"[WARN] Failed to draw correlation search limits (bottom {comp_name}): {e}")

    fig.suptitle(f'Event {eve_id} - Aligned {align_phase} waveforms (3 components)', 
                fontsize=14, fontweight='bold')
    
    # Save combined figure
    save_path = Path("/Users/vidale/Documents/Research/Mingze_SJF/output")
    save_dir = save_path / eve_id
    save_dir.mkdir(parents=True, exist_ok=True)
    save_file = save_dir / f"{eve_id}_3comp_{align_phase}.png"
    #plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Three-component plot saved to: {save_file}")
    print(f"\n✓ Three-component plot created successfully!\n")    
    # plt.show()  # defer until end

    # ===================== Save R–T zero-diff station list (for reuse in Z-only mode) =====================
    zero_rt_diff_stations_saved = None
    if save_rt_filter_station_list and ('R' in all_component_data) and ('T' in all_component_data):
        r_shifts = all_component_data['R']['station_shifts']
        t_shifts = all_component_data['T']['station_shifts']
        common = set(r_shifts.keys()) & set(t_shifts.keys())

        # Optionally restrict to stations selected in BOTH R and T (same logic as RT stats)
        if not rt_stats_use_all_stations:
            r_sel = set(all_component_data['R'].get('selected_ids', []))
            t_sel = set(all_component_data['T'].get('selected_ids', []))
            common = common & r_sel & t_sel

        zero_rt_diff_stations_saved = {
            sta for sta in common
            if np.isclose(
                float(r_shifts[sta]['lag_seconds']) - float(t_shifts[sta]['lag_seconds']),
                0.0,
                atol=1e-12,
            )
        }

        rt_txt = save_dir / f"{eve_id}_rt_zero_diff_stations_{align_phase}.txt"
        try:
            with open(rt_txt, 'w') as f:
                f.write(f"# Event: {eve_id}\n")
                f.write(f"# align_phase: {align_phase}\n")
                f.write(f"# criterion: (R lag_seconds - T lag_seconds) == 0\n")
                f.write(f"# rt_stats_use_all_stations: {rt_stats_use_all_stations}\n")
                f.write(f"# N_zero: {0 if zero_rt_diff_stations_saved is None else len(zero_rt_diff_stations_saved)}\n")
                for sta in sorted(zero_rt_diff_stations_saved, key=lambda s: int(s)):
                    f.write(f"{sta}\n")
            print(f"✓ Saved R–T zero-diff station list to: {rt_txt}")
        except Exception as e:
            print(f"[WARN] Failed to write R–T station list: {e}")

    # ===================== Stack compare plot: with vs without R–T screening =====================
    if show_rt_filter_compare_plot:
        print("Creating stack comparison plot (before vs after R–T screening)...")

        # We use the same zero_rt_diff_stations logic as above:
        #   - If stack_only_zero_rt_diff is False, we still compute zero_rt_diff_stations here
        #     (so this compare plot is always meaningful when R and T are available).
        zero_rt_for_compare = None
        if ('R' in all_component_data) and ('T' in all_component_data):
            r_shifts = all_component_data['R']['station_shifts']
            t_shifts = all_component_data['T']['station_shifts']
            common = set(r_shifts.keys()) & set(t_shifts.keys())

            # Optionally restrict to stations selected in BOTH R and T (same as RT stats)
            if not rt_stats_use_all_stations:
                r_sel = set(all_component_data['R'].get('selected_ids', []))
                t_sel = set(all_component_data['T'].get('selected_ids', []))
                common = common & r_sel & t_sel

            zero_rt_for_compare = {
                sta for sta in common
                if np.isclose(
                    float(r_shifts[sta]['lag_seconds']) - float(t_shifts[sta]['lag_seconds']),
                    0.0,
                    atol=1e-12,
                )
            }

        # Figure layout: 3 rows, 1 column (Z / R / T) — vertical arrangement
        fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(9, 12), sharex=True, sharey=True)
        comp_order = ['DPZ', 'R', 'T']
        comp_titles_cmp = ['Z stack', 'R stack', 'T stack']

        for j, comp_name in enumerate(comp_order):
            axc = axes_cmp[j]
            if comp_name not in all_component_data:
                axc.set_axis_off()
                continue

            data = all_component_data[comp_name]
            t_abs = data['t_abs']
            mask = data['mask']
            start_time = data['start_time']
            end_time = data['end_time']

            # Black: baseline stack (whatever your current stacking rule produced)
            stack_black = data['stack_vec']

            # Red: R–T screened stack (only stations with zero R–T shift difference)
            stack_red = stack_black
            if zero_rt_for_compare is not None and len(zero_rt_for_compare) > 0:
                tr_map = data.get('aligned_traces_by_station', {})
                bank = [
                    tr_map[sta]
                    for sta in sorted(zero_rt_for_compare, key=lambda s: int(s))
                    if sta in tr_map
                ]
                if len(bank) > 0:
                    stack_red = np.mean(np.vstack(bank), axis=0)
                    ms = np.max(np.abs(stack_red)) or 1.0
                    stack_red = stack_red / ms

            axc.plot(t_abs[mask], stack_black[mask], color='k', lw=2, label='No R–T screening')
            axc.plot(t_abs[mask], stack_red[mask], color='r', lw=2, label='With R–T screening')
            axc.axhline(0.0, color='k', lw=0.6, alpha=0.6)
            axc.set_xlim(start_time, end_time)
            axc.set_ylim(-1.1, 1.1)
            axc.grid(alpha=0.2)
            axc.set_title(comp_titles_cmp[j], fontsize=12, fontweight='bold')
            axc.set_xlabel('Time since origin (s)', fontsize=11)
            if j != 2:
                axc.set_xlabel('')
            if j == 0:
                axc.set_ylabel('Stack (norm.)', fontsize=11)
            if j == 0:
                axc.legend(loc='upper right', fontsize=9)

        n_zero = 0 if zero_rt_for_compare is None else len(zero_rt_for_compare)
        fig_cmp.suptitle(
            f'Event {eve_id} - Stack compare (black: no R–T screening; red: with R–T screening) | N_zero={n_zero}',
            fontsize=13,
            fontweight='bold'
        )
        plt.tight_layout()

        # Save comparison figure
        cmp_file = save_dir / f"{eve_id}_rtfilter_stack_compare_{align_phase}.png"
        #plt.savefig(cmp_file, dpi=300, bbox_inches='tight')
        print(f"✓ Stack comparison plot saved to: {cmp_file}")
        # plt.show()

    # ===================== Shift comparison plot: Radial vs Transverse =====================
    if show_diff_rt_plot:
        if 'R' in all_component_data and 'T' in all_component_data:
            print("Creating shift comparison plot (Radial vs Transverse)...")
            print(
                "Shift comparison parameters: "
                f"align_phase={align_phase}, start_time={start_time}, end_time={end_time}, "
                f"short_win_pre={short_win_pre}, short_win_post={short_win_post}, "
                f"move_limit_sec={move_limit_sec}"
            )
            
            r_shifts = all_component_data['R']['station_shifts']
            t_shifts = all_component_data['T']['station_shifts']
            r_corr = all_component_data['R']['station_corr']
            t_corr = all_component_data['T']['station_corr']
            
            # Find common stations
            common_stations = set(r_shifts.keys()) & set(t_shifts.keys())
            common_corr_stations = set(r_corr.keys()) & set(t_corr.keys())

            # Optionally restrict to stations selected in BOTH R and T
            if not rt_stats_use_all_stations:
                r_sel = set(all_component_data['R'].get('selected_ids', []))
                t_sel = set(all_component_data['T'].get('selected_ids', []))
                common_stations = common_stations & r_sel & t_sel
                common_corr_stations = common_corr_stations & r_sel & t_sel
            
            if len(common_stations) > 0:
                # Extract shifts in seconds
                stations = sorted(common_stations, key=lambda s: int(s))
                r_lags = [r_shifts[sta]['lag_seconds'] for sta in stations]
                t_lags = [t_shifts[sta]['lag_seconds'] for sta in stations]
                station_nums = [int(sta) for sta in stations]
                
                # Create comparison figure
                fig_shift, axes = plt.subplots(2, 3, figsize=(18, 10))
                (ax1, ax2, ax5), (ax3, ax4, ax6) = axes
                
                # Panel 1: Scatter plot R vs T
                ax1.scatter(r_lags, t_lags, alpha=0.5, s=20)
                ax1.plot([min(r_lags + t_lags), max(r_lags + t_lags)], 
                        [min(r_lags + t_lags), max(r_lags + t_lags)], 
                        'r--', alpha=0.5, label='1:1 line')
                ax1.set_xlabel('Radial shift (seconds)', fontsize=11)
                ax1.set_ylabel('Transverse shift (seconds)', fontsize=11)
                ax1.set_title('Radial vs Transverse Shifts', fontsize=12, fontweight='bold')
                ax1.grid(alpha=0.3)
                ax1.legend()
                ax1.set_aspect('equal', adjustable='box')
                
                # Panel 2: Difference histogram
                diff_lags = np.array(r_lags) - np.array(t_lags)
                # Fraction of stations with zero R–T shift difference
                # (shifts are derived from integer-sample lags; use tiny atol for float safety)
                zero_diff_frac = float(np.mean(np.isclose(diff_lags, 0.0, atol=1e-12)))
                ax2.hist(diff_lags, bins=30, alpha=0.7, edgecolor='black')
                ax2.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero difference')
                ax2.axvline(np.median(diff_lags), color='g', linestyle='--', linewidth=2, 
                        label=f'Median = {np.median(diff_lags):.3f}s')
                ax2.set_xlabel('R shift - T shift (seconds)', fontsize=11)
                ax2.set_ylabel('Count', fontsize=11)
                ax2.set_title('Shift Difference Distribution', fontsize=12, fontweight='bold')
                ax2.legend()
                ax2.grid(alpha=0.3)

                # Panel 3: Max correlation R vs T
                if len(common_corr_stations) > 0:
                    corr_stations = sorted(common_corr_stations, key=lambda s: int(s))
                    r_corr_vals = [r_corr[sta] for sta in corr_stations]
                    t_corr_vals = [t_corr[sta] for sta in corr_stations]
                    ax5.scatter(r_corr_vals, t_corr_vals, alpha=0.5, s=20)
                    ax5.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='1:1 line')
                    ax5.set_xlabel('Radial max corr', fontsize=11)
                    ax5.set_ylabel('Transverse max corr', fontsize=11)
                    ax5.set_title('Max Correlation: R vs T', fontsize=12, fontweight='bold')
                    ax5.grid(alpha=0.3)
                    ax5.legend()
                    ax5.set_aspect('equal', adjustable='box')
                else:
                    ax5.text(0.5, 0.5, 'No common corr stations', ha='center', va='center')
                    ax5.set_axis_off()
                
                # Panel 3: Shifts vs station number
                ax3.plot(station_nums, r_lags, 'o-', label='Radial', alpha=0.7, markersize=4)
                ax3.plot(station_nums, t_lags, 's-', label='Transverse', alpha=0.7, markersize=4)
                ax3.set_xlabel('Station number', fontsize=11)
                ax3.set_ylabel('Shift (seconds)', fontsize=11)
                ax3.set_title('Shifts vs Station', fontsize=12, fontweight='bold')
                ax3.legend()
                ax3.grid(alpha=0.3)
                
                # Panel 4: Statistics
                ax4.axis('off')
                stats_text = f"""Shift Comparison Statistics
                
    Number of stations: {len(common_stations)}
    RT statistics mode: {'ALL stations' if rt_stats_use_all_stations else 'SELECTED stations only'}

    Radial shifts:
    Mean: {np.mean(r_lags):.4f} s
    Std: {np.std(r_lags):.4f} s
    Range: [{np.min(r_lags):.4f}, {np.max(r_lags):.4f}] s

    Transverse shifts:
    Mean: {np.mean(t_lags):.4f} s
    Std: {np.std(t_lags):.4f} s
    Range: [{np.min(t_lags):.4f}, {np.max(t_lags):.4f}] s

    Difference (R - T):
        Mean: {np.mean(diff_lags):.4f} s
        Median: {np.median(diff_lags):.4f} s
        Std: {np.std(diff_lags):.4f} s
        Zero-difference fraction: {zero_diff_frac*100:.1f}%
    
    Frequency content (bandpass):
        {min_freq:.2f}–{max_freq:.2f} Hz"""
                stats_text += (
                    f"\n\nParameters:\n"
                    f"  align_phase: {align_phase}\n"
                    f"  start_time: {start_time}\n"
                    f"  end_time: {end_time}\n"
                    f"  short_win_pre: {short_win_pre}\n"
                    f"  short_win_post: {short_win_post}\n"
                    f"  move_limit_sec: {move_limit_sec}"
                )
                ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                        verticalalignment='center')

                # Panel 6: leave blank
                ax6.axis('off')
                
                fig_shift.suptitle(f'Event {eve_id} - Shift & Correlation Comparison', 
                                fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Save shift comparison plot
                shift_file = save_dir / f"{eve_id}_shift_comparison_{align_phase}.png"
                plt.savefig(shift_file, dpi=300, bbox_inches='tight')
                print(f"✓ Shift comparison plot saved to: {shift_file}")
            else:
                print("Warning: No common stations found between R and T components")

        # Show all figures together (three-component + shift comparison)
        # plt.show()

    # ===================== Show all figures together at the end =====================
    plt.show()