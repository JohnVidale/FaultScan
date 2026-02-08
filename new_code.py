import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import timedelta, timezone

from obspy import read, UTCDateTime, Stream
from obspy.geodetics import degrees2kilometers, locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel
from obspy.signal.rotate import rotate_ne_rt
from scipy.signal import hilbert
from scipy.signal.windows import gaussian

min_freq, max_freq            = 1.0, 5.0 # Bandpass filter (Hz)
start_time, end_time          = -1990.0, 3690.0 # Plotting time window (seconds since origin)
win_pre, win_post             = 0.5,  0.5 # Correlation window parameters (seconds)
r_window_min                  = 0.7       # Minimum correlation coefficient for trace selection
move_limit_sec                = 0.05      # Maximum allowed shift (seconds) searched in compute_lag

# Run modes
all_channels = True  # If True to process all channels
component   = "R"       # Component selection: 'Z', 'R', or 'T'
align_phase = "S"       # Alignment phase 'P' or 'S'

event       = "CI_40353544" # Single run selection (used when the corresponding "all_*" is False)
events = [event]        # Allows for future modification to process multiple events

# Paths
info_root = Path("/Users/vidale/Documents/Research/Mingze_SJF/20220930_events_cut/event_sta_info")
data_path = Path("/Users/vidale/Documents/Research/Mingze_SJF/20220930_events_cut/07_1hour_20220930")

# plotting / stacking options
stack_all_traces_for_final  = True  # Final stack mode: True = use ALL traces after final lag3; False = use only selected traces
show_diff_rt_plot           = True  # Show R T differences in the combined plot
show_rt_filter_compare_plot = True  # Compare stacks with/without correlation screening
show_est_calc_shift_plot    = True  # Compare estimated shifts vs theoretical (calculated) shifts
show_individual_seismograms = False  # Plot individual seismograms (20 traces per subplot, 5 panels per figure)
show_snippet_compare_plot   = True  # Plot correlation-window snippet comparison (pass vs fail)
show_stack_stage_plot       = True  # Plot overlay of Stage1/Stage2/Final stacks
show_aligned_phase_plot     = True  # Show aligned phase record-section + stack plot
show_three_comp_plot        = True  # Show 3-component record-section + stack plot

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
        # Process a single R/T component, but read both horizontals internally for rotation
        channels = ["DP1"]
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

                # Read the component and slice [origin + start_time, origin + end_time]
                tr = read(str(fpath))[0]
                tr = tr.slice(starttime=origin + start_time, endtime=origin + end_time)
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
        # Theoretical arrival time (nearest station) used for reference in shift calculations/plots
        t_ref = phase_traveltime

        # ---- Keep traces that cover [origin + start_time, origin + end_time] ----
        st_window = Stream()
        kept = 0

        for tr in st_all:
            # Keep the entire [start_time, end_time] record as long as it covers the plot window
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
        ref_trace_dur = float(st_comp[0].stats.npts) / float(st_comp[0].stats.sampling_rate)
        print(
            f"    Reference station (auto): {ref_station_id} (nearest)  "
            f"dist_km={st_comp[0].stats.dist_km:.2f}  dur_s={ref_trace_dur:.2f}"
        )

        num_traces = len(st_comp)
        print(f"    {num_traces} traces on {plot_comp}")
        if num_traces == 0:
            continue

        # ---- Preprocess traces (detrend/taper/filter) ----
        for tr in st_comp:
            tr.detrend(type="demean")
            trace_len_sec = float(tr.stats.npts) / float(tr.stats.sampling_rate)
            taper_pct = min(0.05, 5.0 / trace_len_sec) if trace_len_sec > 0 else 0.0
            tr.taper(max_percentage=taper_pct, type="cosine")
            tr.filter(
                "bandpass",
                freqmin=min_freq,
                freqmax=max_freq,
                corners=4,
                zerophase=True,
            )

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
        win_start = int(max(0, sample_rate * ((center_time - start_time) - win_pre)))
        win_end = int(min(npts, sample_rate * ((center_time - start_time) + win_post)))

        # ---- Normalize per trace using only the correlation window ----
        for tr in st_comp:
            win = tr.data[win_start:win_end]
            mx = np.max(np.abs(win)) if win.size > 0 else 0.0
            if mx > 0:
                tr.data = tr.data / mx

        # ---- Theoretical (TauP) shift per station relative to nearest station ----
        calc_shifts = {}
        if t_ref is not None:
            phase_key = align_phase.upper()
            for tr in st_comp:
                station_id = str(tr.stats.station)
                dist_deg = float(tr.stats.dist_deg)

                tts_sta = model.get_travel_times(
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

        # ===================== Stage 1: align to reference -> aligned_stack =====================
        aligned_stack = np.zeros(npts)
        for tr in st_comp:
            d = tr.data[:npts]
            lag0 = 0
            rolled = shift_left_zeropad(d, lag0)

            # Stage-1 alignment: include TauP expected shift + correlation correction
            station_id = str(tr.stats.station)
            if station_id in calc_shifts:
                expected_shift_samples = int(round(calc_shifts[station_id] * sample_rate))
                rolled_expected = shift_left_zeropad(rolled, expected_shift_samples)
                lag1 = expected_shift_samples + compute_lag(
                    ref, rolled_expected, win_start, win_end
                )
            else:
                lag1 = lag0 + compute_lag(ref, rolled, win_start, win_end)

            aligned_stack += shift_left_zeropad(d, lag1)

        win = aligned_stack[win_start:win_end]
        mx = np.max(np.abs(win)) if win.size > 0 else 0.0
        if mx > 0:
            aligned_stack = aligned_stack / mx

        # ===================== Stage 2: align to aligned_stack -> select traces =====================
        selected_aligned_stack = np.zeros(npts)
        selected_ids = set()
        station_corr = {}
        n_pass_window = 0
        pass_window_ids = set()
        snippet_by_station = {}

        for tr in st_comp:
            d = tr.data[:npts]
            station_id = str(tr.stats.station)

            lag0 = 0
            rolled = shift_left_zeropad(d, lag0)

            # Stage-2 alignment: include TauP expected shift + correlation correction
            if station_id in calc_shifts:
                expected_shift_samples = int(round(calc_shifts[station_id] * sample_rate))
                rolled_expected = shift_left_zeropad(rolled, expected_shift_samples)
                lag2 = expected_shift_samples + compute_lag(
                    aligned_stack, rolled_expected, win_start, win_end
                )
            else:
                lag2 = lag0 + compute_lag(aligned_stack, rolled, win_start, win_end)

            aligned_data = shift_left_zeropad(d, lag2)
            ref_window = aligned_stack[win_start:win_end]
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

        win = selected_aligned_stack[win_start:win_end]
        mx = np.max(np.abs(win)) if win.size > 0 else 0.0
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
                if station_id in calc_shifts:
                    expected_shift_samples = int(round(calc_shifts[station_id] * sample_rate))
                    x_expected = shift_left_zeropad(x, expected_shift_samples)
                    lag_delta = compute_lag(ref, x_expected, win_start, win_end)
                    lag3 = expected_shift_samples + lag_delta
                else:
                    lag3 = compute_lag(ref, x, win_start, win_end)
                y = shift_left_zeropad(x, lag3)

            # Store shift (in samples and seconds)
            station_shifts[station_id] = {
                'lag_samples': lag3,
                'lag_seconds': lag3 / sample_rate
            }

            # Per-trace normalization for plotting/stacking (correlation window only)
            win = y[win_start:win_end]
            my = np.max(np.abs(win)) if win.size > 0 else 1.0
            if my > 0:
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
        t_abs = start_time + (np.arange(npts) / sample_rate)
        mask = (t_abs >= start_time) & (t_abs <= end_time)

        # ---- Final stack (normalized) ----
        bank = aligned_bank_all if stack_all_traces_for_final else aligned_bank
        if len(bank) > 0:
            stack_vec = np.mean(np.vstack(bank), axis=0)
            win = stack_vec[win_start:win_end]
            ms = np.max(np.abs(win)) if win.size > 0 else 1.0
            if ms > 0:
                stack_vec = stack_vec / ms
        else:
            stack_vec = np.zeros_like(t_abs)

        # ---- Plot: superposition of Stage1/Stage2/Final stacks ----
        if show_stack_stage_plot:
            fig_stk, ax_stk = plt.subplots(1, 1, figsize=(10, 3.8))
            ax_stk.plot(t_abs[mask], aligned_stack[mask], color='C0', lw=2, label='Stage 1: aligned_stack')
            ax_stk.plot(t_abs[mask], selected_aligned_stack[mask], color='C1', lw=2, label='Stage 2: selected_aligned_stack')
            ax_stk.plot(t_abs[mask], stack_vec[mask], color='C3', lw=2.2, label='Final stack')
            ax_stk.axhline(0.0, color='k', lw=0.6, alpha=0.6)
            ax_stk.set_xlim(start_time, end_time)
            ax_stk.set_ylim(-1.1, 1.1)
            ax_stk.grid(alpha=0.2)
            ax_stk.set_xlabel('Time since origin (s)')
            ax_stk.set_ylabel('Stack (norm.)')
            ax_stk.set_title(f"Event {eve_id} {plot_comp}: Stage-1/Stage-2/Final stacks")
            ax_stk.legend(loc='upper right', fontsize=9)
            plt.tight_layout()

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
            if t_ref is not None:
                for axi in (ax, ax2):
                    axi.axvline(x=t_ref, color="k", lw=3, alpha=0.5, zorder=6)
        except Exception as e:
            print(f"    [WARN] Failed to draw vertical reference arrival for {align_phase}: {e}")

        # Cross-correlation window bounds as vertical yellow lines
        try:
            t_win_start = start_time + (win_start / sample_rate)
            t_win_end = start_time + (win_end / sample_rate)
            for axi in (ax, ax2):
                axi.axvline(x=t_win_start, color="y", lw=2, alpha=0.9, zorder=7)
                axi.axvline(x=t_win_end, color="y", lw=2, alpha=0.9, zorder=7)
        except Exception as e:
            print(f"    [WARN] Failed to draw correlation window bounds: {e}")

        # Cross-correlation search limits (green): yellow window expanded by move_limit_sec
        try:
            t_explore_start = t_win_start - move_limit_sec
            t_explore_end = t_win_end + move_limit_sec
            t_explore_start = max(start_time, t_explore_start)
            t_explore_end = min(start_time + (npts / sample_rate), t_explore_end)
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
                Line2D([0], [0], color='none', label=f'Pass r_win: {n_pass_window}'),
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
                'origin': origin,
                'station_shifts': station_shifts.copy(),
                'station_corr': station_corr.copy(),
                'calc_shifts': calc_shifts.copy(),
                'n_pass_window': int(n_pass_window),
                'pass_window_ids': sorted(list(pass_window_ids), key=lambda s: int(s)),
                'snippet_by_station': {k: v.copy() for k, v in snippet_by_station.items()},
                'ref_window': ref_window.copy(),
                'p_traveltime': None if p_traveltime is None else float(p_traveltime),
                's_traveltime': None if s_traveltime is None else float(s_traveltime),
                'station_ll': {k: (float(v[0]), float(v[1])) for k, v in name2ll.items()},
                # Stations that passed Stage-2 screening for this component
                'selected_ids': sorted(list(selected_ids), key=lambda s: int(s)),
                'aligned_traces_by_station': {k: v.copy() for k, v in aligned_traces_by_station.items()},
            }
            # Store t_ref
            all_component_data[comp_key]['t_ref'] = t_ref
            
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

            # ===================== Log10 envelope plot (single trace) =====================
            if num_traces == 1:
                try:
                    env = np.abs(hilbert(stack_vec))
                    std_sec = 1.0
                    std_samples = max(1.0, float(sample_rate) * std_sec)
                    win_samples = max(3, int(round(6.0 * std_samples)))
                    gauss = gaussian(win_samples, std_samples)
                    gauss = gauss / np.sum(gauss)
                    env_smooth = np.convolve(env, gauss, mode='same')
                    log_env = np.log10(np.maximum(env_smooth, 1e-12))

                    fig_env, ax_env = plt.subplots(figsize=(12, 4.5))
                    ax_env.plot(t_abs[mask], log_env[mask], color='k', lw=1.5)
                    ax_env.set_xlim(start_time, end_time)
                    ax_env.set_xlabel('Time since origin (s)', fontsize=11)
                    ax_env.set_ylabel('log10 envelope', fontsize=11)
                    ax_env.set_title(
                        f'Event {eve_id} - log10 envelope ({plot_comp})',
                        fontsize=12,
                        fontweight='bold',
                    )
                    ax_env.grid(alpha=0.2)
                    fig_env.subplots_adjust(bottom=0.28)

                    if origin is not None:
                        try:
                            origin_dt_utc = origin.datetime
                            if origin_dt_utc.tzinfo is None:
                                origin_dt_utc = origin_dt_utc.replace(tzinfo=timezone.utc)
                            ax_time = ax_env.twiny()
                            ax_time.set_xlim(ax_env.get_xlim())
                            ax_time.xaxis.set_label_position('bottom')
                            ax_time.xaxis.set_ticks_position('bottom')
                            ax_time.spines['bottom'].set_position(('outward', 36))
                            ax_time.spines['top'].set_visible(False)

                            ticks = ax_env.get_xticks()
                            labels = [
                                (origin_dt_utc + timedelta(seconds=float(t))).astimezone(timezone.utc).strftime('%H:%M:%S')
                                for t in ticks
                            ]
                            ax_time.set_xticks(ticks)
                            ax_time.set_xticklabels(labels)
                            date_str = origin_dt_utc.date().isoformat()
                            ax_time.set_xlabel(f'UTC time ({date_str})', fontsize=10)
                        except Exception as e:
                            print(f"[WARN] Failed to add UTC time axis (single envelope): {e}")

                    env_file = save_dir / f"{eve_id}_{plot_comp}_log10_envelope_{align_phase}.png"
                    #plt.savefig(env_file, dpi=300, bbox_inches='tight')
                    print(f"✓ Log10 envelope plot saved to: {env_file}")
                except Exception as e:
                    print(f"[WARN] Failed to create log10 envelope plot (single trace): {e}")

            # No Z-only R–T screening reuse.
            # ===================== Estimated vs calculated shift plot (single component) =====================
            if show_est_calc_shift_plot:
                common_sta = set(calc_shifts.keys()) & set(station_shifts.keys())
                if len(common_sta) == 0:
                    print("[WARN] No stations with both estimated and calculated shifts for comparison.")
                else:
                    stations = sorted(common_sta, key=lambda s: int(s))
                    est_shift = np.array([station_shifts[s]['lag_seconds'] for s in stations], dtype=float)
                    calc_shift = np.array([calc_shifts[s] for s in stations], dtype=float)

                    fig_ec, ax_ec = plt.subplots(1, 1, figsize=(6.2, 5.2))
                    ax_ec.scatter(calc_shift, est_shift, s=20, alpha=0.6)

                    minv = float(min(np.min(calc_shift), np.min(est_shift)))
                    maxv = float(max(np.max(calc_shift), np.max(est_shift)))
                    ax_ec.plot([minv, maxv], [minv, maxv], 'r--', lw=1.2, alpha=0.7, label='1:1 line')

                    ax_ec.set_xlabel('Calculated shift (s)')
                    ax_ec.set_ylabel('Estimated shift (s)')
                    ax_ec.set_title(f"Event {eve_id} {plot_comp}: Estimated vs Calculated shifts")
                    ax_ec.grid(alpha=0.3)
                    ax_ec.legend(loc='upper left', fontsize=9)
                    plt.tight_layout()

                    estcalc_file = save_dir / f"{eve_id}_{plot_comp}_est_vs_calc_shift_{align_phase}.png"
                    #plt.savefig(estcalc_file, dpi=300, bbox_inches='tight')
                    print(f"✓ Estimated vs calculated shift plot saved to: {estcalc_file}")

            # ===================== Snippet comparison plot (pass vs fail) =====================
            if show_snippet_compare_plot:
                try:
                    t_win = start_time + (np.arange(win_start, win_end) / sample_rate)
                    ref_win = ref_window
                    pass_list = sorted(list(pass_window_ids), key=lambda s: int(s))
                    fail_list = sorted(
                        [s for s in snippet_by_station.keys() if s not in pass_window_ids],
                        key=lambda s: int(s),
                    )

                    n_show = 10
                    pass_show = pass_list[:n_show]
                    fail_show = fail_list[:n_show]

                    fig_snip, (axp, axf) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)

                    for sid in pass_show:
                        axp.plot(t_win, snippet_by_station[sid], color='k', alpha=0.4, lw=1)
                    axp.plot(t_win, ref_win, color='C3', lw=2, label='Ref window')
                    axp.set_title(f"Pass r_win (N={len(pass_list)})")
                    axp.set_xlabel('Time since origin (s)')
                    axp.grid(alpha=0.3)

                    for sid in fail_show:
                        axf.plot(t_win, snippet_by_station[sid], color='k', alpha=0.4, lw=1)
                    axf.plot(t_win, ref_win, color='C3', lw=2, label='Ref window')
                    axf.set_title(f"Fail r_win (N={len(fail_list)})")
                    axf.set_xlabel('Time since origin (s)')
                    axf.grid(alpha=0.3)

                    axp.set_ylabel('Normalized amplitude')
                    axf.legend(loc='upper right', fontsize=8)
                    fig_snip.suptitle(
                        f"Event {eve_id} {plot_comp}: correlation-window snippets",
                        fontsize=12,
                        fontweight='bold',
                    )
                    plt.tight_layout()

                    snip_file = save_dir / f"{eve_id}_{plot_comp}_snippet_compare_{align_phase}.png"
                    #plt.savefig(snip_file, dpi=300, bbox_inches='tight')
                    print(f"✓ Snippet comparison plot saved to: {snip_file}")
                except Exception as e:
                    print(f"[WARN] Failed to create snippet comparison plot: {e}")

            # ===================== Individual seismograms (20 traces per subplot, 5 panels per figure) =====================
            if show_individual_seismograms:
                try:
                    all_rows = selected_rows + rejected_rows
                    all_rows.sort(key=lambda t: int(t[1]))

                    n_traces = len(all_rows)
                    if n_traces > 0:
                        n_per = 20
                        panels_per_fig = 5
                        n_panels = int(np.ceil(n_traces / n_per))
                        n_figs = int(np.ceil(n_panels / panels_per_fig))

                        for fig_idx in range(n_figs):
                            panel_start = fig_idx * panels_per_fig
                            panel_end = min((fig_idx + 1) * panels_per_fig, n_panels)
                            panels_in_fig = panel_end - panel_start

                            fig_ind, axes_ind = plt.subplots(
                                panels_in_fig,
                                1,
                                figsize=(10, 2.2 * panels_in_fig),
                                sharex=True,
                                sharey=False,
                            )
                            if panels_in_fig == 1:
                                axes_ind = [axes_ind]

                            for p in range(panels_in_fig):
                                axp = axes_ind[p]
                                global_panel = panel_start + p
                                start_idx = global_panel * n_per
                                end_idx = min((global_panel + 1) * n_per, n_traces)
                                subset = all_rows[start_idx:end_idx]

                                # Thresholding windows
                                t_win_start = start_time + (win_start / sample_rate)
                                t_win_end = start_time + (win_end / sample_rate)
                                t_explore_start = max(start_time, t_win_start - move_limit_sec)
                                t_explore_end = min(start_time + (npts / sample_rate), t_win_end + move_limit_sec)
                                axp.axvline(x=t_win_start, color='y', lw=1.2, alpha=0.9)
                                axp.axvline(x=t_win_end, color='y', lw=1.2, alpha=0.9)
                                axp.axvline(x=t_explore_start, color='g', lw=1.2, alpha=0.9)
                                axp.axvline(x=t_explore_end, color='g', lw=1.2, alpha=0.9)

                                for idx_in_subset, (_, station_id, y) in enumerate(subset):
                                    i = (len(subset) - 1) - idx_in_subset
                                    passed_win = station_id in pass_window_ids
                                    trace_color = 'k' if passed_win else 'red'
                                    axp.plot(
                                        t_abs[mask],
                                        y[mask] + i,
                                        color=trace_color,
                                        lw=0.7,
                                    )
                                    axp.text(
                                        t_abs[mask][0],
                                        i,
                                        station_id,
                                        fontsize=6,
                                        va='center',
                                    )

                                # Reference stack above traces
                                ref_offset = len(subset) + 1
                                axp.plot(
                                    t_abs[mask],
                                    stack_vec[mask] + ref_offset,
                                    color='C3',
                                    lw=1.2,
                                )

                                axp.set_ylim(-1, len(subset) + 2)
                                axp.grid(alpha=0.2)
                                axp.set_ylabel('Trace index')

                            axes_ind[-1].set_xlabel('Time since origin (s)')
                            fig_ind.suptitle(
                                f"Event {eve_id} {plot_comp}: individual seismograms "
                                f"(20 per panel, fig {fig_idx + 1}/{n_figs})",
                                fontsize=12,
                                fontweight='bold',
                            )
                            plt.tight_layout()

                            ind_file = save_dir / (
                                f"{eve_id}_{plot_comp}_individual_seismograms_{align_phase}_fig{fig_idx + 1}.png"
                            )
                            #plt.savefig(ind_file, dpi=300, bbox_inches='tight')
                            print(f"✓ Individual seismograms plot saved to: {ind_file}")
                except Exception as e:
                    print(f"[WARN] Failed to create individual seismograms plot: {e}")

            # ===================== Station maps: pass each threshold and both =====================
            try:
                tr_map = aligned_traces_by_station
                all_stations = sorted(tr_map.keys(), key=lambda s: int(s))
                pass_win = set(pass_window_ids)

                fig_map, axm = plt.subplots(1, 1, figsize=(6.5, 5.5))
                pass_lats = [name2ll[s][0] for s in all_stations if s in pass_win]
                pass_lons = [name2ll[s][1] for s in all_stations if s in pass_win]
                fail_lats = [name2ll[s][0] for s in all_stations if s not in pass_win]
                fail_lons = [name2ll[s][1] for s in all_stations if s not in pass_win]

                if len(fail_lons) > 0:
                    axm.scatter(fail_lons, fail_lats, s=18, c='0.7', label='Fail')
                if len(pass_lons) > 0:
                    axm.scatter(pass_lons, pass_lats, s=22, c='C3', label='Pass')

                axm.set_title('Pass r_win', fontsize=11, fontweight='bold')
                axm.grid(alpha=0.3)
                axm.set_xlabel('Longitude')
                axm.set_ylabel('Latitude')
                axm.legend(loc='upper right', fontsize=9)

                fig_map.suptitle(
                    f"Event {eve_id} {plot_comp}: stations passing thresholds",
                    fontsize=13,
                    fontweight='bold',
                )
                plt.tight_layout()

                map_file = save_dir / f"{eve_id}_{plot_comp}_station_pass_map_{align_phase}.png"
                #plt.savefig(map_file, dpi=300, bbox_inches='tight')
                print(f"✓ Station pass/fail map saved to: {map_file}")
            except Exception as e:
                print(f"[WARN] Failed to create station pass/fail maps: {e}")

            # Show figures for single-component mode
            if show_aligned_phase_plot:
                plt.show()
            else:
                plt.close(fig)



# ===================== Three-component combined plotting =====================
if show_three_comp_plot and process_as_three_comp and len(all_component_data) == 3:
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
    stack_by_comp = {}
    t_abs = first_data['t_abs']
    mask = first_data['mask']
    sample_rate_env = first_data['sample_rate']
    origin_env = first_data.get('origin')
    
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
            t_win_start = start_time + (win_start / sample_rate)
            t_win_end = start_time + (win_end / sample_rate)
            ax.axvline(x=t_win_start, color='y', lw=2, alpha=0.9, zorder=7)
            ax.axvline(x=t_win_end, color='y', lw=2, alpha=0.9, zorder=7)
        except Exception as e:
            print(f"[WARN] Failed to draw correlation window bounds (top {comp_name}): {e}")

        # Cross-correlation search limits (green): yellow window expanded by move_limit_sec
        try:
            t_explore_start = t_win_start - move_limit_sec
            t_explore_end = t_win_end + move_limit_sec
            t_explore_start = max(start_time, t_explore_start)
            t_explore_end = min(start_time + (npts / sample_rate), t_explore_end)
            ax.axvline(x=t_explore_start, color='g', lw=2, alpha=0.9, zorder=7)
            ax.axvline(x=t_explore_end, color='g', lw=2, alpha=0.9, zorder=7)
        except Exception as e:
            print(f"[WARN] Failed to draw correlation search limits (top {comp_name}): {e}")

        # Legend for window bounds (only once, top-left panel)
        if idx == 0:
            try:
                n_pass_window = int(data.get('n_pass_window', 0))
                legend_handles = [
                    Line2D([0], [0], color='y', lw=2, label='Correlation window'),
                    Line2D([0], [0], color='g', lw=2, label='Correlation search (±move_limit_sec)'),
                    Line2D([0], [0], color='none', label=f'Pass r_win: {n_pass_window}'),
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

        if t_ref is not None:
            ax2.axvline(x=t_ref, color='r', lw=2, alpha=0.6, linestyle='--', zorder=6)
        # Cross-correlation window bounds
        try:
            t_win_start = start_time + (win_start / sample_rate)
            t_win_end = start_time + (win_end / sample_rate)
            ax2.axvline(x=t_win_start, color='y', lw=2, alpha=0.9, zorder=7)
            ax2.axvline(x=t_win_end, color='y', lw=2, alpha=0.9, zorder=7)
        except Exception as e:
            print(f"[WARN] Failed to draw correlation window bounds (bottom {comp_name}): {e}")

        # Cross-correlation search limits (green): yellow window expanded by move_limit_sec
        try:
            t_explore_start = t_win_start - move_limit_sec
            t_explore_end = t_win_end + move_limit_sec
            t_explore_start = max(start_time, t_explore_start)
            t_explore_end = min(start_time + (npts / sample_rate), t_explore_end)
            ax2.axvline(x=t_explore_start, color='g', lw=2, alpha=0.9, zorder=7)
            ax2.axvline(x=t_explore_end, color='g', lw=2, alpha=0.9, zorder=7)
        except Exception as e:
            print(f"[WARN] Failed to draw correlation search limits (bottom {comp_name}): {e}")

        stack_by_comp[comp_name] = stack_vec

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

    # ===================== Log10 envelope of 3-component stack =====================
    try:
        if all(comp in stack_by_comp for comp in comp_order):
            z = stack_by_comp['DPZ']
            r = stack_by_comp['R']
            t = stack_by_comp['T']
            env_z = np.abs(hilbert(z))
            env_r = np.abs(hilbert(r))
            env_t = np.abs(hilbert(t))
            env_rms = np.sqrt((env_z ** 2 + env_r ** 2 + env_t ** 2) / 3.0)
            std_sec = 1.0
            std_samples = max(1.0, float(sample_rate_env) * std_sec)
            win_samples = max(3, int(round(6.0 * std_samples)))
            gauss = gaussian(win_samples, std_samples)
            gauss = gauss / np.sum(gauss)
            env_rms_smooth = np.convolve(env_rms, gauss, mode='same')
            log_env = np.log10(np.maximum(env_rms_smooth, 1e-12))

            fig_env, ax_env = plt.subplots(figsize=(12, 4.5))
            ax_env.plot(t_abs[mask], log_env[mask], color='k', lw=1.5)
            ax_env.set_xlim(start_time, end_time)
            ax_env.set_xlabel('Time since origin (s)', fontsize=11)
            ax_env.set_ylabel('log10 envelope', fontsize=11)
            ax_env.set_title(
                f'Event {eve_id} - log10 RMS envelope of 3-component stack',
                fontsize=12,
                fontweight='bold',
            )
            ax_env.grid(alpha=0.2)
            fig_env.subplots_adjust(bottom=0.28)

            if origin_env is not None:
                try:
                    origin_dt_utc = origin_env.datetime
                    if origin_dt_utc.tzinfo is None:
                        origin_dt_utc = origin_dt_utc.replace(tzinfo=timezone.utc)
                    ax_time = ax_env.twiny()
                    ax_time.set_xlim(ax_env.get_xlim())
                    ax_time.xaxis.set_label_position('bottom')
                    ax_time.xaxis.set_ticks_position('bottom')
                    ax_time.spines['bottom'].set_position(('outward', 36))
                    ax_time.spines['top'].set_visible(False)

                    ticks = ax_env.get_xticks()
                    labels = [
                        (origin_dt_utc + timedelta(seconds=float(t))).astimezone(timezone.utc).strftime('%H:%M:%S')
                        for t in ticks
                    ]
                    ax_time.set_xticks(ticks)
                    ax_time.set_xticklabels(labels)
                    date_str = origin_dt_utc.date().isoformat()
                    ax_time.set_xlabel(f'UTC time ({date_str})', fontsize=10)
                except Exception as e:
                    print(f"[WARN] Failed to add UTC time axis (envelope): {e}")

            env_file = save_dir / f"{eve_id}_3comp_log10_envelope_{align_phase}.png"
            #plt.savefig(env_file, dpi=300, bbox_inches='tight')
            print(f"✓ Log10 envelope plot saved to: {env_file}")
    except Exception as e:
        print(f"[WARN] Failed to create log10 envelope plot: {e}")

    # No R–T zero-diff station list saved.
    # ===================== Stack compare plot: all aligned vs r_min-selected =====================
    if show_rt_filter_compare_plot:
        print("Creating stack comparison plot (all aligned vs r_min-selected)...")

        # Figure layout: 3 rows, 1 column (Z / R / T) — vertical arrangement
        fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(9, 12), sharex=True, sharey=True)
        comp_order = ['DPZ', 'R', 'T']
        comp_titles_cmp = ['Z stack', 'R stack', 'T stack']
        utc_tz = timezone.utc

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
            p_time = data.get('p_traveltime')
            s_time = data.get('s_traveltime')

            tr_map = data.get('aligned_traces_by_station', {})
            all_stations = sorted(tr_map.keys(), key=lambda s: int(s))

            # Black: stack of all aligned traces
            stack_black = np.zeros_like(t_abs)
            if len(all_stations) > 0:
                bank_all = [tr_map[sta] for sta in all_stations]
                stack_black = np.mean(np.vstack(bank_all), axis=0)
                ms = np.max(np.abs(stack_black)) or 1.0
                stack_black = stack_black / ms

            # Red: stack of traces that pass r_min thresholds (selected_ids)
            sel_ids = data.get('selected_ids', [])
            sel_ids = [s for s in sel_ids if s in tr_map]
            n_pass_window = int(data.get('n_pass_window', len(sel_ids)))
            stack_red = stack_black
            if len(sel_ids) > 0:
                bank_sel = [tr_map[sta] for sta in sel_ids]
                stack_red = np.mean(np.vstack(bank_sel), axis=0)
                ms = np.max(np.abs(stack_red)) or 1.0
                stack_red = stack_red / ms

            axc.plot(t_abs[mask], stack_black[mask], color='k', lw=2, label='All aligned traces')
            axc.plot(
                t_abs[mask],
                stack_red[mask],
                color='r',
                lw=2,
                label=f'Pass r_win N={n_pass_window}',
            )
            axc.axhline(0.0, color='k', lw=0.6, alpha=0.6)
            if p_time is not None:
                axc.axvline(x=p_time, color='b', lw=1.5, alpha=0.7, linestyle='--', label='P arrival')
            if s_time is not None:
                axc.axvline(x=s_time, color='g', lw=1.5, alpha=0.7, linestyle='--', label='S arrival')
            axc.set_xlim(start_time, end_time)
            axc.set_ylim(-1.1, 1.1)
            axc.grid(alpha=0.2)
            axc.set_title(comp_titles_cmp[j], fontsize=12, fontweight='bold')
            axc.set_xlabel('Time since origin (s)', fontsize=11)
            if j != 2:
                axc.set_xlabel('')
            if j == 0:
                axc.set_ylabel('Stack (norm.)', fontsize=11)
            axc.legend(loc='upper right', fontsize=9)

            if j == 2:
                try:
                    origin_utc = data.get('origin')
                    if origin_utc is not None:
                        origin_dt_utc = origin_utc.datetime
                        if origin_dt_utc.tzinfo is None:
                            origin_dt_utc = origin_dt_utc.replace(tzinfo=timezone.utc)
                        ax_time = axc.twiny()
                        ax_time.set_xlim(axc.get_xlim())
                        ax_time.xaxis.set_label_position('bottom')
                        ax_time.xaxis.set_ticks_position('bottom')
                        ax_time.spines['bottom'].set_position(('outward', 36))
                        ax_time.spines['top'].set_visible(False)

                        ticks = axc.get_xticks()
                        labels = [
                            (origin_dt_utc + timedelta(seconds=float(t))).astimezone(utc_tz).strftime('%H:%M:%S')
                            for t in ticks
                        ]
                        ax_time.set_xticks(ticks)
                        ax_time.set_xticklabels(labels)
                        date_str = origin_dt_utc.date().isoformat()
                        ax_time.set_xlabel(f'UTC time ({date_str})', fontsize=10)
                except Exception as e:
                    print(f"[WARN] Failed to add UTC time axis: {e}")

        fig_cmp.suptitle(
            f'Event {eve_id} - Stack compare (black: all aligned; red: pass r_min thresholds)',
            fontsize=13,
            fontweight='bold'
        )
        plt.tight_layout()

        # Save comparison figure
        cmp_file = save_dir / f"{eve_id}_rtfilter_stack_compare_{align_phase}.png"
        #plt.savefig(cmp_file, dpi=300, bbox_inches='tight')
        print(f"✓ Stack comparison plot saved to: {cmp_file}")
        # plt.show()

    # ===================== Individual seismograms (20 traces per subplot, 5 panels per figure, 3 components) =====================
    if show_individual_seismograms:
        try:
            for comp_name, comp_title in zip(['DPZ', 'R', 'T'], ['Z', 'R', 'T']):
                if comp_name not in all_component_data:
                    continue

                data = all_component_data[comp_name]
                all_rows = data.get('all_rows', [])
                all_rows = sorted(all_rows, key=lambda t: int(t[1]))
                t_abs = data['t_abs']
                mask = data['mask']
                sample_rate = data['sample_rate']
                win_start = data['win_start']
                win_end = data['win_end']
                move_limit_sec = data['move_limit_sec']
                npts = data['npts']

                n_traces = len(all_rows)
                if n_traces == 0:
                    continue

                n_per = 20
                panels_per_fig = 5
                n_panels = int(np.ceil(n_traces / n_per))
                n_figs = int(np.ceil(n_panels / panels_per_fig))

                for fig_idx in range(n_figs):
                    panel_start = fig_idx * panels_per_fig
                    panel_end = min((fig_idx + 1) * panels_per_fig, n_panels)
                    panels_in_fig = panel_end - panel_start

                    fig_ind, axes_ind = plt.subplots(
                        panels_in_fig,
                        1,
                        figsize=(10, 2.2 * panels_in_fig),
                        sharex=True,
                        sharey=False,
                    )
                    if panels_in_fig == 1:
                        axes_ind = [axes_ind]

                    for p in range(panels_in_fig):
                        axp = axes_ind[p]
                        global_panel = panel_start + p
                        start_idx = global_panel * n_per
                        end_idx = min((global_panel + 1) * n_per, n_traces)
                        subset = all_rows[start_idx:end_idx]

                        # Thresholding windows
                        t_win_start = start_time + (win_start / sample_rate)
                        t_win_end = start_time + (win_end / sample_rate)
                        t_explore_start = max(start_time, t_win_start - move_limit_sec)
                        t_explore_end = min(start_time + (npts / sample_rate), t_win_end + move_limit_sec)
                        axp.axvline(x=t_win_start, color='y', lw=1.2, alpha=0.9)
                        axp.axvline(x=t_win_end, color='y', lw=1.2, alpha=0.9)
                        axp.axvline(x=t_explore_start, color='g', lw=1.2, alpha=0.9)
                        axp.axvline(x=t_explore_end, color='g', lw=1.2, alpha=0.9)

                        for idx_in_subset, (_, station_id, y) in enumerate(subset):
                            i = (len(subset) - 1) - idx_in_subset
                            passed_win = station_id in pass_window_ids
                            trace_color = 'k' if passed_win else 'red'
                            axp.plot(
                                t_abs[mask],
                                y[mask] + i,
                                color=trace_color,
                                lw=0.7,
                            )
                            axp.text(
                                t_abs[mask][0],
                                i,
                                station_id,
                                fontsize=6,
                                va='center',
                            )

                        # Reference stack above traces
                        ref_offset = len(subset) + 1
                        stack_ref = data.get('stack_vec', None)
                        if stack_ref is not None:
                            axp.plot(
                                t_abs[mask],
                                stack_ref[mask] + ref_offset,
                                color='C3',
                                lw=1.2,
                            )

                        axp.set_ylim(-1, len(subset) + 2)
                        axp.grid(alpha=0.2)
                        axp.set_ylabel('Trace index')

                    axes_ind[-1].set_xlabel('Time since origin (s)')
                    fig_ind.suptitle(
                        f"Event {eve_id} {comp_title}: individual seismograms "
                        f"(20 per panel, fig {fig_idx + 1}/{n_figs})",
                        fontsize=12,
                        fontweight='bold',
                    )
                    plt.tight_layout()

                    ind_file = save_dir / (
                        f"{eve_id}_{comp_title}_individual_seismograms_{align_phase}_fig{fig_idx + 1}.png"
                    )
                    #plt.savefig(ind_file, dpi=300, bbox_inches='tight')
                    print(f"✓ Individual seismograms plot saved to: {ind_file}")
        except Exception as e:
            print(f"[WARN] Failed to create individual seismograms plots (3 components): {e}")

    # ===================== Station maps: pass r_win (3 components) =====================
    try:
        fig_map, axes_map = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
        comp_order = ['DPZ', 'R', 'T']
        comp_titles_map = ['Z', 'R', 'T']

        for j, comp_name in enumerate(comp_order):
            axm = axes_map[j]
            if comp_name not in all_component_data:
                axm.set_axis_off()
                continue

            data = all_component_data[comp_name]
            station_ll = data.get('station_ll', {})
            tr_map = data.get('aligned_traces_by_station', {})
            all_stations = sorted(tr_map.keys(), key=lambda s: int(s))
            pass_set = set(data.get('pass_window_ids', []))

            pass_lats = [station_ll[s][0] for s in all_stations if s in pass_set and s in station_ll]
            pass_lons = [station_ll[s][1] for s in all_stations if s in pass_set and s in station_ll]
            fail_lats = [station_ll[s][0] for s in all_stations if s not in pass_set and s in station_ll]
            fail_lons = [station_ll[s][1] for s in all_stations if s not in pass_set and s in station_ll]

            if len(fail_lons) > 0:
                axm.scatter(fail_lons, fail_lats, s=16, c='0.7', label='Fail')
            if len(pass_lons) > 0:
                axm.scatter(pass_lons, pass_lats, s=20, c='C3', label='Pass')

            axm.set_title(comp_titles_map[j], fontsize=12, fontweight='bold')
            if j == 0:
                axm.set_ylabel('Latitude')
            axm.grid(alpha=0.3)
            axm.set_xlabel('Longitude')
            axm.legend(loc='upper right', fontsize=8)

        fig_map.suptitle(
            f'Event {eve_id} - Stations passing thresholds ({align_phase})',
            fontsize=13,
            fontweight='bold'
        )
        plt.tight_layout()

        map_file = save_dir / f"{eve_id}_station_pass_map_{align_phase}.png"
        #plt.savefig(map_file, dpi=300, bbox_inches='tight')
        print(f"✓ Station pass/fail map saved to: {map_file}")
    except Exception as e:
        print(f"[WARN] Failed to create station pass/fail map (3 components): {e}")

    # ===================== Shift comparison plot: Radial vs Transverse =====================
    if show_diff_rt_plot:
        if 'R' in all_component_data and 'T' in all_component_data:
            print("Creating shift comparison plot (Radial vs Transverse)...")
            print(
                "Shift comparison parameters: "
                f"align_phase={align_phase}, start_time={start_time}, end_time={end_time}, "
                f"win_pre={win_pre}, win_post={win_post}, "
                f"move_limit_sec={move_limit_sec}"
            )
            
            r_shifts = all_component_data['R']['station_shifts']
            t_shifts = all_component_data['T']['station_shifts']
            r_corr = all_component_data['R']['station_corr']
            t_corr = all_component_data['T']['station_corr']
            
            # Find common stations
            common_stations = set(r_shifts.keys()) & set(t_shifts.keys())
            common_corr_stations = set(r_corr.keys()) & set(t_corr.keys())

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
                    f"  win_pre: {win_pre}\n"
                    f"  win_post: {win_post}\n"
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

    # ===================== Estimated vs calculated shift plot (3 components) =====================
    if show_est_calc_shift_plot:
        print("Creating estimated vs calculated shift plot (3 components)...")

        fig_ec, axes_ec = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)
        comp_order = ['DPZ', 'R', 'T']
        comp_titles_ec = ['Z', 'R', 'T']

        for j, comp_name in enumerate(comp_order):
            axc = axes_ec[j]
            if comp_name not in all_component_data:
                axc.set_axis_off()
                continue

            data = all_component_data[comp_name]
            station_shifts = data.get('station_shifts', {})
            calc_shifts = data.get('calc_shifts', {})

            common_sta = set(calc_shifts.keys()) & set(station_shifts.keys())
            if len(common_sta) == 0:
                axc.text(0.5, 0.5, 'No common stations', ha='center', va='center')
                axc.set_axis_off()
                continue

            stations = sorted(common_sta, key=lambda s: int(s))
            est_shift = np.array([station_shifts[s]['lag_seconds'] for s in stations], dtype=float)
            calc_shift = np.array([calc_shifts[s] for s in stations], dtype=float)

            axc.scatter(calc_shift, est_shift, s=18, alpha=0.6)

            minv = float(min(np.min(calc_shift), np.min(est_shift)))
            maxv = float(max(np.max(calc_shift), np.max(est_shift)))
            axc.plot([minv, maxv], [minv, maxv], 'r--', lw=1.2, alpha=0.7)

            axc.set_title(comp_titles_ec[j], fontsize=12, fontweight='bold')
            axc.grid(alpha=0.3)
            axc.set_xlabel('Calculated shift (s)', fontsize=10)
            if j == 0:
                axc.set_ylabel('Estimated shift (s)', fontsize=10)

        fig_ec.suptitle(
            f'Event {eve_id} - Estimated vs Calculated shifts ({align_phase})',
            fontsize=13,
            fontweight='bold'
        )
        plt.tight_layout()

        estcalc_file = save_dir / f"{eve_id}_est_vs_calc_shift_{align_phase}.png"
        #plt.savefig(estcalc_file, dpi=300, bbox_inches='tight')
        print(f"✓ Estimated vs calculated shift plot saved to: {estcalc_file}")

        # Show all figures together (three-component + shift comparison)
        # plt.show()

    # ===================== Show all figures together at the end =====================
    plt.show()