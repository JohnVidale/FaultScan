import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from obspy import read, UTCDateTime, Stream
from obspy.geodetics import degrees2kilometers, locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel
from obspy.signal.rotate import rotate_ne_rt

# ===================== Parameter setup =====================
# Bandpass filter (Hz)
min_freq, max_freq = 0.5, 15.0

# Plotting time window (seconds since origin)
time_min, time_max = 0.0, 10.0

# Paths
info_root = Path(
    "/Users/vidale/Documents/Research/Mingze_SJF/20220930_events_cut/event_sta_info"
)
data_path = Path(
    "/Users/vidale/Documents/Research/Mingze_SJF/20220930_events_cut/07_1hour_20220930"
)

# Run modes
all_channels = False
all_events = False

# Single run selection (used when the corresponding "all_*" is False)
event = "CI_40353544"
channel = "DP1"  # DPZ / DP1 / DP2  (DP1~N-like, DP2~E-like) 
                 # either DP1 or DP2 signals that horizontal component is needed

# Horizontal rotation target when channel is DP1/DP2
sel_comp = "T"  # 'R' or 'T'

# Alignment phase
align_phase = "S"  # 'P' or 'S'

# Analysis window relative to the theoretical arrival time of the nearest station
start_time = -1.0
end_time = 1.0
duration = end_time - start_time

# Short-window correlation parameters (seconds)
short_win_pre = 0.5
short_win_post = 0.5

# Maximum allowed shift (samples) searched in compute_lag
move_limit = 35

# Rising search parameters (NOTE: current behavior forces rising_time_shift = 0.0)
search_win_pre = 5.0
search_win_post = 2.0
rising_amp_factor = 1.5

# Final stack mode: True = use ALL traces after final lag3; False = use only selected traces
stack_all_traces_for_final = True

# Travel-time model
model = TauPyModel(model="iasp91")


# ===================== Helper functions =====================
def compute_lag(ref: np.ndarray, d: np.ndarray, win_start: int, win_end: int) -> int:
    """Compute integer lag (samples) by maximizing correlation within a short window.

    This matches the original implementation:
      - ref_window = ref[win_start:win_end]
      - d_window   = d[win_start-move_limit : win_end+move_limit]
      - corr = np.correlate(d_window, ref_window, mode='valid')
      - lag = argmax(corr) - move_limit

    Returns:
        Best lag (integer samples). Positive lag advances the target waveform.
    """
    ref_window = ref[win_start:win_end]
    d_window = d[win_start - move_limit : win_end + move_limit]
    corr = np.correlate(d_window, ref_window, mode="valid")
    return int(np.argmax(corr) - move_limit)


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


# ===================== Channel list =====================
if all_channels:
    channels = ["DPZ", "DP1", "DP2"]
else:
    channels = [channel]


# ===================== Main loop =====================
for channel in channels:
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
                print(fpath)
                if not fpath.exists():
                    print("No such file")
                    continue

                # Read the component and slice [origin, origin + time_max]
                tr = read(str(fpath))[0]
                tr = tr.slice(starttime=origin, endtime=origin + time_max)
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
            abs_pick = p_arrival_time + start_time
            phase_traveltime = float(p_traveltime)  # seconds since origin
        elif align_phase == "S" and s_traveltime is not None:
            abs_pick = s_arrival_time + start_time
            phase_traveltime = float(s_traveltime)
        else:
            print("    No valid phase for alignment. Skip array.")
            continue

        # ---- Keep traces that cover [abs_pick, abs_pick + duration] ----
        st_window = Stream()
        kept = 0

        for tr in st_all:
            # Keep the entire [0, time_max] record as long as it covers the analysis window
            if tr.stats.endtime >= abs_pick and tr.stats.starttime <= abs_pick + duration:
                tr_i = tr.copy()
                if tr_i is not None and tr_i.stats.npts > 0:
                    tr_i.stats.station = tr.stats.station
                    st_window.append(tr_i)
                    kept += 1

        if kept == 0:
            print("  No data in analysis window (global pick from nearest station).")
            continue

        # Label to show on the figure
        plot_comp = channel

        # ---- Rotate horizontal components to R/T (only when channel is DP1/DP2) ----
        if channel in ["DP1", "DP2"]:
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
        print(f"    sample_rate = {sample_rate:.1f} Hz")

        # ---- Correlation window indices ----
        t0 = float(phase_traveltime) if phase_traveltime is not None else 0.0

        # Rising search around the theoretical arrival (kept for completeness).
        # NOTE: The current script forces rising_time_shift = 0.0 (functional behavior unchanged).
        search_win_start = int(max(0, sample_rate * (t0 - search_win_pre)))
        search_win_end = int(min(npts, sample_rate * (t0 + search_win_post)))
        search_window = ref[search_win_start:search_win_end]
        if search_window.size > 0:
            peak_amp = np.max(np.abs(search_window))
            _ = np.argmax(np.abs(search_window) > peak_amp / rising_amp_factor)
            rising_time_shift = 0.0
        else:
            rising_time_shift = 0.0
        print(f"    rising_time_shift = {rising_time_shift:.3f}s")

        # Plotting window around the selected phase (not used for slicing, only for plotting bounds)
        if align_phase == "P" and p_traveltime is not None:
            t_min = start_time + float(p_traveltime) + rising_time_shift
        elif align_phase == "S" and s_traveltime is not None:
            t_min = start_time + float(s_traveltime) + rising_time_shift
        else:
            t_min = start_time + rising_time_shift
        t_max = t_min + duration

        # Correlation window centered at (theoretical arrival + rising_time_shift)
        center_time = t0 + rising_time_shift
        win_start = int(max(0, sample_rate * (center_time - short_win_pre)))
        win_end = int(min(npts, sample_rate * (center_time + short_win_post)))

        # ===================== Stage 1: align to reference -> aligned_stack =====================
        aligned_stack = np.zeros(npts)
        for tr in st_comp:
            d = tr.data[:npts]
            lag0 = 0
            rolled = shift_left_zeropad(d, lag0)
            lag1 = lag0 + compute_lag(ref, rolled, win_start, win_end)
            aligned_stack += shift_left_zeropad(d, lag1)

        mx = np.max(np.abs(aligned_stack))
        if mx > 0:
            aligned_stack = aligned_stack / mx

        # ===================== Stage 2: align to aligned_stack -> select traces =====================
        selected_aligned_stack = np.zeros(npts)
        selected_ids = set()

        for tr in st_comp:
            d = tr.data[:npts]
            station_id = str(tr.stats.station)

            lag0 = 0
            rolled = shift_left_zeropad(d, lag0)
            lag2 = lag0 + compute_lag(aligned_stack, rolled, win_start, win_end)

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

            if r_window >= 0.7 and r_whole >= 0.1:
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

        for tr in st_comp:
            x = tr.data[:npts]
            station_id = str(tr.stats.station)

            # Final alignment: align every trace to the reference station so the x-axis remains
            # the reference station's time since origin.
            if station_id == ref_station_id:
                lag3 = 0
                y = x.copy()
            else:
                lag3 = compute_lag(ref, x, win_start, win_end)
                y = shift_left_zeropad(x, lag3)

            # Per-trace normalization for plotting/stacking
            my = np.max(np.abs(y)) or 1.0
            y = y / my

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
        mask = (t_abs >= time_min) & (t_abs <= time_max)

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

        ax.set_xlim(time_min, time_max)
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

        # Bottom panel: normalized stack
        ax2.plot(t_abs[mask], stack_vec[mask], color="C3", lw=1.5)
        ax2.axhline(0.0, color="k", lw=0.6)
        ax2.set_xlim(time_min, time_max)
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

        # Save figure (same location/pattern as the original script)
        save_path = Path(
            "/Users/vidale/Documents/Research/Mingze_SJF/output"
        )
        save_dir = save_path / eve_id
        save_dir.mkdir(parents=True, exist_ok=True)
        save_file = save_dir / f"{eve_id}_{plot_comp}_{align_phase}.png"
        #plt.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.show()