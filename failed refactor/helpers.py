import numpy as np
import time
from obspy import UTCDateTime

_start_cpu_time = time.process_time()
_start_wall_time = time.perf_counter()
_timing_reported = False

def compute_lag(ref: np.ndarray, d: np.ndarray, win_start: int, win_end: int, move_limit_samples: int) -> int:
    ref_window = ref[win_start:win_end]
    d_window = d[win_start - move_limit_samples : win_end + move_limit_samples]
    corr = np.correlate(d_window, ref_window, mode="valid")
    return int(np.argmax(corr) - move_limit_samples)

def shift_left_zeropad(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x)
    y = np.zeros_like(x)
    if n == 0:
        y[:] = x
    elif n > 0:
        if n < x.size:
            y[:-n] = x[n:]
    else:
        n = -n
        if n < x.size:
            y[n:] = x[:-n]
    return y

def set_figure_title(fig, title: str) -> None:
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass

def report_timing_once() -> None:
    global _timing_reported
    if _timing_reported:
        return
    cpu_sec = time.process_time() - _start_cpu_time
    wall_sec = time.perf_counter() - _start_wall_time
    print(f"\033[31mTiming: cpu={cpu_sec:.2f}s  wall={wall_sec:.2f}s\033[0m")
    _timing_reported = True

def add_catalog_event_lines(ax, origin_time, catalog_df, tmin, tmax) -> None:
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
        try:
            skip_int = int(row.get("skip", 0))
        except Exception:
            skip_int = 0
        ax.axvline(x=dt, color=color_map.get(skip_int, "red"), lw=1.1, alpha=0.8, zorder=6)