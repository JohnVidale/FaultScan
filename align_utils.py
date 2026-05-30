from __future__ import annotations

from datetime import timedelta, timezone

import numpy as np
from obspy import UTCDateTime


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
