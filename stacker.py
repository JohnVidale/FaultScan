import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import NullFormatter
from pathlib import Path
from datetime import timezone
from obspy import read, UTCDateTime

OUTPUT_ROOT = Path("/Users/vidale/Documents/Research/Mingze_SJF/output")
CATALOG_FILE = Path("/Users/vidale/Documents/Research/Mingze_SJF/20220930_events_cut/event_sta_info/catalog_local_hand.xlsx")
ORIGIN_COL = "origin_time"
SHOW_ORIGINAL_PLOT = False
SHOW_CHOPPED_PLOT = False
SHOW_LEGEND = False
SHOW_RADIAL_ONLY = False
SHOW_OFFSET_TRACES = True
SHOW_SUPERIMPOSED_MASKED = True
SHOW_MEDIAN_TRACE = True

MAG_MIN = 0.0
MAG_DIFF_MIN = 0.8
EVENT_WINDOW_START = 0.0
EVENT_WINDOW_END = 25.0
ALIGN_WINDOW_START = 4.6 # 3 to 10 Hz
ALIGN_WINDOW_END   = 6.0
# ALIGN_WINDOW_START = 4.4 # 1 to 5 Hz
# ALIGN_WINDOW_END   = 5.6
OFFSET_STEP = 0.6
REPLACE_LEVEL = np.nan # can be np.nan, 0, or 1

COMPONENTS = ["DPZ", "R", "T"]
COMPONENT_LABELS = {
    "DPZ": "Vertical (Z)",
    "R": "Radial (R)",
    "T": "Transverse (T)",
}


def load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if ORIGIN_COL not in df.columns:
        raise ValueError(f"Catalog missing '{ORIGIN_COL}' column")
    if "evid" not in df.columns:
        raise ValueError("Catalog missing 'evid' column")
    return df


def load_stack_traces(root_dir: Path) -> dict:
    traces = {}
    for comp in COMPONENTS:
        mseed_file = root_dir / f"{comp}_stack.mseed"
        if not mseed_file.exists():
            continue
        st = read(str(mseed_file))
        if len(st) == 0:
            continue
        traces[comp] = st[0]
    return traces


def plot_stacks(
    origin_time: UTCDateTime,
    traces: dict,
    title: str,
    catalog: pd.DataFrame,
    catalog_all: pd.DataFrame,
    show_original: bool,
) -> None:
    if not traces:
        print("[WARN] No stack mseed files found in output directory")
        return

    plot_components = ["R"] if SHOW_RADIAL_ONLY else COMPONENTS

    tmin = None
    tmax = None
    for tr in traces.values():
        t_start = tr.stats.starttime
        t_end = tr.stats.endtime
        if tmin is None or t_start < tmin:
            tmin = t_start
        if tmax is None or t_end > tmax:
            tmax = t_end

    if show_original:
        fig, axes = plt.subplots(len(plot_components), 1, figsize=(12, 6), sharex=True)
        if len(plot_components) == 1:
            axes = [axes]

        origin_dt = origin_time.datetime
        if origin_dt.tzinfo is None:
            origin_dt = origin_dt.replace(tzinfo=timezone.utc)

        event_times = []
        for _, row in catalog.iterrows():
            if str(row.get("skip", "0")) == "1":
                continue
            try:
                evt = UTCDateTime(str(row[ORIGIN_COL])).datetime
                if evt.tzinfo is None:
                    evt = evt.replace(tzinfo=timezone.utc)
                event_times.append(evt)
            except Exception:
                continue

        for idx, comp in enumerate(plot_components):
            ax = axes[idx]
            tr = traces.get(comp)
            if tr is None:
                ax.set_axis_off()
                continue

            t0 = tr.stats.starttime
            npts = tr.stats.npts
            sr = float(tr.stats.sampling_rate)
            t_abs = [
                (t0 + (i / sr)).datetime.replace(tzinfo=timezone.utc)
                for i in range(npts)
            ]

            ax.plot(t_abs, tr.data, lw=1.0, color="k")
            ax.axvline(origin_dt, color="r", lw=1.2, alpha=0.7, linestyle="--")
            for evt in event_times:
                ax.axvline(evt, color="0.5", lw=0.8, alpha=0.6)
            ax.set_ylabel(comp)
            ax.grid(alpha=0.2)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        axes[-1].xaxis.set_minor_formatter(NullFormatter())
        axes[-1].set_xlabel(f"UTC time (origin {origin_dt.isoformat()})")

    def _apply_round_ticks(ax) -> None:
        xmin, xmax = ax.get_xlim()
        span_seconds = max(1.0, (xmax - xmin) * 86400.0)
        span_minutes = span_seconds / 60.0
        target_ticks = 10.0
        raw_interval = span_minutes / target_ticks
        candidates = [0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120, 180, 240, 360]
        interval = min(candidates, key=lambda v: abs(v - raw_interval))
        if interval < 1:
            seconds = int(round(interval * 60.0))
            major_seconds = max(1, seconds)
            ax.xaxis.set_major_locator(mdates.SecondLocator(interval=major_seconds))
            ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=max(1, major_seconds // 5)))
        else:
            major_minutes = int(round(interval))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=major_minutes))
            if major_minutes >= 5:
                ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=max(1, major_minutes // 5)))
            else:
                ax.xaxis.set_minor_locator(mdates.SecondLocator(interval=max(1, int(major_minutes * 60 / 5))))

    if show_original and tmin is not None and tmax is not None:
        tmin_dt = tmin.datetime.replace(tzinfo=timezone.utc)
        tmax_dt = tmax.datetime.replace(tzinfo=timezone.utc)
        axes[-1].set_xlim(tmin_dt, tmax_dt)
        _apply_round_ticks(axes[-1])
        def _on_view_change(_event=None) -> None:
            _apply_round_ticks(axes[-1])
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_release_event", _on_view_change)
        axes[-1].callbacks.connect("xlim_changed", _on_view_change)

    if show_original:
        axes[-1].tick_params(axis="x", which="major", length=6)
        axes[-1].tick_params(axis="x", which="minor", length=3)
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.autofmt_xdate()

    # Overlay: normalized components, 0-30 s after each event within data window
    def _shift_zeropad(x: np.ndarray, n: int) -> np.ndarray:
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

    if tmin is not None and tmax is not None:
        tmin_evt = tmin
        tmax_evt = tmax - EVENT_WINDOW_END
        mag_col = None
        if "magnitude" in catalog_all.columns:
            mag_col = "magnitude"
        elif "mag" in catalog_all.columns:
            mag_col = "mag"
        event_info = []
        for _, row in catalog_all.iterrows():
            try:
                evt_time = UTCDateTime(str(row[ORIGIN_COL]))
            except Exception:
                continue
            mag_val = None
            if mag_col is not None and mag_col in row:
                try:
                    mag_val = float(row[mag_col])
                except Exception:
                    mag_val = None
            event_info.append((evt_time, mag_val))

        fig_seg, axes_seg = plt.subplots(
            len(plot_components),
            1,
            figsize=(10, 6.5 if len(plot_components) == 1 else 8.5),
            sharex=True,
        )
        if len(plot_components) == 1:
            axes_seg = [axes_seg]
        comp_order = plot_components
        counts = {c: 0 for c in comp_order}

        mag_col = None
        if "magnitude" in catalog.columns:
            mag_col = "magnitude"
        elif "mag" in catalog.columns:
            mag_col = "mag"

        processed_by_comp = {comp: [] for comp in comp_order}
        processed_pre_mask = {comp: [] for comp in comp_order}

        for comp, ax in zip(comp_order, axes_seg):
            tr_c = traces.get(comp)
            if tr_c is None:
                ax.set_axis_off()
                continue

            segments = []
            legend_labels = []
            segment_times = []
            segment_mags = []
            for _, row in catalog.iterrows():
                if str(row.get("skip", "0")) == "1":
                    continue
                try:
                    evt = UTCDateTime(str(row[ORIGIN_COL]))
                except Exception:
                    continue
                if evt < tmin_evt or evt > tmax_evt:
                    continue

                seg = tr_c.slice(
                    starttime=evt + EVENT_WINDOW_START,
                    endtime=evt + EVENT_WINDOW_END,
                )
                if seg.stats.npts == 0:
                    continue
                segments.append(seg)
                segment_times.append(evt)

                evid = str(row.get("evid", ""))
                mag_val = None
                if mag_col is not None and mag_col in row:
                    try:
                        mag_val = float(row[mag_col])
                    except Exception:
                        mag_val = None
                if SHOW_LEGEND:
                    if mag_val is not None:
                        legend_labels.append(f"{evid} M{mag_val:.2f}")
                    else:
                        legend_labels.append(f"{evid}")
                segment_mags.append(mag_val)

            if segments:
                ref = segments[0].data.astype(float)
                sr = float(segments[0].stats.sampling_rate)
                w0 = int(round(ALIGN_WINDOW_START * sr))
                w1 = int(round(ALIGN_WINDOW_END * sr))
                ref_win = ref[w0:w1]
                ref_win = ref_win - np.mean(ref_win) if ref_win.size > 0 else ref_win
                ref_den = np.linalg.norm(ref_win) if ref_win.size > 0 else 0.0
                if ref_den > 0:
                    ref_win = ref_win / ref_den

                if not SHOW_LEGEND:
                    legend_labels = ["" for _ in segments]
                for seg, label, evt_time, evt_mag in zip(
                    segments,
                    legend_labels,
                    segment_times,
                    segment_mags,
                ):
                    y = seg.data.astype(float)
                    y_win = y[w0:w1]
                    if ref_win.size > 0 and y_win.size == ref_win.size:
                        y_win = y_win - np.mean(y_win)
                        y_den = np.linalg.norm(y_win)
                        if y_den > 0:
                            y_win = y_win / y_den
                            corr = np.correlate(y_win, ref_win, mode="full")
                            lag = int(np.argmax(corr) - (y_win.size - 1))
                            y = _shift_zeropad(y, lag)

                    mx = float(max(abs(y.min()), abs(y.max()))) if y.size > 0 else 0.0
                    if mx > 0:
                        y = y / mx

                    dt = float(seg.stats.delta)
                    y_pre = y.copy()
                    for other_evt, other_mag in event_info:
                        offset = float(other_evt - evt_time)
                        if abs(offset) < 1e-6:
                            continue
                        if offset < -ALIGN_WINDOW_END or offset > EVENT_WINDOW_END - ALIGN_WINDOW_START:
                            continue
                        if (
                            evt_mag is None
                            or other_mag is None
                               or other_mag < (evt_mag - MAG_DIFF_MIN)
                        ):
                            continue
                        t0 = offset + ALIGN_WINDOW_START - EVENT_WINDOW_START
                        t1 = offset + ALIGN_WINDOW_END - EVENT_WINDOW_START
                        i0 = int(round(t0 / dt))
                        i1 = int(round(t1 / dt))
                        if i1 < 0 or i0 >= y.size:
                            continue
                        i0 = max(0, i0)
                        i1 = min(y.size - 1, i1)
                        if i1 >= i0:
                            y[i0 : i1 + 1] = REPLACE_LEVEL

                    t_rel = [seg.stats.delta * i for i in range(seg.stats.npts)]
                    processed_pre_mask[comp].append((t_rel, y_pre, evt_time))
                    processed_by_comp[comp].append((t_rel, y, evt_time))
                    ax.plot(t_rel, y, lw=1.0, alpha=0.6, label=label)
                    counts[comp] += 1

                if processed_by_comp[comp]:
                    t_rel = processed_by_comp[comp][0][0]
                    y_stack = np.vstack(
                        [np.asarray(t[1], dtype=float) for t in processed_by_comp[comp]]
                    )
                    y_med = np.nanmedian(y_stack, axis=0)
                    ax.plot(t_rel, y_med, lw=2.4, color="k", label="Median")

            ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
            ax.set_ylabel(COMPONENT_LABELS.get(comp, comp))
            ax.grid(alpha=0.2)
            if SHOW_LEGEND:
                ax.legend(loc="upper right", fontsize=7, ncol=1)

        axes_seg[-1].set_xlim(EVENT_WINDOW_START, EVENT_WINDOW_END)
        axes_seg[-1].set_xlabel("Time since origin (s)")
        fig_seg.suptitle(
            f"Stacks: {EVENT_WINDOW_START:.1f}–{EVENT_WINDOW_END:.1f} s after events",
            fontsize=12,
            fontweight="bold",
        )
        fig_seg.tight_layout()
        out_file = OUTPUT_ROOT / "stack_segments_overlay.png"
        fig_seg.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"✓ Wrote plot: {out_file}")

        if SHOW_MEDIAN_TRACE:
            fig_med, axes_med = plt.subplots(
                len(plot_components),
                1,
                figsize=(10, 6.5 if len(plot_components) == 1 else 8.5),
                sharex=True,
            )
            if len(plot_components) == 1:
                axes_med = [axes_med]

            for comp, ax in zip(plot_components, axes_med):
                traces_list = processed_by_comp.get(comp, [])
                if len(traces_list) == 0:
                    ax.set_axis_off()
                    continue

                t_rel = traces_list[0][0]
                y_stack = np.vstack([np.asarray(t[1], dtype=float) for t in traces_list])
                y_med = np.nanmedian(y_stack, axis=0)

                ax.plot(t_rel, y_med, lw=1.4, color="k")
                ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
                ax.set_ylabel(COMPONENT_LABELS.get(comp, comp))
                ax.grid(alpha=0.2)

            axes_med[-1].set_xlim(EVENT_WINDOW_START, EVENT_WINDOW_END)
            axes_med[-1].set_xlabel("Time since origin (s)")
            fig_med.suptitle(
                "Median of selected traces (post-mask)",
                fontsize=12,
                fontweight="bold",
            )
            fig_med.tight_layout()
            out_file = OUTPUT_ROOT / "stack_segments_median.png"
            fig_med.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"✓ Wrote plot: {out_file}")

        if SHOW_CHOPPED_PLOT:
            fig_chop, axes_chop = plt.subplots(
                len(plot_components),
                1,
                figsize=(10, 6.5 if len(plot_components) == 1 else 8.5),
                sharex=True,
            )
            if len(plot_components) == 1:
                axes_chop = [axes_chop]

            for comp, ax in zip(plot_components, axes_chop):
                tr_c = traces.get(comp)
                if tr_c is None:
                    ax.set_axis_off()
                    continue

                for _, row in catalog.iterrows():
                    if str(row.get("skip", "0")) == "1":
                        continue
                    try:
                        evt = UTCDateTime(str(row[ORIGIN_COL]))
                    except Exception:
                        continue
                    if evt < tmin_evt or evt > tmax_evt:
                        continue

                    seg = tr_c.slice(
                        starttime=evt + EVENT_WINDOW_START,
                        endtime=evt + EVENT_WINDOW_END,
                    )
                    if seg.stats.npts == 0:
                        continue
                    y = seg.data.astype(float)
                    mx = float(max(abs(y.min()), abs(y.max()))) if y.size > 0 else 0.0
                    if mx > 0:
                        y = y / mx
                    t_rel = [seg.stats.delta * i for i in range(seg.stats.npts)]
                    ax.plot(t_rel, y, lw=0.9, alpha=0.5)

                ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
                ax.set_ylabel(COMPONENT_LABELS.get(comp, comp))
                ax.grid(alpha=0.2)

            axes_chop[-1].set_xlim(EVENT_WINDOW_START, EVENT_WINDOW_END)
            axes_chop[-1].set_xlabel("Time since origin (s)")
            fig_chop.suptitle(
                "Chopped window (normalized)",
                fontsize=12,
                fontweight="bold",
            )
            fig_chop.tight_layout()
            out_file = OUTPUT_ROOT / "stack_segments_chopped.png"
            fig_chop.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"✓ Wrote plot: {out_file}")

        if SHOW_OFFSET_TRACES:
            fig_off, axes_off = plt.subplots(
                len(plot_components),
                1,
                figsize=(10, 6.5 if len(plot_components) == 1 else 8.5),
                sharex=True,
            )
            if len(plot_components) == 1:
                axes_off = [axes_off]

            for comp, ax in zip(plot_components, axes_off):
                if comp not in processed_pre_mask or len(processed_pre_mask[comp]) == 0:
                    ax.set_axis_off()
                    continue

                offset = 0.0
                sorted_pre = sorted(processed_pre_mask[comp], key=lambda t: t[2])
                for t_rel, y, evt_time in reversed(sorted_pre):
                    ax.plot(t_rel, y + offset, lw=0.9, alpha=0.7)
                    ax.text(
                        t_rel[0],
                        y[0] + offset,
                        evt_time.datetime.strftime("%H:%M:%S"),
                        fontsize=6,
                        va="bottom",
                    )
                    offset += OFFSET_STEP

                ax.set_ylabel(COMPONENT_LABELS.get(comp, comp))
                ax.grid(alpha=0.2)

            axes_off[-1].set_xlim(EVENT_WINDOW_START, EVENT_WINDOW_END)
            axes_off[-1].set_xlabel("Time since origin (s)")
            fig_off.suptitle(
                "Chopped window (offset traces, pre-mask)",
                fontsize=12,
                fontweight="bold",
            )
            fig_off.tight_layout()
            out_file = OUTPUT_ROOT / "stack_segments_offset_pre_mask.png"
            fig_off.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"✓ Wrote plot: {out_file}")

            fig_off2, axes_off2 = plt.subplots(
                len(plot_components),
                1,
                figsize=(10, 6.5 if len(plot_components) == 1 else 8.5),
                sharex=True,
            )
            if len(plot_components) == 1:
                axes_off2 = [axes_off2]

            for comp, ax in zip(plot_components, axes_off2):
                if comp not in processed_by_comp or len(processed_by_comp[comp]) == 0:
                    ax.set_axis_off()
                    continue

                offset = 0.0
                sorted_post = sorted(processed_by_comp[comp], key=lambda t: t[2])
                for t_rel, y, evt_time in reversed(sorted_post):
                    y = np.asarray(y, dtype=float)
                    mx = float(np.max(np.abs(y))) if y.size > 0 else 0.0
                    if mx > 0:
                        y = y / mx
                    ax.plot(t_rel, y + offset, lw=0.9, alpha=0.7)
                    ax.text(
                        t_rel[0],
                        y[0] + offset,
                        evt_time.datetime.strftime("%H:%M:%S"),
                        fontsize=6,
                        va="bottom",
                    )
                    offset += OFFSET_STEP

                ax.set_ylabel(COMPONENT_LABELS.get(comp, comp))
                ax.grid(alpha=0.2)

            axes_off2[-1].set_xlim(EVENT_WINDOW_START, EVENT_WINDOW_END)
            axes_off2[-1].set_xlabel("Time since origin (s)")
            fig_off2.suptitle(
                "Chopped window (offset traces, post-mask)",
                fontsize=12,
                fontweight="bold",
            )
            fig_off2.tight_layout()
            out_file = OUTPUT_ROOT / "stack_segments_offset_post_mask.png"
            fig_off2.savefig(out_file, dpi=300, bbox_inches="tight")
            print(f"✓ Wrote plot: {out_file}")

    plt.show()


def main() -> None:
    catalog_all = load_catalog(CATALOG_FILE)
    if catalog_all.empty:
        raise ValueError("Catalog is empty")

    mag_col = None
    if "magnitude" in catalog_all.columns:
        mag_col = "magnitude"
    elif "mag" in catalog_all.columns:
        mag_col = "mag"
    if mag_col is None:
        raise ValueError("Catalog missing magnitude column (magnitude or mag)")

    catalog = catalog_all[catalog_all[mag_col].astype(float) > MAG_MIN].reset_index(drop=True)
    if catalog.empty:
        raise ValueError(f"No events with magnitude > {MAG_MIN}")

    first_row = catalog.iloc[0]
    event_id = str(first_row["evid"])
    origin_time = UTCDateTime(str(first_row[ORIGIN_COL]))

    traces = load_stack_traces(OUTPUT_ROOT)
    plot_stacks(
        origin_time,
        traces,
        f"Event {event_id} stacked components",
        catalog,
        catalog_all,
        SHOW_ORIGINAL_PLOT,
    )


if __name__ == "__main__":
    main()
