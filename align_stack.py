import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import cast
from matplotlib.lines import Line2D
from pathlib import Path
from datetime import timezone
import time

from obspy import UTCDateTime, Stream, Trace
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel
from scipy.signal import hilbert
from scipy.signal.windows import gaussian

from align_utils import (
    add_catalog_event_lines,
    add_stage_timing,
    add_utc_time_axis,
    build_alignment_products_payload,
    build_component_output_payload,
    compute_phase_travel_times,
    compute_stage1_aligned_stack,
    compute_stage2_screened_stack,
    compute_stage3_finalized_rows,
    compute_taup_station_shifts,
    compute_time_axis_and_stack,
    correlation_time_bounds,
    draw_correlation_markers,
    ensure_utc_datetime,
    get_component_selection,
    load_event_metadata,
    load_station_lookup,
    make_event_output_dir,
    compute_alignment_setup,
    normalize_traces_in_window,
    report_timing_once,
    read_waveforms_for_event,
    rotate_horizontals_to_component,
    resolve_component_key,
    select_reference_trace,
    print_reference_summary,
    preprocess_traces_bandpass,
    set_figure_title,
    TimingState,
    write_component_stack_mseeds,
)

min_freq, max_freq            = 3.0, 10.0 # Bandpass filter (Hz)
start_time, end_time          = -10.0, 20 # Plotting time window (seconds since origin)
# start_time, end_time          = -1990.0, 3690.0 # Plotting time window (seconds since origin)
# start_time, end_time          = -1990.0, 15000 # Plotting time window (seconds since origin)
win_pre, win_post             = 0.5,  0.5 # Correlation window parameters (seconds)
r_window_min                  = 0.6       # Minimum correlation coefficient for trace selection
move_limit_sec                = 0.05      # Maximum allowed shift (seconds) searched in compute_lag

# Run modes
all_channels = True  # If True to process all channels
component   = "R"       # Component selection: 'Z', 'R', or 'T'
align_phase = "S"       # Alignment phase 'P' or 'S'
verbose     = False     # If True, print detailed processing info

# Paths
path_prefix = "/Users/jvidale/Documents/Research/FaultScanR/"
info_root = Path(path_prefix + "20220930_events_cut/event_sta_info")

# sps_rate = "down50"  # Subdirectory name indicating the sampling rate of the data (e.g., "down50", "down100", etc.)
# data_path = Path(path_prefix + "20220930_events_cut/07_1hour_20220930")

sps_rate = "down100"
data_path = Path(path_prefix + "20220930_events_cut/20220930_" + sps_rate)

event       = "CI_40353544" # Single run selection (used when the corresponding "all_*" is False)
# event       = "CI_40353664" # Single run selection (used when the corresponding "all_*" is False)
events = [event]        # Allows for future modification to process multiple events

# plotting options (user-facing)
show_individual_seismograms = False  # Plot individual seismograms (20 traces/plot, 5 panels/figure)
show_record_section_plot = False  # Show aligned record sections (single + 3-comp)


# Timing (cpu and wall)
timing_state = TimingState()


catalog_local_file = info_root / "catalog_local_hand.xlsx"
catalog_local = None
try:
    if catalog_local_file.exists():
        catalog_local = pd.read_excel(catalog_local_file)
        print(f"Loaded catalog: {catalog_local_file}")
    else:
        print(f"[WARN] Catalog not found: {catalog_local_file}")
except Exception as e:
    print(f"[WARN] Failed to read catalog: {catalog_local_file} ({e})")

# Travel-time model
model = TauPyModel(model="iasp91")

# ===================== Helper functions =====================


def plot_stage_stacks(
    eve_id: str,
    plot_comp: str,
    align_phase_name: str,
    t_abs: np.ndarray,
    mask: np.ndarray,
    aligned_stack: np.ndarray,
    selected_aligned_stack: np.ndarray,
    stack_vec: np.ndarray,
    save_dir: Path,
) -> None:
    """Plot and save Stage-1/Stage-2/Final stacks for single-component runs."""
    fig_stk, ax_stk = plt.subplots(1, 1, figsize=(10, 3.8))
    set_figure_title(fig_stk, f"{eve_id} {plot_comp} stage stacks")
    ax_stk.plot(t_abs[mask], aligned_stack[mask], color="C0", lw=2, label="Stage 1: aligned_stack")
    ax_stk.plot(
        t_abs[mask],
        selected_aligned_stack[mask],
        color="C1",
        lw=2,
        label="Stage 2: selected_aligned_stack",
    )
    ax_stk.plot(t_abs[mask], stack_vec[mask], color="C3", lw=2.2, label="Final stack")
    ax_stk.axhline(0.0, color="k", lw=0.6, alpha=0.6)
    ax_stk.set_xlim(start_time, end_time)
    ax_stk.set_ylim(-1.1, 1.1)
    ax_stk.grid(alpha=0.2)
    ax_stk.set_xlabel("Time since origin (s)")
    ax_stk.set_ylabel("Stack (norm.)")
    ax_stk.set_title(f"Event {eve_id} {plot_comp}: Stage-1/Stage-2/Final stacks")
    ax_stk.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    stack_file = save_dir / f"{eve_id}_{plot_comp}_stage_stacks_{align_phase_name}.png"
    fig_stk.savefig(stack_file, dpi=300, bbox_inches="tight")
    print(f"✓ Stage stacks plot saved to: {stack_file}")


def plot_record_section_and_stack(
    show_record: bool,
    eve_id: str,
    plot_comp: str,
    align_phase_name: str,
    selected_rows: list,
    rejected_rows: list,
    t_abs: np.ndarray,
    mask: np.ndarray,
    sample_rate: float,
    t_ref,
    win_start: int,
    win_end: int,
    move_sec: float,
    npts: int,
    n_pass_window: int,
    stack_vec: np.ndarray,
    save_dir: Path,
):
    """Plot and save record section (top) plus normalized stack (bottom)."""
    if not show_record:
        return None

    fig, (ax, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 9),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    set_figure_title(fig, f"{eve_id} {plot_comp} aligned record section")

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
    ax.set_title(f"Aligned {align_phase_name} waveforms Event {eve_id} comp = {plot_comp}")
    ax.grid(alpha=0.2)

    # Theoretical arrival time (reference station) as a vertical reference line
    try:
        if t_ref is not None:
            for axi in (ax, ax2):
                axi.axvline(x=t_ref, color="k", lw=3, alpha=0.5, zorder=6)
    except Exception as e:
        print(f"    [WARN] Failed to draw vertical reference arrival for {align_phase_name}: {e}")

    # Correlation window and search bounds
    try:
        for axi in (ax, ax2):
            draw_correlation_markers(
                axi,
                start_time,
                win_start,
                win_end,
                sample_rate,
                move_sec,
                npts,
            )
    except Exception as e:
        print(f"    [WARN] Failed to draw correlation window bounds: {e}")

    try:
        legend_handles = [
            Line2D([0], [0], color="y", lw=2, label="Correlation window"),
            Line2D([0], [0], color="g", lw=2, label="Correlation search (±move_limit_sec)"),
            Line2D([0], [0], color="none", label=f"Pass r_win: {n_pass_window}"),
        ]
        ax.legend(
            handles=legend_handles,
            loc="upper left",
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
    ax2.set_title("Final stack uses ALL traces (no screening)")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    record_file = save_dir / f"{eve_id}_{plot_comp}_{align_phase_name}.png"
    fig.savefig(record_file, dpi=300, bbox_inches="tight")
    print(f"✓ Record-section plot saved to: {record_file}")
    return fig


def plot_three_component_log_envelope(
    comp_order: list,
    stack_by_comp: dict,
    sample_rate_env: float,
    t_abs: np.ndarray,
    mask: np.ndarray,
    start_time: float,
    end_time: float,
    eve_id: str,
    align_phase_name: str,
    save_dir: Path,
    origin_env,
    catalog_df,
) -> None:
    """Plot and save log10 RMS envelope for combined three-component stacks."""
    try:
        if all(comp in stack_by_comp for comp in comp_order):
            z = stack_by_comp["DPZ"]
            r = stack_by_comp["R"]
            t = stack_by_comp["T"]
            env_z = np.abs(cast(np.ndarray, hilbert(z)))
            env_r = np.abs(cast(np.ndarray, hilbert(r)))
            env_t = np.abs(cast(np.ndarray, hilbert(t)))
            env_rms = np.sqrt((env_z ** 2 + env_r ** 2 + env_t ** 2) / 3.0)
            std_sec = 1.0
            std_samples = max(1.0, float(sample_rate_env) * std_sec)
            win_samples = max(3, int(round(6.0 * std_samples)))
            gauss = gaussian(win_samples, std_samples)
            gauss = gauss / np.sum(gauss)
            env_rms_smooth = np.convolve(env_rms, gauss, mode="same")
            log_env = np.log10(np.maximum(env_rms_smooth, 1e-12))

            fig_env, ax_env = plt.subplots(figsize=(12, 4.5))
            set_figure_title(fig_env, f"{eve_id} 3-comp log10 envelope")
            ax_env.plot(t_abs[mask], log_env[mask], color="k", lw=1.5)
            ax_env.set_xlim(start_time, end_time)
            ax_env.set_xlabel("Time since origin (s)", fontsize=11)
            ax_env.set_ylabel("log10 envelope", fontsize=11)
            ax_env.set_title(
                f"Event {eve_id} - log10 RMS envelope of 3-component stack",
                fontsize=12,
                fontweight="bold",
            )
            ax_env.grid(alpha=0.2)
            add_catalog_event_lines(ax_env, origin_env, catalog_df, start_time, end_time)
            fig_env.subplots_adjust(bottom=0.28)

            if origin_env is not None:
                try:
                    add_utc_time_axis(ax_env, origin_env)
                except Exception as e:
                    print(f"[WARN] Failed to add UTC time axis (envelope): {e}")

            env_file = save_dir / f"{eve_id}_3comp_log10_envelope_{align_phase_name}.png"
            fig_env.savefig(env_file, dpi=300, bbox_inches="tight")
            print(f"✓ Log10 envelope plot saved to: {env_file}")
    except Exception as e:
        print(f"[WARN] Failed to create log10 envelope plot: {e}")


def plot_single_trace_log_envelope(
    num_traces: int,
    stack_vec: np.ndarray,
    sample_rate: float,
    t_abs: np.ndarray,
    mask: np.ndarray,
    start_time: float,
    end_time: float,
    eve_id: str,
    plot_comp: str,
    align_phase_name: str,
    save_dir: Path,
    origin,
    catalog_df,
) -> None:
    """Plot and save log10 envelope when there is a single trace."""
    if num_traces != 1:
        return
    try:
        env = np.abs(cast(np.ndarray, hilbert(stack_vec)))
        std_sec = 1.0
        std_samples = max(1.0, float(sample_rate) * std_sec)
        win_samples = max(3, int(round(6.0 * std_samples)))
        gauss = gaussian(win_samples, std_samples)
        gauss = gauss / np.sum(gauss)
        env_smooth = np.convolve(env, gauss, mode="same")
        log_env = np.log10(np.maximum(env_smooth, 1e-12))

        fig_env, ax_env = plt.subplots(figsize=(12, 4.5))
        set_figure_title(fig_env, f"{eve_id} {plot_comp} log10 envelope")
        ax_env.plot(t_abs[mask], log_env[mask], color="k", lw=1.5)
        ax_env.set_xlim(start_time, end_time)
        ax_env.set_xlabel("Time since origin (s)", fontsize=11)
        ax_env.set_ylabel("log10 envelope", fontsize=11)
        ax_env.set_title(
            f"Event {eve_id} - log10 envelope ({plot_comp})",
            fontsize=12,
            fontweight="bold",
        )
        ax_env.grid(alpha=0.2)
        add_catalog_event_lines(ax_env, origin, catalog_df, start_time, end_time)
        fig_env.subplots_adjust(bottom=0.28)

        if origin is not None:
            try:
                add_utc_time_axis(ax_env, origin)
            except Exception as e:
                print(f"[WARN] Failed to add UTC time axis (single envelope): {e}")

        env_file = save_dir / f"{eve_id}_{plot_comp}_log10_envelope_{align_phase_name}.png"
        fig_env.savefig(env_file, dpi=300, bbox_inches="tight")
        print(f"✓ Log10 envelope plot saved to: {env_file}")
    except Exception as e:
        print(f"[WARN] Failed to create log10 envelope plot (single trace): {e}")


def plot_estimated_vs_calculated_shifts(
    calc_shifts: dict,
    station_shifts: dict,
    eve_id: str,
    plot_comp: str,
    align_phase_name: str,
    save_dir: Path,
) -> None:
    """Plot estimated shift versus TauP-calculated shift for stations with both values."""
    common_sta = set(calc_shifts.keys()) & set(station_shifts.keys())
    if len(common_sta) == 0:
        print("[WARN] No stations with both estimated and calculated shifts for comparison.")
        return

    stations = sorted(common_sta, key=lambda s: int(s))
    est_shift = np.array([station_shifts[s]["lag_seconds"] for s in stations], dtype=float)
    calc_shift = np.array([calc_shifts[s] for s in stations], dtype=float)

    fig_ec, ax_ec = plt.subplots(1, 1, figsize=(6.2, 5.2))
    set_figure_title(fig_ec, f"{eve_id} {plot_comp} est vs calc shifts")
    ax_ec.scatter(calc_shift, est_shift, s=20, alpha=0.6)

    minv = float(min(np.min(calc_shift), np.min(est_shift)))
    maxv = float(max(np.max(calc_shift), np.max(est_shift)))
    ax_ec.plot([minv, maxv], [minv, maxv], "r--", lw=1.2, alpha=0.7, label="1:1 line")

    ax_ec.set_xlabel("Calculated shift (s)")
    ax_ec.set_ylabel("Estimated shift (s)")
    ax_ec.set_title(f"Event {eve_id} {plot_comp}: Estimated vs Calculated shifts")
    ax_ec.grid(alpha=0.3)
    ax_ec.legend(loc="upper left", fontsize=9)
    plt.tight_layout()

    estcalc_file = save_dir / f"{eve_id}_{plot_comp}_est_vs_calc_shift_{align_phase_name}.png"
    fig_ec.savefig(estcalc_file, dpi=300, bbox_inches="tight")
    print(f"✓ Estimated vs calculated shift plot saved to: {estcalc_file}")


def plot_snippet_comparison(
    start_time: float,
    win_start: int,
    win_end: int,
    sample_rate: float,
    ref_window: np.ndarray,
    pass_window_ids: set,
    snippet_by_station: dict,
    eve_id: str,
    plot_comp: str,
    align_phase_name: str,
    save_dir: Path,
) -> None:
    """Plot pass/fail correlation-window snippets against the reference window."""
    try:
        t_win = start_time + (np.arange(win_start, win_end) / sample_rate)
        pass_list = sorted(list(pass_window_ids), key=lambda s: int(s))
        fail_list = sorted(
            [s for s in snippet_by_station.keys() if s not in pass_window_ids],
            key=lambda s: int(s),
        )

        n_show = 10
        pass_show = pass_list[:n_show]
        fail_show = fail_list[:n_show]

        fig_snip, (axp, axf) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        set_figure_title(fig_snip, f"{eve_id} {plot_comp} correlation snippets")

        for sid in pass_show:
            axp.plot(t_win, snippet_by_station[sid], color="k", alpha=0.4, lw=1)
        axp.plot(t_win, ref_window, color="C3", lw=2, label="Ref window")
        axp.set_title(f"Pass r_win (N={len(pass_list)})")
        axp.set_xlabel("Time since origin (s)")
        axp.grid(alpha=0.3)

        for sid in fail_show:
            axf.plot(t_win, snippet_by_station[sid], color="k", alpha=0.4, lw=1)
        axf.plot(t_win, ref_window, color="C3", lw=2, label="Ref window")
        axf.set_title(f"Fail r_win (N={len(fail_list)})")
        axf.set_xlabel("Time since origin (s)")
        axf.grid(alpha=0.3)

        axp.set_ylabel("Normalized amplitude")
        axf.legend(loc="upper right", fontsize=8)
        fig_snip.suptitle(
            f"Event {eve_id} {plot_comp}: correlation-window snippets",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()

        snip_file = save_dir / f"{eve_id}_{plot_comp}_snippet_compare_{align_phase_name}.png"
        fig_snip.savefig(snip_file, dpi=300, bbox_inches="tight")
        print(f"✓ Snippet comparison plot saved to: {snip_file}")
    except Exception as e:
        print(f"[WARN] Failed to create snippet comparison plot: {e}")


def plot_station_pass_map(
    aligned_traces_by_station: dict,
    pass_window_ids: set,
    name2ll: dict,
    eve_id: str,
    plot_comp: str,
    align_phase_name: str,
    save_dir: Path,
) -> None:
    """Plot station locations colored by pass/fail screening status."""
    try:
        all_stations = sorted(aligned_traces_by_station.keys(), key=lambda s: int(s))
        pass_win = set(pass_window_ids)

        fig_map, axm = plt.subplots(1, 1, figsize=(6.5, 5.5))
        set_figure_title(fig_map, f"{eve_id} {plot_comp} station pass map")
        pass_lats = [name2ll[s][0] for s in all_stations if s in pass_win]
        pass_lons = [name2ll[s][1] for s in all_stations if s in pass_win]
        fail_lats = [name2ll[s][0] for s in all_stations if s not in pass_win]
        fail_lons = [name2ll[s][1] for s in all_stations if s not in pass_win]

        if len(fail_lons) > 0:
            axm.scatter(fail_lons, fail_lats, s=18, c="0.7", label="Fail")
        if len(pass_lons) > 0:
            axm.scatter(pass_lons, pass_lats, s=22, c="C3", label="Pass")

        axm.set_title("Pass r_win", fontsize=11, fontweight="bold")
        axm.grid(alpha=0.3)
        axm.set_xlabel("Longitude")
        axm.set_ylabel("Latitude")
        axm.legend(loc="upper right", fontsize=9)

        fig_map.suptitle(
            f"Event {eve_id} {plot_comp}: stations passing thresholds",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        map_file = save_dir / f"{eve_id}_{plot_comp}_station_pass_map_{align_phase_name}.png"
        fig_map.savefig(map_file, dpi=300, bbox_inches="tight")
        print(f"✓ Station pass/fail map saved to: {map_file}")
    except Exception as e:
        print(f"[WARN] Failed to create station pass/fail maps: {e}")


def plot_individual_seismograms_single_component(
    show_individual_seismograms: bool,
    selected_rows: list,
    rejected_rows: list,
    pass_window_ids: set,
    t_abs: np.ndarray,
    mask: np.ndarray,
    stack_vec: np.ndarray,
    start_time: float,
    win_start: int,
    win_end: int,
    sample_rate: float,
    move_limit_sec: float,
    npts: int,
    eve_id: str,
    plot_comp: str,
    align_phase_name: str,
    save_dir: Path,
) -> None:
    """Plot individual seismograms in paged panels for single-component mode."""
    if not show_individual_seismograms:
        return
    try:
        all_rows = selected_rows + rejected_rows
        all_rows.sort(key=lambda t: int(t[1]))

        n_traces = len(all_rows)
        if n_traces <= 0:
            return

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
            set_figure_title(
                fig_ind,
                f"{eve_id} {plot_comp} individual seismograms fig {fig_idx + 1}",
            )
            if panels_in_fig == 1:
                axes_ind = [axes_ind]

            for p in range(panels_in_fig):
                axp = axes_ind[p]
                global_panel = panel_start + p
                start_idx = global_panel * n_per
                end_idx = min((global_panel + 1) * n_per, n_traces)
                subset = all_rows[start_idx:end_idx]

                t_win_start = start_time + (win_start / sample_rate)
                t_win_end = start_time + (win_end / sample_rate)
                t_explore_start = max(start_time, t_win_start - move_limit_sec)
                t_explore_end = min(start_time + (npts / sample_rate), t_win_end + move_limit_sec)
                axp.axvline(x=t_win_start, color="y", lw=1.2, alpha=0.9)
                axp.axvline(x=t_win_end, color="y", lw=1.2, alpha=0.9)
                axp.axvline(x=t_explore_start, color="g", lw=1.2, alpha=0.9)
                axp.axvline(x=t_explore_end, color="g", lw=1.2, alpha=0.9)

                for idx_in_subset, (_, station_id, y) in enumerate(subset):
                    i = (len(subset) - 1) - idx_in_subset
                    passed_win = station_id in pass_window_ids
                    trace_color = "k" if passed_win else "red"
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
                        va="center",
                    )

                ref_offset = len(subset) + 1
                axp.plot(
                    t_abs[mask],
                    stack_vec[mask] + ref_offset,
                    color="C3",
                    lw=1.2,
                )

                axp.set_ylim(-1, len(subset) + 2)
                axp.grid(alpha=0.2)
                axp.set_ylabel("Trace index")

            axes_ind[-1].set_xlabel("Time since origin (s)")
            fig_ind.suptitle(
                f"Event {eve_id} {plot_comp}: individual seismograms "
                f"(20 per panel, fig {fig_idx + 1}/{n_figs})",
                fontsize=12,
                fontweight="bold",
            )
            plt.tight_layout()

            ind_file = save_dir / (
                f"{eve_id}_{plot_comp}_individual_seismograms_{align_phase_name}_fig{fig_idx + 1}.png"
            )
            fig_ind.savefig(ind_file, dpi=300, bbox_inches="tight")
            print(f"✓ Individual seismograms plot saved to: {ind_file}")
    except Exception as e:
        print(f"[WARN] Failed to create individual seismograms plot: {e}")


def plot_three_component_stack_compare(
    all_component_data: dict,
    eve_id: str,
    align_phase_name: str,
    save_dir: Path,
) -> None:
    """Plot black/red stack comparison for Z/R/T components and save figure."""
    print("Creating stack comparison plot (all aligned vs r_min-selected)...")

    fig_cmp, axes_cmp = plt.subplots(3, 1, figsize=(9, 12), sharex=True, sharey=True)
    set_figure_title(fig_cmp, f"{eve_id} stack compare")
    comp_order = ["DPZ", "R", "T"]
    comp_titles_cmp = ["Z stack", "R stack", "T stack"]
    utc_tz = timezone.utc

    for j, comp_name in enumerate(comp_order):
        axc = axes_cmp[j]
        if comp_name not in all_component_data:
            axc.set_axis_off()
            continue

        data = all_component_data[comp_name]
        t_abs = data["t_abs"]
        mask = data["mask"]
        start_time = data["start_time"]
        end_time = data["end_time"]
        p_time = data.get("p_traveltime")
        s_time = data.get("s_traveltime")

        tr_map = data.get("aligned_traces_by_station", {})
        all_stations = sorted(tr_map.keys(), key=lambda s: int(s))

        stack_black = np.zeros_like(t_abs)
        if len(all_stations) > 0:
            bank_all = [tr_map[sta] for sta in all_stations]
            stack_black = np.mean(np.vstack(bank_all), axis=0)
            ms = np.max(np.abs(stack_black)) or 1.0
            stack_black = stack_black / ms

        sel_ids = data.get("selected_ids", [])
        sel_ids = [s for s in sel_ids if s in tr_map]
        n_pass_window = int(data.get("n_pass_window", len(sel_ids)))
        stack_red = stack_black
        if len(sel_ids) > 0:
            bank_sel = [tr_map[sta] for sta in sel_ids]
            stack_red = np.mean(np.vstack(bank_sel), axis=0)
            ms = np.max(np.abs(stack_red)) or 1.0
            stack_red = stack_red / ms

        axc.plot(t_abs[mask], stack_black[mask], color="k", lw=2, label="All aligned traces")
        axc.plot(
            t_abs[mask],
            stack_red[mask],
            color="r",
            lw=2,
            label=f"Pass r_win N={n_pass_window}",
        )
        axc.axhline(0.0, color="k", lw=0.6, alpha=0.6)
        if p_time is not None:
            axc.axvline(x=p_time, color="b", lw=1.5, alpha=0.7, linestyle="--", label="P arrival")
        if s_time is not None:
            axc.axvline(x=s_time, color="g", lw=1.5, alpha=0.7, linestyle="--", label="S arrival")
        axc.set_xlim(start_time, end_time)
        axc.set_ylim(-1.1, 1.1)
        axc.grid(alpha=0.2)
        axc.set_title(comp_titles_cmp[j], fontsize=12, fontweight="bold")
        axc.set_xlabel("Time since origin (s)", fontsize=11)
        if j != 2:
            axc.set_xlabel("")
        if j == 0:
            axc.set_ylabel("Stack (norm.)", fontsize=11)
        axc.legend(loc="upper right", fontsize=9)

        if j == 2:
            try:
                origin_utc = data.get("origin")
                if origin_utc is not None:
                    add_utc_time_axis(axc, origin_utc, tick_tz=utc_tz)
            except Exception as e:
                print(f"[WARN] Failed to add UTC time axis: {e}")

    fig_cmp.suptitle(
        f"Event {eve_id} - Stack compare (black: all aligned; red: pass r_min thresholds)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    cmp_file = save_dir / f"{eve_id}_rtfilter_stack_compare_{align_phase_name}.png"
    fig_cmp.savefig(cmp_file, dpi=300, bbox_inches="tight")
    print(f"✓ Stack comparison plot saved to: {cmp_file}")


def plot_individual_seismograms_three_components(
    show_individual_seismograms: bool,
    all_component_data: dict,
    pass_window_ids: set,
    start_time: float,
    eve_id: str,
    align_phase_name: str,
    save_dir: Path,
) -> None:
    """Plot individual seismograms for Z/R/T components in paged panels."""
    if not show_individual_seismograms:
        return
    try:
        for comp_name, comp_title in zip(["DPZ", "R", "T"], ["Z", "R", "T"]):
            if comp_name not in all_component_data:
                continue

            data = all_component_data[comp_name]
            all_rows = data.get("all_rows", [])
            all_rows = sorted(all_rows, key=lambda t: int(t[1]))
            t_abs = data["t_abs"]
            mask = data["mask"]
            sample_rate = data["sample_rate"]
            win_start = data["win_start"]
            win_end = data["win_end"]
            move_limit_sec = data["move_limit_sec"]
            npts = data["npts"]

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
                set_figure_title(
                    fig_ind,
                    f"{eve_id} {comp_title} individual seismograms fig {fig_idx + 1}",
                )
                if panels_in_fig == 1:
                    axes_ind = [axes_ind]

                for p in range(panels_in_fig):
                    axp = axes_ind[p]
                    global_panel = panel_start + p
                    start_idx = global_panel * n_per
                    end_idx = min((global_panel + 1) * n_per, n_traces)
                    subset = all_rows[start_idx:end_idx]

                    t_win_start = start_time + (win_start / sample_rate)
                    t_win_end = start_time + (win_end / sample_rate)
                    t_explore_start = max(start_time, t_win_start - move_limit_sec)
                    t_explore_end = min(start_time + (npts / sample_rate), t_win_end + move_limit_sec)
                    axp.axvline(x=t_win_start, color="y", lw=1.2, alpha=0.9)
                    axp.axvline(x=t_win_end, color="y", lw=1.2, alpha=0.9)
                    axp.axvline(x=t_explore_start, color="g", lw=1.2, alpha=0.9)
                    axp.axvline(x=t_explore_end, color="g", lw=1.2, alpha=0.9)

                    for idx_in_subset, (_, station_id, y) in enumerate(subset):
                        i = (len(subset) - 1) - idx_in_subset
                        passed_win = station_id in pass_window_ids
                        trace_color = "k" if passed_win else "red"
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
                            va="center",
                        )

                    ref_offset = len(subset) + 1
                    stack_ref = data.get("stack_vec", None)
                    if stack_ref is not None:
                        axp.plot(
                            t_abs[mask],
                            stack_ref[mask] + ref_offset,
                            color="C3",
                            lw=1.2,
                        )

                    axp.set_ylim(-1, len(subset) + 2)
                    axp.grid(alpha=0.2)
                    axp.set_ylabel("Trace index")

                axes_ind[-1].set_xlabel("Time since origin (s)")
                fig_ind.suptitle(
                    f"Event {eve_id} {comp_title}: individual seismograms "
                    f"(20 per panel, fig {fig_idx + 1}/{n_figs})",
                    fontsize=12,
                    fontweight="bold",
                )
                plt.tight_layout()

                ind_file = save_dir / (
                    f"{eve_id}_{comp_title}_individual_seismograms_{align_phase_name}_fig{fig_idx + 1}.png"
                )
                fig_ind.savefig(ind_file, dpi=300, bbox_inches="tight")
                print(f"✓ Individual seismograms plot saved to: {ind_file}")
    except Exception as e:
        print(f"[WARN] Failed to create individual seismograms plots (3 components): {e}")


def plot_three_component_station_pass_map(
    all_component_data: dict,
    eve_id: str,
    align_phase_name: str,
    save_dir: Path,
) -> None:
    """Plot station pass/fail maps for Z/R/T side-by-side."""
    try:
        fig_map, axes_map = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
        set_figure_title(fig_map, f"{eve_id} station pass map (3-comp)")
        comp_order = ["DPZ", "R", "T"]
        comp_titles_map = ["Z", "R", "T"]

        for j, comp_name in enumerate(comp_order):
            axm = axes_map[j]
            if comp_name not in all_component_data:
                axm.set_axis_off()
                continue

            data = all_component_data[comp_name]
            station_ll = data.get("station_ll", {})
            tr_map = data.get("aligned_traces_by_station", {})
            all_stations = sorted(tr_map.keys(), key=lambda s: int(s))
            pass_set = set(data.get("pass_window_ids", []))

            pass_lats = [station_ll[s][0] for s in all_stations if s in pass_set and s in station_ll]
            pass_lons = [station_ll[s][1] for s in all_stations if s in pass_set and s in station_ll]
            fail_lats = [station_ll[s][0] for s in all_stations if s not in pass_set and s in station_ll]
            fail_lons = [station_ll[s][1] for s in all_stations if s not in pass_set and s in station_ll]

            if len(fail_lons) > 0:
                axm.scatter(fail_lons, fail_lats, s=16, c="0.7", label="Fail")
            if len(pass_lons) > 0:
                axm.scatter(pass_lons, pass_lats, s=20, c="C3", label="Pass")

            axm.set_title(comp_titles_map[j], fontsize=12, fontweight="bold")
            if j == 0:
                axm.set_ylabel("Latitude")
            axm.grid(alpha=0.3)
            axm.set_xlabel("Longitude")
            axm.legend(loc="upper right", fontsize=8)

        fig_map.suptitle(
            f"Event {eve_id} - Stations passing thresholds ({align_phase_name})",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        map_file = save_dir / f"{eve_id}_station_pass_map_{align_phase_name}.png"
        fig_map.savefig(map_file, dpi=300, bbox_inches="tight")
        print(f"✓ Station pass/fail map saved to: {map_file}")
    except Exception as e:
        print(f"[WARN] Failed to create station pass/fail map (3 components): {e}")


def plot_three_component_shift_comparison(
    all_component_data: dict,
    eve_id: str,
    align_phase_name: str,
    start_time: float,
    end_time: float,
    win_pre: float,
    win_post: float,
    move_limit_sec: float,
    min_freq: float,
    max_freq: float,
    save_dir: Path,
) -> None:
    """Plot radial/transverse residual shift and correlation comparison panels."""
    if "R" not in all_component_data or "T" not in all_component_data:
        return

    print("Creating shift comparison plot (Radial vs Transverse)...")
    print(
        "Shift comparison parameters: "
        f"align_phase={align_phase_name}, start_time={start_time}, end_time={end_time}, "
        f"win_pre={win_pre}, win_post={win_post}, "
        f"move_limit_sec={move_limit_sec}"
    )

    r_shifts = all_component_data["R"]["station_shifts"]
    t_shifts = all_component_data["T"]["station_shifts"]
    r_corr = all_component_data["R"]["station_corr"]
    t_corr = all_component_data["T"]["station_corr"]
    r_calc = all_component_data["R"].get("calc_shifts", {})
    t_calc = all_component_data["T"].get("calc_shifts", {})

    common_stations = set(r_shifts.keys()) & set(t_shifts.keys())
    common_stations = common_stations & set(r_calc.keys()) & set(t_calc.keys())
    common_corr_stations = set(r_corr.keys()) & set(t_corr.keys())

    if len(common_stations) == 0:
        print("Warning: No common stations found between R and T components")
        return

    stations = sorted(common_stations, key=lambda s: int(s))
    r_lags = np.array([r_shifts[sta]["lag_seconds"] - r_calc[sta] for sta in stations], dtype=float)
    t_lags = np.array([t_shifts[sta]["lag_seconds"] - t_calc[sta] for sta in stations], dtype=float)
    station_nums = np.array([int(sta) for sta in stations], dtype=int)

    pass_r = set(all_component_data["R"].get("pass_window_ids", []))
    pass_t = set(all_component_data["T"].get("pass_window_ids", []))
    pass_mask = np.array([(sta in pass_r) and (sta in pass_t) for sta in stations], dtype=bool)
    fail_mask = ~pass_mask

    fig_shift, axes = plt.subplots(2, 3, figsize=(18, 10))
    set_figure_title(fig_shift, f"{eve_id} shift comparison")
    (ax1, ax2, ax5), (ax3, ax4, ax6) = axes

    ax1.scatter(r_lags[pass_mask], t_lags[pass_mask], alpha=0.6, s=20, color="k", label="Pass r_win")
    if np.any(fail_mask):
        ax1.scatter(r_lags[fail_mask], t_lags[fail_mask], alpha=0.8, s=24, color="red", label="Fail r_win")
    ax1.plot(
        [min(r_lags + t_lags), max(r_lags + t_lags)],
        [min(r_lags + t_lags), max(r_lags + t_lags)],
        "r--",
        alpha=0.5,
        label="1:1 line",
    )
    ax1.set_xlabel("Radial residual shift (s)", fontsize=11)
    ax1.set_ylabel("Transverse residual shift (s)", fontsize=11)
    ax1.set_title("Radial vs Transverse Residuals", fontsize=12, fontweight="bold")
    ax1.axvline(-move_limit_sec, color="0.4", linestyle=":", linewidth=1.2)
    ax1.axvline(move_limit_sec, color="0.4", linestyle=":", linewidth=1.2)
    ax1.axhline(-move_limit_sec, color="0.4", linestyle=":", linewidth=1.2)
    ax1.axhline(move_limit_sec, color="0.4", linestyle=":", linewidth=1.2)
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_aspect("equal", adjustable="box")

    diff_lags = np.array(r_lags) - np.array(t_lags)
    zero_diff_frac = float(np.mean(np.isclose(diff_lags, 0.0, atol=1e-12)))
    ax2.hist(diff_lags, bins=30, alpha=0.7, edgecolor="black")
    ax2.axvline(0, color="r", linestyle="--", linewidth=2, label="Zero difference")
    ax2.axvline(np.median(diff_lags), color="g", linestyle="--", linewidth=2, label=f"Median = {np.median(diff_lags):.3f}s")
    ax2.set_xlabel("R residual - T residual (s)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Residual Difference Distribution", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    if len(common_corr_stations) > 0:
        corr_stations = sorted(common_corr_stations, key=lambda s: int(s))
        r_corr_vals = np.array([r_corr[sta] for sta in corr_stations], dtype=float)
        t_corr_vals = np.array([t_corr[sta] for sta in corr_stations], dtype=float)
        pass_mask_corr = np.array([(sta in pass_r) and (sta in pass_t) for sta in corr_stations], dtype=bool)
        fail_mask_corr = ~pass_mask_corr
        ax5.scatter(r_corr_vals[pass_mask_corr], t_corr_vals[pass_mask_corr], alpha=0.6, s=20, color="k", label="Pass r_win")
        if np.any(fail_mask_corr):
            ax5.scatter(r_corr_vals[fail_mask_corr], t_corr_vals[fail_mask_corr], alpha=0.8, s=24, color="red", label="Fail r_win")
        ax5.plot([0, 1], [0, 1], "r--", alpha=0.5, label="1:1 line")
        ax5.set_xlabel("Radial max corr", fontsize=11)
        ax5.set_ylabel("Transverse max corr", fontsize=11)
        ax5.set_title("Max Correlation: R vs T", fontsize=12, fontweight="bold")
        ax5.grid(alpha=0.3)
        ax5.legend()
        ax5.set_aspect("equal", adjustable="box")
    else:
        ax5.text(0.5, 0.5, "No common corr stations", ha="center", va="center")
        ax5.set_axis_off()

    ax3.plot(station_nums, r_lags, "o-", label="Radial", alpha=0.5, markersize=4, color="0.4")
    ax3.plot(station_nums, t_lags, "s-", label="Transverse", alpha=0.5, markersize=4, color="0.4")
    if np.any(fail_mask):
        ax3.scatter(station_nums[fail_mask], r_lags[fail_mask], color="red", s=24, marker="o", label="Fail r_win")
        ax3.scatter(station_nums[fail_mask], t_lags[fail_mask], color="red", s=24, marker="s")
    ax3.set_xlabel("Station number", fontsize=11)
    ax3.set_ylabel("Residual shift (s)", fontsize=11)
    ax3.set_title("Residuals vs Station", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4.axis("off")
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
    {min_freq:.2f}-{max_freq:.2f} Hz"""
    stats_text += (
        f"\n\nParameters:\n"
        f"  align_phase: {align_phase_name}\n"
        f"  start_time: {start_time}\n"
        f"  end_time: {end_time}\n"
        f"  win_pre: {win_pre}\n"
        f"  win_post: {win_post}\n"
        f"  move_limit_sec: {move_limit_sec}"
    )
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family="monospace", verticalalignment="center")

    ax6.axis("off")

    fig_shift.suptitle(f"Event {eve_id} - Shift & Correlation Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    shift_file = save_dir / f"{eve_id}_shift_comparison_{align_phase_name}.png"
    fig_shift.savefig(shift_file, dpi=300, bbox_inches="tight")
    print(f"✓ Shift comparison plot saved to: {shift_file}")


def plot_three_component_estimated_vs_calculated_shifts(
    all_component_data: dict,
    eve_id: str,
    align_phase_name: str,
    move_limit_sec: float,
    save_dir: Path,
) -> None:
    """Plot estimated-vs-calculated shifts for Z/R/T components."""
    print("Creating estimated vs calculated shift plot (3 components)...")

    fig_ec, axes_ec = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)
    set_figure_title(fig_ec, f"{eve_id} est vs calc shifts (3-comp)")
    comp_order = ["DPZ", "R", "T"]
    comp_titles_ec = ["Z", "R", "T"]

    for j, comp_name in enumerate(comp_order):
        axc = axes_ec[j]
        if comp_name not in all_component_data:
            axc.set_axis_off()
            continue

        data = all_component_data[comp_name]
        station_shifts = data.get("station_shifts", {})
        calc_shifts = data.get("calc_shifts", {})
        pass_set = set(data.get("pass_window_ids", []))

        common_sta = set(calc_shifts.keys()) & set(station_shifts.keys())
        if len(common_sta) == 0:
            axc.text(0.5, 0.5, "No common stations", ha="center", va="center")
            axc.set_axis_off()
            continue

        stations = sorted(common_sta, key=lambda s: int(s))
        est_shift = np.array([station_shifts[s]["lag_seconds"] for s in stations], dtype=float)
        calc_shift = np.array([calc_shifts[s] for s in stations], dtype=float)
        pass_mask = np.array([s in pass_set for s in stations], dtype=bool)
        fail_mask = ~pass_mask

        axc.scatter(calc_shift[pass_mask], est_shift[pass_mask], s=18, alpha=0.6, color="k", label="Pass r_win")
        if np.any(fail_mask):
            axc.scatter(calc_shift[fail_mask], est_shift[fail_mask], s=22, alpha=0.8, color="red", label="Fail r_win")

        minv = float(min(np.min(calc_shift), np.min(est_shift)))
        maxv = float(max(np.max(calc_shift), np.max(est_shift)))
        axc.plot([minv, maxv], [minv, maxv], "r--", lw=1.2, alpha=0.7)
        axc.plot(
            [minv, maxv],
            [minv + move_limit_sec, maxv + move_limit_sec],
            color="0.4",
            linestyle=":",
            lw=1.2,
        )
        axc.plot(
            [minv, maxv],
            [minv - move_limit_sec, maxv - move_limit_sec],
            color="0.4",
            linestyle=":",
            lw=1.2,
        )

        axc.set_title(comp_titles_ec[j], fontsize=12, fontweight="bold")
        axc.grid(alpha=0.3)
        axc.set_xlabel("Calculated shift (s)", fontsize=10)
        if j == 0:
            axc.set_ylabel("Estimated shift (s)", fontsize=10)
        axc.legend(loc="upper left", fontsize=8)

        fig_ec.suptitle(
            f"Event {eve_id} - Estimated vs Calculated shifts ({align_phase_name})",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()

        estcalc_file = save_dir / f"{eve_id}_est_vs_calc_shift_{align_phase_name}.png"
        fig_ec.savefig(estcalc_file, dpi=300, bbox_inches="tight")
        print(f"✓ Estimated vs calculated shift plot saved to: {estcalc_file}")


def render_three_component_panel(
    fig,
    gs,
    idx: int,
    comp_name: str,
    comp_titles: list,
    data: dict,
    start_time: float,
    end_time: float,
) -> None:
    """Render top record-section and bottom stack panel for one component."""
    all_rows = data["all_rows"]
    stack_vec = data["stack_vec"]
    t_abs = data["t_abs"]
    mask = data["mask"]
    sample_rate = data["sample_rate"]
    win_start = data["win_start"]
    win_end = data["win_end"]
    move_limit_sec = data["move_limit_sec"]
    npts = data["npts"]
    t_ref = data.get("t_ref")

    ax = fig.add_subplot(gs[0, idx])

    all_rows.sort(key=lambda t: t[0])
    t_masked = t_abs[mask]

    if len(all_rows) > 0 and np.any(mask):
        A = np.vstack([row[2][mask] for row in all_rows])
        dvec = np.array([row[0] for row in all_rows], dtype=float)

        if len(dvec) == 1:
            y_edges = np.array([dvec[0] - 0.5, dvec[0] + 0.5])
        else:
            mids = 0.5 * (dvec[1:] + dvec[:-1])
            y_edges = np.empty(len(dvec) + 1)
            y_edges[1:-1] = mids
            y_edges[0] = dvec[0] - (mids[0] - dvec[0])
            y_edges[-1] = dvec[-1] + (dvec[-1] - mids[-1])

        if len(t_masked) == 1:
            t_edges = np.array([t_masked[0] - 0.5 / sample_rate, t_masked[0] + 0.5 / sample_rate])
        else:
            tmids = 0.5 * (t_masked[1:] + t_masked[:-1])
            t_edges = np.empty(len(t_masked) + 1)
            t_edges[1:-1] = tmids
            t_edges[0] = t_masked[0] - (tmids[0] - t_masked[0])
            t_edges[-1] = t_masked[-1] + (t_masked[-1] - tmids[-1])

        ax.pcolormesh(t_edges, y_edges, A, cmap="gray", shading="auto", vmin=-1.0, vmax=1.0)

    ax.set_xlim(start_time, end_time)
    if idx == 0:
        ax.set_ylabel("Epicentral distance (km)", fontsize=11)
    ax.set_title(f"{comp_titles[idx]}", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)

    if t_ref is not None:
        ax.axvline(x=t_ref, color="r", lw=2, alpha=0.6, linestyle="--", zorder=6)
    try:
        draw_correlation_markers(
            ax,
            start_time,
            win_start,
            win_end,
            sample_rate,
            move_limit_sec,
            npts,
        )
    except Exception as e:
        print(f"[WARN] Failed to draw correlation window bounds (top {comp_name}): {e}")

    if idx == 0:
        try:
            n_pass_window = int(data.get("n_pass_window", 0))
            legend_handles = [
                Line2D([0], [0], color="y", lw=2, label="Correlation window"),
                Line2D([0], [0], color="g", lw=2, label="Correlation search (±move_limit_sec)"),
                Line2D([0], [0], color="none", label=f"Pass r_win: {n_pass_window}"),
            ]
            ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=9,
            )
        except Exception as e:
            print(f"[WARN] Failed to add legend (top {comp_name}): {e}")

    ax2 = fig.add_subplot(gs[1, idx])
    ax2.plot(t_abs[mask], stack_vec[mask], color="C3", lw=2)
    ax2.axhline(0.0, color="k", lw=0.6)
    ax2.set_xlim(start_time, end_time)
    ax2.set_xlabel("Time since origin (s)", fontsize=11)
    if idx == 0:
        ax2.set_ylabel("Stack (norm.)", fontsize=11)
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(alpha=0.2)

    if t_ref is not None:
        ax2.axvline(x=t_ref, color="r", lw=2, alpha=0.6, linestyle="--", zorder=6)
    try:
        draw_correlation_markers(
            ax2,
            start_time,
            win_start,
            win_end,
            sample_rate,
            move_limit_sec,
            npts,
        )
    except Exception as e:
        print(f"[WARN] Failed to draw correlation window bounds (bottom {comp_name}): {e}")


def setup_three_component_record_figure(show_record: bool, eve_id: str, align_phase_name: str):
    """Create combined 3-component record-section figure and grid spec when enabled."""
    if not show_record:
        return None, None

    fig_local = plt.figure(figsize=(18, 9))
    set_figure_title(fig_local, f"{eve_id} {align_phase_name} 3-comp record section")
    gs_local = fig_local.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3, wspace=0.25)
    return fig_local, gs_local


def finalize_three_component_record_figure(fig, eve_id: str, align_phase_name: str, save_dir: Path) -> None:
    """Apply title and save the combined 3-component record-section figure."""
    fig.suptitle(
        f"Event {eve_id} - Aligned {align_phase_name} waveforms (3 components)",
        fontsize=14,
        fontweight="bold",
    )
    save_file = save_dir / f"{eve_id}_3comp_{align_phase_name}.png"
    fig.savefig(save_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Three-component plot saved to: {save_file}")
    print("\n✓ Three-component plot created successfully!\n")
    # plt.show()  # defer until end


def render_and_collect_three_component_stacks(
    all_component_data: dict,
    comp_order: list,
    comp_titles: list,
    show_record: bool,
    fig,
    gs,
    start_time: float,
    end_time: float,
    t_abs,
    mask,
):
    """Render enabled record-section panels and return stack vectors by component."""
    stack_by_comp = {}

    for idx, comp_name in enumerate(comp_order):
        if comp_name not in all_component_data:
            print(f"Warning: {comp_name} data not found")
            continue

        data = all_component_data[comp_name]
        stack_vec = data["stack_vec"]
        t_abs = data["t_abs"]
        mask = data["mask"]

        if show_record:
            render_three_component_panel(
                fig=fig,
                gs=gs,
                idx=idx,
                comp_name=comp_name,
                comp_titles=comp_titles,
                data=data,
                start_time=start_time,
                end_time=end_time,
            )

        stack_by_comp[comp_name] = stack_vec

    return stack_by_comp, t_abs, mask


def get_three_component_plot_context(all_component_data: dict):
    """Return shared context values used by combined 3-component plotting/output."""
    comp_order = ["DPZ", "R", "T"]
    comp_titles = ["Vertical (Z)", "Radial (R)", "Transverse (T)"]

    first_data = all_component_data[comp_order[0]]
    eve_id = first_data["eve_id"]
    align_phase_name = first_data["align_phase"]
    start_time_local = first_data["start_time"]
    end_time_local = first_data["end_time"]
    t_abs = first_data["t_abs"]
    mask = first_data["mask"]
    sample_rate_env = first_data["sample_rate"]
    origin_env = first_data.get("origin")

    return (
        comp_order,
        comp_titles,
        eve_id,
        align_phase_name,
        start_time_local,
        end_time_local,
        t_abs,
        mask,
        sample_rate_env,
        origin_env,
    )


def plot_three_component_summary_products(
    all_component_data: dict,
    comp_order: list,
    stack_by_comp: dict,
    sample_rate_env: float,
    t_abs,
    mask,
    start_time: float,
    end_time: float,
    eve_id: str,
    align_phase_name: str,
    save_dir: Path,
    origin_env,
    catalog_df,
    pass_window_ids: set,
) -> None:
    """Generate and save all downstream three-component product plots."""
    plot_three_component_log_envelope(
        comp_order=comp_order,
        stack_by_comp=stack_by_comp,
        sample_rate_env=sample_rate_env,
        t_abs=t_abs,
        mask=mask,
        start_time=start_time,
        end_time=end_time,
        eve_id=eve_id,
        align_phase_name=align_phase_name,
        save_dir=save_dir,
        origin_env=origin_env,
        catalog_df=catalog_df,
    )

    plot_three_component_stack_compare(
        all_component_data=all_component_data,
        eve_id=eve_id,
        align_phase_name=align_phase_name,
        save_dir=save_dir,
    )

    plot_individual_seismograms_three_components(
        show_individual_seismograms=show_individual_seismograms,
        all_component_data=all_component_data,
        pass_window_ids=pass_window_ids,
        start_time=start_time,
        eve_id=eve_id,
        align_phase_name=align_phase_name,
        save_dir=save_dir,
    )

    plot_three_component_station_pass_map(
        all_component_data=all_component_data,
        eve_id=eve_id,
        align_phase_name=align_phase_name,
        save_dir=save_dir,
    )

    plot_three_component_shift_comparison(
        all_component_data=all_component_data,
        eve_id=eve_id,
        align_phase_name=align_phase_name,
        start_time=start_time,
        end_time=end_time,
        win_pre=win_pre,
        win_post=win_post,
        move_limit_sec=move_limit_sec,
        min_freq=min_freq,
        max_freq=max_freq,
        save_dir=save_dir,
    )

    plot_three_component_estimated_vs_calculated_shifts(
        all_component_data=all_component_data,
        eve_id=eve_id,
        align_phase_name=align_phase_name,
        move_limit_sec=move_limit_sec,
        save_dir=save_dir,
    )


def plot_single_component_products(
    record_fig,
    save_dir: Path,
    eve_id: str,
    plot_comp: str,
    align_phase_name: str,
    num_traces: int,
    stack_vec,
    sample_rate: float,
    t_abs,
    mask,
    start_time: float,
    end_time: float,
    origin,
    catalog_df,
    calc_shifts: dict,
    station_shifts: dict,
    win_start,
    win_end,
    ref_window,
    pass_window_ids: set,
    snippet_by_station: dict,
    selected_rows: list,
    rejected_rows: list,
    npts: int,
    aligned_traces_by_station: dict,
    name2ll: dict,
) -> None:
    """Generate and show all single-component plots after alignment."""
    save_file = save_dir / f"{eve_id}_{plot_comp}_{align_phase_name}.png"
    if record_fig is not None:
        record_fig.savefig(save_file, dpi=300, bbox_inches="tight")

    plot_single_trace_log_envelope(
        num_traces=num_traces,
        stack_vec=stack_vec,
        sample_rate=sample_rate,
        t_abs=t_abs,
        mask=mask,
        start_time=start_time,
        end_time=end_time,
        eve_id=eve_id,
        plot_comp=plot_comp,
        align_phase_name=align_phase_name,
        save_dir=save_dir,
        origin=origin,
        catalog_df=catalog_df,
    )

    plot_estimated_vs_calculated_shifts(
        calc_shifts=calc_shifts,
        station_shifts=station_shifts,
        eve_id=eve_id,
        plot_comp=plot_comp,
        align_phase_name=align_phase_name,
        save_dir=save_dir,
    )

    plot_snippet_comparison(
        start_time=start_time,
        win_start=win_start,
        win_end=win_end,
        sample_rate=sample_rate,
        ref_window=ref_window,
        pass_window_ids=pass_window_ids,
        snippet_by_station=snippet_by_station,
        eve_id=eve_id,
        plot_comp=plot_comp,
        align_phase_name=align_phase_name,
        save_dir=save_dir,
    )

    plot_individual_seismograms_single_component(
        show_individual_seismograms=show_individual_seismograms,
        selected_rows=selected_rows,
        rejected_rows=rejected_rows,
        pass_window_ids=pass_window_ids,
        t_abs=t_abs,
        mask=mask,
        stack_vec=stack_vec,
        start_time=start_time,
        win_start=win_start,
        win_end=win_end,
        sample_rate=sample_rate,
        move_limit_sec=move_limit_sec,
        npts=npts,
        eve_id=eve_id,
        plot_comp=plot_comp,
        align_phase_name=align_phase_name,
        save_dir=save_dir,
    )

    plot_station_pass_map(
        aligned_traces_by_station=aligned_traces_by_station,
        pass_window_ids=pass_window_ids,
        name2ll=name2ll,
        eve_id=eve_id,
        plot_comp=plot_comp,
        align_phase_name=align_phase_name,
        save_dir=save_dir,
    )


def finalize_single_component_plotting(plot_wall_start: float, plot_cpu_start: float) -> None:
    """Record timing and show figures for single-component plotting mode."""
    add_stage_timing(timing_state, "plot_and_save", plot_wall_start, plot_cpu_start)
    report_timing_once(timing_state)
    plt.show()


def start_plot_timing():
    """Return current wall-clock and CPU timestamps for plotting stages."""
    return time.perf_counter(), time.process_time()


def finalize_three_component_plotting(plot_wall_start: float, plot_cpu_start: float) -> None:
    """Record timing and show figures for combined three-component plotting mode."""
    add_stage_timing(timing_state, "plot_three_component", plot_wall_start, plot_cpu_start)
    # Show all figures together (three-component + shift comparison)
    # plt.show()
    report_timing_once(timing_state)
    print("\a\a\a")
    plt.show()


def print_three_component_banner() -> None:
    """Print section banner for combined three-component plotting."""
    print(f"\n{'='*70}")
    print("Creating combined three-component plot...")
    print(f"{'='*70}\n")
def select_component_stream(
    st_window: Stream,
    sel_comp: str,
    channel: str,
    name2ll: dict,
    eve_lat: float,
    eve_lon: float,
):
    """Return selected component stream and plot label, rotating to R/T when requested."""
    plot_comp = sel_comp

    if sel_comp in ("R", "T"):
        print("Rotating horizontal components (N/E) to R/T ...")
        st_comp, plot_comp = rotate_horizontals_to_component(
            st_window=st_window,
            sel_comp=sel_comp,
            name2ll=name2ll,
            eve_lat=eve_lat,
            eve_lon=eve_lon,
            timing_state=timing_state,
        )
    else:
        st_comp = st_window.select(channel=channel)

    return st_comp, plot_comp


def prepare_reference_and_phase_timing(
    st_comp: Stream,
    name2ll: dict,
    raw_limits_by_station,
    event_depth: float,
    origin,
    align_phase_name: str,
):
    """Select reference trace, print summary, and compute phase timing for alignment."""
    ref_station_id, ref_trace = select_reference_trace(st_comp, name2ll)
    if ref_trace is None or ref_station_id is None:
        return None

    print_reference_summary(ref_station_id, ref_trace, raw_limits_by_station)

    p_traveltime, s_traveltime, p_arrival_time, s_arrival_time, phase_traveltime = compute_phase_travel_times(
        model,
        event_depth,
        ref_trace,
        origin,
        align_phase_name,
    )
    if phase_traveltime is None:
        print("    No valid phase for alignment. Skip array.")
        return None

    t_ref = phase_traveltime
    return (
        ref_station_id,
        ref_trace,
        p_traveltime,
        s_traveltime,
        p_arrival_time,
        s_arrival_time,
        phase_traveltime,
        t_ref,
    )
def load_event_context_and_waveforms(
    eve_id: str,
    channel: str,
    process_as_three_comp: bool,
    horizontal_window_cache: dict,
    horizontal_raw_limits_cache: dict,
):
    """Load event metadata/station lookup and read event waveforms for one channel."""
    event_depth, eve_lat, eve_lon, origin = load_event_metadata(eve_id, info_root)
    save_dir = make_event_output_dir(path_prefix, eve_id)
    name2ll = load_station_lookup(info_root)

    st_window, raw_limits_by_station = read_waveforms_for_event(
        eve_id=eve_id,
        channel=channel,
        process_as_three_comp_mode=process_as_three_comp,
        horizontal_window_cache=horizontal_window_cache,
        horizontal_raw_limits_cache=horizontal_raw_limits_cache,
        name2ll=name2ll,
        eve_lat=eve_lat,
        eve_lon=eve_lon,
        origin=origin,
        data_path=data_path,
        sps_rate=sps_rate,
        start_time=start_time,
        end_time=end_time,
        verbose=verbose,
        timing_state=timing_state,
    )
    if st_window is None:
        return None

    return (
        event_depth,
        eve_lat,
        eve_lon,
        origin,
        save_dir,
        name2ll,
        st_window,
        raw_limits_by_station,
    )
def prepare_stream_reference_context(
    st_window: Stream,
    sel_comp: str,
    channel: str,
    name2ll: dict,
    eve_lat: float,
    eve_lon: float,
    raw_limits_by_station,
    event_depth: float,
    origin,
    align_phase_name: str,
):
    """Prepare component stream, reference timing context, and trace count for one event."""
    st_comp, plot_comp = select_component_stream(
        st_window=st_window,
        sel_comp=sel_comp,
        channel=channel,
        name2ll=name2ll,
        eve_lat=eve_lat,
        eve_lon=eve_lon,
    )

    ref_phase_timing = prepare_reference_and_phase_timing(
        st_comp=st_comp,
        name2ll=name2ll,
        raw_limits_by_station=raw_limits_by_station,
        event_depth=event_depth,
        origin=origin,
        align_phase_name=align_phase_name,
    )
    if ref_phase_timing is None:
        return None

    (
        ref_station_id,
        ref_trace,
        p_traveltime,
        s_traveltime,
        p_arrival_time,
        s_arrival_time,
        phase_traveltime,
        t_ref,
    ) = ref_phase_timing

    num_traces = len(st_comp)
    print(f"    {num_traces} traces on {plot_comp}")
    if num_traces == 0:
        return None

    return (
        st_comp,
        plot_comp,
        ref_station_id,
        ref_trace,
        p_traveltime,
        s_traveltime,
        p_arrival_time,
        s_arrival_time,
        phase_traveltime,
        t_ref,
        num_traces,
    )


def run_alignment_and_unpack(
    st_comp: Stream,
    ref_trace: Trace,
    ref_station_id,
    name2ll: dict,
    eve_lat: float,
    eve_lon: float,
    event_depth: float,
    align_phase_name: str,
    t_ref,
):
    """Compute alignment products and return unpacked values used by run_pipeline."""
    alignment = compute_alignment_products(
        st_comp=st_comp,
        ref_trace=ref_trace,
        ref_station_id=ref_station_id,
        name2ll=name2ll,
        eve_lat=eve_lat,
        eve_lon=eve_lon,
        event_depth=event_depth,
        align_phase_name=align_phase_name,
        t_ref=t_ref,
    )
    return (
        alignment["npts"],
        alignment["sample_rate"],
        alignment["move_limit_samples"],
        alignment["win_start"],
        alignment["win_end"],
        alignment["calc_shifts"],
        alignment["aligned_stack"],
        alignment["selected_aligned_stack"],
        alignment["selected_ids"],
        alignment["station_corr"],
        alignment["n_pass_window"],
        alignment["pass_window_ids"],
        alignment["snippet_by_station"],
        alignment["ref_window"],
        alignment["selected_rows"],
        alignment["rejected_rows"],
        alignment["station_shifts"],
        alignment["aligned_traces_by_station"],
        alignment["t_abs"],
        alignment["mask"],
        alignment["stack_vec"],
    )


def store_three_component_data(
    all_component_data: dict,
    channel: str,
    sel_comp: str,
    record_fig,
    selected_rows: list,
    rejected_rows: list,
    stack_vec,
    t_abs,
    mask,
    sample_rate: float,
    win_start,
    win_end,
    move_limit_samples: int,
    npts: int,
    eve_id: str,
    align_phase_name: str,
    origin,
    station_shifts: dict,
    station_corr: dict,
    calc_shifts: dict,
    n_pass_window: int,
    pass_window_ids: set,
    snippet_by_station: dict,
    ref_window,
    p_traveltime,
    s_traveltime,
    name2ll: dict,
    selected_ids: set,
    aligned_traces_by_station: dict,
    t_ref,
) -> None:
    """Store per-component plotting payload for deferred 3-component rendering."""
    comp_key = resolve_component_key(channel, sel_comp)
    all_component_data[comp_key] = build_component_output_payload(
        record_fig=record_fig,
        selected_rows=selected_rows,
        rejected_rows=rejected_rows,
        stack_vec=stack_vec,
        t_abs=t_abs,
        mask=mask,
        sample_rate=sample_rate,
        win_start=win_start,
        win_end=win_end,
        move_limit_sec_value=move_limit_sec,
        move_limit_samples=move_limit_samples,
        npts=npts,
        start_t=start_time,
        end_t=end_time,
        eve_id=eve_id,
        align_phase_name=align_phase_name,
        origin=origin,
        station_shifts=station_shifts,
        station_corr=station_corr,
        calc_shifts=calc_shifts,
        n_pass_window=n_pass_window,
        pass_window_ids=pass_window_ids,
        snippet_by_station=snippet_by_station,
        ref_window=ref_window,
        p_traveltime=p_traveltime,
        s_traveltime=s_traveltime,
        name2ll=name2ll,
        selected_ids=selected_ids,
        aligned_traces_by_station=aligned_traces_by_station,
        t_ref=t_ref,
    )

    if record_fig is not None:
        plt.close(record_fig)


def persist_three_component_outputs(
    comp_order: list,
    stack_by_comp: dict,
    save_dir: Path,
    eve_id: str,
    origin_env,
    start_time: float,
    sample_rate_env: float,
    show_record: bool,
    fig,
    align_phase_name: str,
) -> Path:
    """Write 3-component stack mSEEDs and optional combined figure, returning output dir."""
    if all(comp in stack_by_comp for comp in comp_order):
        save_dir = make_event_output_dir(path_prefix, eve_id)
        write_component_stack_mseeds(
            comp_order=comp_order,
            stack_by_comp=stack_by_comp,
            save_dir=save_dir,
            eve_id=eve_id,
            origin_env=origin_env,
            start_time=start_time,
            sample_rate_env=sample_rate_env,
        )

    if show_record:
        save_dir = make_event_output_dir(path_prefix, eve_id)
        finalize_three_component_record_figure(
            fig=fig,
            eve_id=eve_id,
            align_phase_name=align_phase_name,
            save_dir=save_dir,
        )

    return save_dir


def compute_alignment_products(
    st_comp: Stream,
    ref_trace: Trace,
    ref_station_id: str,
    name2ll: dict,
    eve_lat: float,
    eve_lon: float,
    event_depth: float,
    align_phase_name: str,
    t_ref,
):
    """Run alignment stages and return all products needed by plotting/output."""
    # ---- Common length / sampling rate / windows ----
    setup = compute_alignment_setup(
        st_comp=st_comp,
        ref_trace=ref_trace,
        move_limit_sec=move_limit_sec,
        start_time=start_time,
        win_pre=win_pre,
        win_post=win_post,
        t_ref=t_ref,
    )
    npts = setup["npts"]
    sample_rate = setup["sample_rate"]
    ref = setup["ref"]
    move_limit_samples = setup["move_limit_samples"]
    win_start = setup["win_start"]
    win_end = setup["win_end"]
    print(f"    sample_rate = {sample_rate:.1f} Hz")

    # ---- Normalize per trace using only the correlation window ----
    normalize_traces_in_window(st_comp, win_start, win_end)

    # ---- Theoretical (TauP) shift per station relative to reference station ----
    calc_shifts = compute_taup_station_shifts(
        model_obj=model,
        st_comp=st_comp,
        event_depth=event_depth,
        align_phase_name=align_phase_name,
        t_ref=t_ref,
        timing_state=timing_state,
    )

    # ===================== Stage 1: align to reference -> aligned_stack =====================
    aligned_stack = compute_stage1_aligned_stack(
        st_comp=st_comp,
        ref=ref,
        npts=npts,
        sample_rate=sample_rate,
        win_start=win_start,
        win_end=win_end,
        move_limit_samples=move_limit_samples,
        calc_shifts=calc_shifts,
        timing_state=timing_state,
    )

    # ===================== Stage 2: align to aligned_stack -> select traces =====================
    stage2_products = compute_stage2_screened_stack(
        st_comp=st_comp,
        aligned_stack=aligned_stack,
        npts=npts,
        sample_rate=sample_rate,
        win_start=win_start,
        win_end=win_end,
        move_limit_samples=move_limit_samples,
        calc_shifts=calc_shifts,
        r_window_min=r_window_min,
        timing_state=timing_state,
    )
    selected_aligned_stack = stage2_products["selected_aligned_stack"]
    selected_ids = stage2_products["selected_ids"]
    station_corr = stage2_products["station_corr"]
    n_pass_window = stage2_products["n_pass_window"]
    pass_window_ids = stage2_products["pass_window_ids"]
    snippet_by_station = stage2_products["snippet_by_station"]
    ref_window = stage2_products["ref_window"]

    # ===================== Final alignment for plotting (lag3 to reference) =====================
    stage3_products = compute_stage3_finalized_rows(
        st_comp=st_comp,
        ref=ref,
        ref_station_id=ref_station_id,
        selected_ids=selected_ids,
        calc_shifts=calc_shifts,
        npts=npts,
        sample_rate=sample_rate,
        win_start=win_start,
        win_end=win_end,
        move_limit_samples=move_limit_samples,
        name2ll=name2ll,
        eve_lat=eve_lat,
        eve_lon=eve_lon,
        timing_state=timing_state,
    )
    selected_rows = stage3_products["selected_rows"]
    rejected_rows = stage3_products["rejected_rows"]
    aligned_bank_all = stage3_products["aligned_bank_all"]
    station_shifts = stage3_products["station_shifts"]
    aligned_traces_by_station = stage3_products["aligned_traces_by_station"]

    # ---- Time axis (seconds since origin) and final stack ----
    axis_stack_products = compute_time_axis_and_stack(
        start_time=start_time,
        end_time=end_time,
        npts=npts,
        sample_rate=sample_rate,
        aligned_bank_all=aligned_bank_all,
        win_start=win_start,
        win_end=win_end,
    )
    t_abs = axis_stack_products["t_abs"]
    mask = axis_stack_products["mask"]
    stack_vec = axis_stack_products["stack_vec"]

    return build_alignment_products_payload(
        npts=npts,
        sample_rate=sample_rate,
        move_limit_samples=move_limit_samples,
        win_start=win_start,
        win_end=win_end,
        calc_shifts=calc_shifts,
        aligned_stack=aligned_stack,
        selected_aligned_stack=selected_aligned_stack,
        selected_ids=selected_ids,
        station_corr=station_corr,
        n_pass_window=n_pass_window,
        pass_window_ids=pass_window_ids,
        snippet_by_station=snippet_by_station,
        ref_window=ref_window,
        selected_rows=selected_rows,
        rejected_rows=rejected_rows,
        station_shifts=station_shifts,
        aligned_traces_by_station=aligned_traces_by_station,
        t_abs=t_abs,
        mask=mask,
        stack_vec=stack_vec,
    )
def run_pipeline() -> None:
    global align_phase, move_limit_sec, start_time, end_time
    save_dir = Path(path_prefix + "output")
    # User-facing components: Z, R, T
    channels, process_as_three_comp, sel_comp_list = get_component_selection(
        all_channels, component
    )

    # Caches are harmless in single-component mode and simplify control flow.
    all_component_data, horizontal_window_cache, horizontal_raw_limits_cache = {}, {}, {}
    
    for idx, channel in enumerate(channels):
        sel_comp = sel_comp_list[idx]
        
        print(f"Processing channel: {channel}")
    
        for eve_id in events:
            print(f"==========Processing event {eve_id}===========")

            event_context = load_event_context_and_waveforms(
                eve_id=eve_id,
                channel=channel,
                process_as_three_comp=process_as_three_comp,
                horizontal_window_cache=horizontal_window_cache,
                horizontal_raw_limits_cache=horizontal_raw_limits_cache,
            )
            if event_context is None:
                continue
            (
                event_depth,
                eve_lat,
                eve_lon,
                origin,
                save_dir,
                name2ll,
                st_window,
                raw_limits_by_station,
            ) = event_context
    
            stream_ref_context = prepare_stream_reference_context(
                st_window=st_window,
                sel_comp=sel_comp,
                channel=channel,
                name2ll=name2ll,
                eve_lat=eve_lat,
                eve_lon=eve_lon,
                raw_limits_by_station=raw_limits_by_station,
                event_depth=event_depth,
                origin=origin,
                align_phase_name=align_phase,
            )
            if stream_ref_context is None:
                continue
            (
                st_comp,
                plot_comp,
                ref_station_id,
                ref_trace,
                p_traveltime,
                s_traveltime,
                _, _, _,
                t_ref,
                num_traces,
            ) = stream_ref_context

            preprocess_traces_bandpass(
                st_comp=st_comp,
                min_freq=min_freq,
                max_freq=max_freq,
                timing_state=timing_state,
            )

            (
                npts,
                sample_rate,
                move_limit_samples,
                win_start,
                win_end,
                calc_shifts,
                aligned_stack,
                selected_aligned_stack,
                selected_ids,
                station_corr,
                n_pass_window,
                pass_window_ids,
                snippet_by_station,
                ref_window,
                selected_rows,
                rejected_rows,
                station_shifts,
                aligned_traces_by_station,
                t_abs,
                mask,
                stack_vec,
            ) = run_alignment_and_unpack(
                st_comp=st_comp,
                ref_trace=ref_trace,
                ref_station_id=ref_station_id,
                name2ll=name2ll,
                eve_lat=eve_lat,
                eve_lon=eve_lon,
                event_depth=event_depth,
                align_phase_name=align_phase,
                t_ref=t_ref,
            )

            _plot_wall_start, _plot_cpu_start = start_plot_timing()
            if not all_channels:
                plot_stage_stacks(
                    eve_id=eve_id,
                    plot_comp=plot_comp,
                    align_phase_name=align_phase,
                    t_abs=t_abs,
                    mask=mask,
                    aligned_stack=aligned_stack,
                    selected_aligned_stack=selected_aligned_stack,
                    stack_vec=stack_vec,
                    save_dir=save_dir,
                )

            record_fig = plot_record_section_and_stack(
                show_record=show_record_section_plot,
                eve_id=eve_id,
                plot_comp=plot_comp,
                align_phase_name=align_phase,
                selected_rows=selected_rows,
                rejected_rows=rejected_rows,
                t_abs=t_abs,
                mask=mask,
                sample_rate=sample_rate,
                t_ref=t_ref,
                win_start=win_start,
                win_end=win_end,
                move_sec=move_limit_sec,
                npts=npts,
                n_pass_window=n_pass_window,
                stack_vec=stack_vec,
                save_dir=save_dir,
            )

            # Store data for three-component plotting or show individual plot
            if process_as_three_comp:
                store_three_component_data(
                    all_component_data=all_component_data,
                    channel=channel,
                    sel_comp=sel_comp,
                    record_fig=record_fig,
                    selected_rows=selected_rows,
                    rejected_rows=rejected_rows,
                    stack_vec=stack_vec,
                    t_abs=t_abs,
                    mask=mask,
                    sample_rate=sample_rate,
                    win_start=win_start,
                    win_end=win_end,
                    move_limit_samples=move_limit_samples,
                    npts=npts,
                    eve_id=eve_id,
                    align_phase_name=align_phase,
                    origin=origin,
                    station_shifts=station_shifts,
                    station_corr=station_corr,
                    calc_shifts=calc_shifts,
                    n_pass_window=n_pass_window,
                    pass_window_ids=pass_window_ids,
                    snippet_by_station=snippet_by_station,
                    ref_window=ref_window,
                    p_traveltime=p_traveltime,
                    s_traveltime=s_traveltime,
                    name2ll=name2ll,
                    selected_ids=selected_ids,
                    aligned_traces_by_station=aligned_traces_by_station,
                    t_ref=t_ref,
                )
            else:
                # No Z-only R-T screening reuse.
                plot_single_component_products(
                    record_fig=record_fig,
                    save_dir=save_dir,
                    eve_id=eve_id,
                    plot_comp=plot_comp,
                    align_phase_name=align_phase,
                    num_traces=num_traces,
                    stack_vec=stack_vec,
                    sample_rate=sample_rate,
                    t_abs=t_abs,
                    mask=mask,
                    start_time=start_time,
                    end_time=end_time,
                    origin=origin,
                    catalog_df=catalog_local,
                    calc_shifts=calc_shifts,
                    station_shifts=station_shifts,
                    win_start=win_start,
                    win_end=win_end,
                    ref_window=ref_window,
                    pass_window_ids=pass_window_ids,
                    snippet_by_station=snippet_by_station,
                    selected_rows=selected_rows,
                    rejected_rows=rejected_rows,
                    npts=npts,
                    aligned_traces_by_station=aligned_traces_by_station,
                    name2ll=name2ll,
                )
                finalize_single_component_plotting(_plot_wall_start, _plot_cpu_start)

    # Three-component combined plotting
    if process_as_three_comp and len(all_component_data) == 3:
        _plot3_wall_start, _plot3_cpu_start = start_plot_timing()
        print_three_component_banner()

        fig, gs = setup_three_component_record_figure(
            show_record=show_record_section_plot,
            eve_id=eve_id,
            align_phase_name=align_phase,
        )

        (
            comp_order,
            comp_titles,
            eve_id,
            align_phase,
            start_time,
            end_time,
            t_abs,
            mask,
            sample_rate_env,
            origin_env,
        ) = get_three_component_plot_context(all_component_data)

        stack_by_comp, t_abs, mask = render_and_collect_three_component_stacks(
            all_component_data=all_component_data,
            comp_order=comp_order,
            comp_titles=comp_titles,
            show_record=show_record_section_plot,
            fig=fig,
            gs=gs,
            start_time=start_time,
            end_time=end_time,
            t_abs=t_abs,
            mask=mask,
        )

        save_dir = persist_three_component_outputs(
            comp_order=comp_order,
            stack_by_comp=stack_by_comp,
            save_dir=save_dir,
            eve_id=eve_id,
            origin_env=origin_env,
            start_time=start_time,
            sample_rate_env=sample_rate_env,
            show_record=show_record_section_plot,
            fig=fig,
            align_phase_name=align_phase,
        )

        # No R-T zero-diff station list saved.
        plot_three_component_summary_products(
            all_component_data=all_component_data,
            comp_order=comp_order,
            stack_by_comp=stack_by_comp,
            sample_rate_env=sample_rate_env,
            t_abs=t_abs,
            mask=mask,
            end_time=end_time,
            start_time=start_time,
            eve_id=eve_id,
            align_phase_name=align_phase,
            save_dir=save_dir,
            origin_env=origin_env,
            catalog_df=catalog_local,
            pass_window_ids=pass_window_ids,
        )

        finalize_three_component_plotting(_plot3_wall_start, _plot3_cpu_start)

def main() -> None:
    run_pipeline()

if __name__ == "__main__":
    main()
