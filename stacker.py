"""Compare the per-event component stacks produced by one align_stack run."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Trace, UTCDateTime, read


OUTPUT_ROOT = Path("/Users/jvidale/Documents/Research/FaultScanR/output")
IMPLICIT_RUN_PREFIX = "2026"
CATALOG_FILE = Path(
    "/Users/jvidale/Documents/Research/FaultScanR/event_sta_info/catalog_local_hand.xlsx"
)
ORIGIN_COL = "origin_time"
COMPONENT_FILE_NAMES = {"Z": "DPZ", "R": "R", "T": "T"}
COMPONENT_LABELS = {
    "Z": "Vertical (Z)",
    "R": "Radial (R)",
    "T": "Transverse (T)",
}

ALIGN_WINDOW_START = 4.6
ALIGN_WINDOW_END = 6.0
MAG_MIN = 0.0
MAG_DIFF_MIN = 0.8
OFFSET_STEP = 0.6
REPLACE_LEVEL = np.nan
SHOW_OFFSET_TRACES = True
SHOW_MEDIAN_TRACE = True
MASK_POLICIES = ("comparable_or_larger", "smaller", "all", "none")

# Defaults used when stacker.py is run with the triangle button in VS Code.
# Edit these values to change an argument-free interactive run.
DEFAULT_RUN = "0724_185013_6649"
DEFAULT_COMPONENTS = ["Z", "R", "T"]
DEFAULT_MASK_OTHER_EVENTS = "smaller"
DEFAULT_OVERLAY_ONLY = True
DEFAULT_SHOW_PLOTS = True
DEFAULT_SAVE_PLOTS = False
DEFAULT_AMPLITUDE_LIMITS = (-0.1, 0.1)


@dataclass
class EventStack:
    event_id: str
    origin: UTCDateTime
    magnitude: float | None
    trace: Trace


@dataclass
class ProcessedStack:
    event_id: str
    origin: UTCDateTime
    time: np.ndarray
    pre_mask: np.ndarray
    post_mask: np.ndarray


def resolve_run_directory(run_suffix: str) -> Path:
    """Resolve a short run suffix beneath the fixed 2026 output prefix."""
    value = run_suffix.strip()
    if value.startswith(IMPLICIT_RUN_PREFIX):
        run_id = value
    else:
        run_id = f"{IMPLICIT_RUN_PREFIX}{value}"

    if not re.fullmatch(r"2026\d{4}_\d{6}_\d{4}", run_id):
        raise ValueError(
            "Run must be the suffix after 2026, such as '0722_182322_4666' "
            "(the complete 2026-prefixed run name is also accepted)."
        )
    return OUTPUT_ROOT / run_id


def normalize_components(components: list[str]) -> list[str]:
    """Normalize component names, reject unknown values, and preserve order."""
    normalized = []
    for component in components:
        value = component.upper()
        if value not in COMPONENT_FILE_NAMES:
            raise ValueError(
                f"Unknown component {component!r}; choose from Z, R, and T"
            )
        if value not in normalized:
            normalized.append(value)
    return normalized


def load_catalog(path: Path) -> pd.DataFrame:
    catalog = pd.read_excel(path, dtype={"evid": str})
    required = {"evid", ORIGIN_COL}
    missing = required - set(catalog.columns)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")
    return catalog


def load_run_parameters(run_dir: Path) -> dict:
    """Read and cross-check the parameter snapshots stored under a run."""
    snapshot_paths = sorted(run_dir.glob("*/rp_*.json"))
    if not snapshot_paths:
        raise FileNotFoundError(f"No run-parameter snapshot found under {run_dir}")

    snapshots = []
    for path in snapshot_paths:
        with path.open("r", encoding="utf-8") as handle:
            snapshots.append((path, json.load(handle)))

    keys_to_match = (
        "min_freq",
        "max_freq",
        "start_time",
        "end_time",
        "align_phase",
        "analysis_hz",
    )
    first_path, first = snapshots[0]
    for path, snapshot in snapshots[1:]:
        differences = [
            key for key in keys_to_match if snapshot.get(key) != first.get(key)
        ]
        if differences:
            raise ValueError(
                f"Run snapshots disagree between {first_path} and {path}: "
                f"{differences}"
            )
    return first


def _magnitude_column(catalog: pd.DataFrame) -> str | None:
    for column in ("magnitude", "mag"):
        if column in catalog.columns:
            return column
    return None


def has_zero_skip(value) -> bool:
    """Return True only when a catalog skip value is numerically zero."""
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def load_event_stacks(
    run_dir: Path,
    components: list[str],
    catalog: pd.DataFrame,
    parameters: dict,
) -> dict[str, list[EventStack]]:
    """Load the requested per-event stack files from one align_stack run."""
    catalog_by_event = catalog.drop_duplicates("evid").set_index("evid")
    magnitude_column = _magnitude_column(catalog)
    configured_events = [str(event_id) for event_id in parameters.get("events", [])]
    event_ids = configured_events or sorted(
        path.name for path in run_dir.iterdir() if path.is_dir()
    )
    stacks_by_component = {component: [] for component in components}

    for event_id in event_ids:
        if event_id not in catalog_by_event.index:
            print(f"[WARN] Event {event_id} is absent from the catalog; skipping it.")
            continue
        row = catalog_by_event.loc[event_id]
        if not has_zero_skip(row.get("skip")):
            continue
        magnitude = None
        if magnitude_column is not None:
            try:
                magnitude = float(row[magnitude_column])
            except (TypeError, ValueError):
                pass
        if magnitude is not None and magnitude <= MAG_MIN:
            continue

        origin = UTCDateTime(str(row[ORIGIN_COL]))
        for component in components:
            file_component = COMPONENT_FILE_NAMES[component]
            path = run_dir / event_id / f"{event_id}_{file_component}_stack.mseed"
            if not path.exists():
                print(f"[WARN] Missing requested {component} stack: {path}")
                continue
            stream = read(str(path))
            if not stream:
                print(f"[WARN] Empty requested {component} stack: {path}")
                continue
            stacks_by_component[component].append(
                EventStack(event_id, origin, magnitude, stream[0])
            )

    missing_components = [
        component
        for component, event_stacks in stacks_by_component.items()
        if not event_stacks
    ]
    if missing_components:
        raise FileNotFoundError(
            f"No stack files found for requested components {missing_components} "
            f"under {run_dir}"
        )
    return stacks_by_component


def shift_left_zeropad(values: np.ndarray, samples: int) -> np.ndarray:
    shifted = np.zeros_like(values)
    if samples == 0:
        shifted[:] = values
    elif samples > 0 and samples < values.size:
        shifted[:-samples] = values[samples:]
    elif samples < 0 and -samples < values.size:
        shifted[-samples:] = values[:samples]
    return shifted


def correlation_lag(
    reference: np.ndarray,
    candidate: np.ndarray,
    window_start: int,
    window_end: int,
    max_shift_samples: int,
) -> int:
    """Return the bounded lag that best aligns candidate with reference."""
    best_lag = 0
    best_correlation = -np.inf
    for lag in range(-max_shift_samples, max_shift_samples + 1):
        shifted = shift_left_zeropad(candidate, lag)
        ref_window = reference[window_start:window_end]
        candidate_window = shifted[window_start:window_end]
        if not ref_window.size or ref_window.size != candidate_window.size:
            continue
        ref_centered = ref_window - np.mean(ref_window)
        candidate_centered = candidate_window - np.mean(candidate_window)
        denominator = np.linalg.norm(ref_centered) * np.linalg.norm(
            candidate_centered
        )
        correlation = (
            float(np.dot(ref_centered, candidate_centered) / denominator)
            if denominator
            else -np.inf
        )
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    return best_lag


def should_mask_other_event(
    event_magnitude: float | None,
    other_magnitude: float | None,
    mask_policy: str,
) -> bool:
    """Return whether another event's arrival should be masked."""
    if mask_policy not in MASK_POLICIES:
        raise ValueError(f"Unknown event-mask policy: {mask_policy!r}")
    if mask_policy == "none":
        return False
    if event_magnitude is None or other_magnitude is None:
        return False
    if mask_policy == "smaller":
        return other_magnitude < event_magnitude
    if mask_policy == "all":
        return True
    return other_magnitude >= event_magnitude - MAG_DIFF_MIN


def process_component_stacks(
    event_stacks: list[EventStack],
    catalog: pd.DataFrame,
    parameters: dict,
    mask_policy: str = "comparable_or_larger",
) -> list[ProcessedStack]:
    """Cross-correlate, normalize, and mask the stacks for one component."""
    sample_rates = {float(item.trace.stats.sampling_rate) for item in event_stacks}
    if len(sample_rates) != 1:
        raise ValueError(f"Stack sample rates differ: {sorted(sample_rates)}")
    sample_rate = sample_rates.pop()
    npts = min(int(item.trace.stats.npts) for item in event_stacks)
    start_time = float(parameters.get("start_time", 0.0))
    align_start = max(0, int(round((ALIGN_WINDOW_START - start_time) * sample_rate)))
    align_end = min(npts, int(round((ALIGN_WINDOW_END - start_time) * sample_rate)))
    if align_end <= align_start:
        raise ValueError(
            f"Alignment window {ALIGN_WINDOW_START}–{ALIGN_WINDOW_END} s is "
            f"outside the stack time range."
        )
    max_shift_sec = float(parameters.get("event_stack_alignment_max_shift_sec", 0.2))
    max_shift_samples = int(round(max_shift_sec * sample_rate))
    requested_reference = str(parameters.get("event_alignment_reference", ""))
    reference_item = next(
        (
            item
            for item in event_stacks
            if requested_reference and item.event_id == requested_reference
        ),
        event_stacks[0],
    )
    if requested_reference and reference_item.event_id != requested_reference:
        print(
            f"[WARN] Reference event {requested_reference} has no stack; "
            f"using {reference_item.event_id}."
        )
    reference = np.asarray(reference_item.trace.data[:npts], dtype=float)
    time_axis = start_time + np.arange(npts) / sample_rate

    magnitude_column = _magnitude_column(catalog)
    catalog_events = []
    for _, row in catalog.iterrows():
        try:
            origin = UTCDateTime(str(row[ORIGIN_COL]))
        except Exception:
            continue
        magnitude = None
        if magnitude_column is not None:
            try:
                magnitude = float(row[magnitude_column])
            except (TypeError, ValueError):
                pass
        catalog_events.append((origin, magnitude))

    processed = []
    for item in event_stacks:
        values = np.asarray(item.trace.data[:npts], dtype=float)
        lag = correlation_lag(
            reference,
            values,
            align_start,
            align_end,
            max_shift_samples,
        )
        values = shift_left_zeropad(values, lag)
        maximum = float(np.max(np.abs(values))) if values.size else 0.0
        if maximum:
            values = values / maximum
        pre_mask = values.copy()
        post_mask = values.copy()

        for other_origin, other_magnitude in catalog_events:
            offset = float(other_origin - item.origin)
            if abs(offset) < 1e-6:
                continue
            if not should_mask_other_event(
                item.magnitude,
                other_magnitude,
                mask_policy,
            ):
                continue
            mask_start = offset + ALIGN_WINDOW_START
            mask_end = offset + ALIGN_WINDOW_END
            mask = (time_axis >= mask_start) & (time_axis <= mask_end)
            post_mask[mask] = REPLACE_LEVEL

        processed.append(
            ProcessedStack(
                item.event_id,
                item.origin,
                time_axis.copy(),
                pre_mask,
                post_mask,
            )
        )
    return processed


def _plot_component_rows(
    processed_by_component: dict[str, list[ProcessedStack]],
    components: list[str],
    data_field: str,
    offset: bool,
    title: str,
    output_path: Path | None,
    amplitude_limits: tuple[float, float] | None = None,
    show_plot: bool = False,
) -> None:
    fig, axes = plt.subplots(
        len(components),
        1,
        figsize=(10, 3.0 * len(components)),
        sharex=True,
        squeeze=False,
    )
    for component, ax in zip(components, axes[:, 0]):
        records = processed_by_component[component]
        ordered = sorted(records, key=lambda item: item.origin)
        if offset:
            ordered = list(reversed(ordered))
        for index, record in enumerate(ordered):
            values = np.asarray(getattr(record, data_field), dtype=float)
            vertical_offset = index * OFFSET_STEP if offset else 0.0
            ax.plot(record.time, values + vertical_offset, lw=0.9, alpha=0.7)
            if offset:
                ax.text(
                    record.time[0],
                    vertical_offset,
                    record.event_id,
                    fontsize=6,
                    va="bottom",
                )
        if not offset:
            matrix = np.vstack(
                [np.asarray(getattr(record, data_field), dtype=float) for record in records]
            )
            ax.plot(records[0].time, np.nanmedian(matrix, axis=0), color="k", lw=2.2)
            ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
            if amplitude_limits is not None:
                ax.set_ylim(*amplitude_limits)
        ax.set_ylabel(COMPONENT_LABELS[component])
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Time since event origin (s)")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Wrote plot: {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _plot_component_medians(
    processed_by_component: dict[str, list[ProcessedStack]],
    components: list[str],
    title: str,
    output_path: Path | None,
    amplitude_limits: tuple[float, float] | None = None,
    show_plot: bool = False,
) -> None:
    fig, axes = plt.subplots(
        len(components),
        1,
        figsize=(10, 3.0 * len(components)),
        sharex=True,
        squeeze=False,
    )
    for component, ax in zip(components, axes[:, 0]):
        records = processed_by_component[component]
        matrix = np.vstack([record.post_mask for record in records])
        ax.plot(records[0].time, np.nanmedian(matrix, axis=0), color="k", lw=1.5)
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
        if amplitude_limits is not None:
            ax.set_ylim(*amplitude_limits)
        ax.set_ylabel(COMPONENT_LABELS[component])
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Time since event origin (s)")
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✓ Wrote plot: {output_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def write_plots(
    processed_by_component: dict[str, list[ProcessedStack]],
    components: list[str],
    run_dir: Path,
    parameters: dict,
    mask_policy: str = "comparable_or_larger",
    overlay_only: bool = False,
    save_plots: bool = True,
    show_plots: bool = False,
    amplitude_limits: tuple[float, float] | None = DEFAULT_AMPLITUDE_LIMITS,
) -> Path:
    output_dir = run_dir / "stacker"
    if save_plots:
        output_dir.mkdir(parents=True, exist_ok=True)
    band = f"{parameters.get('min_freq', '?')}–{parameters.get('max_freq', '?')} Hz"
    common_title = f"{run_dir.name}: {band}; components {' '.join(components)}"
    if mask_policy == "comparable_or_larger":
        filename_suffix = ""
        mask_title = "comparable/larger-event arrivals masked"
    else:
        filename_suffix = f"_{mask_policy}_events_masked"
        mask_title = f"{mask_policy}-event arrivals masked"

    _plot_component_rows(
        processed_by_component,
        components,
        "post_mask",
        False,
        f"Aligned event stacks and median; {mask_title}\n{common_title}",
        (
            output_dir / f"stack_segments_overlay{filename_suffix}.png"
            if save_plots
            else None
        ),
        amplitude_limits=amplitude_limits,
        show_plot=show_plots,
    )
    if SHOW_MEDIAN_TRACE and not overlay_only:
        _plot_component_medians(
            processed_by_component,
            components,
            f"Aligned event-stack medians\n{common_title}",
            output_dir / "stack_segments_median.png" if save_plots else None,
            amplitude_limits=amplitude_limits,
            show_plot=show_plots,
        )
    if SHOW_OFFSET_TRACES and not overlay_only:
        _plot_component_rows(
            processed_by_component,
            components,
            "pre_mask",
            True,
            f"Aligned event stacks before masking\n{common_title}",
            output_dir / "stack_segments_offset_pre_mask.png" if save_plots else None,
            show_plot=show_plots,
        )
        _plot_component_rows(
            processed_by_component,
            components,
            "post_mask",
            True,
            f"Aligned event stacks after masking\n{common_title}",
            output_dir / "stack_segments_offset_post_mask.png" if save_plots else None,
            show_plot=show_plots,
        )
    return output_dir


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare per-event component stacks from one align_stack run."
    )
    parser.add_argument(
        "--run",
        default=DEFAULT_RUN,
        help=(
            "Run suffix after the implicit 2026 prefix, for example "
            f"0722_182322_4666 (default: {DEFAULT_RUN})."
        ),
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=DEFAULT_COMPONENTS,
        metavar="COMPONENT",
        help=(
            "Components to process, chosen from Z R T "
            f"(default: {' '.join(DEFAULT_COMPONENTS)})."
        ),
    )
    parser.add_argument(
        "--mask-other-events",
        choices=MASK_POLICIES,
        default=DEFAULT_MASK_OTHER_EVENTS,
        help=(
            "Which other-event arrivals to mask within each event stack "
            f"(default: {DEFAULT_MASK_OTHER_EVENTS})."
        ),
    )
    parser.add_argument(
        "--amplitude-limits",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=DEFAULT_AMPLITUDE_LIMITS,
        help=(
            "Y-axis limits for non-offset plots, clipping the visible "
            "amplitude range (default: -0.1 0.1)."
        ),
    )
    overlay_group = parser.add_mutually_exclusive_group()
    overlay_group.add_argument(
        "--overlay-only",
        dest="overlay_only",
        action="store_true",
        help="Create only the aligned overlay-and-median plot.",
    )
    overlay_group.add_argument(
        "--all-plots",
        dest="overlay_only",
        action="store_false",
        help="Create the overlay, median, and offset plots.",
    )
    show_group = parser.add_mutually_exclusive_group()
    show_group.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Show plots in interactive Matplotlib windows.",
    )
    show_group.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Do not show interactive Matplotlib windows.",
    )
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save",
        dest="save",
        action="store_true",
        help="Write PNG plot files.",
    )
    save_group.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not write PNG plot files.",
    )
    parser.set_defaults(
        overlay_only=DEFAULT_OVERLAY_ONLY,
        show=DEFAULT_SHOW_PLOTS,
        save=DEFAULT_SAVE_PLOTS,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    amplitude_limits = tuple(args.amplitude_limits)
    if amplitude_limits[0] >= amplitude_limits[1]:
        raise ValueError("--amplitude-limits requires MIN to be less than MAX")
    run_dir = resolve_run_directory(args.run)
    components = normalize_components(args.components)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"align_stack run directory not found: {run_dir}")

    print(f"Using align_stack run: {run_dir}")
    print(f"Processing components: {' '.join(components)}")
    parameters = load_run_parameters(run_dir)
    catalog = load_catalog(CATALOG_FILE)
    stacks_by_component = load_event_stacks(
        run_dir,
        components,
        catalog,
        parameters,
    )
    processed_by_component = {
        component: process_component_stacks(
            stacks_by_component[component],
            catalog,
            parameters,
            mask_policy=args.mask_other_events,
        )
        for component in components
    }
    output_dir = write_plots(
        processed_by_component,
        components,
        run_dir,
        parameters,
        mask_policy=args.mask_other_events,
        overlay_only=args.overlay_only,
        save_plots=args.save,
        show_plots=args.show,
        amplitude_limits=amplitude_limits,
    )
    if args.save:
        print(f"Stacker outputs: {output_dir}")


if __name__ == "__main__":
    main()
