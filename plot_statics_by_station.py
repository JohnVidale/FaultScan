import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_STATICS_DIR = Path("/Users/jvidale/Documents/Research/FaultScanR/output/Statics")
STATIC_COLUMN = "shift_relative_to_predicted_seconds"
CORRECTED_STATIC_COLUMN = "event_baseline_corrected_static_seconds"


def robust_sigma(values: pd.Series) -> float:
    """Estimate scatter as 1.4826 * MAD."""
    x = values.astype(float).to_numpy()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    return 1.4826 * mad


def station_sort_key(station_id: str):
    try:
        return (0, int(station_id))
    except ValueError:
        return (1, station_id)


def load_statics(statics_dir: Path) -> pd.DataFrame:
    files = sorted(statics_dir.glob("*_xcorr_statics.xlsx"))
    if not files:
        raise FileNotFoundError(f"No statics workbooks found in {statics_dir}")

    frames = []
    for path in files:
        df = pd.read_excel(path)
        required = {"event_id", "station", STATIC_COLUMN}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
        df = df.copy()
        df["source_file"] = path.name
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["station"] = out["station"].astype(str)
    out = out[np.isfinite(out[STATIC_COLUMN].astype(float))]
    return out


def robust_inlier_mask(values: pd.Series, mad_threshold: float) -> np.ndarray:
    """Return inliers using a median absolute deviation cutoff."""
    x = values.astype(float).to_numpy()
    finite = np.isfinite(x)
    mask = np.zeros(len(x), dtype=bool)
    if not np.any(finite):
        return mask

    finite_x = x[finite]
    median = float(np.median(finite_x))
    mad = float(np.median(np.abs(finite_x - median)))
    if mad == 0.0:
        mask[finite] = finite_x == median
        if not np.any(mask):
            mask[finite] = True
        return mask

    modified_z = 0.6745 * (finite_x - median) / mad
    mask[finite] = np.abs(modified_z) <= mad_threshold
    return mask


def compute_event_baselines(
    df: pd.DataFrame,
    mad_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    corrected = df.copy()
    corrected["event_baseline_static_seconds"] = np.nan
    corrected["event_baseline_inlier"] = False
    corrected[CORRECTED_STATIC_COLUMN] = np.nan

    baseline_rows = []
    for event_id, sub in corrected.groupby("event_id", sort=True):
        inlier_mask = robust_inlier_mask(sub[STATIC_COLUMN], mad_threshold)
        inlier_values = sub.loc[inlier_mask, STATIC_COLUMN].astype(float)
        all_values = sub[STATIC_COLUMN].astype(float)
        baseline = float(inlier_values.median()) if len(inlier_values) else float(all_values.median())
        baseline_sigma = robust_sigma(inlier_values)
        baseline_uncertainty = baseline_sigma / np.sqrt(len(inlier_values)) if len(inlier_values) else np.nan

        corrected.loc[sub.index, "event_baseline_static_seconds"] = baseline
        corrected.loc[sub.index, "event_baseline_inlier"] = inlier_mask
        corrected.loc[sub.index, CORRECTED_STATIC_COLUMN] = all_values - baseline

        baseline_rows.append(
            {
                "event_id": event_id,
                "event_baseline_static_seconds": baseline,
                "event_baseline_uncertainty_seconds": baseline_uncertainty,
                "event_baseline_robust_sigma_seconds": baseline_sigma,
                "n_statics": int(len(sub)),
                "n_inliers": int(np.sum(inlier_mask)),
                "n_outliers": int(len(sub) - np.sum(inlier_mask)),
                "raw_median_static_seconds": float(all_values.median()),
                "raw_mean_static_seconds": float(all_values.mean()),
                "raw_std_static_seconds": float(all_values.std(ddof=0)),
                "inlier_q25_static_seconds": float(inlier_values.quantile(0.25)) if len(inlier_values) else np.nan,
                "inlier_q75_static_seconds": float(inlier_values.quantile(0.75)) if len(inlier_values) else np.nan,
                "mad_threshold": float(mad_threshold),
            }
        )

    return corrected, pd.DataFrame(baseline_rows)


def compute_station_medians(
    corrected: pd.DataFrame,
    mad_threshold: float,
) -> pd.DataFrame:
    rows = []
    for station, sub in corrected.groupby("station", sort=False):
        inlier_mask = robust_inlier_mask(sub[CORRECTED_STATIC_COLUMN], mad_threshold)
        inlier_values = sub.loc[inlier_mask, CORRECTED_STATIC_COLUMN].astype(float)
        all_values = sub[CORRECTED_STATIC_COLUMN].astype(float)
        median_static = float(inlier_values.median()) if len(inlier_values) else float(all_values.median())
        station_sigma = robust_sigma(inlier_values)
        station_uncertainty = station_sigma / np.sqrt(len(inlier_values)) if len(inlier_values) else np.nan

        rows.append(
            {
                "station": station,
                "median_event_baseline_corrected_static_seconds": median_static,
                "static_uncertainty_seconds": station_uncertainty,
                "static_robust_sigma_seconds": station_sigma,
                "n_events": int(sub["event_id"].nunique()),
                "n_statics": int(len(sub)),
                "n_inliers": int(np.sum(inlier_mask)),
                "n_outliers": int(len(sub) - np.sum(inlier_mask)),
                "uncorrected_median_static_seconds": float(sub[STATIC_COLUMN].astype(float).median()),
                "corrected_mean_static_seconds": float(all_values.mean()),
                "corrected_std_static_seconds": float(all_values.std(ddof=0)),
                "corrected_q25_static_seconds": float(inlier_values.quantile(0.25)) if len(inlier_values) else np.nan,
                "corrected_q75_static_seconds": float(inlier_values.quantile(0.75)) if len(inlier_values) else np.nan,
                "mad_threshold": float(mad_threshold),
            }
        )

    station_df = pd.DataFrame(rows)
    station_df = station_df.sort_values("station", key=lambda s: s.map(station_sort_key))
    return station_df


def write_static_summary_workbook(
    corrected: pd.DataFrame,
    event_baselines: pd.DataFrame,
    station_medians: pd.DataFrame,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file) as writer:
        event_baselines.to_excel(writer, sheet_name="event_baselines", index=False)
        station_medians.to_excel(writer, sheet_name="station_medians", index=False)
        corrected.to_excel(writer, sheet_name="corrected_statics", index=False)


def write_station_statics_workbook(station_medians: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    station_medians.to_excel(output_file, index=False)


def write_event_baselines_workbook(event_baselines: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    event_baselines.to_excel(output_file, index=False)


def plot_statics_by_station(
    df: pd.DataFrame,
    output_file: Path,
    y_column: str = STATIC_COLUMN,
    y_label: str = "Static relative to predicted arrival (s)",
    title_prefix: str = "Radial S-wave statics",
) -> None:
    stations = sorted(df["station"].unique(), key=station_sort_key)
    station_to_x = {station: idx for idx, station in enumerate(stations)}

    fig_width = max(12.0, min(32.0, len(stations) * 0.22))
    fig, ax = plt.subplots(figsize=(fig_width, 7.0))

    for station in stations:
        sub = df[df["station"] == station]
        x0 = station_to_x[station]
        rng = np.random.default_rng(abs(hash(station)) % (2**32))
        jitter = rng.uniform(-0.18, 0.18, size=len(sub))
        x = np.full(len(sub), x0, dtype=float) + jitter
        y = sub[y_column].astype(float).to_numpy()

        if "passed_window_correlation" in sub.columns:
            passed = sub["passed_window_correlation"].astype(bool).to_numpy()
            if np.any(passed):
                ax.scatter(x[passed], y[passed], s=14, c="black", alpha=0.65, linewidths=0)
            if np.any(~passed):
                ax.scatter(x[~passed], y[~passed], s=18, c="red", alpha=0.75, linewidths=0)
        else:
            ax.scatter(x, y, s=14, c="black", alpha=0.65, linewidths=0)

    medians = df.groupby("station")[y_column].median()
    median_x = [station_to_x[s] for s in medians.index if s in station_to_x]
    median_y = [medians[s] for s in medians.index if s in station_to_x]
    ax.scatter(median_x, median_y, s=22, c="dodgerblue", marker="_", linewidths=1.5, zorder=4)

    ax.axhline(0.0, color="0.45", lw=1.0, linestyle="--", alpha=0.8)
    ax.set_xticks(range(len(stations)))
    ax.set_xticklabels(stations, rotation=90, fontsize=7)
    ax.set_xlabel("Station")
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_prefix} by station ({df['event_id'].nunique()} events)")
    ax.grid(axis="y", alpha=0.25)
    ax.margins(x=0.01)
    fig.tight_layout()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all radial S-wave statics by station.")
    parser.add_argument("--statics-dir", type=Path, default=DEFAULT_STATICS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--station-output", type=Path, default=None)
    parser.add_argument("--event-output", type=Path, default=None)
    parser.add_argument("--corrected-output", type=Path, default=None)
    parser.add_argument("--mad-threshold", type=float, default=3.5)
    args = parser.parse_args()

    statics_dir = args.statics_dir
    output_file = args.output or (statics_dir / "radial_s_statics_by_station.png")
    summary_file = args.summary_output or (statics_dir / "radial_s_static_baseline_summary.xlsx")
    station_file = args.station_output or (statics_dir / "radial_s_station_statics.xlsx")
    event_file = args.event_output or (statics_dir / "radial_s_event_baseline_shifts.xlsx")
    corrected_plot_file = args.corrected_output or (
        statics_dir / "radial_s_event_baseline_corrected_statics_by_station.png"
    )

    df = load_statics(statics_dir)
    plot_statics_by_station(df, output_file)
    corrected, event_baselines = compute_event_baselines(df, args.mad_threshold)
    station_medians = compute_station_medians(corrected, args.mad_threshold)
    write_static_summary_workbook(corrected, event_baselines, station_medians, summary_file)
    write_station_statics_workbook(station_medians, station_file)
    write_event_baselines_workbook(event_baselines, event_file)
    plot_statics_by_station(
        corrected,
        corrected_plot_file,
        y_column=CORRECTED_STATIC_COLUMN,
        y_label="Event-baseline-corrected static (s)",
        title_prefix="Event-baseline-corrected radial S-wave statics",
    )
    print(f"Wrote {output_file} from {len(df)} statics in {df['event_id'].nunique()} events")
    print(f"Wrote {summary_file}")
    print(f"Wrote {station_file}")
    print(f"Wrote {event_file}")
    print(f"Wrote {corrected_plot_file}")


if __name__ == "__main__":
    main()
