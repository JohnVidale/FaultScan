"""Compare R- and T-correlation station static corrections directly."""

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_STATIONS_FILE = Path(
    "/Users/jvidale/Documents/Research/FaultScanR/event_sta_info/stations.xlsx"
)
DEFAULT_OUTPUT_DIR = Path("/Users/jvidale/Documents/Research/FaultScanR/output/Statics")


def station_sort_key(station_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(station_id))
    except ValueError:
        return (1, station_id)


def plot_component_comparison(
    stations: pd.DataFrame,
    waveform_component: str,
    output_dir: Path,
) -> Path:
    """Scatter R- against T-correlation static estimates for every station."""
    r_column = f"{waveform_component} static (R correlation) s"
    t_column = f"{waveform_component} static (T correlation) s"
    missing = {r_column, t_column} - set(stations.columns)
    if missing:
        raise ValueError(f"stations workbook is missing columns: {sorted(missing)}")

    r_values = pd.to_numeric(stations[r_column], errors="coerce").to_numpy()
    t_values = pd.to_numeric(stations[t_column], errors="coerce").to_numpy()
    finite = np.isfinite(r_values) & np.isfinite(t_values)
    r_values = r_values[finite]
    t_values = t_values[finite]
    if not len(r_values):
        raise ValueError(f"No paired R/T static values are available for component {waveform_component}")

    limit = max(np.max(np.abs(r_values)), np.max(np.abs(t_values)))
    limit = 1.1 * limit if limit > 0.0 else 0.01
    correlation = float(np.corrcoef(r_values, t_values)[0, 1]) if len(r_values) > 1 else np.nan

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(
        r_values,
        t_values,
        s=15,
        color="tab:purple",
        alpha=0.8,
        linewidths=0,
        label=f"{len(r_values)} stations",
    )
    ax.plot([-limit, limit], [-limit, limit], color="0.3", lw=1.0, linestyle="--", label="1:1")
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.axvline(0.0, color="0.65", lw=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Static from R correlation (s)")
    ax.set_ylabel("Static from T correlation (s)")
    ax.set_title(
        f"{waveform_component} component station statics: R vs T correlation",
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {correlation:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{waveform_component.lower()}_station_statics_R_vs_T_correlation.png"
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot R- and T-correlation station statics for Z, R, and T components."
    )
    parser.add_argument("--stations-file", type=Path, default=DEFAULT_STATIONS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    stations = pd.read_excel(args.stations_file, dtype={"station": str})
    if "station" not in stations.columns:
        raise ValueError(f"{args.stations_file} is missing the required 'station' column")

    for waveform_component in ("Z", "R", "T"):
        output_file = plot_component_comparison(
            stations,
            waveform_component,
            args.output_dir,
        )
        print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
