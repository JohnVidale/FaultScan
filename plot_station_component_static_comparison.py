"""Compare Z, R, and T station statics within each correlation basis."""

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
COMPONENT_PAIRS = (("R", "T"), ("R", "Z"), ("T", "Z"))


def static_column(component: str, correlation_basis: str) -> str:
    return f"{component} static ({correlation_basis} correlation) s"


def plot_component_pair(
    stations: pd.DataFrame,
    correlation_basis: str,
    x_component: str,
    y_component: str,
    output_dir: Path,
) -> Path:
    """Scatter paired station statics for two waveform components."""
    x_column = static_column(x_component, correlation_basis)
    y_column = static_column(y_component, correlation_basis)
    missing = {x_column, y_column} - set(stations.columns)
    if missing:
        raise ValueError(f"stations workbook is missing columns: {sorted(missing)}")

    x_values = pd.to_numeric(stations[x_column], errors="coerce").to_numpy()
    y_values = pd.to_numeric(stations[y_column], errors="coerce").to_numpy()
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if not len(x_values):
        raise ValueError(
            f"No paired {x_component}/{y_component} values for {correlation_basis} correlation"
        )

    limit = max(np.max(np.abs(x_values)), np.max(np.abs(y_values)))
    limit = 1.1 * limit if limit > 0.0 else 0.01
    correlation = float(np.corrcoef(x_values, y_values)[0, 1]) if len(x_values) > 1 else np.nan

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(x_values, y_values, s=15, color="tab:green", alpha=0.8, linewidths=0)
    ax.plot([-limit, limit], [-limit, limit], color="0.3", lw=1.0, linestyle="--", label="1:1")
    ax.axhline(0.0, color="0.65", lw=0.8)
    ax.axvline(0.0, color="0.65", lw=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{x_component} component static ({correlation_basis} correlation) (s)")
    ax.set_ylabel(f"{y_component} component static ({correlation_basis} correlation) (s)")
    ax.set_title(
        f"{correlation_basis}-correlation station statics: {x_component} vs {y_component}",
        fontweight="bold",
    )
    ax.text(
        0.03,
        0.97,
        f"{len(x_values)} stations\nPearson r = {correlation:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (
        f"shift{correlation_basis}_{x_component}_vs_{y_component}_station_statics.png"
    )
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Z, R, and T station statics within R and T correlation bases."
    )
    parser.add_argument("--stations-file", type=Path, default=DEFAULT_STATIONS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    stations = pd.read_excel(args.stations_file)
    for correlation_basis in ("R", "T"):
        for x_component, y_component in COMPONENT_PAIRS:
            output_file = plot_component_pair(
                stations,
                correlation_basis,
                x_component,
                y_component,
                args.output_dir,
            )
            print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
