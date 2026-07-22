"""Plot radial versus transverse event-stack alignment shifts from the catalog."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CATALOG_PATH = Path(
    "/Users/jvidale/Documents/Research/FaultScanR/event_sta_info/catalog_local_hand.xlsx"
)
OUTPUT_PATH = Path(
    "/Users/jvidale/Documents/Research/FaultScanR/output/time_shift_R_vs_T.png"
)


def main() -> None:
    catalog = pd.read_excel(CATALOG_PATH)
    required = {"evid", "time shift", "time shift T"}
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")

    data = catalog.loc[:, ["evid", "time shift", "time shift T"]].copy()
    data["time shift"] = pd.to_numeric(data["time shift"], errors="coerce")
    data["time shift T"] = pd.to_numeric(data["time shift T"], errors="coerce")
    data = data.dropna(subset=["time shift", "time shift T"])
    if data.empty:
        raise ValueError("No events have both radial and transverse time shifts.")

    radial = data["time shift"].to_numpy()
    transverse = data["time shift T"].to_numpy()
    limit = max(0.05, float(np.max(np.abs(np.r_[radial, transverse]))))
    limit *= 1.12

    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    ax.scatter(radial, transverse, s=46, color="C0", edgecolor="white", linewidth=0.7)
    ax.plot([-limit, limit], [-limit, limit], color="0.35", linestyle="--", linewidth=1.2)
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    ax.axvline(0.0, color="0.75", linewidth=0.8)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Radial time shift (s)")
    ax.set_ylabel("Transverse time shift (s)")
    ax.set_title("S-wave event-stack shifts: radial vs transverse")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plotted {len(data)} events: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
