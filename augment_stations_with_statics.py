"""Add robust component statics for R- and T-shift runs to stations.xlsx."""

import argparse
from pathlib import Path

import pandas as pd

from plot_statics_by_station import (
    DEFAULT_STATICS_DIR,
    compute_event_baselines,
    compute_station_medians,
    load_statics,
)


DEFAULT_STATIONS_FILE = Path(
    "/Users/jvidale/Documents/Research/FaultScanR/event_sta_info/stations.xlsx"
)
STATIC_COLUMN = "median_event_baseline_corrected_static_seconds"


def station_static_medians(
    statics_dir: Path,
    phase: str,
    waveform_component: str,
    catalog_shift_component: str,
    mad_threshold: float,
) -> pd.DataFrame:
    """Return one robust station-static column for a component/basis pair."""
    statics = load_statics(
        statics_dir,
        component=waveform_component,
        phase=phase,
        catalog_shift_component=catalog_shift_component,
    )
    corrected, _event_baselines = compute_event_baselines(statics, mad_threshold)
    medians = compute_station_medians(corrected, mad_threshold)
    medians["station"] = medians["station"].astype(str).str.zfill(5)
    column_name = f"{waveform_component} static ({catalog_shift_component} correlation) s"
    return medians[["station", STATIC_COLUMN]].rename(columns={STATIC_COLUMN: column_name})


def augment_stations_workbook(
    stations_file: Path,
    statics_dir: Path,
    phase: str,
    mad_threshold: float,
    output_file: Path,
) -> Path:
    """Merge six robust station-static columns into the station coordinate workbook."""
    stations = pd.read_excel(stations_file, dtype={"station": str})
    if "station" not in stations.columns:
        raise ValueError(f"{stations_file} is missing the required 'station' column")
    stations["station"] = stations["station"].astype(str).str.zfill(5)

    augmented = stations.copy()
    for catalog_shift_component in ("R", "T"):
        for waveform_component in ("Z", "R", "T"):
            medians = station_static_medians(
                statics_dir=statics_dir,
                phase=phase,
                waveform_component=waveform_component,
                catalog_shift_component=catalog_shift_component,
                mad_threshold=mad_threshold,
            )
            column_name = medians.columns[-1]
            augmented = augmented.drop(columns=[column_name], errors="ignore").merge(
                medians,
                on="station",
                how="left",
                validate="one_to_one",
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_excel(output_file, index=False)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add R- and T-correlation station statics to stations.xlsx."
    )
    parser.add_argument("--stations-file", type=Path, default=DEFAULT_STATIONS_FILE)
    parser.add_argument("--statics-dir", type=Path, default=DEFAULT_STATICS_DIR)
    parser.add_argument("--phase", default="S")
    parser.add_argument("--mad-threshold", type=float, default=3.5)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_file = args.output or args.stations_file
    output = augment_stations_workbook(
        stations_file=args.stations_file,
        statics_dir=args.statics_dir,
        phase=args.phase.upper(),
        mad_threshold=args.mad_threshold,
        output_file=output_file,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
