"""Map original and M-relocated catalog event locations side by side."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator


DEFAULT_CATALOG_PATH = Path(
    "/Users/jvidale/Documents/Research/FaultScanR/event_sta_info/catalog_local_hand.xlsx"
)
DEFAULT_OUTPUT_PATH = Path(
    "/Users/jvidale/Documents/Research/FaultScanR/output/catalog_location_comparison.png"
)
REQUIRED_COLUMNS = frozenset(
    {"evid", "skip", "latitude", "longitude", "depth", "M lat", "M lon", "M depth"}
)
# MIN_LONGITUDE = -116.52
# MAX_LONGITUDE = -116.49
# MIN_LATITUDE = 33.48
# MAX_LATITUDE = 33.495
MIN_LONGITUDE = -116.505
MAX_LONGITUDE = -116.498
MIN_LATITUDE = 33.487
MAX_LATITUDE = 33.490
SHOW_CATALOG_LOCATIONS = False
SCALE_BAR_KM = 0.25
KILOMETERS_PER_DEGREE_LATITUDE = 111.32
TICK_INTERVAL_DEGREES = 0.001


def prepare_catalog_locations(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return catalog locations with numeric original and M coordinates."""
    missing = REQUIRED_COLUMNS - set(catalog.columns)
    if missing:
        raise ValueError(f"Catalog is missing required columns: {sorted(missing)}")

    locations = catalog.loc[
        :, ["evid", "skip", "latitude", "longitude", "depth", "M lat", "M lon", "M depth"]
    ].copy()
    locations["label"] = catalog["label"] if "label" in catalog else ""
    for column in ("skip", "latitude", "longitude", "depth", "M lat", "M lon", "M depth"):
        locations[column] = pd.to_numeric(locations[column], errors="coerce")
    locations["evid"] = locations["evid"].astype(str)
    locations["label"] = locations["label"].astype("string").fillna("").str.strip()
    locations["evid_suffix"] = locations["evid"].str[-4:]
    locations["has_original"] = locations[["latitude", "longitude"]].notna().all(axis=1)
    locations["has_m"] = locations[["M lat", "M lon"]].notna().all(axis=1)
    locations = locations.loc[locations["has_original"] | locations["has_m"]].copy()
    locations = locations.loc[~locations["evid"].str.startswith("CI_X")]
    return locations.loc[locations["skip"].isin([0, 1])].copy()


def event_annotation(row: pd.Series, include_evid: bool) -> str:
    """Return the catalog letter, optionally followed by the event suffix."""
    return f"{row['label']} {row['evid_suffix']}".strip() if include_evid else row["label"]


def format_decimal_degrees(value: float, _position: float) -> str:
    """Format decimal-degree ticks without an offset or scientific notation."""
    return f"{value:.3f}"


def map_title(show_catalog_locations: bool) -> str:
    """Return the title appropriate to the selected location set."""
    return "Catalog and M event locations" if show_catalog_locations else "Relocations"


def add_distance_scale_bar(ax: plt.Axes) -> None:
    """Draw a horizontal scale bar whose length is SCALE_BAR_KM at map latitude."""
    reference_latitude = (MIN_LATITUDE + MAX_LATITUDE) / 2
    longitude_degrees = SCALE_BAR_KM / (
        KILOMETERS_PER_DEGREE_LATITUDE * np.cos(np.deg2rad(reference_latitude))
    )
    latitude_degrees = SCALE_BAR_KM / KILOMETERS_PER_DEGREE_LATITUDE
    x_start = MIN_LONGITUDE + 0.06 * (MAX_LONGITUDE - MIN_LONGITUDE)
    y_position = MIN_LATITUDE + 0.06 * (MAX_LATITUDE - MIN_LATITUDE)
    tick_half_height = 0.12 * latitude_degrees

    ax.plot(
        [x_start, x_start + longitude_degrees],
        [y_position, y_position],
        color="black",
        linewidth=2,
        zorder=5,
    )
    for x_position in (x_start, x_start + longitude_degrees):
        ax.plot(
            [x_position, x_position],
            [y_position - tick_half_height, y_position + tick_half_height],
            color="black",
            linewidth=1.4,
            zorder=5,
        )
    ax.annotate(
        f"{SCALE_BAR_KM:g} km",
        (x_start + longitude_degrees / 2, y_position),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
    )


def reference_latitude() -> float:
    """Return the latitude used for local kilometre-per-degree conversions."""
    return (MIN_LATITUDE + MAX_LATITUDE) / 2


def kilometers_per_degree_longitude() -> float:
    """Return local east-west kilometres per degree of longitude."""
    return KILOMETERS_PER_DEGREE_LATITUDE * np.cos(np.deg2rad(reference_latitude()))


def plot_cross_section(
    ax: plt.Axes,
    locations: pd.DataFrame,
    coordinate: str,
    m_coordinate: str,
    coordinate_label: str,
    include_evid_labels: bool,
    show_catalog_locations: bool,
) -> None:
    """Plot one horizontal-coordinate versus depth comparison frame."""
    original = locations.loc[locations[[coordinate, "depth"]].notna().all(axis=1)]
    m_locations = locations.loc[locations[[m_coordinate, "M depth"]].notna().all(axis=1)]
    if show_catalog_locations:
        paired = locations.loc[
            locations[[coordinate, "depth", m_coordinate, "M depth"]].notna().all(axis=1)
        ]
        for _, row in paired.iterrows():
            ax.plot(
                [row[coordinate], row[m_coordinate]],
                [row["depth"], row["M depth"]],
                color="0.55",
                linewidth=0.8,
                zorder=1,
            )
        ax.scatter(
            original[coordinate],
            original["depth"],
            marker="o",
            s=38,
            color="tab:blue",
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
    ax.scatter(
        m_locations[m_coordinate],
        m_locations["M depth"],
        marker="o",
        s=40,
        color="tab:orange",
        edgecolors="white",
        linewidths=0.7,
        zorder=4,
    )
    if show_catalog_locations:
        for _, row in original.iterrows():
            annotation = event_annotation(row, include_evid_labels)
            if annotation:
                ax.annotate(annotation, (row[coordinate], row["depth"]), fontsize=7, color="tab:blue")
    for _, row in m_locations.iterrows():
        annotation = event_annotation(row, include_evid_labels)
        if annotation:
            ax.annotate(annotation, (row[m_coordinate], row["M depth"]), fontsize=7, color="tab:orange")

    ax.set_xlabel(coordinate_label)
    ax.set_ylabel("Depth (km)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(FuncFormatter(format_decimal_degrees))
    if coordinate == "latitude":
        ax.set_aspect(1 / KILOMETERS_PER_DEGREE_LATITUDE)
    else:
        ax.set_aspect(1 / kilometers_per_degree_longitude())
    ax.invert_yaxis()
    ax.grid(alpha=0.25)


def plot_catalog_locations(
    catalog: pd.DataFrame,
    output_path: Path,
    include_evid_labels: bool = False,
    show_catalog_locations: bool | None = None,
) -> Path:
    """Plot map and cross sections of original and M event locations."""
    if show_catalog_locations is None:
        show_catalog_locations = SHOW_CATALOG_LOCATIONS
    if "label" not in catalog and not include_evid_labels:
        raise ValueError(
            "Catalog is missing the 'label' column; add it or use --label-evid."
        )
    locations = prepare_catalog_locations(catalog)
    if locations.empty:
        raise ValueError("Catalog has no finite original or M latitude/longitude pairs.")

    fig = plt.figure(figsize=(14, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    map_ax = fig.add_subplot(grid[0, :])
    latitude_depth_ax = fig.add_subplot(grid[1, 0])
    longitude_depth_ax = fig.add_subplot(grid[1, 1])
    original = locations.loc[locations["has_original"]]
    m_locations = locations.loc[locations["has_m"]]
    if show_catalog_locations:
        paired = locations.loc[locations["has_original"] & locations["has_m"]]
        for _, row in paired.iterrows():
            map_ax.plot(
                [row["longitude"], row["M lon"]],
                [row["latitude"], row["M lat"]],
                color="0.55",
                linewidth=0.8,
                zorder=1,
            )
        map_ax.scatter(
            original["longitude"], original["latitude"], marker="o", s=46,
            color="tab:blue", edgecolors="white", linewidths=0.7,
            label="Catalog location", zorder=3,
        )
    map_ax.scatter(
        m_locations["M lon"], m_locations["M lat"], marker="o", s=48,
        color="tab:orange", edgecolors="white", linewidths=0.7,
        label="M location", zorder=4,
    )
    for entries, x_column, y_column, color, offset in (
        (original if show_catalog_locations else original.iloc[0:0], "longitude", "latitude", "tab:blue", (-4, 4)),
        (m_locations, "M lon", "M lat", "tab:orange", (4, -4)),
    ):
        for _, row in entries.iterrows():
            annotation = event_annotation(row, include_evid_labels)
            if annotation:
                map_ax.annotate(
                    annotation, (row[x_column], row[y_column]), xytext=offset,
                    textcoords="offset points", ha="right" if offset[0] < 0 else "left",
                    va="bottom" if offset[1] > 0 else "top", fontsize=7, color=color,
                )

    map_ax.set_xlim(MIN_LONGITUDE, MAX_LONGITUDE)
    map_ax.set_ylim(MIN_LATITUDE, MAX_LATITUDE)
    map_ax.set_aspect(1.0 / np.cos(np.deg2rad(reference_latitude())))
    add_distance_scale_bar(map_ax)
    map_ax.set_xlabel("Longitude")
    map_ax.set_ylabel("Latitude")
    map_ax.xaxis.set_major_locator(MultipleLocator(TICK_INTERVAL_DEGREES))
    map_ax.yaxis.set_major_locator(MultipleLocator(TICK_INTERVAL_DEGREES))
    map_ax.xaxis.set_major_formatter(FuncFormatter(format_decimal_degrees))
    map_ax.yaxis.set_major_formatter(FuncFormatter(format_decimal_degrees))
    map_ax.set_title(map_title(show_catalog_locations), fontweight="bold")
    map_ax.grid(alpha=0.25)

    plot_cross_section(
        latitude_depth_ax, locations, "latitude", "M lat", "Latitude",
        include_evid_labels, show_catalog_locations,
    )
    latitude_depth_ax.set_title("Latitude–depth", fontweight="bold")
    plot_cross_section(
        longitude_depth_ax, locations, "longitude", "M lon", "Longitude",
        include_evid_labels, show_catalog_locations,
    )
    longitude_depth_ax.set_title("Longitude–depth", fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--label-evid",
        action="store_true",
        help="Append each event's final four evid digits to its catalog letter label; "
        "use suffixes alone when no label column is present.",
    )
    parser.add_argument(
        "--m-only",
        action="store_true",
        help="Plot only M locations, omitting catalog points and location-pair links.",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    catalog = pd.read_excel(args.catalog)
    output_path = plot_catalog_locations(
        catalog,
        args.output,
        args.label_evid,
        show_catalog_locations=SHOW_CATALOG_LOCATIONS and not args.m_only,
    )
    print(f"Plotted {len(prepare_catalog_locations(catalog))} events: {output_path}")


if __name__ == "__main__":
    main()
