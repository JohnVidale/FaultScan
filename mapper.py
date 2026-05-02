#!/usr/bin/env python3
"""Map USGS seismicity near Pinyon Flat Observatory for a selected year.

Default: 2025 earthquakes within 50 km of PFO, using the USGS FDSN Event
Web Service. Outputs a PNG map, CSV catalog, and JSON summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

# PFO / Pinyon Flat Observatory coordinates from FDSN station metadata.
PFO_LAT = 33.6092
PFO_LON = -116.4550
EARTH_RADIUS_KM = 6371.0088
USGS_EVENT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
OUTPUT_DIR = Path("/Users/vidale/Documents/Research/Mingze_SJF/output")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def circle_lon_lat(lat0_deg: float, lon0_deg: float, radius_km: float) -> tuple[list[float], list[float]]:
    """Return lon/lat arrays for a geodesic radius circle."""
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)
    angular_radius = radius_km / EARTH_RADIUS_KM
    lats: list[float] = []
    lons: list[float] = []

    for bearing in np.linspace(0.0, 2.0 * np.pi, 361):
        lat = math.asin(
            math.sin(lat0) * math.cos(angular_radius)
            + math.cos(lat0) * math.sin(angular_radius) * math.cos(bearing)
        )
        lon = lon0 + math.atan2(
            math.sin(bearing) * math.sin(angular_radius) * math.cos(lat0),
            math.cos(angular_radius) - math.sin(lat0) * math.sin(lat),
        )
        lats.append(math.degrees(lat))
        lons.append(math.degrees(lon))

    return lons, lats


def fetch_usgs_events(
    year: int,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    min_magnitude: float | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Fetch events from USGS and return the query URL and simplified rows."""
    params: dict[str, Any] = {
        "format": "geojson",
        "starttime": f"{year}-01-01",
        "endtime": f"{year + 1}-01-01",
        "latitude": center_lat,
        "longitude": center_lon,
        "maxradiuskm": radius_km,
        "orderby": "time-asc",
        "limit": 20000,
    }
    if min_magnitude is not None:
        params["minmagnitude"] = min_magnitude

    url = USGS_EVENT_URL + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)

    rows: list[dict[str, Any]] = []
    for feature in data["features"]:
        properties = feature["properties"]
        lon, lat, depth = feature["geometry"]["coordinates"][:3]
        rows.append(
            {
                "time": datetime.fromtimestamp(
                    properties["time"] / 1000.0, tz=timezone.utc
                ).isoformat(),
                "latitude": lat,
                "longitude": lon,
                "depth_km": depth,
                "magnitude": properties.get("mag"),
                "magType": properties.get("magType"),
                "place": properties.get("place"),
                "event_id": feature["id"],
                "url": properties.get("url"),
                "distance_km": haversine_km(center_lat, center_lon, lat, lon),
                "type": properties.get("type"),
            }
        )

    return url, rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "time",
        "latitude",
        "longitude",
        "depth_km",
        "magnitude",
        "magType",
        "place",
        "event_id",
        "url",
        "distance_km",
        "type",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_map(
    rows: list[dict[str, Any]],
    path: Path,
    year: int,
    center_lat: float,
    center_lon: float,
    radius_km: float,
    min_magnitude: float | None,
) -> None:
    if not rows:
        raise ValueError("No events returned; nothing to plot.")

    lats = np.array([row["latitude"] for row in rows], dtype=float)
    lons = np.array([row["longitude"] for row in rows], dtype=float)
    depths = np.array([row["depth_km"] for row in rows], dtype=float)
    mags = np.array(
        [np.nan if row["magnitude"] is None else row["magnitude"] for row in rows],
        dtype=float,
    )
    circle_lons, circle_lats = circle_lon_lat(center_lat, center_lon, radius_km)

    fig = plt.figure(figsize=(11, 8.5))
    grid = fig.add_gridspec(
        2, 2, height_ratios=[3, 1], width_ratios=[3, 1], hspace=0.28, wspace=0.25
    )

    ax = fig.add_subplot(grid[0, 0])
    sizes = 10.0 + 18.0 * np.clip(mags + 1.0, 0.0, None) ** 1.4
    scatter = ax.scatter(
        lons,
        lats,
        c=depths,
        s=sizes,
        cmap="viridis_r",
        alpha=0.72,
        edgecolor="k",
        linewidth=0.15,
    )
    ax.plot(circle_lons, circle_lats, "r--", lw=1.6, label=f"{radius_km:g} km radius")
    ax.scatter(
        [center_lon],
        [center_lat],
        marker="*",
        s=220,
        c="red",
        edgecolor="k",
        label="Pinyon Flat/PFO",
    )

    for name, lat, lon in [
        ("Anza", 33.555, -116.673),
        ("Idyllwild", 33.740, -116.718),
        ("Borrego Springs", 33.256, -116.375),
        ("Palm Desert", 33.723, -116.376),
    ]:
        ax.text(
            lon,
            lat,
            name,
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "none", "pad": 1},
        )

    min_mag_text = "none" if min_magnitude is None else f"M≥{min_magnitude:g}"
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"USGS seismicity in {year} within {radius_km:.0f} km of Pinyon Flat Observatory\n"
        f"N={len(rows)} events, center=({center_lat:.4f}, {center_lon:.4f}), minmag={min_mag_text}"
    )
    ax.set_aspect(1.0 / np.cos(np.deg2rad(center_lat)))
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=9)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.85)
    colorbar.set_label("Depth (km)")

    ax_depth = fig.add_subplot(grid[0, 1])
    ax_depth.hist(
        depths,
        bins=np.arange(math.floor(np.nanmin(depths)), math.ceil(np.nanmax(depths)) + 1, 1),
        color="0.35",
    )
    ax_depth.set_xlabel("Count")
    ax_depth.set_ylabel("Depth (km)")
    ax_depth.invert_yaxis()
    ax_depth.grid(alpha=0.25)
    ax_depth.set_title("Depth histogram")

    ax_month = fig.add_subplot(grid[1, 0])
    months = np.array([datetime.fromisoformat(row["time"]).month for row in rows])
    monthly_counts = [int(np.sum(months == month)) for month in range(1, 13)]
    ax_month.bar(range(1, 13), monthly_counts, color="0.25")
    ax_month.set_xlim(0.5, 12.5)
    ax_month.set_xticks(range(1, 13))
    ax_month.set_xlabel(f"{year} month")
    ax_month.set_ylabel("Events/month")
    ax_month.grid(axis="y", alpha=0.25)

    ax_mag = fig.add_subplot(grid[1, 1])
    finite_mags = mags[~np.isnan(mags)]
    ax_mag.hist(
        finite_mags,
        bins=np.arange(
            np.floor(np.nanmin(finite_mags) * 2) / 2,
            np.ceil(np.nanmax(finite_mags) * 2) / 2 + 0.5,
            0.25,
        ),
        color="0.35",
    )
    ax_mag.set_xlabel("Magnitude")
    ax_mag.set_ylabel("Count")
    ax_mag.grid(alpha=0.25)
    ax_mag.set_title("Magnitude histogram")

    fig.text(
        0.01,
        0.01,
        "Data: USGS FDSN Event Web Service; no minimum magnitude filter unless stated.",
        fontsize=8,
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")


def write_summary(
    rows: list[dict[str, Any]], query_url: str, path: Path, year: int, center_lat: float, center_lon: float, radius_km: float
) -> None:
    mags = np.array(
        [np.nan if row["magnitude"] is None else row["magnitude"] for row in rows],
        dtype=float,
    )
    depths = np.array([row["depth_km"] for row in rows], dtype=float)
    months = np.array([datetime.fromisoformat(row["time"]).month for row in rows])
    summary = {
        "query_url": query_url,
        "n_events": len(rows),
        "center_lat": center_lat,
        "center_lon": center_lon,
        "radius_km": radius_km,
        "time_start": f"{year}-01-01",
        "time_end": f"{year + 1}-01-01",
        "mag_min": None if len(rows) == 0 else float(np.nanmin(mags)),
        "mag_max": None if len(rows) == 0 else float(np.nanmax(mags)),
        "depth_min_km": None if len(rows) == 0 else float(np.nanmin(depths)),
        "depth_max_km": None if len(rows) == 0 else float(np.nanmax(depths)),
        "largest_events": sorted(
            rows, key=lambda row: -999.0 if row["magnitude"] is None else -float(row["magnitude"])
        )[:10],
        "monthly_counts": [int(np.sum(months == month)) for month in range(1, 13)],
    }
    path.write_text(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--radius-km", type=float, default=50.0)
    parser.add_argument("--lat", type=float, default=PFO_LAT)
    parser.add_argument("--lon", type=float, default=PFO_LON)
    parser.add_argument("--min-magnitude", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"seismicity_{args.year}_pinyon_flats_{args.radius_km:g}km"
    query_url, rows = fetch_usgs_events(
        args.year, args.lat, args.lon, args.radius_km, args.min_magnitude
    )

    csv_path = args.output_dir / f"{stem}_usgs.csv"
    png_path = args.output_dir / f"{stem}_map.png"
    summary_path = args.output_dir / f"{stem}_summary.json"

    write_csv(rows, csv_path)
    plot_map(rows, png_path, args.year, args.lat, args.lon, args.radius_km, args.min_magnitude)
    write_summary(rows, query_url, summary_path, args.year, args.lat, args.lon, args.radius_km)

    print(f"Wrote {png_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")
    print(f"N={len(rows)}")
    print(f"Query: {query_url}")


if __name__ == "__main__":
    main()
