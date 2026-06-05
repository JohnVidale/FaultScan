#!/usr/bin/env python3
"""Populate per-event snippet directories with trimmed seismograms."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime, read


CHANNELS = ("DP1", "DP2", "DPZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create event snippets from continuous waveforms and write them to per-event "
            "directories."
        )
    )
    parser.add_argument("excel_file", type=Path, help="Path to event catalog .xlsx")
    parser.add_argument("source_root", type=Path, help="Root directory containing station folders")
    parser.add_argument("snippets_root", type=Path, help="Root directory with per-event folders")
    parser.add_argument(
        "--event-column",
        default="evid",
        help="Catalog column containing event IDs (default: evid)",
    )
    parser.add_argument(
        "--origin-column",
        default="origin_time",
        help="Catalog column containing origin timestamps (default: origin_time)",
    )
    parser.add_argument(
        "--omit-prefix",
        default="CI_X",
        help="Skip events whose IDs start with this prefix (default: CI_X)",
    )
    parser.add_argument(
        "--pre-seconds",
        type=float,
        default=30.0,
        help="Seconds before origin to include (default: 30)",
    )
    parser.add_argument(
        "--post-seconds",
        type=float,
        default=60.0,
        help="Seconds after origin to include (default: 60)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing files",
    )
    return parser.parse_args()


def load_events(
    excel_file: Path,
    event_col: str,
    origin_col: str,
    omit_prefix: str,
) -> list[tuple[str, UTCDateTime]]:
    df = pd.read_excel(excel_file)
    if event_col not in df.columns:
        raise KeyError(f"Missing event column '{event_col}' in {excel_file}")
    if origin_col not in df.columns:
        raise KeyError(f"Missing origin column '{origin_col}' in {excel_file}")

    events: list[tuple[str, UTCDateTime]] = []
    seen: set[str] = set()

    for _, row in df.iterrows():
        evid_raw = row.get(event_col)
        origin_raw = row.get(origin_col)
        if pd.isna(evid_raw) or pd.isna(origin_raw):
            continue

        evid = str(evid_raw).strip()
        if not evid:
            continue
        if omit_prefix and evid.startswith(omit_prefix):
            continue
        if evid in seen:
            continue

        try:
            origin = UTCDateTime(str(origin_raw))
        except Exception:
            continue

        seen.add(evid)
        events.append((evid, origin))

    return events


def main() -> int:
    args = parse_args()

    if not args.excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {args.excel_file}")
    if not args.source_root.exists():
        raise FileNotFoundError(f"Source root not found: {args.source_root}")

    args.snippets_root.mkdir(parents=True, exist_ok=True)

    events = load_events(
        args.excel_file,
        event_col=args.event_column,
        origin_col=args.origin_column,
        omit_prefix=args.omit_prefix,
    )

    station_dirs = sorted([p for p in args.source_root.iterdir() if p.is_dir() and p.name.isdigit()])

    total_written = 0
    total_empty = 0
    total_missing = 0

    for evid, origin in events:
        event_dir = args.snippets_root / evid
        event_dir.mkdir(parents=True, exist_ok=True)
        start = origin - args.pre_seconds
        end = origin + args.post_seconds

        written_this_event = 0

        for sta_dir in station_dirs:
            for ch in CHANNELS:
                in_file = (
                    sta_dir
                    / f"{ch}.D"
                    / f"7V.{sta_dir.name}.00.{ch}.D.2022.273.down100.mseed"
                )
                if not in_file.exists():
                    total_missing += 1
                    continue

                if args.dry_run:
                    print(f"[dry-run] {evid}: {in_file}")
                    written_this_event += 1
                    continue

                try:
                    st = read(str(in_file), starttime=start, endtime=end)
                except Exception:
                    total_empty += 1
                    continue

                if len(st) == 0 or st[0].stats.npts == 0:
                    total_empty += 1
                    continue

                out_dir = event_dir / sta_dir.name / f"{ch}.D"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / in_file.name
                st.write(str(out_file), format="MSEED")
                written_this_event += 1
                total_written += 1

        print(f"{evid}: wrote {written_this_event} files")

    print(f"Events processed: {len(events)}")
    print(f"Files written: {total_written}")
    print(f"Missing source files skipped: {total_missing}")
    print(f"Empty/unreadable slices skipped: {total_empty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())