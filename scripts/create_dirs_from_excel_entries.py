#!/usr/bin/env python3
"""Create one directory per entry from an Excel column."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


def sanitize_name(raw: str) -> str:
    """Return a filesystem-safe directory name."""
    name = INVALID_CHARS.sub("_", raw.strip())
    name = name.rstrip(". ")
    return name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create directories in a target folder from an Excel column.",
    )
    parser.add_argument("excel_file", type=Path, help="Path to the .xlsx file")
    parser.add_argument("target_dir", type=Path, help="Directory where folders are created")
    parser.add_argument(
        "--column",
        default="evid",
        help="Column name to use for folder names (default: evid)",
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Sheet name or index to read (default: 0, first sheet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print directories that would be created without creating them",
    )
    parser.add_argument(
        "--omit-prefix",
        default="CI_X",
        help="Skip entries starting with this prefix (default: CI_X)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.excel_file.exists():
        raise FileNotFoundError(f"Excel file not found: {args.excel_file}")

    args.target_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.excel_file, sheet_name=args.sheet)
    if args.column not in df.columns:
        raise KeyError(
            f"Column '{args.column}' not found. Available columns: {list(df.columns)}"
        )

    entries = (
        df[args.column]
        .dropna()
        .astype(str)
        .map(str.strip)
    )
    entries = entries[entries != ""]

    omitted = 0
    if args.omit_prefix:
        mask = entries.str.startswith(args.omit_prefix)
        omitted = int(mask.sum())
        entries = entries[~mask]

    unique_names: list[str] = []
    seen: set[str] = set()
    for raw in entries:
        safe_name = sanitize_name(raw)
        if not safe_name or safe_name in seen:
            continue
        seen.add(safe_name)
        unique_names.append(safe_name)

    created = 0
    for name in unique_names:
        folder = args.target_dir / name
        if args.dry_run:
            print(f"[dry-run] {folder}")
            continue
        if not folder.exists():
            folder.mkdir(parents=False, exist_ok=False)
            created += 1

    print(f"Entries read: {len(entries)}")
    if args.omit_prefix:
        print(f"Entries omitted by prefix '{args.omit_prefix}': {omitted}")
    print(f"Unique folder names: {len(unique_names)}")
    if args.dry_run:
        print("No folders created (dry-run).")
    else:
        print(f"Folders created: {created}")
        print(f"Folders already existed: {len(unique_names) - created}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())