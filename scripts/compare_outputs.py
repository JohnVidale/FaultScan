#!/usr/bin/env python3
"""Compare two output directories for regression checks.

Checks:
- File presence by relative path
- PNG file size deltas
- MSEED basic stats (trace count, npts, sampling_rate, max_abs)

Exit code is 0 when no differences are found, else 1.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from obspy import read


@dataclass
class MseedStats:
    trace_count: int
    npts_total: int
    sampling_rates: tuple[float, ...]
    max_abs: float


@dataclass
class ComparisonResult:
    ok: bool
    lines: list[str]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def list_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file():
            files[str(p.relative_to(root))] = p
    return files


def mseed_stats(path: Path) -> MseedStats:
    st = read(str(path))
    sampling_rates = tuple(float(tr.stats.sampling_rate) for tr in st)
    npts_total = int(sum(int(tr.stats.npts) for tr in st))
    max_abs = 0.0
    for tr in st:
        if tr.data.size == 0:
            continue
        candidate = float(abs(tr.data).max())
        if candidate > max_abs:
            max_abs = candidate
    return MseedStats(
        trace_count=len(st),
        npts_total=npts_total,
        sampling_rates=sampling_rates,
        max_abs=max_abs,
    )


def compare_dirs(
    baseline: Path,
    candidate: Path,
    png_size_tol_pct: float,
    mseed_amp_tol: float,
    check_hash: bool,
) -> ComparisonResult:
    lines: list[str] = []
    ok = True

    base_files = list_files(baseline)
    cand_files = list_files(candidate)

    base_set = set(base_files.keys())
    cand_set = set(cand_files.keys())

    missing = sorted(base_set - cand_set)
    extra = sorted(cand_set - base_set)

    if missing:
        ok = False
        lines.append("Missing files in candidate:")
        lines.extend(f"  - {m}" for m in missing)
    if extra:
        ok = False
        lines.append("Extra files in candidate:")
        lines.extend(f"  - {e}" for e in extra)

    common = sorted(base_set & cand_set)

    for rel in common:
        b = base_files[rel]
        c = cand_files[rel]

        if rel.lower().endswith(".png"):
            b_size = b.stat().st_size
            c_size = c.stat().st_size
            denom = max(1, b_size)
            pct = 100.0 * abs(c_size - b_size) / denom
            if pct > png_size_tol_pct:
                ok = False
                lines.append(
                    f"PNG size delta too large: {rel} baseline={b_size} candidate={c_size} delta={pct:.2f}%"
                )
            if check_hash and sha256_file(b) != sha256_file(c):
                lines.append(f"PNG hash changed: {rel}")

        elif rel.lower().endswith(".mseed"):
            bs = mseed_stats(b)
            cs = mseed_stats(c)

            if bs.trace_count != cs.trace_count:
                ok = False
                lines.append(
                    f"MSEED trace count changed: {rel} baseline={bs.trace_count} candidate={cs.trace_count}"
                )
            if bs.npts_total != cs.npts_total:
                ok = False
                lines.append(
                    f"MSEED npts_total changed: {rel} baseline={bs.npts_total} candidate={cs.npts_total}"
                )
            if bs.sampling_rates != cs.sampling_rates:
                ok = False
                lines.append(
                    f"MSEED sampling_rates changed: {rel} baseline={bs.sampling_rates} candidate={cs.sampling_rates}"
                )

            if not math.isclose(bs.max_abs, cs.max_abs, rel_tol=0.0, abs_tol=mseed_amp_tol):
                ok = False
                lines.append(
                    f"MSEED max_abs changed: {rel} baseline={bs.max_abs:.6g} candidate={cs.max_abs:.6g}"
                )

    if ok:
        lines.append("No regressions detected under current comparison rules.")

    return ComparisonResult(ok=ok, lines=lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="Baseline output directory")
    parser.add_argument("candidate", type=Path, help="Candidate output directory")
    parser.add_argument(
        "--png-size-tol-pct",
        type=float,
        default=2.0,
        help="Allowed PNG size delta percentage (default: 2.0)",
    )
    parser.add_argument(
        "--mseed-amp-abs-tol",
        type=float,
        default=1e-6,
        help="Allowed absolute tolerance for MSEED max amplitude (default: 1e-6)",
    )
    parser.add_argument(
        "--check-hash",
        action="store_true",
        help="Also report content hash differences for PNGs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = args.baseline.resolve()
    candidate = args.candidate.resolve()

    if not baseline.is_dir():
        print(f"Baseline directory not found: {baseline}")
        return 2
    if not candidate.is_dir():
        print(f"Candidate directory not found: {candidate}")
        return 2

    result = compare_dirs(
        baseline=baseline,
        candidate=candidate,
        png_size_tol_pct=float(args.png_size_tol_pct),
        mseed_amp_tol=float(args.mseed_amp_abs_tol),
        check_hash=bool(args.check_hash),
    )

    for line in result.lines:
        print(line)

    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
