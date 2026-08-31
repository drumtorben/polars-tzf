"""Throughput benchmark for polars-tzf.

Run:  uv run python scripts/bench.py [--rows N] [--pool K]

The interesting comparison is `h3 unique` vs `h3 repeated`: the first is the
worst case for the dedup cache (every lookup misses, so the cache is pure
overhead), the second is the case it exists for. If `h3 unique` keeps pace with
the raw `lat/lng` baseline, the cache costs nothing when it cannot help.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time

import h3
import polars as pl

import polars_tzf as tzf

# Spread the sample over several timezones so the finder's stripe index is
# exercised rather than a single hot polygon.
METROS = [
    (52.5200, 13.4050),  # Berlin
    (40.7128, -74.0060),  # New York City
    (35.6762, 139.6503),  # Tokyo
    (-23.5505, -46.6333),  # Sao Paulo
    (19.0760, 72.8777),  # Mumbai
    (-33.8688, 151.2093),  # Sydney
]
RES = 9


def build_pool(size: int, seed: int = 0) -> list[int]:
    """`size` distinct H3 indices, as contiguous patches around each metro.

    Real H3 columns cover areas rather than scattered points, so the pool is
    built from grid disks. It also keeps generation in C: filling a large pool
    with per-point `latlng_to_cell` calls from Python takes minutes.
    """
    rng = random.Random(seed)
    per_metro = size // len(METROS) + 1
    # 1 + 3k(k+1) cells in a disk of radius k; solve for the k we need.
    k = 1
    while 1 + 3 * k * (k + 1) < per_metro:
        k += 1

    cells: list[int] = []
    for lat, lon in METROS:
        disk = h3.grid_disk(h3.latlng_to_cell(lat, lon, RES), k)
        cells.extend(h3.str_to_int(c) for c in disk[:per_metro])
    rng.shuffle(cells)
    return cells[:size]


def timed(label: str, frame: pl.DataFrame, expr: pl.Expr, rows: int) -> float:
    frame.head(1024).select(expr)  # warm the finder index and the cache
    start = time.perf_counter()
    out = frame.select(expr)
    elapsed = time.perf_counter() - start
    matched = out.to_series().null_count()
    print(
        f"  {label:<34} {elapsed:7.3f} s"
        f"   {rows / elapsed / 1e6:7.2f} M rows/s"
        f"   nulls={matched}"
    )
    return elapsed


def measure_one(distinct: int, rows: int) -> tuple[float, float]:
    """One sweep point, as (cold, warm) in M rows/s.

    The two differ by more than noise and answer different questions. `cold` is
    the first call on a fresh cache, so it pays to resolve every distinct cell;
    it is what a one-shot query costs. `warm` is the steady state once those
    cells are memoised, which is what repeated calls, many row groups, or a
    grouped context actually see.

    The memo is thread-local and lives as long as the process, so sweeping sizes
    in one process would measure each point against a cache the earlier points
    already filled. Each point therefore runs in a fresh interpreter.
    """
    pool = build_pool(distinct, seed=99)
    rng = random.Random(5)
    frame = pl.DataFrame(
        {"cell": [pool[rng.randrange(distinct)] for _ in range(rows)]},
        schema={"cell": pl.UInt64},
    )

    start = time.perf_counter()
    frame.select(tzf.tz_name_from_h3("cell"))
    cold = rows / (time.perf_counter() - start) / 1e6

    warm = max(
        rows / t / 1e6
        for t in (
            _time(lambda: frame.select(tzf.tz_name_from_h3("cell"))) for _ in range(2)
        )
    )
    return cold, warm


def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5_000_000)
    ap.add_argument("--pool", type=int, default=50_000)
    ap.add_argument("--sweep-point", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.sweep_point:
        cold, warm = measure_one(args.sweep_point, args.rows)
        print(f"{cold:.1f} {warm:.1f}")
        return

    print(f"building a pool of {args.pool:,} distinct cells at resolution {RES} ...")
    pool = build_pool(args.pool)

    rng = random.Random(1)
    repeated = [pool[rng.randrange(len(pool))] for _ in range(args.rows)]

    print(f"\nrows={args.rows:,}  distinct cells={len(pool):,}  "
          f"repetition={args.rows / len(pool):.0f}x\n")

    centres = [h3.cell_to_latlng(h3.int_to_str(c)) for c in pool]
    coords = pl.DataFrame(
        {
            "lat": [centres[rng.randrange(len(centres))][0] for _ in range(args.rows)],
            "lon": [centres[rng.randrange(len(centres))][1] for _ in range(args.rows)],
        }
    )
    rep_u64 = pl.DataFrame({"cell": repeated}, schema={"cell": pl.UInt64})
    uniq_u64 = pl.DataFrame({"cell": pool}, schema={"cell": pl.UInt64})
    rep_hex = rep_u64.select(pl.col("cell").map_elements(
        lambda c: h3.int_to_str(c), return_dtype=pl.String))

    print("baseline")
    timed("lat/lng", coords, tzf.tz_name("lat", "lon"), args.rows)

    print("\nh3 (cache worst case: every cell distinct)")
    timed("h3 uint64, unique", uniq_u64, tzf.tz_name_from_h3("cell"), len(pool))

    print("\nh3 (cache working)")
    timed("h3 uint64, repeated", rep_u64, tzf.tz_name_from_h3("cell"), args.rows)
    timed("h3 hex string, repeated", rep_hex, tzf.tz_name_from_h3("cell"), args.rows)

    print("\nalternative: dedup in polars instead of in rust")
    start = time.perf_counter()
    lut = (
        rep_u64.select("cell").unique()
        .with_columns(tz=tzf.tz_name_from_h3("cell"))
    )
    rep_u64.join(lut, on="cell", how="left")
    elapsed = time.perf_counter() - start
    print(f"  {'unique + lookup + join':<34} {elapsed:7.3f} s"
          f"   {args.rows / elapsed / 1e6:7.2f} M rows/s")

    # Where does the time in `h3 uint64, repeated` actually go? Once the cache
    # is warm the lookups are nearly free, and what remains is the cost of
    # materialising one string per row into the output column.
    print("\nbreakdown of the repeated case")
    distinct = rep_u64.select("cell").unique()
    distinct.head(1024).select(tzf.tz_name_from_h3("cell"))
    start = time.perf_counter()
    resolved = distinct.select(tz=tzf.tz_name_from_h3("cell"))["tz"]
    lookup_only = time.perf_counter() - start

    positions = pl.Series(
        "i", [rng.randrange(len(pool)) for _ in range(args.rows)], dtype=pl.UInt32
    )
    start = time.perf_counter()
    resolved.gather(positions)
    gather_only = time.perf_counter() - start

    print(f"  {'resolving the distinct cells':<34} {lookup_only:7.3f} s")
    print(f"  {'gathering one string per row':<34} {gather_only:7.3f} s")
    print(f"  {'-> lookups are no longer the cost':<34}")

    # Throughput tracks the number of *distinct* cells, not the row count: the
    # cache holds up to 2^19 of them per thread, and past that the extra cells
    # fall back to a full polygon lookup. The curve should sag, not cliff.
    # Throughput tracks the number of *distinct* cells, not the row count. The
    # cache holds up to 2^19 of them per thread; past that the surplus falls
    # back to a full polygon lookup, so the curve should sag, not cliff.
    print("\nsensitivity to the size of the working set (fresh process per row)")
    print(f"  {'distinct cells':>16} {'cold':>10} {'warm':>10}   (M rows/s)")
    for distinct in (20_000, 120_000, 400_000, 700_000, 1_500_000):
        cold, warm = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--sweep-point", str(distinct), "--rows", str(args.rows)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        print(f"  {distinct:>16,} {cold:>10} {warm:>10}")


if __name__ == "__main__":
    main()
