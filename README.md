# polars-tzf

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)

Fast timezone lookup for [Polars](https://pola.rs): resolve IANA timezone names from latitude/longitude coordinates or H3 cells, powered by [tzf-rs](https://github.com/ringsaturn/tzf-rs).

Runs as a native Polars expression plugin written in Rust — no Python loops, no `map_elements`, fully parallelized by the Polars engine.

## Installation

```bash
pip install polars-tzf
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add polars-tzf
```

## Usage

```python
import polars as pl
import polars_tzf as tzf

df = pl.DataFrame(
    {
        "place": ["Berlin", "New York City", "Tokyo", "São Paulo", "Null Island"],
        "lat": [52.5200, 40.7128, 35.6762, -23.5505, 0.0],
        "lon": [13.4050, -74.0060, 139.6503, -46.6333, 0.0],
    }
)

df.with_columns(tz=tzf.tz_name("lat", "lon"))
```

```text
shape: (5, 4)
┌───────────────┬──────────┬───────────┬───────────────────┐
│ place         ┆ lat      ┆ lon       ┆ tz                │
│ ---           ┆ ---      ┆ ---       ┆ ---               │
│ str           ┆ f64      ┆ f64       ┆ str               │
╞═══════════════╪══════════╪═══════════╪═══════════════════╡
│ Berlin        ┆ 52.52    ┆ 13.405    ┆ Europe/Berlin     │
│ New York City ┆ 40.7128  ┆ -74.006   ┆ America/New_York  │
│ Tokyo         ┆ 35.6762  ┆ 139.6503  ┆ Asia/Tokyo        │
│ São Paulo     ┆ -23.5505 ┆ -46.6333  ┆ America/Sao_Paulo │
│ Null Island   ┆ 0.0      ┆ 0.0       ┆ Etc/GMT           │
└───────────────┴──────────┴───────────┴───────────────────┘
```

If your data is already indexed with [H3](https://h3geo.org) cells, resolve
straight from the cell — no coordinate columns needed:

```python
df = pl.DataFrame(
    {
        "cell": [
            "871f1d489ffffff",
            "872a1072cffffff",
            "872f5a363ffffff",
        ],
    }
)

df.with_columns(tz=tzf.tz_name_from_h3("cell"))
```

```text
shape: (3, 2)
┌─────────────────┬──────────────────┐
│ cell            ┆ tz               │
│ ---             ┆ ---              │
│ str             ┆ str              │
╞═════════════════╪══════════════════╡
│ 871f1d489ffffff ┆ Europe/Berlin    │
│ 872a1072cffffff ┆ America/New_York │
│ 872f5a363ffffff ┆ Asia/Tokyo       │
└─────────────────┴──────────────────┘
```

It works anywhere a Polars expression works, including lazy queries:

```python
(
    pl.scan_parquet("events.parquet")
    .with_columns(tz=tzf.tz_name("latitude", "longitude"))
    .collect()
)
```

## API

### `tz_name(lat, lng) -> pl.Expr`

Resolves each `(lat, lng)` pair to an IANA timezone name (e.g. `"Europe/Berlin"`).

- **`lat`**, **`lng`** — column names, expressions, or anything else Polars accepts as an expression input (`IntoExpr`). Both must be `Float64`.
- **Returns** a `String` expression. The operation is elementwise, so it parallelizes across chunks and is safe to use in `group_by`/`over` contexts.
- **Nulls**: if either coordinate is null, or no timezone matches, the result is null. Points in international waters resolve to `Etc/GMT±N` zones.

### `tz_name_from_h3(cell) -> pl.Expr`

Resolves each H3 cell to the IANA timezone containing the cell's **centre**.

- **`cell`** — a column of H3 indices, either as an integer (`UInt64`, `Int64`, …) or as the hex string form (`"871f1d489ffffff"`, upper or lower case). Any other dtype raises.
- **Returns** a `String` expression, elementwise like `tz_name`.
- **Nulls**: null input, an index that is not a valid H3 cell, and an unparseable hex string all yield null.
- **Coarse cells**: a low-resolution cell can span several timezones; you always get the one containing its centre. From resolution 7 (~1.2 km edges) downward this is effectively never ambiguous.

## Performance

- The timezone index (`tzf-rs` `DefaultFinder`) is built **once per process** and shared across all threads, so the one-time startup cost (~100 MB of polygon index) is amortized over the whole session.
- Lookups run in native Rust with zero Python overhead per row. Tens of millions of rows resolve in seconds on a laptop.
- `tz_name_from_h3` memoizes on the H3 index in a **thread-local cache** and builds its output by gathering from a dictionary of zone names rather than writing one string per row. Both matter: on 3M rows over 50k distinct cells at resolution 9, throughput goes from ~5M rows/s (a plain per-row lookup) to ~160M rows/s. The cache needs no locking and survives across chunks and across calls on the same worker thread.
- Throughput therefore tracks the number of **distinct cells**, not the row count, and it matters whether the cache is warm. *Cold* is a one-shot query, which pays to resolve every distinct cell; *warm* is what repeated calls, many row groups, or a grouped context see. Measured on 3M rows, one fresh process per row:

  | distinct cells | cold | warm |
  |---:|---:|---:|
  | 20,000 | 34 | 169 |
  | 120,000 | 25 | 145 |
  | 400,000 | 14 | 122 |
  | 700,000 | 7 | 16 |
  | 1,500,000 | 4 | 6 |

  The cache holds 2^19 (~524k) cells per worker thread, which covers a metro area at resolution 9 comfortably. Beyond that the surplus cells fall back to a full polygon lookup and throughput converges on the uncached cost — decoding the H3 index makes that floor about 25% below `tz_name` on raw coordinates, so if your cells are near-unique you may as well look up coordinates directly.

Reproduce with `uv run python scripts/bench.py`.

## Accuracy

`tzf-rs` uses simplified timezone polygons for speed. Points very close to timezone borders (roughly within ~100 m) may be assigned to the neighboring zone. This is the right trade-off for bulk analytical workloads; if you need survey-grade precision at borders, use a full-polygon lookup instead.

Timezone data comes from [timezone-boundary-builder](https://github.com/evansiroky/timezone-boundary-builder) via the [tzf](https://github.com/ringsaturn/tzf) project.

## Development

Requires [Rust](https://rustup.rs), Python ≥ 3.10, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/drumtorben/polars-tzf
cd polars-tzf
uv sync                          # create venv and install dev dependencies
uv run maturin develop --release # build the Rust extension into the venv
```

Then test your changes:

```bash
uv run python -c "
import polars as pl, polars_tzf as tzf
print(pl.select(tz=tzf.tz_name(pl.lit(52.52), pl.lit(13.405))))
"
```

Run the test suite:

```bash
uv run pytest
```

> **Note**: always build with `--release` when measuring performance — debug builds are dramatically slower.

## Credits

- [tzf-rs](https://github.com/ringsaturn/tzf-rs) by [@ringsaturn](https://github.com/ringsaturn) does the actual geometry work.
- [pyo3-polars](https://github.com/pola-rs/pyo3-polars) provides the Polars plugin machinery.

## License

MIT — see [LICENSE](LICENSE).
