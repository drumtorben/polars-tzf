# polars-tzf

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)

Fast timezone lookup for [Polars](https://pola.rs): resolve IANA timezone names from latitude/longitude coordinates, powered by [tzf-rs](https://github.com/ringsaturn/tzf-rs).

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

## Performance

- The timezone index (`tzf-rs` `DefaultFinder`) is built **once per process** and shared across all threads, so the one-time startup cost (~100 MB of polygon index) is amortized over the whole session.
- Lookups run in native Rust with zero Python overhead per row. Tens of millions of rows resolve in seconds on a laptop.

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

> **Note**: always build with `--release` when measuring performance — debug builds are dramatically slower.

## Credits

- [tzf-rs](https://github.com/ringsaturn/tzf-rs) by [@ringsaturn](https://github.com/ringsaturn) does the actual geometry work.
- [pyo3-polars](https://github.com/pola-rs/pyo3-polars) provides the Polars plugin machinery.

## License

MIT — see [LICENSE](LICENSE).
