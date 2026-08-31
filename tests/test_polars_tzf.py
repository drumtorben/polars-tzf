"""Tests for polars-tzf.

Expected H3 values are derived with Uber's reference `h3` library rather than
hardcoded, so these tests also cross-check our `h3o`-based implementation
against the canonical one.
"""

from __future__ import annotations

import h3
import polars as pl
import pytest

import polars_tzf as tzf

# (name, lat, lon, expected IANA zone)
CITIES = [
    ("Berlin", 52.5200, 13.4050, "Europe/Berlin"),
    ("New York City", 40.7128, -74.0060, "America/New_York"),
    ("Tokyo", 35.6762, 139.6503, "Asia/Tokyo"),
    ("Sao Paulo", -23.5505, -46.6333, "America/Sao_Paulo"),
]

RES = 9


def cell_hex(lat: float, lon: float, res: int = RES) -> str:
    return h3.latlng_to_cell(lat, lon, res)


def cell_int(lat: float, lon: float, res: int = RES) -> int:
    return h3.str_to_int(cell_hex(lat, lon, res))


# --- baseline: existing lat/lng lookup --------------------------------------


def test_tz_name_resolves_known_cities():
    df = pl.DataFrame(
        {
            "lat": [lat for _, lat, _, _ in CITIES],
            "lon": [lon for _, _, lon, _ in CITIES],
        }
    )
    got = df.select(tz=tzf.tz_name("lat", "lon"))["tz"].to_list()
    assert got == [tz for _, _, _, tz in CITIES]


# --- tz_name_from_h3: happy paths -------------------------------------------


def test_h3_uint64_resolves_known_cities():
    df = pl.DataFrame(
        {"cell": [cell_int(lat, lon) for _, lat, lon, _ in CITIES]},
        schema={"cell": pl.UInt64},
    )
    got = df.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list()
    assert got == [tz for _, _, _, tz in CITIES]


def test_h3_hex_string_resolves_known_cities():
    df = pl.DataFrame({"cell": [cell_hex(lat, lon) for _, lat, lon, _ in CITIES]})
    got = df.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list()
    assert got == [tz for _, _, _, tz in CITIES]


def test_h3_hex_string_and_uint64_agree():
    df = pl.DataFrame(
        {
            "as_hex": [cell_hex(lat, lon) for _, lat, lon, _ in CITIES],
            "as_int": [cell_int(lat, lon) for _, lat, lon, _ in CITIES],
        },
        schema={"as_hex": pl.String, "as_int": pl.UInt64},
    )
    got = df.select(
        from_hex=tzf.tz_name_from_h3("as_hex"),
        from_int=tzf.tz_name_from_h3("as_int"),
    )
    assert got["from_hex"].to_list() == got["from_int"].to_list()


def test_h3_hex_string_is_case_insensitive():
    df = pl.DataFrame(
        {
            "lower": [cell_hex(lat, lon).lower() for _, lat, lon, _ in CITIES],
            "upper": [cell_hex(lat, lon).upper() for _, lat, lon, _ in CITIES],
        }
    )
    got = df.select(
        lo=tzf.tz_name_from_h3("lower"),
        up=tzf.tz_name_from_h3("upper"),
    )
    assert got["lo"].to_list() == got["up"].to_list()


def test_h3_accepts_int64_dtype():
    """polars-h3 and Parquet round-trips often hand back Int64, not UInt64."""
    cells = [cell_int(lat, lon) for _, lat, lon, _ in CITIES]
    df = pl.DataFrame({"cell": cells}, schema={"cell": pl.UInt64}).with_columns(
        pl.col("cell").cast(pl.Int64)
    )
    got = df.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list()
    assert got == [tz for _, _, _, tz in CITIES]


def test_h3_matches_latlng_lookup_at_cell_centre():
    """The H3 result must equal a direct lookup of the cell's centre point."""
    cells = [cell_hex(lat, lon) for _, lat, lon, _ in CITIES]
    centres = [h3.cell_to_latlng(c) for c in cells]
    df = pl.DataFrame(
        {
            "cell": cells,
            "lat": [c[0] for c in centres],
            "lon": [c[1] for c in centres],
        }
    )
    got = df.select(
        from_h3=tzf.tz_name_from_h3("cell"),
        from_coords=tzf.tz_name("lat", "lon"),
    )
    assert got["from_h3"].to_list() == got["from_coords"].to_list()


def test_h3_resolves_across_many_resolutions():
    for res in (5, 7, 9, 12, 15):
        df = pl.DataFrame({"cell": [cell_hex(52.5200, 13.4050, res)]})
        got = df.select(tz=tzf.tz_name_from_h3("cell"))["tz"].item()
        assert got == "Europe/Berlin", f"resolution {res} gave {got!r}"


# --- tz_name_from_h3: nulls and invalid input -------------------------------


def test_h3_null_input_yields_null():
    df = pl.DataFrame({"cell": [None, None]}, schema={"cell": pl.UInt64})
    assert df.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list() == [None, None]


def test_h3_null_string_yields_null():
    df = pl.DataFrame({"cell": [None]}, schema={"cell": pl.String})
    assert df.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list() == [None]


def test_h3_invalid_index_yields_null():
    df = pl.DataFrame(
        {"cell": [0, 1, 0xFFFFFFFFFFFFFFFF]}, schema={"cell": pl.UInt64}
    )
    assert df.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list() == [None] * 3


def test_h3_unparseable_string_yields_null():
    df = pl.DataFrame({"cell": ["", "not-hex", "zzzz"]})
    assert df.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list() == [None] * 3


def test_h3_valid_and_invalid_mixed_keeps_row_alignment():
    berlin = cell_hex(52.5200, 13.4050)
    tokyo = cell_hex(35.6762, 139.6503)
    df = pl.DataFrame({"cell": [berlin, "not-hex", tokyo, None]})
    got = df.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list()
    assert got == ["Europe/Berlin", None, "Asia/Tokyo", None]


def test_h3_rejects_unsupported_dtype():
    df = pl.DataFrame({"cell": [1.5, 2.5]})
    with pytest.raises(pl.exceptions.PolarsError, match="(?i)h3"):
        df.select(tz=tzf.tz_name_from_h3("cell"))


# --- cache behaviour ---------------------------------------------------------


def test_h3_repeated_cells_resolve_consistently():
    """Exercises the dedup cache: a repeated cell must not change its answer."""
    berlin = cell_int(52.5200, 13.4050)
    tokyo = cell_int(35.6762, 139.6503)
    cells = [berlin, tokyo] * 500
    df = pl.DataFrame({"cell": cells}, schema={"cell": pl.UInt64})
    got = df.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list()
    assert got == ["Europe/Berlin", "Asia/Tokyo"] * 500


def test_h3_cache_is_consistent_across_chunks():
    """A multi-chunk Series must give the same answers as a single-chunk one."""
    cells = [cell_int(lat, lon) for _, lat, lon, _ in CITIES]
    schema = {"cell": pl.UInt64}
    single = pl.DataFrame({"cell": cells * 3}, schema=schema)
    chunked = pl.concat(
        [pl.DataFrame({"cell": cells}, schema=schema) for _ in range(3)],
        rechunk=False,
    )
    assert chunked.n_chunks() > 1
    expected = single.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list()
    assert chunked.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list() == expected


def test_h3_works_in_lazy_and_grouped_contexts():
    cells = [cell_int(lat, lon) for _, lat, lon, _ in CITIES]
    df = pl.LazyFrame(
        {"cell": cells * 4, "grp": [0, 0, 1, 1] * 4},
        schema={"cell": pl.UInt64, "grp": pl.Int64},
    )
    got = (
        df.with_columns(tz=tzf.tz_name_from_h3("cell"))
        .group_by("grp")
        .agg(pl.col("tz").unique().sort())
        .sort("grp")
        .collect()
    )
    assert got["tz"].to_list() == [
        ["America/New_York", "Europe/Berlin"],
        ["America/Sao_Paulo", "Asia/Tokyo"],
    ]


# --- many distinct zones in one call ----------------------------------------

# Deliberately spread over many zones: the plugin builds its output from an
# internal dictionary of zone names, so a call touching dozens of zones
# exercises far more of that path than the four-city fixture above.
WORLD = [
    (51.5074, -0.1278, "Europe/London"),
    (48.8566, 2.3522, "Europe/Paris"),
    (40.4168, -3.7038, "Europe/Madrid"),
    (41.9028, 12.4964, "Europe/Rome"),
    (55.7558, 37.6173, "Europe/Moscow"),
    (41.0082, 28.9784, "Europe/Istanbul"),
    (64.1466, -21.9426, "Atlantic/Reykjavik"),
    (30.0444, 31.2357, "Africa/Cairo"),
    (6.5244, 3.3792, "Africa/Lagos"),
    (-1.2921, 36.8219, "Africa/Nairobi"),
    (-26.2041, 28.0473, "Africa/Johannesburg"),
    (25.2048, 55.2708, "Asia/Dubai"),
    (24.8607, 67.0011, "Asia/Karachi"),
    (19.0760, 72.8777, "Asia/Kolkata"),
    (23.8103, 90.4125, "Asia/Dhaka"),
    (13.7563, 100.5018, "Asia/Bangkok"),
    (1.3521, 103.8198, "Asia/Singapore"),
    (-6.2088, 106.8456, "Asia/Jakarta"),
    (22.3193, 114.1694, "Asia/Hong_Kong"),
    (31.2304, 121.4737, "Asia/Shanghai"),
    (37.5665, 126.9780, "Asia/Seoul"),
    (35.6762, 139.6503, "Asia/Tokyo"),
    (-33.8688, 151.2093, "Australia/Sydney"),
    (-37.8136, 144.9631, "Australia/Melbourne"),
    (-36.8485, 174.7633, "Pacific/Auckland"),
    (21.3099, -157.8581, "Pacific/Honolulu"),
    (61.2181, -149.9003, "America/Anchorage"),
    (34.0522, -118.2437, "America/Los_Angeles"),
    (39.7392, -104.9903, "America/Denver"),
    (41.8781, -87.6298, "America/Chicago"),
    (19.4326, -99.1332, "America/Mexico_City"),
    (4.7110, -74.0721, "America/Bogota"),
    (-12.0464, -77.0428, "America/Lima"),
    (-33.4489, -70.6693, "America/Santiago"),
    (-34.6037, -58.3816, "America/Argentina/Buenos_Aires"),
]


def test_h3_resolves_many_distinct_zones_in_one_call():
    df = pl.DataFrame({"cell": [cell_hex(lat, lon) for lat, lon, _ in WORLD]})
    got = df.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list()
    assert got == [tz for _, _, tz in WORLD]


def test_h3_many_zones_survive_repetition_and_nulls():
    """Same zones, but interleaved with nulls and repeats to catch misalignment."""
    cells, expected = [], []
    for lat, lon, tz in WORLD:
        cells += [cell_hex(lat, lon), None, cell_hex(lat, lon)]
        expected += [tz, None, tz]
    df = pl.DataFrame({"cell": cells})
    assert df.select(tz=tzf.tz_name_from_h3("cell"))["tz"].to_list() == expected


def test_h3_output_dtype_is_string():
    df = pl.DataFrame({"cell": [cell_hex(52.52, 13.405)]})
    assert df.select(tz=tzf.tz_name_from_h3("cell")).schema["tz"] == pl.String


# --- cache eviction ----------------------------------------------------------


def test_h3_stays_correct_when_the_cache_evicts():
    """More distinct cells than the cache holds must not return stale answers.

    The memo is a fixed-size direct-mapped cache, so cells that collide evict
    one another. If the key comparison guarding a slot were wrong, a colliding
    cell would silently inherit its neighbour's timezone -- and since the wrong
    zone is still a plausible-looking string, only a comparison against the
    coordinate lookup catches it.
    """
    # A disk of cells around the Alps: ~44k cells spanning several zones.
    centre = h3.latlng_to_cell(46.8, 9.5, 7)
    cells = h3.grid_disk(centre, 120)
    assert len(cells) > 40_000, len(cells)

    centres = [h3.cell_to_latlng(c) for c in cells]
    df = pl.DataFrame(
        {
            "cell": cells,
            "lat": [c[0] for c in centres],
            "lon": [c[1] for c in centres],
        }
    )
    got = df.select(
        from_h3=tzf.tz_name_from_h3("cell"),
        from_coords=tzf.tz_name("lat", "lon"),
    )
    assert got["from_h3"].to_list() == got["from_coords"].to_list()
    # Guard against the comparison passing because everything came back null.
    assert got["from_h3"].n_unique() > 3
