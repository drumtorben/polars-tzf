from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExpr

LIB = Path(__file__).parent


def tz_name(lat: IntoExpr, lng: IntoExpr) -> pl.Expr:
    """Resolve IANA timezone names from (lat, lng). Null when there is no match."""
    return register_plugin_function(
        plugin_path=LIB,
        function_name="tz_name",
        args=[lat, lng],
        is_elementwise=True,
    )
