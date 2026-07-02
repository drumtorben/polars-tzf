use polars::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{pymodule, Bound, PyResult};
use pyo3_polars::derive::polars_expr;
use pyo3_polars::PolarsAllocator;
use std::sync::LazyLock;
use tzf_rs::DefaultFinder;

// The finder is expensive to build (y_stripes index ~120 MB) and Send+Sync,
// so build it exactly once per process and share it across all threads.
static FINDER: LazyLock<DefaultFinder> = LazyLock::new(DefaultFinder::new);

#[polars_expr(output_type=String)]
fn tz_name(inputs: &[Series]) -> PolarsResult<Series> {
    let lat = inputs[0].f64()?;
    let lng = inputs[1].f64()?;
    polars_ensure!(
        lat.len() == lng.len(),
        ShapeMismatch: "lat and lng must have the same length"
    );

    let out: StringChunked = lat
        .iter()
        .zip(lng.iter())
        .map(|(lat, lng)| match (lat, lng) {
            // NOTE: tzf-rs expects (lng, lat) — not (lat, lng)!
            (Some(lat), Some(lng)) => match FINDER.get_tz_name(lng, lat) {
                "" => None,           // no match -> null
                tz => Some(tz),
            },
            _ => None,
        })
        .collect();

    Ok(out.into_series())
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
