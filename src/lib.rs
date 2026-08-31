use h3o::{CellIndex, LatLng};
use polars::chunked_array::cast::CastOptions;
use polars::prelude::*;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{pymodule, Bound, PyResult};
use pyo3_polars::derive::polars_expr;
use pyo3_polars::PolarsAllocator;
use rustc_hash::FxHashMap;
use std::cell::RefCell;
use std::sync::LazyLock;
use tzf_rs::DefaultFinder;

// The finder is expensive to build (y_stripes index ~120 MB) and Send+Sync,
// so build it exactly once per process and share it across all threads.
static FINDER: LazyLock<DefaultFinder> = LazyLock::new(DefaultFinder::new);

/// Borrow the finder as `&'static`, which makes the names it returns `&'static
/// str` too (`get_tz_name(&'a self, ..) -> &'a str`). That lets the cache below
/// store zone names as bare slices instead of owned strings.
fn finder() -> &'static DefaultFinder {
    &FINDER
}

/// Resolve one point. `None` means "no zone matched" and becomes a null.
///
/// NOTE: tzf-rs expects (lng, lat) — not (lat, lng)!
fn lookup(lng: f64, lat: f64) -> Option<&'static str> {
    match finder().get_tz_name(lng, lat) {
        "" => None,
        tz => Some(tz),
    }
}

/// Upper bound on memoised cells per worker thread. At 2^19 entries the map
/// costs roughly 8 MB, next to the ~100 MB polygon index already resident.
const CACHE_CAPACITY: usize = 1 << 19;

/// Reserved dictionary slot standing for "no zone", so that every row has a
/// valid gather index and the index array never needs a validity mask.
const NO_ZONE: IdxSize = 0;

/// Per-thread memo of H3 index -> timezone, holding *positions* into a small
/// dictionary of zone names rather than the names themselves.
///
/// Storing positions is what makes the output cheap to build. There are only
/// ~450 distinct zones, so instead of appending one name per row we hand Polars
/// the dictionary plus an index per row and let it gather: gathering copies
/// 16-byte string views and shares the underlying bytes, whereas appending
/// copies every name in full, once per row. Once the cache is warm that is the
/// difference between most of the runtime and almost none of it.
///
/// On overflow the map stops accepting new cells but keeps what it has. That
/// asymmetry is deliberate and was measured. Clearing it wholesale instead
/// falls off a cliff the moment the working set exceeds the cap — every clear
/// discards everything, so a column with more distinct cells than the cap gets
/// almost no hits while still paying to insert: at 2M rows that was 150M rows/s
/// just under the cap against 6.6M just over it, slower than not caching at
/// all. A fixed-size direct-mapped table avoids the cliff too, but it is the
/// same size whether it holds twenty thousand cells or five hundred thousand,
/// and that cost 3x on the common small-working-set case. Letting the map size
/// itself and simply capping its growth keeps the fast case fast, and degrades
/// to "one failed probe per row" rather than to something worse than no cache.
///
/// The cache is thread-local: no locking, and it survives across chunks and
/// across plugin invocations on the same worker thread.
#[derive(Default)]
struct TzCache {
    cells: FxHashMap<u64, IdxSize>,
    /// Zone names by position; slot `NO_ZONE` is a placeholder, never emitted.
    names: Vec<&'static str>,
    positions: FxHashMap<&'static str, IdxSize>,
}

impl TzCache {
    fn new() -> Self {
        Self {
            names: vec![""],
            ..Default::default()
        }
    }

    /// Dictionary position of the zone containing this cell's centre, or
    /// `NO_ZONE` when the index is invalid or no zone matches. Invalid indices
    /// are memoised too — rejecting them is not free.
    ///
    /// NOTE: do not "improve" this by keying on a coarser parent cell to raise
    /// the hit rate. A parent cell can straddle a timezone border, so its centre
    /// says nothing about the children near that border — it would silently
    /// return the wrong zone.
    fn position_of(&mut self, index: u64) -> IdxSize {
        if let Some(&hit) = self.cells.get(&index) {
            return hit;
        }

        let resolved = CellIndex::try_from(index).ok().and_then(|cell| {
            let centre = LatLng::from(cell);
            lookup(centre.lng(), centre.lat())
        });

        let position = match resolved {
            None => NO_ZONE,
            Some(name) => match self.positions.get(name) {
                Some(&known) => known,
                None => {
                    self.names.push(name);
                    let position = (self.names.len() - 1) as IdxSize;
                    self.positions.insert(name, position);
                    position
                }
            },
        };

        if self.cells.len() < CACHE_CAPACITY {
            self.cells.insert(index, position);
        }
        position
    }

    /// The dictionary as a column, with the reserved slot as null.
    fn dictionary(&self) -> StringChunked {
        self.names
            .iter()
            .enumerate()
            .map(|(position, name)| (position != NO_ZONE as usize).then_some(*name))
            .collect()
    }
}

thread_local! {
    static TZ_CACHE: RefCell<TzCache> = RefCell::new(TzCache::new());
}

/// Map cells to dictionary positions, and hand back the dictionary to gather
/// from. Both come from the same borrow so they cannot drift apart.
fn resolve(cells: impl Iterator<Item = Option<u64>>) -> (Vec<IdxSize>, StringChunked) {
    TZ_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let positions = cells
            .map(|cell| cell.map_or(NO_ZONE, |index| cache.position_of(index)))
            .collect();
        (positions, cache.dictionary())
    })
}

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
            (Some(lat), Some(lng)) => lookup(lng, lat),
            _ => None,
        })
        .collect();

    Ok(out.into_series())
}

/// Resolve the timezone at the *centre* of each H3 cell.
///
/// Accepts either the 64-bit index or its hex string form. A cell coarse enough
/// to span several zones still yields exactly one — the one containing its
/// centre; from resolution 7 (~1.2 km edges) down that is effectively never
/// ambiguous.
#[polars_expr(output_type=String)]
fn tz_name_from_h3(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];

    let (positions, dictionary) = match s.dtype() {
        // `from_str_radix` accepts either case; anything unparseable, and any
        // parsed value that is not a valid index, falls through to null.
        DataType::String => {
            let cells = s.str()?;
            resolve(
                cells
                    .iter()
                    .map(|cell| cell.and_then(|hex| u64::from_str_radix(hex, 16).ok())),
            )
        }

        dt if dt.is_integer() => {
            // Non-strict: a negative Int64 cannot be an H3 index (bit 63 is
            // always 0), so let it become null rather than an error.
            let cells = s.cast_with_options(&DataType::UInt64, CastOptions::NonStrict)?;
            resolve(cells.u64()?.iter())
        }

        dt => polars_bail!(
            InvalidOperation:
            "tz_name_from_h3 expects an H3 index as an integer or a hex string, got: {}",
            dt
        ),
    };

    let positions = IdxCa::from_vec(PlSmallStr::EMPTY, positions);
    Ok(dictionary.take(&positions)?.into_series())
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();
