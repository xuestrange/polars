use std::iter::{zip, Zip};

use polars_core::prelude::*;
use polars_core::series::Series;
use polars_time::{time_range_impl, ClosedWindow, Duration};

use super::utils::{ensure_range_bounds_contain_exactly_one_value, temporal_series_to_i64_scalar};

const CAPACITY_FACTOR: usize = 5;

pub(super) fn time_range(
    s: &[Series],
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];

    ensure_range_bounds_contain_exactly_one_value(start, end)?;

    let dtype = DataType::Time;
    let start = temporal_series_to_i64_scalar(&start.cast(&dtype)?)
        .ok_or_else(|| polars_err!(ComputeError: "start is an out-of-range time."))?;
    let end = temporal_series_to_i64_scalar(&end.cast(&dtype)?)
        .ok_or_else(|| polars_err!(ComputeError: "end is an out-of-range time."))?;

    let out = time_range_impl("time", start, end, interval, closed)?;
    Ok(out.cast(&dtype).unwrap().into_series())
}

pub(super) fn time_ranges(
    s: &[Series],
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Series> {
    let start = &s[0];
    let end = &s[1];

    let start = start.cast(&DataType::Time)?;
    let end = end.cast(&DataType::Time)?;

    let start_phys = start.to_physical_repr();
    let end_phys = end.to_physical_repr();
    let start = start_phys.i64().unwrap();
    let end = end_phys.i64().unwrap();

    // let (start_iter, end_iter) = broadcast_scalar_inputs_iter(start_ca, end_ca)?;
    // let start_end_iter = std::iter::zip(start_iter, end_iter);

    match (start.len(), end.len()) {
        (len_start, len_end) if len_start == len_end => {
            let start_end_iter = zip(start, end);
            time_ranges_impl(start_end_iter, len_start, interval, closed)
        },
        (1, len_end) => {
            let start_scalar = unsafe { start.get_unchecked(0) };
            let start_iter = std::iter::repeat(start_scalar).take(len_end);
            let start_end_iter = zip(start_iter, end);
            time_ranges_impl(start_end_iter, len_end, interval, closed)
        },
        (len_start, 1) => {
            let end_scalar = unsafe { end.get_unchecked(0) };
            let end_iter = std::iter::repeat(end_scalar).take(len_start);
            let start_end_iter = zip(start, end_iter);
            time_ranges_impl(start_end_iter, len_start, interval, closed)
        },
        (len_start, len_end) => {
            polars_bail!(
                ComputeError:
                "lengths of `start` ({}) and `end` ({}) do not match",
                len_start, len_end
            )
        },
    }
}

fn time_ranges_impl(
    start_end_iter: Zip<impl Iterator<Item = Option<i64>>, impl Iterator<Item = Option<i64>>>,
    len: usize,
    interval: Duration,
    closed: ClosedWindow,
) -> PolarsResult<Series> {
    let mut builder = ListPrimitiveChunkedBuilder::<Int64Type>::new(
        "time_range",
        len,
        len * CAPACITY_FACTOR,
        DataType::Int64,
    );
    for (start, end) in start_end_iter {
        match (start, end) {
            (Some(start), Some(end)) => {
                let rng = time_range_impl("", start, end, interval, closed)?;
                builder.append_slice(rng.cont_slice().unwrap())
            },
            _ => builder.append_null(),
        }
    }
    let list = builder.finish().into_series();

    let to_type = DataType::List(Box::new(DataType::Time));
    list.cast(&to_type)
}
