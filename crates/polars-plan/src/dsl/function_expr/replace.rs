use polars_core::prelude::{DataFrame, Series, *};
use polars_lazy::prelude::*;

// use polars_error::{polars_bail, polars_ensure, PolarsResult};
use super::*;
use crate::dsl::{col, lit, when};

pub(super) fn replace(s: &[Series]) -> PolarsResult<Series> {
    replace_impl(&s[0], &mut s[1], &mut s[2])
}
pub(super) fn replace_with_default(s: &[Series]) -> PolarsResult<Series> {
    replace_with_default_impl(&s[0], &mut s[1], &mut s[2], &mut s[3])
}

fn replace_impl(s: &Series, old: &mut Series, new: &mut Series) -> PolarsResult<Series> {
    if old.len() == 0 {
        return Ok(s.clone());
    }

    polars_ensure!(
        new.len() == 1 || (new.len() == old.len()),
        ComputeError: "`new` input for `replace` must have length 1 or be the same length as `old`"
    );

    let df = DataFrame::new_no_checks(vec![s.clone()]);

    // Special case: if a single value is replaced, use `when/then/otherwise`
    if old.len() == 1 {
        let old_value = unsafe { old.get_unchecked(0) };
        let new_value = unsafe { new.get_unchecked(0) };
        let x = df.
        let out = df
            .lazy()
            .select([when(col(s.name()).eq(lit(old_value.into())))
                .then(lit(new_value.into()))
                .otherwise(col(s.name()))])?;
        return out.column(s.name());
    }

    // TODO: Allow 'broadcasting' `new` here for many-to-one replace?
    // polars_ensure!(
    //     old.len() == new.len(),
    //     ComputeError: "`old` and `new` inputs for `replace` must have the same length"
    // );

    old.rename("__POLARS_REPLACE_OLD");
    new.rename("__POLARS_REPLACE_NEW");
    let df = DataFrame::new_no_checks(vec![old, new]);

    Ok(s.clone())
}

fn replace_with_default_impl(
    s: &Series,
    old: &mut Series,
    new: &mut Series,
    default: &mut Series,
) -> PolarsResult<Series> {
    if old.len() == 0 {
        return Ok(default.clone());
    }

    let replaced = replace_impl(s, old, new);

    replaced
}
