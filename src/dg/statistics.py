"""Statistical contracts and mouse-level inference for Dynamic Gating.

This module deliberately operates on already-derived scalar estimates.  It
does not read neural data and it never treats sessions, units, or trials as
independent biological replicates when a mouse-level estimate is available.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import math
import numbers
from typing import Any

import numpy as np
import polars as pl

STATISTICS_SCHEMA = pl.Schema(
    {
        "analysis_id": pl.String,
        "result_id": pl.String,
        "contrast_id": pl.String,
        "hypothesis": pl.String,
        "dandiset_version": pl.String,
        "code_version": pl.String,
        "seed": pl.Int64,
        "analysis_tier": pl.String,
        "inclusion_definition": pl.String,
        "missingness_stratum": pl.String,
        "estimate": pl.Float64,
        "scale": pl.String,
        "ci_low": pl.Float64,
        "ci_high": pl.Float64,
        "confidence_level": pl.Float64,
        "ci_method": pl.String,
        "test_statistic": pl.Float64,
        "test_method": pl.String,
        "p_value": pl.Float64,
        "adjusted_p_value": pl.Float64,
        "adjustment_method": pl.String,
        "multiplicity_family": pl.String,
        "sidedness": pl.String,
        "n_mice": pl.Int64,
        "n_sessions": pl.Int64,
        "n_probes": pl.Int64,
        "n_units": pl.Int64,
        "n_trials": pl.Int64,
        "aggregation": pl.String,
        "bootstrap_id": pl.String,
        "model_formula": pl.String,
        "cv_grouping": pl.String,
        "status": pl.String,
        "reason": pl.String,
    }
)

_REQUIRED_TEXT_COLUMNS = (
    "analysis_id",
    "result_id",
    "contrast_id",
    "hypothesis",
    "dandiset_version",
    "code_version",
    "analysis_tier",
    "inclusion_definition",
    "scale",
    "aggregation",
    "status",
)
_TEXT_COLUMNS = tuple(name for name, dtype in STATISTICS_SCHEMA.items() if dtype == pl.String)
_FLOAT_COLUMNS = tuple(name for name, dtype in STATISTICS_SCHEMA.items() if dtype == pl.Float64)
_COUNT_COLUMNS = ("n_mice", "n_sessions", "n_probes", "n_units", "n_trials")
_ANALYSIS_TIERS = frozenset({"discovery", "confirmation", "exploratory"})
_SIDEDNESS_VALUES = frozenset({"two-sided", "greater", "less", "not_applicable"})
_STATUS_VALUES = frozenset({"pass", "null", "fragile", "not_estimable", "failed"})
_REASON_REQUIRED_STATUSES = frozenset({"fragile", "not_estimable", "failed"})
_MAX_EXACT_SIGN_FLIP_MICE = 24


@dataclasses.dataclass(frozen=True, slots=True)
class BootstrapMeanResult:
    """Percentile bootstrap interval for an equally weighted mouse mean."""

    estimate: float
    ci_low: float
    ci_high: float
    confidence_level: float
    n_resamples: int
    seed: int
    n_mice: int


@dataclasses.dataclass(frozen=True, slots=True)
class SignFlipResult:
    """Result of a two-sided randomization test over mouse-level signs."""

    statistic: float
    p_value: float
    method: str
    n_permutations: int
    seed: int | None
    n_mice: int


def empty_statistics_table() -> pl.DataFrame:
    """Return an empty table with the canonical publication-statistics schema."""

    return pl.DataFrame(schema=STATISTICS_SCHEMA)


def normalize_statistics_table(table: pl.DataFrame) -> pl.DataFrame:
    """Normalize and validate a table against the canonical statistics contract.

    Identity and provenance columns are required. Missing columns that are
    analysis-dependent are added as typed nulls. Unknown columns are rejected
    so a misspelling or schema change cannot silently enter a manuscript table.
    Text is stripped, numeric columns are cast strictly, and semantic checks are
    delegated to :func:`validate_statistics_table`.
    """

    if not isinstance(table, pl.DataFrame):
        raise TypeError("table must be a polars DataFrame")

    schema_names = set(STATISTICS_SCHEMA.names())
    table_names = set(table.columns)
    unknown = table_names.difference(schema_names)
    if unknown:
        raise ValueError(f"statistics table has unknown columns: {sorted(unknown)}")

    missing_required = set(_REQUIRED_TEXT_COLUMNS).difference(table_names)
    if missing_required:
        raise ValueError(
            f"statistics table is missing required columns: {sorted(missing_required)}"
        )

    numeric_columns = ("seed", *_FLOAT_COLUMNS, *_COUNT_COLUMNS)
    for column in numeric_columns:
        if column not in table.columns:
            continue
        for row_index, value in enumerate(table.get_column(column)):
            if value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
                raise ValueError(f"row {row_index}: {column} must be numeric or null")
            if column == "seed" or column in _COUNT_COLUMNS:
                numeric = float(value)
                if not math.isfinite(numeric) or not numeric.is_integer():
                    raise ValueError(f"row {row_index}: {column} must be an integer or null")

    normalized = table
    for column, dtype in STATISTICS_SCHEMA.items():
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None, dtype=dtype).alias(column))

    try:
        normalized = normalized.select(
            *(
                pl.col(column).cast(dtype, strict=True).alias(column)
                for column, dtype in STATISTICS_SCHEMA.items()
            )
        )
    except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError) as error:
        message = f"statistics table contains a value with an invalid type: {error}"
        raise ValueError(message) from error

    normalized = normalized.with_columns(
        *(
            pl.when(pl.col(column).is_null())
            .then(None)
            .otherwise(pl.col(column).str.strip_chars())
            .alias(column)
            for column in _TEXT_COLUMNS
        )
    ).with_columns(
        *(
            pl.when(pl.col(column) == "").then(None).otherwise(pl.col(column)).alias(column)
            for column in _TEXT_COLUMNS
        )
    )
    validate_statistics_table(normalized)
    return normalized


def validate_statistics_table(table: pl.DataFrame) -> None:
    """Raise ``ValueError`` when a canonical statistics table is inconsistent."""

    if not isinstance(table, pl.DataFrame):
        raise TypeError("table must be a polars DataFrame")
    if table.columns != STATISTICS_SCHEMA.names():
        raise ValueError("statistics table columns are not in canonical order")
    if table.schema != STATISTICS_SCHEMA:
        raise ValueError("statistics table dtypes do not match the canonical schema")

    for row_index, row in enumerate(table.iter_rows(named=True)):
        for column in _REQUIRED_TEXT_COLUMNS:
            if row[column] is None or not row[column].strip():
                raise ValueError(f"row {row_index}: {column} must be non-empty")

        if row["analysis_tier"] not in _ANALYSIS_TIERS:
            raise ValueError(
                f"row {row_index}: analysis_tier must be one of {sorted(_ANALYSIS_TIERS)}"
            )
        if row["status"] not in _STATUS_VALUES:
            raise ValueError(f"row {row_index}: status must be one of {sorted(_STATUS_VALUES)}")
        if row["sidedness"] is not None and row["sidedness"] not in _SIDEDNESS_VALUES:
            raise ValueError(
                f"row {row_index}: sidedness must be one of {sorted(_SIDEDNESS_VALUES)}"
            )
        if row["status"] in _REASON_REQUIRED_STATUSES and row["reason"] is None:
            raise ValueError(f"row {row_index}: status {row['status']!r} requires a reason")
        if row["status"] in {"pass", "null", "fragile"} and row["estimate"] is None:
            raise ValueError(f"row {row_index}: status {row['status']!r} requires an estimate")

        for column in _FLOAT_COLUMNS:
            value = row[column]
            if value is not None and not math.isfinite(value):
                raise ValueError(f"row {row_index}: {column} must be finite or null")
        for column in _COUNT_COLUMNS:
            value = row[column]
            if value is not None and value < 0:
                raise ValueError(f"row {row_index}: {column} must be non-negative or null")
        if row["seed"] is not None and row["seed"] < 0:
            raise ValueError(f"row {row_index}: seed must be non-negative or null")

        for column in ("p_value", "adjusted_p_value"):
            value = row[column]
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"row {row_index}: {column} must be in [0, 1] or null")
        confidence_level = row["confidence_level"]
        if confidence_level is not None and not 0 < confidence_level < 1:
            raise ValueError(f"row {row_index}: confidence_level must be in (0, 1) or null")

        ci_low = row["ci_low"]
        ci_high = row["ci_high"]
        if (ci_low is None) != (ci_high is None):
            raise ValueError(f"row {row_index}: ci_low and ci_high must both be set or null")
        if ci_low is not None and ci_high is not None:
            if ci_low > ci_high:
                raise ValueError(f"row {row_index}: ci_low must not exceed ci_high")
            if confidence_level is None or row["ci_method"] is None:
                raise ValueError(
                    f"row {row_index}: confidence interval requires confidence_level and ci_method"
                )

        if row["p_value"] is not None:
            for column in ("test_method", "sidedness", "multiplicity_family"):
                if row[column] is None:
                    raise ValueError(f"row {row_index}: p_value requires {column}")
        if row["adjusted_p_value"] is not None:
            if row["p_value"] is None:
                raise ValueError(f"row {row_index}: adjusted_p_value requires p_value")
            if row["adjustment_method"] is None:
                raise ValueError(f"row {row_index}: adjusted_p_value requires adjustment_method")


def holm_adjust(
    p_values: collections.abc.Sequence[float | None],
) -> list[float | None]:
    """Return Holm family-wise-error adjusted p-values, preserving nulls."""

    values = list(p_values)
    indexed: list[tuple[int, float]] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
            raise ValueError(f"p_values[{index}] must be numeric or null")
        numeric = float(value)
        if not math.isfinite(numeric) or not 0 <= numeric <= 1:
            raise ValueError(f"p_values[{index}] must be finite and in [0, 1], or null")
        indexed.append((index, numeric))

    adjusted: list[float | None] = [None] * len(values)
    if not indexed:
        return adjusted

    ordered = sorted(indexed, key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    running_maximum = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * p_value)
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = running_maximum
    return adjusted


def add_holm_adjustment(
    table: pl.DataFrame,
    *,
    family_columns: collections.abc.Sequence[str] = ("multiplicity_family",),
) -> pl.DataFrame:
    """Add Holm-adjusted p-values independently within each declared family."""

    normalized = normalize_statistics_table(table)
    if isinstance(family_columns, str) or not family_columns:
        raise ValueError("family_columns must contain at least one column")
    missing = set(family_columns).difference(normalized.columns)
    if missing:
        raise ValueError(f"family columns are missing: {sorted(missing)}")
    if normalized.select(
        pl.col("adjusted_p_value").is_not_null().any()
        | pl.col("adjustment_method").is_not_null().any()
    ).item():
        raise ValueError("adjustment columns are already populated")

    records = normalized.to_dicts()
    rows_by_family: dict[tuple[Any, ...], list[int]] = {}
    for row_index, row in enumerate(records):
        if row["p_value"] is None:
            continue
        family = tuple(row[column] for column in family_columns)
        if any(value is None for value in family):
            raise ValueError(f"row {row_index}: Holm adjustment requires a complete family key")
        rows_by_family.setdefault(family, []).append(row_index)

    for row_indices in rows_by_family.values():
        adjusted = holm_adjust([records[index]["p_value"] for index in row_indices])
        for row_index, adjusted_p_value in zip(row_indices, adjusted, strict=True):
            records[row_index]["adjusted_p_value"] = adjusted_p_value
            records[row_index]["adjustment_method"] = "holm"

    if not records:
        return normalized
    return normalize_statistics_table(pl.DataFrame(records, schema=STATISTICS_SCHEMA))


def aggregate_reversible_contrast_by_mouse(
    session_estimates: pl.DataFrame,
    *,
    mouse_column: str = "mouse_id",
    session_column: str = "session_id",
    engaged_1_column: str = "engaged_1",
    no_reward_column: str = "no_reward",
    engaged_2_column: str = "engaged_2",
) -> pl.DataFrame:
    """Average session-level three-block contrasts within each mouse.

    Every input row must be one complete session. The reversible contrast is
    ``0.5 * engaged_1 - no_reward + 0.5 * engaged_2``. Sessions receive equal
    weight within mouse; downstream inference should then weight the returned
    mouse rows equally.
    """

    if not isinstance(session_estimates, pl.DataFrame):
        raise TypeError("session_estimates must be a polars DataFrame")
    required = {
        mouse_column,
        session_column,
        engaged_1_column,
        no_reward_column,
        engaged_2_column,
    }
    missing = required.difference(session_estimates.columns)
    if missing:
        raise ValueError(f"session_estimates is missing columns: {sorted(missing)}")
    if session_estimates.is_empty():
        raise ValueError("session_estimates must contain at least one session")

    keys = session_estimates.select(mouse_column, session_column)
    if keys.null_count().sum_horizontal().item() > 0:
        raise ValueError("mouse and session keys must not be null")
    if keys.is_duplicated().any():
        raise ValueError("session_estimates must have unique mouse/session keys")

    block_columns = (engaged_1_column, no_reward_column, engaged_2_column)
    try:
        sessions = session_estimates.select(
            mouse_column,
            session_column,
            *(pl.col(column).cast(pl.Float64, strict=True) for column in block_columns),
        )
    except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError) as error:
        raise ValueError(f"block estimates must be numeric: {error}") from error

    for column in block_columns:
        invalid = pl.col(column).is_null() | ~pl.col(column).is_finite()
        if sessions.select(invalid.any()).item():
            raise ValueError(f"{column} must contain only finite, non-null estimates")

    sessions = sessions.with_columns(
        (
            0.5 * pl.col(engaged_1_column)
            - pl.col(no_reward_column)
            + 0.5 * pl.col(engaged_2_column)
        ).alias("reversible_contrast")
    )
    return (
        sessions.group_by(mouse_column)
        .agg(
            pl.len().alias("n_sessions"),
            pl.col(engaged_1_column).mean().alias("engaged_1_mean"),
            pl.col(no_reward_column).mean().alias("no_reward_mean"),
            pl.col(engaged_2_column).mean().alias("engaged_2_mean"),
            pl.col("reversible_contrast").mean(),
        )
        .sort(mouse_column)
    )


def bootstrap_mouse_mean(
    mouse_estimates: pl.DataFrame,
    *,
    seed: int,
    mouse_column: str = "mouse_id",
    value_column: str = "reversible_contrast",
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    batch_size: int = 10_000,
) -> BootstrapMeanResult:
    """Estimate a seeded percentile interval by resampling whole mice."""

    _validate_seed(seed)
    _validate_positive_integer("n_resamples", n_resamples)
    _validate_positive_integer("batch_size", batch_size)
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")

    values = _extract_unique_mouse_values(
        mouse_estimates,
        mouse_column=mouse_column,
        value_column=value_column,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_resamples, dtype=float)
    n_mice = values.size
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(0, n_mice, size=(stop - start, n_mice))
        bootstrap_means[start:stop] = values[indices].mean(axis=1)

    tail_probability = (1 - confidence_level) / 2
    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [tail_probability, 1 - tail_probability],
        method="linear",
    )
    return BootstrapMeanResult(
        estimate=float(values.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        seed=seed,
        n_mice=n_mice,
    )


def two_sided_sign_flip_test(
    mouse_estimates: pl.DataFrame,
    *,
    seed: int,
    mouse_column: str = "mouse_id",
    value_column: str = "reversible_contrast",
    exact_max_mice: int = 20,
    n_resamples: int = 100_000,
    batch_size: int = 10_000,
) -> SignFlipResult:
    """Test a zero mouse-level mean with exact or Monte Carlo sign flips.

    All sign assignments are enumerated when ``n_mice <= exact_max_mice``.
    Larger samples use seeded Monte Carlo draws and the plus-one correction,
    preventing a reported p-value of zero.
    """

    _validate_seed(seed)
    if isinstance(exact_max_mice, bool) or not isinstance(exact_max_mice, int):
        raise ValueError("exact_max_mice must be a non-negative integer")
    if not 0 <= exact_max_mice <= _MAX_EXACT_SIGN_FLIP_MICE:
        raise ValueError(f"exact_max_mice must be between 0 and {_MAX_EXACT_SIGN_FLIP_MICE}")
    _validate_positive_integer("n_resamples", n_resamples)
    _validate_positive_integer("batch_size", batch_size)

    values = _extract_unique_mouse_values(
        mouse_estimates,
        mouse_column=mouse_column,
        value_column=value_column,
    )
    n_mice = values.size
    observed = abs(float(values.mean()))
    tolerance = np.finfo(float).eps * max(1.0, observed) * 8

    if n_mice <= exact_max_mice:
        n_permutations = 1 << n_mice
        extreme = 0
        bit_positions = np.arange(n_mice, dtype=np.uint64)
        for start in range(0, n_permutations, batch_size):
            stop = min(start + batch_size, n_permutations)
            assignments = np.arange(start, stop, dtype=np.uint64)[:, np.newaxis]
            positive = ((assignments >> bit_positions) & 1).astype(bool)
            signs = np.where(positive, 1.0, -1.0)
            statistics = np.abs(signs @ values / n_mice)
            extreme += int(np.count_nonzero(statistics >= observed - tolerance))
        return SignFlipResult(
            statistic=observed,
            p_value=extreme / n_permutations,
            method="exact_two_sided_sign_flip",
            n_permutations=n_permutations,
            seed=None,
            n_mice=n_mice,
        )

    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        signs = rng.integers(0, 2, size=(stop - start, n_mice)) * 2 - 1
        statistics = np.abs(signs @ values / n_mice)
        extreme += int(np.count_nonzero(statistics >= observed - tolerance))
    return SignFlipResult(
        statistic=observed,
        p_value=(extreme + 1) / (n_resamples + 1),
        method="monte_carlo_two_sided_sign_flip",
        n_permutations=n_resamples,
        seed=seed,
        n_mice=n_mice,
    )


def _extract_unique_mouse_values(
    mouse_estimates: pl.DataFrame,
    *,
    mouse_column: str,
    value_column: str,
) -> np.ndarray:
    if not isinstance(mouse_estimates, pl.DataFrame):
        raise TypeError("mouse_estimates must be a polars DataFrame")
    missing = {mouse_column, value_column}.difference(mouse_estimates.columns)
    if missing:
        raise ValueError(f"mouse_estimates is missing columns: {sorted(missing)}")
    if mouse_estimates.height < 2:
        raise ValueError("mouse-level inference requires at least two mice")
    if mouse_estimates.get_column(mouse_column).null_count() > 0:
        raise ValueError("mouse identifiers must not be null")
    if mouse_estimates.get_column(mouse_column).n_unique() != mouse_estimates.height:
        raise ValueError("mouse_estimates must contain exactly one row per mouse")

    try:
        value_series = mouse_estimates.get_column(value_column).cast(pl.Float64, strict=True)
    except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError) as error:
        raise ValueError(f"{value_column} must be numeric: {error}") from error
    values = value_series.to_numpy()
    if not np.isfinite(values).all():
        raise ValueError(f"{value_column} must contain only finite, non-null estimates")
    return values


def _validate_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
