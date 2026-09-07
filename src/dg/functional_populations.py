"""Discovery-only functional-population analyses for Figure 3.

The functions in this module operate on already selected behavioral events and
bounded batches of spike trains.  NWB access stays in the orchestration script.
The primary result compares an early visual-change window with a later,
pre-response window and asks how much of the reversible reward-state contrast
remains after adjustment for measured licking and running.

Discrete labels are deliberately conservative.  A unit receives a named
response archetype only when independently interleaved trial halves agree on
both the dominant response component and its sign.  All units retain their
continuous response features regardless of label stability.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import math
from typing import Any

import numpy as np
import polars as pl

ENGAGED_1 = "engaged_1"
NO_REWARD = "no_reward"
ENGAGED_2 = "engaged_2"
REWARD_BLOCKS = (ENGAGED_1, NO_REWARD, ENGAGED_2)
RESPONSE_CLASSES = ("early_sensory", "late_prelick", "action_related")


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionalPopulationConfig:
    """Frozen event windows, binning, and regularization search space."""

    change_start_seconds: float = -0.250
    change_stop_seconds: float = 0.750
    bin_width_seconds: float = 0.025
    baseline_window: tuple[float, float] = (-0.250, 0.0)
    early_window: tuple[float, float] = (0.025, 0.150)
    late_window: tuple[float, float] = (0.150, 0.600)
    lick_start_seconds: float = -0.400
    lick_stop_seconds: float = 0.200
    lick_baseline_window: tuple[float, float] = (-0.400, -0.200)
    prelick_window: tuple[float, float] = (-0.200, 0.0)
    minimum_trials_per_block: int = 8
    minimum_prelick_trials_per_block: int = 5
    minimum_response_trials_for_action: int = 8
    outer_folds: int = 3
    inner_folds: int = 2
    ridge_lambdas: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    running_max_gap_seconds: float = 0.100

    def __post_init__(self) -> None:
        positive = ("bin_width_seconds", "running_max_gap_seconds")
        if any(
            not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0 for name in positive
        ):
            raise ValueError("bin width and running gap must be finite and positive")
        if not self.change_start_seconds < 0 < self.change_stop_seconds:
            raise ValueError("change-aligned interval must straddle zero")
        if not self.lick_start_seconds < 0 < self.lick_stop_seconds:
            raise ValueError("lick-aligned interval must straddle zero")
        for name in (
            "baseline_window",
            "early_window",
            "late_window",
            "lick_baseline_window",
            "prelick_window",
        ):
            start, stop = getattr(self, name)
            if not (math.isfinite(start) and math.isfinite(stop) and start < stop):
                raise ValueError(f"{name} must be a finite increasing interval")
        if (
            self.baseline_window[0] < self.change_start_seconds
            or self.late_window[1] > self.change_stop_seconds
        ):
            raise ValueError("change response windows must fit inside the binned interval")
        if (
            self.lick_baseline_window[0] < self.lick_start_seconds
            or self.prelick_window[1] > self.lick_stop_seconds
        ):
            raise ValueError("lick response windows must fit inside the binned interval")
        for name in (
            "minimum_trials_per_block",
            "minimum_prelick_trials_per_block",
            "minimum_response_trials_for_action",
            "outer_folds",
            "inner_folds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        if not self.ridge_lambdas or any(
            not math.isfinite(value) or value <= 0 for value in self.ridge_lambdas
        ):
            raise ValueError("ridge_lambdas must contain finite positive values")
        if tuple(sorted(set(self.ridge_lambdas))) != self.ridge_lambdas:
            raise ValueError("ridge_lambdas must be unique and increasing")
        for start, stop in (
            (self.change_start_seconds, self.change_stop_seconds),
            (self.lick_start_seconds, self.lick_stop_seconds),
        ):
            bins = (stop - start) / self.bin_width_seconds
            if not math.isclose(bins, round(bins), abs_tol=1e-9):
                raise ValueError("aligned intervals must contain an integer number of bins")


DEFAULT_FUNCTIONAL_POPULATION_CONFIG = FunctionalPopulationConfig()


@dataclasses.dataclass(frozen=True, slots=True)
class RidgeBatchResult:
    """Nested-contiguous-CV output for multiple unit outcomes."""

    coefficients: np.ndarray
    intercepts: np.ndarray
    selected_lambdas: np.ndarray
    oof_r2: np.ndarray
    oof_predictions: np.ndarray


@dataclasses.dataclass(frozen=True, slots=True)
class BinnedSpikeBatch:
    """Bounded change- and lick-aligned spike counts."""

    change_counts: np.ndarray
    change_bin_centers: np.ndarray
    lick_counts: np.ndarray
    lick_bin_centers: np.ndarray


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedRunningSamples:
    """Canonical running samples plus explicit raw-clock diagnostics."""

    timestamps: np.ndarray
    speed: np.ndarray
    n_raw_rows: int
    n_finite_pairs: int
    n_unique_timestamps: int
    n_nonincreasing_adjacent_raw: int
    n_decreasing_adjacent_raw: int
    n_duplicate_adjacent_raw: int
    n_exact_duplicates_removed: int
    timestamp_min: float | None
    timestamp_max: float | None
    usable: bool
    status: str


def select_functional_population_trials(
    trials: pl.DataFrame | pl.LazyFrame,
    *,
    config: FunctionalPopulationConfig = DEFAULT_FUNCTIONAL_POPULATION_CONFIG,
) -> pl.DataFrame:
    """Select completed familiar full-contrast physical image changes.

    Trial response labels come only from the exact-deduplicated raw-lick
    derivation.  The first response lick is retained for action alignment.
    """

    frame = trials.collect() if isinstance(trials, pl.LazyFrame) else trials
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("trials must be a polars DataFrame or LazyFrame")
    required = {
        "_table_index",
        "change_time",
        "change_image_name",
        "novel_image_id",
        "physical_image_change",
        "reward_block",
        "aborted",
        "auto_rewarded",
        "response_in_window",
        "response_latency_from_licks",
        "lick_times_valid",
    }
    _require_columns(frame, required, frame_name="trials")
    selected = frame.filter(
        ~pl.col("aborted").fill_null(True)
        & ~pl.col("auto_rewarded").fill_null(True)
        & pl.col("physical_image_change").fill_null(False)
        & pl.col("lick_times_valid").fill_null(False)
        & pl.col("change_time").is_not_null()
        & pl.col("change_time").is_finite()
        & pl.col("change_image_name").is_not_null()
        & pl.col("novel_image_id").is_not_null()
        & (pl.col("change_image_name") != pl.col("novel_image_id"))
        & pl.col("change_image_name").str.ends_with("_r-1.0")
        & pl.col("reward_block").is_in(REWARD_BLOCKS)
        & pl.col("response_in_window").is_not_null()
    ).with_columns(
        pl.when(pl.col("response_in_window"))
        .then(pl.col("change_time") + pl.col("response_latency_from_licks"))
        .otherwise(None)
        .alias("first_response_lick_time")
    )
    invalid_response = selected.filter(
        pl.col("response_in_window")
        & (
            pl.col("first_response_lick_time").is_null()
            | ~pl.col("first_response_lick_time").is_finite()
        )
    )
    if invalid_response.height:
        raise ValueError("responded trials require a finite raw-lick response latency")
    sort_columns = [
        column
        for column in ("_nwb_path", "change_time", "_table_index")
        if column in selected.columns
    ]
    selected = selected.sort(*sort_columns)
    rank_expression = pl.int_range(pl.len()).over("reward_block")
    selected = selected.with_columns(
        pl.when(pl.col("response_in_window"))
        .then(
            pl.min_horizontal(
                pl.col("response_latency_from_licks"),
                pl.lit(config.late_window[1]),
            )
        )
        .otherwise(pl.lit(config.late_window[1]))
        .alias("late_censor_stop_seconds")
    )
    return selected.with_columns(
        (rank_expression % 2).cast(pl.Int8).alias("assignment_half"),
        (
            pl.col("late_censor_stop_seconds") >= config.late_window[0] + config.bin_width_seconds
        ).alias("prelick_late_eligible"),
        pl.lit(config.bin_width_seconds).alias("functional_bin_width_seconds"),
    )


def summarize_running_at_events(
    timestamps: collections.abc.Sequence[float] | np.ndarray,
    speed: collections.abc.Sequence[float] | np.ndarray,
    event_times: collections.abc.Sequence[float] | np.ndarray,
    *,
    config: FunctionalPopulationConfig = DEFAULT_FUNCTIONAL_POPULATION_CONFIG,
) -> pl.DataFrame:
    """Interpolate running speed at baseline, early, and late window centers."""

    prepared = prepare_running_samples(timestamps, speed)
    if not prepared.usable:
        raise ValueError("running timestamps have fewer than two unique finite samples")
    times = prepared.timestamps
    values = prepared.speed
    events = np.asarray(event_times, dtype=float)
    if events.ndim != 1 or not np.isfinite(events).all():
        raise ValueError("event_times must be a finite one-dimensional array")

    offsets = {
        "running_baseline": sum(config.baseline_window) / 2,
        "running_early": sum(config.early_window) / 2,
        "running_late": sum(config.late_window) / 2,
    }
    columns: dict[str, Any] = {}
    for name, offset in offsets.items():
        queries = events + offset
        right = np.searchsorted(times, queries, side="left")
        right = np.clip(right, 1, times.size - 1)
        left = right - 1
        choose_right = np.abs(times[right] - queries) < np.abs(queries - times[left])
        nearest = np.where(choose_right, right, left)
        distance = np.abs(times[nearest] - queries)
        sampled = values[nearest].astype(float, copy=True)
        sampled[distance > config.running_max_gap_seconds] = np.nan
        columns[name] = sampled
    result = pl.DataFrame(columns).with_columns(
        (pl.col("running_early") - pl.col("running_baseline")).alias("running_early_delta"),
        (pl.col("running_late") - pl.col("running_baseline")).alias("running_late_delta"),
    )
    return result


def prepare_running_samples(
    timestamps: collections.abc.Sequence[float] | np.ndarray,
    speed: collections.abc.Sequence[float] | np.ndarray,
) -> PreparedRunningSamples:
    """Sort and exact-deduplicate finite running samples without interpolation.

    Speeds at exactly repeated timestamps are averaged.  A decreasing raw
    timestamp is therefore repaired only by ordering observed samples; no new
    time or speed values are fabricated.  Fewer than two unique finite samples
    is returned as explicitly unusable rather than raised here.
    """

    raw_times = np.asarray(timestamps, dtype=float)
    raw_speed = np.asarray(speed, dtype=float)
    if raw_times.ndim != 1 or raw_speed.ndim != 1 or raw_times.size != raw_speed.size:
        raise ValueError("running timestamps and speed must be equal one-dimensional arrays")
    finite = np.isfinite(raw_times) & np.isfinite(raw_speed)
    times = raw_times[finite]
    values = raw_speed[finite]
    raw_differences = np.diff(times)
    n_nonincreasing = int(np.sum(raw_differences <= 0))
    n_decreasing = int(np.sum(raw_differences < 0))
    n_duplicate_adjacent = int(np.sum(raw_differences == 0))
    if times.size:
        order = np.argsort(times, kind="stable")
        sorted_times = times[order]
        sorted_values = values[order]
        unique_times, starts, counts = np.unique(
            sorted_times,
            return_index=True,
            return_counts=True,
        )
        value_sums = np.add.reduceat(sorted_values, starts)
        unique_values = value_sums / counts
    else:
        unique_times = np.empty(0, dtype=float)
        unique_values = np.empty(0, dtype=float)
    usable = unique_times.size >= 2
    changed = (
        n_nonincreasing > 0
        or int(finite.sum()) != raw_times.size
        or unique_times.size != times.size
    )
    if not usable:
        status = "unavailable_fewer_than_two_unique_finite_timestamps"
    elif changed:
        status = "pass_sorted_exact_deduplicated_finite_pairs"
    else:
        status = "pass_raw_strictly_increasing"
    return PreparedRunningSamples(
        timestamps=unique_times,
        speed=unique_values,
        n_raw_rows=int(raw_times.size),
        n_finite_pairs=int(finite.sum()),
        n_unique_timestamps=int(unique_times.size),
        n_nonincreasing_adjacent_raw=n_nonincreasing,
        n_decreasing_adjacent_raw=n_decreasing,
        n_duplicate_adjacent_raw=n_duplicate_adjacent,
        n_exact_duplicates_removed=int(times.size - unique_times.size),
        timestamp_min=float(unique_times[0]) if unique_times.size else None,
        timestamp_max=float(unique_times[-1]) if unique_times.size else None,
        usable=usable,
        status=status,
    )


def bin_spike_batch(
    spike_times_by_unit: collections.abc.Sequence[collections.abc.Sequence[float] | np.ndarray],
    change_times: collections.abc.Sequence[float] | np.ndarray,
    response_lick_times: collections.abc.Sequence[float] | np.ndarray,
    *,
    config: FunctionalPopulationConfig = DEFAULT_FUNCTIONAL_POPULATION_CONFIG,
) -> BinnedSpikeBatch:
    """Count spikes in 25-ms bins without retaining raw spike vectors."""

    changes = np.asarray(change_times, dtype=float)
    licks = np.asarray(response_lick_times, dtype=float)
    if changes.ndim != 1 or not np.isfinite(changes).all():
        raise ValueError("change_times must be a finite one-dimensional array")
    if licks.ndim != 1 or licks.shape != changes.shape:
        raise ValueError("response_lick_times must match change_times")
    change_edges = _bin_edges(
        config.change_start_seconds,
        config.change_stop_seconds,
        config.bin_width_seconds,
    )
    lick_edges = _bin_edges(
        config.lick_start_seconds,
        config.lick_stop_seconds,
        config.bin_width_seconds,
    )
    valid_licks = np.isfinite(licks)
    units = list(spike_times_by_unit)
    change_counts = np.empty((changes.size, change_edges.size - 1, len(units)), dtype=np.int32)
    lick_counts = np.empty(
        (int(valid_licks.sum()), lick_edges.size - 1, len(units)),
        dtype=np.int32,
    )
    for unit_index, row in enumerate(units):
        spikes = np.asarray(row, dtype=float)
        if spikes.ndim != 1 or not np.isfinite(spikes).all():
            raise ValueError(f"unit {unit_index} spike times must be finite and one-dimensional")
        if spikes.size > 1 and np.any(np.diff(spikes) < 0):
            raise ValueError(f"unit {unit_index} spike times must be sorted")
        change_counts[:, :, unit_index] = _counts_around_events(spikes, changes, change_edges)
        if valid_licks.any():
            lick_counts[:, :, unit_index] = _counts_around_events(
                spikes,
                licks[valid_licks],
                lick_edges,
            )
    return BinnedSpikeBatch(
        change_counts=change_counts,
        change_bin_centers=(change_edges[:-1] + change_edges[1:]) / 2,
        lick_counts=lick_counts,
        lick_bin_centers=(lick_edges[:-1] + lick_edges[1:]) / 2,
    )


def variance_stabilized_window_response(
    counts: np.ndarray,
    bin_centers: np.ndarray,
    response_window: tuple[float, float],
    baseline_window: tuple[float, float],
    *,
    bin_width_seconds: float,
) -> np.ndarray:
    """Return trial x unit Anscombe-rate response minus baseline."""

    values = np.asarray(counts)
    centers = np.asarray(bin_centers, dtype=float)
    if values.ndim != 3 or centers.ndim != 1 or values.shape[1] != centers.size:
        raise ValueError("counts must be trial x bin x unit and match bin_centers")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("counts must contain finite non-negative values")
    response_mask = _window_mask(centers, response_window)
    baseline_mask = _window_mask(centers, baseline_window)
    response_duration = response_mask.sum() * bin_width_seconds
    baseline_duration = baseline_mask.sum() * bin_width_seconds
    response_counts = values[:, response_mask, :].sum(axis=1)
    baseline_counts = values[:, baseline_mask, :].sum(axis=1)
    response_rate = np.sqrt(response_counts + 3.0 / 8.0) / math.sqrt(response_duration)
    baseline_rate = np.sqrt(baseline_counts + 3.0 / 8.0) / math.sqrt(baseline_duration)
    return response_rate - baseline_rate


def censored_variance_stabilized_window_response(
    counts: np.ndarray,
    bin_centers: np.ndarray,
    censor_stop_seconds: collections.abc.Sequence[float] | np.ndarray,
    response_window: tuple[float, float],
    baseline_window: tuple[float, float],
    *,
    bin_width_seconds: float,
) -> np.ndarray:
    """Return pre-censor trial responses with explicit variable exposure.

    A response bin contributes only when its right edge does not exceed the
    per-trial censor time. Rows with no complete response bin are ``NaN``.
    Baseline is fixed and always precedes the event.
    """

    values = np.asarray(counts)
    centers = np.asarray(bin_centers, dtype=float)
    stops = np.asarray(censor_stop_seconds, dtype=float)
    if values.ndim != 3 or centers.ndim != 1 or values.shape[1] != centers.size:
        raise ValueError("counts must be trial x bin x unit and match bin_centers")
    if stops.shape != (values.shape[0],):
        raise ValueError("censor_stop_seconds must contain one value per trial")
    if np.any(values < 0) or not np.isfinite(values).all():
        raise ValueError("counts must contain finite non-negative values")
    baseline_mask = _window_mask(centers, baseline_window)
    candidate = (centers >= response_window[0]) & (centers < response_window[1])
    right_edges = centers + bin_width_seconds / 2
    included = candidate[np.newaxis, :] & (right_edges[np.newaxis, :] <= stops[:, np.newaxis])
    n_bins = included.sum(axis=1)
    response_counts = np.einsum("tbu,tb->tu", values, included, optimize=True)
    baseline_counts = values[:, baseline_mask, :].sum(axis=1)
    baseline_duration = baseline_mask.sum() * bin_width_seconds
    response_duration = n_bins * bin_width_seconds
    with np.errstate(divide="ignore", invalid="ignore"):
        response_rate = np.sqrt(response_counts + 3.0 / 8.0) / np.sqrt(
            response_duration[:, np.newaxis]
        )
    baseline_rate = np.sqrt(baseline_counts + 3.0 / 8.0) / math.sqrt(baseline_duration)
    response = response_rate - baseline_rate
    response[n_bins == 0] = np.nan
    return response


def build_model_design(
    trials: pl.DataFrame,
    *,
    window: str,
    adjusted: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build an interpretable reward-state design with optional movement terms."""

    if window not in {"early", "late"}:
        raise ValueError("window must be 'early' or 'late'")
    required = {
        "reward_block",
        "change_time",
        "change_image_name",
        "response_in_window",
        "response_latency_from_licks",
    }
    if adjusted:
        required.update({"running_baseline", f"running_{window}", f"running_{window}_delta"})
    _require_columns(trials, required, frame_name="trials")
    if trials.is_empty():
        raise ValueError("trials must not be empty")
    blocks = trials.get_column("reward_block").to_list()
    if any(block not in REWARD_BLOCKS for block in blocks):
        raise ValueError("reward_block contains an unknown label")
    times = trials.get_column("change_time").cast(pl.Float64).to_numpy()
    time_centered = (times - times.mean()) / max(float(times.std()), 1e-12)
    reversible = np.array(
        [0.5 if block in {ENGAGED_1, ENGAGED_2} else -1.0 for block in blocks],
        dtype=float,
    )
    drift = np.array(
        [-0.5 if block == ENGAGED_1 else 0.5 if block == ENGAGED_2 else 0.0 for block in blocks],
        dtype=float,
    )
    columns = [reversible, drift, time_centered, time_centered**2]
    names = ["reward_reversible", "engaged_drift", "session_time", "session_time_squared"]
    identities = trials.get_column("change_image_name").cast(pl.String).to_list()
    levels = sorted(set(identities))
    for level in levels[1:]:
        columns.append(np.asarray([value == level for value in identities], dtype=float))
        names.append(f"image[{level}]")
    if adjusted:
        responded = trials.get_column("response_in_window").cast(pl.Float64).to_numpy()
        latency = trials.get_column("response_latency_from_licks").cast(pl.Float64).to_numpy()
        missing_latency = ~np.isfinite(latency)
        finite_latency = latency[~missing_latency]
        fill = float(np.median(finite_latency)) if finite_latency.size else 0.45
        latency = latency.copy()
        latency[missing_latency] = fill
        columns.extend(
            [
                responded,
                latency,
                trials.get_column("running_baseline").cast(pl.Float64).to_numpy(),
                trials.get_column(f"running_{window}_delta").cast(pl.Float64).to_numpy(),
            ]
        )
        names.extend(
            [
                "lick_response",
                "lick_latency",
                "running_baseline",
                f"running_{window}_delta",
            ]
        )
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError("model design contains missing or non-finite values")
    return matrix, tuple(names)


def nested_contiguous_ridge(
    design: np.ndarray,
    outcomes: np.ndarray,
    reward_blocks: collections.abc.Sequence[str] | np.ndarray,
    event_times: collections.abc.Sequence[float] | np.ndarray,
    *,
    config: FunctionalPopulationConfig = DEFAULT_FUNCTIONAL_POPULATION_CONFIG,
) -> RidgeBatchResult:
    """Fit ridge models with outer OOF prediction and inner lambda selection.

    Folds are contiguous independently within each reward block.  Scaling,
    lambda selection, and intercept estimation are recomputed within each
    training fold.  Lambdas are selected separately for each unit.
    """

    x = np.asarray(design, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    blocks = np.asarray(reward_blocks, dtype=object)
    times = np.asarray(event_times, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("design and outcomes must be row-aligned two-dimensional arrays")
    if blocks.shape != (x.shape[0],) or times.shape != (x.shape[0],):
        raise ValueError("reward_blocks and event_times must align with design rows")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(times).all():
        raise ValueError("model inputs must be finite")
    if x.shape[0] < config.outer_folds * 2 or y.shape[1] < 1:
        raise ValueError("insufficient rows or outcomes for nested cross-validation")
    outer_ids = contiguous_block_folds(blocks, times, config.outer_folds)
    oof = np.full_like(y, np.nan, dtype=float)
    chosen_outer = np.empty((config.outer_folds, y.shape[1]), dtype=float)
    lambdas = np.asarray(config.ridge_lambdas, dtype=float)
    for outer_fold in range(config.outer_folds):
        test = outer_ids == outer_fold
        train = ~test
        if not test.any() or train.sum() < config.inner_folds * 2:
            raise ValueError("an outer fold lacks train or test observations")
        inner_ids = contiguous_block_folds(
            blocks[train],
            times[train],
            config.inner_folds,
        )
        errors = np.zeros((lambdas.size, y.shape[1]), dtype=float)
        for inner_fold in range(config.inner_folds):
            inner_test = inner_ids == inner_fold
            inner_train = ~inner_test
            for lambda_index, ridge_lambda in enumerate(lambdas):
                coef, intercept = _ridge_fit(
                    x[train][inner_train],
                    y[train][inner_train],
                    ridge_lambda,
                )
                prediction = x[train][inner_test] @ coef + intercept
                errors[lambda_index] += np.square(y[train][inner_test] - prediction).sum(axis=0)
        selected = np.argmin(errors, axis=0)
        chosen_outer[outer_fold] = lambdas[selected]
        for lambda_index, ridge_lambda in enumerate(lambdas):
            unit_mask = selected == lambda_index
            if not unit_mask.any():
                continue
            coef, intercept = _ridge_fit(x[train], y[train][:, unit_mask], ridge_lambda)
            oof[test][:, unit_mask] = x[test] @ coef + intercept
    if not np.isfinite(oof).all():
        # Boolean indexing creates a copy for two-dimensional assignment above.
        # Repeat assignment with explicit row/column indices to guarantee fill.
        oof[:] = np.nan
        for outer_fold in range(config.outer_folds):
            test_rows = np.flatnonzero(outer_ids == outer_fold)
            train = outer_ids != outer_fold
            selected_values = chosen_outer[outer_fold]
            for ridge_lambda in lambdas:
                unit_indices = np.flatnonzero(selected_values == ridge_lambda)
                if not unit_indices.size:
                    continue
                coef, intercept = _ridge_fit(x[train], y[train][:, unit_indices], ridge_lambda)
                oof[np.ix_(test_rows, unit_indices)] = x[test_rows] @ coef + intercept
    residual = np.square(y - oof).sum(axis=0)
    total = np.square(y - y.mean(axis=0)).sum(axis=0)
    fraction = np.full_like(total, np.nan, dtype=float)
    np.divide(residual, total, out=fraction, where=total > 0)
    oof_r2 = 1.0 - fraction

    full_fold_ids = contiguous_block_folds(blocks, times, config.outer_folds)
    errors = np.zeros((lambdas.size, y.shape[1]), dtype=float)
    for fold in range(config.outer_folds):
        test = full_fold_ids == fold
        train = ~test
        for lambda_index, ridge_lambda in enumerate(lambdas):
            coef, intercept = _ridge_fit(x[train], y[train], ridge_lambda)
            prediction = x[test] @ coef + intercept
            errors[lambda_index] += np.square(y[test] - prediction).sum(axis=0)
    selected = np.argmin(errors, axis=0)
    final_coefficients = np.empty((x.shape[1], y.shape[1]), dtype=float)
    final_intercepts = np.empty(y.shape[1], dtype=float)
    for lambda_index, ridge_lambda in enumerate(lambdas):
        unit_mask = selected == lambda_index
        if not unit_mask.any():
            continue
        coef, intercept = _ridge_fit(x, y[:, unit_mask], ridge_lambda)
        final_coefficients[:, unit_mask] = coef
        final_intercepts[unit_mask] = intercept
    return RidgeBatchResult(
        coefficients=final_coefficients,
        intercepts=final_intercepts,
        selected_lambdas=lambdas[selected],
        oof_r2=oof_r2,
        oof_predictions=oof,
    )


def contiguous_block_folds(
    reward_blocks: collections.abc.Sequence[str] | np.ndarray,
    event_times: collections.abc.Sequence[float] | np.ndarray,
    n_folds: int,
) -> np.ndarray:
    """Assign contiguous, within-state temporal segments to fold IDs."""

    if isinstance(n_folds, bool) or not isinstance(n_folds, int) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least two")
    blocks = np.asarray(reward_blocks, dtype=object)
    times = np.asarray(event_times, dtype=float)
    if blocks.ndim != 1 or times.shape != blocks.shape or not np.isfinite(times).all():
        raise ValueError("reward_blocks and event_times must be aligned one-dimensional arrays")
    if any(block not in REWARD_BLOCKS for block in blocks):
        raise ValueError("reward_blocks contains an unknown state")
    folds = np.full(blocks.size, -1, dtype=int)
    for block in REWARD_BLOCKS:
        indices = np.flatnonzero(blocks == block)
        if indices.size < n_folds:
            raise ValueError(f"{block} has fewer observations than folds")
        ordered = indices[np.argsort(times[indices], kind="stable")]
        for fold, segment in enumerate(np.array_split(ordered, n_folds)):
            folds[segment] = fold
    if np.any(folds < 0):
        raise RuntimeError("fold assignment left observations unassigned")
    return folds


def classify_response_archetypes(
    early_response: np.ndarray,
    late_response: np.ndarray,
    action_response: np.ndarray,
    assignment_halves: collections.abc.Sequence[int] | np.ndarray,
    responded: collections.abc.Sequence[bool] | np.ndarray,
) -> pl.DataFrame:
    """Assign stable dominant response archetypes from two trial halves."""

    early = np.asarray(early_response, dtype=float)
    late = np.asarray(late_response, dtype=float)
    action = np.asarray(action_response, dtype=float)
    halves = np.asarray(assignment_halves, dtype=int)
    response_mask = np.asarray(responded, dtype=bool)
    if early.ndim != 2 or late.shape != early.shape:
        raise ValueError("early and late responses must be aligned trial x unit matrices")
    if halves.shape != (early.shape[0],) or response_mask.shape != (early.shape[0],):
        raise ValueError("assignment halves and response mask must align with trial rows")
    if (
        action.ndim != 2
        or action.shape[0] != response_mask.sum()
        or action.shape[1] != early.shape[1]
    ):
        raise ValueError("action responses must contain one row per responded trial")
    if set(np.unique(halves)) != {0, 1}:
        raise ValueError("assignment_halves must contain both 0 and 1")

    response_halves = halves[response_mask]
    labels_by_half: list[np.ndarray] = []
    signs_by_half: list[np.ndarray] = []
    margins_by_half: list[np.ndarray] = []
    amplitudes: list[np.ndarray] = []
    for half in (0, 1):
        components = np.vstack(
            [
                np.nanmean(early[halves == half], axis=0),
                np.nanmean(late[halves == half], axis=0),
                np.nanmean(action[response_halves == half], axis=0),
            ]
        )
        absolute = np.abs(components)
        dominant = np.argmax(absolute, axis=0)
        ordered = np.sort(absolute, axis=0)
        top = ordered[-1]
        second = ordered[-2]
        margins_by_half.append((top - second) / np.maximum(top, 1e-12))
        labels_by_half.append(dominant)
        signs_by_half.append(np.sign(components[dominant, np.arange(components.shape[1])]))
        amplitudes.append(components)
    stable = (labels_by_half[0] == labels_by_half[1]) & (signs_by_half[0] == signs_by_half[1])
    stable &= signs_by_half[0] != 0
    final_class = np.where(
        stable,
        np.asarray(RESPONSE_CLASSES, dtype=object)[labels_by_half[0]],
        "continuous_only",
    )
    dominant_sign = np.where(stable, signs_by_half[0], np.nan)
    return pl.DataFrame(
        {
            "assignment_half_0_class": np.asarray(RESPONSE_CLASSES, dtype=object)[
                labels_by_half[0]
            ],
            "assignment_half_1_class": np.asarray(RESPONSE_CLASSES, dtype=object)[
                labels_by_half[1]
            ],
            "class_assignment_stable": stable,
            "response_class": final_class,
            "dominant_response_sign": dominant_sign,
            "assignment_margin": np.minimum(margins_by_half[0], margins_by_half[1]),
            "early_response": np.nanmean(early, axis=0),
            "late_response": np.nanmean(late, axis=0),
            "action_response": np.nanmean(action, axis=0),
        }
    )


def reversible_contrast(
    values: np.ndarray,
    reward_blocks: collections.abc.Sequence[str],
) -> np.ndarray:
    """Return unit-wise ``0.5*E1 - NR + 0.5*E2`` block contrast."""

    matrix = np.asarray(values, dtype=float)
    blocks = np.asarray(reward_blocks, dtype=object)
    if matrix.ndim != 2 or blocks.shape != (matrix.shape[0],):
        raise ValueError("values must be trial x unit and align with reward_blocks")
    means = []
    for block in REWARD_BLOCKS:
        selected = matrix[blocks == block]
        if selected.size == 0:
            raise ValueError(f"reward block {block!r} has no observations")
        means.append(np.nanmean(selected, axis=0))
    return 0.5 * means[0] - means[1] + 0.5 * means[2]


def _ridge_fit(x: np.ndarray, y: np.ndarray, ridge_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale <= 1e-12] = 1.0
    y_mean = y.mean(axis=0)
    standardized = (x - x_mean) / x_scale
    gram = standardized.T @ standardized
    penalty = ridge_lambda * np.eye(gram.shape[0])
    standardized_coef = np.linalg.solve(gram + penalty, standardized.T @ (y - y_mean))
    coef = standardized_coef / x_scale[:, np.newaxis]
    intercept = y_mean - x_mean @ coef
    return coef, intercept


def _counts_around_events(spikes: np.ndarray, events: np.ndarray, edges: np.ndarray) -> np.ndarray:
    absolute_edges = events[:, np.newaxis] + edges[np.newaxis, :]
    positions = np.searchsorted(spikes, absolute_edges, side="left")
    return np.diff(positions, axis=1).astype(np.int32, copy=False)


def _bin_edges(start: float, stop: float, width: float) -> np.ndarray:
    n_bins = int(round((stop - start) / width))
    return start + np.arange(n_bins + 1, dtype=float) * width


def _window_mask(centers: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    mask = (centers >= window[0]) & (centers < window[1])
    if not mask.any():
        raise ValueError(f"window {window!r} contains no bins")
    return mask


def _require_columns(frame: pl.DataFrame, columns: set[str], *, frame_name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {sorted(missing)}")
