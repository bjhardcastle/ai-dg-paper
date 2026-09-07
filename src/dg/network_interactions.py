"""Discovery-only simultaneous inter-regional predictive interactions.

The functions in this module operate on already-reduced trial-by-unit spike
features.  NWB access belongs in the orchestration script.  For each ordered
source-target region pair, a nested contiguous-fold model asks whether early
source activity improves held-out prediction of later target activity beyond
stimulus/session-time covariates and the target population's own early
response.  A within-block cyclic trial shift is fit with the same procedure.

The estimand is predictive association, not anatomical direction or causal
information flow.  Sessions are summarized within mouse before inference.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import json
import math
from typing import Any

import numpy as np
import polars as pl

import dg.statistics

ENGAGED_1 = "engaged_1"
NO_REWARD = "no_reward"
ENGAGED_2 = "engaged_2"
REWARD_BLOCKS = (ENGAGED_1, NO_REWARD, ENGAGED_2)


@dataclasses.dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Fixed windows, support thresholds, and nested-CV search space."""

    baseline_start_seconds: float = -0.150
    baseline_stop_seconds: float = 0.0
    source_start_seconds: float = 0.0
    source_stop_seconds: float = 0.150
    target_start_seconds: float = 0.150
    target_stop_seconds: float = 0.300
    minimum_trials_per_block: int = 12
    minimum_units_per_region: int = 8
    maximum_units_per_region: int = 24
    minimum_pair_mice: int = 5
    minimum_pair_sessions: int = 8
    minimum_median_units_per_mouse: int = 10
    outer_folds: int = 4
    inner_folds: int = 3
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    ranks: tuple[int, ...] = (1, 2, 4, 8)

    def __post_init__(self) -> None:
        windows = (
            self.baseline_start_seconds,
            self.baseline_stop_seconds,
            self.source_start_seconds,
            self.source_stop_seconds,
            self.target_start_seconds,
            self.target_stop_seconds,
        )
        if not all(math.isfinite(value) for value in windows):
            raise ValueError("network windows must be finite")
        if not (
            self.baseline_start_seconds
            < self.baseline_stop_seconds
            <= self.source_start_seconds
            < self.source_stop_seconds
            <= self.target_start_seconds
            < self.target_stop_seconds
        ):
            raise ValueError("network windows must be positive-lag, ordered, and non-overlapping")
        for name in (
            "minimum_trials_per_block",
            "minimum_units_per_region",
            "maximum_units_per_region",
            "minimum_pair_mice",
            "minimum_pair_sessions",
            "minimum_median_units_per_mouse",
            "outer_folds",
            "inner_folds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        if self.maximum_units_per_region < self.minimum_units_per_region:
            raise ValueError("maximum_units_per_region must not be below the minimum")
        if tuple(sorted(set(self.ridge_alphas))) != self.ridge_alphas or any(
            not math.isfinite(value) or value <= 0 for value in self.ridge_alphas
        ):
            raise ValueError("ridge_alphas must be unique, increasing, finite, and positive")
        if tuple(sorted(set(self.ranks))) != self.ranks or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.ranks
        ):
            raise ValueError("ranks must be unique, increasing positive integers")


DEFAULT_NETWORK_CONFIG = NetworkConfig()


@dataclasses.dataclass(frozen=True, slots=True)
class NetworkFitTables:
    """Out-of-fold block and fold estimates for one population pair."""

    block_performance: pl.DataFrame
    fold_performance: pl.DataFrame
    balanced_trial_indices: np.ndarray


def select_network_trials(
    trials: pl.DataFrame | pl.LazyFrame,
    *,
    config: NetworkConfig = DEFAULT_NETWORK_CONFIG,
) -> pl.DataFrame:
    """Select familiar full-contrast changes with no lick through 300 ms.

    The closed censoring interval spans the baseline start through the target
    window endpoint.  A lick exactly at 300 ms therefore excludes a trial even
    though the spike-count endpoint itself is half-open.
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
        "lick_times",
    }
    _require_columns(frame, required, frame_name="trials")
    candidate = frame.filter(
        ~pl.col("aborted").fill_null(True)
        & ~pl.col("auto_rewarded").fill_null(True)
        & pl.col("physical_image_change").fill_null(False)
        & pl.col("change_time").is_not_null()
        & pl.col("change_time").is_finite()
        & pl.col("change_image_name").is_not_null()
        & pl.col("novel_image_id").is_not_null()
        & (pl.col("change_image_name") != pl.col("novel_image_id"))
        & pl.col("change_image_name").str.ends_with("_r-1.0")
        & pl.col("reward_block").is_in(REWARD_BLOCKS)
        & pl.col("lick_times").is_not_null()
    )
    if "lick_times_valid" in candidate.columns:
        candidate = candidate.filter(pl.col("lick_times_valid").fill_null(False))

    censor_start = config.baseline_start_seconds
    censor_stop = config.target_stop_seconds
    keep: list[bool] = []
    for row in candidate.select("lick_times", "change_time").iter_rows(named=True):
        licks = np.asarray(row["lick_times"], dtype=float)
        if licks.ndim != 1 or not np.isfinite(licks).all():
            keep.append(False)
            continue
        event_time = float(row["change_time"])
        keep.append(
            not bool(
                np.any((licks >= event_time + censor_start) & (licks <= event_time + censor_stop))
            )
        )
    sort_columns = [
        column
        for column in ("_nwb_path", "change_time", "_table_index")
        if column in candidate.columns
    ]
    return (
        candidate.with_columns(
            pl.Series("network_lick_free", keep, dtype=pl.Boolean),
            pl.lit(censor_start).alias("network_lick_censor_start_seconds"),
            pl.lit(censor_stop).alias("network_lick_censor_stop_seconds"),
        )
        .filter(pl.col("network_lick_free"))
        .sort(*sort_columns)
    )


def build_windowed_population_features(
    spike_times_by_unit: collections.abc.Sequence[collections.abc.Sequence[float] | np.ndarray],
    event_times: collections.abc.Sequence[float] | np.ndarray,
    *,
    config: NetworkConfig = DEFAULT_NETWORK_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Return variance-stabilized early and late responses, trials by units."""

    events = np.asarray(event_times, dtype=float)
    if events.ndim != 1 or not np.isfinite(events).all():
        raise ValueError("event_times must be a one-dimensional finite array")
    units = list(spike_times_by_unit)
    early = np.empty((events.size, len(units)), dtype=float)
    late = np.empty_like(early)
    for unit_index, values in enumerate(units):
        spikes = np.asarray(values, dtype=float)
        if spikes.ndim != 1 or not np.isfinite(spikes).all():
            raise ValueError(f"unit {unit_index} spike times must be one-dimensional and finite")
        if spikes.size > 1 and np.any(np.diff(spikes) < 0):
            raise ValueError(f"unit {unit_index} spike times must be sorted")
        baseline = _window_counts(
            spikes,
            events + config.baseline_start_seconds,
            events + config.baseline_stop_seconds,
        )
        source = _window_counts(
            spikes,
            events + config.source_start_seconds,
            events + config.source_stop_seconds,
        )
        target = _window_counts(
            spikes,
            events + config.target_start_seconds,
            events + config.target_stop_seconds,
        )
        transformed_baseline = np.sqrt(baseline + 3.0 / 8.0)
        early[:, unit_index] = np.sqrt(source + 3.0 / 8.0) - transformed_baseline
        late[:, unit_index] = np.sqrt(target + 3.0 / 8.0) - transformed_baseline
    return early, late


def make_trial_covariates(trials: pl.DataFrame) -> tuple[np.ndarray, tuple[str, ...]]:
    """Construct prespecified stimulus, slow-time, and response covariates."""

    if not isinstance(trials, pl.DataFrame):
        raise TypeError("trials must be a polars DataFrame")
    _require_columns(
        trials,
        {"change_time", "change_image_name", "response_in_window"},
        frame_name="trials",
    )
    if trials.is_empty():
        raise ValueError("trials must not be empty")
    times = trials.get_column("change_time").cast(pl.Float64).to_numpy()
    if not np.isfinite(times).all():
        raise ValueError("change_time must be finite")
    midpoint = 0.5 * (times.min() + times.max())
    half_range = max(0.5 * (times.max() - times.min()), 1.0)
    slow_time = (times - midpoint) / half_range
    columns = [slow_time, slow_time**2]
    names = ["session_time_linear", "session_time_quadratic"]

    identities = sorted(trials.get_column("change_image_name").drop_nulls().unique().to_list())
    identity_values = np.asarray(trials.get_column("change_image_name"), dtype=object)
    for identity in identities[1:]:
        columns.append((identity_values == identity).astype(float))
        names.append(f"image_identity:{identity}")

    responses = trials.get_column("response_in_window").fill_null(False).cast(pl.Float64).to_numpy()
    columns.append(responses)
    names.append("eventual_response_in_window")
    if "response_latency_from_licks" in trials.columns:
        latency = trials.get_column("response_latency_from_licks").cast(pl.Float64).to_numpy()
        finite = np.isfinite(latency)
        fill = float(np.median(latency[finite])) if finite.any() else 0.0
        columns.append(np.where(finite, latency, fill))
        columns.append((~finite).astype(float))
        names.extend(("eventual_response_latency", "response_latency_missing"))
    matrix = np.column_stack(columns).astype(float, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("trial covariates must be finite after declared missingness encoding")
    return matrix, tuple(names)


def fit_network_pair_by_block(
    source_early: np.ndarray,
    target_early: np.ndarray,
    target_late: np.ndarray,
    nuisance_covariates: np.ndarray,
    reward_blocks: collections.abc.Sequence[str] | np.ndarray,
    event_times: collections.abc.Sequence[float] | np.ndarray,
    *,
    config: NetworkConfig = DEFAULT_NETWORK_CONFIG,
) -> NetworkFitTables:
    """Fit nested-CV real and trial-shift source increments in each block."""

    source = _as_finite_matrix(source_early, "source_early")
    target_history = _as_finite_matrix(target_early, "target_early")
    target = _as_finite_matrix(target_late, "target_late")
    nuisance = _as_finite_matrix(nuisance_covariates, "nuisance_covariates")
    blocks = np.asarray(reward_blocks, dtype=object)
    times = np.asarray(event_times, dtype=float)
    n_rows = source.shape[0]
    if any(matrix.shape[0] != n_rows for matrix in (target_history, target, nuisance)):
        raise ValueError("all feature matrices must have equal row counts")
    if blocks.ndim != 1 or times.ndim != 1 or blocks.size != n_rows or times.size != n_rows:
        raise ValueError("blocks and times must be one-dimensional and match feature rows")
    if not np.isfinite(times).all():
        raise ValueError("event_times must be finite")
    unknown = set(blocks.tolist()).difference(REWARD_BLOCKS)
    if unknown:
        raise ValueError(f"reward_blocks contains unknown values: {sorted(unknown)}")
    if source.shape[1] < config.minimum_units_per_region:
        raise ValueError("source population is below minimum_units_per_region")
    if target.shape[1] < config.minimum_units_per_region:
        raise ValueError("target population is below minimum_units_per_region")
    if target_history.shape[1] != target.shape[1]:
        raise ValueError("target early and late matrices must contain the same units")

    balanced = _balanced_trial_indices(blocks, times, config.minimum_trials_per_block)
    fold_rows: list[dict[str, Any]] = []
    block_rows: list[dict[str, Any]] = []
    for block in REWARD_BLOCKS:
        indices = balanced[blocks[balanced] == block]
        order = indices[np.argsort(times[indices], kind="stable")]
        block_source = source[order]
        shifted_source = np.roll(block_source, shift=max(1, order.size // 2), axis=0)
        block_history = target_history[order]
        block_target = target[order]
        block_nuisance = nuisance[order]
        base = np.column_stack((block_nuisance, block_history))
        n_outer = min(config.outer_folds, order.size)
        outer_ids = _contiguous_fold_ids(order.size, n_outer)
        observed_parts: list[np.ndarray] = []
        base_parts: list[np.ndarray] = []
        full_parts: list[np.ndarray] = []
        shift_parts: list[np.ndarray] = []
        selected_real: list[tuple[float, float, int]] = []
        selected_shift: list[tuple[float, float, int]] = []
        for fold_id in range(n_outer):
            validation = outer_ids == fold_id
            training = ~validation
            base_alpha = _select_base_alpha(
                base[training],
                block_target[training],
                times[order][training],
                config=config,
            )
            real_alpha, real_rank = _select_hyperparameters(
                base[training],
                block_source[training],
                block_target[training],
                times[order][training],
                base_alpha=base_alpha,
                config=config,
            )
            # Keep model complexity identical for the shifted control. Retuning
            # would change the base prediction and confound the null comparison.
            shift_alpha, shift_rank = real_alpha, real_rank
            real_prediction = _fit_predict_residual_rrr(
                base[training],
                block_source[training],
                block_target[training],
                base[validation],
                block_source[validation],
                block_target[validation],
                base_alpha=base_alpha,
                source_alpha=real_alpha,
                rank=real_rank,
            )
            shift_prediction = _fit_predict_residual_rrr(
                base[training],
                shifted_source[training],
                block_target[training],
                base[validation],
                shifted_source[validation],
                block_target[validation],
                base_alpha=base_alpha,
                source_alpha=shift_alpha,
                rank=shift_rank,
            )
            y_observed = real_prediction["observed"]
            observed_parts.append(y_observed)
            base_parts.append(real_prediction["base_prediction"])
            full_parts.append(real_prediction["full_prediction"])
            shift_parts.append(shift_prediction["full_prediction"])
            selected_real.append((base_alpha, real_alpha, real_rank))
            selected_shift.append((base_alpha, shift_alpha, shift_rank))
            fold_rows.append(
                {
                    "reward_block": block,
                    "outer_fold": fold_id,
                    "n_train_trials": int(training.sum()),
                    "n_validation_trials": int(validation.sum()),
                    "base_alpha": base_alpha,
                    "real_alpha": real_alpha,
                    "real_rank": real_rank,
                    "shift_alpha": shift_alpha,
                    "shift_rank": shift_rank,
                    "base_r2": _r2(y_observed, real_prediction["base_prediction"]),
                    "full_r2": _r2(y_observed, real_prediction["full_prediction"]),
                    "shift_full_r2": _r2(y_observed, shift_prediction["full_prediction"]),
                }
            )
        observed = np.vstack(observed_parts)
        base_prediction = np.vstack(base_parts)
        full_prediction = np.vstack(full_parts)
        shift_prediction = np.vstack(shift_parts)
        base_r2 = _r2(observed, base_prediction)
        full_r2 = _r2(observed, full_prediction)
        shift_full_r2 = _r2(observed, shift_prediction)
        block_rows.append(
            {
                "reward_block": block,
                "n_trials": int(order.size),
                "n_source_units": int(source.shape[1]),
                "n_target_units": int(target.shape[1]),
                "base_r2": base_r2,
                "full_r2": full_r2,
                "source_added_r2": full_r2 - base_r2,
                "shift_full_r2": shift_full_r2,
                "shift_source_added_r2": shift_full_r2 - base_r2,
                "real_minus_shift_source_added_r2": full_r2 - shift_full_r2,
                "outer_real_hyperparameters": json.dumps(selected_real),
                "outer_shift_hyperparameters": json.dumps(selected_shift),
                "model_status": "pass",
            }
        )
    return NetworkFitTables(
        block_performance=pl.DataFrame(block_rows),
        fold_performance=pl.DataFrame(fold_rows),
        balanced_trial_indices=balanced,
    )


def build_region_pair_coverage(
    unit_anatomy: pl.DataFrame,
    *,
    config: NetworkConfig = DEFAULT_NETWORK_CONFIG,
) -> pl.DataFrame:
    """Summarize ordered simultaneous-region coverage before model fitting."""

    if not isinstance(unit_anatomy, pl.DataFrame):
        raise TypeError("unit_anatomy must be a polars DataFrame")
    _require_columns(
        unit_anatomy,
        {"_nwb_path", "subject_id", "network_region", "well_isolated"},
        frame_name="unit_anatomy",
    )
    eligible = unit_anatomy.filter(
        pl.col("well_isolated").fill_null(False)
        & pl.col("network_region").is_not_null()
        & (pl.col("network_region").str.strip_chars() != "")
    )
    session_regions = eligible.group_by("_nwb_path", "subject_id", "network_region").agg(
        pl.len().alias("n_units")
    )
    rows: list[dict[str, Any]] = []
    for session in session_regions.partition_by("_nwb_path", maintain_order=True):
        values = session.sort("network_region").to_dicts()
        for source in values:
            for target in values:
                if source["network_region"] == target["network_region"]:
                    continue
                rows.append(
                    {
                        "_nwb_path": source["_nwb_path"],
                        "subject_id": str(source["subject_id"]),
                        "source_region": source["network_region"],
                        "target_region": target["network_region"],
                        "n_source_units": int(source["n_units"]),
                        "n_target_units": int(target["n_units"]),
                    }
                )
    schema = {
        "_nwb_path": pl.String,
        "subject_id": pl.String,
        "source_region": pl.String,
        "target_region": pl.String,
        "n_source_units": pl.Int64,
        "n_target_units": pl.Int64,
    }
    session_pairs = pl.DataFrame(rows, schema=schema)
    if session_pairs.is_empty():
        return pl.DataFrame(
            schema={
                "source_region": pl.String,
                "target_region": pl.String,
                "n_mice": pl.Int64,
                "n_sessions": pl.Int64,
                "median_source_units_per_mouse": pl.Float64,
                "median_target_units_per_mouse": pl.Float64,
                "minimum_source_units_per_session": pl.Int64,
                "minimum_target_units_per_session": pl.Int64,
                "coverage_eligible": pl.Boolean,
                "coverage_reason": pl.String,
            }
        )
    qualifying_sessions = session_pairs.filter(
        (pl.col("n_source_units") >= config.minimum_units_per_region)
        & (pl.col("n_target_units") >= config.minimum_units_per_region)
    )
    output: list[dict[str, Any]] = []
    for pair in (
        session_pairs.select("source_region", "target_region")
        .unique()
        .sort("source_region", "target_region")
        .iter_rows(named=True)
    ):
        pair_rows = qualifying_sessions.filter(
            (pl.col("source_region") == pair["source_region"])
            & (pl.col("target_region") == pair["target_region"])
        )
        by_mouse = pair_rows.group_by("subject_id").agg(
            pl.col("n_source_units").mean().alias("source_units"),
            pl.col("n_target_units").mean().alias("target_units"),
        )
        n_mice = by_mouse.height
        n_sessions = pair_rows.height
        median_source = float(by_mouse.get_column("source_units").median()) if n_mice else 0.0
        median_target = float(by_mouse.get_column("target_units").median()) if n_mice else 0.0
        minimum_source = int(pair_rows.get_column("n_source_units").min()) if n_sessions else 0
        minimum_target = int(pair_rows.get_column("n_target_units").min()) if n_sessions else 0
        reasons = []
        if n_mice < config.minimum_pair_mice:
            reasons.append("insufficient_mice")
        if n_sessions < config.minimum_pair_sessions:
            reasons.append("insufficient_sessions")
        if min(median_source, median_target) < config.minimum_median_units_per_mouse:
            reasons.append("insufficient_median_units_per_mouse")
        output.append(
            {
                **pair,
                "n_mice": n_mice,
                "n_sessions": n_sessions,
                "median_source_units_per_mouse": median_source,
                "median_target_units_per_mouse": median_target,
                "minimum_source_units_per_session": minimum_source,
                "minimum_target_units_per_session": minimum_target,
                "coverage_eligible": not reasons,
                "coverage_reason": "pass" if not reasons else ";".join(reasons),
            }
        )
    return pl.DataFrame(output).sort(
        pl.col("coverage_eligible").cast(pl.Int8), "n_mice", "n_sessions", descending=True
    )


def summarize_network_by_mouse(
    session_block_performance: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Average sessions within mouse and derive real/shift state contrasts."""

    if not isinstance(session_block_performance, pl.DataFrame):
        raise TypeError("session_block_performance must be a polars DataFrame")
    required = {
        "_nwb_path",
        "subject_id",
        "source_region",
        "target_region",
        "reward_block",
        "source_added_r2",
        "shift_source_added_r2",
        "real_minus_shift_source_added_r2",
    }
    _require_columns(session_block_performance, required, frame_name="session_block_performance")
    passed = session_block_performance
    if "model_status" in passed.columns:
        passed = passed.filter(pl.col("model_status") == "pass")
    duplicate = passed.select(
        "_nwb_path", "source_region", "target_region", "reward_block"
    ).is_duplicated()
    if duplicate.any():
        raise ValueError("session block performance contains duplicate pair/block rows")
    mouse_blocks = (
        passed.group_by("subject_id", "source_region", "target_region", "reward_block")
        .agg(
            pl.col("_nwb_path").n_unique().alias("n_sessions"),
            pl.col("source_added_r2").mean(),
            pl.col("shift_source_added_r2").mean(),
            pl.col("real_minus_shift_source_added_r2").mean(),
            pl.col("n_trials").sum(),
        )
        .sort("source_region", "target_region", "subject_id", "reward_block")
    )
    effect_rows: list[dict[str, Any]] = []
    for key, group in mouse_blocks.partition_by(
        "subject_id", "source_region", "target_region", as_dict=True
    ).items():
        lookup = {row["reward_block"]: row for row in group.iter_rows(named=True)}
        if set(lookup) != set(REWARD_BLOCKS):
            continue
        mouse, source_region, target_region = key
        real = {block: float(lookup[block]["source_added_r2"]) for block in REWARD_BLOCKS}
        shifted = {block: float(lookup[block]["shift_source_added_r2"]) for block in REWARD_BLOCKS}
        real_reversible = 0.5 * real[ENGAGED_1] - real[NO_REWARD] + 0.5 * real[ENGAGED_2]
        shift_reversible = 0.5 * shifted[ENGAGED_1] - shifted[NO_REWARD] + 0.5 * shifted[ENGAGED_2]
        effect_rows.append(
            {
                "subject_id": str(mouse),
                "source_region": source_region,
                "target_region": target_region,
                "n_sessions": min(int(lookup[block]["n_sessions"]) for block in REWARD_BLOCKS),
                "n_trials": sum(int(lookup[block]["n_trials"]) for block in REWARD_BLOCKS),
                "engaged_1_source_added_r2": real[ENGAGED_1],
                "no_reward_source_added_r2": real[NO_REWARD],
                "engaged_2_source_added_r2": real[ENGAGED_2],
                "real_reversible_contrast": real_reversible,
                "shift_reversible_contrast": shift_reversible,
                "controlled_reversible_contrast": real_reversible - shift_reversible,
                "real_hysteresis_contrast": real[ENGAGED_2] - real[ENGAGED_1],
            }
        )
    return mouse_blocks, pl.DataFrame(effect_rows).sort(
        "source_region", "target_region", "subject_id"
    )


def infer_and_nominate_network_pair(
    mouse_effects: pl.DataFrame,
    *,
    seed: int,
    n_bootstrap: int = 10_000,
    n_sign_flips: int = 100_000,
    config: NetworkConfig = DEFAULT_NETWORK_CONFIG,
) -> pl.DataFrame:
    """Infer every screened pair and nominate at most one discovery pair."""

    if not isinstance(mouse_effects, pl.DataFrame):
        raise TypeError("mouse_effects must be a polars DataFrame")
    _require_columns(
        mouse_effects,
        {
            "subject_id",
            "source_region",
            "target_region",
            "controlled_reversible_contrast",
            "n_sessions",
            "n_trials",
        },
        frame_name="mouse_effects",
    )
    rows: list[dict[str, Any]] = []
    groups = mouse_effects.partition_by("source_region", "target_region", as_dict=True)
    for pair_index, (pair, group) in enumerate(sorted(groups.items()), start=1):
        source_region, target_region = pair
        n_sessions = int(group.get_column("n_sessions").sum())
        if (
            group.get_column("subject_id").n_unique() < config.minimum_pair_mice
            or n_sessions < config.minimum_pair_sessions
        ):
            continue
        values = group.select(
            pl.col("subject_id").alias("mouse_id"),
            pl.col("controlled_reversible_contrast").alias("reversible_contrast"),
        )
        if values.height < 2:
            continue
        bootstrap = dg.statistics.bootstrap_mouse_mean(
            values,
            seed=seed + pair_index,
            n_resamples=n_bootstrap,
        )
        test = dg.statistics.two_sided_sign_flip_test(
            values,
            seed=seed + 10_000 + pair_index,
            n_resamples=n_sign_flips,
        )
        rows.append(
            {
                "source_region": source_region,
                "target_region": target_region,
                "estimate": bootstrap.estimate,
                "ci_low": bootstrap.ci_low,
                "ci_high": bootstrap.ci_high,
                "p_value": test.p_value,
                "test_method": test.method,
                "n_mice": group.get_column("subject_id").n_unique(),
                "n_sessions": n_sessions,
                "n_trials": int(group.get_column("n_trials").sum()),
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "source_region": pl.String,
                "target_region": pl.String,
                "estimate": pl.Float64,
                "ci_low": pl.Float64,
                "ci_high": pl.Float64,
                "p_value": pl.Float64,
                "adjusted_p_value": pl.Float64,
                "test_method": pl.String,
                "n_mice": pl.Int64,
                "n_sessions": pl.Int64,
                "n_trials": pl.Int64,
                "nominated": pl.Boolean,
                "status": pl.String,
                "reason": pl.String,
            }
        )
    adjusted = dg.statistics.holm_adjust([row["p_value"] for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["adjusted_p_value"] = value
        row["nominated"] = False
        row["status"] = "pass" if row["estimate"] > 0 and value < 0.05 else "null"
        row["reason"] = "discovery_screen_selection_biased"
    best = max(range(len(rows)), key=lambda index: (rows[index]["estimate"], -index))
    rows[best]["nominated"] = True
    if rows[best]["estimate"] > 0 and rows[best]["adjusted_p_value"] >= 0.05:
        rows[best]["status"] = "fragile"
        rows[best]["reason"] = "positive_discovery_nomination_not_familywise_significant"
    elif rows[best]["estimate"] <= 0:
        rows[best]["reason"] = "no_positive_controlled_pair_in_discovery_screen"
    return pl.DataFrame(rows).sort(pl.col("nominated").cast(pl.Int8), "estimate", descending=True)


def _select_base_alpha(
    base: np.ndarray,
    target: np.ndarray,
    times: np.ndarray,
    *,
    config: NetworkConfig,
) -> float:
    n_inner = min(config.inner_folds, base.shape[0])
    order = np.argsort(times, kind="stable")
    fold_ids = np.empty(base.shape[0], dtype=int)
    fold_ids[order] = _contiguous_fold_ids(base.shape[0], n_inner)
    candidates = []
    for alpha in config.ridge_alphas:
        squared_error = 0.0
        n_values = 0
        for fold_id in range(n_inner):
            validation = fold_ids == fold_id
            training = ~validation
            observed, predicted = _fit_predict_base(
                base[training],
                target[training],
                base[validation],
                target[validation],
                alpha=alpha,
            )
            squared_error += float(np.square(observed - predicted).sum())
            n_values += int(observed.size)
        candidates.append((alpha, squared_error / max(n_values, 1)))
    alpha, _ = min(candidates, key=lambda item: (item[1], -item[0]))
    return alpha


def _select_hyperparameters(
    base: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    times: np.ndarray,
    *,
    base_alpha: float,
    config: NetworkConfig,
) -> tuple[float, int]:
    n_inner = min(config.inner_folds, base.shape[0])
    if n_inner < 2:
        return config.ridge_alphas[-1], 1
    order = np.argsort(times, kind="stable")
    fold_ids = np.empty(base.shape[0], dtype=int)
    fold_ids[order] = _contiguous_fold_ids(base.shape[0], n_inner)
    candidates: list[tuple[float, int, float]] = []
    maximum_rank = min(source.shape[1], target.shape[1], max(1, base.shape[0] - 1))
    ranks = [rank for rank in config.ranks if rank <= maximum_rank] or [1]
    for alpha in config.ridge_alphas:
        for rank in ranks:
            squared_error = 0.0
            n_values = 0
            for fold_id in range(n_inner):
                validation = fold_ids == fold_id
                training = ~validation
                prediction = _fit_predict_residual_rrr(
                    base[training],
                    source[training],
                    target[training],
                    base[validation],
                    source[validation],
                    target[validation],
                    base_alpha=base_alpha,
                    source_alpha=alpha,
                    rank=rank,
                )
                squared_error += float(
                    np.square(prediction["observed"] - prediction["full_prediction"]).sum()
                )
                n_values += int(prediction["observed"].size)
            candidates.append((alpha, rank, squared_error / max(n_values, 1)))
    alpha, rank, _ = min(candidates, key=lambda item: (item[2], item[1], -item[0]))
    return alpha, rank


def _fit_predict_residual_rrr(
    base_train: np.ndarray,
    source_train: np.ndarray,
    target_train: np.ndarray,
    base_validation: np.ndarray,
    source_validation: np.ndarray,
    target_validation: np.ndarray,
    *,
    base_alpha: float,
    source_alpha: float,
    rank: int,
) -> dict[str, np.ndarray]:
    base_train_z, base_validation_z = _standardize_train_validation(base_train, base_validation)
    source_train_z, source_validation_z = _standardize_train_validation(
        source_train, source_validation
    )
    target_train_z, target_validation_z = _standardize_train_validation(
        target_train, target_validation
    )

    base_to_target = _ridge_coefficients(base_train_z, target_train_z, base_alpha)
    base_to_source = _ridge_coefficients(base_train_z, source_train_z, base_alpha)
    target_residual = target_train_z - base_train_z @ base_to_target
    source_residual = source_train_z - base_train_z @ base_to_source
    source_to_target = _ridge_coefficients(source_residual, target_residual, source_alpha)
    fitted_increment = source_residual @ source_to_target
    _, _, right = np.linalg.svd(fitted_increment, full_matrices=False)
    retained_rank = min(rank, right.shape[0])
    projection = right[:retained_rank].T @ right[:retained_rank]
    reduced = source_to_target @ projection
    validation_source_residual = source_validation_z - base_validation_z @ base_to_source
    base_prediction = base_validation_z @ base_to_target
    full_prediction = base_prediction + validation_source_residual @ reduced
    return {
        "observed": target_validation_z,
        "base_prediction": base_prediction,
        "full_prediction": full_prediction,
    }


def _fit_predict_base(
    base_train: np.ndarray,
    target_train: np.ndarray,
    base_validation: np.ndarray,
    target_validation: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    base_train_z, base_validation_z = _standardize_train_validation(base_train, base_validation)
    target_train_z, target_validation_z = _standardize_train_validation(
        target_train, target_validation
    )
    coefficients = _ridge_coefficients(base_train_z, target_train_z, alpha)
    return target_validation_z, base_validation_z @ coefficients


def _standardize_train_validation(
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (train - mean) / scale, (validation - mean) / scale


def _ridge_coefficients(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    gram = features.T @ features
    penalty = np.eye(features.shape[1], dtype=float) * alpha
    return np.linalg.solve(gram + penalty, features.T @ target)


def _r2(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(np.square(observed - predicted).sum())
    denominator = float(np.square(observed).sum())
    if denominator <= np.finfo(float).eps:
        return 0.0 if residual <= np.finfo(float).eps else -residual / np.finfo(float).eps
    return 1.0 - residual / denominator


def _balanced_trial_indices(blocks: np.ndarray, times: np.ndarray, minimum: int) -> np.ndarray:
    indices_by_block = {
        block: np.flatnonzero(blocks == block)[np.argsort(times[blocks == block], kind="stable")]
        for block in REWARD_BLOCKS
    }
    counts = {block: values.size for block, values in indices_by_block.items()}
    n_balanced = min(counts.values())
    if n_balanced < minimum:
        raise ValueError(
            f"network model requires at least {minimum} trials in every block; received {counts}"
        )
    selected = []
    for block in REWARD_BLOCKS:
        values = indices_by_block[block]
        positions = np.floor((np.arange(n_balanced) + 0.5) * values.size / n_balanced).astype(int)
        selected.extend(values[positions].tolist())
    return np.asarray(selected, dtype=int)


def _contiguous_fold_ids(n_rows: int, n_folds: int) -> np.ndarray:
    fold_ids = np.empty(n_rows, dtype=int)
    for fold_id, indices in enumerate(np.array_split(np.arange(n_rows), n_folds)):
        fold_ids[indices] = fold_id
    return fold_ids


def _window_counts(spikes: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    return np.searchsorted(spikes, stops, side="left") - np.searchsorted(
        spikes, starts, side="left"
    )


def _as_finite_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a two-dimensional matrix with at least one column")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    return matrix


def _require_columns(frame: pl.DataFrame, required: set[str], *, frame_name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} missing columns: {sorted(missing)}")
