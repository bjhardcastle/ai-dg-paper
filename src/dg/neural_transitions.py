"""Discovery-only neural reward-state transition analysis.

This module contains computation on already-selected trials and spike times;
it deliberately has no NWB I/O.  The reward-state axis is trained only on
engaged-1 (E1) and no-reward (NR) trials.  E1/NR scores are out-of-fold, while
engaged-2 (E2) scores are a held-out transfer test from a final E1-versus-NR
fit.  Folds are contiguous within each reward state so nearby trials are not
randomly interleaved between training and validation sets.

The transition summaries retain real and within-source-state pseudo
boundaries separately.  A session contributes to a six-minute step only
when both causal sides contain the declared minimum number of trials.  Mouse
summaries first average sessions, then give mice equal weight.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import math
from typing import Any

import numpy as np
import polars as pl

import dg.statistics

SOURCE_COLUMN = "_nwb_path"
MOUSE_COLUMN = "subject_id"

ENGAGED_1 = "engaged_1"
NO_REWARD = "no_reward"
ENGAGED_2 = "engaged_2"
REWARD_BLOCKS = (ENGAGED_1, NO_REWARD, ENGAGED_2)
TRANSITION_IDS = ("reward_withdrawal", "reward_restoration")


@dataclasses.dataclass(frozen=True, slots=True)
class NeuralTransitionConfig:
    """Fixed windows, support rules, and ridge grid for the neural analysis."""

    spike_window_seconds: float = 0.150
    lick_exclusion_seconds: float = 0.150
    effect_window_seconds: float = 360.0
    trajectory_window_seconds: float = 360.0
    trajectory_bin_width_seconds: float = 120.0
    minimum_trials_per_effect_side: int = 5
    minimum_axis_trials_per_state: int = 6
    outer_folds: int = 5
    inner_folds: int = 3
    ridge_lambdas: tuple[float, ...] = (
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
        1000.0,
    )

    def __post_init__(self) -> None:
        for name in (
            "spike_window_seconds",
            "lick_exclusion_seconds",
            "effect_window_seconds",
            "trajectory_window_seconds",
            "trajectory_bin_width_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "minimum_trials_per_effect_side",
            "minimum_axis_trials_per_state",
            "outer_folds",
            "inner_folds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least two")
        n_bins = self.trajectory_window_seconds / self.trajectory_bin_width_seconds
        if not n_bins.is_integer():
            raise ValueError(
                "trajectory_window_seconds must be divisible by trajectory_bin_width_seconds"
            )
        if self.effect_window_seconds > self.trajectory_window_seconds:
            raise ValueError("effect_window_seconds cannot exceed trajectory_window_seconds")
        if not self.ridge_lambdas:
            raise ValueError("ridge_lambdas must not be empty")
        if any(not math.isfinite(value) or value <= 0 for value in self.ridge_lambdas):
            raise ValueError("ridge_lambdas must contain finite positive values")
        if tuple(sorted(set(self.ridge_lambdas))) != self.ridge_lambdas:
            raise ValueError("ridge_lambdas must be unique and strictly increasing")


DEFAULT_NEURAL_TRANSITION_CONFIG = NeuralTransitionConfig()


@dataclasses.dataclass(frozen=True, slots=True)
class RidgeAxisResult:
    """Scores and final E1-versus-NR linear model for one session.

    ``scores`` follows the input row order. E1 and NR values are out-of-fold;
    E2 values use the final model fit to all E1 and NR rows. ``feature_weights``
    and ``intercept`` map the original, unstandardized feature matrix to the
    final calibrated score.
    """

    scores: np.ndarray
    score_roles: tuple[str, ...]
    feature_weights: np.ndarray
    intercept: float
    selected_lambda: float
    outer_selected_lambdas: tuple[float, ...]
    oof_auc: float
    n_engaged_1: int
    n_no_reward: int
    n_engaged_2: int


@dataclasses.dataclass(frozen=True, slots=True)
class TransitionScoreTables:
    """Session- and mouse-level transition summaries."""

    session_trajectories: pl.DataFrame
    mouse_trajectories: pl.DataFrame
    session_anchor_effects: pl.DataFrame
    mouse_anchor_effects: pl.DataFrame
    session_contrasts: pl.DataFrame
    mouse_contrasts: pl.DataFrame


def select_lick_free_familiar_trials(
    trials: pl.DataFrame | pl.LazyFrame,
    *,
    config: NeuralTransitionConfig = DEFAULT_NEURAL_TRANSITION_CONFIG,
) -> pl.DataFrame:
    """Select completed, lick-free, familiar full-contrast image changes.

    Licks exactly on either boundary of ``change_time +/- lick_exclusion`` are
    excluded. Null or non-finite lick arrays are treated as invalid rather than
    as evidence for an absence of licking.
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

    lick_free = [
        _lick_array_is_free(
            row["lick_times"],
            event_time=float(row["change_time"]),
            half_width=config.lick_exclusion_seconds,
        )
        for row in candidate.select("lick_times", "change_time").iter_rows(named=True)
    ]
    return (
        candidate.with_columns(
            pl.Series("lick_free_for_neural_analysis", lick_free, dtype=pl.Boolean),
            pl.lit(config.lick_exclusion_seconds).alias("lick_exclusion_half_width_seconds"),
        )
        .filter(pl.col("lick_free_for_neural_analysis"))
        .sort(
            *(
                column
                for column in (SOURCE_COLUMN, "change_time", "_table_index")
                if column in candidate.columns
            )
        )
    )


def build_spike_feature_matrix(
    spike_times_by_unit: collections.abc.Sequence[collections.abc.Sequence[float] | np.ndarray],
    event_times: collections.abc.Sequence[float] | np.ndarray,
    *,
    window_seconds: float = DEFAULT_NEURAL_TRANSITION_CONFIG.spike_window_seconds,
) -> np.ndarray:
    """Return trials x units sqrt-count changes around each event.

    For each unit and event, the feature is
    ``sqrt(n[0, window) + 3/8) - sqrt(n[-window, 0) + 3/8)``. Half-open
    windows make spikes exactly at the event causal post-event observations
    and exclude spikes exactly at the positive endpoint.
    """

    if not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive")
    events = np.asarray(event_times, dtype=float)
    if events.ndim != 1 or not np.isfinite(events).all():
        raise ValueError("event_times must be a one-dimensional finite array")
    units = list(spike_times_by_unit)
    features = np.empty((events.size, len(units)), dtype=float)
    for unit_index, values in enumerate(units):
        spikes = np.asarray(values, dtype=float)
        if spikes.ndim != 1 or not np.isfinite(spikes).all():
            raise ValueError(f"unit {unit_index} spike times must be one-dimensional and finite")
        if spikes.size > 1 and np.any(np.diff(spikes) < 0):
            raise ValueError(f"unit {unit_index} spike times must be sorted")
        pre_count = np.searchsorted(spikes, events, side="left") - np.searchsorted(
            spikes, events - window_seconds, side="left"
        )
        post_count = np.searchsorted(
            spikes, events + window_seconds, side="left"
        ) - np.searchsorted(spikes, events, side="left")
        features[:, unit_index] = np.sqrt(post_count + 3.0 / 8.0) - np.sqrt(pre_count + 3.0 / 8.0)
    return features


def fit_reward_state_axis(
    features: np.ndarray,
    reward_blocks: collections.abc.Sequence[str] | np.ndarray,
    event_times: collections.abc.Sequence[float] | np.ndarray,
    *,
    config: NeuralTransitionConfig = DEFAULT_NEURAL_TRANSITION_CONFIG,
) -> RidgeAxisResult:
    """Fit a nested-CV E1-versus-NR ridge axis and score held-out E2 trials.

    Hyperparameter selection, standardization, calibration, and all fitted
    coefficients use E1 and NR rows only. E2 labels and features are accessed
    only once the final E1/NR model has been frozen.
    """

    matrix = np.asarray(features, dtype=float)
    blocks = np.asarray(reward_blocks, dtype=object)
    times = np.asarray(event_times, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] < 1:
        raise ValueError("features must be a two-dimensional array with at least one unit")
    if blocks.ndim != 1 or times.ndim != 1:
        raise ValueError("reward_blocks and event_times must be one-dimensional")
    if matrix.shape[0] != blocks.size or blocks.size != times.size:
        raise ValueError("features, reward_blocks, and event_times must have equal row counts")
    if not np.isfinite(matrix).all() or not np.isfinite(times).all():
        raise ValueError("features and event_times must be finite")
    unknown = set(blocks.tolist()).difference(REWARD_BLOCKS)
    if unknown:
        raise ValueError(f"reward_blocks contains unknown values: {sorted(unknown)}")

    e1_indices = np.flatnonzero(blocks == ENGAGED_1)
    nr_indices = np.flatnonzero(blocks == NO_REWARD)
    e2_indices = np.flatnonzero(blocks == ENGAGED_2)
    minimum = config.minimum_axis_trials_per_state
    if e1_indices.size < minimum or nr_indices.size < minimum:
        raise ValueError(
            f"axis requires at least {minimum} E1 and {minimum} NR trials; "
            f"received {e1_indices.size} and {nr_indices.size}"
        )

    fit_indices = np.concatenate([e1_indices, nr_indices])
    fit_targets = np.concatenate(
        [np.ones(e1_indices.size, dtype=float), -np.ones(nr_indices.size, dtype=float)]
    )
    fit_times = times[fit_indices]
    n_outer = min(config.outer_folds, e1_indices.size, nr_indices.size)
    fold_ids = _contiguous_state_fold_ids(fit_targets, fit_times, n_folds=n_outer)
    scores = np.full(blocks.size, np.nan, dtype=float)
    outer_lambdas: list[float] = []

    for fold_id in range(n_outer):
        validation = fold_ids == fold_id
        training = ~validation
        ridge_lambda = _select_ridge_lambda(
            matrix[fit_indices][training],
            fit_targets[training],
            fit_times[training],
            config=config,
        )
        outer_lambdas.append(ridge_lambda)
        model = _fit_calibrated_ridge(
            matrix[fit_indices][training], fit_targets[training], ridge_lambda
        )
        scores[fit_indices[validation]] = _apply_calibrated_ridge(
            matrix[fit_indices][validation], model
        )

    if not np.isfinite(scores[fit_indices]).all():
        raise RuntimeError("out-of-fold scoring did not cover every E1/NR trial")

    selected_lambda = _select_ridge_lambda(
        matrix[fit_indices],
        fit_targets,
        fit_times,
        config=config,
    )
    final_model = _fit_calibrated_ridge(matrix[fit_indices], fit_targets, selected_lambda)
    if e2_indices.size:
        scores[e2_indices] = _apply_calibrated_ridge(matrix[e2_indices], final_model)

    feature_weights, intercept = _original_space_model(final_model)
    score_roles = tuple(
        "held_out_e2_transfer" if block == ENGAGED_2 else "e1_nr_out_of_fold" for block in blocks
    )
    return RidgeAxisResult(
        scores=scores,
        score_roles=score_roles,
        feature_weights=feature_weights,
        intercept=intercept,
        selected_lambda=selected_lambda,
        outer_selected_lambdas=tuple(outer_lambdas),
        oof_auc=_binary_auc(scores[fit_indices], fit_targets),
        n_engaged_1=int(e1_indices.size),
        n_no_reward=int(nr_indices.size),
        n_engaged_2=int(e2_indices.size),
    )


def summarize_transition_scores(
    trial_scores: pl.DataFrame | pl.LazyFrame,
    boundaries: pl.DataFrame | pl.LazyFrame,
    *,
    config: NeuralTransitionConfig = DEFAULT_NEURAL_TRANSITION_CONFIG,
    source_column: str = SOURCE_COLUMN,
    mouse_column: str = MOUSE_COLUMN,
    score_column: str = "axis_score",
) -> TransitionScoreTables:
    """Summarize neural-axis trajectories and real-minus-pseudo steps."""

    scores = trial_scores.collect() if isinstance(trial_scores, pl.LazyFrame) else trial_scores
    bounds = boundaries.collect() if isinstance(boundaries, pl.LazyFrame) else boundaries
    if not isinstance(scores, pl.DataFrame) or not isinstance(bounds, pl.DataFrame):
        raise TypeError("trial_scores and boundaries must be polars frames")
    _require_columns(
        scores,
        {source_column, mouse_column, "change_time", "reward_block", score_column},
        frame_name="trial_scores",
    )
    _require_columns(
        bounds,
        {
            source_column,
            mouse_column,
            "transition_id",
            "pre_block",
            "post_block",
            "pseudo_block",
            "effect_direction_multiplier",
            "real_boundary_time",
            "pseudo_boundary_time",
        },
        frame_name="boundaries",
    )
    duplicate_bounds = (
        bounds.group_by(source_column, "transition_id").len().filter(pl.col("len") != 1)
    )
    if duplicate_bounds.height:
        raise ValueError("boundaries must contain one row per session and transition")
    unknown_transitions = set(bounds.get_column("transition_id").drop_nulls()).difference(
        TRANSITION_IDS
    )
    if unknown_transitions:
        raise ValueError(f"boundaries contains unknown transitions: {sorted(unknown_transitions)}")

    valid_scores = scores.filter(
        pl.col("change_time").is_not_null()
        & pl.col("change_time").is_finite()
        & pl.col(score_column).is_not_null()
        & pl.col(score_column).is_finite()
        & pl.col("reward_block").is_in(REWARD_BLOCKS)
    )
    by_source = {
        key[0] if isinstance(key, tuple) else key: value
        for key, value in valid_scores.partition_by(source_column, as_dict=True).items()
    }
    trajectory_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    n_each_side = int(config.trajectory_window_seconds / config.trajectory_bin_width_seconds)

    for boundary in bounds.sort(mouse_column, source_column, "transition_id").iter_rows(named=True):
        source = boundary[source_column]
        mouse = boundary[mouse_column]
        session = by_source.get(source)
        included = bool(boundary.get("included_in_transition_analysis", True))
        session_times = (
            np.asarray(session.get_column("change_time"), dtype=float)
            if session is not None
            else np.empty(0, dtype=float)
        )
        session_blocks = (
            np.asarray(session.get_column("reward_block"), dtype=object)
            if session is not None
            else np.empty(0, dtype=object)
        )
        session_scores = (
            np.asarray(session.get_column(score_column), dtype=float)
            if session is not None
            else np.empty(0, dtype=float)
        )
        anchor_effects: dict[str, dict[str, Any]] = {}

        for anchor_type in ("real", "pseudo"):
            anchor_time = boundary[f"{anchor_type}_boundary_time"]
            anchor_valid = (
                included and anchor_time is not None and math.isfinite(float(anchor_time))
            )
            anchor = float(anchor_time) if anchor_valid else math.nan
            if anchor_type == "real":
                pre_block = boundary["pre_block"]
                post_block = boundary["post_block"]
            else:
                pre_block = boundary["pseudo_block"]
                post_block = boundary["pseudo_block"]
            relative = (
                session_times - anchor if anchor_valid else np.full(session_times.size, np.nan)
            )

            for bin_index in range(-n_each_side, n_each_side):
                bin_start = bin_index * config.trajectory_bin_width_seconds
                bin_stop = (bin_index + 1) * config.trajectory_bin_width_seconds
                expected_block = pre_block if bin_index < 0 else post_block
                mask = (
                    (relative >= bin_start)
                    & (relative < bin_stop)
                    & (session_blocks == expected_block)
                )
                values = session_scores[mask]
                trajectory_rows.append(
                    {
                        source_column: source,
                        mouse_column: mouse,
                        "transition_id": boundary["transition_id"],
                        "anchor_type": anchor_type,
                        "bin_index": bin_index,
                        "relative_bin_start_seconds": float(bin_start),
                        "relative_bin_stop_seconds": float(bin_stop),
                        "relative_bin_center_seconds": float((bin_start + bin_stop) / 2),
                        "expected_reward_block": expected_block,
                        "n_trials": int(values.size),
                        "mean_axis_score": float(values.mean()) if values.size else None,
                        "status": "pass"
                        if values.size
                        else ("no_trials" if anchor_valid else "invalid_boundary"),
                    }
                )

            effect_window = config.effect_window_seconds
            pre_mask = (relative >= -effect_window) & (relative < 0) & (session_blocks == pre_block)
            post_mask = (
                (relative >= 0) & (relative < effect_window) & (session_blocks == post_block)
            )
            pre_values = session_scores[pre_mask]
            post_values = session_scores[post_mask]
            enough_support = (
                anchor_valid
                and pre_values.size >= config.minimum_trials_per_effect_side
                and post_values.size >= config.minimum_trials_per_effect_side
            )
            pre_mean = float(pre_values.mean()) if pre_values.size else None
            post_mean = float(post_values.mean()) if post_values.size else None
            signed_step = (
                float(boundary["effect_direction_multiplier"]) * (post_mean - pre_mean)
                if enough_support and pre_mean is not None and post_mean is not None
                else None
            )
            effect = {
                source_column: source,
                mouse_column: mouse,
                "transition_id": boundary["transition_id"],
                "anchor_type": anchor_type,
                "anchor_time": float(anchor) if anchor_valid else None,
                "pre_reward_block": pre_block,
                "post_reward_block": post_block,
                "n_pre_trials": int(pre_values.size),
                "n_post_trials": int(post_values.size),
                "pre_mean_axis_score": pre_mean,
                "post_mean_axis_score": post_mean,
                "effect_direction_multiplier": float(boundary["effect_direction_multiplier"]),
                "signed_step": signed_step,
                "status": (
                    "pass"
                    if enough_support
                    else ("insufficient_trials" if anchor_valid else "invalid_boundary")
                ),
            }
            effect_rows.append(effect)
            anchor_effects[anchor_type] = effect

        real_effect = anchor_effects["real"]
        pseudo_effect = anchor_effects["pseudo"]
        contrast_valid = real_effect["status"] == "pass" and pseudo_effect["status"] == "pass"
        contrast_rows.append(
            {
                source_column: source,
                mouse_column: mouse,
                "transition_id": boundary["transition_id"],
                "real_signed_step": real_effect["signed_step"],
                "pseudo_signed_step": pseudo_effect["signed_step"],
                "real_minus_pseudo": (
                    float(real_effect["signed_step"] - pseudo_effect["signed_step"])
                    if contrast_valid
                    else None
                ),
                "n_real_pre_trials": real_effect["n_pre_trials"],
                "n_real_post_trials": real_effect["n_post_trials"],
                "n_pseudo_pre_trials": pseudo_effect["n_pre_trials"],
                "n_pseudo_post_trials": pseudo_effect["n_post_trials"],
                "status": "pass" if contrast_valid else "insufficient_anchor_support",
            }
        )

    session_trajectories = pl.DataFrame(trajectory_rows)
    session_anchor_effects = pl.DataFrame(effect_rows)
    session_contrasts = pl.DataFrame(contrast_rows)
    mouse_trajectories = (
        session_trajectories.filter(pl.col("status") == "pass")
        .group_by(
            mouse_column,
            "transition_id",
            "anchor_type",
            "bin_index",
            "relative_bin_start_seconds",
            "relative_bin_stop_seconds",
            "relative_bin_center_seconds",
        )
        .agg(
            pl.len().alias("n_sessions"),
            pl.col("n_trials").sum().alias("n_trials"),
            pl.col("mean_axis_score").mean().alias("mean_axis_score"),
        )
        .sort(mouse_column, "transition_id", "anchor_type", "bin_index")
    )
    mouse_anchor_effects = (
        session_anchor_effects.filter(pl.col("status") == "pass")
        .group_by(mouse_column, "transition_id", "anchor_type")
        .agg(
            pl.len().alias("n_sessions"),
            pl.col("n_pre_trials").sum().alias("n_pre_trials"),
            pl.col("n_post_trials").sum().alias("n_post_trials"),
            pl.col("signed_step").mean().alias("signed_step"),
        )
        .sort(mouse_column, "transition_id", "anchor_type")
    )
    mouse_contrasts = (
        session_contrasts.filter(pl.col("status") == "pass")
        .group_by(mouse_column, "transition_id")
        .agg(
            pl.len().alias("n_sessions"),
            pl.col("real_minus_pseudo").mean().alias("real_minus_pseudo"),
        )
        .with_columns(pl.lit("pass").alias("status"))
        .sort(mouse_column, "transition_id")
    )
    return TransitionScoreTables(
        session_trajectories=session_trajectories,
        mouse_trajectories=mouse_trajectories,
        session_anchor_effects=session_anchor_effects,
        mouse_anchor_effects=mouse_anchor_effects,
        session_contrasts=session_contrasts,
        mouse_contrasts=mouse_contrasts,
    )


def infer_mouse_transition_contrasts(
    mouse_contrasts: pl.DataFrame,
    *,
    seed: int,
    n_bootstrap: int = 10_000,
    n_sign_flips: int = 100_000,
    mouse_column: str = MOUSE_COLUMN,
) -> pl.DataFrame:
    """Return mouse-bootstrap intervals and Holm-adjusted sign-flip tests."""

    _require_columns(
        mouse_contrasts,
        {mouse_column, "transition_id", "real_minus_pseudo"},
        frame_name="mouse_contrasts",
    )
    rows: list[dict[str, Any]] = []
    for index, transition_id in enumerate(TRANSITION_IDS):
        values = mouse_contrasts.filter(
            (pl.col("transition_id") == transition_id)
            & pl.col("real_minus_pseudo").is_not_null()
            & pl.col("real_minus_pseudo").is_finite()
        ).select(mouse_column, "real_minus_pseudo")
        if values.height == 0:
            rows.append(
                {
                    "transition_id": transition_id,
                    "estimate": None,
                    "ci_low": None,
                    "ci_high": None,
                    "p_value": None,
                    "adjusted_p_value": None,
                    "n_mice": 0,
                    "bootstrap_seed": seed + index,
                    "test_method": None,
                    "status": "not_estimable",
                }
            )
            continue
        bootstrap = dg.statistics.bootstrap_mouse_mean(
            values,
            seed=seed + index,
            mouse_column=mouse_column,
            value_column="real_minus_pseudo",
            n_resamples=n_bootstrap,
        )
        test = dg.statistics.two_sided_sign_flip_test(
            values,
            seed=seed + index,
            mouse_column=mouse_column,
            value_column="real_minus_pseudo",
            n_resamples=n_sign_flips,
        )
        rows.append(
            {
                "transition_id": transition_id,
                "estimate": bootstrap.estimate,
                "ci_low": bootstrap.ci_low,
                "ci_high": bootstrap.ci_high,
                "p_value": test.p_value,
                "adjusted_p_value": None,
                "n_mice": bootstrap.n_mice,
                "bootstrap_seed": seed + index,
                "test_method": test.method,
                "status": "pass",
            }
        )
    adjusted = _holm_adjust([row["p_value"] for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["adjusted_p_value"] = value
    return pl.DataFrame(rows)


@dataclasses.dataclass(frozen=True, slots=True)
class _CalibratedRidge:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    calibration_midpoint: float
    calibration_difference: float


def _fit_calibrated_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    ridge_lambda: float,
) -> _CalibratedRidge:
    mean, scale = _balanced_location_scale(features, targets)
    standardized = (features - mean) / scale
    weights = _balanced_sample_weights(targets)
    weighted_features = standardized * np.sqrt(weights)[:, np.newaxis]
    weighted_targets = targets * np.sqrt(weights)
    gram = weighted_features @ weighted_features.T
    dual = np.linalg.solve(
        gram + ridge_lambda * np.eye(gram.shape[0], dtype=float),
        weighted_targets,
    )
    coefficients = weighted_features.T @ dual
    raw = standardized @ coefficients
    positive_mean = float(raw[targets > 0].mean())
    negative_mean = float(raw[targets < 0].mean())
    difference = positive_mean - negative_mean
    tolerance = np.finfo(float).eps * max(1.0, abs(positive_mean), abs(negative_mean)) * 32
    if not math.isfinite(difference) or difference <= tolerance:
        raise ValueError("reward-state axis is degenerate for these features")
    return _CalibratedRidge(
        mean=mean,
        scale=scale,
        coefficients=coefficients,
        calibration_midpoint=(positive_mean + negative_mean) / 2,
        calibration_difference=difference,
    )


def _apply_calibrated_ridge(features: np.ndarray, model: _CalibratedRidge) -> np.ndarray:
    raw = ((features - model.mean) / model.scale) @ model.coefficients
    return (raw - model.calibration_midpoint) / model.calibration_difference


def _original_space_model(model: _CalibratedRidge) -> tuple[np.ndarray, float]:
    weights = model.coefficients / model.scale / model.calibration_difference
    intercept = (
        -float((model.mean / model.scale) @ model.coefficients) - model.calibration_midpoint
    ) / model.calibration_difference
    return weights, intercept


def _balanced_location_scale(
    features: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    positive = features[targets > 0]
    negative = features[targets < 0]
    mean = 0.5 * positive.mean(axis=0) + 0.5 * negative.mean(axis=0)
    second_moment = 0.5 * np.square(positive - mean).mean(axis=0) + 0.5 * np.square(
        negative - mean
    ).mean(axis=0)
    scale = np.sqrt(second_moment)
    scale[~np.isfinite(scale) | (scale <= np.finfo(float).eps)] = 1.0
    return mean, scale


def _balanced_sample_weights(targets: np.ndarray) -> np.ndarray:
    n_positive = int(np.count_nonzero(targets > 0))
    n_negative = int(np.count_nonzero(targets < 0))
    if not n_positive or not n_negative:
        raise ValueError("ridge training requires both reward states")
    return np.where(targets > 0, 0.5 / n_positive, 0.5 / n_negative)


def _select_ridge_lambda(
    features: np.ndarray,
    targets: np.ndarray,
    times: np.ndarray,
    *,
    config: NeuralTransitionConfig,
) -> float:
    n_positive = int(np.count_nonzero(targets > 0))
    n_negative = int(np.count_nonzero(targets < 0))
    n_folds = min(config.inner_folds, n_positive, n_negative)
    if n_folds < 2:
        raise ValueError("ridge tuning requires at least two trials per reward state")
    folds = _contiguous_state_fold_ids(targets, times, n_folds=n_folds)
    losses: list[float] = []
    for ridge_lambda in config.ridge_lambdas:
        fold_losses: list[float] = []
        for fold_id in range(n_folds):
            validation = folds == fold_id
            training = ~validation
            model = _fit_calibrated_ridge(features[training], targets[training], ridge_lambda)
            predictions = _apply_calibrated_ridge(features[validation], model)
            fold_losses.append(_balanced_mean_squared_error(predictions, targets[validation]))
        losses.append(float(np.mean(fold_losses)))
    return float(config.ridge_lambdas[int(np.argmin(np.asarray(losses)))])


def _contiguous_state_fold_ids(
    targets: np.ndarray,
    times: np.ndarray,
    *,
    n_folds: int,
) -> np.ndarray:
    folds = np.full(targets.size, -1, dtype=int)
    for sign in (1, -1):
        state_indices = np.flatnonzero(targets == sign)
        ordered = state_indices[np.argsort(times[state_indices], kind="stable")]
        for fold_id, chunk in enumerate(np.array_split(ordered, n_folds)):
            folds[chunk] = fold_id
    if np.any(folds < 0):
        raise ValueError("fold construction received targets other than -1 and 1")
    return folds


def _balanced_mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    positive_loss = np.square(predictions[targets > 0] - 0.5).mean()
    negative_loss = np.square(predictions[targets < 0] + 0.5).mean()
    return float(0.5 * positive_loss + 0.5 * negative_loss)


def _binary_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    positive = scores[targets > 0]
    negative = scores[targets < 0]
    comparisons = positive[:, np.newaxis] - negative[np.newaxis, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def _lick_array_is_free(
    lick_times: collections.abc.Sequence[float] | np.ndarray | None,
    *,
    event_time: float,
    half_width: float,
) -> bool:
    if lick_times is None:
        return False
    values = np.asarray(lick_times, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        return False
    tolerance = np.finfo(float).eps * max(1.0, abs(event_time), half_width) * 8
    return not bool(np.any(np.abs(values - event_time) <= half_width + tolerance))


def _holm_adjust(p_values: collections.abc.Sequence[float | None]) -> list[float | None]:
    adjusted: list[float | None] = [None] * len(p_values)
    present = [(index, float(value)) for index, value in enumerate(p_values) if value is not None]
    ordered = sorted(present, key=lambda item: item[1])
    running = 0.0
    total = len(ordered)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted


def _require_columns(
    frame: pl.DataFrame,
    required: collections.abc.Iterable[str],
    *,
    frame_name: str,
) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {sorted(missing)}")
