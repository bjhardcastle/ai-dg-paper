"""Validation and source-data summaries for functional-population Figure 3."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import polars as pl

import dg.functional_populations
import dg.statistics

ANALYSIS_ID = "functional_populations"
RESPONSE_GROUPS = (
    ("early_sensory", 1, "Early sensory"),
    ("late_action", 2, "Late / action"),
)
BLOCKS = (
    ("engaged_1", 1, "E1"),
    ("no_reward", 2, "NR"),
    ("engaged_2", 3, "E2"),
)
PRIMARY_STATISTIC_IDS = (
    "late_minus_early_model_free",
    "late_minus_early_model_total",
    "late_minus_early_model_adjusted",
    "late_state_model_total",
    "late_state_model_adjusted",
)
BOOTSTRAP_SEED = 3351
BOOTSTRAP_RESAMPLES = 10_000


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionalPopulationFigureData:
    """Publication-facing Figure 3 tables."""

    mouse_psths: pl.DataFrame
    psth_summary: pl.DataFrame
    mouse_effects: pl.DataFrame
    statistics: pl.DataFrame
    class_stability: pl.DataFrame


def prepare_functional_population_figure_data(
    mouse_psths: pl.DataFrame,
    mouse_effects: pl.DataFrame,
    statistics: pl.DataFrame,
    class_stability: pl.DataFrame,
) -> FunctionalPopulationFigureData:
    """Validate Figure 3 inputs and summarize PSTHs with equal mouse weighting."""

    prepared_psths = _prepare_mouse_psths(mouse_psths)
    prepared_effects = _prepare_mouse_effects(mouse_effects)
    prepared_statistics = _prepare_statistics(statistics, prepared_effects)
    prepared_stability = _prepare_class_stability(class_stability)
    return FunctionalPopulationFigureData(
        mouse_psths=prepared_psths,
        psth_summary=_summarize_psths(prepared_psths),
        mouse_effects=prepared_effects,
        statistics=prepared_statistics,
        class_stability=prepared_stability,
    )


def _prepare_mouse_psths(frame: pl.DataFrame) -> pl.DataFrame:
    _require_columns(
        frame,
        {
            "subject_id",
            "aligned_event",
            "response_group",
            "reward_block",
            "bin_center_seconds",
            "mean_sign_aligned_rate_hz",
            "n_sessions",
            "n_unit_session_contributions",
        },
        frame_name="mouse_psths",
    )
    selected = frame.filter(
        (pl.col("aligned_event") == "change")
        & pl.col("response_group").is_in([row[0] for row in RESPONSE_GROUPS])
    ).select(
        pl.col("subject_id").cast(pl.String),
        pl.col("response_group").cast(pl.String),
        pl.col("reward_block").cast(pl.String),
        pl.col("bin_center_seconds").cast(pl.Float64),
        pl.col("mean_sign_aligned_rate_hz").cast(pl.Float64),
        pl.col("n_sessions").cast(pl.Int64),
        pl.col("n_unit_session_contributions").cast(pl.Int64),
    )
    if selected.is_empty():
        raise ValueError("mouse_psths has no change-aligned stable-class rows")
    if selected.filter(
        ~pl.col("response_group").is_in([row[0] for row in RESPONSE_GROUPS])
        | ~pl.col("reward_block").is_in([row[0] for row in BLOCKS])
    ).height:
        raise ValueError("mouse_psths contains unknown group or reward block")
    if selected.select(
        pl.struct("subject_id", "response_group", "reward_block", "bin_center_seconds")
        .is_duplicated()
        .any()
    ).item():
        raise ValueError("mouse_psths has duplicate mouse/group/block/bin rows")
    if selected.filter(
        ~pl.col("mean_sign_aligned_rate_hz").is_finite()
        | (pl.col("n_sessions") <= 0)
        | (pl.col("n_unit_session_contributions") <= 0)
    ).height:
        raise ValueError("mouse_psths contains invalid estimates or denominators")
    expected_centers = _expected_change_bin_centers()
    observed_centers = np.sort(selected.get_column("bin_center_seconds").unique().to_numpy())
    if observed_centers.shape != expected_centers.shape or not np.allclose(
        observed_centers, expected_centers, atol=1e-10
    ):
        raise ValueError("mouse_psths does not contain the frozen 25-ms change bins")
    coverage = selected.group_by("response_group", "reward_block").agg(
        pl.col("bin_center_seconds").n_unique().alias("n_bins"),
        pl.col("subject_id").n_unique().alias("n_mice"),
    )
    if coverage.height != len(RESPONSE_GROUPS) * len(BLOCKS):
        raise ValueError("mouse_psths lacks a response-group/reward-block combination")
    if coverage.filter((pl.col("n_bins") != expected_centers.size) | (pl.col("n_mice") < 2)).height:
        raise ValueError("each Figure 3 PSTH requires complete bins and at least two mice")
    mouse_coverage = selected.group_by("subject_id", "response_group", "reward_block").agg(
        pl.col("bin_center_seconds").n_unique().alias("n_bins")
    )
    if mouse_coverage.filter(pl.col("n_bins") != expected_centers.size).height:
        raise ValueError("each mouse PSTH must contain complete bins")
    return (
        selected.join(_group_lookup(), on="response_group", validate="m:1")
        .join(_block_lookup(), on="reward_block", validate="m:1")
        .sort("group_order", "block_order", "subject_id", "bin_center_seconds")
    )


def _prepare_mouse_effects(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "subject_id",
        "n_sessions",
        "n_units",
        "n_trials",
        "early_raw",
        "late_raw",
        "late_minus_early_raw",
        "early_total",
        "late_total",
        "late_minus_early_total",
        "early_adjusted",
        "late_adjusted",
        "late_minus_early_adjusted",
        "late_adjustment_change",
    }
    _require_columns(frame, required, frame_name="mouse_effects")
    selected = frame.select(
        pl.col("subject_id").cast(pl.String),
        pl.col("n_sessions").cast(pl.Int64),
        pl.col("n_units").cast(pl.Int64),
        pl.col("n_trials").cast(pl.Int64),
        *(
            pl.col(column).cast(pl.Float64)
            for column in sorted(required - {"subject_id", "n_sessions", "n_units", "n_trials"})
        ),
    ).sort("subject_id")
    if selected.height < 2 or selected.get_column("subject_id").n_unique() != selected.height:
        raise ValueError("mouse_effects must contain one row for at least two mice")
    if selected.filter(
        (pl.col("n_sessions") <= 0) | (pl.col("n_units") <= 0) | (pl.col("n_trials") <= 0)
    ).height:
        raise ValueError("mouse_effects denominators must be positive")
    always_finite = {
        "early_raw",
        "late_raw",
        "late_minus_early_raw",
        "early_total",
        "late_total",
        "late_minus_early_total",
    }
    for column in always_finite:
        if selected.filter(~pl.col(column).is_finite()).height:
            raise ValueError(f"mouse_effects {column} must be finite")
    adjusted_columns = {
        "early_adjusted",
        "late_adjusted",
        "late_minus_early_adjusted",
        "late_adjustment_change",
    }
    for column in adjusted_columns:
        if selected.filter(pl.col(column).is_not_null() & ~pl.col(column).is_finite()).height:
            raise ValueError(f"mouse_effects {column} must be finite or null")
    if (
        selected.filter(
            pl.col("early_adjusted").is_not_null() & pl.col("late_adjusted").is_not_null()
        ).height
        < 2
    ):
        raise ValueError("mouse_effects requires adjusted estimates for at least two mice")
    identities = (
        ("late_minus_early_raw", "late_raw", "early_raw"),
        ("late_minus_early_total", "late_total", "early_total"),
        ("late_minus_early_adjusted", "late_adjusted", "early_adjusted"),
        ("late_adjustment_change", "late_adjusted", "late_total"),
    )
    for result, minuend, subtrahend in identities:
        disagreement = selected.filter(
            pl.all_horizontal(
                pl.col(result).is_not_null(),
                pl.col(minuend).is_not_null(),
                pl.col(subtrahend).is_not_null(),
            )
            & ((pl.col(result) - (pl.col(minuend) - pl.col(subtrahend))).abs() > 1e-10)
        )
        if disagreement.height:
            raise ValueError(f"mouse_effects {result} does not reconstruct")
    return selected


def _prepare_statistics(frame: pl.DataFrame, effects: pl.DataFrame) -> pl.DataFrame:
    normalized = dg.statistics.normalize_statistics_table(frame)
    selected = normalized.filter(
        (pl.col("analysis_id") == ANALYSIS_ID) & pl.col("result_id").is_in(PRIMARY_STATISTIC_IDS)
    )
    if selected.height != len(PRIMARY_STATISTIC_IDS):
        raise ValueError("statistics lacks one row per Figure 3 primary result")
    if selected.get_column("result_id").n_unique() != selected.height:
        raise ValueError("statistics has duplicate Figure 3 primary results")
    value_by_result = {
        "late_minus_early_model_free": "late_minus_early_raw",
        "late_minus_early_model_total": "late_minus_early_total",
        "late_minus_early_model_adjusted": "late_minus_early_adjusted",
        "late_state_model_total": "late_total",
        "late_state_model_adjusted": "late_adjusted",
    }
    for result_id, value_column in value_by_result.items():
        row = selected.filter(pl.col("result_id") == result_id).row(0, named=True)
        expected = float(effects.get_column(value_column).mean())
        if row["estimate"] is not None and not math.isclose(
            float(row["estimate"]), expected, abs_tol=1e-10
        ):
            raise ValueError(f"statistics {result_id} does not match mouse source values")
    return selected.sort("result_id")


def _prepare_class_stability(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "subject_id",
        "n_sessions",
        "n_units",
        "n_stable_units",
        "n_early_sensory",
        "n_late_prelick",
        "n_action_related",
        "stable_assignment_fraction",
        "mean_assignment_margin",
    }
    _require_columns(frame, required, frame_name="class_stability")
    selected = frame.select(
        pl.col("subject_id").cast(pl.String),
        pl.col("n_sessions").cast(pl.Int64),
        pl.col("n_units").cast(pl.Int64),
        pl.col("n_stable_units").cast(pl.Int64),
        pl.col("n_early_sensory").cast(pl.Int64),
        pl.col("n_late_prelick").cast(pl.Int64),
        pl.col("n_action_related").cast(pl.Int64),
        pl.col("stable_assignment_fraction").cast(pl.Float64),
        pl.col("mean_assignment_margin").cast(pl.Float64),
    ).sort("subject_id")
    if selected.get_column("subject_id").n_unique() != selected.height:
        raise ValueError("class_stability must contain one row per mouse")
    if selected.filter(
        (pl.col("n_units") <= 0)
        | (pl.col("n_stable_units") < 0)
        | (pl.col("n_stable_units") > pl.col("n_units"))
        | (pl.col("stable_assignment_fraction") < 0)
        | (pl.col("stable_assignment_fraction") > 1)
    ).height:
        raise ValueError("class_stability contains invalid counts or fractions")
    class_sum = pl.col("n_early_sensory") + pl.col("n_late_prelick") + pl.col("n_action_related")
    if selected.filter(class_sum != pl.col("n_stable_units")).height:
        raise ValueError("stable response-class counts do not sum")
    return selected


def _summarize_psths(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = frame.group_by("response_group", "reward_block", "bin_center_seconds")
    for index, (key, group) in enumerate(grouped):
        response_group, reward_block, bin_center = key
        values = group.select("subject_id", "mean_sign_aligned_rate_hz")
        bootstrap = dg.statistics.bootstrap_mouse_mean(
            values,
            seed=BOOTSTRAP_SEED + index,
            mouse_column="subject_id",
            value_column="mean_sign_aligned_rate_hz",
            n_resamples=BOOTSTRAP_RESAMPLES,
        )
        rows.append(
            {
                "response_group": response_group,
                "reward_block": reward_block,
                "bin_center_seconds": float(bin_center),
                "mean_sign_aligned_rate_hz": bootstrap.estimate,
                "ci_low": bootstrap.ci_low,
                "ci_high": bootstrap.ci_high,
                "n_mice": bootstrap.n_mice,
                "n_sessions": int(group.get_column("n_sessions").sum()),
                "n_unit_session_contributions": int(
                    group.get_column("n_unit_session_contributions").sum()
                ),
            }
        )
    return (
        pl.DataFrame(rows)
        .join(_group_lookup(), on="response_group", validate="m:1")
        .join(_block_lookup(), on="reward_block", validate="m:1")
        .sort("group_order", "block_order", "bin_center_seconds")
    )


def _expected_change_bin_centers() -> np.ndarray:
    config = dg.functional_populations.DEFAULT_FUNCTIONAL_POPULATION_CONFIG
    count = int(
        round((config.change_stop_seconds - config.change_start_seconds) / config.bin_width_seconds)
    )
    edges = config.change_start_seconds + np.arange(count + 1) * config.bin_width_seconds
    return (edges[:-1] + edges[1:]) / 2


def _group_lookup() -> pl.DataFrame:
    return pl.DataFrame(
        RESPONSE_GROUPS,
        schema=("response_group", "group_order", "response_group_label"),
        orient="row",
    )


def _block_lookup() -> pl.DataFrame:
    return pl.DataFrame(
        BLOCKS,
        schema=("reward_block", "block_order", "reward_block_label"),
        orient="row",
    )


def _require_columns(frame: pl.DataFrame, columns: set[str], *, frame_name: str) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{frame_name} must be a polars DataFrame")
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {sorted(missing)}")
