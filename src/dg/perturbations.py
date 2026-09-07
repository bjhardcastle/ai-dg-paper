"""Discovery-only novelty and contrast probes for the frozen neural axis.

This module operates on already-materialized behavior rows and neural scores.
It contains no NWB I/O.  Reduced-contrast and designated-novel trials are kept
out of the familiar full-contrast representation used to fit the Figure 2
reward-state axis.
"""

from __future__ import annotations

import dataclasses
import math

import polars as pl

import dg.neural_transitions

SOURCE_COLUMN = "_nwb_path"
MOUSE_COLUMN = "subject_id"
SESSION_COLUMN = "ecephys_session_id"
BLOCKS = (
    dg.neural_transitions.ENGAGED_1,
    dg.neural_transitions.NO_REWARD,
    dg.neural_transitions.ENGAGED_2,
)


@dataclasses.dataclass(frozen=True, slots=True)
class PerturbationConfig:
    """Fixed trial eligibility and support rules for Figure 6."""

    lick_exclusion_seconds: float = 0.150
    minimum_trials_per_session_block_condition: int = 5

    def __post_init__(self) -> None:
        if not math.isfinite(self.lick_exclusion_seconds) or self.lick_exclusion_seconds <= 0:
            raise ValueError("lick_exclusion_seconds must be finite and positive")
        value = self.minimum_trials_per_session_block_condition
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                "minimum_trials_per_session_block_condition must be a positive integer"
            )


DEFAULT_PERTURBATION_CONFIG = PerturbationConfig()


def select_perturbation_trials(
    trials: pl.DataFrame | pl.LazyFrame,
    *,
    config: PerturbationConfig = DEFAULT_PERTURBATION_CONFIG,
) -> pl.DataFrame:
    """Select completed physical changes with an early lick-free interval.

    Image identity and relative contrast are parsed only from the audited raw
    image token.  ``novel_image_id`` is retained as a designated identity
    label; it is not interpreted as first exposure.
    """

    frame = trials.collect() if isinstance(trials, pl.LazyFrame) else trials
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("trials must be a polars DataFrame or LazyFrame")
    required = {
        SOURCE_COLUMN,
        MOUSE_COLUMN,
        SESSION_COLUMN,
        "_table_index",
        "change_time",
        "change_image_name",
        "novel_image_id",
        "physical_image_change",
        "reward_block",
        "aborted",
        "auto_rewarded",
        "lick_times",
        "response_in_window",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"trials is missing columns: {sorted(missing)}")

    selected = frame.filter(
        ~pl.col("aborted").fill_null(True)
        & ~pl.col("auto_rewarded").fill_null(True)
        & pl.col("physical_image_change").fill_null(False)
        & pl.col("change_time").is_not_null()
        & pl.col("change_time").is_finite()
        & pl.col("change_image_name").is_not_null()
        & pl.col("novel_image_id").is_not_null()
        & pl.col("reward_block").is_in(BLOCKS)
        & pl.col("lick_times").is_not_null()
        & pl.col("response_in_window").is_not_null()
    )
    if "lick_times_valid" in selected.columns:
        selected = selected.filter(pl.col("lick_times_valid").fill_null(False))

    selected = selected.with_columns(
        pl.col("change_image_name")
        .str.extract(r"^(im\d+)_r-", group_index=1)
        .alias("change_image_id"),
        pl.col("change_image_name")
        .str.extract(r"_r-([0-9]+(?:\.[0-9]+)?)$", group_index=1)
        .cast(pl.Float64, strict=False)
        .alias("change_relative_contrast"),
        (pl.col("change_image_name") == pl.col("novel_image_id")).alias("is_designated_novel"),
    ).filter(
        pl.col("change_image_id").is_not_null()
        & pl.col("change_relative_contrast").is_in([0.7, 1.0])
    )

    lick_free = [
        _lick_array_is_free(
            row["lick_times"],
            event_time=float(row["change_time"]),
            half_width=config.lick_exclusion_seconds,
        )
        for row in selected.select("lick_times", "change_time").iter_rows(named=True)
    ]
    return (
        selected.with_columns(
            pl.Series("lick_free_for_early_neural", lick_free, dtype=pl.Boolean),
            pl.lit(config.lick_exclusion_seconds).alias("lick_exclusion_half_width_seconds"),
        )
        .filter(pl.col("lick_free_for_early_neural"))
        .sort(SOURCE_COLUMN, "change_time", "_table_index")
    )


def expand_probe_conditions(trials: pl.DataFrame) -> pl.DataFrame:
    """Expand eligible rows into crossed contrast and novelty probes.

    Contrast is restricted to the same physical identity (``im115``).
    Novelty compares the session-designated novel identity with familiar
    full-contrast identities and therefore retains the known identity/day
    confound in an explicit ``interpretation_scope`` column.
    """

    required = {
        "change_image_id",
        "change_relative_contrast",
        "is_designated_novel",
    }
    missing = required.difference(trials.columns)
    if missing:
        raise ValueError(f"trials is missing columns: {sorted(missing)}")

    contrast = (
        trials.filter(pl.col("change_image_id") == "im115")
        .with_columns(
            pl.lit("contrast").alias("perturbation_family"),
            pl.when(pl.col("change_relative_contrast") == 0.7)
            .then(pl.lit("reduced"))
            .when(pl.col("change_relative_contrast") == 1.0)
            .then(pl.lit("full"))
            .otherwise(None)
            .alias("condition"),
            pl.lit("within_identity_im115_contrast").alias("interpretation_scope"),
        )
        .filter(pl.col("condition").is_not_null())
    )

    novelty = trials.filter(pl.col("change_relative_contrast") == 1.0).with_columns(
        pl.lit("novelty").alias("perturbation_family"),
        pl.when(pl.col("is_designated_novel"))
        .then(pl.lit("designated_novel"))
        .otherwise(pl.lit("familiar"))
        .alias("condition"),
        pl.lit("designated_identity_not_first_exposure;identity_day_confounded").alias(
            "interpretation_scope"
        ),
    )
    return pl.concat([contrast, novelty], how="diagonal_relaxed").sort(
        SOURCE_COLUMN,
        "_table_index",
        "perturbation_family",
    )


def summarize_session_blocks(
    scored_trials: pl.DataFrame,
) -> pl.DataFrame:
    """Return session-level behavior and frozen-axis summaries by block."""

    required = {
        SOURCE_COLUMN,
        MOUSE_COLUMN,
        SESSION_COLUMN,
        "reward_block",
        "perturbation_family",
        "condition",
        "interpretation_scope",
        "response_in_window",
        "axis_score",
    }
    missing = required.difference(scored_trials.columns)
    if missing:
        raise ValueError(f"scored_trials is missing columns: {sorted(missing)}")
    if scored_trials.is_empty():
        raise ValueError("scored_trials must not be empty")
    if scored_trials.select(pl.col("axis_score").is_null().any()).item():
        raise ValueError("axis_score must not be null")
    return (
        scored_trials.group_by(
            SOURCE_COLUMN,
            MOUSE_COLUMN,
            SESSION_COLUMN,
            "perturbation_family",
            "condition",
            "interpretation_scope",
            "reward_block",
        )
        .agg(
            pl.len().alias("n_trials"),
            pl.col("response_in_window").cast(pl.Float64).mean().alias("response_probability"),
            pl.col("axis_score").mean().alias("mean_axis_score"),
        )
        .sort(
            "perturbation_family",
            "condition",
            MOUSE_COLUMN,
            SESSION_COLUMN,
            "reward_block",
        )
    )


def compute_session_reversible_contrasts(
    session_blocks: pl.DataFrame,
    *,
    config: PerturbationConfig = DEFAULT_PERTURBATION_CONFIG,
) -> pl.DataFrame:
    """Compute three-block reversible contrasts for both response modalities."""

    required = {
        SOURCE_COLUMN,
        MOUSE_COLUMN,
        SESSION_COLUMN,
        "perturbation_family",
        "condition",
        "interpretation_scope",
        "reward_block",
        "n_trials",
        "response_probability",
        "mean_axis_score",
    }
    missing = required.difference(session_blocks.columns)
    if missing:
        raise ValueError(f"session_blocks is missing columns: {sorted(missing)}")

    keys = [
        SOURCE_COLUMN,
        MOUSE_COLUMN,
        SESSION_COLUMN,
        "perturbation_family",
        "condition",
        "interpretation_scope",
    ]
    rows: list[dict[str, object]] = []
    for key, group in session_blocks.group_by(keys, maintain_order=True):
        key_values = key if isinstance(key, tuple) else (key,)
        base = dict(zip(keys, key_values, strict=True))
        by_block = {row["reward_block"]: row for row in group.iter_rows(named=True)}
        complete = set(by_block) == set(BLOCKS)
        support = complete and all(
            int(by_block[block]["n_trials"]) >= config.minimum_trials_per_session_block_condition
            for block in BLOCKS
        )
        for metric, value_column in (
            ("behavior_response", "response_probability"),
            ("early_reward_axis", "mean_axis_score"),
        ):
            values = {
                block: float(by_block[block][value_column]) if complete else None
                for block in BLOCKS
            }
            contrast = (
                0.5 * values[BLOCKS[0]] - values[BLOCKS[1]] + 0.5 * values[BLOCKS[2]]
                if support
                else None
            )
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "engaged_1": values[BLOCKS[0]],
                    "no_reward": values[BLOCKS[1]],
                    "engaged_2": values[BLOCKS[2]],
                    "reversible_contrast": contrast,
                    "n_trials": sum(int(by_block[block]["n_trials"]) for block in BLOCKS)
                    if complete
                    else sum(int(row["n_trials"]) for row in by_block.values()),
                    "minimum_block_trials": min(
                        (int(by_block[block]["n_trials"]) for block in BLOCKS),
                        default=0,
                    ),
                    "status": "pass" if support else "insufficient_trial_support",
                }
            )
    return pl.DataFrame(rows).sort(
        "perturbation_family", "condition", "metric", MOUSE_COLUMN, SESSION_COLUMN
    )


def aggregate_mouse_contrasts(session_contrasts: pl.DataFrame) -> pl.DataFrame:
    """Average valid session contrasts within mouse with equal session weight."""

    required = {
        MOUSE_COLUMN,
        SESSION_COLUMN,
        "perturbation_family",
        "condition",
        "metric",
        "reversible_contrast",
        "n_trials",
        "status",
    }
    missing = required.difference(session_contrasts.columns)
    if missing:
        raise ValueError(f"session_contrasts is missing columns: {sorted(missing)}")
    return (
        session_contrasts.filter(
            (pl.col("status") == "pass")
            & pl.col("reversible_contrast").is_not_null()
            & pl.col("reversible_contrast").is_finite()
        )
        .group_by(MOUSE_COLUMN, "perturbation_family", "condition", "metric")
        .agg(
            pl.col(SESSION_COLUMN).n_unique().alias("n_sessions"),
            pl.col("n_trials").sum(),
            pl.col("reversible_contrast").mean(),
        )
        .sort("perturbation_family", "condition", "metric", MOUSE_COLUMN)
    )


def compute_mouse_perturbation_interactions(
    mouse_contrasts: pl.DataFrame,
) -> pl.DataFrame:
    """Pair perturbation and reference reversible contrasts within mouse."""

    required = {
        MOUSE_COLUMN,
        "perturbation_family",
        "condition",
        "metric",
        "reversible_contrast",
    }
    missing = required.difference(mouse_contrasts.columns)
    if missing:
        raise ValueError(f"mouse_contrasts is missing columns: {sorted(missing)}")

    condition_pairs = {
        "contrast": ("reduced", "full"),
        "novelty": ("designated_novel", "familiar"),
    }
    rows: list[dict[str, object]] = []
    for family, (test_condition, reference_condition) in condition_pairs.items():
        family_rows = mouse_contrasts.filter(pl.col("perturbation_family") == family)
        for metric in family_rows.get_column("metric").unique().sort().to_list():
            metric_rows = family_rows.filter(pl.col("metric") == metric)
            test = metric_rows.filter(pl.col("condition") == test_condition).select(
                MOUSE_COLUMN,
                pl.col("reversible_contrast").alias("test_reversible_contrast"),
                pl.col("n_sessions").alias("test_n_sessions"),
                pl.col("n_trials").alias("test_n_trials"),
            )
            reference = metric_rows.filter(pl.col("condition") == reference_condition).select(
                MOUSE_COLUMN,
                pl.col("reversible_contrast").alias("reference_reversible_contrast"),
                pl.col("n_sessions").alias("reference_n_sessions"),
                pl.col("n_trials").alias("reference_n_trials"),
            )
            paired = test.join(reference, on=MOUSE_COLUMN, how="inner", validate="1:1")
            for row in paired.iter_rows(named=True):
                rows.append(
                    {
                        MOUSE_COLUMN: row[MOUSE_COLUMN],
                        "perturbation_family": family,
                        "metric": metric,
                        "test_condition": test_condition,
                        "reference_condition": reference_condition,
                        "test_reversible_contrast": row["test_reversible_contrast"],
                        "reference_reversible_contrast": row["reference_reversible_contrast"],
                        "interaction": row["test_reversible_contrast"]
                        - row["reference_reversible_contrast"],
                        "n_sessions": min(row["test_n_sessions"], row["reference_n_sessions"]),
                        "n_trials": row["test_n_trials"] + row["reference_n_trials"],
                    }
                )
    return pl.DataFrame(rows).sort("perturbation_family", "metric", MOUSE_COLUMN)


def _lick_array_is_free(
    values: object,
    *,
    event_time: float,
    half_width: float,
) -> bool:
    if values is None or not math.isfinite(event_time):
        return False
    try:
        licks = [float(value) for value in values]  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if any(not math.isfinite(value) for value in licks):
        return False
    lower = event_time - half_width
    upper = event_time + half_width
    return not any(lower <= value <= upper for value in licks)
