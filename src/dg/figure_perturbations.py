"""Validated publication-facing data for perturbation Figure 6."""

from __future__ import annotations

import dataclasses
import math

import polars as pl

import dg.statistics

BLOCKS = (
    ("engaged_1", 1, "E1"),
    ("no_reward", 2, "NR"),
    ("engaged_2", 3, "E2"),
)
CONDITIONS = (
    ("contrast", "full", 1, "Contrast: full"),
    ("contrast", "reduced", 2, "Contrast: 70%"),
    ("novelty", "familiar", 3, "Identity: familiar"),
    ("novelty", "designated_novel", 4, "Identity: designated novel"),
)
METRICS = (
    ("behavior_response", "Response probability"),
    ("early_reward_axis", "Early frozen-axis score (a.u.)"),
)
BOOTSTRAP_SEED = 6600
BOOTSTRAP_RESAMPLES = 10_000


@dataclasses.dataclass(frozen=True, slots=True)
class PerturbationFigureData:
    mouse_blocks: pl.DataFrame
    block_summary: pl.DataFrame
    mouse_contrasts: pl.DataFrame
    mouse_interactions: pl.DataFrame
    panel_statistics: pl.DataFrame


def prepare_perturbation_figure_data(
    session_blocks: pl.DataFrame,
    mouse_contrasts: pl.DataFrame,
    mouse_interactions: pl.DataFrame,
    statistics: pl.DataFrame,
) -> PerturbationFigureData:
    """Validate artifact identities and compute equal-mouse block summaries."""

    for name, frame in (
        ("session_blocks", session_blocks),
        ("mouse_contrasts", mouse_contrasts),
        ("mouse_interactions", mouse_interactions),
        ("statistics", statistics),
    ):
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"{name} must be a polars DataFrame")

    mouse_blocks = _prepare_mouse_blocks(session_blocks)
    prepared_contrasts = _prepare_mouse_contrasts(mouse_contrasts)
    prepared_interactions = _prepare_mouse_interactions(mouse_interactions)
    prepared_statistics = _prepare_statistics(
        statistics,
        prepared_contrasts,
        prepared_interactions,
    )
    return PerturbationFigureData(
        mouse_blocks=mouse_blocks,
        block_summary=_summarize_blocks(mouse_blocks),
        mouse_contrasts=prepared_contrasts,
        mouse_interactions=prepared_interactions,
        panel_statistics=prepared_statistics,
    )


def _prepare_mouse_blocks(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "subject_id",
        "ecephys_session_id",
        "perturbation_family",
        "condition",
        "reward_block",
        "n_trials",
        "response_probability",
        "mean_axis_score",
    }
    _require_columns(frame, required, frame_name="session_blocks")
    expanded = pl.concat(
        [
            frame.select(
                "subject_id",
                "ecephys_session_id",
                "perturbation_family",
                "condition",
                "reward_block",
                "n_trials",
                pl.lit("behavior_response").alias("metric"),
                pl.col("response_probability").alias("value"),
            ),
            frame.select(
                "subject_id",
                "ecephys_session_id",
                "perturbation_family",
                "condition",
                "reward_block",
                "n_trials",
                pl.lit("early_reward_axis").alias("metric"),
                pl.col("mean_axis_score").alias("value"),
            ),
        ],
        how="vertical",
    ).filter(pl.col("n_trials") >= 5)
    selected = (
        expanded.group_by(
            "subject_id",
            "perturbation_family",
            "condition",
            "reward_block",
            "metric",
        )
        .agg(
            pl.col("ecephys_session_id").n_unique().alias("n_sessions"),
            pl.col("n_trials").sum(),
            pl.col("value").mean(),
        )
        .join(_condition_lookup(), on=("perturbation_family", "condition"), how="left")
        .join(_block_lookup(), on="reward_block", how="left")
        .sort("metric", "condition_order", "subject_id", "block_order")
    )
    if selected.is_empty():
        raise ValueError("session_blocks has no supported rows")
    if selected.filter(
        pl.col("condition_order").is_null() | pl.col("block_order").is_null()
    ).height:
        raise ValueError("session_blocks contains an unexpected condition or block")
    _validate_unique(
        selected,
        ("subject_id", "perturbation_family", "condition", "reward_block", "metric"),
        frame_name="mouse_blocks",
    )
    _validate_finite(selected, "value", frame_name="mouse_blocks")
    return selected


def _prepare_mouse_contrasts(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "subject_id",
        "perturbation_family",
        "condition",
        "metric",
        "reversible_contrast",
        "n_sessions",
        "n_trials",
    }
    _require_columns(frame, required, frame_name="mouse_contrasts")
    selected = frame.join(
        _condition_lookup(),
        on=("perturbation_family", "condition"),
        how="left",
    ).sort("metric", "condition_order", "subject_id")
    if selected.filter(pl.col("condition_order").is_null()).height:
        raise ValueError("mouse_contrasts contains an unexpected condition")
    _validate_unique(
        selected,
        ("subject_id", "perturbation_family", "condition", "metric"),
        frame_name="mouse_contrasts",
    )
    _validate_finite(selected, "reversible_contrast", frame_name="mouse_contrasts")
    return selected


def _prepare_mouse_interactions(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "subject_id",
        "perturbation_family",
        "metric",
        "interaction",
        "n_sessions",
        "n_trials",
    }
    _require_columns(frame, required, frame_name="mouse_interactions")
    selected = frame.select(
        pl.col("subject_id").cast(pl.String),
        pl.col("perturbation_family").cast(pl.String),
        pl.col("metric").cast(pl.String),
        pl.col("interaction").cast(pl.Float64),
        pl.col("n_sessions").cast(pl.Int64),
        pl.col("n_trials").cast(pl.Int64),
    ).sort("metric", "perturbation_family", "subject_id")
    _validate_unique(
        selected,
        ("subject_id", "perturbation_family", "metric"),
        frame_name="mouse_interactions",
    )
    _validate_finite(selected, "interaction", frame_name="mouse_interactions")
    expected = {(family, metric) for family in ("contrast", "novelty") for metric, _ in METRICS}
    observed = set(selected.select("perturbation_family", "metric").iter_rows())
    if observed != expected:
        raise ValueError("mouse_interactions lacks a planned family/metric combination")
    return selected


def _prepare_statistics(
    frame: pl.DataFrame,
    contrasts: pl.DataFrame,
    interactions: pl.DataFrame,
) -> pl.DataFrame:
    required = {
        "analysis_id",
        "result_id",
        "estimate",
        "ci_low",
        "ci_high",
        "p_value",
        "adjusted_p_value",
        "n_mice",
        "status",
    }
    _require_columns(frame, required, frame_name="statistics")
    expected_ids = {
        "behavior_response_contrast_state_by_perturbation_interaction_mouse_mean",
        "behavior_response_novelty_state_by_perturbation_interaction_mouse_mean",
        "early_reward_axis_contrast_reduced_reversible_mouse_mean",
        "early_reward_axis_novelty_designated_novel_reversible_mouse_mean",
    }
    selected = frame.filter(pl.col("result_id").is_in(sorted(expected_ids))).with_columns(
        pl.col("result_id")
        .str.extract(r"^(behavior_response|early_reward_axis)_", group_index=1)
        .alias("metric"),
        pl.col("result_id")
        .str.extract(r"_(contrast|novelty)_", group_index=1)
        .alias("perturbation_family"),
        pl.when(pl.col("result_id").str.contains("state_by_perturbation"))
        .then(pl.lit("interaction"))
        .otherwise(pl.lit("heldout_condition"))
        .alias("panel_estimand"),
    )
    if (
        selected.height != len(expected_ids)
        or set(selected.get_column("result_id")) != expected_ids
    ):
        raise ValueError("statistics lacks exactly the four perturbation interactions")
    if selected.filter(
        (pl.col("analysis_id") != "heldout_perturbation_axis")
        | ~pl.col("status").is_in(("pass", "null"))
    ).height:
        raise ValueError("a perturbation interaction statistic is not reportable")
    for column in ("estimate", "ci_low", "ci_high", "p_value", "adjusted_p_value"):
        _validate_finite(selected, column, frame_name="statistics")
    for row in selected.iter_rows(named=True):
        if row["panel_estimand"] == "interaction":
            values = interactions.filter(
                (pl.col("metric") == row["metric"])
                & (pl.col("perturbation_family") == row["perturbation_family"])
            )
            value_column = "interaction"
        else:
            condition = (
                "reduced" if row["perturbation_family"] == "contrast" else "designated_novel"
            )
            values = contrasts.filter(
                (pl.col("metric") == row["metric"])
                & (pl.col("perturbation_family") == row["perturbation_family"])
                & (pl.col("condition") == condition)
            )
            value_column = "reversible_contrast"
        mean = float(values.get_column(value_column).mean())
        if not math.isclose(mean, row["estimate"], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("a perturbation statistic disagrees with plotted mouse values")
        if values.height != row["n_mice"]:
            raise ValueError("a perturbation statistic has the wrong mouse count")
    return selected.sort("metric", "perturbation_family")


def _summarize_blocks(mouse_blocks: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for index, (key, group) in enumerate(
        mouse_blocks.group_by(
            "metric",
            "perturbation_family",
            "condition",
            "condition_order",
            "condition_label",
            "reward_block",
            "block_order",
            "block_label",
            maintain_order=True,
        )
    ):
        values = group.select(
            pl.col("subject_id").alias("mouse_id"),
            pl.col("value"),
        )
        interval = dg.statistics.bootstrap_mouse_mean(
            values,
            seed=BOOTSTRAP_SEED + index,
            value_column="value",
            n_resamples=BOOTSTRAP_RESAMPLES,
        )
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(
            zip(
                (
                    "metric",
                    "perturbation_family",
                    "condition",
                    "condition_order",
                    "condition_label",
                    "reward_block",
                    "block_order",
                    "block_label",
                ),
                key_tuple,
                strict=True,
            )
        )
        rows.append(
            {
                **row,
                "estimate": interval.estimate,
                "ci_low": interval.ci_low,
                "ci_high": interval.ci_high,
                "n_mice": interval.n_mice,
                "n_sessions": int(group.get_column("n_sessions").sum()),
                "n_trials": int(group.get_column("n_trials").sum()),
            }
        )
    return pl.DataFrame(rows).sort("metric", "condition_order", "block_order")


def _condition_lookup() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "perturbation_family": [row[0] for row in CONDITIONS],
            "condition": [row[1] for row in CONDITIONS],
            "condition_order": [row[2] for row in CONDITIONS],
            "condition_label": [row[3] for row in CONDITIONS],
        }
    )


def _block_lookup() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "reward_block": [row[0] for row in BLOCKS],
            "block_order": [row[1] for row in BLOCKS],
            "block_label": [row[2] for row in BLOCKS],
        }
    )


def _require_columns(
    frame: pl.DataFrame,
    columns: set[str],
    *,
    frame_name: str,
) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {sorted(missing)}")


def _validate_unique(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    if frame.select(*columns).is_duplicated().any():
        raise ValueError(f"{frame_name} contains duplicate rows for {columns}")


def _validate_finite(frame: pl.DataFrame, column: str, *, frame_name: str) -> None:
    if frame.select(pl.col(column).is_null().any() | ~pl.col(column).is_finite().all()).item():
        raise ValueError(f"{frame_name}.{column} must be finite and non-null")
