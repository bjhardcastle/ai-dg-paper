"""Validated source-data preparation for anatomical-enrichment Figure 4."""

from __future__ import annotations

import dataclasses
import math

import polars as pl

import dg.anatomical_enrichment


@dataclasses.dataclass(frozen=True, slots=True)
class AnatomyFigureData:
    """Publication-facing Figure 4 tables and lead-screen identity."""

    unit_coordinates: pl.DataFrame
    region_coverage: pl.DataFrame
    mouse_effects: pl.DataFrame
    statistics: pl.DataFrame
    lead_permutation_null: pl.DataFrame
    lead_region_id: int
    lead_region_acronym: str
    lead_region_name: str


def prepare_anatomy_figure_data(
    unit_scores: pl.DataFrame,
    region_coverage: pl.DataFrame,
    mouse_effects: pl.DataFrame,
    statistics: pl.DataFrame,
    permutation_null: pl.DataFrame,
) -> AnatomyFigureData:
    """Validate artifact contracts and select the transparent discovery lead."""

    for name, frame in (
        ("unit_scores", unit_scores),
        ("region_coverage", region_coverage),
        ("mouse_effects", mouse_effects),
        ("statistics", statistics),
        ("permutation_null", permutation_null),
    ):
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"{name} must be a polars DataFrame")

    coverage = _prepare_coverage(region_coverage)
    eligible = coverage.filter(pl.col("coverage_eligible"))
    if eligible.is_empty():
        raise ValueError("Figure 4 requires at least one coverage-eligible parent region")
    coordinates = _prepare_coordinates(unit_scores, coverage)
    effects = _prepare_mouse_effects(mouse_effects, eligible)
    prepared_statistics = _prepare_statistics(statistics, effects, eligible)
    lead = prepared_statistics.sort(pl.col("estimate"), descending=True).row(0, named=True)
    lead_null = _prepare_lead_null(
        permutation_null,
        parent_region_id=lead["parent_region_id"],
    )
    return AnatomyFigureData(
        unit_coordinates=coordinates,
        region_coverage=coverage,
        mouse_effects=effects,
        statistics=prepared_statistics,
        lead_permutation_null=lead_null,
        lead_region_id=int(lead["parent_region_id"]),
        lead_region_acronym=str(lead["parent_region_acronym"]),
        lead_region_name=str(lead["parent_region_name"]),
    )


def _prepare_coverage(frame: pl.DataFrame) -> pl.DataFrame:
    required = {
        "parent_region_order",
        "parent_region_id",
        "parent_region_acronym",
        "parent_region_name",
        "n_mice",
        "n_sessions",
        "n_units",
        "median_units_per_represented_mouse",
        "coverage_eligible",
        "minimum_mice",
        "minimum_sessions",
        "minimum_median_units_per_mouse",
    }
    _require_columns(frame, required, frame_name="region_coverage")
    selected = frame.select(
        pl.col("parent_region_order").cast(pl.Int64),
        pl.col("parent_region_id").cast(pl.Int64),
        pl.col("parent_region_acronym").cast(pl.String),
        pl.col("parent_region_name").cast(pl.String),
        pl.col("n_mice").cast(pl.Int64),
        pl.col("n_sessions").cast(pl.Int64),
        pl.col("n_units").cast(pl.Int64),
        pl.col("median_units_per_represented_mouse").cast(pl.Float64),
        pl.col("coverage_eligible").cast(pl.Boolean),
        pl.col("minimum_mice").cast(pl.Int64),
        pl.col("minimum_sessions").cast(pl.Int64),
        pl.col("minimum_median_units_per_mouse").cast(pl.Int64),
    ).sort("parent_region_order")
    contract = dg.anatomical_enrichment.parent_region_contract()
    if selected.select(*contract.columns[:4]).rows() != contract.rows():
        raise ValueError("region_coverage differs from the complete locked parent-region level")
    if set(selected.get_column("minimum_mice")) != {dg.anatomical_enrichment.MIN_REGION_MICE}:
        raise ValueError("region_coverage has an unexpected mouse minimum")
    if set(selected.get_column("minimum_sessions")) != {
        dg.anatomical_enrichment.MIN_REGION_SESSIONS
    }:
        raise ValueError("region_coverage has an unexpected session minimum")
    if set(selected.get_column("minimum_median_units_per_mouse")) != {
        dg.anatomical_enrichment.MIN_MEDIAN_UNITS_PER_MOUSE
    }:
        raise ValueError("region_coverage has an unexpected unit minimum")
    reconstructed = (
        (pl.col("n_mice") >= dg.anatomical_enrichment.MIN_REGION_MICE)
        & (pl.col("n_sessions") >= dg.anatomical_enrichment.MIN_REGION_SESSIONS)
        & (
            pl.col("median_units_per_represented_mouse")
            >= dg.anatomical_enrichment.MIN_MEDIAN_UNITS_PER_MOUSE
        )
    )
    if selected.filter(pl.col("coverage_eligible") != reconstructed).height:
        raise ValueError("coverage_eligible does not reconstruct from the locked minima")
    return selected


def _prepare_coordinates(unit_scores: pl.DataFrame, coverage: pl.DataFrame) -> pl.DataFrame:
    required = {
        "_nwb_path",
        "_table_index",
        "subject_id",
        "parent_region_id",
        "parent_region_acronym",
        "parent_region_name",
        "anterior_posterior_ccf_coordinate",
        "dorsal_ventral_ccf_coordinate",
        "left_right_ccf_coordinate",
        "well_isolated",
        "axis_weight",
    }
    _require_columns(unit_scores, required, frame_name="unit_scores")
    if unit_scores.select("_nwb_path", "_table_index").n_unique() != unit_scores.height:
        raise ValueError("unit_scores contains duplicate unit keys")
    selected = unit_scores.select(
        "_nwb_path",
        pl.col("_table_index").cast(pl.UInt32),
        pl.col("subject_id").cast(pl.String),
        pl.col("parent_region_id").cast(pl.Int64),
        pl.col("parent_region_acronym").cast(pl.String),
        pl.col("parent_region_name").cast(pl.String),
        pl.col("anterior_posterior_ccf_coordinate").cast(pl.Float64),
        pl.col("dorsal_ventral_ccf_coordinate").cast(pl.Float64),
        pl.col("left_right_ccf_coordinate").cast(pl.Float64),
        pl.col("well_isolated").cast(pl.Boolean),
        pl.col("axis_weight").cast(pl.Float64),
    ).join(
        coverage.select("parent_region_id", "parent_region_order", "coverage_eligible"),
        on="parent_region_id",
        how="left",
        validate="m:1",
    )
    selected = selected.filter(
        pl.col("well_isolated")
        & pl.col("axis_weight").is_finite()
        & pl.col("parent_region_id").is_not_null()
        & pl.col("anterior_posterior_ccf_coordinate").is_finite()
        & pl.col("dorsal_ventral_ccf_coordinate").is_finite()
    ).sort("parent_region_order", "subject_id", "_nwb_path", "_table_index")
    if selected.is_empty():
        raise ValueError("unit_scores contains no mapped finite CCF coordinates")
    return selected


def _prepare_mouse_effects(frame: pl.DataFrame, eligible: pl.DataFrame) -> pl.DataFrame:
    required = {
        "subject_id",
        "parent_region_id",
        "parent_region_acronym",
        "parent_region_name",
        "score_type",
        "mouse_effect",
        "n_sessions",
        "n_units",
    }
    _require_columns(frame, required, frame_name="mouse_effects")
    ids = eligible.get_column("parent_region_id").to_list()
    selected = (
        frame.filter(
            (pl.col("score_type") == "absolute_loading") & pl.col("parent_region_id").is_in(ids)
        )
        .select(
            pl.col("subject_id").cast(pl.String),
            pl.col("parent_region_id").cast(pl.Int64),
            pl.col("parent_region_acronym").cast(pl.String),
            pl.col("parent_region_name").cast(pl.String),
            pl.lit("absolute_loading").alias("score_type"),
            pl.col("mouse_effect").cast(pl.Float64),
            pl.col("n_sessions").cast(pl.Int64),
            pl.col("n_units").cast(pl.Int64),
        )
        .sort("parent_region_id", "subject_id")
    )
    if selected.select("subject_id", "parent_region_id").n_unique() != selected.height:
        raise ValueError("mouse_effects contains duplicate mouse-region rows")
    if selected.filter(
        ~pl.col("mouse_effect").is_finite() | (pl.col("n_sessions") <= 0) | (pl.col("n_units") <= 0)
    ).height:
        raise ValueError("mouse_effects contains invalid estimates or denominators")
    for row in eligible.iter_rows(named=True):
        observed = selected.filter(pl.col("parent_region_id") == row["parent_region_id"])
        if observed.height != row["n_mice"]:
            raise ValueError("mouse_effects n_mice disagrees with region_coverage")
    return selected


def _prepare_statistics(
    frame: pl.DataFrame,
    mouse_effects: pl.DataFrame,
    eligible: pl.DataFrame,
) -> pl.DataFrame:
    required = {
        "analysis_id",
        "result_id",
        "estimate",
        "ci_low",
        "ci_high",
        "p_value",
        "adjusted_p_value",
        "adjustment_method",
        "n_mice",
        "n_sessions",
        "n_units",
        "status",
    }
    _require_columns(frame, required, frame_name="statistics")
    selected = frame.filter(pl.col("result_id").str.starts_with("absolute_loading__")).with_columns(
        pl.col("result_id").str.split("__").list.get(1).alias("parent_region_acronym")
    )
    selected = selected.join(
        eligible.select(
            "parent_region_id",
            "parent_region_order",
            "parent_region_acronym",
            "parent_region_name",
        ),
        on="parent_region_acronym",
        how="inner",
        validate="1:1",
    ).sort("parent_region_order")
    if selected.height != eligible.height:
        raise ValueError("statistics lacks one absolute-loading row per eligible region")
    if selected.filter(
        (pl.col("analysis_id") != dg.anatomical_enrichment.ANALYSIS_ID)
        | (pl.col("adjustment_method") != "holm")
        | ~pl.col("status").is_in(("pass", "null"))
    ).height:
        raise ValueError("statistics contains an invalid analysis identity or status")
    for column in ("estimate", "ci_low", "ci_high", "p_value", "adjusted_p_value"):
        if selected.filter(~pl.col(column).cast(pl.Float64).is_finite()).height:
            raise ValueError(f"statistics contains nonfinite {column}")
    if selected.filter(
        (pl.col("ci_low") > pl.col("estimate"))
        | (pl.col("ci_high") < pl.col("estimate"))
        | (pl.col("p_value") < 0)
        | (pl.col("p_value") > 1)
        | (pl.col("adjusted_p_value") < 0)
        | (pl.col("adjusted_p_value") > 1)
    ).height:
        raise ValueError("statistics contains an invalid interval or P value")
    for row in selected.iter_rows(named=True):
        mice = mouse_effects.filter(pl.col("parent_region_id") == row["parent_region_id"])
        observed = float(mice.get_column("mouse_effect").mean())
        if not math.isclose(observed, row["estimate"], rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("a regional statistic disagrees with plotted mouse effects")
        if row["n_mice"] != mice.height:
            raise ValueError("a regional statistic n_mice disagrees with plotted mice")
    return selected


def _prepare_lead_null(frame: pl.DataFrame, *, parent_region_id: int) -> pl.DataFrame:
    required = {
        "score_type",
        "parent_region_id",
        "permutation_index",
        "null_estimate",
        "permutation_seed",
    }
    _require_columns(frame, required, frame_name="permutation_null")
    selected = frame.filter(
        (pl.col("score_type") == "absolute_loading")
        & (pl.col("parent_region_id") == parent_region_id)
    ).sort("permutation_index")
    if selected.is_empty():
        raise ValueError("permutation_null lacks the lead absolute-loading region")
    if selected.get_column("permutation_index").n_unique() != selected.height:
        raise ValueError("permutation_null contains duplicate draws")
    expected = list(range(selected.height))
    if selected.get_column("permutation_index").to_list() != expected:
        raise ValueError("permutation_null draws are not contiguous from zero")
    if selected.filter(~pl.col("null_estimate").is_finite()).height:
        raise ValueError("permutation_null contains nonfinite estimates")
    if selected.get_column("permutation_seed").n_unique() != 1:
        raise ValueError("permutation_null contains multiple seeds")
    return selected


def _require_columns(frame: pl.DataFrame, columns: set[str], *, frame_name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {sorted(missing)}")
