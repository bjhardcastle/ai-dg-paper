"""Discovery-only anatomical enrichment of reward-state-axis loadings.

The Figure 2 decoder is session specific, so raw coefficient scales cannot be
compared across recordings.  This module converts each session's coefficients
to two dimensionless quantities before looking at anatomy: a signed z score
and a z score of coefficient magnitude.  Regional estimates are then formed
session first, mouse second.  The randomization null shuffles those fixed
scores over anatomical labels within each session, preserving every session's
unit counts and recording footprint.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import polars as pl

import dg.data
import dg.statistics

ANALYSIS_ID = "anatomical_axis_enrichment"
ANALYSIS_TIER = "discovery"
DEFAULT_SEED = 1051
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_PERMUTATIONS = 10_000
MIN_REGION_MICE = 5
MIN_REGION_SESSIONS = 8
MIN_MEDIAN_UNITS_PER_MOUSE = 10

# D09 parent level: Allen CCFv3 graph 1, official 12 major divisions
# (structure set 687527670).  Keeping the complete level here prevents a
# result-dependent choice of parent regions; coverage rules decide which rows
# are estimable.
PARENT_REGIONS = (
    (1097, "HY", "Hypothalamus"),
    (313, "MB", "Midbrain"),
    (698, "OLF", "Olfactory areas"),
    (803, "PAL", "Pallidum"),
    (315, "Isocortex", "Isocortex"),
    (703, "CTXsp", "Cortical subplate"),
    (477, "STR", "Striatum"),
    (354, "MY", "Medulla"),
    (549, "TH", "Thalamus"),
    (512, "CB", "Cerebellum"),
    (1089, "HPF", "Hippocampal formation"),
    (771, "P", "Pons"),
)
SCORE_TYPES = (
    ("absolute_loading", "Absolute reward-state-axis loading"),
    ("signed_loading", "Signed E1-versus-NR axis loading"),
)


@dataclasses.dataclass(frozen=True, slots=True)
class AnatomicalEnrichmentTables:
    """All analysis-ready and publication-facing Figure 4 tables."""

    unit_scores: pl.DataFrame
    region_coverage: pl.DataFrame
    session_effects: pl.DataFrame
    mouse_effects: pl.DataFrame
    statistics: pl.DataFrame
    permutation_null: pl.DataFrame


def parent_region_contract() -> pl.DataFrame:
    """Return the complete, ordered D09 parent-region level."""

    return pl.DataFrame(
        {
            "parent_region_order": list(range(1, len(PARENT_REGIONS) + 1)),
            "parent_region_id": [row[0] for row in PARENT_REGIONS],
            "parent_region_acronym": [row[1] for row in PARENT_REGIONS],
            "parent_region_name": [row[2] for row in PARENT_REGIONS],
        },
        schema={
            "parent_region_order": pl.Int64,
            "parent_region_id": pl.Int64,
            "parent_region_acronym": pl.String,
            "parent_region_name": pl.String,
        },
    )


def assemble_unit_anatomy(
    unit_qc: pl.DataFrame,
    structure_labels: pl.DataFrame,
    electrodes: pl.DataFrame,
    ontology_structures: pl.DataFrame,
) -> pl.DataFrame:
    """Attach raw CCF labels, parent regions, and peak-channel coordinates.

    ``structure_labels`` is expected to come from
    :func:`dg.lazynwb_obstore.read_vlen_string_column` on
    ``/units/structure_layer``.  ``electrodes`` contains scalar-only fields;
    no spike or waveform array belongs in any input.
    """

    for name, frame in (
        ("unit_qc", unit_qc),
        ("structure_labels", structure_labels),
        ("electrodes", electrodes),
        ("ontology_structures", ontology_structures),
    ):
        if not isinstance(frame, pl.DataFrame):
            raise TypeError(f"{name} must be a polars DataFrame")
    _require_columns(
        unit_qc,
        {
            "_nwb_path",
            "_table_index",
            "id",
            "peak_channel_id",
            "subject_id",
            "ecephys_session_id",
            "well_isolated",
            "axis_weight",
            "unit_inclusion_status",
            "engaged_block_consistency_status",
        },
        frame_name="unit_qc",
    )
    _require_columns(
        structure_labels,
        {"_nwb_path", "_table_index", "structure_layer"},
        frame_name="structure_labels",
    )
    _require_columns(
        electrodes,
        {"_nwb_path", "id", "x", "y", "z", "probe_id"},
        frame_name="electrodes",
    )
    _require_columns(
        ontology_structures,
        {
            "acronym",
            "structure_id",
            "structure_name",
            "major_division_id",
            "major_division_acronym",
            "major_division_name",
        },
        frame_name="ontology_structures",
    )
    _validate_unique(unit_qc, ("_nwb_path", "_table_index"), frame_name="unit_qc")
    _validate_unique(
        structure_labels,
        ("_nwb_path", "_table_index"),
        frame_name="structure_labels",
    )
    _validate_unique(electrodes, ("_nwb_path", "id"), frame_name="electrodes")
    if structure_labels.height != unit_qc.height:
        raise ValueError("structure_labels must cover every unit row exactly")

    region_contract = parent_region_contract()
    structures = ontology_structures.select(
        pl.col("acronym").cast(pl.String).alias("structure_acronym"),
        pl.col("structure_id").cast(pl.Int64),
        pl.col("structure_name").cast(pl.String),
        pl.col("major_division_id").cast(pl.Int64).alias("parent_region_id"),
        pl.col("major_division_acronym").cast(pl.String).alias("ontology_parent_region_acronym"),
        pl.col("major_division_name").cast(pl.String).alias("ontology_parent_region_name"),
    )
    _validate_unique(structures, ("structure_acronym",), frame_name="ontology_structures")
    electrode_projection = electrodes.select(
        "_nwb_path",
        pl.col("id").cast(pl.Int64).alias("peak_channel_id"),
        pl.col("probe_id").cast(pl.Int64),
        pl.col("x").cast(pl.Float64).alias("anterior_posterior_ccf_coordinate"),
        pl.col("y").cast(pl.Float64).alias("dorsal_ventral_ccf_coordinate"),
        pl.col("z").cast(pl.Float64).alias("left_right_ccf_coordinate"),
    )
    labels = structure_labels.select(
        "_nwb_path",
        pl.col("_table_index").cast(pl.UInt32),
        pl.col("structure_layer").cast(pl.String).str.strip_chars().alias("structure_acronym"),
    )
    joined = (
        unit_qc.join(
            labels,
            on=["_nwb_path", "_table_index"],
            how="left",
            validate="1:1",
        )
        .join(
            electrode_projection,
            on=["_nwb_path", "peak_channel_id"],
            how="left",
            validate="m:1",
        )
        .join(structures, on="structure_acronym", how="left", validate="m:1")
        .join(region_contract, on="parent_region_id", how="left", validate="m:1")
    )
    conflict = joined.filter(
        pl.col("parent_region_id").is_not_null()
        & (
            (pl.col("parent_region_acronym") != pl.col("ontology_parent_region_acronym"))
            | (pl.col("parent_region_name") != pl.col("ontology_parent_region_name"))
        )
    )
    if conflict.height:
        raise RuntimeError("ontology parent mapping conflicts with the frozen D09 contract")
    result = joined.with_columns(
        pl.col("structure_id").is_not_null().alias("raw_structure_recognized"),
        pl.col("parent_region_id").is_not_null().alias("parent_region_mapped"),
        (
            pl.col("well_isolated")
            & pl.col("axis_weight").is_not_null()
            & pl.col("axis_weight").is_finite()
            & pl.col("parent_region_id").is_not_null()
        ).alias("anatomy_analysis_included"),
    ).with_columns(
        pl.when(~pl.col("well_isolated"))
        .then(pl.lit("excluded_isolation_qc"))
        .when(pl.col("axis_weight").is_null() | ~pl.col("axis_weight").is_finite())
        .then(pl.lit("excluded_missing_axis_weight"))
        .when(pl.col("structure_id").is_null())
        .then(pl.lit("excluded_unrecognized_structure"))
        .when(pl.col("parent_region_id").is_null())
        .then(pl.lit("excluded_outside_locked_parent_level"))
        .otherwise(pl.lit("included"))
        .alias("anatomy_inclusion_status"),
        pl.lit("Allen CCFv3 graph 1; official 12 major divisions; structure set 687527670").alias(
            "parent_region_mapping"
        ),
        pl.lit("isolation_only_preliminary_d04_not_applied").alias("unit_cohort"),
        pl.lit(True).alias("spike_arrays_intentionally_not_loaded"),
        pl.lit(False).alias("confirmation_accessed"),
    )
    _validate_unique(result, ("_nwb_path", "_table_index"), frame_name="unit anatomy")
    return result.select(
        "_nwb_path",
        "_table_index",
        pl.col("id").alias("unit_id"),
        "subject_id",
        "ecephys_session_id",
        "peak_channel_id",
        "probe_id",
        "structure_acronym",
        "structure_id",
        "structure_name",
        "raw_structure_recognized",
        "parent_region_id",
        "parent_region_order",
        "parent_region_acronym",
        "parent_region_name",
        "parent_region_mapped",
        "anterior_posterior_ccf_coordinate",
        "dorsal_ventral_ccf_coordinate",
        "left_right_ccf_coordinate",
        "well_isolated",
        "unit_inclusion_status",
        "engaged_block_consistency_status",
        "unit_cohort",
        "axis_weight",
        "anatomy_analysis_included",
        "anatomy_inclusion_status",
        "parent_region_mapping",
        "spike_arrays_intentionally_not_loaded",
        "confirmation_accessed",
    ).sort("subject_id", "ecephys_session_id", "_table_index")


def analyze_anatomical_enrichment(
    unit_anatomy: pl.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    code_version: str = "unknown",
) -> AnatomicalEnrichmentTables:
    """Compute coverage, mouse effects, and the session-preserving null."""

    if not isinstance(unit_anatomy, pl.DataFrame):
        raise TypeError("unit_anatomy must be a polars DataFrame")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for name, value in (("n_bootstrap", n_bootstrap), ("n_permutations", n_permutations)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    _require_columns(
        unit_anatomy,
        {
            "_nwb_path",
            "_table_index",
            "subject_id",
            "ecephys_session_id",
            "probe_id",
            "axis_weight",
            "well_isolated",
            "parent_region_id",
            "parent_region_acronym",
            "parent_region_name",
            "anatomy_analysis_included",
        },
        frame_name="unit_anatomy",
    )
    _validate_unique(unit_anatomy, ("_nwb_path", "_table_index"), frame_name="unit_anatomy")
    base = unit_anatomy.filter(
        pl.col("well_isolated")
        & pl.col("axis_weight").is_not_null()
        & pl.col("axis_weight").is_finite()
    )
    if base.is_empty():
        raise ValueError("unit_anatomy contains no finite well-isolated axis weights")
    if base.select("subject_id", "_nwb_path").null_count().sum_horizontal().item():
        raise ValueError("eligible units contain null mouse or session identity")

    unit_scores = _standardize_session_loadings(base)
    coverage = _region_coverage(unit_scores)
    session_effects = _session_region_effects(unit_scores)
    mouse_effects = _mouse_region_effects(session_effects)
    permutation_null = _session_preserving_permutation_null(
        unit_scores,
        coverage,
        n_permutations=n_permutations,
        seed=seed,
    )
    statistics = _region_statistics(
        mouse_effects,
        coverage,
        permutation_null,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        seed=seed,
        code_version=code_version,
    )
    return AnatomicalEnrichmentTables(
        unit_scores=unit_scores,
        region_coverage=coverage,
        session_effects=session_effects,
        mouse_effects=mouse_effects,
        statistics=statistics,
        permutation_null=permutation_null,
    )


def _standardize_session_loadings(frame: pl.DataFrame) -> pl.DataFrame:
    signed_mean = pl.col("axis_weight").mean().over("_nwb_path")
    signed_scale = pl.col("axis_weight").std(ddof=0).over("_nwb_path")
    absolute = pl.col("axis_weight").abs()
    absolute_mean = absolute.mean().over("_nwb_path")
    absolute_scale = absolute.std(ddof=0).over("_nwb_path")
    scored = frame.with_columns(
        signed_mean.alias("session_axis_weight_mean"),
        signed_scale.alias("session_axis_weight_sd"),
        absolute.alias("absolute_axis_weight"),
        absolute_mean.alias("session_absolute_weight_mean"),
        absolute_scale.alias("session_absolute_weight_sd"),
    )
    invalid = scored.filter(
        (pl.col("session_axis_weight_sd") <= 0)
        | ~pl.col("session_axis_weight_sd").is_finite()
        | (pl.col("session_absolute_weight_sd") <= 0)
        | ~pl.col("session_absolute_weight_sd").is_finite()
    )
    if invalid.height:
        sessions = invalid.get_column("_nwb_path").unique().to_list()
        raise ValueError(f"axis weights cannot be standardized in sessions: {sessions[:3]}")
    return scored.with_columns(
        (
            (pl.col("axis_weight") - pl.col("session_axis_weight_mean"))
            / pl.col("session_axis_weight_sd")
        ).alias("signed_loading_z"),
        (
            (pl.col("absolute_axis_weight") - pl.col("session_absolute_weight_mean"))
            / pl.col("session_absolute_weight_sd")
        ).alias("absolute_loading_z"),
    ).sort("subject_id", "ecephys_session_id", "_table_index")


def _region_coverage(unit_scores: pl.DataFrame) -> pl.DataFrame:
    mapped = unit_scores.filter(pl.col("parent_region_id").is_not_null())
    mouse_counts = mapped.group_by("parent_region_id", "subject_id").agg(
        pl.len().cast(pl.Int64).alias("n_units_in_mouse"),
        pl.col("_nwb_path").n_unique().cast(pl.Int64).alias("n_sessions_in_mouse"),
    )
    summary = mapped.group_by("parent_region_id").agg(
        pl.col("subject_id").n_unique().cast(pl.Int64).alias("n_mice"),
        pl.col("_nwb_path").n_unique().cast(pl.Int64).alias("n_sessions"),
        pl.concat_str(
            pl.col("_nwb_path"),
            pl.lit(":"),
            pl.col("probe_id").cast(pl.String),
            ignore_nulls=False,
        )
        .drop_nulls()
        .n_unique()
        .cast(pl.Int64)
        .alias("n_probes"),
        pl.len().cast(pl.Int64).alias("n_units"),
    )
    medians = mouse_counts.group_by("parent_region_id").agg(
        pl.col("n_units_in_mouse").median().alias("median_units_per_represented_mouse"),
        pl.col("n_sessions_in_mouse").median().alias("median_sessions_per_represented_mouse"),
    )
    all_units = unit_scores.height
    all_sessions = unit_scores.get_column("_nwb_path").n_unique()
    all_mice = unit_scores.get_column("subject_id").n_unique()
    return (
        parent_region_contract()
        .join(summary, on="parent_region_id", how="left", validate="1:1")
        .join(medians, on="parent_region_id", how="left", validate="1:1")
        .with_columns(
            pl.col("n_mice").fill_null(0),
            pl.col("n_sessions").fill_null(0),
            pl.col("n_probes").fill_null(0),
            pl.col("n_units").fill_null(0),
            pl.col("median_units_per_represented_mouse").fill_null(0.0),
            pl.col("median_sessions_per_represented_mouse").fill_null(0.0),
            pl.lit(all_mice).cast(pl.Int64).alias("eligible_denominator_mice"),
            pl.lit(all_sessions).cast(pl.Int64).alias("eligible_denominator_sessions"),
            pl.lit(all_units).cast(pl.Int64).alias("eligible_denominator_units"),
        )
        .with_columns(
            (
                (pl.col("n_mice") >= MIN_REGION_MICE)
                & (pl.col("n_sessions") >= MIN_REGION_SESSIONS)
                & (pl.col("median_units_per_represented_mouse") >= MIN_MEDIAN_UNITS_PER_MOUSE)
            ).alias("coverage_eligible"),
            pl.lit(MIN_REGION_MICE).alias("minimum_mice"),
            pl.lit(MIN_REGION_SESSIONS).alias("minimum_sessions"),
            pl.lit(MIN_MEDIAN_UNITS_PER_MOUSE).alias("minimum_median_units_per_mouse"),
            pl.lit("well-isolated finite-weight units; D04 not yet applied").alias(
                "coverage_population"
            ),
        )
        .sort("parent_region_order")
    )


def _session_region_effects(unit_scores: pl.DataFrame) -> pl.DataFrame:
    mapped = unit_scores.filter(pl.col("parent_region_id").is_not_null())
    id_columns = (
        "subject_id",
        "_nwb_path",
        "ecephys_session_id",
        "parent_region_id",
        "parent_region_acronym",
        "parent_region_name",
    )
    common = mapped.group_by(*id_columns).agg(
        pl.len().cast(pl.Int64).alias("n_units"),
        pl.col("probe_id").drop_nulls().n_unique().cast(pl.Int64).alias("n_probes"),
        pl.col("signed_loading_z").mean().alias("signed_loading"),
        pl.col("absolute_loading_z").mean().alias("absolute_loading"),
    )
    return (
        common.unpivot(
            index=[*id_columns, "n_units", "n_probes"],
            on=["signed_loading", "absolute_loading"],
            variable_name="score_type",
            value_name="session_effect",
        )
        .with_columns(
            pl.col("score_type").replace_strict(dict(SCORE_TYPES)).alias("score_type_label")
        )
        .sort("score_type", "parent_region_id", "subject_id", "ecephys_session_id")
    )


def _mouse_region_effects(session_effects: pl.DataFrame) -> pl.DataFrame:
    return (
        session_effects.group_by(
            "subject_id",
            "parent_region_id",
            "parent_region_acronym",
            "parent_region_name",
            "score_type",
            "score_type_label",
        )
        .agg(
            pl.col("session_effect").mean().alias("mouse_effect"),
            pl.col("_nwb_path").n_unique().cast(pl.Int64).alias("n_sessions"),
            pl.col("n_probes").sum().cast(pl.Int64).alias("n_session_probes"),
            pl.col("n_units").sum().cast(pl.Int64).alias("n_units"),
        )
        .sort("score_type", "parent_region_id", "subject_id")
    )


def _session_preserving_permutation_null(
    unit_scores: pl.DataFrame,
    coverage: pl.DataFrame,
    *,
    n_permutations: int,
    seed: int,
) -> pl.DataFrame:
    eligible_ids = (
        coverage.filter(pl.col("coverage_eligible")).get_column("parent_region_id").to_list()
    )
    if not eligible_ids:
        return pl.DataFrame(
            schema={
                "score_type": pl.String,
                "parent_region_id": pl.Int64,
                "permutation_index": pl.Int64,
                "null_estimate": pl.Float64,
                "permutation_seed": pl.Int64,
            }
        )
    parent_ids = [row[0] for row in PARENT_REGIONS]
    region_index = {identifier: index for index, identifier in enumerate(parent_ids)}
    mouse_ids = sorted(unit_scores.get_column("subject_id").unique().to_list())
    mouse_index = {mouse: index for index, mouse in enumerate(mouse_ids)}
    n_regions = len(parent_ids)
    n_mice = len(mouse_ids)
    rng = np.random.default_rng(seed)
    output: list[pl.DataFrame] = []

    for score_type, _ in SCORE_TYPES:
        value_column = f"{score_type}_z"
        sums = np.zeros((n_permutations, n_mice, n_regions), dtype=np.float64)
        session_counts = np.zeros((n_mice, n_regions), dtype=np.int64)
        sessions = unit_scores.partition_by("_nwb_path", as_dict=False, maintain_order=True)
        for session in sessions:
            mouse_values = session.get_column("subject_id").unique().to_list()
            if len(mouse_values) != 1:
                raise ValueError("each session must belong to exactly one mouse")
            mouse_position = mouse_index[mouse_values[0]]
            values = session.get_column(value_column).to_numpy()
            codes = np.array(
                [
                    region_index.get(value, -1) if value is not None else -1
                    for value in session.get_column("parent_region_id")
                ],
                dtype=np.int64,
            )
            counts = np.bincount(codes[codes >= 0], minlength=n_regions)
            represented = counts > 0
            session_counts[mouse_position, represented] += 1
            for permutation_index in range(n_permutations):
                permuted = rng.permutation(values)
                region_sums = np.bincount(
                    codes[codes >= 0],
                    weights=permuted[codes >= 0],
                    minlength=n_regions,
                )
                sums[permutation_index, mouse_position, represented] += (
                    region_sums[represented] / counts[represented]
                )
        with np.errstate(invalid="ignore", divide="ignore"):
            mouse_null = sums / session_counts[np.newaxis, :, :]
        for parent_region_id in eligible_ids:
            region_position = region_index[parent_region_id]
            represented_mice = session_counts[:, region_position] > 0
            null_values = mouse_null[:, represented_mice, region_position].mean(axis=1)
            output.append(
                pl.DataFrame(
                    {
                        "score_type": [score_type] * n_permutations,
                        "parent_region_id": [parent_region_id] * n_permutations,
                        "permutation_index": np.arange(n_permutations, dtype=np.int64),
                        "null_estimate": null_values,
                        "permutation_seed": [seed] * n_permutations,
                    }
                )
            )
    return pl.concat(output).sort("score_type", "parent_region_id", "permutation_index")


def _region_statistics(
    mouse_effects: pl.DataFrame,
    coverage: pl.DataFrame,
    permutation_null: pl.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int,
    code_version: str,
) -> pl.DataFrame:
    eligible = coverage.filter(pl.col("coverage_eligible"))
    if eligible.is_empty():
        return dg.statistics.empty_statistics_table()
    rows: list[dict[str, Any]] = []
    row_index = 0
    for score_type, score_label in SCORE_TYPES:
        for coverage_row in eligible.sort("parent_region_order").iter_rows(named=True):
            region_id = coverage_row["parent_region_id"]
            region_acronym = coverage_row["parent_region_acronym"]
            mice = mouse_effects.filter(
                (pl.col("score_type") == score_type) & (pl.col("parent_region_id") == region_id)
            )
            if mice.height != coverage_row["n_mice"]:
                raise RuntimeError("mouse-effect and coverage denominators disagree")
            bootstrap = dg.statistics.bootstrap_mouse_mean(
                mice,
                seed=seed + row_index + 1,
                mouse_column="subject_id",
                value_column="mouse_effect",
                n_resamples=n_bootstrap,
            )
            null = (
                permutation_null.filter(
                    (pl.col("score_type") == score_type) & (pl.col("parent_region_id") == region_id)
                )
                .get_column("null_estimate")
                .to_numpy()
            )
            if null.size != n_permutations or not np.isfinite(null).all():
                raise RuntimeError("permutation null is incomplete or nonfinite")
            observed = bootstrap.estimate
            tolerance = np.finfo(float).eps * max(1.0, abs(observed)) * 8
            p_value = (1 + np.count_nonzero(np.abs(null) >= abs(observed) - tolerance)) / (
                n_permutations + 1
            )
            rows.append(
                {
                    "analysis_id": ANALYSIS_ID,
                    "result_id": f"{score_type}__{region_acronym}",
                    "contrast_id": (
                        f"{score_type}_region_minus_session_population__{region_acronym}"
                    ),
                    "hypothesis": (
                        f"{score_label} differs in {coverage_row['parent_region_name']} "
                        "relative to the same sessions' full well-isolated population"
                    ),
                    "dandiset_version": dg.data.DANDISET_VERSION,
                    "code_version": code_version,
                    "seed": seed,
                    "analysis_tier": ANALYSIS_TIER,
                    "inclusion_definition": (
                        "behavior-eligible discovery sessions; quality good, isi violations "
                        "<0.5, amplitude cutoff <0.1; finite Figure 2 axis weight; Allen CCFv3 "
                        "official-major-division mapping; D04 not yet applied"
                    ),
                    "missingness_stratum": (
                        "region passes locked 5-mouse/8-session/median-10-unit coverage"
                    ),
                    "estimate": observed,
                    "scale": "within-session standardized decoder coefficient",
                    "ci_low": bootstrap.ci_low,
                    "ci_high": bootstrap.ci_high,
                    "confidence_level": bootstrap.confidence_level,
                    "ci_method": "mouse percentile bootstrap",
                    "test_statistic": abs(observed),
                    "test_method": "two-sided session-preserving anatomy-label permutation",
                    "p_value": float(p_value),
                    "adjusted_p_value": None,
                    "adjustment_method": None,
                    "multiplicity_family": "anatomical_axis_region_screen",
                    "sidedness": "two-sided",
                    "n_mice": int(coverage_row["n_mice"]),
                    "n_sessions": int(coverage_row["n_sessions"]),
                    "n_probes": int(coverage_row["n_probes"]),
                    "n_units": int(coverage_row["n_units"]),
                    "n_trials": None,
                    "aggregation": (
                        "region mean within session; equal sessions within mouse; equal mice"
                    ),
                    "bootstrap_id": f"mouse_percentile_{n_bootstrap}_seed_{seed + row_index + 1}",
                    "model_formula": (
                        "axis_weight z-scored within session; absolute magnitude "
                        "analyzed separately"
                    ),
                    "cv_grouping": "inherits Figure 2 nested contiguous E1-vs-NR folds",
                    "status": "pass",
                    "reason": None,
                }
            )
            row_index += 1
    normalized = dg.statistics.normalize_statistics_table(pl.DataFrame(rows))
    return dg.statistics.add_holm_adjustment(normalized)


def add_screen_annotations(
    statistics: pl.DataFrame,
    coverage: pl.DataFrame,
) -> pl.DataFrame:
    """Add transparent discovery rank and conservative nomination flags."""

    if statistics.is_empty():
        return statistics.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("discovery_rank"),
            pl.lit(False).alias("screen_lead"),
            pl.lit(False).alias("nominated_for_confirmation"),
        )
    lookup = coverage.select(
        "parent_region_id",
        "parent_region_order",
        "parent_region_acronym",
        "parent_region_name",
    )
    annotated = statistics.with_columns(
        pl.col("result_id").str.split("__").list.get(0).alias("score_type"),
        pl.col("result_id").str.split("__").list.get(1).alias("parent_region_acronym"),
    ).join(lookup, on="parent_region_acronym", how="left", validate="m:1")
    absolute = annotated.filter(pl.col("score_type") == "absolute_loading").sort(
        pl.col("estimate"), descending=True
    )
    rank = absolute.with_row_index("discovery_rank", offset=1).select(
        "parent_region_id", "discovery_rank"
    )
    return (
        annotated.join(rank, on="parent_region_id", how="left", validate="m:1")
        .with_columns(
            ((pl.col("score_type") == "absolute_loading") & (pl.col("discovery_rank") == 1)).alias(
                "screen_lead"
            ),
            (
                (pl.col("score_type") == "absolute_loading")
                & (pl.col("estimate") > 0)
                & (pl.col("ci_low") > 0)
                & (pl.col("adjusted_p_value") < 0.05)
            ).alias("nominated_for_confirmation"),
        )
        .sort("score_type", "parent_region_order")
    )


def _require_columns(frame: pl.DataFrame, columns: set[str], *, frame_name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {sorted(missing)}")


def _validate_unique(
    frame: pl.DataFrame,
    columns: tuple[str, ...],
    *,
    frame_name: str,
) -> None:
    if frame.select(*columns).n_unique() != frame.height:
        raise ValueError(f"{frame_name} contains duplicate rows for {columns}")
