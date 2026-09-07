"""Coverage-only neural and stimulus inventories for the frozen Dandiset.

This module deliberately does not read spike-time arrays or derive neural
responses.  It builds the Milestone 0 denominators needed to decide whether
later analyses are feasible, while preserving raw NWB provenance and reporting
missing values rather than filling them.
"""

from __future__ import annotations

import collections.abc
from typing import Any

import lazynwb
import polars as pl

import dg.data
import dg.quality

SOURCE_COLUMN = lazynwb.NWB_PATH_COLUMN_NAME
TABLE_PATH_COLUMN = lazynwb.TABLE_PATH_COLUMN_NAME
TABLE_INDEX_COLUMN = lazynwb.TABLE_INDEX_COLUMN_NAME

UNIT_REQUIRED_COLUMNS = (
    "id",
    "peak_channel_id",
    "isi_violations",
    "amplitude_cutoff",
    "quality",
)
UNIT_REQUIRED_DTYPES: dict[str, pl.DataType] = {
    "id": pl.Int64,
    "peak_channel_id": pl.Int64,
    "isi_violations": pl.Float64,
    "amplitude_cutoff": pl.Float64,
    "quality": pl.String,
}
UNIT_OPTIONAL_SCALAR_DTYPES: dict[str, pl.DataType] = {
    "presence_ratio": pl.Float64,
    "snr": pl.Float64,
    "amplitude_cv_median": pl.Float64,
    "amplitude_cv_range": pl.Float64,
    "amplitude_median": pl.Float64,
    "waveform_duration": pl.Float64,
    "waveform_halfwidth": pl.Float64,
    "waveform_repolarization_slope": pl.Float64,
    "waveform_recovery_slope": pl.Float64,
    "waveform_spread": pl.Float64,
    "structure_layer": pl.String,
}

ELECTRODE_REQUIRED_COLUMNS = ("id", "location", "x", "y", "z", "probe_id")
ELECTRODE_REQUIRED_DTYPES: dict[str, pl.DataType] = {
    "id": pl.Int64,
    "location": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
    "probe_id": pl.Int64,
}
ELECTRODE_OPTIONAL_DTYPES: dict[str, pl.DataType] = {
    "probe_channel_number": pl.Int64,
    "probe_horizontal_position": pl.Float64,
    "probe_vertical_position": pl.Float64,
    "valid_data": pl.Boolean,
}

PRESENTATION_REQUIRED_COLUMNS = (
    "id",
    "start_time",
    "stop_time",
    "start_frame",
    "image_name",
    "is_change",
    "active",
    "is_image_novel",
    "flashes_since_change",
    "stimulus_block",
    "rewarded",
    "omitted",
)
PRESENTATION_REQUIRED_DTYPES: dict[str, pl.DataType] = {
    "id": pl.Int64,
    "start_time": pl.Float64,
    "stop_time": pl.Float64,
    "start_frame": pl.Int64,
    "image_name": pl.String,
    "is_change": pl.Boolean,
    "active": pl.Boolean,
    "is_image_novel": pl.Boolean,
    "flashes_since_change": pl.Float64,
    "stimulus_block": pl.Int64,
    "rewarded": pl.Boolean,
    "omitted": pl.Boolean,
}
PRESENTATION_OPTIONAL_DTYPES: dict[str, pl.DataType] = {
    "stimulus_name": pl.String,
}

TRIAL_CLOCK_COLUMNS = (
    "id",
    "start_time",
    "stop_time",
    "change_time",
    "change_frame",
    "initial_image_name",
    "change_image_name",
    "aborted",
    "go",
    "catch",
    "auto_rewarded",
    "is_change",
    "is_sham_change",
    "no_reward_epoch",
    "omitted_reward",
)

SESSION_METADATA_COLUMNS = (
    "dandiset_id",
    "dandiset_version",
    "asset_id",
    "path",
    "subject_id",
    "ecephys_session_id",
    "recording_day",
    "session_number",
    "date_of_acquisition",
    "companion_unit_count",
    "novel_image_id",
)


def scan_unit_inventory(session_inventory: pl.DataFrame) -> pl.LazyFrame:
    """Lazily scan scalar unit/QC and peak-electrode fields for every session.

    The projection is resolved before collection, so ``spike_times``,
    ``spike_amplitudes``, and waveform arrays are never read.  Optional scalar
    QC fields are retained when present and represented by typed nulls when the
    published schema does not contain them.
    """

    _validate_session_inventory(session_inventory)
    sources = session_inventory.get_column(SOURCE_COLUMN).to_list()
    units = _scan_projected_table(
        sources,
        dg.data.UNITS_PATH,
        required_columns=UNIT_REQUIRED_COLUMNS,
        required_dtypes=UNIT_REQUIRED_DTYPES,
        optional_dtypes=UNIT_OPTIONAL_SCALAR_DTYPES,
    )
    electrodes = _scan_projected_table(
        sources,
        dg.data.ELECTRODES_PATH,
        required_columns=ELECTRODE_REQUIRED_COLUMNS,
        required_dtypes=ELECTRODE_REQUIRED_DTYPES,
        optional_dtypes=ELECTRODE_OPTIONAL_DTYPES,
    )
    located = dg.data.join_unit_locations(units, electrodes)
    flagged = dg.quality.add_well_isolated_unit_flag(located)
    metadata_columns = [
        SOURCE_COLUMN,
        *(column for column in SESSION_METADATA_COLUMNS if column in session_inventory.columns),
    ]
    metadata = session_inventory.select(*metadata_columns).lazy()
    return (
        flagged.join(metadata, on=SOURCE_COLUMN, how="left", validate="m:1")
        .with_columns(
            pl.col("id").alias("unit_id"),
            pl.concat_str(
                pl.col("asset_id"),
                pl.lit(":"),
                pl.col("id").cast(pl.String),
            ).alias("unit_key"),
            pl.lit(True).alias("spike_arrays_intentionally_not_loaded"),
        )
        .rename(
            {
                TABLE_PATH_COLUMN: "unit_table_path",
                TABLE_INDEX_COLUMN: "unit_table_index",
            }
        )
        .select(
            "unit_key",
            "unit_id",
            "id",
            "unit_table_path",
            "unit_table_index",
            SOURCE_COLUMN,
            *(column for column in SESSION_METADATA_COLUMNS if column in session_inventory.columns),
            "peak_channel_id",
            *UNIT_OPTIONAL_SCALAR_DTYPES,
            "isi_violations",
            "amplitude_cutoff",
            "quality",
            "well_isolated",
            "unit_quality_exclusion_reasons",
            "structure_acronym",
            "anterior_posterior_ccf_coordinate",
            "dorsal_ventral_ccf_coordinate",
            "left_right_ccf_coordinate",
            "probe_id",
            "probe_channel_number",
            "probe_horizontal_position",
            "probe_vertical_position",
            "valid_data",
            "electrode_table_path",
            "electrode_table_index",
            "spike_arrays_intentionally_not_loaded",
        )
    )


def validate_unit_inventory(
    unit_inventory: pl.DataFrame,
    session_inventory: pl.DataFrame,
) -> None:
    """Fail on lost/duplicated unit identities or failed session reconciliation."""

    required = {
        "unit_key",
        "unit_id",
        "unit_table_path",
        "unit_table_index",
        SOURCE_COLUMN,
        "asset_id",
        "subject_id",
        "peak_channel_id",
        "electrode_table_index",
        "well_isolated",
        "unit_quality_exclusion_reasons",
    }
    _require_columns(unit_inventory, required, frame_name="unit_inventory")
    _validate_session_inventory(session_inventory)
    if unit_inventory.is_empty():
        raise ValueError("unit_inventory contains no units")
    if unit_inventory.get_column("unit_key").null_count():
        raise ValueError("unit_inventory contains null unit keys")
    if unit_inventory.get_column("unit_key").n_unique() != unit_inventory.height:
        raise ValueError("unit_inventory contains duplicate session-qualified unit keys")
    if (
        unit_inventory.select(
            "unit_id",
            "unit_table_path",
            "unit_table_index",
            SOURCE_COLUMN,
        )
        .null_count()
        .sum_horizontal()
        .item()
    ):
        raise ValueError("unit_inventory contains null unit-table identity/provenance")
    if unit_inventory.filter(
        ~pl.col("unit_table_path").is_in([dg.data.UNITS_PATH, dg.data.UNITS_PATH.removeprefix("/")])
    ).height:
        raise ValueError("unit_inventory contains an unexpected NWB unit table path")
    if unit_inventory.select(SOURCE_COLUMN, "unit_table_index").n_unique() != unit_inventory.height:
        raise ValueError("unit_inventory contains duplicate source/table-index provenance")
    if unit_inventory.select("asset_id", "subject_id").null_count().row(0) != (0, 0):
        raise ValueError("unit_inventory contains units without reconciled session metadata")
    if unit_inventory.get_column("well_isolated").null_count():
        raise ValueError("unit_inventory contains null well-isolated flags")
    unmatched_peak = unit_inventory.filter(
        pl.col("peak_channel_id").is_null() | pl.col("electrode_table_index").is_null()
    )
    if unmatched_peak.height:
        raise ValueError(
            "unit_inventory contains "
            f"{unmatched_peak.height} units without a matched peak electrode"
        )

    expected_sources = set(session_inventory.get_column(SOURCE_COLUMN).to_list())
    observed_sources = set(unit_inventory.get_column(SOURCE_COLUMN).to_list())
    if observed_sources != expected_sources:
        missing = sorted(expected_sources.difference(observed_sources))
        unexpected = sorted(observed_sources.difference(expected_sources))
        raise ValueError(
            "unit_inventory source coverage does not match the frozen session inventory; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def summarize_unit_session_reconciliation(
    unit_inventory: pl.DataFrame,
    session_inventory: pl.DataFrame,
) -> pl.DataFrame:
    """Compare observed NWB unit rows with companion counts without excluding either."""

    validate_unit_inventory(unit_inventory, session_inventory)
    metadata_columns = [
        SOURCE_COLUMN,
        *(column for column in SESSION_METADATA_COLUMNS if column in session_inventory.columns),
    ]
    observed = unit_inventory.group_by(SOURCE_COLUMN).agg(
        pl.len().cast(pl.Int64).alias("n_nwb_units"),
        pl.col("unit_id").n_unique().cast(pl.Int64).alias("n_unique_nwb_unit_ids"),
        pl.col("well_isolated").sum().cast(pl.Int64).alias("n_well_isolated_units"),
        pl.col("peak_channel_id").is_null().sum().cast(pl.Int64).alias("n_missing_peak_channels"),
        pl.col("electrode_table_index")
        .is_null()
        .sum()
        .cast(pl.Int64)
        .alias("n_unmatched_peak_electrodes"),
        pl.col("structure_acronym")
        .is_null()
        .sum()
        .cast(pl.Int64)
        .alias("n_missing_structure_acronyms"),
        pl.col("probe_id").is_null().sum().cast(pl.Int64).alias("n_missing_probe_ids"),
    )
    reconciled = session_inventory.select(*metadata_columns).join(
        observed,
        on=SOURCE_COLUMN,
        how="left",
        validate="1:1",
    )
    if "companion_unit_count" not in reconciled.columns:
        return reconciled.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("companion_unit_count"),
            pl.lit(None, dtype=pl.Int64).alias("nwb_minus_companion_units"),
            pl.lit(None, dtype=pl.Boolean).alias("companion_unit_count_agrees"),
        ).sort("subject_id", SOURCE_COLUMN)
    return reconciled.with_columns(
        (pl.col("n_nwb_units") - pl.col("companion_unit_count")).alias("nwb_minus_companion_units"),
        (pl.col("n_nwb_units") == pl.col("companion_unit_count")).alias(
            "companion_unit_count_agrees"
        ),
    ).sort("subject_id", SOURCE_COLUMN)


def summarize_unit_anatomy_coverage(unit_inventory: pl.DataFrame) -> pl.DataFrame:
    """Summarize raw recorded anatomy coverage without defining a parent ontology."""

    required = {
        "structure_acronym",
        "subject_id",
        SOURCE_COLUMN,
        "probe_id",
        "unit_key",
        "well_isolated",
    }
    _require_columns(unit_inventory, required, frame_name="unit_inventory")
    return (
        unit_inventory.with_columns(
            pl.col("structure_acronym").fill_null("[missing]").alias("structure_acronym")
        )
        .group_by("structure_acronym")
        .agg(
            pl.col("subject_id").n_unique().cast(pl.Int64).alias("n_mice"),
            pl.col(SOURCE_COLUMN).n_unique().cast(pl.Int64).alias("n_sessions"),
            pl.col("probe_id").drop_nulls().n_unique().cast(pl.Int64).alias("n_probes"),
            pl.col("unit_key").n_unique().cast(pl.Int64).alias("n_units"),
            pl.col("well_isolated").sum().cast(pl.Int64).alias("n_well_isolated_units"),
        )
        .with_columns(
            (pl.col("n_well_isolated_units") / pl.col("n_units")).alias("well_isolated_fraction"),
            pl.lit("raw_allen_acronym_no_parent_mapping").alias("anatomy_scope"),
            pl.lit("recorded_sampling_denominator_not_biological_abundance").alias(
                "interpretation_scope"
            ),
        )
        .sort("structure_acronym")
    )


def summarize_unit_session_anatomy_coverage(unit_inventory: pl.DataFrame) -> pl.DataFrame:
    """Preserve session/mouse sampling by raw acronym for outcome-blind splitting."""

    required = {
        "structure_acronym",
        "subject_id",
        SOURCE_COLUMN,
        "probe_id",
        "unit_key",
        "well_isolated",
    }
    _require_columns(unit_inventory, required, frame_name="unit_inventory")
    identity = ["subject_id", SOURCE_COLUMN]
    identity.extend(
        column
        for column in ("ecephys_session_id", "recording_day", "session_number")
        if column in unit_inventory.columns
    )
    return (
        unit_inventory.with_columns(
            pl.col("structure_acronym").fill_null("[missing]").alias("structure_acronym")
        )
        .group_by(*identity, "structure_acronym")
        .agg(
            pl.col("probe_id").drop_nulls().n_unique().cast(pl.Int64).alias("n_probes"),
            pl.col("unit_key").n_unique().cast(pl.Int64).alias("n_units"),
            pl.col("well_isolated").sum().cast(pl.Int64).alias("n_well_isolated_units"),
        )
        .with_columns(
            (pl.col("n_well_isolated_units") / pl.col("n_units")).alias("well_isolated_fraction"),
            pl.lit("raw_allen_acronym_no_parent_mapping").alias("anatomy_scope"),
            pl.lit("coverage_only_no_neural_activity").alias("analysis_scope"),
        )
        .sort("subject_id", SOURCE_COLUMN, "structure_acronym")
    )


def resolve_task_presentation_paths(
    session_inventory: pl.DataFrame,
    schema_audit: pl.DataFrame,
) -> pl.DataFrame:
    """Return exactly one schema-audited task-image table path per session."""

    _validate_session_inventory(session_inventory)
    required = {
        SOURCE_COLUMN,
        "n_image_presentation_candidates",
        "image_presentation_paths",
        "audit_error",
    }
    _require_columns(schema_audit, required, frame_name="schema_audit")
    mapping = schema_audit.select(*required).unique().sort(SOURCE_COLUMN)
    invalid = mapping.filter(
        pl.col("audit_error").is_not_null()
        | (pl.col("n_image_presentation_candidates") != 1)
        | pl.col("image_presentation_paths").is_null()
        | pl.col("image_presentation_paths").str.contains(";")
    )
    if invalid.height:
        raise ValueError(
            "schema audit does not resolve exactly one task-image table per session:\n"
            f"{invalid.head(5)}"
        )
    expected_sources = set(session_inventory.get_column(SOURCE_COLUMN).to_list())
    observed_sources = set(mapping.get_column(SOURCE_COLUMN).to_list())
    if observed_sources != expected_sources:
        raise ValueError("task-image path audit does not exactly cover the session inventory")
    return mapping.select(
        SOURCE_COLUMN,
        pl.col("image_presentation_paths").alias("task_presentation_path"),
    )


def scan_task_presentations(
    session_inventory: pl.DataFrame,
    schema_audit: pl.DataFrame,
) -> pl.LazyFrame:
    """Lazily scan projected task-presentation frames, times, and raw labels.

    The result is the row-level M0 stimulus inventory.  It deliberately omits
    array-valued columns and does not derive reward state from presentation
    ``active``, ``rewarded``, or ``stimulus_block`` labels.
    """

    paths = resolve_task_presentation_paths(session_inventory, schema_audit)
    frames: list[pl.LazyFrame] = []
    for table_path in paths.get_column("task_presentation_path").unique().sort():
        sources = paths.filter(pl.col("task_presentation_path") == table_path).get_column(
            SOURCE_COLUMN
        )
        frame = _scan_projected_table(
            sources.to_list(),
            table_path,
            required_columns=PRESENTATION_REQUIRED_COLUMNS,
            required_dtypes=PRESENTATION_REQUIRED_DTYPES,
            optional_dtypes=PRESENTATION_OPTIONAL_DTYPES,
        ).with_columns(pl.lit(table_path).alias("task_presentation_path"))
        frames.append(frame)
    if not frames:
        raise ValueError("no task-presentation tables were resolved")
    presentations = frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")
    metadata_columns = [
        SOURCE_COLUMN,
        *(column for column in SESSION_METADATA_COLUMNS if column in session_inventory.columns),
    ]
    joined = (
        dg.data.add_image_name_factors(presentations)
        .join(
            session_inventory.select(*metadata_columns).lazy(),
            on=SOURCE_COLUMN,
            how="left",
            validate="m:1",
        )
        .with_columns(
            pl.concat_str(
                pl.col("asset_id"),
                pl.lit(":"),
                pl.col(TABLE_INDEX_COLUMN).cast(pl.String),
            ).alias("presentation_row_key"),
            pl.col("novel_image_id")
            .str.extract(r"^(im\d+)_r-", group_index=1)
            .alias("metadata_novel_image_id"),
            pl.col("novel_image_id")
            .str.extract(r"_r-([0-9]+(?:\.[0-9]+)?)$", group_index=1)
            .cast(pl.Float64, strict=False)
            .alias("metadata_novel_relative_contrast"),
            pl.lit("raw_presentation_labels_not_reward_state").alias("presentation_label_scope"),
        )
    )
    ordered = joined.sort(SOURCE_COLUMN, "start_frame", TABLE_INDEX_COLUMN)
    return ordered.with_columns(
        pl.when(pl.col("image_id").is_not_null())
        .then(pl.col("image_id").cum_count().over(SOURCE_COLUMN, "image_id"))
        .otherwise(pl.lit(None, dtype=pl.UInt32))
        .alias("within_session_identity_exposure_index"),
        (pl.col("image_id") == pl.col("metadata_novel_image_id"))
        .fill_null(False)
        .alias("is_metadata_designated_novel_identity"),
    )


def validate_task_presentations(
    presentations: pl.DataFrame,
    session_inventory: pl.DataFrame,
) -> None:
    """Fail when task-presentation rows lose stable identity or session coverage."""

    required = {
        SOURCE_COLUMN,
        TABLE_INDEX_COLUMN,
        "id",
        "start_frame",
        "asset_id",
        "subject_id",
        "presentation_row_key",
        "task_presentation_path",
        "presentation_label_scope",
    }
    _require_columns(presentations, required, frame_name="presentations")
    _validate_session_inventory(session_inventory)
    if presentations.is_empty():
        raise ValueError("task presentations contain no rows")
    if (
        presentations.select(
            SOURCE_COLUMN,
            TABLE_PATH_COLUMN,
            TABLE_INDEX_COLUMN,
            "id",
        )
        .null_count()
        .sum_horizontal()
        .item()
    ):
        raise ValueError("task presentations contain null row identity/provenance")
    duplicate_keys = presentations.select(SOURCE_COLUMN, TABLE_INDEX_COLUMN).is_duplicated()
    if duplicate_keys.any():
        raise ValueError("task presentations contain duplicate source/table-row keys")
    if presentations.select(SOURCE_COLUMN, "id").n_unique() != presentations.height:
        raise ValueError("task presentation IDs must be unique within each session")
    if presentations.select("asset_id", "subject_id").null_count().row(0) != (0, 0):
        raise ValueError("task presentations contain unreconciled session metadata")
    if presentations.filter(
        pl.col(TABLE_PATH_COLUMN).str.strip_prefix("/")
        != pl.col("task_presentation_path").str.strip_prefix("/")
    ).height:
        raise ValueError("task presentations disagree with their schema-discovered table paths")
    expected_sources = set(session_inventory.get_column(SOURCE_COLUMN).to_list())
    observed_sources = set(presentations.get_column(SOURCE_COLUMN).to_list())
    if observed_sources != expected_sources:
        raise ValueError("task-presentation source coverage does not match session inventory")
    if presentations.get_column("presentation_row_key").n_unique() != presentations.height:
        raise ValueError("task presentations contain duplicate stable row keys")
    invalid_frames = presentations.filter(
        pl.col("start_frame").is_null() | (pl.col("start_frame") < 0)
    )
    if invalid_frames.height:
        raise ValueError("task presentations contain missing or negative start frames")
    duplicate_frames = presentations.select(SOURCE_COLUMN, "start_frame").is_duplicated()
    if duplicate_frames.any():
        raise ValueError("task presentations contain duplicate within-session start frames")


def scan_trial_inventory(
    session_inventory: pl.DataFrame,
    companion_reward_epochs: pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    """Scan one projected trial table and attach audited reward-block labels.

    Only scalar trial fields needed for provenance, interval screening, frame
    alignment, and stimulus-design coverage are requested.  Reward block comes
    from the NWB ``no_reward_epoch`` field with the pinned companion table used
    only where that NWB column is absent; it never comes from presentation
    ``active``, ``rewarded``, or ``stimulus_block`` labels.
    """

    _validate_session_inventory(session_inventory)
    trials = dg.data.scan_trials(
        session_inventory.get_column(SOURCE_COLUMN).to_list(),
        columns=TRIAL_CLOCK_COLUMNS,
    )
    resolved = dg.data.add_no_reward_epoch_from_companion(
        trials,
        session_inventory,
        companion_reward_epochs,
        source_column=SOURCE_COLUMN,
    )
    labeled = dg.quality.label_reward_blocks(resolved, source_column=SOURCE_COLUMN)
    existing = set(labeled.collect_schema().names())
    metadata_columns = [SOURCE_COLUMN]
    metadata_columns.extend(
        column
        for column in SESSION_METADATA_COLUMNS
        if column in session_inventory.columns and column not in existing
    )
    joined = labeled.join(
        session_inventory.select(*metadata_columns).lazy(),
        on=SOURCE_COLUMN,
        how="left",
        validate="m:1",
    )
    initial_name = pl.col("initial_image_name")
    change_name = pl.col("change_image_name")
    return joined.with_columns(
        pl.concat_str(
            pl.col("asset_id"),
            pl.lit(":"),
            pl.col(TABLE_INDEX_COLUMN).cast(pl.String),
        ).alias("trial_row_key"),
        initial_name.str.extract(r"^(im\d+)_r-", group_index=1).alias("initial_image_id"),
        initial_name.str.extract(r"_r-([0-9]+(?:\.[0-9]+)?)$", group_index=1)
        .cast(pl.Float64, strict=False)
        .alias("initial_image_relative_contrast"),
        change_name.str.extract(r"^(im\d+)_r-", group_index=1).alias("change_image_id"),
        change_name.str.extract(r"_r-([0-9]+(?:\.[0-9]+)?)$", group_index=1)
        .cast(pl.Float64, strict=False)
        .alias("change_image_relative_contrast"),
        pl.when(initial_name.is_null() | change_name.is_null())
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(initial_name != change_name)
        .alias("stimulus_token_changed"),
        pl.lit("audited_trial_no_reward_epoch_not_presentation_labels").alias(
            "reward_block_source"
        ),
        pl.lit("projected_scalar_trial_fields_no_lick_vectors").alias("trial_projection_scope"),
    ).with_columns(
        pl.when(pl.col("initial_image_id").is_null() | pl.col("change_image_id").is_null())
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(pl.col("initial_image_id") != pl.col("change_image_id"))
        .alias("identity_changed"),
        pl.when(
            pl.col("initial_image_relative_contrast").is_null()
            | pl.col("change_image_relative_contrast").is_null()
        )
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(
            pl.col("initial_image_relative_contrast") != pl.col("change_image_relative_contrast")
        )
        .alias("contrast_changed"),
    )


def validate_trial_inventory(
    trials: pl.DataFrame,
    session_inventory: pl.DataFrame,
) -> pl.DataFrame:
    """Fail closed on trial provenance, reward blocks, and change anchors.

    The returned one-row-per-session task-structure table is suitable for an
    audit artifact and avoids recomputing the structural diagnostics.
    """

    required = {
        SOURCE_COLUMN,
        TABLE_PATH_COLUMN,
        TABLE_INDEX_COLUMN,
        "id",
        "trial_row_key",
        "asset_id",
        "subject_id",
        "start_time",
        "stop_time",
        "change_time",
        "change_frame",
        "initial_image_name",
        "change_image_name",
        "aborted",
        "go",
        "catch",
        "auto_rewarded",
        "is_change",
        "is_sham_change",
        "stimulus_token_changed",
        "no_reward_epoch",
        "no_reward_epoch_conflict",
        "reward_block",
        "reward_block_source",
    }
    _require_columns(trials, required, frame_name="trial_inventory")
    _validate_session_inventory(session_inventory)
    if trials.is_empty():
        raise ValueError("trial inventory contains no rows")
    if (
        trials.select(SOURCE_COLUMN, TABLE_PATH_COLUMN, TABLE_INDEX_COLUMN, "id", "trial_row_key")
        .null_count()
        .sum_horizontal()
        .item()
    ):
        raise ValueError("trial inventory contains null row identity/provenance")
    if trials.select(SOURCE_COLUMN, TABLE_INDEX_COLUMN).n_unique() != trials.height:
        raise ValueError("trial inventory contains duplicate source/table-row provenance")
    if trials.select(SOURCE_COLUMN, "id").n_unique() != trials.height:
        raise ValueError("trial IDs must be unique within each session")
    if trials.get_column("trial_row_key").n_unique() != trials.height:
        raise ValueError("trial inventory contains duplicate stable row keys")
    if trials.filter(
        ~pl.col(TABLE_PATH_COLUMN).is_in(
            [dg.data.TRIALS_PATH, dg.data.TRIALS_PATH.removeprefix("/")]
        )
    ).height:
        raise ValueError("trial inventory contains an unexpected NWB table path")
    expected_sources = set(session_inventory.get_column(SOURCE_COLUMN).to_list())
    observed_sources = set(trials.get_column(SOURCE_COLUMN).to_list())
    if observed_sources != expected_sources:
        raise ValueError("trial inventory source coverage does not match session inventory")
    if trials.select("asset_id", "subject_id").null_count().row(0) != (0, 0):
        raise ValueError("trial inventory contains unreconciled session metadata")
    trial_type_columns = ("aborted", "go", "catch", "auto_rewarded", "is_change")
    if trials.select(*trial_type_columns).null_count().sum_horizontal().item():
        raise ValueError("trial inventory contains null task-type labels")

    change_time_missing = pl.col("change_time").is_null() | pl.col("change_time").is_nan()
    change_frame_missing = pl.col("change_frame").is_null() | pl.col("change_frame").is_nan()
    partial_anchors = trials.filter(change_time_missing != change_frame_missing)
    if partial_anchors.height:
        raise ValueError("trial inventory contains partial time/frame change anchors")
    bad_aborted = trials.filter(pl.col("aborted") & ~(change_time_missing & change_frame_missing))
    if bad_aborted.height:
        raise ValueError("aborted trials must have both change anchors missing")
    bad_completed = trials.filter(~pl.col("aborted") & (change_time_missing | change_frame_missing))
    if bad_completed.height:
        raise ValueError("completed trials must have both change anchors present")
    invalid_anchor_values = trials.filter(
        ~change_time_missing
        & (
            ~pl.col("change_time").is_finite()
            | ~pl.col("change_frame").is_finite()
            | (pl.col("change_frame") < 0)
            | (pl.col("change_frame") != pl.col("change_frame").round(0))
        )
    )
    if invalid_anchor_values.height:
        raise ValueError("trial inventory contains nonfinite, negative, or noninteger anchors")
    finite_anchors = trials.filter(~change_frame_missing)
    if finite_anchors.select(SOURCE_COLUMN, "change_frame").n_unique() != finite_anchors.height:
        raise ValueError("trial change_frame must be unique within each session")
    if trials.filter(
        ~change_time_missing
        & (pl.col("initial_image_name").is_null() | pl.col("change_image_name").is_null())
    ).height:
        raise ValueError("completed trial anchors contain missing stimulus tokens")
    if trials.get_column("no_reward_epoch").null_count():
        raise ValueError("trial inventory contains unresolved reward-epoch labels")
    if trials.get_column("no_reward_epoch_conflict").fill_null(False).any():
        raise ValueError("trial inventory contains reward-epoch source conflicts")

    label_consistency = (
        trials.group_by(SOURCE_COLUMN)
        .agg(
            (pl.col("go") & pl.col("catch")).sum().alias("n_go_catch_overlaps"),
            (pl.col("go") & pl.col("auto_rewarded")).sum().alias("n_go_auto_rewarded_overlaps"),
            (pl.col("catch") & pl.col("auto_rewarded"))
            .sum()
            .alias("n_catch_auto_rewarded_combinations_descriptive"),
            (pl.col("aborted") & (pl.col("go") | pl.col("catch") | pl.col("auto_rewarded")))
            .sum()
            .alias("n_aborted_task_type_overlaps"),
            (~pl.col("aborted") & ~(pl.col("go") | pl.col("catch") | pl.col("auto_rewarded")))
            .sum()
            .alias("n_completed_unclassified_trials"),
            (pl.col("go") & ~pl.col("is_change")).sum().alias("n_go_without_change_label"),
            (pl.col("catch") & pl.col("is_change"))
            .sum()
            .alias("n_catch_with_task_change_label_descriptive"),
            (pl.col("auto_rewarded") & ~pl.col("catch") & ~pl.col("is_change"))
            .sum()
            .alias("n_noncatch_auto_rewarded_without_change_label"),
            (pl.col("aborted") & pl.col("is_change")).sum().alias("n_aborted_with_change_label"),
            (pl.col("go") & ~pl.col("stimulus_token_changed"))
            .sum()
            .alias("n_go_without_physical_token_change"),
            (pl.col("catch") & pl.col("stimulus_token_changed"))
            .sum()
            .alias("n_catch_with_physical_token_change"),
            (pl.col("auto_rewarded") & ~pl.col("catch") & ~pl.col("stimulus_token_changed"))
            .sum()
            .alias("n_noncatch_auto_rewarded_without_physical_token_change"),
            (~pl.col("aborted") & pl.col("is_sham_change").is_null())
            .sum()
            .alias("n_nonaborted_missing_sham_change_labels_descriptive"),
            (
                ~pl.col("aborted")
                & pl.col("is_sham_change").is_not_null()
                & (pl.col("catch") != pl.col("is_sham_change"))
            )
            .sum()
            .alias("n_nonaborted_catch_sham_mismatches"),
        )
        .with_columns(
            pl.sum_horizontal(
                "n_go_catch_overlaps",
                "n_go_auto_rewarded_overlaps",
                "n_aborted_task_type_overlaps",
                "n_completed_unclassified_trials",
                "n_go_without_change_label",
                "n_noncatch_auto_rewarded_without_change_label",
                "n_aborted_with_change_label",
                "n_go_without_physical_token_change",
                "n_catch_with_physical_token_change",
                "n_noncatch_auto_rewarded_without_physical_token_change",
                "n_nonaborted_catch_sham_mismatches",
            )
            .eq(0)
            .alias("trial_label_consistency_pass")
        )
    )
    label_failures = label_consistency.filter(
        ~pl.col("trial_label_consistency_pass").fill_null(False)
    )
    if label_failures.height:
        raise ValueError(f"trial task-type label consistency failed:\n{label_failures.head(5)}")

    structure = dg.quality.summarize_task_structure(
        trials,
        source_column=SOURCE_COLUMN,
        table_index_column=TABLE_INDEX_COLUMN,
    ).collect()
    if structure.height != session_inventory.height:
        raise ValueError("trial task-structure audit does not exactly cover every session")
    failures = structure.filter(~pl.col("task_structure_valid").fill_null(False))
    if failures.height:
        raise ValueError(f"trial task-structure audit failed:\n{failures.head(5)}")
    return structure.join(
        label_consistency,
        on=SOURCE_COLUMN,
        how="left",
        validate="1:1",
    ).sort(SOURCE_COLUMN)


def _infer_session_frame_offsets(
    finite_anchors: pl.DataFrame,
    presentations: pl.DataFrame,
) -> pl.DataFrame:
    """Infer session-local integer frame offsets from exact stimulus labels.

    Each candidate originates from an exact, non-omitted trial/presentation
    label pair.  It remains valid only if applying it gives every finite trial
    anchor exactly one presentation with the same raw token and physical-change
    label, without reusing a presentation.  Timestamps are deliberately absent
    from this inference.
    """

    scope = "session_local_exact_labels_full_anchor_coverage_no_timestamp_inference"
    source_dtype = finite_anchors.schema[SOURCE_COLUMN]
    result_schema = {
        SOURCE_COLUMN: source_dtype,
        "session_frame_offset": pl.Int64,
        "n_session_frame_offset_candidates": pl.Int64,
        "frame_offset_inference_pass": pl.Boolean,
        "frame_offset_inference_scope": pl.String,
    }
    if finite_anchors.is_empty():
        return pl.DataFrame(schema=result_schema)

    anchors_by_source: dict[Any, list[dict[str, Any]]] = {}
    for row in finite_anchors.select(
        SOURCE_COLUMN,
        "change_frame",
        "change_image_name",
        "stimulus_token_changed",
    ).iter_rows(named=True):
        anchors_by_source.setdefault(row[SOURCE_COLUMN], []).append(row)

    presentations_by_source: dict[Any, dict[int, list[dict[str, Any]]]] = {}
    eligible_by_source_and_label: dict[Any, dict[tuple[Any, Any], list[int]]] = {}
    for row_number, row in enumerate(
        presentations.select(
            SOURCE_COLUMN,
            "start_frame",
            "image_name",
            "is_change",
            "omitted",
        ).iter_rows(named=True)
    ):
        source = row[SOURCE_COLUMN]
        frame = int(row["start_frame"])
        row["_frame_offset_inference_row_number"] = row_number
        presentations_by_source.setdefault(source, {}).setdefault(frame, []).append(row)
        if (
            row["omitted"] is False
            and row["image_name"] is not None
            and row["is_change"] is not None
        ):
            label = (row["image_name"], row["is_change"])
            eligible_by_source_and_label.setdefault(source, {}).setdefault(label, []).append(frame)

    result_rows: list[dict[str, Any]] = []
    for source, anchors in anchors_by_source.items():
        eligible_by_label = eligible_by_source_and_label.get(source, {})
        seed_anchor = min(
            anchors,
            key=lambda anchor: len(
                eligible_by_label.get(
                    (anchor["change_image_name"], anchor["stimulus_token_changed"]),
                    (),
                )
            ),
        )
        seed_label = (
            seed_anchor["change_image_name"],
            seed_anchor["stimulus_token_changed"],
        )
        candidate_offsets = {
            presentation_frame - int(seed_anchor["change_frame"])
            for presentation_frame in eligible_by_label.get(seed_label, ())
        }
        frame_lookup = presentations_by_source.get(source, {})
        valid_offsets: list[int] = []
        for candidate_offset in sorted(candidate_offsets):
            used_presentations: set[int] = set()
            candidate_is_valid = True
            for anchor in anchors:
                target_frame = int(anchor["change_frame"]) + candidate_offset
                matches = frame_lookup.get(target_frame, ())
                if len(matches) != 1:
                    candidate_is_valid = False
                    break
                presentation = matches[0]
                presentation_identity = presentation["_frame_offset_inference_row_number"]
                if (
                    presentation_identity in used_presentations
                    or presentation["omitted"] is not False
                    or anchor["change_image_name"] is None
                    or presentation["image_name"] is None
                    or anchor["change_image_name"] != presentation["image_name"]
                    or anchor["stimulus_token_changed"] is None
                    or presentation["is_change"] is None
                    or anchor["stimulus_token_changed"] != presentation["is_change"]
                ):
                    candidate_is_valid = False
                    break
                used_presentations.add(presentation_identity)
            if candidate_is_valid and len(used_presentations) == len(anchors):
                valid_offsets.append(candidate_offset)

        inference_pass = len(valid_offsets) == 1
        result_rows.append(
            {
                SOURCE_COLUMN: source,
                "session_frame_offset": valid_offsets[0] if inference_pass else None,
                "n_session_frame_offset_candidates": len(valid_offsets),
                "frame_offset_inference_pass": inference_pass,
                "frame_offset_inference_scope": scope,
            }
        )

    return pl.DataFrame(result_rows, schema=result_schema)


def align_trials_to_presentations(
    trials: pl.DataFrame,
    presentations: pl.DataFrame,
) -> pl.DataFrame:
    """Join completed trials after inferring each session's frame origin.

    Candidate integer offsets use only exact non-omitted stimulus-label pairs
    and full one-to-one frame coverage.  Presentation time must subsequently
    fall strictly inside its matched trial, but time never infers or selects an
    offset.  ``change_time - start_time`` remains a descriptive residual.
    """

    _require_columns(
        trials,
        {
            SOURCE_COLUMN,
            TABLE_INDEX_COLUMN,
            "trial_row_key",
            "id",
            "subject_id",
            "ecephys_session_id",
            "recording_day",
            "reward_block",
            "change_time",
            "change_frame",
            "initial_image_name",
            "change_image_name",
            "initial_image_id",
            "change_image_id",
            "initial_image_relative_contrast",
            "change_image_relative_contrast",
            "stimulus_token_changed",
            "identity_changed",
            "go",
            "catch",
            "auto_rewarded",
            "aborted",
            "is_change",
            "is_sham_change",
        },
        frame_name="trial_inventory",
    )
    _require_columns(
        presentations,
        {
            SOURCE_COLUMN,
            TABLE_INDEX_COLUMN,
            "presentation_row_key",
            "id",
            "start_time",
            "stop_time",
            "start_frame",
            "image_name",
            "image_id",
            "image_relative_contrast",
            "is_change",
            "is_image_novel",
            "omitted",
        },
        frame_name="task_presentations",
    )
    invalid_trial_anchors = trials.filter(
        pl.col("change_frame").is_not_null()
        & ~pl.col("change_frame").is_nan()
        & (
            ~pl.col("change_frame").is_finite()
            | (pl.col("change_frame") != pl.col("change_frame").round(0))
        )
    )
    if invalid_trial_anchors.height:
        raise ValueError("finite trial change_frame anchors must be integral")
    invalid_presentation_frames = presentations.filter(
        pl.col("start_frame").is_null()
        | ~pl.col("start_frame").is_finite()
        | (pl.col("start_frame") != pl.col("start_frame").round(0))
    )
    if invalid_presentation_frames.height:
        raise ValueError("presentation start_frame values must be finite integers")

    finite_anchors = trials.filter(
        pl.col("change_frame").is_not_null() & pl.col("change_frame").is_finite()
    ).with_columns(pl.col("change_frame").cast(pl.Int64).alias("_integer_change_frame"))
    session_offsets = _infer_session_frame_offsets(finite_anchors, presentations)
    trial_rows = finite_anchors.select(
        SOURCE_COLUMN,
        "trial_row_key",
        pl.col(TABLE_INDEX_COLUMN).alias("trial_table_index"),
        pl.col("id").alias("trial_id"),
        "subject_id",
        "ecephys_session_id",
        "recording_day",
        "reward_block",
        "reward_block_source",
        pl.col("start_time").alias("trial_start_time"),
        pl.col("stop_time").alias("trial_stop_time"),
        pl.col("change_time").alias("trial_change_time"),
        pl.col("change_frame").alias("trial_change_frame"),
        "_integer_change_frame",
        pl.col("initial_image_name").alias("trial_initial_image_name"),
        pl.col("change_image_name").alias("trial_change_image_name"),
        pl.col("initial_image_id").alias("trial_initial_image_id"),
        pl.col("change_image_id").alias("trial_change_image_id"),
        pl.col("initial_image_relative_contrast").alias("trial_initial_image_relative_contrast"),
        pl.col("change_image_relative_contrast").alias("trial_change_image_relative_contrast"),
        pl.col("stimulus_token_changed").alias("trial_stimulus_token_changed"),
        pl.col("identity_changed").alias("trial_identity_changed"),
        pl.col("go").alias("trial_go"),
        pl.col("catch").alias("trial_catch"),
        pl.col("auto_rewarded").alias("trial_auto_rewarded"),
        pl.col("aborted").alias("trial_aborted"),
        pl.col("is_change").alias("trial_is_change_label"),
        pl.col("is_sham_change").alias("trial_is_sham_change"),
    ).join(session_offsets, on=SOURCE_COLUMN, how="left", validate="m:1")
    trial_rows = trial_rows.with_columns(
        (pl.col("_integer_change_frame") + pl.col("session_frame_offset")).alias("_alignment_frame")
    )
    presentation_lookup = (
        presentations.with_columns(pl.col("start_frame").cast(pl.Int64).alias("_alignment_frame"))
        .group_by(SOURCE_COLUMN, "_alignment_frame")
        .agg(
            pl.len().cast(pl.Int64).alias("n_presentation_frame_matches"),
            pl.col("presentation_row_key").first(),
            pl.col(TABLE_INDEX_COLUMN).first().alias("presentation_table_index"),
            pl.col("id").first().alias("presentation_id"),
            pl.col("start_time").first().alias("presentation_start_time"),
            pl.col("stop_time").first().alias("presentation_stop_time"),
            pl.col("start_frame").first().alias("presentation_start_frame"),
            pl.col("image_name").first().alias("presentation_image_name"),
            pl.col("image_id").first().alias("presentation_image_id"),
            pl.col("image_relative_contrast").first().alias("presentation_image_relative_contrast"),
            pl.col("is_change").first().alias("presentation_is_change"),
            pl.col("is_image_novel").first().alias("presentation_is_image_novel"),
            pl.col("omitted").first().alias("presentation_omitted"),
        )
    )
    aligned = trial_rows.join(
        presentation_lookup,
        on=[SOURCE_COLUMN, "_alignment_frame"],
        how="left",
        validate="m:1",
    ).with_columns(pl.col("n_presentation_frame_matches").fill_null(0))
    exactly_one_frame_match = pl.col("n_presentation_frame_matches") == 1
    aligned = aligned.with_columns(
        pl.when(exactly_one_frame_match)
        .then(pl.col("trial_change_image_name") == pl.col("presentation_image_name"))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("stimulus_token_agrees"),
        pl.when(exactly_one_frame_match)
        .then(pl.col("trial_change_image_id") == pl.col("presentation_image_id"))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("image_identity_agrees"),
        pl.when(exactly_one_frame_match)
        .then(
            pl.col("trial_change_image_relative_contrast")
            == pl.col("presentation_image_relative_contrast")
        )
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("image_contrast_agrees"),
        pl.when(exactly_one_frame_match)
        .then(pl.col("trial_stimulus_token_changed") == pl.col("presentation_is_change"))
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("physical_change_label_agrees"),
        pl.when(exactly_one_frame_match)
        .then(
            pl.col("presentation_start_time").is_not_null()
            & pl.col("presentation_start_time").is_finite()
            & pl.col("trial_start_time").is_not_null()
            & pl.col("trial_start_time").is_finite()
            & pl.col("trial_stop_time").is_not_null()
            & pl.col("trial_stop_time").is_finite()
            & (pl.col("presentation_start_time") > pl.col("trial_start_time"))
            & (pl.col("presentation_start_time") < pl.col("trial_stop_time"))
        )
        .otherwise(pl.lit(None, dtype=pl.Boolean))
        .alias("presentation_start_within_trial_interval"),
        (pl.col("trial_change_time") - pl.col("presentation_start_time")).alias(
            "trial_change_minus_presentation_start_seconds"
        ),
    )
    return (
        aligned.with_columns(
            (
                pl.col("frame_offset_inference_pass").fill_null(False)
                & (pl.col("n_presentation_frame_matches") == 1)
                & pl.col("stimulus_token_agrees").fill_null(False)
                & pl.col("image_identity_agrees").fill_null(False)
                & pl.col("image_contrast_agrees").fill_null(False)
                & pl.col("physical_change_label_agrees").fill_null(False)
                & ~pl.col("presentation_omitted").fill_null(True)
                & pl.col("presentation_start_within_trial_interval").fill_null(False)
            ).alias("trial_presentation_alignment_pass"),
            pl.concat_str(
                pl.when(~pl.col("frame_offset_inference_pass").fill_null(False))
                .then(pl.lit("frame_offset_inference_failed;"))
                .otherwise(pl.lit("")),
                pl.when(
                    pl.col("frame_offset_inference_pass").fill_null(False)
                    & (pl.col("n_presentation_frame_matches") == 0)
                )
                .then(pl.lit("missing_frame_match;"))
                .when(
                    pl.col("frame_offset_inference_pass").fill_null(False)
                    & (pl.col("n_presentation_frame_matches") > 1)
                )
                .then(pl.lit("duplicate_frame_match;"))
                .otherwise(pl.lit("")),
                pl.when(pl.col("stimulus_token_agrees").eq(False).fill_null(False))
                .then(pl.lit("stimulus_token_mismatch;"))
                .otherwise(pl.lit("")),
                pl.when(pl.col("image_identity_agrees").eq(False).fill_null(False))
                .then(pl.lit("image_identity_mismatch;"))
                .otherwise(pl.lit("")),
                pl.when(pl.col("image_contrast_agrees").eq(False).fill_null(False))
                .then(pl.lit("image_contrast_mismatch;"))
                .otherwise(pl.lit("")),
                pl.when(pl.col("physical_change_label_agrees").eq(False).fill_null(False))
                .then(pl.lit("physical_change_label_mismatch;"))
                .otherwise(pl.lit("")),
                pl.when(exactly_one_frame_match & pl.col("presentation_omitted").fill_null(True))
                .then(pl.lit("omitted_presentation;"))
                .otherwise(pl.lit("")),
                pl.when(
                    pl.col("presentation_start_within_trial_interval").eq(False).fill_null(False)
                )
                .then(pl.lit("presentation_start_outside_trial_interval;"))
                .otherwise(pl.lit("")),
            )
            .str.strip_chars_end(";")
            .alias("alignment_failure_reasons"),
            pl.lit(
                "same_session_start_frame_equals_change_frame_plus_inferred_integer_offset"
            ).alias("alignment_join_rule"),
            pl.lit("time_residual_descriptive_not_used_for_frame_offset_inference").alias(
                "timing_offset_scope"
            ),
        )
        .drop("_integer_change_frame", "_alignment_frame")
        .sort("subject_id", SOURCE_COLUMN, "trial_table_index")
    )


def summarize_trial_presentation_alignment(alignment: pl.DataFrame) -> pl.DataFrame:
    """Summarize rowwise frame/label alignment once per session."""

    _require_columns(
        alignment,
        {
            SOURCE_COLUMN,
            "subject_id",
            "ecephys_session_id",
            "trial_row_key",
            "session_frame_offset",
            "n_session_frame_offset_candidates",
            "frame_offset_inference_pass",
            "frame_offset_inference_scope",
            "n_presentation_frame_matches",
            "stimulus_token_agrees",
            "image_identity_agrees",
            "image_contrast_agrees",
            "physical_change_label_agrees",
            "presentation_start_within_trial_interval",
            "trial_presentation_alignment_pass",
            "trial_change_minus_presentation_start_seconds",
        },
        frame_name="trial_presentation_alignment",
    )
    return (
        alignment.group_by(SOURCE_COLUMN, "subject_id", "ecephys_session_id")
        .agg(
            pl.len().cast(pl.Int64).alias("n_trial_change_anchors"),
            pl.col("session_frame_offset").first(),
            pl.col("n_session_frame_offset_candidates").first(),
            pl.col("frame_offset_inference_pass").all(),
            pl.col("frame_offset_inference_scope").first(),
            (
                pl.col("frame_offset_inference_pass").fill_null(False)
                & (pl.col("n_presentation_frame_matches") == 0)
            )
            .sum()
            .cast(pl.Int64)
            .alias("n_missing_frame_matches"),
            (
                pl.col("frame_offset_inference_pass").fill_null(False)
                & (pl.col("n_presentation_frame_matches") > 1)
            )
            .sum()
            .cast(pl.Int64)
            .alias("n_duplicate_frame_matches"),
            pl.col("stimulus_token_agrees")
            .eq(False)
            .fill_null(False)
            .sum()
            .cast(pl.Int64)
            .alias("n_stimulus_token_mismatches"),
            pl.col("image_identity_agrees")
            .eq(False)
            .fill_null(False)
            .sum()
            .cast(pl.Int64)
            .alias("n_image_identity_mismatches"),
            pl.col("image_contrast_agrees")
            .eq(False)
            .fill_null(False)
            .sum()
            .cast(pl.Int64)
            .alias("n_image_contrast_mismatches"),
            pl.col("physical_change_label_agrees")
            .eq(False)
            .fill_null(False)
            .sum()
            .cast(pl.Int64)
            .alias("n_physical_change_label_mismatches"),
            pl.col("presentation_start_within_trial_interval")
            .eq(False)
            .fill_null(False)
            .sum()
            .cast(pl.Int64)
            .alias("n_presentation_starts_outside_trial_interval"),
            pl.col("trial_presentation_alignment_pass")
            .fill_null(False)
            .sum()
            .cast(pl.Int64)
            .alias("n_alignment_pass"),
            pl.col("trial_change_minus_presentation_start_seconds")
            .min()
            .alias("minimum_change_start_offset_seconds"),
            pl.col("trial_change_minus_presentation_start_seconds")
            .median()
            .alias("median_change_start_offset_seconds"),
            pl.col("trial_change_minus_presentation_start_seconds")
            .max()
            .alias("maximum_change_start_offset_seconds"),
        )
        .with_columns(
            (
                pl.col("frame_offset_inference_pass").fill_null(False)
                & (pl.col("n_session_frame_offset_candidates") == 1)
                & (pl.col("n_alignment_pass") == pl.col("n_trial_change_anchors"))
            ).alias("trial_presentation_alignment_pass"),
            pl.lit("timing_residual_descriptive_not_frame_offset_inference").alias("offset_scope"),
        )
        .sort("subject_id", SOURCE_COLUMN)
    )


def validate_trial_presentation_alignment(
    alignment: pl.DataFrame,
    trials: pl.DataFrame,
    presentations: pl.DataFrame,
    session_inventory: pl.DataFrame,
) -> pl.DataFrame:
    """Fail unless every completed trial has one structurally consistent match."""

    summary = summarize_trial_presentation_alignment(alignment)
    expected_anchor_rows = trials.filter(
        pl.col("change_frame").is_not_null() & pl.col("change_frame").is_finite()
    ).height
    if alignment.height != expected_anchor_rows:
        raise ValueError("alignment does not contain exactly one row per finite trial anchor")
    if alignment.get_column("trial_row_key").n_unique() != alignment.height:
        raise ValueError("alignment contains duplicate trial row keys")
    if alignment.select(SOURCE_COLUMN, "trial_change_frame").n_unique() != alignment.height:
        raise ValueError("alignment contains reused trial change_frame anchors")
    matched_presentations = alignment.filter(pl.col("presentation_row_key").is_not_null())
    if (
        matched_presentations.select(SOURCE_COLUMN, "presentation_row_key").n_unique()
        != matched_presentations.height
    ):
        raise ValueError("alignment reuses a presentation row across trial anchors")
    if presentations.select(SOURCE_COLUMN, "start_frame").n_unique() != presentations.height:
        raise ValueError("presentation start_frame must be unique within session")
    expected_sources = set(session_inventory.get_column(SOURCE_COLUMN).to_list())
    observed_sources = set(summary.get_column(SOURCE_COLUMN).to_list())
    if observed_sources != expected_sources:
        raise ValueError("trial-presentation alignment does not cover every session")
    offset_failures = summary.filter(
        ~pl.col("frame_offset_inference_pass").fill_null(False)
        | (pl.col("n_session_frame_offset_candidates") != 1)
        | pl.col("session_frame_offset").is_null()
    )
    if offset_failures.height:
        raise ValueError(
            "frame-offset inference did not yield exactly one valid candidate per session:\n"
            f"{offset_failures.select(SOURCE_COLUMN, 'n_session_frame_offset_candidates').head(5)}"
        )
    failures = alignment.filter(~pl.col("trial_presentation_alignment_pass").fill_null(False))
    if failures.height:
        raise ValueError(
            "trial-presentation frame/label alignment failed for one or more rows:\n"
            f"{failures.select('trial_row_key', 'alignment_failure_reasons').head(5)}"
        )
    return summary


def summarize_stimulus_semantics(
    presentations: pl.DataFrame,
    alignment: pl.DataFrame,
) -> pl.DataFrame:
    """Audit raw novelty/contrast labels without claiming exposure history."""

    _require_columns(
        presentations,
        {
            SOURCE_COLUMN,
            "subject_id",
            "ecephys_session_id",
            "recording_day",
            "image_name",
            "image_id",
            "image_relative_contrast",
            "is_image_novel",
            "omitted",
            "metadata_novel_image_id",
            "metadata_novel_relative_contrast",
            "is_metadata_designated_novel_identity",
            "within_session_identity_exposure_index",
        },
        frame_name="task_presentations",
    )
    _require_columns(
        alignment,
        {SOURCE_COLUMN, "trial_presentation_alignment_pass"},
        frame_name="trial_presentation_alignment",
    )
    nonomitted = ~pl.col("omitted").fill_null(False)
    valid_contrast = pl.col("image_relative_contrast").is_in([0.7, 1.0])
    presentation_summary = presentations.group_by(
        SOURCE_COLUMN,
        "subject_id",
        "ecephys_session_id",
        "recording_day",
    ).agg(
        pl.len().cast(pl.Int64).alias("n_task_presentations"),
        nonomitted.sum().cast(pl.Int64).alias("n_non_omitted_presentations"),
        (nonomitted & (pl.col("image_id").is_null() | pl.col("image_relative_contrast").is_null()))
        .sum()
        .cast(pl.Int64)
        .alias("n_unparsed_non_omitted_image_names"),
        (nonomitted & ~valid_contrast.fill_null(False))
        .sum()
        .cast(pl.Int64)
        .alias("n_invalid_non_omitted_contrasts"),
        (nonomitted & pl.col("is_image_novel").is_null())
        .sum()
        .cast(pl.Int64)
        .alias("n_missing_novel_labels"),
        (
            nonomitted
            & (
                pl.col("is_image_novel") != pl.col("is_metadata_designated_novel_identity")
            ).fill_null(True)
        )
        .sum()
        .cast(pl.Int64)
        .alias("n_novel_label_metadata_mismatches"),
        (nonomitted & pl.col("within_session_identity_exposure_index").is_null())
        .sum()
        .cast(pl.Int64)
        .alias("n_missing_identity_exposure_indices"),
        pl.col("metadata_novel_image_id")
        .drop_nulls()
        .n_unique()
        .cast(pl.Int64)
        .alias("n_metadata_novel_identities"),
        pl.col("metadata_novel_relative_contrast")
        .drop_nulls()
        .n_unique()
        .cast(pl.Int64)
        .alias("n_metadata_novel_contrasts"),
        (pl.col("metadata_novel_relative_contrast").drop_nulls().eq(1.0).all()).alias(
            "metadata_novel_contrast_is_full"
        ),
        pl.col("is_metadata_designated_novel_identity")
        .sum()
        .cast(pl.Int64)
        .alias("n_designated_novel_presentations"),
    )
    alignment_summary = alignment.group_by(SOURCE_COLUMN).agg(
        (~pl.col("trial_presentation_alignment_pass"))
        .sum()
        .cast(pl.Int64)
        .alias("n_trial_presentation_semantic_mismatches")
    )
    return (
        presentation_summary.join(
            alignment_summary,
            on=SOURCE_COLUMN,
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("n_trial_presentation_semantic_mismatches").fill_null(0),
        )
        .with_columns(
            (
                (pl.col("n_unparsed_non_omitted_image_names") == 0)
                & (pl.col("n_invalid_non_omitted_contrasts") == 0)
                & (pl.col("n_missing_novel_labels") == 0)
                & (pl.col("n_novel_label_metadata_mismatches") == 0)
                & (pl.col("n_missing_identity_exposure_indices") == 0)
                & (pl.col("n_metadata_novel_identities") == 1)
                & (pl.col("n_metadata_novel_contrasts") == 1)
                & pl.col("metadata_novel_contrast_is_full").fill_null(False)
                & (pl.col("n_designated_novel_presentations") > 0)
                & (pl.col("n_trial_presentation_semantic_mismatches") == 0)
            ).alias("stimulus_semantics_internal_consistency_pass"),
            pl.lit("internally_consistent_history_unverified").alias("stimulus_semantics_status"),
            pl.lit("novel_label_designation_not_first_exposure_history").alias(
                "novelty_interpretation_scope"
            ),
        )
        .sort("subject_id", SOURCE_COLUMN)
    )


def validate_stimulus_semantics(
    semantics: pl.DataFrame,
    session_inventory: pl.DataFrame,
) -> None:
    """Fail on raw label/metadata/alignment inconsistency in any session."""

    _require_columns(
        semantics,
        {SOURCE_COLUMN, "stimulus_semantics_internal_consistency_pass"},
        frame_name="stimulus_semantics_audit",
    )
    expected_sources = set(session_inventory.get_column(SOURCE_COLUMN).to_list())
    observed_sources = set(semantics.get_column(SOURCE_COLUMN).to_list())
    if semantics.height != session_inventory.height or observed_sources != expected_sources:
        raise ValueError("stimulus semantics audit does not exactly cover every session")
    failures = semantics.filter(
        ~pl.col("stimulus_semantics_internal_consistency_pass").fill_null(False)
    )
    if failures.height:
        raise ValueError(f"stimulus semantics internal consistency failed:\n{failures.head(5)}")


def summarize_stimulus_estimability(
    alignment: pl.DataFrame,
    session_inventory: pl.DataFrame,
) -> pl.DataFrame:
    """Report state-by-contrast coverage at aligned ``im115`` trial anchors."""

    _require_columns(
        alignment,
        {
            SOURCE_COLUMN,
            "reward_block",
            "presentation_image_id",
            "presentation_image_relative_contrast",
            "trial_presentation_alignment_pass",
        },
        frame_name="trial_presentation_alignment",
    )
    _validate_session_inventory(session_inventory)
    session_columns = [
        SOURCE_COLUMN,
        *(
            column
            for column in ("subject_id", "ecephys_session_id", "recording_day")
            if column in session_inventory.columns
        ),
    ]
    states = pl.DataFrame({"reward_block": ["engaged_1", "no_reward", "engaged_2"]})
    grid = session_inventory.select(*session_columns).join(states, how="cross")
    eligible = alignment.filter(
        pl.col("trial_presentation_alignment_pass") & (pl.col("presentation_image_id") == "im115")
    )
    observed = eligible.group_by(SOURCE_COLUMN, "reward_block").agg(
        pl.len().cast(pl.Int64).alias("n_im115_trial_anchors"),
        (pl.col("presentation_image_relative_contrast") == 0.7)
        .sum()
        .cast(pl.Int64)
        .alias("n_im115_contrast_0_7_anchors"),
        (pl.col("presentation_image_relative_contrast") == 1.0)
        .sum()
        .cast(pl.Int64)
        .alias("n_im115_contrast_1_0_anchors"),
    )
    return (
        grid.join(
            observed,
            on=[SOURCE_COLUMN, "reward_block"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("n_im115_trial_anchors").fill_null(0),
            pl.col("n_im115_contrast_0_7_anchors").fill_null(0),
            pl.col("n_im115_contrast_1_0_anchors").fill_null(0),
        )
        .with_columns(
            (
                (pl.col("n_im115_contrast_0_7_anchors") > 0)
                & (pl.col("n_im115_contrast_1_0_anchors") > 0)
            ).alias("both_im115_contrasts_present"),
            pl.lit("audited_trial_reward_block_at_frame_aligned_anchor").alias(
                "reward_state_source"
            ),
            pl.lit("estimable_if_both_contrasts_present_in_each_session_state").alias(
                "state_by_contrast_scope"
            ),
            pl.lit("assessed_in_stimulus_design_aliases").alias("pure_novelty_main_effect_status"),
            pl.lit("assessed_in_stimulus_design_aliases").alias("novelty_by_contrast_status"),
        )
        .sort("subject_id", SOURCE_COLUMN, "reward_block")
    )


def validate_stimulus_estimability(
    estimability: pl.DataFrame,
    session_inventory: pl.DataFrame,
) -> None:
    """Require crossed ``im115`` contrast coverage in every session/state."""

    _require_columns(
        estimability,
        {SOURCE_COLUMN, "reward_block", "both_im115_contrasts_present"},
        frame_name="stimulus_estimability_summary",
    )
    expected_rows = session_inventory.height * 3
    if estimability.height != expected_rows:
        raise ValueError(
            f"stimulus estimability expected {expected_rows} session-state rows, "
            f"observed {estimability.height}"
        )
    expected_sources = set(session_inventory.get_column(SOURCE_COLUMN).to_list())
    if set(estimability.get_column(SOURCE_COLUMN).to_list()) != expected_sources:
        raise ValueError("stimulus estimability does not exactly cover every session")
    if estimability.select(SOURCE_COLUMN, "reward_block").n_unique() != expected_rows:
        raise ValueError("stimulus estimability contains duplicate session-state rows")
    failures = estimability.filter(~pl.col("both_im115_contrasts_present").fill_null(False))
    if failures.height:
        raise ValueError(
            "state-by-contrast within im115 is not crossed in every session/state:\n"
            f"{failures.head(5)}"
        )


def build_stimulus_design_aliases(
    presentations: pl.DataFrame,
    estimability: pl.DataFrame,
) -> pl.DataFrame:
    """Create an explicit estimand-level alias and rank-status report."""

    _require_columns(
        presentations,
        {
            "recording_day",
            "image_id",
            "metadata_novel_image_id",
            "is_metadata_designated_novel_identity",
            "image_relative_contrast",
        },
        frame_name="task_presentations",
    )
    _require_columns(
        estimability,
        {"both_im115_contrasts_present"},
        frame_name="stimulus_estimability_summary",
    )
    day_novel = presentations.select("recording_day", "metadata_novel_image_id").unique()
    day_novel_one_to_one = (
        day_novel.get_column("recording_day").n_unique() == day_novel.height
        and day_novel.get_column("metadata_novel_image_id").n_unique() == day_novel.height
    )
    identity_novel_status = (
        presentations.filter(pl.col("image_id").is_not_null())
        .select("image_id", "is_metadata_designated_novel_identity")
        .unique()
    )
    identity_status_counts = identity_novel_status.group_by("image_id").agg(
        pl.col("is_metadata_designated_novel_identity").n_unique().alias("n_novel_status_values")
    )
    n_identities_spanning_novel_and_familiar = identity_status_counts.filter(
        pl.col("n_novel_status_values") > 1
    ).height
    novel_identity_status_disjoint = (
        identity_status_counts.height > 0 and n_identities_spanning_novel_and_familiar == 0
    )
    novel_contrasts = (
        presentations.filter(pl.col("is_metadata_designated_novel_identity"))
        .get_column("image_relative_contrast")
        .drop_nulls()
    )
    novel_only_full_contrast = bool(len(novel_contrasts) > 0 and novel_contrasts.eq(1.0).all())
    state_contrast_crossed = bool(estimability.get_column("both_im115_contrasts_present").all())
    pure_novelty_confounded = day_novel_one_to_one and novel_identity_status_disjoint
    common = {
        "n_session_state_rows": estimability.height,
        "n_recording_days": day_novel.get_column("recording_day").n_unique(),
        "n_designated_novel_identities": day_novel.get_column("metadata_novel_image_id").n_unique(),
        "day_novel_mapping_one_to_one": day_novel_one_to_one,
        "n_image_identities": identity_status_counts.height,
        "n_identities_spanning_novel_and_familiar": n_identities_spanning_novel_and_familiar,
        "novel_identity_status_disjoint": novel_identity_status_disjoint,
        "novel_only_full_contrast": novel_only_full_contrast,
        "reward_state_source": "audited_trial_reward_block_not_presentation_labels",
    }
    return pl.DataFrame(
        [
            {
                **common,
                "estimand": "state_x_contrast_within_im115",
                "rank_status": "estimable" if state_contrast_crossed else "not_estimable",
                "evidence": "both_0.7_and_1.0_at_aligned_trial_anchors_every_session_state",
                "interpretation_scope": "confirmatory_candidate_pending_held_out_lock",
            },
            {
                **common,
                "estimand": "pure_novelty_main_effect",
                "rank_status": (
                    "unidentifiable"
                    if pure_novelty_confounded
                    else "not_established_from_inventory"
                ),
                "evidence": (
                    "day_to_novel_identity_one_to_one_and_no_identity_spans_novel_familiar"
                    if pure_novelty_confounded
                    else "required_identity_day_alias_evidence_not_satisfied"
                ),
                "interpretation_scope": (
                    "identity_and_day_alias_cannot_be_adjusted_away"
                    if pure_novelty_confounded
                    else "do_not_claim_pure_novelty_rank_from_this_inventory"
                ),
            },
            {
                **common,
                "estimand": "novelty_x_contrast",
                "rank_status": (
                    "unidentifiable"
                    if novel_only_full_contrast
                    else "not_established_from_inventory"
                ),
                "evidence": (
                    "designated_novel_images_occur_only_at_full_contrast"
                    if novel_only_full_contrast
                    else "novel_contrast_restriction_not_satisfied"
                ),
                "interpretation_scope": (
                    "novelty_contrast_cells_structurally_absent"
                    if novel_only_full_contrast
                    else "do_not_claim_novelty_contrast_rank_from_this_inventory"
                ),
            },
            {
                **common,
                "estimand": "within_session_novel_identity_exposure_trajectory",
                "rank_status": "estimable_as_labeled_identity_adaptation",
                "evidence": "ordered_exposure_index_available_within_session_identity",
                "interpretation_scope": "not_a_pure_first_exposure_novelty_effect",
            },
        ],
        infer_schema_length=None,
    ).sort("estimand")


def validate_stimulus_design_aliases(aliases: pl.DataFrame) -> None:
    """Fail if estimand rank labels contradict their recorded design evidence."""

    _require_columns(
        aliases,
        {
            "estimand",
            "rank_status",
            "day_novel_mapping_one_to_one",
            "novel_identity_status_disjoint",
            "novel_only_full_contrast",
        },
        frame_name="stimulus_design_aliases",
    )
    expected_estimands = {
        "state_x_contrast_within_im115",
        "pure_novelty_main_effect",
        "novelty_x_contrast",
        "within_session_novel_identity_exposure_trajectory",
    }
    if aliases.height != len(expected_estimands) or set(aliases.get_column("estimand")) != (
        expected_estimands
    ):
        raise ValueError("stimulus design aliases do not contain the exact expected estimands")
    pure = aliases.filter(pl.col("estimand") == "pure_novelty_main_effect").row(0, named=True)
    expected_pure = (
        "unidentifiable"
        if pure["day_novel_mapping_one_to_one"] and pure["novel_identity_status_disjoint"]
        else "not_established_from_inventory"
    )
    interaction = aliases.filter(pl.col("estimand") == "novelty_x_contrast").row(0, named=True)
    expected_interaction = (
        "unidentifiable"
        if interaction["novel_only_full_contrast"]
        else "not_established_from_inventory"
    )
    if pure["rank_status"] != expected_pure or interaction["rank_status"] != expected_interaction:
        raise ValueError("stimulus design rank status contradicts recorded alias evidence")


def summarize_stimulus_inventory(presentations: pl.DataFrame) -> pl.DataFrame:
    """Count raw task presentations without validating condition semantics."""

    required = {
        SOURCE_COLUMN,
        "asset_id",
        "subject_id",
        "task_presentation_path",
        "image_name",
        "image_id",
        "image_relative_contrast",
        "is_change",
        "active",
        "is_image_novel",
        "rewarded",
        "omitted",
        "start_time",
        "stop_time",
    }
    _require_columns(presentations, required, frame_name="presentations")
    identity = [
        SOURCE_COLUMN,
        *(column for column in SESSION_METADATA_COLUMNS if column in presentations.columns),
        "task_presentation_path",
    ]
    conditions = [
        "image_name",
        "image_id",
        "image_relative_contrast",
        "is_change",
        "active",
        "is_image_novel",
        "rewarded",
        "omitted",
    ]
    return (
        presentations.group_by(*identity, *conditions, maintain_order=False)
        .agg(
            pl.len().cast(pl.Int64).alias("n_presentations"),
            pl.col("start_time").is_null().sum().cast(pl.Int64).alias("n_missing_start_times"),
            pl.col("stop_time").is_null().sum().cast(pl.Int64).alias("n_missing_stop_times"),
            pl.col("start_time").min().alias("first_start_time"),
            pl.col("stop_time").max().alias("last_stop_time"),
        )
        .with_columns(
            pl.lit("observed_nwb_rows_no_imputation").alias("inventory_scope"),
            pl.lit("raw_presentation_labels_not_reward_state").alias("presentation_label_scope"),
        )
        .sort("subject_id", SOURCE_COLUMN, "image_name", "is_change")
    )


def summarize_stimulus_coverage(stimulus_inventory: pl.DataFrame) -> pl.DataFrame:
    """Aggregate raw-label coverage without novelty/contrast interpretation."""

    required = {
        "subject_id",
        SOURCE_COLUMN,
        "recording_day",
        "image_name",
        "image_id",
        "image_relative_contrast",
        "is_change",
        "active",
        "is_image_novel",
        "rewarded",
        "omitted",
        "n_presentations",
    }
    _require_columns(stimulus_inventory, required, frame_name="stimulus_inventory")
    conditions = [
        "recording_day",
        "image_name",
        "image_id",
        "image_relative_contrast",
        "is_change",
        "active",
        "is_image_novel",
        "rewarded",
        "omitted",
    ]
    return (
        stimulus_inventory.group_by(*conditions)
        .agg(
            pl.col("subject_id").n_unique().cast(pl.Int64).alias("n_mice"),
            pl.col(SOURCE_COLUMN).n_unique().cast(pl.Int64).alias("n_sessions"),
            pl.col("n_presentations").sum().cast(pl.Int64).alias("n_presentations"),
        )
        .with_columns(
            pl.lit("coverage_only_no_neural_or_behavioral_outcomes").alias("analysis_scope"),
            pl.lit("active_and_rewarded_are_presentation_labels_not_reward_state").alias(
                "interpretation_scope"
            ),
            pl.lit("not_validated_by_raw_inventory").alias("novelty_contrast_semantics_status"),
        )
        .sort(*conditions)
    )


def summarize_interval_clock(
    intervals: pl.DataFrame | pl.LazyFrame,
    *,
    table_kind: str,
    event_time_column: str | None = None,
) -> pl.DataFrame:
    """Summarize ordered interval and optional in-interval event integrity.

    Null event timestamps are counted but are not automatically failures because
    the NWB trial table legitimately contains rows without a change event.
    Non-null nonfinite event timestamps and finite events outside their enclosing
    interval are hard failures.
    """

    frame = intervals.collect() if isinstance(intervals, pl.LazyFrame) else intervals
    required = {SOURCE_COLUMN, TABLE_INDEX_COLUMN, "id", "start_time", "stop_time"}
    if event_time_column is not None:
        required.add(event_time_column)
    _require_columns(frame, required, frame_name=table_kind)
    if frame.is_empty():
        raise ValueError(f"{table_kind} contains no rows")

    start = pl.col("start_time")
    stop = pl.col("stop_time")
    ordered = frame.sort(SOURCE_COLUMN, TABLE_INDEX_COLUMN).with_columns(
        (start.diff().over(SOURCE_COLUMN) < 0).fill_null(False).alias("_start_decreased")
    )
    invalid_start = start.is_null() | ~start.is_finite()
    invalid_stop = stop.is_null() | ~stop.is_finite()
    nonpositive_duration = (~invalid_start & ~invalid_stop) & (stop <= start)
    aggregations: list[pl.Expr] = [
        pl.len().cast(pl.Int64).alias("n_rows"),
        pl.col(TABLE_INDEX_COLUMN).n_unique().cast(pl.Int64).alias("n_unique_table_indices"),
        pl.col(TABLE_INDEX_COLUMN).is_null().sum().cast(pl.Int64).alias("n_missing_table_indices"),
        invalid_start.sum().cast(pl.Int64).alias("n_invalid_start_times"),
        invalid_stop.sum().cast(pl.Int64).alias("n_invalid_stop_times"),
        nonpositive_duration.sum().cast(pl.Int64).alias("n_nonpositive_durations"),
        pl.col("_start_decreased").sum().cast(pl.Int64).alias("n_decreasing_start_times"),
        start.is_duplicated().sum().cast(pl.Int64).alias("n_rows_with_duplicate_start_times"),
        start.min().alias("first_start_time"),
        stop.max().alias("last_stop_time"),
    ]
    if event_time_column is not None:
        event = pl.col(event_time_column)
        # NWB exports commonly use NaN as the missing value for optional event
        # anchors (for example, aborted trials without a change). Treat null and
        # NaN as reported missingness; reserve hard failure for +/- infinity.
        event_missing = event.is_null() | event.is_nan()
        event_nonfinite = event.is_not_null() & ~event.is_nan() & ~event.is_finite()
        event_outside = (
            event.is_not_null()
            & event.is_finite()
            & ~invalid_start
            & ~invalid_stop
            & ((event < start) | (event > stop))
        )
        aggregations.extend(
            [
                event_missing.sum().cast(pl.Int64).alias("n_missing_event_times"),
                event_nonfinite.sum().cast(pl.Int64).alias("n_nonfinite_event_times"),
                event_outside.sum().cast(pl.Int64).alias("n_event_times_outside_interval"),
            ]
        )
    else:
        aggregations.extend(
            [
                pl.lit(None, dtype=pl.Int64).alias("n_missing_event_times"),
                pl.lit(None, dtype=pl.Int64).alias("n_nonfinite_event_times"),
                pl.lit(None, dtype=pl.Int64).alias("n_event_times_outside_interval"),
            ]
        )
    return (
        ordered.group_by(SOURCE_COLUMN)
        .agg(*aggregations)
        .with_columns(
            pl.lit(table_kind).alias("table_kind"),
            (pl.col("n_rows") - pl.col("n_unique_table_indices")).alias(
                "n_duplicate_table_indices"
            ),
        )
        .with_columns(
            (
                (pl.col("n_duplicate_table_indices") == 0)
                & (pl.col("n_missing_table_indices") == 0)
                & (pl.col("n_invalid_start_times") == 0)
                & (pl.col("n_invalid_stop_times") == 0)
                & (pl.col("n_nonpositive_durations") == 0)
                & (pl.col("n_decreasing_start_times") == 0)
                & (pl.col("n_rows_with_duplicate_start_times") == 0)
                & pl.col("n_nonfinite_event_times").fill_null(0).eq(0)
                & pl.col("n_event_times_outside_interval").fill_null(0).eq(0)
            ).alias("scalar_interval_clock_pass")
        )
        .select("table_kind", SOURCE_COLUMN, pl.exclude("table_kind", SOURCE_COLUMN))
        .sort(SOURCE_COLUMN)
    )


def combine_event_clock_audits(
    trial_clock: pl.DataFrame,
    presentation_clock: pl.DataFrame,
    session_inventory: pl.DataFrame,
) -> pl.DataFrame:
    """Combine basic scalar interval screens and report temporal-range overlap."""

    _validate_session_inventory(session_inventory)
    for name, frame in (("trial_clock", trial_clock), ("presentation_clock", presentation_clock)):
        _require_columns(
            frame,
            {
                SOURCE_COLUMN,
                "first_start_time",
                "last_stop_time",
                "scalar_interval_clock_pass",
            },
            frame_name=name,
        )
        if frame.get_column(SOURCE_COLUMN).n_unique() != frame.height:
            raise ValueError(f"{name} must have one row per session source")
    trial = trial_clock.select(
        SOURCE_COLUMN,
        *(
            pl.col(column).alias(f"trial_{column}")
            for column in trial_clock.columns
            if column != SOURCE_COLUMN
        ),
    )
    presentation = presentation_clock.select(
        SOURCE_COLUMN,
        *(
            pl.col(column).alias(f"presentation_{column}")
            for column in presentation_clock.columns
            if column != SOURCE_COLUMN
        ),
    )
    metadata_columns = [
        SOURCE_COLUMN,
        *(column for column in SESSION_METADATA_COLUMNS if column in session_inventory.columns),
    ]
    combined = (
        session_inventory.select(*metadata_columns)
        .join(trial, on=SOURCE_COLUMN, how="left", validate="1:1")
        .join(presentation, on=SOURCE_COLUMN, how="left", validate="1:1")
        .with_columns(
            (
                (pl.col("trial_first_start_time") <= pl.col("presentation_last_stop_time"))
                & (pl.col("presentation_first_start_time") <= pl.col("trial_last_stop_time"))
            ).alias("trial_presentation_clock_ranges_overlap"),
            (pl.col("presentation_first_start_time") - pl.col("trial_first_start_time")).alias(
                "presentation_minus_trial_first_start_seconds"
            ),
            (pl.col("presentation_last_stop_time") - pl.col("trial_last_stop_time")).alias(
                "presentation_minus_trial_last_stop_seconds"
            ),
        )
        .with_columns(
            (
                pl.col("trial_scalar_interval_clock_pass").fill_null(False)
                & pl.col("presentation_scalar_interval_clock_pass").fill_null(False)
                & pl.col("trial_presentation_clock_ranges_overlap").fill_null(False)
            ).alias("scalar_interval_clock_screen_pass")
        )
        .sort("subject_id", SOURCE_COLUMN)
    )
    return combined


def validate_event_clock_audit(
    event_clock_audit: pl.DataFrame,
    session_inventory: pl.DataFrame,
) -> None:
    """Fail unless every frozen session passes both required interval clocks."""

    required = {SOURCE_COLUMN, "scalar_interval_clock_screen_pass"}
    _require_columns(event_clock_audit, required, frame_name="event_clock_audit")
    _validate_session_inventory(session_inventory)
    if event_clock_audit.get_column(SOURCE_COLUMN).n_unique() != event_clock_audit.height:
        raise ValueError("event_clock_audit must contain exactly one row per session")
    expected_sources = set(session_inventory.get_column(SOURCE_COLUMN).to_list())
    observed_sources = set(event_clock_audit.get_column(SOURCE_COLUMN).to_list())
    if observed_sources != expected_sources:
        raise ValueError("event_clock_audit does not exactly cover the session inventory")
    failures = event_clock_audit.filter(
        ~pl.col("scalar_interval_clock_screen_pass").fill_null(False)
    )
    if failures.height:
        raise ValueError(
            "required scalar interval-clock screen failed for one or more sessions:\n"
            f"{failures.head(5)}"
        )


def build_event_dictionary(
    event_clock_audit: pl.DataFrame,
    alignment_summary: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Describe audited event fields, clock rules, and semantic restrictions."""

    _require_columns(
        event_clock_audit,
        {"scalar_interval_clock_screen_pass"},
        frame_name="event_clock_audit",
    )
    n_sessions = event_clock_audit.height
    n_sessions_pass = int(
        event_clock_audit.get_column("scalar_interval_clock_screen_pass").fill_null(False).sum()
    )
    alignment_status = "not_scanned_by_scalar_clock_audit"
    if alignment_summary is not None:
        _require_columns(
            alignment_summary,
            {"trial_presentation_alignment_pass"},
            frame_name="trial_presentation_alignment_summary",
        )
        alignment_status = (
            "pass"
            if alignment_summary.height == n_sessions
            and alignment_summary.get_column("trial_presentation_alignment_pass")
            .fill_null(False)
            .all()
            else "fail"
        )
    common: dict[str, Any] = {
        "clock_reference": "NWB session time, seconds",
        "n_sessions_audited": n_sessions,
        "n_sessions_passing_scalar_interval_screen": n_sessions_pass,
        "scalar_interval_clock_screen_status": (
            "pass" if n_sessions == n_sessions_pass and n_sessions else "fail"
        ),
    }
    rows = [
        {
            **common,
            "event_name": "trial_interval",
            "source_table": dg.data.TRIALS_PATH,
            "timestamp_fields": "start_time;stop_time",
            "representation": "closed enclosing interval for row-level integrity checks",
            "missingness_policy": "required finite endpoints",
            "validation_rule": "ordered starts and stop_time > start_time",
            "field_validation_scope": "all-session scalar clock audit",
            "field_validation_status": common["scalar_interval_clock_screen_status"],
            "semantic_restriction": "does not itself define reward availability",
        },
        {
            **common,
            "event_name": "trial_change_anchor",
            "source_table": dg.data.TRIALS_PATH,
            "timestamp_fields": "change_time",
            "representation": "absolute NWB session timestamp",
            "missingness_policy": "nulls counted; non-null values must be finite",
            "validation_rule": "start_time <= change_time <= stop_time when present",
            "field_validation_scope": "all-session scalar clock audit",
            "field_validation_status": common["scalar_interval_clock_screen_status"],
            "semantic_restriction": (
                "physical stimulus-token change uses initial_image_name != change_image_name; "
                "identity and contrast changes remain separate factors"
            ),
        },
        {
            **common,
            "event_name": "trial_change_frame_anchor",
            "source_table": dg.data.TRIALS_PATH,
            "timestamp_fields": "change_frame",
            "representation": "integer-valued stimulus frame index",
            "missingness_policy": "both frame and change_time missing only for aborted trials",
            "validation_rule": (
                "same-session exact join to one task presentation start_frame; labels agree"
            ),
            "field_validation_scope": "rowwise frame and stimulus-label alignment",
            "field_validation_status": alignment_status,
            "semantic_restriction": "change_time equality is not required or assumed",
        },
        {
            **common,
            "event_name": "task_image_presentation_interval",
            "source_table": "schema-discovered task *_presentations table",
            "timestamp_fields": "start_time;stop_time",
            "representation": "absolute NWB session timestamps",
            "missingness_policy": "required finite endpoints",
            "validation_rule": "ordered starts and stop_time > start_time",
            "field_validation_scope": "all-session scalar clock audit",
            "field_validation_status": common["scalar_interval_clock_screen_status"],
            "semantic_restriction": (
                "active, rewarded, and stimulus_block are presentation labels, not reward state"
            ),
        },
        {
            **common,
            "event_name": "task_presentation_start_frame",
            "source_table": "schema-discovered task *_presentations table",
            "timestamp_fields": "start_frame",
            "representation": "integer stimulus frame index unique within session",
            "missingness_policy": "required nonnegative frame for every task presentation",
            "validation_rule": "unique and used for rowwise trial change_frame alignment",
            "field_validation_scope": "rowwise frame and stimulus-label alignment",
            "field_validation_status": alignment_status,
            "semantic_restriction": "frame alignment does not validate all acquisition clocks",
        },
        {
            **common,
            "event_name": "raw_trial_lick_timestamps",
            "source_table": dg.data.TRIALS_PATH,
            "timestamp_fields": "lick_times",
            "representation": "array of absolute NWB session timestamps",
            "missingness_policy": "invalid/missing vectors reported; no imputation",
            "validation_rule": (
                "finite nondecreasing values; exact duplicates retained in raw audit and "
                "deduplicated only for response derivation"
            ),
            "field_validation_scope": "schema-only in Milestone 0; values audited by behavior run",
            "field_validation_status": "not_scanned_by_scalar_clock_audit",
            "semantic_restriction": "response uses the separately audited (0.150, 0.750] s window",
        },
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def _scan_projected_table(
    sources: collections.abc.Iterable[str],
    table_path: str,
    *,
    required_columns: collections.abc.Iterable[str],
    required_dtypes: collections.abc.Mapping[str, pl.DataType],
    optional_dtypes: collections.abc.Mapping[str, pl.DataType],
) -> pl.LazyFrame:
    """Scan a table and project scalar fields before any row collection."""

    source_list = list(sources)
    if not source_list:
        raise ValueError("at least one NWB source is required")
    if set(required_columns) != set(required_dtypes):
        raise ValueError("required column names and schema overrides must agree")
    frame = dg.data.scan_nwb_table(
        source_list,
        table_path,
        columns=None,
        schema_overrides={**required_dtypes, **optional_dtypes},
    )
    schema_names = set(frame.collect_schema().names())
    missing_required = set(required_columns).difference(schema_names)
    if missing_required:
        raise ValueError(
            f"{table_path} is missing required inventory columns: {sorted(missing_required)}"
        )
    expressions: list[pl.Expr | str] = [*required_columns]
    expressions.extend(
        pl.col(column) if column in schema_names else pl.lit(None, dtype=dtype).alias(column)
        for column, dtype in optional_dtypes.items()
    )
    expressions.extend((SOURCE_COLUMN, TABLE_PATH_COLUMN, TABLE_INDEX_COLUMN))
    return frame.select(*expressions)


def _validate_session_inventory(session_inventory: pl.DataFrame) -> None:
    required = {SOURCE_COLUMN, "asset_id", "subject_id"}
    _require_columns(session_inventory, required, frame_name="session_inventory")
    if session_inventory.is_empty():
        raise ValueError("session_inventory contains no sessions")
    if session_inventory.get_column(SOURCE_COLUMN).null_count():
        raise ValueError("session_inventory contains null NWB paths")
    if session_inventory.get_column(SOURCE_COLUMN).n_unique() != session_inventory.height:
        raise ValueError("session_inventory must contain one row per NWB path")
    if session_inventory.get_column("asset_id").n_unique() != session_inventory.height:
        raise ValueError("session_inventory must contain one row per asset ID")


def _require_columns(
    frame: pl.DataFrame,
    required: collections.abc.Iterable[str],
    *,
    frame_name: str,
) -> None:
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {sorted(missing)}")
