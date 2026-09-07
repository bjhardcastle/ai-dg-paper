"""Metadata-only Milestone 0 schema and behavioral-clock integrity audits.

The audits in this module inspect NWB path, schema, shape, dtype, and unit
metadata.  They deliberately do not slice array-valued datasets, load spike
times, calculate neural responses, or claim that timestamp vectors have been
validated.  Remote readers are injectable so normalization and gates can be
tested entirely with in-memory metadata fixtures.
"""

from __future__ import annotations

import collections.abc
import concurrent.futures
import dataclasses
import hashlib
import json
import pathlib
import re
from typing import Any

import lazynwb
import polars as pl

import dg.data

SOURCE_COLUMN = lazynwb.NWB_PATH_COLUMN_NAME


@dataclasses.dataclass(frozen=True)
class SchemaFieldExpectation:
    """Expected column name, requirement, and broad logical dtype family."""

    column_name: str
    logical_dtype: str
    required: bool = True


@dataclasses.dataclass(frozen=True)
class ExactTableSpec:
    """One logical table and its exact fixed or per-source-resolved NWB path."""

    table_id: str
    exact_path: str | None
    fields: tuple[SchemaFieldExpectation, ...]
    excluded_observed_columns: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ClockComponentSpec:
    """A metadata-only child dataset expected beneath a TimeSeries path."""

    name: str
    required_when_series_present: bool = True


@dataclasses.dataclass(frozen=True)
class ClockSeriesSpec:
    """One clock-bearing signal path audited in every session.

    ``shared_timestamps_path`` records NWB containers whose measurements use
    the timestamps of a sibling TimeSeries instead of owning a timestamps
    dataset.  This is an explicit structural relationship, not an inferred
    clock equivalence.
    """

    signal_id: str
    exact_path: str
    coverage_requirement: str
    components: tuple[ClockComponentSpec, ...] = (ClockComponentSpec("data"),)
    shared_timestamps_path: str | None = None

    def __post_init__(self) -> None:
        if self.coverage_requirement not in {"core", "optional"}:
            raise ValueError("coverage_requirement must be 'core' or 'optional'")


@dataclasses.dataclass(frozen=True)
class ClockMetadataAudit:
    """Normalized metadata-only clock coverage artifacts."""

    series: pl.DataFrame
    components: pl.DataFrame
    raw_video: pl.DataFrame


_UNIT_FIELDS = (
    SchemaFieldExpectation("id", "integer"),
    SchemaFieldExpectation("peak_channel_id", "integer"),
    SchemaFieldExpectation("spike_times", "list_numeric"),
    SchemaFieldExpectation("isi_violations", "numeric"),
    SchemaFieldExpectation("amplitude_cutoff", "numeric"),
    SchemaFieldExpectation("quality", "string"),
    SchemaFieldExpectation("presence_ratio", "numeric", required=False),
    SchemaFieldExpectation("snr", "numeric", required=False),
    SchemaFieldExpectation("structure_layer", "string", required=False),
)

_TRIAL_FIELDS = (
    SchemaFieldExpectation("id", "integer"),
    SchemaFieldExpectation("start_time", "numeric"),
    SchemaFieldExpectation("stop_time", "numeric"),
    SchemaFieldExpectation("change_time", "numeric"),
    SchemaFieldExpectation("change_frame", "numeric"),
    SchemaFieldExpectation("initial_image_name", "string"),
    SchemaFieldExpectation("change_image_name", "string"),
    SchemaFieldExpectation("is_change", "boolean"),
    SchemaFieldExpectation("go", "boolean"),
    SchemaFieldExpectation("catch", "boolean"),
    SchemaFieldExpectation("aborted", "boolean"),
    SchemaFieldExpectation("auto_rewarded", "boolean"),
    SchemaFieldExpectation("hit", "boolean"),
    SchemaFieldExpectation("miss", "boolean"),
    SchemaFieldExpectation("false_alarm", "boolean"),
    SchemaFieldExpectation("correct_reject", "boolean"),
    SchemaFieldExpectation("lick_times", "list_numeric"),
    SchemaFieldExpectation("is_sham_change", "boolean", required=False),
    SchemaFieldExpectation("no_reward_epoch", "boolean", required=False),
    SchemaFieldExpectation("omitted_reward", "boolean", required=False),
    SchemaFieldExpectation("response_latency", "numeric", required=False),
)

_ELECTRODE_FIELDS = (
    SchemaFieldExpectation("id", "integer"),
    SchemaFieldExpectation("location", "string"),
    SchemaFieldExpectation("x", "numeric"),
    SchemaFieldExpectation("y", "numeric"),
    SchemaFieldExpectation("z", "numeric"),
    SchemaFieldExpectation("probe_id", "integer"),
    SchemaFieldExpectation("probe_channel_number", "integer", required=False),
    SchemaFieldExpectation("probe_horizontal_position", "numeric", required=False),
    SchemaFieldExpectation("probe_vertical_position", "numeric", required=False),
    SchemaFieldExpectation("valid_data", "boolean", required=False),
)

TASK_PRESENTATION_FIELDS = (
    SchemaFieldExpectation("id", "integer"),
    SchemaFieldExpectation("start_time", "numeric"),
    SchemaFieldExpectation("stop_time", "numeric"),
    SchemaFieldExpectation("start_frame", "integer"),
    SchemaFieldExpectation("image_name", "string"),
    SchemaFieldExpectation("is_change", "boolean"),
    SchemaFieldExpectation("active", "boolean"),
    SchemaFieldExpectation("stimulus_name", "string", required=False),
    SchemaFieldExpectation("stimulus_block", "integer"),
    SchemaFieldExpectation("is_image_novel", "boolean"),
    SchemaFieldExpectation("flashes_since_change", "numeric"),
    SchemaFieldExpectation("rewarded", "boolean"),
    SchemaFieldExpectation("omitted", "boolean"),
)

DEFAULT_EXACT_TABLE_SPECS = (
    ExactTableSpec(
        "units",
        dg.data.UNITS_PATH,
        _UNIT_FIELDS,
        excluded_observed_columns=("firing_rate",),
    ),
    ExactTableSpec("trials", dg.data.TRIALS_PATH, _TRIAL_FIELDS),
    ExactTableSpec("electrodes", dg.data.ELECTRODES_PATH, _ELECTRODE_FIELDS),
)

TASK_PRESENTATION_TABLE_SPEC = ExactTableSpec(
    "task_image_presentations",
    None,
    TASK_PRESENTATION_FIELDS,
)

FROZEN_EXPECTED_SCHEMA_VARIANT_COUNTS = {
    "units": 1,
    "electrodes": 1,
    "task_image_presentations": 1,
    "trials": 2,
}

CORE_CLOCK_SERIES_SPECS = (
    ClockSeriesSpec("licks", "/processing/licking/licks", "core"),
    ClockSeriesSpec("reward_volume", "/processing/rewards/volume", "core"),
    ClockSeriesSpec("autorewarded", "/processing/rewards/autorewarded", "core"),
    ClockSeriesSpec("running_dx", "/processing/running/dx", "core"),
    ClockSeriesSpec("running_speed", "/processing/running/speed", "core"),
    ClockSeriesSpec(
        "running_speed_unfiltered",
        "/processing/running/speed_unfiltered",
        "core",
    ),
    ClockSeriesSpec("stimulus_timestamps", "/processing/stimulus/timestamps", "core"),
    ClockSeriesSpec("v_in", "/acquisition/v_in", "core"),
    ClockSeriesSpec("v_sig", "/acquisition/v_sig", "core"),
    ClockSeriesSpec(
        "optotagging",
        "/processing/optotagging/optotagging",
        "core",
    ),
)

_EYE_MEASUREMENT_COMPONENTS = tuple(
    ClockComponentSpec(name) for name in ("data", "angle", "area", "area_raw", "height", "width")
)

OPTIONAL_EYE_CLOCK_SERIES_SPECS = (
    ClockSeriesSpec(
        "eye_tracking",
        "/acquisition/EyeTracking/eye_tracking",
        "optional",
        _EYE_MEASUREMENT_COMPONENTS,
    ),
    ClockSeriesSpec(
        "pupil_tracking",
        "/acquisition/EyeTracking/pupil_tracking",
        "optional",
        _EYE_MEASUREMENT_COMPONENTS,
        shared_timestamps_path="/acquisition/EyeTracking/eye_tracking/timestamps",
    ),
    ClockSeriesSpec(
        "corneal_reflection_tracking",
        "/acquisition/EyeTracking/corneal_reflection_tracking",
        "optional",
        _EYE_MEASUREMENT_COMPONENTS,
        shared_timestamps_path="/acquisition/EyeTracking/eye_tracking/timestamps",
    ),
    ClockSeriesSpec(
        "likely_blink",
        "/acquisition/EyeTracking/likely_blink",
        "optional",
        shared_timestamps_path="/acquisition/EyeTracking/eye_tracking/timestamps",
    ),
)

DEFAULT_CLOCK_SERIES_SPECS = CORE_CLOCK_SERIES_SPECS + OPTIONAL_EYE_CLOCK_SERIES_SPECS

_LOGICAL_DTYPES = {
    "any",
    "integer",
    "float",
    "numeric",
    "boolean",
    "string",
    "temporal",
    "list_integer",
    "list_float",
    "list_numeric",
    "list_boolean",
    "list_string",
}
_RAW_VIDEO_PATTERN = re.compile(r"(?:^|[/_.-])(?:video|movie)(?:$|[/_.-])", re.IGNORECASE)
_RAW_VIDEO_SUFFIXES = {".avi", ".mkv", ".mov", ".mp4", ".webm"}

_SCHEMA_AUDIT_SCHEMA = {
    "asset_id": pl.String,
    "path": pl.String,
    "subject_id": pl.String,
    SOURCE_COLUMN: pl.String,
    "table_id": pl.String,
    "table_path": pl.String,
    "column_name": pl.String,
    "column_requirement": pl.String,
    "expected_logical_dtype": pl.String,
    "observed_physical_dtype": pl.String,
    "present": pl.Boolean,
    "dtype_compatible": pl.Boolean,
    "n_observed_columns": pl.Int64,
    "observed_schema_json": pl.String,
    "schema_variant_sha256": pl.String,
    "table_audit_status": pl.String,
    "audit_error": pl.String,
    "metadata_only": pl.Boolean,
    "array_values_loaded": pl.Boolean,
    "neural_outcomes_accessed": pl.Boolean,
}

_CLOCK_SERIES_AUDIT_SCHEMA = {
    "asset_id": pl.String,
    "path": pl.String,
    "subject_id": pl.String,
    "ecephys_session_id": pl.Int64,
    SOURCE_COLUMN: pl.String,
    "signal_id": pl.String,
    "timeseries_path": pl.String,
    "coverage_requirement": pl.String,
    "series_present": pl.Boolean,
    "classified_as_timeseries": pl.Boolean,
    "data_present": pl.Boolean,
    "data_shape": pl.String,
    "data_leading_length": pl.Int64,
    "data_dtype": pl.String,
    "data_unit": pl.String,
    "explicit_timestamps_present": pl.Boolean,
    "timestamps_shape": pl.String,
    "timestamps_leading_length": pl.Int64,
    "timestamps_dtype": pl.String,
    "timestamps_dtype_numeric": pl.Boolean,
    "timestamps_unit": pl.String,
    "starting_time_present": pl.Boolean,
    "starting_time_dtype": pl.String,
    "starting_time_dtype_numeric": pl.Boolean,
    "sampling_rate_hz": pl.Float64,
    "timing_representation": pl.String,
    "timing_source_path": pl.String,
    "leading_lengths_agree": pl.Boolean,
    "required_component_lengths_agree": pl.Boolean,
    "n_required_component_length_mismatches": pl.Int64,
    "series_metadata_status": pl.String,
    "audit_error": pl.String,
    "metadata_only": pl.Boolean,
    "array_values_loaded": pl.Boolean,
    "timestamp_vectors_validated": pl.Boolean,
    "full_vector_clock_status": pl.String,
}

_CLOCK_COMPONENT_AUDIT_SCHEMA = {
    "asset_id": pl.String,
    "path": pl.String,
    "subject_id": pl.String,
    "ecephys_session_id": pl.Int64,
    SOURCE_COLUMN: pl.String,
    "signal_id": pl.String,
    "timeseries_path": pl.String,
    "coverage_requirement": pl.String,
    "component_name": pl.String,
    "component_path": pl.String,
    "component_requirement": pl.String,
    "series_present": pl.Boolean,
    "present": pl.Boolean,
    "shape": pl.String,
    "physical_dtype": pl.String,
    "unit": pl.String,
    "audit_error": pl.String,
    "metadata_only": pl.Boolean,
    "array_values_loaded": pl.Boolean,
}

_RAW_VIDEO_AUDIT_SCHEMA = {
    "asset_id": pl.String,
    "path": pl.String,
    "subject_id": pl.String,
    "ecephys_session_id": pl.Int64,
    SOURCE_COLUMN: pl.String,
    "raw_video_internal_paths": pl.String,
    "n_raw_video_internal_paths": pl.Int64,
    "raw_video_timeseries_present": pl.Boolean,
    "raw_video_timeseries_status": pl.String,
    "candidate_detection_scope": pl.String,
    "audit_error": pl.String,
    "metadata_only": pl.Boolean,
    "video_values_loaded": pl.Boolean,
}


def read_exact_table_schema(source: str, exact_table_path: str) -> pl.Schema:
    """Read one exact table's schema without materializing any column values."""

    lazynwb.config.anon = True
    return lazynwb.get_table_schema(
        source,
        exact_table_path,
        exclude_array_columns=False,
        exclude_internal_columns=True,
        raise_on_missing=True,
    )


def canonical_physical_dtype(dtype: Any) -> str:
    """Return a deterministic human-readable physical dtype representation."""

    if dtype is None:
        raise ValueError("a physical dtype cannot be None")
    text = str(dtype).strip()
    if not text:
        raise ValueError("a physical dtype must have a non-empty representation")
    return text


def logical_dtype_compatible(observed_dtype: Any, expected_logical_dtype: str) -> bool:
    """Test a physical dtype against a deliberately broad logical dtype family."""

    if expected_logical_dtype not in _LOGICAL_DTYPES:
        raise ValueError(f"unsupported logical dtype: {expected_logical_dtype!r}")
    if expected_logical_dtype == "any":
        return True

    family, inner_family = _dtype_families(observed_dtype)
    if expected_logical_dtype.startswith("list_"):
        expected_inner = expected_logical_dtype.removeprefix("list_")
        return family == "list" and _scalar_family_compatible(inner_family, expected_inner)
    return _scalar_family_compatible(family, expected_logical_dtype)


def stable_schema_variant_sha256(
    exact_table_path: str,
    schema: collections.abc.Mapping[str, Any] | pl.Schema,
) -> tuple[str, str]:
    """Return canonical schema JSON and its stable SHA-256 variant identifier.

    The exact path is validated but deliberately excluded from the digest.
    Thus equivalent task-presentation schemas under different session-specific
    table names count as one logical schema variant.
    """

    if not exact_table_path.startswith("/"):
        raise ValueError("exact_table_path must be an absolute NWB path")

    normalized = [
        {"column_name": str(column), "physical_dtype": canonical_physical_dtype(dtype)}
        for column, dtype in sorted(dict(schema).items(), key=lambda item: str(item[0]))
    ]
    payload = {"schema_format": "nwb_physical_columns_v1", "columns": normalized}
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return canonical_json, hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def audit_exact_table_schema_metadata(
    session_inventory: pl.DataFrame,
    *,
    table_specs: collections.abc.Sequence[ExactTableSpec] = DEFAULT_EXACT_TABLE_SPECS,
    source_table_paths: pl.DataFrame | None = None,
    schema_reader: collections.abc.Callable[[str, str], collections.abc.Mapping[str, Any]] = (
        read_exact_table_schema
    ),
    max_workers: int = 8,
) -> pl.DataFrame:
    """Audit exact per-session schemas, including physical dtype drift.

    ``source_table_paths`` resolves specs whose ``exact_path`` is ``None`` and
    must contain exactly one ``(_nwb_path, table_id, table_path)`` row for each
    affected source.  Reader exceptions are retained as source-scoped rows so
    a failed remote read cannot masquerade as a missing optional field.
    """

    inventory = _validated_session_inventory(session_inventory)
    specs = _validate_table_specs(table_specs)
    requests = _resolve_exact_table_requests(inventory, specs, source_table_paths)
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    def audit_request(request: dict[str, Any]) -> list[dict[str, Any]]:
        spec = specs[request["table_id"]]
        return _audit_one_exact_table(request, spec, schema_reader)

    request_rows = requests.to_dicts()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        nested = list(executor.map(audit_request, request_rows))
    rows = [row for table_rows in nested for row in table_rows]
    return pl.DataFrame(rows, schema=_SCHEMA_AUDIT_SCHEMA).sort(
        "path", "table_id", "table_path", "column_name"
    )


def validate_exact_table_schema_audit(
    schema_audit: pl.DataFrame,
    session_inventory: pl.DataFrame,
    *,
    table_specs: collections.abc.Sequence[ExactTableSpec] = DEFAULT_EXACT_TABLE_SPECS,
    source_table_paths: pl.DataFrame | None = None,
    expected_schema_variant_counts: collections.abc.Mapping[str, int] | None = None,
    maximum_schema_variant_counts: collections.abc.Mapping[str, int] | None = None,
) -> None:
    """Fail unless every expected source/table has a compatible exact schema."""

    inventory = _validated_session_inventory(session_inventory)
    specs = _validate_table_specs(table_specs)
    requests = _resolve_exact_table_requests(inventory, specs, source_table_paths)
    _require_columns(schema_audit, set(_SCHEMA_AUDIT_SCHEMA), "schema_audit")
    expected_sources = set(inventory.get_column(SOURCE_COLUMN).to_list())
    observed_sources = set(schema_audit.get_column(SOURCE_COLUMN).unique().to_list())
    if observed_sources != expected_sources:
        raise ValueError("schema audit does not exactly cover session inventory sources")
    _validate_session_metadata_columns(schema_audit, inventory, "schema audit")

    expected_tables = set(requests.select(SOURCE_COLUMN, "table_id", "table_path").iter_rows())
    observed_tables = set(
        schema_audit.select(SOURCE_COLUMN, "table_id", "table_path").unique().iter_rows()
    )
    if observed_tables != expected_tables:
        raise ValueError("schema audit does not exactly cover requested source/table pairs")

    expected_rows: set[tuple[str, str, str, str]] = set()
    request_lookup = {
        (row[SOURCE_COLUMN], row["table_id"]): row["table_path"]
        for row in requests.iter_rows(named=True)
    }
    for (source, table_id), table_path in request_lookup.items():
        for field in specs[table_id].fields:
            expected_rows.add((source, table_id, table_path, field.column_name))
    expected_audit = schema_audit.filter(pl.col("column_requirement") != "unexpected")
    observed_rows = set(
        expected_audit.select(SOURCE_COLUMN, "table_id", "table_path", "column_name").iter_rows()
    )
    if observed_rows != expected_rows or expected_audit.height != len(expected_rows):
        raise ValueError("schema audit does not contain exactly one row per expected field")

    if schema_audit.filter(
        ~pl.col("metadata_only").fill_null(False)
        | pl.col("array_values_loaded").fill_null(True)
        | pl.col("neural_outcomes_accessed").fill_null(True)
    ).height:
        raise ValueError("schema audit violates its metadata-only/no-neural-outcome contract")

    successful_tables = schema_audit.filter(pl.col("audit_error").is_null())
    for _, table_rows in successful_tables.group_by(
        SOURCE_COLUMN, "table_id", "table_path", maintain_order=True
    ):
        metadata_rows = table_rows.select(
            "observed_schema_json", "schema_variant_sha256", "n_observed_columns"
        ).unique()
        if metadata_rows.height != 1:
            raise ValueError("schema audit table rows disagree on schema metadata")
        row = metadata_rows.row(0, named=True)
        try:
            payload = json.loads(row["observed_schema_json"])
            canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("schema audit contains invalid canonical schema JSON") from error
        expected_digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        if row["schema_variant_sha256"] != expected_digest:
            raise ValueError("schema audit contains an invalid schema variant SHA-256")
        columns = payload.get("columns")
        if not isinstance(columns, list):
            raise ValueError("schema audit canonical JSON has no column list")
        physical_dtypes = {
            item.get("column_name"): item.get("physical_dtype")
            for item in columns
            if isinstance(item, collections.abc.Mapping)
        }
        present_rows = table_rows.filter(pl.col("present").fill_null(False))
        observed_physical_dtypes = dict(
            present_rows.select("column_name", "observed_physical_dtype").iter_rows()
        )
        if (
            len(physical_dtypes) != len(columns)
            or physical_dtypes != observed_physical_dtypes
            or row["n_observed_columns"] != len(columns)
        ):
            raise ValueError("schema audit field rows disagree with canonical schema JSON")

    failures = (
        schema_audit.filter(
            pl.col("audit_error").is_not_null()
            | ~pl.col("table_audit_status").eq("pass").fill_null(False)
        )
        .select(
            SOURCE_COLUMN,
            "table_id",
            "table_path",
            "table_audit_status",
            "audit_error",
        )
        .unique()
    )
    if failures.height:
        raise ValueError(f"required exact-table schema audit failed:\n{failures.head(10)}")
    _validate_schema_variant_counts(
        schema_audit,
        specs,
        expected_counts=expected_schema_variant_counts,
        maximum_counts=maximum_schema_variant_counts,
    )


def summarize_exact_table_schema_audit(schema_audit: pl.DataFrame) -> pl.DataFrame:
    """Summarize presence, compatibility, and schema variants by exact field."""

    _require_columns(schema_audit, set(_SCHEMA_AUDIT_SCHEMA), "schema_audit")
    return (
        schema_audit.group_by(
            "table_id",
            "table_path",
            "column_name",
            "column_requirement",
            "expected_logical_dtype",
        )
        .agg(
            pl.col(SOURCE_COLUMN).n_unique().alias("n_sources_audited"),
            pl.col("present").fill_null(False).sum().alias("n_sources_present"),
            (~pl.col("present").fill_null(False)).sum().alias("n_sources_missing_or_unread"),
            pl.col("dtype_compatible").fill_null(False).sum().alias("n_sources_compatible"),
            (pl.col("present").fill_null(False) & ~pl.col("dtype_compatible").fill_null(False))
            .sum()
            .alias("n_sources_incompatible"),
            pl.col("audit_error").is_not_null().sum().alias("n_sources_read_error"),
            pl.col("schema_variant_sha256").drop_nulls().n_unique().alias("n_schema_variants"),
            pl.col("schema_variant_sha256")
            .drop_nulls()
            .unique()
            .sort()
            .str.join(";")
            .alias("schema_variant_sha256s"),
            pl.col("observed_physical_dtype")
            .drop_nulls()
            .unique()
            .sort()
            .str.join(";")
            .alias("observed_physical_dtypes"),
        )
        .with_columns(
            pl.when(pl.col("n_sources_read_error") > 0)
            .then(pl.lit("read_error"))
            .when(pl.col("n_sources_incompatible") > 0)
            .then(pl.lit("dtype_incompatible"))
            .when(
                (pl.col("column_requirement") == "required")
                & (pl.col("n_sources_missing_or_unread") > 0)
            )
            .then(pl.lit("required_missing"))
            .when(pl.col("n_sources_missing_or_unread") > 0)
            .then(pl.lit("optional_missing"))
            .otherwise(pl.lit("pass"))
            .alias("field_audit_status")
        )
        .sort("table_id", "table_path", "column_name")
    )


def read_clock_source_metadata(
    source: str,
    series_specs: collections.abc.Sequence[ClockSeriesSpec] = DEFAULT_CLOCK_SERIES_SPECS,
) -> dict[str, Any]:
    """Read path/accessor metadata for clock-relevant TimeSeries without slicing arrays."""

    lazynwb.config.anon = True
    path_info = lazynwb.get_internal_path_info(
        source,
        include_child_datasets=True,
        include_table_columns=False,
        include_metadata=False,
        parents=True,
    )
    representative_timeseries_path = next(
        (
            path
            for path, info in path_info.items()
            if isinstance(info, collections.abc.Mapping) and info.get("is_timeseries")
        ),
        None,
    )
    if representative_timeseries_path is None:
        raise ValueError("clock metadata discovery found no TimeSeries paths")
    # lazynwb 1.0.0dev8 exposes the shared FileAccessor as ``_file``. Retain
    # exactly one accessor per source so timing-dtype reads do not reopen the
    # remote store for every signal/component; no dataset values are sliced.
    with lazynwb.TimeSeries(source, representative_timeseries_path)._file as metadata_file:
        return _normalize_clock_path_metadata(path_info, metadata_file, series_specs)


def _normalize_clock_path_metadata(
    path_info: collections.abc.Mapping[str, collections.abc.Mapping[str, Any]],
    metadata_file: Any,
    series_specs: collections.abc.Sequence[ClockSeriesSpec],
) -> dict[str, Any]:
    """Normalize one discovered path catalog using one managed file accessor."""

    series_metadata: dict[str, Any] = {}
    component_metadata_cache: dict[str, dict[str, Any]] = {}
    for spec in series_specs:
        root_info = path_info.get(spec.exact_path)
        components: dict[str, Any] = {}
        component_names = {component.name for component in spec.components} | {
            "timestamps",
            "starting_time",
        }
        for component_name in sorted(component_names):
            component_path = _clock_component_path(spec, component_name)
            if component_path in component_metadata_cache:
                components[component_name] = dict(component_metadata_cache[component_path])
                continue
            info = path_info.get(component_path)
            metadata_dtype = None if info is None else info.get("dtype")
            component = {
                "present": info is not None,
                "shape": None if info is None else info.get("shape"),
                "dtype": (
                    None if metadata_dtype is None else canonical_physical_dtype(metadata_dtype)
                ),
                "unit": _metadata_unit(info),
                "attrs": {} if info is None else _json_safe_attrs(info.get("attrs")),
            }
            # Shape/attrs come from the one catalog traversal. Only timing
            # datasets need an additional accessor lookup because their dtype
            # participates in the integrity gate; other dtypes are descriptive.
            if info is not None and component_name in {"timestamps", "starting_time"}:
                try:
                    accessor = metadata_file[component_path]
                    component["shape"] = getattr(accessor, "shape", component["shape"])
                    dtype = getattr(accessor, "dtype", None)
                    if dtype is not None:
                        component["dtype"] = canonical_physical_dtype(dtype)
                    component["unit"] = _accessor_unit(accessor) or component["unit"]
                except (AttributeError, KeyError):
                    # The internal path still establishes presence.  Missing accessor
                    # conveniences only limit dtype/unit metadata coverage.
                    pass
            component_metadata_cache[component_path] = dict(component)
            components[component_name] = component
        starting_attrs = components.get("starting_time", {}).get("attrs", {})
        series_metadata[spec.exact_path] = {
            "present": root_info is not None,
            "is_timeseries": bool(root_info and root_info.get("is_timeseries")),
            "attrs": {} if root_info is None else _json_safe_attrs(root_info.get("attrs")),
            "components": components,
            "sampling_rate_hz": _finite_float_or_none(starting_attrs.get("rate")),
        }

    raw_video_paths = sorted(path for path in path_info if _is_raw_video_path(path))
    return {"series": series_metadata, "raw_video_internal_paths": raw_video_paths}


def audit_behavior_acquisition_clock_metadata(
    session_inventory: pl.DataFrame,
    *,
    series_specs: collections.abc.Sequence[ClockSeriesSpec] = DEFAULT_CLOCK_SERIES_SPECS,
    metadata_reader: collections.abc.Callable[
        [str, collections.abc.Sequence[ClockSeriesSpec]], collections.abc.Mapping[str, Any]
    ] = read_clock_source_metadata,
    max_workers: int = 8,
) -> ClockMetadataAudit:
    """Audit clock-relevant path and accessor metadata for every frozen session.

    This is intentionally not a full clock audit: timestamp vector values are
    not read, so monotonicity, duplicates, dropped samples, offsets, and
    row-wise event alignment remain unvalidated.
    """

    inventory = _validated_session_inventory(session_inventory)
    specs = _validate_clock_specs(series_specs)
    if max_workers < 1:
        raise ValueError("max_workers must be positive")

    def read_one(session: dict[str, Any]) -> tuple[dict[str, Any], Any, str | None]:
        try:
            metadata = metadata_reader(session[SOURCE_COLUMN], tuple(specs.values()))
            if not isinstance(metadata, collections.abc.Mapping):
                raise TypeError("clock metadata reader must return a mapping")
            return session, metadata, None
        except Exception as error:  # pragma: no cover - remote errors use same pure path
            return session, {}, f"{type(error).__name__}: {error}"

    sessions = _clock_session_metadata(inventory).to_dicts()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        read_results = list(executor.map(read_one, sessions))

    series_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    raw_video_rows: list[dict[str, Any]] = []
    for session, metadata, audit_error in read_results:
        normalized = _normalize_clock_reader_result(metadata)
        by_path = normalized["series"]
        for spec in specs.values():
            one_series, one_components = _normalize_one_clock_series(
                session,
                spec,
                by_path.get(spec.exact_path, {}),
                audit_error,
            )
            series_rows.append(one_series)
            component_rows.extend(one_components)
        candidates = normalized["raw_video_internal_paths"]
        raw_video_rows.append(
            {
                **session,
                "raw_video_internal_paths": _json_text(candidates),
                "n_raw_video_internal_paths": len(candidates),
                "raw_video_timeseries_present": bool(candidates) if audit_error is None else None,
                "raw_video_timeseries_status": (
                    "metadata_read_error"
                    if audit_error is not None
                    else "present"
                    if candidates
                    else "no_name_matched_candidate_in_session_nwb"
                ),
                "candidate_detection_scope": (
                    "case_insensitive_internal_path_token_regex_video_or_movie"
                ),
                "audit_error": audit_error,
                "metadata_only": True,
                "video_values_loaded": False,
            }
        )

    return ClockMetadataAudit(
        series=pl.DataFrame(series_rows, schema=_CLOCK_SERIES_AUDIT_SCHEMA).sort(
            "path", "signal_id"
        ),
        components=pl.DataFrame(component_rows, schema=_CLOCK_COMPONENT_AUDIT_SCHEMA).sort(
            "path", "signal_id", "component_name"
        ),
        raw_video=pl.DataFrame(raw_video_rows, schema=_RAW_VIDEO_AUDIT_SCHEMA).sort("path"),
    )


def validate_behavior_acquisition_clock_metadata(
    audit: ClockMetadataAudit,
    session_inventory: pl.DataFrame,
    *,
    series_specs: collections.abc.Sequence[ClockSeriesSpec] = DEFAULT_CLOCK_SERIES_SPECS,
    expected_optional_absent_sources: collections.abc.Mapping[str, collections.abc.Collection[str]]
    | None = None,
) -> None:
    """Gate core path metadata and optionally verify known optional absences.

    Optional absence expectations are generic mappings from ``signal_id`` to
    NWB source collections.  No dataset-specific missing-session outcomes are
    embedded in this module.
    """

    inventory = _validated_session_inventory(session_inventory)
    specs = _validate_clock_specs(series_specs)
    _require_columns(audit.series, set(_CLOCK_SERIES_AUDIT_SCHEMA), "clock series audit")
    _require_columns(
        audit.components,
        set(_CLOCK_COMPONENT_AUDIT_SCHEMA),
        "clock component audit",
    )
    _require_columns(audit.raw_video, set(_RAW_VIDEO_AUDIT_SCHEMA), "raw video audit")
    expected_sources = set(inventory.get_column(SOURCE_COLUMN).to_list())
    for frame_name, frame in (
        ("clock series audit", audit.series),
        ("clock component audit", audit.components),
        ("raw video audit", audit.raw_video),
    ):
        if set(frame.get_column(SOURCE_COLUMN).unique().to_list()) != expected_sources:
            raise ValueError(f"{frame_name} does not exactly cover session inventory sources")
        _validate_session_metadata_columns(frame, inventory, frame_name)

    expected_series_keys = {
        (source, signal_id) for source in expected_sources for signal_id in specs
    }
    observed_series_keys = set(audit.series.select(SOURCE_COLUMN, "signal_id").iter_rows())
    if observed_series_keys != expected_series_keys or audit.series.height != len(
        expected_series_keys
    ):
        raise ValueError("clock series audit does not have one row per source/signal")

    expected_component_keys: set[tuple[str, str, str]] = set()
    for source in expected_sources:
        for spec in specs.values():
            component_names = {component.name for component in spec.components} | {
                "timestamps",
                "starting_time",
            }
            expected_component_keys.update(
                (source, spec.signal_id, component_name) for component_name in component_names
            )
    observed_component_keys = set(
        audit.components.select(SOURCE_COLUMN, "signal_id", "component_name").iter_rows()
    )
    if observed_component_keys != expected_component_keys or audit.components.height != len(
        expected_component_keys
    ):
        raise ValueError("clock component audit does not have exact expected coverage")

    if audit.raw_video.height != len(expected_sources) or audit.raw_video.get_column(
        SOURCE_COLUMN
    ).n_unique() != len(expected_sources):
        raise ValueError("raw video audit does not have one row per source")

    if audit.series.filter(
        ~pl.col("metadata_only").fill_null(False)
        | pl.col("array_values_loaded").fill_null(True)
        | pl.col("timestamp_vectors_validated").fill_null(True)
        | (pl.col("full_vector_clock_status") != "not_validated_metadata_only")
    ).height:
        raise ValueError("clock audit overstates or violates its metadata-only scope")

    timing_source_errors: list[tuple[str, str, str | None, str | None]] = []
    for row in audit.series.select(
        "signal_id", "timing_representation", "timing_source_path", SOURCE_COLUMN
    ).iter_rows(named=True):
        spec = specs[row["signal_id"]]
        representation = row["timing_representation"]
        expected_path = (
            _clock_component_path(spec, "timestamps")
            if representation in {"explicit_timestamps", "shared_explicit_timestamps"}
            else _clock_component_path(spec, "starting_time")
            if representation == "rate_derived"
            else None
        )
        expected_representation = (
            "shared_explicit_timestamps"
            if spec.shared_timestamps_path is not None
            else "explicit_timestamps"
        )
        invalid_representation = (
            representation
            in {
                "explicit_timestamps",
                "shared_explicit_timestamps",
            }
            and representation != expected_representation
        )
        if invalid_representation or row["timing_source_path"] != expected_path:
            timing_source_errors.append(
                (
                    row[SOURCE_COLUMN],
                    row["signal_id"],
                    representation,
                    row["timing_source_path"],
                )
            )
    if timing_source_errors:
        raise ValueError(
            "clock audit contains inconsistent timing-source provenance: "
            f"{timing_source_errors[:5]!r}"
        )
    if audit.components.filter(
        ~pl.col("metadata_only").fill_null(False) | pl.col("array_values_loaded").fill_null(True)
    ).height:
        raise ValueError("clock component audit violates its metadata-only scope")
    if audit.raw_video.filter(
        ~pl.col("metadata_only").fill_null(False) | pl.col("video_values_loaded").fill_null(True)
    ).height:
        raise ValueError("raw video audit violates its metadata-only scope")

    integrity_failures = audit.series.filter(
        ~pl.col("series_metadata_status").is_in(["metadata_present", "optional_absent"])
    )
    if integrity_failures.height:
        raise ValueError(
            f"behavioral/acquisition clock metadata integrity failed:\n{integrity_failures}"
        )

    optional_expectations = expected_optional_absent_sources or {}
    unknown_signals = set(optional_expectations).difference(specs)
    if unknown_signals:
        raise ValueError(
            f"optional absence expectations contain unknown signals: {unknown_signals}"
        )
    for signal_id in optional_expectations:
        if specs[signal_id].coverage_requirement != "optional":
            raise ValueError(f"absence expectation for non-optional signal {signal_id!r}")
    for signal_id, spec in specs.items():
        if spec.coverage_requirement != "optional":
            continue
        expected_absent = optional_expectations.get(signal_id, ())
        expected_absent_set = set(expected_absent)
        if not expected_absent_set.issubset(expected_sources):
            raise ValueError(f"absence expectation for {signal_id!r} contains unknown sources")
        observed_absent = set(
            audit.series.filter(
                (pl.col("signal_id") == signal_id)
                & ~pl.col("series_present").fill_null(False)
                & pl.col("audit_error").is_null()
            )
            .get_column(SOURCE_COLUMN)
            .to_list()
        )
        if observed_absent != expected_absent_set:
            raise ValueError(
                f"optional absence coverage for {signal_id!r} differs: "
                f"expected {sorted(expected_absent_set)!r}, observed {sorted(observed_absent)!r}"
            )


def summarize_behavior_acquisition_clock_metadata(
    series_audit: pl.DataFrame,
) -> pl.DataFrame:
    """Summarize core/optional path and cheap metadata coverage by signal."""

    _require_columns(series_audit, set(_CLOCK_SERIES_AUDIT_SCHEMA), "clock series audit")
    return (
        series_audit.group_by("signal_id", "timeseries_path", "coverage_requirement")
        .agg(
            pl.col(SOURCE_COLUMN).n_unique().alias("n_sources_audited"),
            pl.col("series_present").fill_null(False).sum().alias("n_sources_present"),
            (~pl.col("series_present").fill_null(False)).sum().alias("n_sources_absent_or_unread"),
            pl.col("audit_error").is_not_null().sum().alias("n_sources_read_error"),
            pl.col("timing_representation")
            .is_in(["explicit_timestamps", "shared_explicit_timestamps"])
            .sum()
            .alias("n_sources_explicit_timestamps"),
            (pl.col("timing_representation") == "shared_explicit_timestamps")
            .sum()
            .alias("n_sources_shared_explicit_timestamps"),
            (pl.col("timing_representation") == "rate_derived")
            .sum()
            .alias("n_sources_rate_derived"),
            (
                pl.col("leading_lengths_agree").eq(False)
                | pl.col("required_component_lengths_agree").eq(False)
            )
            .fill_null(False)
            .sum()
            .alias("n_sources_length_mismatch"),
            pl.col("data_dtype").drop_nulls().unique().sort().str.join(";").alias("data_dtypes"),
            pl.col("timestamps_dtype")
            .drop_nulls()
            .unique()
            .sort()
            .str.join(";")
            .alias("timestamp_dtypes"),
        )
        .with_columns(
            pl.when(pl.col("n_sources_read_error") > 0)
            .then(pl.lit("read_error"))
            .when(
                (pl.col("coverage_requirement") == "core")
                & (pl.col("n_sources_absent_or_unread") > 0)
            )
            .then(pl.lit("core_missing"))
            .when(pl.col("n_sources_length_mismatch") > 0)
            .then(pl.lit("metadata_length_mismatch"))
            .when(pl.col("n_sources_absent_or_unread") > 0)
            .then(pl.lit("optional_missing"))
            .otherwise(pl.lit("metadata_present"))
            .alias("coverage_status"),
            pl.lit("not_validated_metadata_only").alias("full_vector_clock_status"),
        )
        .sort("coverage_requirement", "signal_id")
    )


def audit_raw_video_asset_absence(asset_inventory: pl.DataFrame) -> pl.DataFrame:
    """Record raw-video candidate absence/presence in the frozen asset manifest."""

    _require_columns(asset_inventory, {"path"}, "asset_inventory")
    paths = asset_inventory.get_column("path").cast(pl.String).to_list()
    if any(path is None or not path for path in paths):
        raise ValueError("asset_inventory contains a null or empty path")
    candidates = sorted(path for path in paths if _is_raw_video_asset_path(path))
    return pl.DataFrame(
        [
            {
                "n_assets_audited": asset_inventory.height,
                "n_raw_video_asset_candidates": len(candidates),
                "raw_video_asset_paths": _json_text(candidates),
                "raw_video_asset_status": (
                    "present_in_frozen_asset_inventory"
                    if candidates
                    else "no_name_matched_candidate_in_frozen_asset_inventory"
                ),
                "candidate_detection_scope": (
                    "case_insensitive_asset_path_token_regex_video_or_movie"
                ),
                "metadata_only": True,
                "asset_values_loaded": False,
            }
        ]
    )


def _audit_one_exact_table(
    request: dict[str, Any],
    spec: ExactTableSpec,
    schema_reader: collections.abc.Callable[[str, str], collections.abc.Mapping[str, Any]],
) -> list[dict[str, Any]]:
    common = {key: request[key] for key in ("asset_id", "path", "subject_id", SOURCE_COLUMN)}
    table_path = request["table_path"]
    schema: dict[str, Any] = {}
    audit_error: str | None = request.get("resolution_error")
    if audit_error is None:
        try:
            schema_result = schema_reader(request[SOURCE_COLUMN], table_path)
            if not isinstance(schema_result, collections.abc.Mapping):
                raise TypeError("schema reader must return a mapping or polars Schema")
            schema = dict(schema_result)
            if any(not isinstance(column, str) or not column for column in schema):
                raise ValueError("observed schema column names must be non-empty strings")
            # Scalar activity outcomes are outside this metadata-only milestone.
            # Exclude even their dtype/name from the published schema artifact;
            # no values from these fields are ever requested.
            schema = {
                column: dtype
                for column, dtype in schema.items()
                if column not in spec.excluded_observed_columns
            }
        except Exception as error:
            audit_error = f"{type(error).__name__}: {error}"

    if audit_error is None:
        canonical_json, variant_hash = stable_schema_variant_sha256(table_path, schema)
    else:
        canonical_json = None
        variant_hash = None
    expectations = {field.column_name: field for field in spec.fields}
    missing_required = any(
        field.required and field.column_name not in schema for field in spec.fields
    )
    incompatible = any(
        field.column_name in schema
        and not logical_dtype_compatible(schema[field.column_name], field.logical_dtype)
        for field in spec.fields
    )
    table_status = (
        "read_error"
        if audit_error is not None
        else "required_missing"
        if missing_required
        else "dtype_incompatible"
        if incompatible
        else "pass"
    )

    output: list[dict[str, Any]] = []
    columns = sorted(set(expectations) | set(schema))
    for column in columns:
        expectation = expectations.get(column)
        present = None if audit_error is not None else column in schema
        output.append(
            {
                **common,
                "table_id": spec.table_id,
                "table_path": table_path,
                "column_name": column,
                "column_requirement": (
                    "unexpected"
                    if expectation is None
                    else "required"
                    if expectation.required
                    else "optional"
                ),
                "expected_logical_dtype": (
                    None if expectation is None else expectation.logical_dtype
                ),
                "observed_physical_dtype": (
                    canonical_physical_dtype(schema[column]) if column in schema else None
                ),
                "present": present,
                "dtype_compatible": (
                    None
                    if expectation is None or not present
                    else logical_dtype_compatible(schema[column], expectation.logical_dtype)
                ),
                "n_observed_columns": None if audit_error is not None else len(schema),
                "observed_schema_json": canonical_json,
                "schema_variant_sha256": variant_hash,
                "table_audit_status": table_status,
                "audit_error": audit_error,
                "metadata_only": True,
                "array_values_loaded": False,
                "neural_outcomes_accessed": False,
            }
        )
    return output


def _normalize_one_clock_series(
    session: dict[str, Any],
    spec: ClockSeriesSpec,
    raw_metadata: Any,
    audit_error: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = raw_metadata if isinstance(raw_metadata, collections.abc.Mapping) else {}
    components_value = metadata.get("components", {})
    components = components_value if isinstance(components_value, collections.abc.Mapping) else {}
    series_present = bool(metadata.get("present", False)) if audit_error is None else None
    classified = bool(metadata.get("is_timeseries", False)) if audit_error is None else None
    component_specs = {component.name: component for component in spec.components}
    component_names = set(component_specs) | {"timestamps", "starting_time"}
    component_rows: list[dict[str, Any]] = []
    normalized_components: dict[str, dict[str, Any]] = {}
    for component_name in sorted(component_names):
        raw_component = components.get(component_name, {})
        component = raw_component if isinstance(raw_component, collections.abc.Mapping) else {}
        present = bool(component.get("present", False)) if audit_error is None else None
        normalized_components[component_name] = {
            "present": present,
            "shape": _normalize_shape(component.get("shape")),
            "dtype": _optional_dtype(component.get("dtype")),
            "unit": _optional_text(component.get("unit")),
        }
        expected = component_specs.get(component_name)
        component_rows.append(
            {
                **session,
                "signal_id": spec.signal_id,
                "timeseries_path": spec.exact_path,
                "coverage_requirement": spec.coverage_requirement,
                "component_name": component_name,
                "component_path": _clock_component_path(spec, component_name),
                "component_requirement": (
                    "shared_timing_source"
                    if component_name == "timestamps" and spec.shared_timestamps_path is not None
                    else "timing_alternative"
                    if component_name in {"timestamps", "starting_time"}
                    else "required_when_series_present"
                    if expected is not None and expected.required_when_series_present
                    else "metadata_if_present"
                ),
                "series_present": series_present,
                "present": present,
                "shape": normalized_components[component_name]["shape"],
                "physical_dtype": normalized_components[component_name]["dtype"],
                "unit": normalized_components[component_name]["unit"],
                "audit_error": audit_error,
                "metadata_only": True,
                "array_values_loaded": False,
            }
        )

    data = normalized_components.get("data", {})
    timestamps = normalized_components["timestamps"]
    starting_time = normalized_components["starting_time"]
    rate = _finite_float_or_none(metadata.get("sampling_rate_hz"))
    if audit_error is not None:
        timing_representation = "metadata_read_error"
    elif not series_present:
        timing_representation = "series_absent"
    elif timestamps.get("present"):
        timing_representation = (
            "shared_explicit_timestamps"
            if spec.shared_timestamps_path is not None
            else "explicit_timestamps"
        )
    elif starting_time.get("present") and rate is not None and rate > 0:
        timing_representation = "rate_derived"
    else:
        timing_representation = "missing"

    if timing_representation in {"explicit_timestamps", "shared_explicit_timestamps"}:
        timing_source_path = _clock_component_path(spec, "timestamps")
    elif timing_representation == "rate_derived":
        timing_source_path = _clock_component_path(spec, "starting_time")
    else:
        timing_source_path = None

    required_components_missing = bool(series_present) and any(
        component.required_when_series_present
        and not normalized_components[component.name]["present"]
        for component in spec.components
    )
    data_length = _leading_length_from_json_shape(data.get("shape"))
    timestamp_length = _leading_length_from_json_shape(timestamps.get("shape"))
    required_component_shape_missing = bool(series_present) and any(
        component.required_when_series_present
        and (
            (
                length := _leading_length_from_json_shape(
                    normalized_components[component.name]["shape"]
                )
            )
            is None
            or length <= 0
        )
        for component in spec.components
    )
    timestamps_dtype = timestamps.get("dtype")
    timestamps_dtype_numeric = (
        logical_dtype_compatible(timestamps_dtype, "numeric")
        if timestamps_dtype is not None
        else None
    )
    starting_time_dtype = starting_time.get("dtype")
    starting_time_dtype_numeric = (
        logical_dtype_compatible(starting_time_dtype, "numeric")
        if starting_time_dtype is not None
        else None
    )
    leading_lengths_agree = (
        data_length == timestamp_length
        if data_length is not None and timestamp_length is not None
        else None
    )
    timing_length = (
        timestamp_length
        if timing_representation in {"explicit_timestamps", "shared_explicit_timestamps"}
        else data_length
        if timing_representation == "rate_derived"
        else None
    )
    required_component_lengths = [
        _leading_length_from_json_shape(normalized_components[component.name]["shape"])
        for component in spec.components
        if component.required_when_series_present
    ]
    component_lengths_comparable = (
        bool(series_present)
        and timing_length is not None
        and timing_length > 0
        and all(length is not None and length > 0 for length in required_component_lengths)
    )
    n_required_component_length_mismatches = (
        sum(length != timing_length for length in required_component_lengths)
        if component_lengths_comparable
        else None
    )
    required_component_lengths_agree = (
        n_required_component_length_mismatches == 0
        if n_required_component_length_mismatches is not None
        else None
    )
    if audit_error is not None:
        status = "metadata_read_error"
    elif not series_present:
        status = "core_path_missing" if spec.coverage_requirement == "core" else "optional_absent"
    elif not classified and spec.shared_timestamps_path is None:
        status = "not_classified_as_timeseries"
    elif required_components_missing:
        status = "required_component_missing"
    elif timing_representation == "missing":
        status = "timing_metadata_missing"
    elif required_component_shape_missing:
        status = "required_component_shape_missing_scalar_or_empty"
    elif timing_representation in {"explicit_timestamps", "shared_explicit_timestamps"} and (
        timestamp_length is None or timestamp_length <= 0
    ):
        status = "timestamp_shape_missing_scalar_or_empty"
    elif timing_representation in {"explicit_timestamps", "shared_explicit_timestamps"} and (
        not timestamps_dtype_numeric
    ):
        status = "timestamp_dtype_missing_or_nonnumeric"
    elif timing_representation == "rate_derived" and not starting_time_dtype_numeric:
        status = "starting_time_dtype_missing_or_nonnumeric"
    elif required_component_lengths_agree is False:
        status = "required_component_length_mismatch"
    elif leading_lengths_agree is False:
        status = "metadata_length_mismatch"
    else:
        status = "metadata_present"

    root_attrs = metadata.get("attrs", {})
    root_attrs = root_attrs if isinstance(root_attrs, collections.abc.Mapping) else {}
    series_row = {
        **session,
        "signal_id": spec.signal_id,
        "timeseries_path": spec.exact_path,
        "coverage_requirement": spec.coverage_requirement,
        "series_present": series_present,
        "classified_as_timeseries": classified,
        "data_present": data.get("present"),
        "data_shape": data.get("shape"),
        "data_leading_length": data_length,
        "data_dtype": data.get("dtype"),
        "data_unit": data.get("unit") or _optional_text(root_attrs.get("unit")),
        "explicit_timestamps_present": timestamps.get("present"),
        "timestamps_shape": timestamps.get("shape"),
        "timestamps_leading_length": timestamp_length,
        "timestamps_dtype": timestamps_dtype,
        "timestamps_dtype_numeric": timestamps_dtype_numeric,
        "timestamps_unit": timestamps.get("unit")
        or _optional_text(root_attrs.get("timestamps_unit")),
        "starting_time_present": starting_time.get("present"),
        "starting_time_dtype": starting_time_dtype,
        "starting_time_dtype_numeric": starting_time_dtype_numeric,
        "sampling_rate_hz": rate,
        "timing_representation": timing_representation,
        "timing_source_path": timing_source_path,
        "leading_lengths_agree": leading_lengths_agree,
        "required_component_lengths_agree": required_component_lengths_agree,
        "n_required_component_length_mismatches": n_required_component_length_mismatches,
        "series_metadata_status": status,
        "audit_error": audit_error,
        "metadata_only": True,
        "array_values_loaded": False,
        "timestamp_vectors_validated": False,
        "full_vector_clock_status": "not_validated_metadata_only",
    }
    return series_row, component_rows


def _validated_session_inventory(session_inventory: pl.DataFrame) -> pl.DataFrame:
    required = {"asset_id", "path", "subject_id", SOURCE_COLUMN}
    _require_columns(session_inventory, required, "session_inventory")
    if session_inventory.is_empty():
        raise ValueError("session_inventory contains no sessions")
    if session_inventory.select(*sorted(required)).null_count().sum_horizontal().item():
        raise ValueError("session_inventory contains null required metadata")
    if session_inventory.get_column(SOURCE_COLUMN).n_unique() != session_inventory.height:
        raise ValueError("session_inventory must have one row per NWB source")
    if session_inventory.get_column("asset_id").n_unique() != session_inventory.height:
        raise ValueError("session_inventory must have unique asset IDs")
    return session_inventory.sort("path")


def _clock_session_metadata(session_inventory: pl.DataFrame) -> pl.DataFrame:
    columns = ["asset_id", "path", "subject_id", SOURCE_COLUMN]
    metadata = session_inventory.select(*columns)
    if "ecephys_session_id" in session_inventory.columns:
        metadata = metadata.with_columns(
            session_inventory.get_column("ecephys_session_id").cast(pl.Int64)
        )
    else:
        metadata = metadata.with_columns(pl.lit(None, dtype=pl.Int64).alias("ecephys_session_id"))
    return metadata.select("asset_id", "path", "subject_id", "ecephys_session_id", SOURCE_COLUMN)


def _validate_table_specs(
    table_specs: collections.abc.Sequence[ExactTableSpec],
) -> dict[str, ExactTableSpec]:
    if not table_specs:
        raise ValueError("at least one exact table spec is required")
    specs: dict[str, ExactTableSpec] = {}
    for spec in table_specs:
        if not spec.table_id or spec.table_id in specs:
            raise ValueError("exact table specs require unique non-empty table IDs")
        if spec.exact_path is not None and not spec.exact_path.startswith("/"):
            raise ValueError("fixed table paths must be absolute NWB paths")
        if not spec.fields:
            raise ValueError(f"exact table spec {spec.table_id!r} has no fields")
        names = [field.column_name for field in spec.fields]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError(f"exact table spec {spec.table_id!r} has invalid field names")
        excluded = spec.excluded_observed_columns
        if (
            any(not name for name in excluded)
            or len(excluded) != len(set(excluded))
            or set(excluded).intersection(names)
        ):
            raise ValueError(
                f"exact table spec {spec.table_id!r} has invalid excluded observed columns"
            )
        invalid_logical = {
            field.logical_dtype
            for field in spec.fields
            if field.logical_dtype not in _LOGICAL_DTYPES
        }
        if invalid_logical:
            raise ValueError(f"unsupported logical dtypes: {sorted(invalid_logical)!r}")
        specs[spec.table_id] = spec
    return specs


def _validate_schema_variant_counts(
    schema_audit: pl.DataFrame,
    specs: collections.abc.Mapping[str, ExactTableSpec],
    *,
    expected_counts: collections.abc.Mapping[str, int] | None,
    maximum_counts: collections.abc.Mapping[str, int] | None,
) -> None:
    expected_counts = expected_counts or {}
    maximum_counts = maximum_counts or {}
    configured_ids = set(expected_counts) | set(maximum_counts)
    unknown = configured_ids.difference(specs)
    if unknown:
        raise ValueError(f"schema variant limits contain unknown table IDs: {sorted(unknown)!r}")
    invalid = {
        table_id: count
        for configured_counts in (expected_counts, maximum_counts)
        for table_id, count in configured_counts.items()
        if isinstance(count, bool) or not isinstance(count, int) or count < 1
    }
    if invalid:
        raise ValueError(f"schema variant limits must be positive integers: {invalid!r}")
    inconsistent = {
        table_id: {
            "expected": expected_counts[table_id],
            "maximum": maximum_counts[table_id],
        }
        for table_id in set(expected_counts) & set(maximum_counts)
        if expected_counts[table_id] > maximum_counts[table_id]
    }
    if inconsistent:
        raise ValueError(f"schema variant count configuration is inconsistent: {inconsistent!r}")

    observed = {
        row["table_id"]: row["n_schema_variants"]
        for row in schema_audit.group_by("table_id")
        .agg(pl.col("schema_variant_sha256").drop_nulls().n_unique().alias("n_schema_variants"))
        .iter_rows(named=True)
    }
    mismatches = {
        table_id: {"expected": count, "observed": observed.get(table_id, 0)}
        for table_id, count in expected_counts.items()
        if observed.get(table_id, 0) != count
    }
    excess = {
        table_id: {"maximum": count, "observed": observed.get(table_id, 0)}
        for table_id, count in maximum_counts.items()
        if observed.get(table_id, 0) > count
    }
    if mismatches or excess:
        raise ValueError(
            "schema variant count gate failed: "
            f"expected_count_mismatches={mismatches!r}, maximum_count_excess={excess!r}"
        )


def _resolve_exact_table_requests(
    inventory: pl.DataFrame,
    specs: collections.abc.Mapping[str, ExactTableSpec],
    source_table_paths: pl.DataFrame | None,
) -> pl.DataFrame:
    resolved: dict[tuple[str, str], str] = {}
    if source_table_paths is not None:
        _require_columns(
            source_table_paths,
            {SOURCE_COLUMN, "table_id", "table_path"},
            "source_table_paths",
        )
        if (
            source_table_paths.select(SOURCE_COLUMN, "table_id").n_unique()
            != source_table_paths.height
        ):
            raise ValueError("source_table_paths contains duplicate source/table IDs")
        resolved = {
            (row[SOURCE_COLUMN], row["table_id"]): row["table_path"]
            for row in source_table_paths.iter_rows(named=True)
        }
        unknown = set(source_table_paths.get_column("table_id").to_list()).difference(specs)
        if unknown:
            raise ValueError(f"source_table_paths contains unknown table IDs: {unknown}")

    rows: list[dict[str, Any]] = []
    expected_dynamic_keys: set[tuple[str, str]] = set()
    for session in inventory.select("asset_id", "path", "subject_id", SOURCE_COLUMN).iter_rows(
        named=True
    ):
        for spec in specs.values():
            key = (session[SOURCE_COLUMN], spec.table_id)
            if spec.exact_path is None:
                expected_dynamic_keys.add(key)
            table_path = spec.exact_path if spec.exact_path is not None else resolved.get(key)
            resolution_error = None
            if not isinstance(table_path, str) or not table_path.startswith("/"):
                resolution_error = "unresolved exact table path"
                table_path = f"[unresolved:{spec.table_id}]"
            rows.append(
                {
                    **session,
                    "table_id": spec.table_id,
                    "table_path": table_path,
                    "resolution_error": resolution_error,
                }
            )
    if set(resolved) != expected_dynamic_keys:
        extra = set(resolved).difference(expected_dynamic_keys)
        missing = expected_dynamic_keys.difference(resolved)
        raise ValueError(
            "source_table_paths does not exactly resolve dynamic table requests: "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort("path", "table_id")


def _validate_clock_specs(
    series_specs: collections.abc.Sequence[ClockSeriesSpec],
) -> dict[str, ClockSeriesSpec]:
    if not series_specs:
        raise ValueError("at least one clock series spec is required")
    specs: dict[str, ClockSeriesSpec] = {}
    paths: set[str] = set()
    for spec in series_specs:
        if not spec.signal_id or spec.signal_id in specs:
            raise ValueError("clock specs require unique non-empty signal IDs")
        if not spec.exact_path.startswith("/") or spec.exact_path in paths:
            raise ValueError("clock specs require unique absolute NWB paths")
        if spec.shared_timestamps_path is not None and (
            not spec.shared_timestamps_path.startswith("/")
            or not spec.shared_timestamps_path.endswith("/timestamps")
            or spec.shared_timestamps_path == f"{spec.exact_path}/timestamps"
        ):
            raise ValueError(
                "shared timestamp paths must be distinct absolute /timestamps dataset paths"
            )
        names = [component.name for component in spec.components]
        if any(not name or "/" in name for name in names) or len(names) != len(set(names)):
            raise ValueError(f"clock spec {spec.signal_id!r} has invalid component names")
        specs[spec.signal_id] = spec
        paths.add(spec.exact_path)
    return specs


def _clock_component_path(spec: ClockSeriesSpec, component_name: str) -> str:
    if component_name == "timestamps" and spec.shared_timestamps_path is not None:
        return spec.shared_timestamps_path
    return f"{spec.exact_path}/{component_name}"


def _normalize_clock_reader_result(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, collections.abc.Mapping):
        return {"series": {}, "raw_video_internal_paths": []}
    series = metadata.get("series", {})
    if not isinstance(series, collections.abc.Mapping):
        series = {}
    raw_video = metadata.get("raw_video_internal_paths", [])
    if isinstance(raw_video, (str, bytes)) or not isinstance(raw_video, collections.abc.Iterable):
        raw_video = []
    return {
        "series": series,
        "raw_video_internal_paths": sorted({str(path) for path in raw_video}),
    }


def _dtype_families(dtype: Any) -> tuple[str, str | None]:
    if hasattr(dtype, "base_type"):
        try:
            if dtype == pl.Boolean:
                return "boolean", None
            if dtype == pl.String or dtype == pl.Categorical or dtype == pl.Enum:
                return "string", None
            if dtype.is_integer():
                return "integer", None
            if dtype.is_float():
                return "float", None
            if dtype.is_temporal():
                return "temporal", None
            if dtype.base_type() in {pl.List, pl.Array}:
                inner = getattr(dtype, "inner", None)
                inner_family, _ = _dtype_families(inner)
                return "list", inner_family
            if dtype.is_decimal():
                return "numeric", None
        except (AttributeError, TypeError):
            pass

    text = canonical_physical_dtype(dtype).lower().replace(" ", "")
    nested = re.fullmatch(r"(?:list|array)\((.+?)(?:,shape=.*)?\)", text)
    if nested:
        inner_family, _ = _dtype_families(nested.group(1))
        return "list", inner_family
    if re.match(r"u?int\d*", text):
        return "integer", None
    if re.match(r"(?:float|double)\d*", text):
        return "float", None
    if text.startswith("decimal"):
        return "numeric", None
    if text in {"bool", "boolean"}:
        return "boolean", None
    if text in {"str", "string", "utf8", "categorical", "enum", "object"}:
        return "string", None
    if text.startswith(("date", "datetime", "duration", "time")):
        return "temporal", None
    return "other", None


def _scalar_family_compatible(observed_family: str | None, expected_family: str) -> bool:
    if expected_family == "numeric":
        return observed_family in {"integer", "float", "numeric"}
    if expected_family == "float":
        return observed_family == "float"
    return observed_family == expected_family


def _metadata_unit(info: Any) -> str | None:
    if not isinstance(info, collections.abc.Mapping):
        return None
    attrs = info.get("attrs", {})
    if not isinstance(attrs, collections.abc.Mapping):
        return None
    return _optional_text(attrs.get("unit"))


def _accessor_unit(accessor: Any) -> str | None:
    attrs = getattr(accessor, "attrs", {})
    if not isinstance(attrs, collections.abc.Mapping):
        return None
    return _optional_text(attrs.get("unit"))


def _json_safe_attrs(attrs: Any) -> dict[str, Any]:
    if not isinstance(attrs, collections.abc.Mapping):
        return {}
    output: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            output[str(key)] = value
        else:
            output[str(key)] = str(value)
    return output


def _normalize_shape(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        value = parsed
    if isinstance(value, int):
        value = [value]
    try:
        dimensions = [int(dimension) for dimension in value]
    except (TypeError, ValueError):
        return None
    if any(dimension < 0 for dimension in dimensions):
        return None
    return _json_text(dimensions)


def _leading_length_from_json_shape(value: str | None) -> int | None:
    if value is None:
        return None
    shape = json.loads(value)
    return int(shape[0]) if shape else None


def _optional_dtype(value: Any) -> str | None:
    return None if value is None else canonical_physical_dtype(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value)
    return text if text else None


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _is_raw_video_path(path: str) -> bool:
    return (
        _RAW_VIDEO_PATTERN.search(path) is not None
        or pathlib.PurePosixPath(path).suffix.lower() in _RAW_VIDEO_SUFFIXES
    )


def _is_raw_video_asset_path(path: str) -> bool:
    return _is_raw_video_path(path)


def _require_columns(frame: pl.DataFrame, required: set[str], frame_name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{frame_name} is missing columns: {sorted(missing)}")


def _validate_session_metadata_columns(
    frame: pl.DataFrame,
    session_inventory: pl.DataFrame,
    frame_name: str,
) -> None:
    identity_columns = ["asset_id", "path", "subject_id"]
    if "ecephys_session_id" in frame.columns:
        identity_columns.append("ecephys_session_id")
    expected_metadata = _clock_session_metadata(session_inventory)
    expected = {
        row[SOURCE_COLUMN]: tuple(row[column] for column in identity_columns)
        for row in expected_metadata.select(SOURCE_COLUMN, *identity_columns).iter_rows(named=True)
    }
    observed = frame.select(SOURCE_COLUMN, *identity_columns).unique()
    if observed.get_column(SOURCE_COLUMN).n_unique() != observed.height:
        raise ValueError(f"{frame_name} maps a source to conflicting session metadata")
    observed_lookup = {
        row[SOURCE_COLUMN]: tuple(row[column] for column in identity_columns)
        for row in observed.iter_rows(named=True)
    }
    if observed_lookup != expected:
        raise ValueError(f"{frame_name} session metadata differs from session inventory")
