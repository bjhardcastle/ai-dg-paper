"""Pure tests for metadata-only Milestone 0 integrity audits."""

import hashlib
import json

import polars as pl
import pytest

import dg.m0_integrity


def _session_inventory() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "asset_id": ["asset-a", "asset-b"],
            "path": ["sub-1/session-a.nwb", "sub-2/session-b.nwb"],
            "subject_id": ["1", "2"],
            "_nwb_path": ["s3://bucket/a.nwb", "s3://bucket/b.nwb"],
        }
    )


def _schema_spec(*, exact_path: str | None = "/intervals/example"):
    return dg.m0_integrity.ExactTableSpec(
        "example",
        exact_path,
        (
            dg.m0_integrity.SchemaFieldExpectation("id", "integer"),
            dg.m0_integrity.SchemaFieldExpectation("value", "numeric"),
            dg.m0_integrity.SchemaFieldExpectation("labels", "list_string"),
            dg.m0_integrity.SchemaFieldExpectation("optional", "boolean", required=False),
        ),
    )


@pytest.mark.parametrize(
    ("physical", "logical", "compatible"),
    [
        (pl.Int32, "integer", True),
        (pl.UInt64, "numeric", True),
        (pl.Float32, "numeric", True),
        (pl.Boolean, "integer", False),
        (pl.String, "string", True),
        (pl.List(pl.Float64), "list_numeric", True),
        (pl.List(pl.String), "list_string", True),
        (pl.List(pl.Boolean), "list_numeric", False),
        ("Array(Float32, shape=(3,))", "list_numeric", True),
    ],
)
def test_logical_dtype_compatibility(physical, logical, compatible) -> None:
    assert dg.m0_integrity.logical_dtype_compatible(physical, logical) is compatible


def test_schema_variant_hash_is_order_invariant_and_dtype_sensitive() -> None:
    first_json, first_hash = dg.m0_integrity.stable_schema_variant_sha256(
        "/units", {"b": pl.Float64, "a": pl.Int64}
    )
    second_json, second_hash = dg.m0_integrity.stable_schema_variant_sha256(
        "/units", {"a": pl.Int64, "b": pl.Float64}
    )
    _, drift_hash = dg.m0_integrity.stable_schema_variant_sha256(
        "/units", {"a": pl.Int32, "b": pl.Float64}
    )
    alternate_path_json, alternate_path_hash = dg.m0_integrity.stable_schema_variant_sha256(
        "/session_specific_units_name", {"a": pl.Int64, "b": pl.Float64}
    )

    assert first_json == second_json
    assert first_hash == second_hash
    assert drift_hash != first_hash
    assert first_hash == hashlib.sha256(first_json.encode()).hexdigest()
    assert alternate_path_json == first_json
    assert alternate_path_hash == first_hash


def test_exact_schema_audit_records_optional_missing_unexpected_and_variants() -> None:
    def reader(source, exact_path):
        assert exact_path == "/intervals/example"
        id_dtype = pl.Int32 if source.endswith("a.nwb") else pl.Int64
        return pl.Schema(
            {
                "id": id_dtype,
                "value": pl.Float64,
                "labels": pl.List(pl.String),
                "extra": pl.String,
            }
        )

    result = dg.m0_integrity.audit_exact_table_schema_metadata(
        _session_inventory(), table_specs=[_schema_spec()], schema_reader=reader, max_workers=2
    )
    dg.m0_integrity.validate_exact_table_schema_audit(
        result, _session_inventory(), table_specs=[_schema_spec()]
    )
    optional = result.filter(pl.col("column_name") == "optional")
    unexpected = result.filter(pl.col("column_name") == "extra")

    assert optional.height == 2
    assert not optional.get_column("present").any()
    assert unexpected.get_column("column_requirement").unique().to_list() == ["unexpected"]
    assert result.get_column("schema_variant_sha256").n_unique() == 2
    assert result.get_column("metadata_only").all()
    assert not result.get_column("array_values_loaded").any()
    assert not result.get_column("neural_outcomes_accessed").any()

    summary = dg.m0_integrity.summarize_exact_table_schema_audit(result)
    optional_summary = summary.filter(pl.col("column_name") == "optional").row(0, named=True)
    assert optional_summary["n_sources_audited"] == 2
    assert optional_summary["n_sources_missing_or_unread"] == 2
    assert optional_summary["field_audit_status"] == "optional_missing"


def test_exact_schema_audit_omits_explicitly_excluded_activity_outcome_metadata() -> None:
    spec = dg.m0_integrity.ExactTableSpec(
        "units",
        "/units",
        (dg.m0_integrity.SchemaFieldExpectation("id", "integer"),),
        excluded_observed_columns=("firing_rate",),
    )
    result = dg.m0_integrity.audit_exact_table_schema_metadata(
        _session_inventory(),
        table_specs=[spec],
        schema_reader=lambda source, path: pl.Schema({"id": pl.Int64, "firing_rate": pl.Float64}),
        max_workers=1,
    )
    dg.m0_integrity.validate_exact_table_schema_audit(
        result,
        _session_inventory(),
        table_specs=[spec],
    )

    assert "firing_rate" not in result.get_column("column_name").to_list()
    assert all("firing_rate" not in value for value in result.get_column("observed_schema_json"))


def test_exact_schema_audit_fails_closed_on_dtype_drift() -> None:
    result = dg.m0_integrity.audit_exact_table_schema_metadata(
        _session_inventory(),
        table_specs=[_schema_spec()],
        schema_reader=lambda source, path: {
            "id": pl.String,
            "value": pl.Float64,
            "labels": pl.List(pl.String),
        },
        max_workers=1,
    )

    incompatible = result.filter(pl.col("column_name") == "id")
    assert not incompatible.get_column("dtype_compatible").any()
    assert incompatible.get_column("table_audit_status").unique().to_list() == [
        "dtype_incompatible"
    ]
    with pytest.raises(ValueError, match="exact-table schema audit failed"):
        dg.m0_integrity.validate_exact_table_schema_audit(
            result, _session_inventory(), table_specs=[_schema_spec()]
        )


def test_exact_schema_audit_scopes_reader_errors_and_gates_source_coverage() -> None:
    def reader(source, path):
        if source.endswith("b.nwb"):
            raise RuntimeError("metadata unavailable")
        return {"id": pl.Int64, "value": pl.Float64, "labels": pl.List(pl.String)}

    result = dg.m0_integrity.audit_exact_table_schema_metadata(
        _session_inventory(), table_specs=[_schema_spec()], schema_reader=reader, max_workers=2
    )
    failed = result.filter(pl.col("_nwb_path") == "s3://bucket/b.nwb")
    assert failed.get_column("audit_error").str.contains("metadata unavailable").all()
    assert failed.get_column("present").null_count() == failed.height

    with pytest.raises(ValueError, match="exact-table schema audit failed"):
        dg.m0_integrity.validate_exact_table_schema_audit(
            result, _session_inventory(), table_specs=[_schema_spec()]
        )
    with pytest.raises(ValueError, match="exactly cover session inventory sources"):
        dg.m0_integrity.validate_exact_table_schema_audit(
            result.filter(pl.col("_nwb_path") != "s3://bucket/b.nwb"),
            _session_inventory(),
            table_specs=[_schema_spec()],
        )


def test_dynamic_exact_paths_must_resolve_every_source_once() -> None:
    targets = pl.DataFrame(
        {
            "_nwb_path": ["s3://bucket/a.nwb", "s3://bucket/b.nwb"],
            "table_id": ["example", "example"],
            "table_path": ["/intervals/images_a", "/intervals/images_b"],
        }
    )
    observed_paths = []

    def reader(source, path):
        observed_paths.append((source, path))
        return {"id": pl.Int64, "value": pl.Float64, "labels": pl.List(pl.String)}

    result = dg.m0_integrity.audit_exact_table_schema_metadata(
        _session_inventory(),
        table_specs=[_schema_spec(exact_path=None)],
        source_table_paths=targets,
        schema_reader=reader,
        max_workers=1,
    )
    dg.m0_integrity.validate_exact_table_schema_audit(
        result,
        _session_inventory(),
        table_specs=[_schema_spec(exact_path=None)],
        source_table_paths=targets,
        expected_schema_variant_counts={"example": 1},
        maximum_schema_variant_counts={"example": 1},
    )
    assert result.get_column("schema_variant_sha256").n_unique() == 1
    assert observed_paths == [
        ("s3://bucket/a.nwb", "/intervals/images_a"),
        ("s3://bucket/b.nwb", "/intervals/images_b"),
    ]

    with pytest.raises(ValueError, match="exactly resolve dynamic table requests"):
        dg.m0_integrity.audit_exact_table_schema_metadata(
            _session_inventory(),
            table_specs=[_schema_spec(exact_path=None)],
            source_table_paths=targets.head(1),
            schema_reader=reader,
            max_workers=1,
        )


def test_exact_schema_variant_count_gate_rejects_unexpected_variant() -> None:
    result = dg.m0_integrity.audit_exact_table_schema_metadata(
        _session_inventory(),
        table_specs=[_schema_spec()],
        schema_reader=lambda source, path: {
            "id": pl.Int64,
            "value": pl.Float64,
            "labels": pl.List(pl.String),
            **({"new_column": pl.Float32} if source.endswith("b.nwb") else {}),
        },
        max_workers=1,
    )

    dg.m0_integrity.validate_exact_table_schema_audit(
        result,
        _session_inventory(),
        table_specs=[_schema_spec()],
        expected_schema_variant_counts={"example": 2},
        maximum_schema_variant_counts={"example": 2},
    )
    with pytest.raises(ValueError, match="schema variant count gate failed"):
        dg.m0_integrity.validate_exact_table_schema_audit(
            result,
            _session_inventory(),
            table_specs=[_schema_spec()],
            expected_schema_variant_counts={"example": 1},
        )
    with pytest.raises(ValueError, match="schema variant count gate failed"):
        dg.m0_integrity.validate_exact_table_schema_audit(
            result,
            _session_inventory(),
            table_specs=[_schema_spec()],
            maximum_schema_variant_counts={"example": 1},
        )


def test_read_exact_table_schema_requests_array_dtypes_without_loading_values(monkeypatch) -> None:
    observed = {}

    def get_table_schema(source, path, **kwargs):
        observed.update({"source": source, "path": path, **kwargs})
        return pl.Schema({"spike_times": pl.List(pl.Float64)})

    monkeypatch.setattr(dg.m0_integrity.lazynwb, "get_table_schema", get_table_schema)
    result = dg.m0_integrity.read_exact_table_schema("s3://bucket/a.nwb", "/units")

    assert result == pl.Schema({"spike_times": pl.List(pl.Float64)})
    assert observed == {
        "source": "s3://bucket/a.nwb",
        "path": "/units",
        "exclude_array_columns": False,
        "exclude_internal_columns": True,
        "raise_on_missing": True,
    }


def _clock_specs():
    return (
        dg.m0_integrity.ClockSeriesSpec("core", "/processing/core", "core"),
        dg.m0_integrity.ClockSeriesSpec(
            "eye",
            "/acquisition/EyeTracking/eye",
            "optional",
            (
                dg.m0_integrity.ClockComponentSpec("data"),
                dg.m0_integrity.ClockComponentSpec("area"),
            ),
        ),
    )


def _clock_reader(source, specs):
    core_length = 10 if source.endswith("a.nwb") else 12
    eye_present = source.endswith("a.nwb")
    return {
        "series": {
            "/processing/core": {
                "present": True,
                "is_timeseries": True,
                "components": {
                    "data": {
                        "present": True,
                        "shape": [core_length],
                        "dtype": "float32",
                        "unit": "cm/s",
                    },
                    "timestamps": {
                        "present": True,
                        "shape": [core_length],
                        "dtype": "float64",
                        "unit": "seconds",
                    },
                    "starting_time": {"present": False},
                },
            },
            "/acquisition/EyeTracking/eye": {
                "present": eye_present,
                "is_timeseries": eye_present,
                "components": {
                    "data": {"present": eye_present, "shape": [10, 2], "dtype": "float32"},
                    "area": {"present": eye_present, "shape": [10], "dtype": "float32"},
                    "timestamps": {
                        "present": eye_present,
                        "shape": [10],
                        "dtype": "float64",
                        "unit": "seconds",
                    },
                    "starting_time": {"present": False},
                },
            },
        },
        "raw_video_internal_paths": [],
    }


def test_clock_metadata_audit_supports_generic_optional_eye_absence() -> None:
    audit = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
        _session_inventory(),
        series_specs=_clock_specs(),
        metadata_reader=_clock_reader,
        max_workers=2,
    )
    dg.m0_integrity.validate_behavior_acquisition_clock_metadata(
        audit,
        _session_inventory(),
        series_specs=_clock_specs(),
        expected_optional_absent_sources={"eye": {"s3://bucket/b.nwb"}},
    )
    with pytest.raises(ValueError, match="optional absence coverage"):
        dg.m0_integrity.validate_behavior_acquisition_clock_metadata(
            audit,
            _session_inventory(),
            series_specs=_clock_specs(),
        )
    core = audit.series.filter(pl.col("signal_id") == "core")
    eye_b = audit.series.filter(
        (pl.col("signal_id") == "eye") & (pl.col("_nwb_path") == "s3://bucket/b.nwb")
    ).row(0, named=True)

    assert core.get_column("series_metadata_status").unique().to_list() == ["metadata_present"]
    assert core.get_column("leading_lengths_agree").all()
    assert eye_b["series_metadata_status"] == "optional_absent"
    assert not audit.series.get_column("timestamp_vectors_validated").any()
    assert audit.series.get_column("full_vector_clock_status").unique().to_list() == [
        "not_validated_metadata_only"
    ]
    assert not audit.components.get_column("array_values_loaded").any()
    assert audit.raw_video.get_column("raw_video_timeseries_status").unique().to_list() == [
        "no_name_matched_candidate_in_session_nwb"
    ]

    summary = dg.m0_integrity.summarize_behavior_acquisition_clock_metadata(audit.series)
    eye_summary = summary.filter(pl.col("signal_id") == "eye").row(0, named=True)
    assert eye_summary["n_sources_present"] == 1
    assert eye_summary["n_sources_absent_or_unread"] == 1
    assert eye_summary["coverage_status"] == "optional_missing"


def test_clock_metadata_gate_rejects_missing_core_and_length_mismatch() -> None:
    def missing_core_reader(source, specs):
        result = _clock_reader(source, specs)
        result["series"]["/processing/core"] = {"present": False}
        return result

    missing = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
        _session_inventory(),
        series_specs=_clock_specs(),
        metadata_reader=missing_core_reader,
        max_workers=1,
    )
    with pytest.raises(ValueError, match="clock metadata integrity failed"):
        dg.m0_integrity.validate_behavior_acquisition_clock_metadata(
            missing, _session_inventory(), series_specs=_clock_specs()
        )

    def mismatch_reader(source, specs):
        result = _clock_reader(source, specs)
        result["series"]["/processing/core"]["components"]["timestamps"]["shape"] = [9]
        return result

    mismatch = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
        _session_inventory(),
        series_specs=_clock_specs(),
        metadata_reader=mismatch_reader,
        max_workers=1,
    )
    with pytest.raises(ValueError, match="clock metadata integrity failed"):
        dg.m0_integrity.validate_behavior_acquisition_clock_metadata(
            mismatch, _session_inventory(), series_specs=_clock_specs()
        )


@pytest.mark.parametrize(
    ("component", "field", "value", "status"),
    [
        ("data", "shape", None, "required_component_shape_missing_scalar_or_empty"),
        ("data", "shape", [0], "required_component_shape_missing_scalar_or_empty"),
        ("timestamps", "shape", [], "timestamp_shape_missing_scalar_or_empty"),
        ("timestamps", "shape", [0], "timestamp_shape_missing_scalar_or_empty"),
        ("timestamps", "dtype", "String", "timestamp_dtype_missing_or_nonnumeric"),
    ],
)
def test_clock_metadata_gate_requires_shapes_and_numeric_explicit_timestamps(
    component,
    field,
    value,
    status,
) -> None:
    def reader(source, specs):
        result = _clock_reader(source, specs)
        result["series"]["/processing/core"]["components"][component][field] = value
        return result

    audit = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
        _session_inventory(), series_specs=_clock_specs(), metadata_reader=reader, max_workers=1
    )
    assert status in audit.series.filter(pl.col("signal_id") == "core").get_column(
        "series_metadata_status"
    )
    with pytest.raises(ValueError, match="clock metadata integrity failed"):
        dg.m0_integrity.validate_behavior_acquisition_clock_metadata(
            audit, _session_inventory(), series_specs=_clock_specs()
        )


def test_rate_derived_clock_requires_numeric_starting_time_dtype() -> None:
    spec = dg.m0_integrity.ClockSeriesSpec("core", "/processing/core", "core")

    def reader(dtype):
        return {
            "series": {
                "/processing/core": {
                    "present": True,
                    "is_timeseries": True,
                    "sampling_rate_hz": 60.0,
                    "components": {
                        "data": {"present": True, "shape": [10], "dtype": "Float32"},
                        "timestamps": {"present": False},
                        "starting_time": {"present": True, "shape": [], "dtype": dtype},
                    },
                }
            }
        }

    passing = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
        _session_inventory(),
        series_specs=[spec],
        metadata_reader=lambda source, specs: reader("Float64"),
        max_workers=1,
    )
    dg.m0_integrity.validate_behavior_acquisition_clock_metadata(
        passing, _session_inventory(), series_specs=[spec]
    )
    assert passing.series.get_column("timing_representation").unique().to_list() == ["rate_derived"]

    failing = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
        _session_inventory(),
        series_specs=[spec],
        metadata_reader=lambda source, specs: reader("String"),
        max_workers=1,
    )
    assert failing.series.get_column("series_metadata_status").unique().to_list() == [
        "starting_time_dtype_missing_or_nonnumeric"
    ]
    with pytest.raises(ValueError, match="clock metadata integrity failed"):
        dg.m0_integrity.validate_behavior_acquisition_clock_metadata(
            failing, _session_inventory(), series_specs=[spec]
        )


def test_clock_metadata_gate_rejects_malformed_present_optional_series() -> None:
    def reader(source, specs):
        result = _clock_reader(source, specs)
        if source.endswith("a.nwb"):
            result["series"]["/acquisition/EyeTracking/eye"]["components"]["area"] = {
                "present": False
            }
        return result

    audit = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
        _session_inventory(), series_specs=_clock_specs(), metadata_reader=reader, max_workers=1
    )
    malformed = audit.series.filter((pl.col("signal_id") == "eye") & pl.col("series_present")).row(
        0, named=True
    )
    assert malformed["series_metadata_status"] == "required_component_missing"
    with pytest.raises(ValueError, match="clock metadata integrity failed"):
        dg.m0_integrity.validate_behavior_acquisition_clock_metadata(
            audit,
            _session_inventory(),
            series_specs=_clock_specs(),
            expected_optional_absent_sources={"eye": {"s3://bucket/b.nwb"}},
        )


def test_clock_metadata_gate_rejects_eye_component_length_mismatch() -> None:
    def reader(source, specs):
        result = _clock_reader(source, specs)
        if source.endswith("a.nwb"):
            result["series"]["/acquisition/EyeTracking/eye"]["components"]["area"]["shape"] = [9]
        return result

    audit = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
        _session_inventory(), series_specs=_clock_specs(), metadata_reader=reader, max_workers=1
    )
    eye = audit.series.filter(
        (pl.col("signal_id") == "eye") & (pl.col("_nwb_path") == "s3://bucket/a.nwb")
    ).row(0, named=True)
    assert eye["series_metadata_status"] == "required_component_length_mismatch"
    assert eye["n_required_component_length_mismatches"] == 1
    assert not eye["required_component_lengths_agree"]
    with pytest.raises(ValueError, match="clock metadata integrity failed"):
        dg.m0_integrity.validate_behavior_acquisition_clock_metadata(
            audit,
            _session_inventory(),
            series_specs=_clock_specs(),
            expected_optional_absent_sources={"eye": {"s3://bucket/b.nwb"}},
        )


def test_clock_metadata_reader_errors_are_source_scoped() -> None:
    def reader(source, specs):
        if source.endswith("b.nwb"):
            raise RuntimeError("catalog unavailable")
        return _clock_reader(source, specs)

    audit = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
        _session_inventory(), series_specs=_clock_specs(), metadata_reader=reader, max_workers=2
    )
    failed = audit.series.filter(pl.col("_nwb_path") == "s3://bucket/b.nwb")
    passed = audit.series.filter(pl.col("_nwb_path") == "s3://bucket/a.nwb")

    assert failed.get_column("audit_error").str.contains("catalog unavailable").all()
    assert passed.get_column("audit_error").null_count() == passed.height
    with pytest.raises(ValueError, match="clock metadata integrity failed"):
        dg.m0_integrity.validate_behavior_acquisition_clock_metadata(
            audit, _session_inventory(), series_specs=_clock_specs()
        )


def test_default_clock_reader_reads_only_metadata_accessors(monkeypatch) -> None:
    class FakeAccessor:
        def __init__(self, shape, dtype, unit=None):
            self.shape = shape
            self.dtype = dtype
            self.attrs = {} if unit is None else {"unit": unit}

        def __getitem__(self, key):
            raise AssertionError("array values must not be sliced")

    class FakeFile(dict):
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.closed = True

    fake_file = FakeFile({"/processing/core/timestamps": FakeAccessor((5,), "float64", "seconds")})

    class FakeTimeSeries:
        _file = fake_file

    spec = dg.m0_integrity.ClockSeriesSpec("core", "/processing/core", "core")
    timeseries_calls = []
    path_info = {
        "/processing/core": {"is_timeseries": True, "attrs": {}, "shape": None},
        "/processing/core/data": {
            "is_timeseries": False,
            "attrs": {"unit": "cm/s"},
            "shape": (5,),
            "dtype": "float32",
        },
        "/processing/core/timestamps": {
            "is_timeseries": False,
            "attrs": {"unit": "seconds"},
            "shape": (5,),
        },
    }
    monkeypatch.setattr(
        dg.m0_integrity.lazynwb,
        "get_internal_path_info",
        lambda *args, **kwargs: path_info,
    )
    monkeypatch.setattr(
        dg.m0_integrity.lazynwb,
        "TimeSeries",
        lambda *args, **kwargs: timeseries_calls.append((args, kwargs)) or FakeTimeSeries(),
    )

    result = dg.m0_integrity.read_clock_source_metadata("s3://bucket/a.nwb", [spec])
    metadata = result["series"]["/processing/core"]

    assert metadata["components"]["data"]["shape"] == (5,)
    assert metadata["components"]["data"]["dtype"] == "float32"
    assert metadata["components"]["timestamps"]["dtype"] == "float64"
    assert result["raw_video_internal_paths"] == []
    assert len(timeseries_calls) == 1
    assert fake_file.closed


def test_shared_eye_timestamps_are_explicitly_provenanced_without_array_reads(
    monkeypatch,
) -> None:
    class FakeAccessor:
        def __init__(self, shape, dtype, unit=None):
            self.shape = shape
            self.dtype = dtype
            self.attrs = {} if unit is None else {"unit": unit}

        def __getitem__(self, key):
            raise AssertionError("array values must not be sliced")

    pupil_path = "/acquisition/EyeTracking/pupil_tracking"
    shared_timestamps = "/acquisition/EyeTracking/eye_tracking/timestamps"
    eye_path = "/acquisition/EyeTracking/eye_tracking"

    class FakeFile(dict):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    class FakeEyeTimeSeries:
        _file = FakeFile({shared_timestamps: FakeAccessor((5,), "float64", "seconds")})

    spec = dg.m0_integrity.ClockSeriesSpec(
        "pupil_tracking",
        pupil_path,
        "optional",
        (dg.m0_integrity.ClockComponentSpec("data"),),
        shared_timestamps_path=shared_timestamps,
    )
    path_info = {
        eye_path: {"is_timeseries": True, "attrs": {}, "shape": None},
        pupil_path: {"is_timeseries": False, "attrs": {}, "shape": None},
        f"{pupil_path}/data": {
            "is_timeseries": False,
            "attrs": {"unit": "pixels"},
            "shape": (5,),
            "dtype": "float32",
        },
        shared_timestamps: {
            "is_timeseries": False,
            "attrs": {"unit": "seconds"},
            "shape": (5,),
            "dtype": "float64",
        },
    }
    monkeypatch.setattr(
        dg.m0_integrity.lazynwb,
        "get_internal_path_info",
        lambda *args, **kwargs: path_info,
    )
    monkeypatch.setattr(
        dg.m0_integrity.lazynwb,
        "TimeSeries",
        lambda *args, **kwargs: FakeEyeTimeSeries(),
    )

    metadata = dg.m0_integrity.read_clock_source_metadata("s3://bucket/a.nwb", [spec])
    pupil = metadata["series"][pupil_path]
    assert not pupil["is_timeseries"]
    assert pupil["components"]["data"]["dtype"] == "float32"
    assert pupil["components"]["timestamps"]["dtype"] == "float64"

    audit = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
        _session_inventory(),
        series_specs=[spec],
        metadata_reader=lambda source, specs: metadata,
        max_workers=1,
    )
    dg.m0_integrity.validate_behavior_acquisition_clock_metadata(
        audit,
        _session_inventory(),
        series_specs=[spec],
    )
    assert audit.series.get_column("series_metadata_status").unique().to_list() == [
        "metadata_present"
    ]
    assert audit.series.get_column("timing_representation").unique().to_list() == [
        "shared_explicit_timestamps"
    ]
    assert audit.series.get_column("timing_source_path").unique().to_list() == [shared_timestamps]
    timestamps = audit.components.filter(pl.col("component_name") == "timestamps")
    assert timestamps.get_column("component_path").unique().to_list() == [shared_timestamps]
    assert timestamps.get_column("component_requirement").unique().to_list() == [
        "shared_timing_source"
    ]


def test_raw_video_paths_and_frozen_asset_absence_are_recorded() -> None:
    def reader(source, specs):
        result = _clock_reader(source, specs)
        if source.endswith("a.nwb"):
            result["raw_video_internal_paths"] = ["/acquisition/behavior_video/data"]
        return result

    audit = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
        _session_inventory(), series_specs=_clock_specs(), metadata_reader=reader, max_workers=1
    )
    assert audit.raw_video.get_column("n_raw_video_internal_paths").to_list() == [1, 0]
    assert json.loads(audit.raw_video.row(0, named=True)["raw_video_internal_paths"]) == [
        "/acquisition/behavior_video/data"
    ]

    absent = dg.m0_integrity.audit_raw_video_asset_absence(
        pl.DataFrame({"path": ["sub-1/session.nwb", "sub-1/probe.nwb"]})
    ).row(0, named=True)
    present = dg.m0_integrity.audit_raw_video_asset_absence(
        pl.DataFrame({"path": ["sub-1/session.nwb", "sub-1/behavior.mp4"]})
    ).row(0, named=True)
    assert absent["raw_video_asset_status"] == "no_name_matched_candidate_in_frozen_asset_inventory"
    assert present["raw_video_asset_status"] == "present_in_frozen_asset_inventory"
    assert present["n_raw_video_asset_candidates"] == 1
    assert not present["asset_values_loaded"]


def test_default_specs_enumerate_the_frozen_behavior_and_eye_paths() -> None:
    observed = {spec.exact_path for spec in dg.m0_integrity.DEFAULT_CLOCK_SERIES_SPECS}
    expected = {
        "/processing/licking/licks",
        "/processing/rewards/volume",
        "/processing/rewards/autorewarded",
        "/processing/running/dx",
        "/processing/running/speed",
        "/processing/running/speed_unfiltered",
        "/processing/stimulus/timestamps",
        "/acquisition/v_in",
        "/acquisition/v_sig",
        "/processing/optotagging/optotagging",
        "/acquisition/EyeTracking/eye_tracking",
        "/acquisition/EyeTracking/pupil_tracking",
        "/acquisition/EyeTracking/corneal_reflection_tracking",
        "/acquisition/EyeTracking/likely_blink",
    }
    assert observed == expected
    assert all(
        spec.coverage_requirement == "core" for spec in dg.m0_integrity.CORE_CLOCK_SERIES_SPECS
    )
    assert all(
        spec.coverage_requirement == "optional"
        for spec in dg.m0_integrity.OPTIONAL_EYE_CLOCK_SERIES_SPECS
    )
    eye_specs = {spec.signal_id: spec for spec in dg.m0_integrity.OPTIONAL_EYE_CLOCK_SERIES_SPECS}
    assert eye_specs["eye_tracking"].shared_timestamps_path is None
    assert {
        eye_specs[name].shared_timestamps_path
        for name in ("pupil_tracking", "corneal_reflection_tracking", "likely_blink")
    } == {"/acquisition/EyeTracking/eye_tracking/timestamps"}


def test_consolidated_schema_specs_include_required_projection_fields() -> None:
    trials = next(
        spec for spec in dg.m0_integrity.DEFAULT_EXACT_TABLE_SPECS if spec.table_id == "trials"
    )
    trial_fields = {field.column_name: field for field in trials.fields}
    task_fields = {field.column_name: field for field in dg.m0_integrity.TASK_PRESENTATION_FIELDS}

    assert trial_fields["change_frame"].required
    assert trial_fields["change_frame"].logical_dtype == "numeric"
    assert task_fields["start_frame"].required
    assert task_fields["start_frame"].logical_dtype == "integer"
    assert task_fields["flashes_since_change"].logical_dtype == "numeric"
    assert all(
        task_fields[name].required
        for name in (
            "is_image_novel",
            "flashes_since_change",
            "stimulus_block",
            "rewarded",
            "omitted",
        )
    )
    assert dg.m0_integrity.FROZEN_EXPECTED_SCHEMA_VARIANT_COUNTS == {
        "units": 1,
        "electrodes": 1,
        "task_image_presentations": 1,
        "trials": 2,
    }


def test_clock_artifacts_preserve_ecephys_session_id_when_available() -> None:
    inventory = _session_inventory().with_columns(
        pl.Series("ecephys_session_id", [101, 202], dtype=pl.Int64)
    )
    audit = dg.m0_integrity.audit_behavior_acquisition_clock_metadata(
        inventory, series_specs=_clock_specs(), metadata_reader=_clock_reader, max_workers=1
    )

    assert audit.series.get_column("ecephys_session_id").unique().sort().to_list() == [101, 202]
    assert audit.components.get_column("ecephys_session_id").unique().sort().to_list() == [
        101,
        202,
    ]
    assert audit.raw_video.get_column("ecephys_session_id").to_list() == [101, 202]
