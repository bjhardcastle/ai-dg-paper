"""Tests for immutable DANDI inventory, schema, and task-parameter audits."""

import datetime
import hashlib
import stat
import threading

import polars as pl
import pytest

import dg.audit
import dg.data


def test_verified_companion_cache_is_readable(tmp_path) -> None:
    cached = tmp_path / "companion.csv"
    content = b"session_id,value\n1,2\n"
    cached.write_bytes(content)
    cached.chmod(0o600)

    provenance = dg.audit.cache_companion_trials(
        cached,
        source_url="https://unused.example/companion.csv",
        expected_sha256=hashlib.sha256(content).hexdigest(),
    )

    assert provenance.get_column("cache_status").item() == "verified_existing"
    assert stat.S_IMODE(cached.stat().st_mode) == 0o644


def _published_asset_record() -> dict:
    # Metadata for DANDI asset c8015581 from published version 0.260825.2232.
    return {
        "id": "dandiasset:c8015581-fc08-4a4f-a39a-ea3910b0d425",
        "identifier": "c8015581-fc08-4a4f-a39a-ea3910b0d425",
        "path": "sub-604914/sub-604914_ses-20220427T041046.nwb",
        "contentSize": 2_008_698_308,
        "contentUrl": [
            "https://api.dandiarchive.org/api/assets/c8015581-fc08-4a4f-a39a-ea3910b0d425/download/",
            "https://dandiarchive.s3.amazonaws.com/blobs/585/411/"
            "5854110b-2b31-4bbf-881e-3280f7b87b95",
        ],
        "digest": {
            "dandi:dandi-etag": "4812334c2c04e316d83a99791ab73884-30",
            "dandi:sha2-256": "1ea4768963085486073d512261d8a53101eac2fba331d563addcec6876db2c0a",
        },
        "encodingFormat": "application/x-nwb",
        "dateModified": "2024-06-07T17:30:16.957999-07:00",
        "datePublished": "2026-08-25T22:32:58.124059+00:00",
        "wasAttributedTo": [
            {
                "schemaKey": "Participant",
                "identifier": "604914",
                "genotype": "Sst-IRES-Cre/wt;Ai32(RCL-ChR2(H134R)_EYFP)/wt",
                "sex": {"name": "Female"},
                "species": {"name": "Mus musculus - House mouse"},
                "age": {"value": "P203D"},
            }
        ],
        "wasGeneratedBy": [
            {
                "schemaKey": "Session",
                "startDate": "2022-04-27T04:10:46Z",
                "description": "EPHYS_2",
                "used": [
                    {"schemaKey": "Equipment", "identifier": "probe:627"},
                    {"schemaKey": "Equipment", "identifier": "probe:628"},
                ],
            }
        ],
    }


def test_normalize_published_session_asset() -> None:
    retrieved_at = datetime.datetime(2026, 9, 6, 12, tzinfo=datetime.UTC)

    result = dg.audit.normalize_asset_manifest(
        [_published_asset_record()],
        retrieved_at=retrieved_at,
    )

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["dandiset_version"] == "0.260825.2232"
    assert row["asset_kind"] == "session_nwb"
    assert row["subject_id"] == "604914"
    assert row["recording_day"] == "EPHYS_2"
    assert row["n_probes"] == 2
    assert row["dandi_etag"] == "4812334c2c04e316d83a99791ab73884-30"
    assert row["retrieved_at_utc"] == "2026-09-06T12:00:00+00:00"


def test_manifest_requires_timezone_aware_retrieval_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        dg.audit.normalize_asset_manifest(
            [_published_asset_record()],
            retrieved_at=datetime.datetime(2026, 9, 6),
        )


def test_manifest_rejects_duplicate_asset_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        dg.audit.normalize_asset_manifest([_published_asset_record()] * 2)


def test_schema_audit_records_required_and_optional_fields(monkeypatch) -> None:
    manifest = dg.audit.normalize_asset_manifest([_published_asset_record()])
    internal_paths = [
        f"{table}/{column}"
        for table, columns in dg.audit.REQUIRED_TABLE_COLUMNS.items()
        for column in columns
    ]
    internal_paths.extend(
        [
            "/intervals/trials/no_reward_epoch",
            "/intervals/dynamic_routing_image_set_presentations/image_name",
            "/intervals/dynamic_routing_image_set_presentations/is_change",
            "/intervals/dynamic_routing_image_set_presentations/active",
        ]
    )
    monkeypatch.setattr(
        dg.audit.lazynwb,
        "get_internal_paths",
        lambda *args, **kwargs: internal_paths,
    )
    monkeypatch.setattr(
        dg.audit.lazynwb,
        "get_table_schema",
        lambda *args, **kwargs: pl.Schema(
            {
                column: pl.String
                for column in dg.audit.REQUIRED_TABLE_COLUMNS[dg.data.ELECTRODES_PATH]
            }
        ),
    )

    result = dg.audit.audit_nwb_schemas(
        manifest.filter(pl.col("asset_kind") == "session_nwb"),
        max_workers=1,
    )

    no_reward = result.filter(pl.col("column_name") == "no_reward_epoch").row(0, named=True)
    assert no_reward["present"]
    assert not no_reward["required"]
    assert no_reward["n_image_presentation_candidates"] == 1
    assert result.filter(pl.col("required") & ~pl.col("present")).is_empty()


def test_schema_audit_scopes_internal_path_errors_away_from_electrodes(monkeypatch) -> None:
    manifest = dg.audit.normalize_asset_manifest([_published_asset_record()])

    def fail_internal_paths(*args, **kwargs):
        raise RuntimeError("internal path failure")

    monkeypatch.setattr(dg.audit.lazynwb, "get_internal_paths", fail_internal_paths)
    monkeypatch.setattr(
        dg.audit.lazynwb,
        "get_table_schema",
        lambda *args, **kwargs: pl.Schema(
            {
                column: pl.String
                for column in dg.audit.REQUIRED_TABLE_COLUMNS[dg.data.ELECTRODES_PATH]
            }
        ),
    )

    result = dg.audit.audit_nwb_schemas(
        manifest.filter(pl.col("asset_kind") == "session_nwb"),
        max_workers=1,
    )
    electrodes = result.filter(pl.col("table_path") == dg.data.ELECTRODES_PATH)
    other_tables = result.filter(pl.col("table_path") != dg.data.ELECTRODES_PATH)

    assert electrodes.get_column("present").all()
    assert electrodes.get_column("audit_error").null_count() == electrodes.height
    assert other_tables.get_column("present").null_count() == other_tables.height
    assert other_tables.get_column("audit_error").str.contains("internal path failure").all()


def test_schema_audit_scopes_electrode_errors_away_from_other_tables(monkeypatch) -> None:
    manifest = dg.audit.normalize_asset_manifest([_published_asset_record()])
    internal_paths = [
        f"{table}/{column}"
        for table, columns in dg.audit.REQUIRED_TABLE_COLUMNS.items()
        if table != dg.data.ELECTRODES_PATH
        for column in columns
    ]

    def fail_electrode_schema(*args, **kwargs):
        raise RuntimeError("electrode schema failure")

    monkeypatch.setattr(
        dg.audit.lazynwb,
        "get_internal_paths",
        lambda *args, **kwargs: internal_paths,
    )
    monkeypatch.setattr(dg.audit.lazynwb, "get_table_schema", fail_electrode_schema)

    result = dg.audit.audit_nwb_schemas(
        manifest.filter(pl.col("asset_kind") == "session_nwb"),
        max_workers=1,
    )
    electrodes = result.filter(pl.col("table_path") == dg.data.ELECTRODES_PATH)
    required_other = result.filter(
        (pl.col("table_path") != dg.data.ELECTRODES_PATH) & pl.col("required")
    )

    assert electrodes.get_column("present").null_count() == electrodes.height
    assert electrodes.get_column("audit_error").str.contains("electrode schema failure").all()
    assert required_other.get_column("present").all()
    assert required_other.get_column("audit_error").null_count() == required_other.height


def test_normalize_response_window_sec_accepts_published_attribute_encoding() -> None:
    assert dg.audit.normalize_response_window_sec(["0.15", "0.75"]) == (0.15, 0.75)
    assert dg.audit.normalize_response_window_sec("[0.15, 0.75]") == (0.15, 0.75)


@pytest.mark.parametrize(
    "value",
    [None, [0.15], [0.15, 0.75, 1.0], [True, 0.75], [0.75, 0.15], [0.15, "bad"]],
)
def test_normalize_response_window_sec_rejects_invalid_values(value) -> None:
    with pytest.raises(ValueError, match="response_window_sec"):
        dg.audit.normalize_response_window_sec(value)


def test_task_parameter_audit_records_one_row_per_session_and_scopes_errors(
    monkeypatch,
) -> None:
    first = _published_asset_record()
    second = _published_asset_record() | {
        "identifier": "00000000-0000-0000-0000-000000000002",
        "path": "sub-604915/sub-604915_ses-20220428T041046.nwb",
        "contentUrl": ["https://dandiarchive.s3.amazonaws.com/blobs/test/second"],
    }
    manifest = dg.audit.normalize_asset_manifest([second, first])
    first_url = manifest.filter(pl.col("subject_id") == "604914")["s3_url"].item()

    def get_task_parameters(source):
        if source == first_url:
            return {"response_window_sec": ["0.15", "0.75"]}
        raise RuntimeError("task-parameter read failed")

    monkeypatch.setattr(dg.data, "get_task_parameters", get_task_parameters)

    result = dg.audit.audit_task_parameters(manifest, max_workers=2)

    assert result.height == 2
    assert result.get_column("path").is_sorted()
    success = result.filter(pl.col("subject_id") == "604914").row(0, named=True)
    failure = result.filter(pl.col("subject_id") == "604915").row(0, named=True)
    assert success["task_parameters_path"] == dg.data.TASK_PARAMETERS_PATH
    assert success["response_window_sec"] == '["0.15","0.75"]'
    assert success["response_window_start_seconds"] == 0.15
    assert success["response_window_stop_seconds"] == 0.75
    assert success["response_window_matches_expected"]
    assert success["audit_error"] is None
    assert failure["response_window_sec"] is None
    assert failure["response_window_start_seconds"] is None
    assert not failure["response_window_matches_expected"]
    assert failure["audit_error"] == "RuntimeError: task-parameter read failed"
    assert "response_window_sec" in result.write_csv()
    with pytest.raises(ValueError, match="required task-parameter audit failed"):
        dg.audit.validate_task_parameters_audit(result)


def test_task_parameter_audit_reads_sessions_concurrently(monkeypatch) -> None:
    first = _published_asset_record()
    second = _published_asset_record() | {
        "identifier": "00000000-0000-0000-0000-000000000002",
        "path": "sub-604915/sub-604915_ses-20220428T041046.nwb",
        "contentUrl": ["https://dandiarchive.s3.amazonaws.com/blobs/test/second"],
    }
    manifest = dg.audit.normalize_asset_manifest([first, second])
    both_workers_started = threading.Barrier(2)

    def get_task_parameters(source):
        both_workers_started.wait(timeout=2.0)
        return {"response_window_sec": ["0.15", "0.75"]}

    monkeypatch.setattr(dg.data, "get_task_parameters", get_task_parameters)

    result = dg.audit.audit_task_parameters(manifest, max_workers=2)

    assert result.get_column("audit_error").null_count() == 2


def test_required_task_parameter_validation_accepts_expected_window(monkeypatch) -> None:
    manifest = dg.audit.normalize_asset_manifest([_published_asset_record()])
    monkeypatch.setattr(
        dg.data,
        "get_task_parameters",
        lambda source: {"response_window_sec": ["0.15", "0.75"]},
    )
    audit = dg.audit.audit_task_parameters(manifest, max_workers=1)

    assert dg.audit.validate_task_parameters_audit(audit) is None


def test_required_task_parameter_validation_rejects_unexpected_window(monkeypatch) -> None:
    manifest = dg.audit.normalize_asset_manifest([_published_asset_record()])
    monkeypatch.setattr(
        dg.data,
        "get_task_parameters",
        lambda source: {"response_window_sec": ["0.10", "0.75"]},
    )
    audit = dg.audit.audit_task_parameters(manifest, max_workers=1)

    assert not audit.get_column("response_window_matches_expected").item()
    with pytest.raises(ValueError, match=r"expected response_window_sec \[0.15, 0.75\]"):
        dg.audit.validate_task_parameters_audit(audit)


def test_nwb_root_identifier_audit_reconciles_subject_and_session(monkeypatch) -> None:
    inventory = pl.DataFrame(
        {
            "asset_id": ["asset-a"],
            "path": ["sub-604914/session.nwb"],
            "_nwb_path": ["session-a"],
            "subject_id": ["604914"],
            "ecephys_session_id": [1173189336],
        }
    )
    monkeypatch.setattr(
        dg.audit,
        "read_nwb_root_identity",
        lambda source: {
            "subject_id": "sub-604914",
            "identifier": "1173189336",
            "session_id": "1173189336",
        },
    )

    result = dg.audit.audit_nwb_root_identifiers(inventory, max_workers=1)

    assert result.get_column("subject_id_agrees").item()
    assert result.get_column("ecephys_session_id_agrees").item()
    assert result.get_column("root_identifier_audit_pass").item()
    dg.audit.validate_nwb_root_identifier_audit(result, inventory)


def test_nwb_root_identifier_audit_scopes_errors_and_fails_closed(monkeypatch) -> None:
    inventory = pl.DataFrame(
        {
            "asset_id": ["asset-a", "asset-b"],
            "path": ["a.nwb", "b.nwb"],
            "_nwb_path": ["session-a", "session-b"],
            "subject_id": ["mouse-a", "mouse-b"],
            "ecephys_session_id": [1, 2],
        }
    )

    def get_metadata(source):
        if source == "session-b":
            raise RuntimeError("metadata unavailable")
        return {"subject_id": "mouse-a", "identifier": "1", "session_id": "1"}

    monkeypatch.setattr(dg.audit, "read_nwb_root_identity", get_metadata)

    result = dg.audit.audit_nwb_root_identifiers(inventory, max_workers=2)

    assert result.height == 2
    failure = result.filter(pl.col("_nwb_path") == "session-b").row(0, named=True)
    assert failure["audit_error"] == "RuntimeError: metadata unavailable"
    assert not failure["root_identifier_audit_pass"]
    with pytest.raises(ValueError, match="root identifier reconciliation failed"):
        dg.audit.validate_nwb_root_identifier_audit(result, inventory)


def test_manifest_summary_and_session_source() -> None:
    manifest = dg.audit.normalize_asset_manifest([_published_asset_record()])

    summary = dg.audit.summarize_asset_manifest(manifest)

    assert summary.to_dicts() == [
        {
            "asset_kind": "session_nwb",
            "n_assets": 1,
            "content_size_bytes": 2_008_698_308,
            "n_subjects": 1,
        }
    ]
    assert dg.audit.get_session_sources(manifest) == [manifest["s3_url"][0]]
