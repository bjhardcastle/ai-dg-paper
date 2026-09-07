"""Tests for resilient, process-isolated task-presentation materialization."""

import collections
import pathlib
import subprocess
import threading

import polars as pl
import polars.testing
import pytest

import dg.artifacts
import dg.data
import dg.inventory
import dg.task_presentations


def _session_inventory(*sources: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_nwb_path": list(sources),
            "dandiset_id": ["001051"] * len(sources),
            "dandiset_version": ["0.260825.2232"] * len(sources),
            "asset_id": [f"asset-{index}" for index in range(len(sources))],
            "path": [f"sub-{index}/session-{index}.nwb" for index in range(len(sources))],
            "subject_id": [f"mouse-{index}" for index in range(len(sources))],
            "ecephys_session_id": [1000 + index for index in range(len(sources))],
            "recording_day": ["EPHYS_1"] * len(sources),
            "session_number": [1] * len(sources),
            "date_of_acquisition": ["2024-01-01"] * len(sources),
            "companion_unit_count": [1] * len(sources),
            "novel_image_id": ["im104_r-1.0"] * len(sources),
        }
    )


def _schema_audit(sources: tuple[str, ...]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_nwb_path": list(sources),
            "n_image_presentation_candidates": [1] * len(sources),
            "image_presentation_paths": [
                f"/intervals/task_presentations_{index}" for index in range(len(sources))
            ],
            "audit_error": [None] * len(sources),
        }
    )


def _raw_presentations(source: str, table_path: str, offset: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 0],
            "start_time": [1.25, 1.0],
            "stop_time": [1.5, 1.25],
            "start_frame": [75 + offset, 60 + offset],
            "image_name": ["im104_r-0.7", "im104_r-1.0"],
            "is_change": [True, False],
            "active": [True, True],
            "is_image_novel": [True, True],
            "flashes_since_change": [2.0, 1.0],
            "stimulus_block": [0, 0],
            "rewarded": [True, False],
            "omitted": [False, False],
            "_nwb_path": [source, source],
            "_table_path": [table_path, table_path],
            "_table_index": [1, 0],
        }
    )


def _forced_multibatch_presentations(source: str, table_path: str) -> pl.DataFrame:
    """Return deliberately unsorted rows with identities repeated across batches."""

    table_indices = [5, 0, 4, 1, 6, 2, 3]
    image_names_by_index = [
        "im104_r-1.0",
        "im115_r-1.0",
        "im104_r-0.7",
        "im115_r-0.7",
        "im104_r-1.0",
        "im115_r-1.0",
        "im104_r-0.7",
    ]
    return pl.DataFrame(
        {
            "id": table_indices,
            "start_time": [index * 0.25 for index in table_indices],
            "stop_time": [index * 0.25 + 0.25 for index in table_indices],
            "start_frame": [index * 15 for index in table_indices],
            "image_name": [image_names_by_index[index] for index in table_indices],
            "is_change": [index > 0 for index in table_indices],
            "active": [True] * len(table_indices),
            "is_image_novel": [False] * len(table_indices),
            "flashes_since_change": [float(index + 1) for index in table_indices],
            "stimulus_block": [0] * len(table_indices),
            "rewarded": [index % 2 == 0 for index in table_indices],
            "omitted": [False] * len(table_indices),
            "_nwb_path": [source] * len(table_indices),
            "_table_path": [table_path] * len(table_indices),
            "_table_index": table_indices,
        }
    )


def _install_local_attempt_runner(monkeypatch, raw_by_source):
    def scan_table(sources, table_path, *, columns=None, **kwargs):
        assert columns is None
        assert len(sources) == 1
        return raw_by_source[sources[0]].lazy()

    def run_attempt(
        job,
        *,
        session_inventory_path,
        schema_audit_path,
        output_path,
        batch_size,
        timeout_seconds,
        python_executable,
    ):
        del timeout_seconds, python_executable
        sessions = pl.read_parquet(session_inventory_path).filter(pl.col("_nwb_path") == job.source)
        audit = pl.read_parquet(schema_audit_path).filter(pl.col("_nwb_path") == job.source)
        frame = dg.task_presentations._collect_single_session(
            sessions,
            audit,
            batch_size=batch_size,
        )
        dg.artifacts.write_frame(frame, output_path)

    monkeypatch.setattr(dg.data, "scan_nwb_table", scan_table)
    monkeypatch.setattr(dg.task_presentations, "_run_worker_attempt", run_attempt)
    return run_attempt


def test_isolated_materialization_preserves_every_source_and_deterministic_order(
    tmp_path,
    monkeypatch,
) -> None:
    sources = ("session-b", "session-a")
    sessions = _session_inventory(*sources)
    audit = _schema_audit(sources)
    paths = dict(audit.select("_nwb_path", "image_presentation_paths").iter_rows())
    raw_by_source = {
        source: _raw_presentations(source, paths[source], index * 100)
        for index, source in enumerate(sources)
    }
    _install_local_attempt_runner(monkeypatch, raw_by_source)

    result = dg.task_presentations.materialize_isolated(
        sessions,
        audit,
        max_workers=2,
        timeout_seconds=1.0,
        max_attempts=2,
        retry_delay_seconds=0.0,
        batch_size=50_000,
        temporary_root=tmp_path,
    )

    assert result.presentations.get_column("_nwb_path").to_list() == [
        "session-a",
        "session-a",
        "session-b",
        "session-b",
    ]
    assert result.presentations.group_by("_nwb_path").agg("start_frame").sort(
        "_nwb_path"
    ).get_column("start_frame").to_list() == [[160, 175], [60, 75]]
    assert result.audit.get_column("_nwb_path").sort().to_list() == [
        "session-a",
        "session-b",
    ]
    assert result.audit.get_column("materialization_status").unique().to_list() == ["pass"]
    assert result.audit.get_column("attempt_count").to_list() == [1, 1]
    assert result.audit.get_column("n_presentations").sum() == 4
    assert result.audit.get_column("requested_collect_batch_size").unique().to_list() == [50_000]
    dg.inventory.validate_task_presentations(result.presentations, sessions)


def test_timeout_is_retried_but_only_validated_success_is_returned(tmp_path, monkeypatch) -> None:
    sources = ("session-a",)
    sessions = _session_inventory(*sources)
    audit = _schema_audit(sources)
    table_path = audit.get_column("image_presentation_paths").item()
    raw_by_source = {sources[0]: _raw_presentations(sources[0], table_path, 0)}
    successful_attempt = _install_local_attempt_runner(monkeypatch, raw_by_source)
    attempts = collections.Counter()

    def time_out_twice(job, **kwargs):
        attempts[job.source] += 1
        if attempts[job.source] < 3:
            raise subprocess.TimeoutExpired(cmd="task-presentation-worker", timeout=0.01)
        successful_attempt(job, **kwargs)

    monkeypatch.setattr(dg.task_presentations, "_run_worker_attempt", time_out_twice)

    result = dg.task_presentations.materialize_isolated(
        sessions,
        audit,
        max_workers=1,
        timeout_seconds=0.01,
        max_attempts=3,
        retry_delay_seconds=0.0,
        temporary_root=tmp_path,
    )

    assert attempts == {"session-a": 3}
    assert result.audit.get_column("attempt_count").item() == 3
    assert result.audit.get_column("prior_failure_categories").item() == "timeout;timeout"


def test_exhausted_source_failure_fails_the_entire_materialization(tmp_path, monkeypatch) -> None:
    sources = ("session-a", "session-b")
    sessions = _session_inventory(*sources)
    audit = _schema_audit(sources)
    attempts = collections.Counter()

    def always_timeout(job, **kwargs):
        del kwargs
        attempts[job.source] += 1
        raise subprocess.TimeoutExpired(cmd="task-presentation-worker", timeout=0.01)

    monkeypatch.setattr(dg.task_presentations, "_run_worker_attempt", always_timeout)

    with pytest.raises(
        dg.task_presentations.TaskPresentationMaterializationError,
        match="2 of 2.*session-a.*session-b",
    ):
        dg.task_presentations.materialize_isolated(
            sessions,
            audit,
            max_workers=2,
            timeout_seconds=0.01,
            max_attempts=2,
            retry_delay_seconds=0.0,
            temporary_root=tmp_path,
        )

    assert attempts == {"session-a": 2, "session-b": 2}


def test_supervisor_cancellation_prevents_retries(tmp_path, monkeypatch) -> None:
    sessions = _session_inventory("session-a")
    job = dg.task_presentations._MaterializationJob(
        index=0,
        source="session-a",
        table_path="/intervals/task_presentations",
    )
    stop_event = threading.Event()
    attempts = []

    def cancelled_attempt(job, **kwargs):
        del job, kwargs
        attempts.append("started")
        stop_event.set()
        raise subprocess.TimeoutExpired(cmd="task-presentation-worker", timeout=0.01)

    monkeypatch.setattr(dg.task_presentations, "_run_worker_attempt", cancelled_attempt)

    with pytest.raises(
        dg.task_presentations.TaskPresentationMaterializationError,
        match="supervisor cancelled",
    ):
        dg.task_presentations._materialize_job_with_retries(
            job,
            stop_event=stop_event,
            session_inventory=sessions,
            session_inventory_path=tmp_path / "sessions.parquet",
            schema_audit_path=tmp_path / "schema.parquet",
            temporary_directory=tmp_path,
            batch_size=50_000,
            timeout_seconds=0.01,
            max_attempts=3,
            retry_delay_seconds=0.0,
            python_executable=pathlib.Path("/example/python"),
        )

    assert attempts == ["started"]


def test_wrong_source_output_is_rejected_and_never_silently_substituted(
    tmp_path,
    monkeypatch,
) -> None:
    sources = ("session-a",)
    sessions = _session_inventory(*sources)
    audit = _schema_audit(sources)
    table_path = audit.get_column("image_presentation_paths").item()
    wrong = _raw_presentations("different-session", table_path, 0)

    def wrong_source(job, **kwargs):
        del job
        frame = wrong.with_columns(
            pl.lit("asset-0").alias("asset_id"),
            pl.lit("mouse-0").alias("subject_id"),
            pl.concat_str(pl.lit("asset-0:"), pl.col("_table_index")).alias("presentation_row_key"),
            pl.lit(table_path).alias("task_presentation_path"),
            pl.lit("raw_presentation_labels_not_reward_state").alias("presentation_label_scope"),
            pl.lit(None, dtype=pl.String).alias("stimulus_name"),
        )
        dg.artifacts.write_frame(frame, kwargs["output_path"])

    monkeypatch.setattr(dg.task_presentations, "_run_worker_attempt", wrong_source)

    with pytest.raises(
        dg.task_presentations.TaskPresentationMaterializationError,
        match="canonical source",
    ):
        dg.task_presentations.materialize_isolated(
            sessions,
            audit,
            max_workers=1,
            timeout_seconds=1.0,
            max_attempts=1,
            retry_delay_seconds=0.0,
            temporary_root=tmp_path,
        )


def test_single_session_worker_requests_large_collect_batch(monkeypatch) -> None:
    sessions = _session_inventory("session-a")
    audit = _schema_audit(("session-a",))
    observed = {}
    table_path = audit.get_column("image_presentation_paths").item()
    output = _raw_presentations("session-a", table_path, 0).with_columns(
        pl.lit("asset-0").alias("asset_id"),
        pl.lit("mouse-0").alias("subject_id"),
        pl.concat_str(pl.lit("asset-0:"), pl.col("_table_index")).alias("presentation_row_key"),
        pl.lit(table_path).alias("task_presentation_path"),
        pl.lit("raw_presentation_labels_not_reward_state").alias("presentation_label_scope"),
        pl.lit(None, dtype=pl.String).alias("stimulus_name"),
    )

    class BatchProbe:
        def collect_batches(self, *, chunk_size, maintain_order):
            observed["chunk_size"] = chunk_size
            observed["maintain_order"] = maintain_order
            yield output

    monkeypatch.setattr(
        dg.inventory,
        "scan_task_presentations",
        lambda session_inventory, schema_audit: BatchProbe(),
    )

    frame = dg.task_presentations._collect_single_session(
        sessions,
        audit,
        batch_size=50_000,
    )

    assert frame.height == 2
    assert observed == {"chunk_size": 50_000, "maintain_order": True}


def test_single_session_forced_multibatch_matches_direct_real_plan_exactly(monkeypatch) -> None:
    source = "session-a"
    sessions = _session_inventory(source)
    audit = _schema_audit((source,))
    table_path = audit.get_column("image_presentation_paths").item()
    raw = _forced_multibatch_presentations(source, table_path)

    def scan_table(sources, observed_path, *, columns=None, **kwargs):
        del kwargs
        assert sources == [source]
        assert observed_path == table_path
        assert columns is None
        return raw.lazy()

    monkeypatch.setattr(dg.data, "scan_nwb_table", scan_table)

    direct_plan = dg.inventory.scan_task_presentations(sessions, audit)
    direct_batches = tuple(direct_plan.collect_batches(chunk_size=2, maintain_order=True))
    assert [batch.height for batch in direct_batches] == [2, 2, 2, 1]
    expected = direct_plan.collect()
    observed = dg.task_presentations._collect_single_session(
        sessions,
        audit,
        batch_size=2,
    )

    polars.testing.assert_frame_equal(
        observed,
        expected,
        check_row_order=True,
        check_column_order=True,
        check_dtypes=True,
        check_exact=True,
    )
    assert observed.get_column("within_session_identity_exposure_index").to_list() == [
        1,
        1,
        2,
        2,
        3,
        3,
        4,
    ]


def test_attempt_output_with_dropped_table_row_is_rejected(tmp_path, monkeypatch) -> None:
    source = "session-a"
    sessions = _session_inventory(source)
    audit = _schema_audit((source,))
    table_path = audit.get_column("image_presentation_paths").item()
    raw = _forced_multibatch_presentations(source, table_path)

    def scan_table(sources, observed_path, *, columns=None, **kwargs):
        del kwargs
        assert sources == [source]
        assert observed_path == table_path
        assert columns is None
        return raw.lazy()

    monkeypatch.setattr(dg.data, "scan_nwb_table", scan_table)
    incomplete = (
        dg.inventory.scan_task_presentations(sessions, audit)
        .collect()
        .filter(pl.col("_table_index") != 3)
    )
    output_path = tmp_path / "incomplete.parquet"
    dg.artifacts.write_frame(incomplete, output_path)
    job = dg.task_presentations._MaterializationJob(
        index=0,
        source=source,
        table_path=table_path,
    )

    with pytest.raises(
        dg.task_presentations.TaskPresentationMaterializationError,
        match="contiguous.*0.*row count",
    ):
        dg.task_presentations._read_and_validate_attempt_output(
            output_path,
            job=job,
            session_row=sessions,
        )


def test_subprocess_attempt_has_hard_timeout_and_reports_child_failure(
    tmp_path,
    monkeypatch,
) -> None:
    job = dg.task_presentations._MaterializationJob(
        index=0,
        source="session-a",
        table_path="/intervals/task_presentations",
    )
    observed = {}

    def failed_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 7, stdout="child-out", stderr="child-error")

    monkeypatch.setattr(subprocess, "run", failed_run)

    with pytest.raises(dg.task_presentations._WorkerProcessError, match="child-error"):
        dg.task_presentations._run_worker_attempt(
            job,
            session_inventory_path=tmp_path / "sessions.parquet",
            schema_audit_path=tmp_path / "schema.parquet",
            output_path=tmp_path / "output.parquet",
            batch_size=50_000,
            timeout_seconds=123.0,
            python_executable=pathlib.Path("/example/python"),
        )

    assert observed["timeout"] == 123.0
    assert observed["check"] is False
    assert observed["capture_output"] is True
    assert observed["text"] is True
    assert "--source" in observed["command"]
    assert observed["command"][0] == "/example/python"


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("max_workers", 0),
        ("max_workers", 9),
        ("timeout_seconds", 0.0),
        ("max_attempts", 0),
        ("max_attempts", 6),
        ("retry_delay_seconds", -1.0),
        ("batch_size", 0),
    ],
)
def test_materialization_bounds_are_validated(keyword, value) -> None:
    arguments = {
        "max_workers": 1,
        "timeout_seconds": 1.0,
        "max_attempts": 1,
        "retry_delay_seconds": 0.0,
        "batch_size": 50_000,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError):
        dg.task_presentations.materialize_isolated(
            _session_inventory("session-a"),
            _schema_audit(("session-a",)),
            **arguments,
        )
