"""Resilient materialization of projected task-presentation rows.

Remote HDF5 reads can block below Python while holding h5py's process-wide
lock.  This module contains that failure inside one short-lived subprocess per
session.  The supervisor applies a wall-clock timeout, bounded retries, and
accepts an attempt only after its atomic output passes the ordinary inventory
validation against the canonical frozen source.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import math
import pathlib
import subprocess
import sys
import tempfile
import threading

import polars as pl

_SOURCE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPOSITORY_ROOT = _SOURCE_ROOT.parent
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

import dg.artifacts  # noqa: E402
import dg.inventory  # noqa: E402

DEFAULT_BATCH_SIZE = 50_000
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_WORKERS = 4
DEFAULT_RETRY_DELAY_SECONDS = 1.0
DEFAULT_TIMEOUT_SECONDS = 300.0
MAX_ATTEMPTS = 5
MAX_WORKERS = 8

_DERIVED_PRESENTATION_COLUMNS = (
    "image_id",
    "image_relative_contrast",
    "task_presentation_path",
    "presentation_row_key",
    "metadata_novel_image_id",
    "metadata_novel_relative_contrast",
    "presentation_label_scope",
    "within_session_identity_exposure_index",
    "is_metadata_designated_novel_identity",
)


class TaskPresentationMaterializationError(RuntimeError):
    """Raised when any canonical session cannot be materialized and validated."""


class _WorkerProcessError(RuntimeError):
    """Raised when a task-presentation worker exits unsuccessfully."""


@dataclasses.dataclass(frozen=True, slots=True)
class TaskPresentationMaterializationResult:
    """Fully reconciled presentation rows and one audit record per session."""

    presentations: pl.DataFrame
    audit: pl.DataFrame


@dataclasses.dataclass(frozen=True, slots=True)
class _MaterializationJob:
    index: int
    source: str
    table_path: str


@dataclasses.dataclass(frozen=True, slots=True)
class _CompletedJob:
    job: _MaterializationJob
    frame: pl.DataFrame
    attempt_count: int
    prior_failure_categories: tuple[str, ...]


def materialize_isolated(
    session_inventory: pl.DataFrame,
    schema_audit: pl.DataFrame,
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temporary_root: str | pathlib.Path | None = None,
    python_executable: str | pathlib.Path | None = None,
) -> TaskPresentationMaterializationResult:
    """Materialize all sessions through bounded, independently killable workers.

    Temporary outputs are deliberately not a reusable cache.  A worker writes
    one attempt atomically, the supervisor re-reads and validates it, and the
    directory is discarded after all source rows have been concatenated.  Any
    exhausted source fails the entire call.
    """

    _validate_runtime_bounds(
        max_workers=max_workers,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        batch_size=batch_size,
    )
    path_mapping = dg.inventory.resolve_task_presentation_paths(
        session_inventory,
        schema_audit,
    ).sort(dg.inventory.SOURCE_COLUMN)
    jobs = tuple(
        _MaterializationJob(
            index=index,
            source=row[dg.inventory.SOURCE_COLUMN],
            table_path=row["task_presentation_path"],
        )
        for index, row in enumerate(path_mapping.iter_rows(named=True))
    )
    if not jobs:
        raise TaskPresentationMaterializationError(
            "no canonical task-presentation sessions were available to materialize"
        )

    # Preserve a virtual-environment launcher symlink. Resolving it can bypass
    # the adjacent pyvenv.cfg and start a child without the locked environment.
    executable = pathlib.Path(python_executable or sys.executable).expanduser().absolute()
    root = pathlib.Path(temporary_root).resolve() if temporary_root is not None else None
    completed_by_index: dict[int, _CompletedJob] = {}
    failures: list[tuple[_MaterializationJob, Exception]] = []
    with tempfile.TemporaryDirectory(prefix="dg-m0-task-presentations-", dir=root) as name:
        temporary_directory = pathlib.Path(name)
        session_inventory_path = temporary_directory / "session_inventory.parquet"
        schema_audit_path = temporary_directory / "schema_audit.parquet"
        dg.artifacts.write_frame(session_inventory, session_inventory_path)
        dg.artifacts.write_frame(schema_audit, schema_audit_path)

        worker_count = min(max_workers, len(jobs))
        stop_event = threading.Event()
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="dg-presentation-supervisor",
        )
        future_to_job: dict[concurrent.futures.Future[_CompletedJob], _MaterializationJob] = {}
        try:
            for job in jobs:
                future = executor.submit(
                    _materialize_job_with_retries,
                    job,
                    stop_event=stop_event,
                    session_inventory=session_inventory,
                    session_inventory_path=session_inventory_path,
                    schema_audit_path=schema_audit_path,
                    temporary_directory=temporary_directory,
                    batch_size=batch_size,
                    timeout_seconds=timeout_seconds,
                    max_attempts=max_attempts,
                    retry_delay_seconds=retry_delay_seconds,
                    python_executable=executable,
                )
                future_to_job[future] = job
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    completed = future.result()
                except Exception as error:
                    failures.append((job, error))
                else:
                    completed_by_index[job.index] = completed
        except BaseException:
            stop_event.set()
            for future in future_to_job:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        if failures:
            details = "; ".join(
                f"{job.source}: {error}"
                for job, error in sorted(failures, key=lambda x: x[0].index)
            )
            raise TaskPresentationMaterializationError(
                f"{len(failures)} of {len(jobs)} canonical task-presentation sources failed: "
                f"{details}"
            )
        if set(completed_by_index) != {job.index for job in jobs}:
            raise TaskPresentationMaterializationError(
                "task-presentation supervisor lost one or more completed source results"
            )

        completed_jobs = tuple(completed_by_index[job.index] for job in jobs)
        presentations = _concatenate_completed_jobs(completed_jobs, session_inventory)
        audit = _build_materialization_audit(
            completed_jobs,
            timeout_seconds=timeout_seconds,
            batch_size=batch_size,
        )
        _validate_materialization_audit(audit, path_mapping)

    return TaskPresentationMaterializationResult(
        presentations=presentations,
        audit=audit,
    )


def _materialize_job_with_retries(
    job: _MaterializationJob,
    *,
    stop_event: threading.Event,
    session_inventory: pl.DataFrame,
    session_inventory_path: pathlib.Path,
    schema_audit_path: pathlib.Path,
    temporary_directory: pathlib.Path,
    batch_size: int,
    timeout_seconds: float,
    max_attempts: int,
    retry_delay_seconds: float,
    python_executable: pathlib.Path,
) -> _CompletedJob:
    session_row = session_inventory.filter(pl.col(dg.inventory.SOURCE_COLUMN) == job.source)
    if session_row.height != 1:
        raise TaskPresentationMaterializationError(
            f"canonical source {job.source!r} resolved to {session_row.height} session rows"
        )

    failure_categories: list[str] = []
    last_error: Exception | None = None
    for attempt_number in range(1, max_attempts + 1):
        if stop_event.is_set():
            raise TaskPresentationMaterializationError(
                f"supervisor cancelled canonical source {job.source!r} before its next attempt"
            )
        output_path = temporary_directory / (
            f"session-{job.index:03d}-attempt-{attempt_number:02d}.parquet"
        )
        output_path.unlink(missing_ok=True)
        try:
            _run_worker_attempt(
                job,
                session_inventory_path=session_inventory_path,
                schema_audit_path=schema_audit_path,
                output_path=output_path,
                batch_size=batch_size,
                timeout_seconds=timeout_seconds,
                python_executable=python_executable,
            )
            frame = _read_and_validate_attempt_output(
                output_path,
                job=job,
                session_row=session_row,
            )
        except Exception as error:
            output_path.unlink(missing_ok=True)
            last_error = error
            failure_categories.append(_failure_category(error))
            if stop_event.is_set():
                raise TaskPresentationMaterializationError(
                    f"supervisor cancelled canonical source {job.source!r} after its active attempt"
                ) from error
            if attempt_number < max_attempts and retry_delay_seconds:
                delay = min(30.0, retry_delay_seconds * (2 ** (attempt_number - 1)))
                stop_event.wait(delay)
            continue
        return _CompletedJob(
            job=job,
            frame=frame,
            attempt_count=attempt_number,
            prior_failure_categories=tuple(failure_categories),
        )

    assert last_error is not None
    raise TaskPresentationMaterializationError(
        f"canonical source {job.source!r} exhausted {max_attempts} attempts; "
        f"last error: {last_error}"
    ) from last_error


def _run_worker_attempt(
    job: _MaterializationJob,
    *,
    session_inventory_path: pathlib.Path,
    schema_audit_path: pathlib.Path,
    output_path: pathlib.Path,
    batch_size: int,
    timeout_seconds: float,
    python_executable: pathlib.Path,
) -> None:
    command = (
        str(python_executable),
        str(pathlib.Path(__file__).resolve()),
        "--worker",
        "--session-inventory",
        str(session_inventory_path),
        "--schema-audit",
        str(schema_audit_path),
        "--source",
        job.source,
        "--output",
        str(output_path),
        "--batch-size",
        str(batch_size),
    )
    completed = subprocess.run(
        command,
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode:
        diagnostic = (completed.stderr.strip() or completed.stdout.strip())[-4_000:]
        raise _WorkerProcessError(
            f"worker exited with status {completed.returncode} for canonical source "
            f"{job.source!r}: {diagnostic or 'no child diagnostic'}"
        )


def _read_and_validate_attempt_output(
    output_path: pathlib.Path,
    *,
    job: _MaterializationJob,
    session_row: pl.DataFrame,
) -> pl.DataFrame:
    if not output_path.is_file():
        raise TaskPresentationMaterializationError(
            f"worker produced no atomic output for canonical source {job.source!r}"
        )
    frame = pl.read_parquet(output_path)
    if frame.is_empty():
        raise TaskPresentationMaterializationError(
            f"worker produced no presentation rows for canonical source {job.source!r}"
        )

    observed_sources = set(frame.get_column(dg.inventory.SOURCE_COLUMN).to_list())
    if observed_sources != {job.source}:
        raise TaskPresentationMaterializationError(
            "worker output canonical source mismatch: "
            f"expected {job.source!r}, observed {sorted(observed_sources)!r}"
        )
    observed_paths = set(frame.get_column("task_presentation_path").to_list())
    if observed_paths != {job.table_path}:
        raise TaskPresentationMaterializationError(
            f"worker output table-path mismatch for canonical source {job.source!r}: "
            f"expected {job.table_path!r}, observed {sorted(observed_paths)!r}"
        )
    required_columns = {
        *dg.inventory.PRESENTATION_REQUIRED_COLUMNS,
        *dg.inventory.PRESENTATION_OPTIONAL_DTYPES,
        dg.inventory.SOURCE_COLUMN,
        dg.inventory.TABLE_PATH_COLUMN,
        dg.inventory.TABLE_INDEX_COLUMN,
        *_DERIVED_PRESENTATION_COLUMNS,
        *(
            column
            for column in dg.inventory.SESSION_METADATA_COLUMNS
            if column in session_row.columns
        ),
    }
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise TaskPresentationMaterializationError(
            f"worker output for canonical source {job.source!r} is missing projected columns: "
            f"{sorted(missing_columns)}"
        )
    dg.inventory.validate_task_presentations(frame, session_row)
    _validate_complete_table_indices(frame)
    return frame.sort(
        dg.inventory.SOURCE_COLUMN,
        "start_frame",
        dg.inventory.TABLE_INDEX_COLUMN,
    )


def _concatenate_completed_jobs(
    completed_jobs: tuple[_CompletedJob, ...],
    session_inventory: pl.DataFrame,
) -> pl.DataFrame:
    reference_schema = completed_jobs[0].frame.schema
    schema_mismatches = [
        completed.job.source
        for completed in completed_jobs[1:]
        if completed.frame.schema != reference_schema
    ]
    if schema_mismatches:
        raise TaskPresentationMaterializationError(
            "validated per-session task-presentation schemas differ for canonical sources: "
            f"{schema_mismatches}"
        )
    presentations = pl.concat(
        (completed.frame for completed in completed_jobs),
        how="vertical",
        rechunk=True,
    ).sort(
        dg.inventory.SOURCE_COLUMN,
        "start_frame",
        dg.inventory.TABLE_INDEX_COLUMN,
    )
    dg.inventory.validate_task_presentations(presentations, session_inventory)
    _validate_complete_table_indices(presentations)
    return presentations


def _build_materialization_audit(
    completed_jobs: tuple[_CompletedJob, ...],
    *,
    timeout_seconds: float,
    batch_size: int,
) -> pl.DataFrame:
    return pl.DataFrame(
        (
            {
                dg.inventory.SOURCE_COLUMN: completed.job.source,
                "task_presentation_path": completed.job.table_path,
                "attempt_count": completed.attempt_count,
                "prior_failure_categories": ";".join(completed.prior_failure_categories),
                "n_presentations": completed.frame.height,
                "worker_timeout_seconds": timeout_seconds,
                "requested_collect_batch_size": batch_size,
                "worker_isolation": "one_canonical_session_per_subprocess",
                "attempt_output_scope": "atomic_temporary_non_reusable",
                "materialization_status": "pass",
            }
            for completed in completed_jobs
        ),
        infer_schema_length=None,
    ).sort(dg.inventory.SOURCE_COLUMN)


def _validate_materialization_audit(
    audit: pl.DataFrame,
    path_mapping: pl.DataFrame,
) -> None:
    expected = path_mapping.sort(dg.inventory.SOURCE_COLUMN)
    observed = audit.select(
        dg.inventory.SOURCE_COLUMN,
        "task_presentation_path",
    ).sort(dg.inventory.SOURCE_COLUMN)
    if observed.to_dicts() != expected.to_dicts():
        raise TaskPresentationMaterializationError(
            "task-presentation materialization audit does not exactly cover canonical sources"
        )
    if (
        audit.height != expected.height
        or not audit.get_column("materialization_status").eq("pass").all()
    ):
        raise TaskPresentationMaterializationError(
            "task-presentation materialization audit contains missing or failed sources"
        )
    if (audit.get_column("attempt_count") < 1).any() or (
        audit.get_column("n_presentations") < 1
    ).any():
        raise TaskPresentationMaterializationError(
            "task-presentation materialization audit contains invalid attempt or row counts"
        )


def _collect_single_session(
    session_inventory: pl.DataFrame,
    schema_audit: pl.DataFrame,
    *,
    batch_size: int,
) -> pl.DataFrame:
    """Collect one projected session with a large requested IO-plugin batch."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if session_inventory.height != 1:
        raise ValueError("task-presentation worker requires exactly one session row")
    lazy_presentations = dg.inventory.scan_task_presentations(
        session_inventory,
        schema_audit,
    )
    batches = tuple(
        lazy_presentations.collect_batches(
            chunk_size=batch_size,
            maintain_order=True,
        )
    )
    if not batches:
        raise ValueError("task-presentation worker collected no batches")
    frame = batches[0] if len(batches) == 1 else pl.concat(batches, how="vertical", rechunk=True)
    dg.inventory.validate_task_presentations(frame, session_inventory)
    _validate_complete_table_indices(frame)
    return frame


def _validate_complete_table_indices(presentations: pl.DataFrame) -> None:
    """Require every source's original row indices to cover its complete table.

    ``scan_task_presentations`` does not filter rows, and lazynwb's
    ``_table_index`` is the zero-based original DynamicTable row position.
    Therefore every canonical source must cover exactly ``0..height - 1``.
    """

    index_audit = presentations.group_by(dg.inventory.SOURCE_COLUMN).agg(
        pl.len().cast(pl.Int64).alias("row_count"),
        pl.col(dg.inventory.TABLE_INDEX_COLUMN).cast(pl.Int64).min().alias("minimum_table_index"),
        pl.col(dg.inventory.TABLE_INDEX_COLUMN).cast(pl.Int64).max().alias("maximum_table_index"),
        pl.col(dg.inventory.TABLE_INDEX_COLUMN)
        .n_unique()
        .cast(pl.Int64)
        .alias("unique_table_index_count"),
    )
    invalid = index_audit.filter(
        (pl.col("minimum_table_index") != 0)
        | (pl.col("maximum_table_index") != pl.col("row_count") - 1)
        | (pl.col("unique_table_index_count") != pl.col("row_count"))
    )
    if invalid.height:
        raise TaskPresentationMaterializationError(
            "task-presentation _table_index values must be contiguous from 0 through "
            f"row count minus one within each source:\n{invalid.head(5)}"
        )


def _failure_category(error: Exception) -> str:
    if isinstance(error, subprocess.TimeoutExpired):
        return "timeout"
    if isinstance(error, _WorkerProcessError):
        return "worker_process_error"
    if isinstance(error, TaskPresentationMaterializationError):
        return "invalid_worker_output"
    if isinstance(error, OSError):
        return "io_error"
    return type(error).__name__


def _validate_runtime_bounds(
    *,
    max_workers: int,
    timeout_seconds: float,
    max_attempts: int,
    retry_delay_seconds: float,
    batch_size: int,
) -> None:
    if not 1 <= max_workers <= MAX_WORKERS:
        raise ValueError(f"max_workers must be between 1 and {MAX_WORKERS}")
    if not 1 <= max_attempts <= MAX_ATTEMPTS:
        raise ValueError(f"max_attempts must be between 1 and {MAX_ATTEMPTS}")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    if not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be nonnegative and finite")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")


def _parse_worker_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Private task-presentation materialization worker")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--session-inventory", type=pathlib.Path, required=True)
    parser.add_argument("--schema-audit", type=pathlib.Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    return parser.parse_args()


def _worker_main() -> None:
    arguments = _parse_worker_arguments()
    session_inventory = pl.read_parquet(arguments.session_inventory)
    schema_audit = pl.read_parquet(arguments.schema_audit)
    session_row = session_inventory.filter(pl.col(dg.inventory.SOURCE_COLUMN) == arguments.source)
    source_schema = schema_audit.filter(pl.col(dg.inventory.SOURCE_COLUMN) == arguments.source)
    if session_row.height != 1:
        raise ValueError(
            f"worker source resolved to {session_row.height} session rows, expected exactly one"
        )
    if source_schema.is_empty():
        raise ValueError("worker source has no schema-audit rows")
    frame = _collect_single_session(
        session_row,
        source_schema,
        batch_size=arguments.batch_size,
    )
    dg.artifacts.write_frame(frame, arguments.output)


if __name__ == "__main__":
    _worker_main()
