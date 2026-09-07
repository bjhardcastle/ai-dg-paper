"""Validated, atomic result-artifact writing and provenance capture."""

from __future__ import annotations

import datetime
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile
from typing import Any

import polars as pl

RESULT_SUBDIRECTORIES = (
    "manifests",
    "tables",
    "models",
    "figures/main",
    "figures/supplementary",
    "figures/all_units",
    "unit_browser/units",
)


def initialize_results_tree(root: str | os.PathLike[str]) -> pathlib.Path:
    """Create the planned artifact directories and return the resolved root."""

    result_root = pathlib.Path(root).resolve()
    for relative_path in RESULT_SUBDIRECTORIES:
        (result_root / relative_path).mkdir(parents=True, exist_ok=True)
    return result_root


def write_frame(
    frame: pl.DataFrame | pl.LazyFrame,
    destination: str | os.PathLike[str],
) -> pathlib.Path:
    """Atomically write a Polars frame based on a supported file extension."""

    output = pathlib.Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    collected = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    suffix = output.suffix.lower()
    if suffix not in {".parquet", ".csv", ".json", ".ndjson"}:
        raise ValueError(f"unsupported table extension: {suffix!r}")

    temporary = _temporary_sibling(output)
    try:
        if suffix == ".parquet":
            collected.write_parquet(temporary, compression="zstd", statistics=True)
        elif suffix == ".csv":
            collected.write_csv(temporary)
        elif suffix == ".json":
            collected.write_json(temporary)
        else:
            collected.write_ndjson(temporary)
        temporary.chmod(0o644)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_text(text: str, destination: str | os.PathLike[str]) -> pathlib.Path:
    """Atomically write UTF-8 text."""

    output = pathlib.Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(output)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.chmod(0o644)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def write_json(
    value: Any,
    destination: str | os.PathLike[str],
    *,
    indent: int = 2,
) -> pathlib.Path:
    """Atomically write stable, newline-terminated JSON."""

    return write_text(
        json.dumps(value, indent=indent, sort_keys=True, default=str) + "\n",
        destination,
    )


def capture_software_environment(
    *,
    repository: str | os.PathLike[str] | None = None,
    package_names: tuple[str, ...] = (
        "lazynwb",
        "numpy",
        "polars",
        "pyarrow",
        "pytest",
        "ruff",
    ),
) -> str:
    """Return a human-readable, deterministic software/provenance record."""

    recorded_at = datetime.datetime.now(datetime.UTC).isoformat()
    lines = [
        f"recorded_at_utc={recorded_at}",
        f"python={platform.python_version()}",
        f"python_executable={sys.executable}",
        f"platform={platform.platform()}",
    ]
    for package_name in package_names:
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = "not-installed"
        lines.append(f"package.{package_name}={version}")

    if repository is not None:
        repo = pathlib.Path(repository).resolve()
        revision = _run_git(repo, "rev-parse", "HEAD")
        status = _run_git(repo, "status", "--porcelain")
        lines.extend(
            [
                f"git.revision={revision or 'unavailable'}",
                f"git.dirty={'true' if status else 'false'}",
            ]
        )
    return "\n".join(lines) + "\n"


def _temporary_sibling(destination: pathlib.Path) -> pathlib.Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return pathlib.Path(name)


def _run_git(repository: pathlib.Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()
