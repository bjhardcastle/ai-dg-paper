"""Tests for result-artifact writing and software provenance."""

import pathlib

import polars as pl
import pytest

import dg.artifacts


@pytest.mark.parametrize("extension", ["parquet", "csv", "json", "ndjson"])
def test_write_frame_round_trip(tmp_path: pathlib.Path, extension: str) -> None:
    # These identifiers are from published session sub-604914, 2022-04-27.
    frame = pl.DataFrame({"ecephys_session_id": [1173189336], "trial_id": [130]})
    output = tmp_path / "nested" / f"table.{extension}"

    dg.artifacts.write_frame(frame.lazy(), output)

    if extension == "parquet":
        observed = pl.read_parquet(output)
    elif extension == "csv":
        observed = pl.read_csv(output)
    elif extension == "json":
        observed = pl.read_json(output)
    else:
        observed = pl.read_ndjson(output)
    assert observed.to_dicts() == frame.to_dicts()
    assert output.stat().st_mode & 0o777 == 0o644
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_write_frame_rejects_unknown_extension(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="unsupported"):
        dg.artifacts.write_frame(pl.DataFrame({"trial_id": [130]}), tmp_path / "table.txt")


def test_initialize_results_tree(tmp_path: pathlib.Path) -> None:
    root = dg.artifacts.initialize_results_tree(tmp_path / "results")

    assert all((root / path).is_dir() for path in dg.artifacts.RESULT_SUBDIRECTORIES)


def test_capture_software_environment_has_required_provenance() -> None:
    result = dg.artifacts.capture_software_environment(repository=pathlib.Path.cwd())

    assert "package.lazynwb=1.0.0.dev8" in result
    assert "git.revision=" in result
    assert "git.dirty=" in result
