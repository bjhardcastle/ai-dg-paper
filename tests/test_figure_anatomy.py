"""Focused data and rendering contracts for anatomical Figure 4."""

import argparse
import datetime
import hashlib
import importlib.util
import json
import pathlib
import sys

import polars as pl
import pytest

import dg.anatomical_enrichment
import dg.figure_anatomy

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "11_plot_anatomy.py"
SPEC = importlib.util.spec_from_file_location("dg_plot_anatomy_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
PLOT_ANATOMY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT_ANATOMY
SPEC.loader.exec_module(PLOT_ANATOMY)


def _unit_anatomy_fixture() -> pl.DataFrame:
    rows = []
    for mouse_index in range(5):
        for session_index in range(2):
            source = f"fixture_{mouse_index}_{session_index}.nwb"
            for unit_index in range(30):
                if unit_index < 10:
                    parent_id, acronym, name, magnitude = 315, "Isocortex", "Isocortex", 3.0
                elif unit_index < 20:
                    parent_id, acronym, name, magnitude = 549, "TH", "Thalamus", 1.0
                else:
                    parent_id, acronym, name, magnitude = None, None, None, 0.5
                rows.append(
                    {
                        "_nwb_path": source,
                        "_table_index": unit_index,
                        "subject_id": f"mouse_{mouse_index}",
                        "ecephys_session_id": mouse_index * 2 + session_index,
                        "probe_id": session_index,
                        "axis_weight": magnitude * (-1 if unit_index % 2 else 1),
                        "well_isolated": True,
                        "parent_region_id": parent_id,
                        "parent_region_acronym": acronym,
                        "parent_region_name": name,
                        "anatomy_analysis_included": parent_id is not None,
                        "anterior_posterior_ccf_coordinate": 2_000 + unit_index * 20,
                        "dorsal_ventral_ccf_coordinate": 1_000 + unit_index * 15,
                        "left_right_ccf_coordinate": 4_000 + unit_index * 10,
                    }
                )
    return pl.DataFrame(rows).with_columns(pl.col("_table_index").cast(pl.UInt32))


def _tables():
    return dg.anatomical_enrichment.analyze_anatomical_enrichment(
        _unit_anatomy_fixture(),
        seed=5,
        n_bootstrap=100,
        n_permutations=100,
        code_version="fixture",
    )


def _prepared() -> dg.figure_anatomy.AnatomyFigureData:
    tables = _tables()
    return dg.figure_anatomy.prepare_anatomy_figure_data(
        tables.unit_scores,
        tables.region_coverage,
        tables.mouse_effects,
        tables.statistics,
        tables.permutation_null,
    )


def test_prepare_selects_absolute_loading_lead_and_preserves_all_mice() -> None:
    prepared = _prepared()

    assert prepared.lead_region_acronym == "Isocortex"
    assert prepared.statistics.height == 2
    assert prepared.mouse_effects.height == 10
    assert prepared.lead_permutation_null.height == 100
    assert prepared.unit_coordinates.height == 200


def test_prepare_rejects_statistic_that_disagrees_with_mouse_rows() -> None:
    tables = _tables()
    changed = tables.statistics.with_columns(
        pl.when(pl.col("result_id") == "absolute_loading__Isocortex")
        .then(pl.col("estimate") + 0.1)
        .otherwise(pl.col("estimate"))
        .alias("estimate"),
        pl.when(pl.col("result_id") == "absolute_loading__Isocortex")
        .then(pl.col("ci_low") + 0.1)
        .otherwise(pl.col("ci_low"))
        .alias("ci_low"),
        pl.when(pl.col("result_id") == "absolute_loading__Isocortex")
        .then(pl.col("ci_high") + 0.1)
        .otherwise(pl.col("ci_high"))
        .alias("ci_high"),
    )

    with pytest.raises(ValueError, match="disagrees with plotted mouse"):
        dg.figure_anatomy.prepare_anatomy_figure_data(
            tables.unit_scores,
            tables.region_coverage,
            tables.mouse_effects,
            changed,
            tables.permutation_null,
        )


def test_prepare_rejects_changed_locked_coverage_threshold() -> None:
    tables = _tables()
    changed = tables.region_coverage.with_columns(pl.lit(4).alias("minimum_mice"))

    with pytest.raises(ValueError, match="unexpected mouse minimum"):
        dg.figure_anatomy.prepare_anatomy_figure_data(
            tables.unit_scores,
            changed,
            tables.mouse_effects,
            tables.statistics,
            tables.permutation_null,
        )


def test_create_figure_has_three_labeled_axes() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = PLOT_ANATOMY.create_figure(_prepared())
    try:
        assert len(figure.axes) == 3
        labels = {text.get_text() for axis in figure.axes for text in axis.texts}
        assert {"A", "B", "C"}.issubset(labels)
    finally:
        plt.close(figure)


def test_analysis_manifest_binds_figure_inputs(tmp_path: pathlib.Path) -> None:
    inputs = {
        key: tmp_path / pathlib.Path(relative).name
        for key, relative in PLOT_ANATOMY.INPUT_KEYS.items()
    }
    outputs = {}
    for key, path in inputs.items():
        path.write_text(key, encoding="utf-8")
        outputs[key] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    manifest = tmp_path / "anatomy_analysis_run.json"
    manifest.write_text(
        json.dumps(
            {
                "analysis_id": "anatomical_axis_enrichment",
                "analysis_tier": "discovery",
                "run_status": "complete",
                "authoritative": True,
                "confirmation_accessed": False,
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )

    assert (
        PLOT_ANATOMY._validated_analysis_manifest(
            manifest,
            inputs,
            results_root=tmp_path,
        )["run_status"]
        == "complete"
    )
    inputs["anatomy_region_coverage"].write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after"):
        PLOT_ANATOMY._validated_analysis_manifest(
            manifest,
            inputs,
            results_root=tmp_path,
        )


def test_run_figure_writes_three_formats_and_source_data(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("matplotlib")
    tables = _tables()
    results_root = tmp_path / "results"
    (results_root / "tables").mkdir(parents=True)
    frames = {
        "anatomical_axis_unit_scores": tables.unit_scores,
        "anatomy_region_coverage": tables.region_coverage,
        "mouse_anatomical_enrichment": tables.mouse_effects,
        "anatomy_enrichment_statistics": tables.statistics,
        "anatomy_permutation_null": tables.permutation_null,
    }
    outputs = {}
    for key, frame in frames.items():
        destination = results_root / PLOT_ANATOMY.INPUT_KEYS[key]
        if destination.suffix == ".parquet":
            frame.write_parquet(destination)
        else:
            frame.write_csv(destination)
        outputs[key] = {
            "path": str(destination.relative_to(results_root)),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "size_bytes": destination.stat().st_size,
        }
    manifest_path = results_root / "manifests" / "anatomy_analysis_run.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "analysis_id": "anatomical_axis_enrichment",
                "analysis_tier": "discovery",
                "run_status": "complete",
                "authoritative": True,
                "confirmation_accessed": False,
                "dandiset_version": "fixture",
                "code_version": "fixture",
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    figure_root = results_root / "figures" / "main"
    figure_root.mkdir(parents=True)
    figure_manifest = figure_root / "figure_4_anatomy_manifest.json"

    PLOT_ANATOMY._run_figure(
        argparse.Namespace(dpi=120),
        started_at=datetime.datetime.now(datetime.UTC),
        results_root=results_root,
        figure_root=figure_root,
        figure_manifest_path=figure_manifest,
    )

    for suffix in ("svg", "pdf", "png"):
        assert (figure_root / f"figure_4_anatomy.{suffix}").stat().st_size > 1_000
    source_root = figure_root / "figure_4_anatomy_source_data"
    assert (source_root / "manifest.json").is_file()
    assert len(list(source_root.glob("*.csv"))) == 5
    assert json.loads(figure_manifest.read_text())["run_status"] == "complete"


@pytest.mark.parametrize(("value", "expected"), ((0.02, "0.020"), (0.0009, "<0.001")))
def test_p_value_format(value: float, expected: str) -> None:
    assert PLOT_ANATOMY._format_p(value) == expected
