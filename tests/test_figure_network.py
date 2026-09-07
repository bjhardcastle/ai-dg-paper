"""Source-data and rendering contracts for simultaneous-network Figure 5."""

import argparse
import datetime
import hashlib
import importlib.util
import json
import pathlib
import sys

import polars as pl
import pytest

import dg.figure_network

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "13_plot_network.py"
SPEC = importlib.util.spec_from_file_location("dg_plot_network_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
PLOT_NETWORK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT_NETWORK
SPEC.loader.exec_module(PLOT_NETWORK)


def _mouse_blocks() -> pl.DataFrame:
    rows = []
    for mouse_index in range(5):
        real = {
            "engaged_1": 0.06 + 0.002 * mouse_index,
            "no_reward": 0.01,
            "engaged_2": 0.05 + 0.002 * mouse_index,
        }
        for block in ("engaged_1", "no_reward", "engaged_2"):
            rows.append(
                {
                    "subject_id": f"m{mouse_index}",
                    "source_region": "VISp",
                    "target_region": "MOs",
                    "reward_block": block,
                    "n_sessions": 2,
                    "n_trials": 40,
                    "source_added_r2": real[block],
                    "shift_source_added_r2": 0.01,
                    "real_minus_shift_source_added_r2": real[block] - 0.01,
                }
            )
    return pl.DataFrame(rows)


def _mouse_effects() -> pl.DataFrame:
    rows = []
    for mouse_index in range(5):
        real = 0.045 + 0.002 * mouse_index
        rows.append(
            {
                "subject_id": f"m{mouse_index}",
                "source_region": "VISp",
                "target_region": "MOs",
                "n_sessions": 2,
                "n_trials": 120,
                "engaged_1_source_added_r2": 0.06 + 0.002 * mouse_index,
                "no_reward_source_added_r2": 0.01,
                "engaged_2_source_added_r2": 0.05 + 0.002 * mouse_index,
                "real_reversible_contrast": real,
                "shift_reversible_contrast": 0.0,
                "controlled_reversible_contrast": real,
                "real_hysteresis_contrast": -0.01,
            }
        )
    return pl.DataFrame(rows)


def _statistics() -> pl.DataFrame:
    estimate = float(_mouse_effects().get_column("controlled_reversible_contrast").mean())
    return pl.DataFrame(
        {
            "analysis_id": ["simultaneous_network_interaction"],
            "result_id": ["VISp_to_MOs_controlled_reversible_source_added_r2"],
            "estimate": [estimate],
            "ci_low": [estimate - 0.01],
            "ci_high": [estimate + 0.01],
            "p_value": [0.03125],
            "adjusted_p_value": [0.03125],
            "n_mice": [5],
            "status": ["pass"],
        }
    )


def _prepared() -> dg.figure_network.NetworkFigureData:
    return dg.figure_network.prepare_network_figure_data(
        _mouse_blocks(),
        _mouse_effects(),
        _statistics(),
        source_region="VISp",
        target_region="MOs",
    )


def test_prepare_binds_blocks_effects_and_statistic() -> None:
    prepared = _prepared()

    assert prepared.mouse_blocks.height == 15
    assert prepared.mouse_effects.height == 5
    assert prepared.block_summary.height == 3
    assert prepared.statistic.height == 1
    e1 = prepared.block_summary.filter(pl.col("reward_block") == "engaged_1").row(0, named=True)
    assert e1["mean_source_added_r2"] == pytest.approx(0.064)
    assert e1["n_mice"] == 5
    assert e1["n_sessions"] == 10


def test_prepare_rejects_effect_not_reconstructed_from_blocks() -> None:
    effects = _mouse_effects().with_columns(
        pl.when(pl.col("subject_id") == "m0")
        .then(pl.col("real_reversible_contrast") + 0.01)
        .otherwise(pl.col("real_reversible_contrast"))
        .alias("real_reversible_contrast")
    )
    with pytest.raises(ValueError, match="reconstruct|disagrees"):
        dg.figure_network.prepare_network_figure_data(
            _mouse_blocks(),
            effects,
            _statistics(),
            source_region="VISp",
            target_region="MOs",
        )


def test_create_figure_has_three_panel_labels() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = PLOT_NETWORK.create_figure(_prepared())
    try:
        assert len(figure.axes) == 3
        labels = {text.get_text() for axis in figure.axes for text in axis.texts}
        assert {"A", "B", "C"}.issubset(labels)
    finally:
        plt.close(figure)


def test_analysis_manifest_binds_inputs_and_keeps_confirmation_sealed(
    tmp_path: pathlib.Path,
) -> None:
    inputs = {
        key: tmp_path / pathlib.Path(relative).name
        for key, relative in PLOT_NETWORK.INPUT_KEYS.items()
    }
    outputs = {}
    for key, path in inputs.items():
        path.write_text(key, encoding="utf-8")
        outputs[key] = {
            "path": f"tables/{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    manifest = tmp_path / "network_analysis_run.json"
    manifest.write_text(
        json.dumps(
            {
                "analysis_id": "simultaneous_network_interaction",
                "analysis_tier": "discovery",
                "run_status": "complete",
                "authoritative": True,
                "confirmation_accessed": False,
                "nominated_pair": {"source_region": "VISp", "target_region": "MOs"},
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )

    result = PLOT_NETWORK._validated_analysis_manifest(manifest, inputs)
    assert result["confirmation_accessed"] is False
    inputs["network_statistics"].write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after"):
        PLOT_NETWORK._validated_analysis_manifest(manifest, inputs)


def test_run_figure_writes_vector_raster_and_source_data(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("matplotlib")
    results_root = tmp_path / "results"
    tables = results_root / "tables"
    manifests = results_root / "manifests"
    figures = results_root / "figures" / "main"
    tables.mkdir(parents=True)
    manifests.mkdir(parents=True)
    figures.mkdir(parents=True)
    frames = {
        "mouse_network_block_performance": _mouse_blocks(),
        "mouse_network_effects": _mouse_effects(),
        "network_statistics": _statistics(),
    }
    outputs = {}
    for key, frame in frames.items():
        path = results_root / PLOT_NETWORK.INPUT_KEYS[key]
        frame.write_csv(path)
        outputs[key] = {
            "path": str(path.relative_to(results_root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    analysis_manifest = manifests / "network_analysis_run.json"
    analysis_manifest.write_text(
        json.dumps(
            {
                "analysis_id": "simultaneous_network_interaction",
                "analysis_tier": "discovery",
                "run_status": "complete",
                "authoritative": True,
                "confirmation_accessed": False,
                "nominated_pair": {"source_region": "VISp", "target_region": "MOs"},
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    figure_manifest = figures / "figure_5_network_interaction_manifest.json"

    PLOT_NETWORK._run_figure(
        argparse.Namespace(dpi=120),
        started_at=datetime.datetime.now(datetime.UTC),
        results_root=results_root,
        figure_root=figures,
        figure_manifest_path=figure_manifest,
    )

    for suffix in ("svg", "pdf", "png"):
        assert (figures / f"figure_5_network_interaction.{suffix}").stat().st_size > 1_000
    source_root = figures / "figure_5_network_interaction_source_data"
    assert (source_root / "manifest.json").is_file()
    assert len(list(source_root.glob("*.csv"))) == 4
    assert json.loads(figure_manifest.read_text())["run_status"] == "complete"


@pytest.mark.parametrize(("value", "expected"), ((0.02, "0.020"), (0.0009, "<0.001")))
def test_adjusted_p_format(value: float, expected: str) -> None:
    assert PLOT_NETWORK._format_adjusted_p(value) == expected
