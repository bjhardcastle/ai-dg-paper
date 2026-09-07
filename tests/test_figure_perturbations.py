import importlib.util
import pathlib
import sys

import polars as pl
import pytest

import dg.figure_perturbations

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "15_plot_perturbations.py"
SPEC = importlib.util.spec_from_file_location("dg_plot_perturbations_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
PLOT_PERTURBATIONS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT_PERTURBATIONS
SPEC.loader.exec_module(PLOT_PERTURBATIONS)


def _session_blocks() -> pl.DataFrame:
    rows = []
    conditions = {
        "contrast": ("full", "reduced"),
        "novelty": ("familiar", "designated_novel"),
    }
    for mouse_index, mouse in enumerate(("m1", "m2", "m3")):
        for family, family_conditions in conditions.items():
            for condition_index, condition in enumerate(family_conditions):
                for block_index, block in enumerate(("engaged_1", "no_reward", "engaged_2")):
                    rows.append(
                        {
                            "subject_id": mouse,
                            "ecephys_session_id": mouse_index + 1,
                            "perturbation_family": family,
                            "condition": condition,
                            "reward_block": block,
                            "n_trials": 10,
                            "response_probability": 0.7
                            - 0.3 * (block_index == 1)
                            + 0.02 * condition_index,
                            "mean_axis_score": 0.2
                            - 0.25 * (block_index == 1)
                            + 0.01 * condition_index,
                        }
                    )
    return pl.DataFrame(rows)


def _mouse_contrasts() -> pl.DataFrame:
    rows = []
    for mouse in ("m1", "m2", "m3"):
        for family, conditions in {
            "contrast": ("full", "reduced"),
            "novelty": ("familiar", "designated_novel"),
        }.items():
            for condition_index, condition in enumerate(conditions):
                for metric in ("behavior_response", "early_reward_axis"):
                    rows.append(
                        {
                            "subject_id": mouse,
                            "perturbation_family": family,
                            "condition": condition,
                            "metric": metric,
                            "reversible_contrast": 0.3 + 0.02 * condition_index,
                            "n_sessions": 1,
                            "n_trials": 30,
                        }
                    )
    return pl.DataFrame(rows)


def _interactions() -> pl.DataFrame:
    rows = []
    for mouse in ("m1", "m2", "m3"):
        for family in ("contrast", "novelty"):
            for metric in ("behavior_response", "early_reward_axis"):
                rows.append(
                    {
                        "subject_id": mouse,
                        "perturbation_family": family,
                        "metric": metric,
                        "interaction": 0.02,
                        "n_sessions": 1,
                        "n_trials": 60,
                    }
                )
    return pl.DataFrame(rows)


def _statistics() -> pl.DataFrame:
    rows = []
    for family in ("contrast", "novelty"):
        rows.append(
            {
                "analysis_id": "heldout_perturbation_axis",
                "result_id": (
                    f"behavior_response_{family}_state_by_perturbation_interaction_mouse_mean"
                ),
                "estimate": 0.02,
                "ci_low": -0.01,
                "ci_high": 0.05,
                "p_value": 0.5,
                "adjusted_p_value": 0.5,
                "n_mice": 3,
                "status": "pass",
            }
        )
    for family, condition in (
        ("contrast", "reduced"),
        ("novelty", "designated_novel"),
    ):
        rows.append(
            {
                "analysis_id": "heldout_perturbation_axis",
                "result_id": f"early_reward_axis_{family}_{condition}_reversible_mouse_mean",
                "estimate": 0.32,
                "ci_low": 0.29,
                "ci_high": 0.35,
                "p_value": 0.01,
                "adjusted_p_value": 0.02,
                "n_mice": 3,
                "status": "pass",
            }
        )
    return pl.DataFrame(rows)


def test_prepare_figure_data_uses_equal_mouse_summaries() -> None:
    prepared = dg.figure_perturbations.prepare_perturbation_figure_data(
        _session_blocks(),
        _mouse_contrasts(),
        _interactions(),
        _statistics(),
    )
    assert prepared.mouse_blocks.height == 72
    assert prepared.block_summary.height == 24
    assert prepared.panel_statistics.height == 4
    row = prepared.block_summary.filter(
        (pl.col("metric") == "behavior_response")
        & (pl.col("perturbation_family") == "contrast")
        & (pl.col("condition") == "full")
        & (pl.col("reward_block") == "engaged_1")
    ).row(0, named=True)
    assert row["estimate"] == pytest.approx(0.7)
    assert row["n_mice"] == 3


def test_prepare_rejects_statistic_disagreement() -> None:
    statistics = _statistics().with_columns(
        pl.when(pl.col("result_id").str.starts_with("behavior_response_contrast"))
        .then(0.5)
        .otherwise(pl.col("estimate"))
        .alias("estimate")
    )
    with pytest.raises(ValueError, match="disagrees"):
        dg.figure_perturbations.prepare_perturbation_figure_data(
            _session_blocks(),
            _mouse_contrasts(),
            _interactions(),
            statistics,
        )


def test_prepare_rejects_missing_condition() -> None:
    blocks = _session_blocks().filter(pl.col("condition") != "reduced")
    prepared = dg.figure_perturbations.prepare_perturbation_figure_data(
        blocks,
        _mouse_contrasts(),
        _interactions(),
        _statistics(),
    )
    assert set(prepared.block_summary.get_column("condition")) == {
        "full",
        "familiar",
        "designated_novel",
    }


def test_create_figure_has_four_panel_labels() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prepared = dg.figure_perturbations.prepare_perturbation_figure_data(
        _session_blocks(),
        _mouse_contrasts(),
        _interactions(),
        _statistics(),
    )
    figure = PLOT_PERTURBATIONS.create_figure(prepared)
    try:
        assert len(figure.axes) == 4
        labels = {text.get_text() for axis in figure.axes for text in axis.texts}
        assert {"A", "B", "C", "D"}.issubset(labels)
    finally:
        plt.close(figure)
