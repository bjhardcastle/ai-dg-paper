import polars as pl
import pytest

import dg.perturbations


def _trials() -> pl.DataFrame:
    rows = []
    images = [
        ("im115_r-1.0", "im999_r-1.0"),
        ("im115_r-0.7", "im999_r-1.0"),
        ("im999_r-1.0", "im999_r-1.0"),
        ("im047_r-1.0", "im999_r-1.0"),
    ]
    for index, (image, novel) in enumerate(images):
        rows.append(
            {
                "_nwb_path": "s1",
                "subject_id": "m1",
                "ecephys_session_id": 1,
                "_table_index": index,
                "change_time": 10.0 + index,
                "change_image_name": image,
                "novel_image_id": novel,
                "physical_image_change": True,
                "reward_block": "engaged_1",
                "aborted": False,
                "auto_rewarded": False,
                "lick_times": [20.0],
                "lick_times_valid": True,
                "response_in_window": index % 2 == 0,
            }
        )
    return pl.DataFrame(rows)


def test_select_and_expand_probe_conditions() -> None:
    selected = dg.perturbations.select_perturbation_trials(_trials())
    assert selected.height == 4
    assert selected.get_column("change_relative_contrast").to_list() == [1.0, 0.7, 1.0, 1.0]
    expanded = dg.perturbations.expand_probe_conditions(selected)
    assert expanded.filter(pl.col("perturbation_family") == "contrast").height == 2
    novelty = expanded.filter(pl.col("perturbation_family") == "novelty")
    assert set(novelty.get_column("condition")) == {"designated_novel", "familiar"}
    assert novelty.filter(pl.col("condition") == "familiar").height == 2


def test_peristimulus_lick_boundaries_are_excluded() -> None:
    trials = _trials().with_columns(
        pl.when(pl.col("_table_index") == 0)
        .then(pl.lit([9.85]))
        .otherwise(pl.col("lick_times"))
        .alias("lick_times")
    )
    selected = dg.perturbations.select_perturbation_trials(trials)
    assert selected.get_column("_table_index").to_list() == [1, 2, 3]


def test_session_and_mouse_reversible_summaries() -> None:
    rows = []
    for mouse, session in (("m1", 1), ("m1", 2), ("m2", 3)):
        for block, value in (("engaged_1", 0.8), ("no_reward", 0.2), ("engaged_2", 0.6)):
            for condition, offset in (("full", 0.0), ("reduced", -0.1)):
                for trial in range(5):
                    rows.append(
                        {
                            "_nwb_path": f"s{session}",
                            "subject_id": mouse,
                            "ecephys_session_id": session,
                            "reward_block": block,
                            "perturbation_family": "contrast",
                            "condition": condition,
                            "interpretation_scope": "within_identity_im115_contrast",
                            "response_in_window": value + offset > 0.5,
                            "axis_score": value + offset,
                            "trial": trial,
                        }
                    )
    blocks = dg.perturbations.summarize_session_blocks(pl.DataFrame(rows))
    sessions = dg.perturbations.compute_session_reversible_contrasts(blocks)
    assert sessions.filter(pl.col("status") == "pass").height == 12
    neural = sessions.filter(
        (pl.col("metric") == "early_reward_axis") & (pl.col("condition") == "full")
    )
    assert neural.get_column("reversible_contrast").to_list() == pytest.approx([0.5, 0.5, 0.5])
    mice = dg.perturbations.aggregate_mouse_contrasts(sessions)
    interactions = dg.perturbations.compute_mouse_perturbation_interactions(mice)
    assert interactions.height == 4
    assert interactions.filter(pl.col("metric") == "early_reward_axis").get_column(
        "interaction"
    ).to_list() == pytest.approx([0.0, 0.0])


def test_config_rejects_invalid_support() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        dg.perturbations.PerturbationConfig(minimum_trials_per_session_block_condition=0)
