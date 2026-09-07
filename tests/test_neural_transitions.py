"""Algorithmic tests for the neural transition analysis core.

The arrays below are arithmetic fixtures for checking implementation
contracts. They are not synthetic observations or substitutes for DANDI data.
"""

import numpy as np
import polars as pl
import pytest

import dg.neural_transitions


def test_default_transition_windows_match_figure_two_contract() -> None:
    config = dg.neural_transitions.DEFAULT_NEURAL_TRANSITION_CONFIG

    assert config.effect_window_seconds == 360.0
    assert config.minimum_trials_per_effect_side == 5
    assert config.trajectory_window_seconds == 360.0
    assert config.trajectory_bin_width_seconds == 120.0


def test_trial_selection_requires_familiar_full_contrast_and_lick_free_window() -> None:
    trials = pl.DataFrame(
        {
            "_table_index": list(range(8)),
            "change_time": [10.0] * 8,
            "change_image_name": [
                "im001_r-1.0",
                "im002_r-1.0",
                "im001_r-0.5",
                "im001_r-1.0",
                "im001_r-1.0",
                "im001_r-1.0",
                "im001_r-1.0",
                "im001_r-1.0",
            ],
            "novel_image_id": ["im002_r-1.0"] * 8,
            "physical_image_change": [True, True, True, True, True, True, False, True],
            "reward_block": ["engaged_1"] * 7 + ["unclassified"],
            "aborted": [False, False, False, False, True, False, False, False],
            "auto_rewarded": [False, False, False, True, False, False, False, False],
            "lick_times": [
                [9.0, 11.0],
                [],
                [],
                [],
                [],
                [10.15],
                [],
                [],
            ],
            "lick_times_valid": [True] * 8,
        }
    )

    selected = dg.neural_transitions.select_lick_free_familiar_trials(trials)

    assert selected.get_column("_table_index").to_list() == [0]
    assert selected.get_column("lick_free_for_neural_analysis").to_list() == [True]
    assert selected.get_column("lick_exclusion_half_width_seconds").item() == pytest.approx(0.15)


def test_spike_features_use_causal_half_open_windows() -> None:
    event_times = np.asarray([1.0])
    spike_times = [
        np.asarray([0.85, 0.999, 1.0, 1.149, 1.15]),
        np.asarray([1.0, 1.15]),
    ]

    features = dg.neural_transitions.build_spike_feature_matrix(
        spike_times,
        event_times,
        window_seconds=0.15,
    )

    assert features.shape == (1, 2)
    assert features[0, 0] == pytest.approx(0.0)
    assert features[0, 1] == pytest.approx(np.sqrt(1 + 3 / 8) - np.sqrt(3 / 8))


def _axis_fixture(e2_offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e1_time = np.arange(12, dtype=float)
    nr_time = np.arange(20, 32, dtype=float)
    e2_time = np.arange(40, 48, dtype=float)
    times = np.concatenate([e1_time, nr_time, e2_time])
    blocks = np.asarray(
        ["engaged_1"] * e1_time.size + ["no_reward"] * nr_time.size + ["engaged_2"] * e2_time.size,
        dtype=object,
    )
    temporal_nuisance = np.linspace(-0.2, 0.2, times.size)
    features = np.column_stack(
        [
            np.concatenate(
                [
                    np.linspace(0.8, 1.2, e1_time.size),
                    np.linspace(-1.2, -0.8, nr_time.size),
                    np.linspace(e2_offset, e2_offset + 0.2, e2_time.size),
                ]
            ),
            temporal_nuisance,
        ]
    )
    return features, blocks, times


def test_ridge_axis_orients_e1_above_nr_and_e2_cannot_leak_into_fit() -> None:
    first_features, blocks, times = _axis_fixture(e2_offset=0.3)
    second_features, _, _ = _axis_fixture(e2_offset=1_000.0)

    first = dg.neural_transitions.fit_reward_state_axis(first_features, blocks, times)
    second = dg.neural_transitions.fit_reward_state_axis(second_features, blocks, times)

    fit_rows = blocks != "engaged_2"
    assert np.array_equal(first.scores[fit_rows], second.scores[fit_rows])
    assert np.array_equal(first.feature_weights, second.feature_weights)
    assert first.intercept == second.intercept
    assert first.selected_lambda == second.selected_lambda
    assert first.outer_selected_lambdas == second.outer_selected_lambdas
    assert first.scores[blocks == "engaged_1"].mean() > first.scores[blocks == "no_reward"].mean()
    assert first.oof_auc == pytest.approx(1.0)
    assert set(first.score_roles[blocks.size - 8 :]) == {"held_out_e2_transfer"}
    assert not np.array_equal(
        first.scores[blocks == "engaged_2"], second.scores[blocks == "engaged_2"]
    )
    final_projection = (
        first_features[blocks == "engaged_2"] @ first.feature_weights + first.intercept
    )
    assert np.allclose(first.scores[blocks == "engaged_2"], final_projection)


def _session_score_rows(
    source: str,
    mouse: str,
    *,
    withdrawal_post_score: float = -0.5,
    omit_last_withdrawal_post: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(times, block, score):
        rows.extend(
            {
                "_nwb_path": source,
                "subject_id": mouse,
                "change_time": float(time),
                "reward_block": block,
                "axis_score": float(score),
            }
            for time in times
        )

    add([30, 34, 38, 42, 46], "engaged_1", 0.5)
    add([50, 54, 58, 62, 66], "engaged_1", 0.5)
    add([80, 84, 88, 92, 96], "engaged_1", 0.5)
    withdrawal_post = [100, 104, 108, 112, 116]
    if omit_last_withdrawal_post:
        withdrawal_post = withdrawal_post[:-1]
    add(withdrawal_post, "no_reward", withdrawal_post_score)
    # These large values are on the correct side in time but in the wrong state.
    add([96], "no_reward", -100.0)
    add([104], "engaged_1", 100.0)

    add([130, 134, 138, 142, 146], "no_reward", -0.5)
    add([150, 154, 158, 162, 166], "no_reward", -0.5)
    add([180, 184, 188, 192, 196], "no_reward", -0.5)
    add([200, 204, 208, 212, 216], "engaged_2", 0.4)
    return rows


def _boundary_rows(source: str, mouse: str) -> list[dict[str, object]]:
    common = {
        "_nwb_path": source,
        "subject_id": mouse,
        "included_in_transition_analysis": True,
    }
    return [
        {
            **common,
            "transition_id": "reward_withdrawal",
            "pre_block": "engaged_1",
            "post_block": "no_reward",
            "pseudo_block": "engaged_1",
            "effect_direction_multiplier": -1.0,
            "real_boundary_time": 100.0,
            "pseudo_boundary_time": 50.0,
        },
        {
            **common,
            "transition_id": "reward_restoration",
            "pre_block": "no_reward",
            "post_block": "engaged_2",
            "pseudo_block": "no_reward",
            "effect_direction_multiplier": 1.0,
            "real_boundary_time": 200.0,
            "pseudo_boundary_time": 150.0,
        },
    ]


def test_transition_effects_respect_state_causality_and_support_thresholds() -> None:
    scores = pl.DataFrame(
        _session_score_rows("session_a", "mouse_a")
        + _session_score_rows(
            "session_b",
            "mouse_b",
            omit_last_withdrawal_post=True,
        )
    )
    boundaries = pl.DataFrame(
        _boundary_rows("session_a", "mouse_a") + _boundary_rows("session_b", "mouse_b")
    )
    config = dg.neural_transitions.NeuralTransitionConfig(
        effect_window_seconds=30.0,
        trajectory_window_seconds=60.0,
        trajectory_bin_width_seconds=20.0,
        minimum_trials_per_effect_side=5,
    )

    tables = dg.neural_transitions.summarize_transition_scores(
        scores,
        boundaries,
        config=config,
    )

    effects = tables.session_anchor_effects
    withdrawal_real = effects.filter(
        (pl.col("_nwb_path") == "session_a")
        & (pl.col("transition_id") == "reward_withdrawal")
        & (pl.col("anchor_type") == "real")
    ).row(0, named=True)
    assert withdrawal_real["n_pre_trials"] == 5
    assert withdrawal_real["n_post_trials"] == 5
    assert withdrawal_real["signed_step"] == pytest.approx(1.0)

    withdrawal_pseudo = effects.filter(
        (pl.col("_nwb_path") == "session_a")
        & (pl.col("transition_id") == "reward_withdrawal")
        & (pl.col("anchor_type") == "pseudo")
    ).row(0, named=True)
    assert withdrawal_pseudo["signed_step"] == pytest.approx(0.0)

    restoration = tables.session_contrasts.filter(
        (pl.col("_nwb_path") == "session_a") & (pl.col("transition_id") == "reward_restoration")
    ).row(0, named=True)
    assert restoration["real_minus_pseudo"] == pytest.approx(0.9)

    unsupported = tables.session_contrasts.filter(
        (pl.col("_nwb_path") == "session_b") & (pl.col("transition_id") == "reward_withdrawal")
    ).row(0, named=True)
    assert unsupported["status"] == "insufficient_anchor_support"
    assert unsupported["real_minus_pseudo"] is None

    first_post_bin = tables.session_trajectories.filter(
        (pl.col("_nwb_path") == "session_a")
        & (pl.col("transition_id") == "reward_withdrawal")
        & (pl.col("anchor_type") == "real")
        & (pl.col("bin_index") == 0)
    ).row(0, named=True)
    assert first_post_bin["n_trials"] == 5
    assert first_post_bin["mean_axis_score"] == pytest.approx(-0.5)


def test_mouse_aggregation_weights_sessions_equally() -> None:
    scores = pl.DataFrame(
        _session_score_rows("session_a", "mouse_a", withdrawal_post_score=-0.5)
        + _session_score_rows("session_b", "mouse_a", withdrawal_post_score=0.0)
    )
    boundaries = pl.DataFrame(
        _boundary_rows("session_a", "mouse_a") + _boundary_rows("session_b", "mouse_a")
    )
    config = dg.neural_transitions.NeuralTransitionConfig(
        effect_window_seconds=30.0,
        trajectory_window_seconds=60.0,
        trajectory_bin_width_seconds=20.0,
        minimum_trials_per_effect_side=5,
    )

    tables = dg.neural_transitions.summarize_transition_scores(
        scores,
        boundaries,
        config=config,
    )

    withdrawal = tables.mouse_contrasts.filter(pl.col("transition_id") == "reward_withdrawal").row(
        0, named=True
    )
    assert withdrawal["n_sessions"] == 2
    assert withdrawal["real_minus_pseudo"] == pytest.approx(0.75)


def test_mouse_inference_applies_holm_correction_across_transitions() -> None:
    mouse_contrasts = pl.DataFrame(
        {
            "subject_id": ["m1", "m2", "m3", "m1", "m2", "m3"],
            "transition_id": ["reward_withdrawal"] * 3 + ["reward_restoration"] * 3,
            "real_minus_pseudo": [1.0, 1.1, 0.9, 0.3, -0.1, 0.2],
        }
    )

    result = dg.neural_transitions.infer_mouse_transition_contrasts(
        mouse_contrasts,
        seed=1051,
        n_bootstrap=200,
        n_sign_flips=200,
    )

    assert result.height == 2
    assert result.get_column("n_mice").to_list() == [3, 3]
    assert (result.get_column("adjusted_p_value") >= result.get_column("p_value")).all()
