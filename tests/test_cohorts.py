"""Arithmetic contract tests for metadata-only cohort allocation.

All small tables below are explicitly labeled fixtures.  They are not
synthetic neuroscience observations and are never used as scientific results.
"""

import itertools

import polars as pl
import pytest

import dg.cohorts


def _allocation_contract_fixture() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return a small, labeled arithmetic fixture with six eligible mice."""

    inventory_rows = []
    qc_rows = []
    mouse_specs = (
        ("fixture_mouse_01", "F", "g1", "project_a", 2, (1,)),
        ("fixture_mouse_02", "M", "g1", "project_b", 2, (1, 2)),
        ("fixture_mouse_03", "F", "g2", "project_a", 1, (1,)),
        ("fixture_mouse_04", "M", "g2", "project_b", 2, (2,)),
        ("fixture_mouse_05", "F", "g3", "project_a", 1, (1,)),
        ("fixture_mouse_06", "M", "g3", "project_b", 2, (1, 2)),
        ("fixture_mouse_excluded", "M", "g2", "project_a", 2, ()),
    )
    session_counter = 0
    for subject_id, sex, genotype, project, n_sessions, selected_days in mouse_specs:
        for session_number in range(1, n_sessions + 1):
            session_counter += 1
            session_path = f"fixture_session_{session_counter:02d}.nwb"
            inventory_rows.append(
                {
                    "_nwb_path": session_path,
                    "subject_id": subject_id,
                    "sex": sex,
                    "genotype": genotype,
                    "project_code": project,
                    "recording_day": f"EPHYS_{session_number}",
                    "session_number": session_number,
                    "asset_id": f"fixture_asset_{session_counter:02d}",
                    "path": f"sub-{subject_id}/{session_path}",
                    "ecephys_session_id": session_counter,
                }
            )
            selected = session_number in selected_days
            qc_rows.append(
                {
                    "_nwb_path": session_path,
                    "subject_id": subject_id,
                    "is_good_session": selected,
                    "session_exclusion_reasons": None if selected else "fixture_qc_failure",
                }
            )
    return pl.DataFrame(inventory_rows), pl.DataFrame(qc_rows)


def _major_division_contract_fixture(
    qc: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return labeled presence-only fixtures for the 12-division contract."""

    divisions = pl.DataFrame(
        {
            "major_division_id": list(range(1, 13)),
            "major_division_acronym": [f"fixture_division_{index:02d}" for index in range(1, 13)],
            "major_division_name": [f"Fixture division {index:02d}" for index in range(1, 13)],
        }
    )
    selected = qc.filter(pl.col("is_good_session")).sort("subject_id", "_nwb_path")
    eligible = sorted(selected.get_column("subject_id").unique().to_list())
    first_selected = {
        mouse: rows.get_column("_nwb_path").item(0)
        for mouse, rows in selected.partition_by("subject_id", as_dict=True).items()
    }
    first_selected = {
        (mouse[0] if isinstance(mouse, tuple) else mouse): path
        for mouse, path in first_selected.items()
    }
    rows = []
    for row in selected.iter_rows(named=True):
        mouse_index = eligible.index(row["subject_id"])
        # Division 1 is present in five mice and varies because selected-session
        # denominators differ. Division 2 is too sparse and all others absent.
        identifiers = []
        if mouse_index < 5 and row["_nwb_path"] == first_selected[row["subject_id"]]:
            identifiers.append(1)
        if mouse_index < 2:
            identifiers.append(2)
        for identifier in identifiers:
            division = divisions.filter(pl.col("major_division_id") == identifier).row(
                0, named=True
            )
            rows.append(
                {
                    "_nwb_path": row["_nwb_path"],
                    "subject_id": row["subject_id"],
                    **division,
                    "has_any_raw_located_unit_in_division": True,
                    "unit_population": "all_units_presence_only",
                    "count_policy": "unit_count_magnitude_not_used",
                }
            )
    return pl.DataFrame(rows), divisions


def test_allocation_is_exhaustive_mouse_grouped_and_retains_sessions() -> None:
    inventory, qc = _allocation_contract_fixture()

    result = dg.cohorts.allocate_mouse_cohorts(
        inventory,
        qc,
        discovery_mice=4,
        confirmation_mice=2,
        seed=1051,
    )

    counts = dict(result.mice.group_by("cohort_assignment").len().iter_rows())
    assert counts == {"discovery": 4, "confirmation": 2, "excluded": 1}
    assert result.sessions.height == inventory.height
    assert result.sessions.get_column("_nwb_path").n_unique() == inventory.height
    assert (
        result.sessions.group_by("subject_id")
        .agg(pl.col("cohort_assignment").n_unique().alias("n"))
        .get_column("n")
        .to_list()
        == [1] * 7
    )
    summary = result.summary.row(0, named=True)
    assert summary["n_candidates_evaluated"] == 15
    assert summary["search_method"] == "exhaustive"
    assert summary["neural_activity_used_for_allocation"] is False
    assert summary["ready_for_neural_discovery"] is False
    assert summary["allocation_status"] == ("provisional_pending_coarse_regional_coverage")
    assert "analysis_included" not in result.sessions.columns
    assert not result.sessions.get_column("discovery_analysis_included").any()
    assert not result.sessions.get_column("confirmation_analysis_included").any()

    excluded = result.sessions.filter(pl.col("cohort_assignment") == "excluded")
    assert (
        excluded.get_column("session_split_exclusion_reason")
        .str.starts_with("mouse_has_no_threshold_selected_sessions")
        .all()
    )
    eligible_but_failed = result.sessions.filter(
        pl.col("mouse_behavior_eligible") & ~pl.col("session_behavior_eligible")
    )
    assert (
        eligible_but_failed.get_column("session_split_exclusion_reason")
        .str.starts_with("session_failed_behavior_qc")
        .all()
    )


def test_selected_split_is_the_exact_global_minimum() -> None:
    inventory, qc = _allocation_contract_fixture()
    result = dg.cohorts.allocate_mouse_cohorts(
        inventory,
        qc,
        discovery_mice=4,
        confirmation_mice=2,
        seed=1051,
    )
    eligible = result.mice.filter(pl.col("mouse_behavior_eligible")).sort("subject_id")
    features = dg.cohorts._build_features(eligible)
    mouse_ids = tuple(eligible.get_column("subject_id").to_list())
    candidate_scores = []
    for confirmation in itertools.combinations(mouse_ids, 2):
        confirmation_set = frozenset(confirmation)
        discovery = tuple(mouse for mouse in mouse_ids if mouse not in confirmation_set)
        candidate_scores.append(
            dg.cohorts._score_allocation(features, discovery, confirmation).objective
        )

    exact_selected = result.summary.item(0, "mean_family_mean_squared_smd_exact")
    assert exact_selected == str(min(candidate_scores))
    weighted_sum = result.balance.get_column("weighted_objective_contribution").sum()
    assert weighted_sum == pytest.approx(float(min(candidate_scores)))


def test_allocation_is_stable_to_input_order_and_ignores_outcome_columns() -> None:
    inventory, qc = _allocation_contract_fixture()
    qc_with_unused_outcome = qc.with_columns(
        pl.int_range(pl.len()).cast(pl.Float64).alias("unused_behavior_or_neural_outcome")
    )
    expected = dg.cohorts.allocate_mouse_cohorts(
        inventory,
        qc,
        discovery_mice=4,
        confirmation_mice=2,
        seed=1051,
    )
    observed = dg.cohorts.allocate_mouse_cohorts(
        inventory.reverse().with_columns(pl.lit(999.0).alias("unused_unit_metric")),
        qc_with_unused_outcome.reverse(),
        discovery_mice=4,
        confirmation_mice=2,
        seed=1051,
    )

    assert observed.mice.equals(expected.mice)
    assert observed.sessions.equals(expected.sessions)
    assert observed.balance.equals(expected.balance)
    assert observed.summary.equals(expected.summary)


def test_seed_only_resolves_exact_balance_ties() -> None:
    inventory, qc = _allocation_contract_fixture()
    seed_one = dg.cohorts.allocate_mouse_cohorts(
        inventory,
        qc,
        discovery_mice=4,
        confirmation_mice=2,
        seed=1,
    )
    seed_two = dg.cohorts.allocate_mouse_cohorts(
        inventory,
        qc,
        discovery_mice=4,
        confirmation_mice=2,
        seed=2,
    )

    assert seed_one.summary.item(0, "n_primary_objective_ties") > 1
    assert seed_one.summary.item(0, "mean_family_mean_squared_smd_exact") == seed_two.summary.item(
        0, "mean_family_mean_squared_smd_exact"
    )
    confirmation_one = seed_one.mice.filter(
        pl.col("cohort_assignment") == "confirmation"
    ).get_column("subject_id")
    confirmation_two = seed_two.mice.filter(
        pl.col("cohort_assignment") == "confirmation"
    ).get_column("subject_id")
    assert confirmation_one.to_list() != confirmation_two.to_list()


def test_allocation_rejects_session_coverage_or_metadata_conflicts() -> None:
    inventory, qc = _allocation_contract_fixture()
    with pytest.raises(ValueError, match="coverage differs"):
        dg.cohorts.allocate_mouse_cohorts(
            inventory,
            qc.head(qc.height - 1),
            discovery_mice=4,
            confirmation_mice=2,
        )

    conflicting = inventory.with_columns(
        pl.when(pl.col("_nwb_path") == "fixture_session_02.nwb")
        .then(pl.lit("different_genotype"))
        .otherwise(pl.col("genotype"))
        .alias("genotype")
    )
    with pytest.raises(ValueError, match="inconsistent within mouse"):
        dg.cohorts.allocate_mouse_cohorts(
            conflicting,
            qc,
            discovery_mice=4,
            confirmation_mice=2,
        )


@pytest.mark.parametrize("table_name", ["inventory", "qc"])
@pytest.mark.parametrize("invalid_subject", [None, "", "   "])
def test_allocation_rejects_null_or_empty_subject_ids(
    table_name: str,
    invalid_subject: str | None,
) -> None:
    inventory, qc = _allocation_contract_fixture()
    target = inventory if table_name == "inventory" else qc
    invalid = target.with_columns(
        pl.when(pl.col("_nwb_path") == "fixture_session_01.nwb")
        .then(pl.lit(invalid_subject, dtype=pl.String))
        .otherwise(pl.col("subject_id"))
        .alias("subject_id")
    )
    if table_name == "inventory":
        inventory = invalid
    else:
        qc = invalid
    with pytest.raises(ValueError, match="subject_id must be"):
        dg.cohorts.allocate_mouse_cohorts(
            inventory,
            qc,
            discovery_mice=4,
            confirmation_mice=2,
        )


def test_output_validation_detects_a_lost_session_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, qc = _allocation_contract_fixture()
    original = dg.cohorts._build_session_assignments

    def drop_last_session(
        inventory_frame: pl.DataFrame,
        qc_frame: pl.DataFrame,
        mice_frame: pl.DataFrame,
        *,
        allocation_status: str,
    ) -> pl.DataFrame:
        return original(
            inventory_frame,
            qc_frame,
            mice_frame,
            allocation_status=allocation_status,
        ).head(-1)

    monkeypatch.setattr(dg.cohorts, "_build_session_assignments", drop_last_session)
    with pytest.raises(RuntimeError, match="row count differs"):
        dg.cohorts.allocate_mouse_cohorts(
            inventory,
            qc,
            discovery_mice=4,
            confirmation_mice=2,
        )


def test_output_validation_detects_a_replaced_session_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory, qc = _allocation_contract_fixture()
    original = dg.cohorts._build_session_assignments

    def replace_last_session_key(
        inventory_frame: pl.DataFrame,
        qc_frame: pl.DataFrame,
        mice_frame: pl.DataFrame,
        *,
        allocation_status: str,
    ) -> pl.DataFrame:
        result = original(
            inventory_frame,
            qc_frame,
            mice_frame,
            allocation_status=allocation_status,
        )
        last_key = result.get_column("_nwb_path").tail(1).item()
        return result.with_columns(
            pl.when(pl.col("_nwb_path") == last_key)
            .then(pl.lit("unexpected_fixture_session.nwb"))
            .otherwise(pl.col("_nwb_path"))
            .alias("_nwb_path")
        )

    monkeypatch.setattr(
        dg.cohorts,
        "_build_session_assignments",
        replace_last_session_key,
    )
    with pytest.raises(RuntimeError, match="keys differ from inventory"):
        dg.cohorts.allocate_mouse_cohorts(
            inventory,
            qc,
            discovery_mice=4,
            confirmation_mice=2,
        )


def test_discovery_source_selector_fails_closed_and_blocks_holdout_leakage() -> None:
    inventory, qc = _allocation_contract_fixture()
    result = dg.cohorts.allocate_mouse_cohorts(
        inventory,
        qc,
        discovery_mice=4,
        confirmation_mice=2,
    )
    expected_sources = inventory.get_column("_nwb_path").to_list()
    with pytest.raises(RuntimeError, match="locked until one uniform final allocation"):
        dg.cohorts._select_discovery_session_sources_from_validated_table(
            result.sessions,
            expected_session_sources=expected_sources,
        )

    finalized = result.sessions.with_columns(
        pl.lit(dg.cohorts.FINAL_ALLOCATION_STATUS).alias("allocation_status"),
        pl.lit(True).alias("ready_for_neural_discovery"),
        (pl.col("cohort_assignment") == "discovery").alias("discovery_mouse_included"),
        ((pl.col("cohort_assignment") == "discovery") & pl.col("session_behavior_eligible")).alias(
            "discovery_analysis_included"
        ),
    )
    expected = tuple(
        finalized.filter(pl.col("discovery_analysis_included"))
        .get_column("_nwb_path")
        .sort()
        .to_list()
    )
    assert (
        dg.cohorts._select_discovery_session_sources_from_validated_table(
            finalized,
            expected_session_sources=expected_sources,
        )
        == expected
    )

    leaked = finalized.with_columns(
        (pl.col("cohort_assignment") == "confirmation").alias("confirmation_analysis_included")
    )
    with pytest.raises(RuntimeError, match="refuses an activated confirmation holdout"):
        dg.cohorts._select_discovery_session_sources_from_validated_table(
            leaked,
            expected_session_sources=expected_sources,
        )

    with pytest.raises(RuntimeError, match="exactly cover expected"):
        dg.cohorts._select_discovery_session_sources_from_validated_table(
            finalized.head(-1),
            expected_session_sources=expected_sources,
        )

    original_cohort = finalized.filter(pl.col("subject_id") == "fixture_mouse_02").item(
        0, "cohort_assignment"
    )
    replacement = "confirmation" if original_cohort == "discovery" else "discovery"
    split_mouse = finalized.with_columns(
        pl.when(pl.col("_nwb_path") == "fixture_session_03.nwb")
        .then(pl.lit(replacement))
        .otherwise(pl.col("cohort_assignment"))
        .alias("cohort_assignment")
    ).with_columns(
        (pl.col("cohort_assignment") == "discovery").alias("discovery_mouse_included"),
        ((pl.col("cohort_assignment") == "discovery") & pl.col("session_behavior_eligible")).alias(
            "discovery_analysis_included"
        ),
    )
    with pytest.raises(RuntimeError, match="mouse split across cohorts"):
        dg.cohorts._select_discovery_session_sources_from_validated_table(
            split_mouse,
            expected_session_sources=expected_sources,
        )


def test_allocation_rejects_wrong_eligible_mouse_count_and_day_conflict() -> None:
    inventory, qc = _allocation_contract_fixture()
    with pytest.raises(ValueError, match="eligible mouse count"):
        dg.cohorts.allocate_mouse_cohorts(
            inventory,
            qc,
            discovery_mice=3,
            confirmation_mice=2,
        )

    conflicting_day = inventory.with_columns(
        pl.when(pl.col("_nwb_path") == "fixture_session_01.nwb")
        .then(pl.lit("EPHYS_4"))
        .otherwise(pl.col("recording_day"))
        .alias("recording_day")
    )
    with pytest.raises(ValueError, match="disagree"):
        dg.cohorts.allocate_mouse_cohorts(
            conflicting_day,
            qc,
            discovery_mice=4,
            confirmation_mice=2,
        )


def test_public_dataframe_only_discovery_selector_always_fails_closed() -> None:
    inventory, qc = _allocation_contract_fixture()
    result = dg.cohorts.allocate_mouse_cohorts(
        inventory,
        qc,
        discovery_mice=4,
        confirmation_mice=2,
    )

    with pytest.raises(RuntimeError, match="DataFrame-only discovery selection is unauthenticated"):
        dg.cohorts.select_discovery_session_sources(
            result.sessions,
            expected_session_sources=inventory.get_column("_nwb_path").to_list(),
        )


def test_allocation_requires_boolean_threshold_selection() -> None:
    inventory, qc = _allocation_contract_fixture()
    invalid_qc = qc.with_columns(pl.col("is_good_session").cast(pl.Int8))
    with pytest.raises(TypeError, match="must be Boolean"):
        dg.cohorts.allocate_mouse_cohorts(
            inventory,
            invalid_qc,
            discovery_mice=4,
            confirmation_mice=2,
        )


def test_major_division_coverage_uses_exact_selected_session_fractions() -> None:
    _, qc = _allocation_contract_fixture()
    presence, divisions = _major_division_contract_fixture(qc)

    coverage = dg.cohorts.summarize_mouse_major_division_coverage(
        presence,
        qc,
        divisions,
    )

    assert coverage.height == 6 * 12
    division_one = coverage.filter(pl.col("major_division_id") == 1)
    assert division_one.get_column("n_eligible_mice_represented").unique().to_list() == [5]
    assert division_one.get_column("include_in_balance").all()
    assert set(division_one.get_column("threshold_selected_session_fraction_exact")) <= {
        "0",
        "1/2",
        "1",
    }
    division_two = coverage.filter(pl.col("major_division_id") == 2)
    assert not division_two.get_column("include_in_balance").any()
    assert division_two.get_column("balance_exclusion_reason").unique().to_list() == [
        "represented_in_fewer_than_minimum_eligible_mice"
    ]
    assert not coverage.get_column("neural_activity_outcomes_used").any()


def test_complete_major_division_grid_finalizes_discovery_but_not_confirmation() -> None:
    inventory, qc = _allocation_contract_fixture()
    presence, divisions = _major_division_contract_fixture(qc)
    coverage = dg.cohorts.summarize_mouse_major_division_coverage(
        presence,
        qc,
        divisions,
    )

    result = dg.cohorts.allocate_mouse_cohorts(
        inventory,
        qc,
        mouse_major_division_coverage=coverage,
        discovery_mice=4,
        confirmation_mice=2,
        seed=1051,
    )

    assert result.summary.item(0, "allocation_status") == dg.cohorts.FINAL_ALLOCATION_STATUS
    assert result.summary.item(0, "ready_for_neural_discovery") is True
    regional = result.balance.filter(pl.col("metadata_family") == "coarse_regional_coverage")
    assert regional.height == 1
    assert regional.item(0, "feature_type") == "selected_session_presence_fraction"
    expected_discovery = (pl.col("cohort_assignment") == "discovery") & pl.col(
        "session_behavior_eligible"
    )
    assert result.sessions.filter(
        pl.col("discovery_analysis_included") != expected_discovery
    ).is_empty()
    assert not result.sessions.get_column("confirmation_analysis_included").any()
    assert not result.mice.get_column("confirmation_mouse_included").any()
    expected_sources = tuple(
        result.sessions.filter(pl.col("discovery_analysis_included"))
        .get_column("_nwb_path")
        .sort()
        .to_list()
    )
    assert (
        dg.cohorts._select_discovery_session_sources_from_validated_table(
            result.sessions,
            expected_session_sources=inventory.get_column("_nwb_path").to_list(),
        )
        == expected_sources
    )


def test_final_allocation_recomputes_and_rejects_regional_diagnostics() -> None:
    inventory, qc = _allocation_contract_fixture()
    presence, divisions = _major_division_contract_fixture(qc)
    coverage = dg.cohorts.summarize_mouse_major_division_coverage(
        presence,
        qc,
        divisions,
    )
    tampered = coverage.with_columns(
        pl.when(pl.col("major_division_id") == 1)
        .then(pl.lit("999/1"))
        .otherwise(pl.col("eligible_mouse_fraction_variance_exact"))
        .alias("eligible_mouse_fraction_variance_exact")
    )

    with pytest.raises(ValueError, match="exact variance diagnostic"):
        dg.cohorts.allocate_mouse_cohorts(
            inventory,
            qc,
            mouse_major_division_coverage=tampered,
            discovery_mice=4,
            confirmation_mice=2,
        )
