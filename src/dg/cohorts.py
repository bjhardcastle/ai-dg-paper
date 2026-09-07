"""Mouse-grouped discovery/confirmation allocation using locked covariates only.

The primary balance objective is the mean, across metadata families, of the
mean squared standardized difference between discovery and confirmation.
Each feature is standardized by its sample variance across all eligible mice.
Categorical metadata are represented by one indicator per observed level;
numeric summaries are represented directly.  Equal family weighting prevents
a categorical variable with more levels from receiving more influence merely
because it expands to more indicators.

The exhaustive search is evaluated with :class:`fractions.Fraction`.  Thus,
the selected allocation exactly minimizes the documented squared-SMD
objective rather than a rounded floating-point approximation.  Exact primary
ties are resolved by the smallest maximum squared SMD, then a SHA-256 rank
derived from the frozen seed and sorted confirmation IDs, then lexical order.
No neural or behavioral outcome magnitude is accepted by this module. The only
behavior-derived input is the D03 threshold-selection boolean. The optional
regional family is likewise outcome-blind: for each mouse and official major
division it uses only the fraction of selected sessions containing at least one
raw located unit. Unit quality, firing rate, activity, and count magnitude are
not inputs. Without this frozen regional table an allocation is provisional
and cannot activate neural discovery.
"""

from __future__ import annotations

import dataclasses
import fractions
import hashlib
import itertools
import json
import math
import os
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import polars as pl

SESSION_KEY = "_nwb_path"
MISSING_CATEGORY = "<missing>"
CATEGORICAL_METADATA = ("sex", "genotype", "project_code")
FAMILY_ORDER = (
    "sex",
    "genotype",
    "project_cohort",
    "recording_day",
    "session_count",
    "coarse_regional_coverage",
)
DEFAULT_DISCOVERY_MICE = 12
DEFAULT_CONFIRMATION_MICE = 6
DEFAULT_SPLIT_SEED = 1051
MINIMUM_REGION_REPRESENTED_MICE = 5
PROVISIONAL_ALLOCATION_STATUS = "provisional_pending_coarse_regional_coverage"
# Backwards-compatible name for readers of the initial provisional contract.
ALLOCATION_STATUS = PROVISIONAL_ALLOCATION_STATUS
FINAL_ALLOCATION_STATUS = "final_coarse_regional_coverage_balanced"
COHORT_MANIFEST_SCHEMA_VERSION = 3
COHORT_ANALYSIS_ID = "d05_mouse_grouped_cohort_allocation"
D05_REQUIRED_LOCAL_SOURCE_PATHS = (
    "src/dg/artifacts.py",
    "src/dg/cohorts.py",
    "src/dg/gates.py",
    "src/dg/ontology.py",
    "scripts/03_allocate_cohorts.py",
)
D05_REQUIRED_DEPENDENCY_PATHS = ("pyproject.toml", "uv.lock")

_REQUIRED_INVENTORY_COLUMNS = (
    SESSION_KEY,
    "subject_id",
    "sex",
    "genotype",
    "project_code",
    "recording_day",
    "session_number",
)
_OPTIONAL_SESSION_ID_COLUMNS = (
    "asset_id",
    "path",
    "ecephys_session_id",
    "behavior_session_id",
    "date_of_acquisition",
)
_REQUIRED_QC_COLUMNS = (
    SESSION_KEY,
    "subject_id",
    "is_good_session",
    "session_exclusion_reasons",
)


@dataclasses.dataclass(frozen=True)
class CohortAllocation:
    """Complete allocation artifacts and exact exhaustive-search diagnostics."""

    mice: pl.DataFrame
    sessions: pl.DataFrame
    balance: pl.DataFrame
    summary: pl.DataFrame


@dataclasses.dataclass(frozen=True)
class _Feature:
    family: str
    name: str
    feature_type: str
    values: Mapping[str, fractions.Fraction]
    eligible_mean: fractions.Fraction
    eligible_variance: fractions.Fraction


@dataclasses.dataclass(frozen=True)
class _Score:
    objective: fractions.Fraction
    maximum_squared_smd: fractions.Fraction
    squared_smds: Mapping[str, fractions.Fraction]
    family_objectives: Mapping[str, fractions.Fraction]


def summarize_mouse_major_division_coverage(
    session_division_presence: pl.DataFrame | pl.LazyFrame,
    session_qc: pl.DataFrame | pl.LazyFrame,
    major_divisions: pl.DataFrame | pl.LazyFrame,
    *,
    minimum_represented_mice: int = MINIMUM_REGION_REPRESENTED_MICE,
) -> pl.DataFrame:
    """Build exact D05 mouse-level regional covariates from session presence.

    For every behavior-eligible mouse and each of the 12 official major
    divisions, the covariate is ``selected sessions with presence / selected
    sessions``. Presence means at least one raw located unit of any quality;
    unit count magnitude is deliberately discarded upstream and rejected here.
    A division enters the balance objective only when it occurs in at least
    ``minimum_represented_mice`` eligible mice and its fractions vary.
    """

    if (
        isinstance(minimum_represented_mice, bool)
        or not isinstance(minimum_represented_mice, int)
        or minimum_represented_mice <= 0
    ):
        raise ValueError("minimum_represented_mice must be a positive integer")
    presence = _collect(session_division_presence)
    qc = _collect(session_qc)
    divisions = _collect(major_divisions)
    _require_columns(qc, _REQUIRED_QC_COLUMNS, table="session QC")
    _validate_unique_nonnull_key(qc, table="session QC")
    _validate_subject_ids(qc, table="session QC")
    if qc.get_column("is_good_session").dtype != pl.Boolean:
        raise TypeError("session QC is_good_session must be Boolean")
    if qc.get_column("is_good_session").null_count():
        raise ValueError("session QC is_good_session must not contain nulls")

    division_columns = (
        "major_division_id",
        "major_division_acronym",
        "major_division_name",
    )
    _require_columns(divisions, division_columns, table="major divisions")
    divisions = divisions.select(division_columns).sort("major_division_id")
    if divisions.height != 12:
        raise ValueError(f"major divisions must contain exactly 12 rows, got {divisions.height}")
    for column in division_columns:
        if divisions.get_column(column).null_count():
            raise ValueError(f"major divisions {column} contains nulls")
        if divisions.get_column(column).n_unique() != divisions.height:
            raise ValueError(f"major divisions {column} must be unique")
    if not divisions.get_column("major_division_id").dtype.is_integer():
        raise TypeError("major divisions major_division_id must be an integer")
    for column in ("major_division_acronym", "major_division_name"):
        if divisions.get_column(column).dtype != pl.String:
            raise TypeError(f"major divisions {column} must be String")
        if divisions.filter(pl.col(column).str.strip_chars() == "").height:
            raise ValueError(f"major divisions {column} contains empty values")

    presence_columns = (
        SESSION_KEY,
        "subject_id",
        *division_columns,
        "has_any_raw_located_unit_in_division",
        "unit_population",
        "count_policy",
    )
    _require_columns(presence, presence_columns, table="session division presence")
    presence = presence.select(presence_columns)
    if presence.is_empty():
        raise ValueError("session division presence is empty")
    _validate_subject_ids(presence, table="session division presence")
    if presence.get_column(SESSION_KEY).dtype != pl.String:
        raise TypeError(f"session division presence {SESSION_KEY} must be String")
    if (
        presence.get_column(SESSION_KEY).null_count()
        or presence.filter(pl.col(SESSION_KEY).str.strip_chars() == "").height
    ):
        raise ValueError(f"session division presence {SESSION_KEY} contains null/empty values")
    if presence.select(SESSION_KEY, "major_division_id").n_unique() != presence.height:
        raise ValueError("session division presence must be unique by session/division")
    boolean = presence.get_column("has_any_raw_located_unit_in_division")
    if boolean.dtype != pl.Boolean or boolean.null_count() or not boolean.all():
        raise ValueError("session division presence flags must all be Boolean true")
    if set(presence.get_column("unit_population").unique().to_list()) != {
        "all_units_presence_only"
    }:
        raise ValueError("session division presence must use all units")
    if set(presence.get_column("count_policy").unique().to_list()) != {
        "unit_count_magnitude_not_used"
    }:
        raise ValueError("session division presence must discard unit count magnitude")

    definition_check = presence.join(
        divisions,
        on="major_division_id",
        how="left",
        validate="m:1",
        suffix="_definition",
    )
    invalid_definitions = definition_check.filter(
        pl.col("major_division_acronym_definition").is_null()
        | (pl.col("major_division_acronym") != pl.col("major_division_acronym_definition"))
        | (pl.col("major_division_name") != pl.col("major_division_name_definition"))
    )
    if invalid_definitions.height:
        raise ValueError("session division presence conflicts with major-division definitions")

    qc_identity = qc.select(SESSION_KEY, "subject_id")
    identity_check = (
        presence.select(SESSION_KEY, "subject_id")
        .unique()
        .join(
            qc_identity,
            on=SESSION_KEY,
            how="left",
            validate="1:1",
            suffix="_qc",
        )
    )
    if identity_check.filter(
        pl.col("subject_id_qc").is_null() | (pl.col("subject_id") != pl.col("subject_id_qc"))
    ).height:
        raise ValueError("session division presence identity differs from session QC")

    selected = qc.filter(pl.col("is_good_session")).select(SESSION_KEY, "subject_id")
    if selected.is_empty():
        raise ValueError("session QC contains no threshold-selected sessions")
    selected_counts = dict(selected.group_by("subject_id").len(name="n_selected").iter_rows())
    selected_keys = set(selected.get_column(SESSION_KEY).to_list())
    selected_presence = presence.filter(pl.col(SESSION_KEY).is_in(selected_keys))
    present_pairs = set(selected_presence.select("subject_id", "major_division_id").iter_rows())
    present_session_counts = {
        (row["subject_id"], row["major_division_id"]): row["n_present"]
        for row in selected_presence.group_by("subject_id", "major_division_id")
        .len(name="n_present")
        .iter_rows(named=True)
    }

    fractions_by_division: dict[int, dict[str, fractions.Fraction]] = {}
    for division in divisions.iter_rows(named=True):
        identifier = int(division["major_division_id"])
        fractions_by_division[identifier] = {
            mouse: fractions.Fraction(
                present_session_counts.get((mouse, identifier), 0), denominator
            )
            for mouse, denominator in selected_counts.items()
        }

    summaries: dict[int, dict[str, Any]] = {}
    for identifier, values in fractions_by_division.items():
        mean, variance = _mean_and_sample_variance(values.values())
        represented = sum(value > 0 for value in values.values())
        include = represented >= minimum_represented_mice and variance > 0
        if represented < minimum_represented_mice:
            exclusion = "represented_in_fewer_than_minimum_eligible_mice"
        elif variance == 0:
            exclusion = "zero_eligible_mouse_variance"
        else:
            exclusion = None
        summaries[identifier] = {
            "n_eligible_mice_represented": represented,
            "eligible_mouse_fraction_mean": float(mean),
            "eligible_mouse_fraction_mean_exact": str(mean),
            "eligible_mouse_fraction_variance": float(variance),
            "eligible_mouse_fraction_variance_exact": str(variance),
            "nonzero_eligible_mouse_variance": variance > 0,
            "include_in_balance": include,
            "balance_exclusion_reason": exclusion,
        }

    rows = []
    for mouse in sorted(selected_counts):
        denominator = selected_counts[mouse]
        for division in divisions.iter_rows(named=True):
            identifier = int(division["major_division_id"])
            numerator = present_session_counts.get((mouse, identifier), 0)
            value = fractions.Fraction(numerator, denominator)
            rows.append(
                {
                    "subject_id": mouse,
                    **division,
                    "n_threshold_selected_sessions": denominator,
                    "n_threshold_selected_sessions_with_division": numerator,
                    "threshold_selected_session_fraction": float(value),
                    "threshold_selected_session_fraction_exact": str(value),
                    **summaries[identifier],
                    "minimum_represented_mice_for_balance": minimum_represented_mice,
                    "feature_family": "coarse_regional_coverage",
                    "coverage_definition": (
                        "fraction_of_d03_selected_sessions_with_any_raw_located_unit"
                    ),
                    "unit_population": "all_units_presence_only",
                    "count_policy": "unit_count_magnitude_not_used",
                    "neural_activity_outcomes_used": False,
                }
            )
    result = pl.DataFrame(rows, infer_schema_length=None).sort("subject_id", "major_division_id")
    expected_pairs = {
        (mouse, int(division_id))
        for mouse in selected_counts
        for division_id in divisions.get_column("major_division_id").to_list()
    }
    if set(result.select("subject_id", "major_division_id").iter_rows()) != expected_pairs:
        raise RuntimeError("mouse major-division coverage grid is incomplete")
    if present_pairs.difference(expected_pairs):
        raise RuntimeError("selected session presence produced an unexpected mouse/division")
    return result


def allocate_mouse_cohorts(
    session_inventory: pl.DataFrame | pl.LazyFrame,
    session_qc: pl.DataFrame | pl.LazyFrame,
    *,
    mouse_major_division_coverage: pl.DataFrame | pl.LazyFrame | None = None,
    discovery_mice: int = DEFAULT_DISCOVERY_MICE,
    confirmation_mice: int = DEFAULT_CONFIRMATION_MICE,
    seed: int = DEFAULT_SPLIT_SEED,
) -> CohortAllocation:
    """Allocate all behavior-eligible mice and retain every inventory session.

    A mouse is eligible when at least one of its sessions has
    ``is_good_session == True``. All inventory sessions belonging to an
    eligible mouse inherit that mouse's split. Mice with no selected session
    are retained with the ``excluded`` allocation.

    Omitting ``mouse_major_division_coverage`` deliberately produces a
    provisional allocation with every downstream inclusion flag false. A
    complete, validated coverage grid produces the final D05 allocation and
    activates only D03-selected discovery sessions. Confirmation inclusion is
    always false here and requires the separate one-time confirmation gate.
    """

    if discovery_mice <= 0 or confirmation_mice <= 0:
        raise ValueError("discovery_mice and confirmation_mice must both be positive")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    inventory = _collect(session_inventory)
    qc = _collect(session_qc)
    inventory, qc = _validate_and_select_inputs(inventory, qc)
    base_mice = _summarize_mice(inventory, qc)
    eligible = base_mice.filter(pl.col("mouse_behavior_eligible")).sort("subject_id")
    expected_eligible = discovery_mice + confirmation_mice
    if eligible.height != expected_eligible:
        raise ValueError(
            "eligible mouse count must equal the requested split sizes: "
            f"observed={eligible.height}, requested={discovery_mice}+{confirmation_mice}"
        )

    eligible_ids = tuple(eligible.get_column("subject_id").to_list())
    regional_coverage = (
        None if mouse_major_division_coverage is None else _collect(mouse_major_division_coverage)
    )
    is_final = regional_coverage is not None
    features = _build_features(eligible, regional_coverage)
    if not features:
        raise ValueError("no variable metadata features are available for balancing")
    (
        confirmation_ids,
        chosen_score,
        candidates_evaluated,
        primary_ties,
        secondary_ties,
        tie_break_sha256,
    ) = _exhaustive_confirmation_search(
        eligible_ids,
        features,
        confirmation_mice=confirmation_mice,
        seed=seed,
    )
    confirmation_set = frozenset(confirmation_ids)
    discovery_ids = tuple(mouse for mouse in eligible_ids if mouse not in confirmation_set)

    allocation_lookup = {
        subject_id: ("confirmation" if subject_id in confirmation_set else "discovery")
        for subject_id in eligible_ids
    }
    allocation_status = FINAL_ALLOCATION_STATUS if is_final else PROVISIONAL_ALLOCATION_STATUS
    mice = _add_mouse_assignments(
        base_mice,
        allocation_lookup,
        seed=seed,
        allocation_status=allocation_status,
    )
    sessions = _build_session_assignments(
        inventory,
        qc,
        mice,
        allocation_status=allocation_status,
    )
    balance = _build_balance_table(
        features,
        discovery_ids=discovery_ids,
        confirmation_ids=confirmation_ids,
        score=chosen_score,
    )
    summary = _build_summary_table(
        mice,
        sessions,
        score=chosen_score,
        seed=seed,
        discovery_mice=discovery_mice,
        confirmation_mice=confirmation_mice,
        candidates_evaluated=candidates_evaluated,
        primary_ties=primary_ties,
        secondary_ties=secondary_ties,
        tie_break_sha256=tie_break_sha256,
        n_features=len(features),
        n_families=len({feature.family for feature in features}),
        allocation_status=allocation_status,
    )
    _validate_outputs(
        mice,
        sessions,
        inventory=inventory,
        discovery_mice=discovery_mice,
        confirmation_mice=confirmation_mice,
        allocation_status=allocation_status,
        seed=seed,
    )
    return CohortAllocation(mice=mice, sessions=sessions, balance=balance, summary=summary)


def select_discovery_session_sources(
    session_assignments: pl.DataFrame | pl.LazyFrame,
    *,
    expected_session_sources: Sequence[str],
) -> tuple[str, ...]:
    """Reject unauthenticated DataFrame-only discovery selection.

    A structurally valid table cannot prove that the exact D05 allocation or
    its provenance has not been replaced. Neural analyses must call
    :func:`select_authenticated_discovery_session_sources` with the on-disk
    assignment, D05 manifest, and current authoritative parent artifacts.
    """

    del session_assignments, expected_session_sources
    raise RuntimeError(
        "DataFrame-only discovery selection is unauthenticated; use "
        "select_authenticated_discovery_session_sources"
    )


def _select_discovery_session_sources_from_validated_table(
    session_assignments: pl.DataFrame | pl.LazyFrame,
    *,
    expected_session_sources: Sequence[str],
) -> tuple[str, ...]:
    """Return discovery sources after the file-level API authenticates D05.

    This private helper verifies the final table's structural invariants. It is
    deliberately not sufficient by itself because whole-mouse swaps can retain
    all of those invariants.
    """

    assignments = _collect(session_assignments)
    required = (
        SESSION_KEY,
        "subject_id",
        "cohort_assignment",
        "session_behavior_eligible",
        "mouse_behavior_eligible",
        "allocation_status",
        "split_seed",
        "ready_for_neural_discovery",
        "discovery_mouse_included",
        "confirmation_mouse_included",
        "discovery_analysis_included",
        "confirmation_analysis_included",
        "neural_activity_used_for_allocation",
    )
    _require_columns(assignments, required, table="session cohort assignments")
    _validate_unique_nonnull_key(assignments, table="session cohort assignments")
    _validate_subject_ids(assignments, table="session cohort assignments")
    if assignments.is_empty():
        raise RuntimeError("session cohort assignments are empty")
    expected_sources = tuple(expected_session_sources)
    if not expected_sources or any(
        not isinstance(value, str) or not value for value in expected_sources
    ):
        raise ValueError("expected_session_sources must contain non-empty strings")
    if len(set(expected_sources)) != len(expected_sources):
        raise ValueError("expected_session_sources must be unique")
    observed_sources = set(assignments.get_column(SESSION_KEY).to_list())
    if assignments.height != len(expected_sources) or observed_sources != set(expected_sources):
        raise RuntimeError(
            "session cohort assignments do not exactly cover expected session sources"
        )
    for column in (
        "session_behavior_eligible",
        "mouse_behavior_eligible",
        "ready_for_neural_discovery",
        "discovery_mouse_included",
        "confirmation_mouse_included",
        "discovery_analysis_included",
        "confirmation_analysis_included",
        "neural_activity_used_for_allocation",
    ):
        if assignments.get_column(column).dtype != pl.Boolean:
            raise TypeError(f"session cohort assignments {column} must be Boolean")
        if assignments.get_column(column).null_count():
            raise RuntimeError(f"session cohort assignments {column} contains nulls")

    if assignments.get_column("allocation_status").null_count():
        raise RuntimeError("session cohort assignments allocation_status contains nulls")
    statuses = assignments.get_column("allocation_status").unique().to_list()
    if statuses != [FINAL_ALLOCATION_STATUS]:
        raise RuntimeError(
            "discovery sources are locked until one uniform final allocation is present; "
            f"observed statuses={sorted(statuses)}"
        )
    if not assignments.get_column("ready_for_neural_discovery").all():
        raise RuntimeError(
            "discovery sources are locked because readiness is not true for all rows"
        )
    if assignments.get_column("confirmation_analysis_included").any():
        raise RuntimeError("discovery selector refuses an activated confirmation holdout")
    if assignments.get_column("confirmation_mouse_included").any():
        raise RuntimeError("discovery selector refuses activated confirmation mice")
    if assignments.get_column("neural_activity_used_for_allocation").any():
        raise RuntimeError("discovery selector refuses an outcome-informed allocation")
    seeds = assignments.get_column("split_seed").unique().to_list()
    if seeds != [DEFAULT_SPLIT_SEED]:
        raise RuntimeError(
            f"discovery selector requires split seed {DEFAULT_SPLIT_SEED}; observed={seeds}"
        )
    if assignments.get_column("cohort_assignment").null_count():
        raise RuntimeError("session cohort assignments cohort_assignment contains nulls")
    observed_cohorts = set(assignments.get_column("cohort_assignment").unique().to_list())
    if not {"discovery", "confirmation"}.issubset(observed_cohorts):
        raise RuntimeError("session cohort assignments lack discovery or confirmation rows")
    unexpected_cohorts = observed_cohorts.difference({"discovery", "confirmation", "excluded"})
    if unexpected_cohorts:
        raise RuntimeError(f"session cohort assignments have invalid cohorts: {unexpected_cohorts}")
    per_mouse = assignments.group_by("subject_id").agg(
        pl.col("cohort_assignment").n_unique().alias("n_cohort_assignments")
    )
    if per_mouse.filter(pl.col("n_cohort_assignments") != 1).height:
        raise RuntimeError("discovery selector refuses a mouse split across cohorts")
    expected_mouse_behavior_eligible = pl.col("cohort_assignment") != "excluded"
    if assignments.filter(
        pl.col("mouse_behavior_eligible") != expected_mouse_behavior_eligible
    ).height:
        raise RuntimeError("mouse eligibility flags differ from cohort assignments")
    expected_discovery_mouse = pl.col("cohort_assignment") == "discovery"
    if assignments.filter(pl.col("discovery_mouse_included") != expected_discovery_mouse).height:
        raise RuntimeError("discovery mouse flags differ from cohort assignments")

    expected_discovery = (pl.col("cohort_assignment") == "discovery") & pl.col(
        "session_behavior_eligible"
    )
    invalid_flags = assignments.filter(pl.col("discovery_analysis_included") != expected_discovery)
    if invalid_flags.height:
        raise RuntimeError(
            "discovery inclusion flags are incomplete or expose a non-discovery/failed-QC row"
        )
    sources = (
        assignments.filter(pl.col("discovery_analysis_included"))
        .get_column(SESSION_KEY)
        .sort()
        .to_list()
    )
    if not sources:
        raise RuntimeError("final allocation contains no discovery sources")
    return tuple(sources)


def select_authenticated_discovery_session_sources(
    *,
    session_assignments_path: str | os.PathLike[str],
    cohort_manifest_path: str | os.PathLike[str],
    session_inventory_path: str | os.PathLike[str],
    session_qc_path: str | os.PathLike[str],
    repository_root: str | os.PathLike[str],
) -> tuple[str, ...]:
    """Authenticate final D05 artifacts before returning discovery sources.

    This is the only supported file-to-neural-analysis bridge. It binds the
    assignment Parquet to the completed authoritative D05
    manifest, verifies every recorded input/output/source/dependency hash, and
    rechecks the allocation against the current authoritative inventory and
    behavior-QC Parquets before exposing any source.
    """

    root = pathlib.Path(repository_root).resolve()
    manifest_path = pathlib.Path(cohort_manifest_path).resolve()
    assignments_path = pathlib.Path(session_assignments_path).resolve()
    inventory_path = pathlib.Path(session_inventory_path).resolve()
    qc_path = pathlib.Path(session_qc_path).resolve()
    for label, path in (
        ("cohort manifest", manifest_path),
        ("session assignments", assignments_path),
        ("session inventory", inventory_path),
        ("session QC", qc_path),
    ):
        if not path.is_relative_to(root):
            raise RuntimeError(f"{label} path escapes the repository root")

    manifest = _read_stable_json_object(manifest_path, label="D05 cohort manifest")
    expected_manifest_fields = {
        "schema_version": COHORT_MANIFEST_SCHEMA_VERSION,
        "analysis_id": COHORT_ANALYSIS_ID,
        "run_status": "complete",
        "authoritative": True,
        "analysis_status": "pass",
        "allocation_status": "final",
        "assignment_table_status": FINAL_ALLOCATION_STATUS,
        "ready_for_neural_discovery": True,
        "confirmation_holdout_accessed": False,
        "neural_activity_used_for_allocation": False,
        "split_seed": DEFAULT_SPLIT_SEED,
        "requested_discovery_mice": DEFAULT_DISCOVERY_MICE,
        "requested_confirmation_mice": DEFAULT_CONFIRMATION_MICE,
    }
    for field, expected in expected_manifest_fields.items():
        if manifest.get(field) != expected:
            raise RuntimeError(f"D05 cohort manifest {field} is not the authenticated final value")

    inputs = manifest.get("inputs")
    outputs = manifest.get("outputs")
    dependencies = manifest.get("dependency_lockfiles")
    local_sources = manifest.get("local_sources")
    if not isinstance(inputs, dict) or not inputs:
        raise RuntimeError("D05 cohort manifest lacks its input hash chain")
    if not isinstance(outputs, dict) or not outputs:
        raise RuntimeError("D05 cohort manifest lacks its output hash chain")
    if not isinstance(dependencies, dict) or not dependencies:
        raise RuntimeError("D05 cohort manifest lacks dependency lockfile hashes")
    if not isinstance(local_sources, list) or not local_sources:
        raise RuntimeError("D05 cohort manifest lacks local-source hashes")
    required_inputs = {
        "analysis_lock",
        "m0_audit_run",
        "behavior_analysis_run",
        "session_inventory",
        "session_qc",
        "m0_artifact__session_inventory",
        "m0_artifact__task_parameters_audit",
        "m0_artifact__unit_session_anatomy_coverage",
        "behavior_input_m0_manifest",
        "behavior_task_parameters_audit",
        "behavior_trials_checkpoint",
        "behavior_current_trials",
        "behavior_current_session_qc",
    }
    missing_inputs = sorted(required_inputs.difference(inputs))
    if missing_inputs:
        raise RuntimeError(f"D05 cohort manifest lacks required inputs: {missing_inputs}")
    required_outputs = {
        "mouse_assignments_parquet",
        "session_assignments_parquet",
        "mouse_major_division_coverage_parquet",
        "software_environment",
    }
    missing_outputs = sorted(required_outputs.difference(outputs))
    if missing_outputs:
        raise RuntimeError(f"D05 cohort manifest lacks required outputs: {missing_outputs}")

    validated_inputs = {
        name: _authenticate_manifest_file_record(record, root=root, label=f"input {name}")
        for name, record in sorted(inputs.items())
    }
    validated_outputs = {
        name: _authenticate_manifest_file_record(record, root=root, label=f"output {name}")
        for name, record in sorted(outputs.items())
    }
    validated_dependencies = {
        name: _authenticate_manifest_file_record(
            record,
            root=root,
            label=f"dependency lockfile {name}",
        )
        for name, record in sorted(dependencies.items())
    }
    expected_dependency_paths = {
        name: (root / name).resolve() for name in D05_REQUIRED_DEPENDENCY_PATHS
    }
    if validated_dependencies != expected_dependency_paths:
        raise RuntimeError("D05 dependency lockfiles are missing, unexpected, or misbound")
    validated_local_sources = [
        _authenticate_manifest_file_record(
            record,
            root=root,
            label=f"local source {index}",
        )
        for index, record in enumerate(local_sources)
    ]
    expected_local_sources = {(root / path).resolve() for path in D05_REQUIRED_LOCAL_SOURCE_PATHS}
    if set(validated_local_sources) != expected_local_sources or len(
        validated_local_sources
    ) != len(expected_local_sources):
        raise RuntimeError("D05 local-source hashes are missing, unexpected, or duplicated")
    expected_paths = {
        "session_assignments_parquet": assignments_path,
        "session_inventory": inventory_path,
        "session_qc": qc_path,
    }
    for name, expected_path in expected_paths.items():
        observed_path = (
            validated_outputs[name]
            if name == "session_assignments_parquet"
            else validated_inputs[name]
        )
        if observed_path != expected_path:
            raise RuntimeError(f"authenticated D05 {name} path differs from the supplied path")

    parent_hashes = manifest.get("parent_hashes")
    if not isinstance(parent_hashes, dict):
        raise RuntimeError("D05 cohort manifest lacks exact parent hashes")
    expected_parent_hashes = {
        "m0_audit_run_sha256": inputs["m0_audit_run"].get("sha256"),
        "behavior_analysis_run_sha256": inputs["behavior_analysis_run"].get("sha256"),
        "session_inventory_sha256": inputs["session_inventory"].get("sha256"),
        "session_qc_sha256": inputs["session_qc"].get("sha256"),
    }
    if parent_hashes != expected_parent_hashes:
        raise RuntimeError("D05 exact parent hashes differ from its input records")
    assignment_hashes = manifest.get("assignment_hashes")
    expected_assignment_hashes = {
        "mouse_assignments_parquet_sha256": outputs["mouse_assignments_parquet"].get("sha256"),
        "session_assignments_parquet_sha256": outputs["session_assignments_parquet"].get("sha256"),
    }
    if assignment_hashes != expected_assignment_hashes:
        raise RuntimeError("D05 exact assignment hashes differ from its output records")

    m0 = _read_stable_json_object(validated_inputs["m0_audit_run"], label="bound M0 manifest")
    dandiset = manifest.get("dandiset")
    expected_dandiset_version = (
        dandiset.get("dandiset_version") if isinstance(dandiset, dict) else None
    )
    if (
        m0.get("manifest_schema_version") != 3
        or m0.get("run_status") != "complete"
        or m0.get("authoritative") is not True
        or m0.get("milestone_0_status") not in {"partial", "complete"}
        or not isinstance(expected_dandiset_version, str)
        or m0.get("dandiset_version") != expected_dandiset_version
        or m0.get("analysis_lock_status") != "approved"
        or m0.get("asset_inventory_status") != "pass"
        or m0.get("nwb_root_metadata_audit_status") != "pass"
        or m0.get("unit_inventory_status") != "pass"
        or m0.get("unit_metadata_inventory_status") != "pass_scalar_metadata_no_spike_arrays"
        or m0.get("neural_outcomes_accessed") is not False
        or m0.get("spike_arrays_loaded") is not False
    ):
        raise RuntimeError("bound M0 manifest is not authoritative and D05-safe")
    m0_base = validated_inputs["m0_audit_run"].parent.parent
    m0_artifacts = m0.get("artifacts")
    m0_inputs = m0.get("inputs")
    m0_sources = m0.get("local_sources")
    if not isinstance(m0_artifacts, dict) or not isinstance(m0_inputs, dict) or not m0_inputs:
        raise RuntimeError("bound M0 manifest lacks its artifact/input hash chain")
    if not isinstance(m0_sources, dict) or not m0_sources:
        raise RuntimeError("bound M0 manifest lacks its local-source hash chain")
    required_m0_artifacts = {
        "session_inventory",
        "task_parameters_audit",
        "unit_session_anatomy_coverage",
    }
    if missing_m0 := sorted(required_m0_artifacts.difference(m0_artifacts)):
        raise RuntimeError(f"bound M0 manifest lacks D05 artifacts: {missing_m0}")
    for name, record in sorted(m0_artifacts.items()):
        artifact_path = _authenticate_manifest_file_record(
            record,
            root=root,
            base=m0_base,
            label=f"bound M0 artifact {name}",
        )
        _require_path_bound_in_d05_inputs(
            artifact_path,
            validated_inputs=validated_inputs,
            label=f"M0 artifact {name}",
        )
    for name, record in sorted(m0_inputs.items()):
        input_path = _authenticate_manifest_file_record(
            record,
            root=root,
            base=m0_base,
            label=f"bound M0 input {name}",
        )
        _require_path_bound_in_d05_inputs(
            input_path,
            validated_inputs=validated_inputs,
            label=f"M0 input {name}",
        )
    for name, record in sorted(m0_sources.items()):
        source_path = _authenticate_manifest_file_record(
            record,
            root=root,
            label=f"bound M0 local source {name}",
        )
        _require_path_bound_in_d05_inputs(
            source_path,
            validated_inputs=validated_inputs,
            label=f"M0 local source {name}",
        )
    m0_inventory_path = _authenticate_manifest_file_record(
        m0_artifacts["session_inventory"],
        root=root,
        base=m0_base,
        label="bound M0 session inventory",
    )
    if (
        m0_inventory_path != inventory_path
        or validated_inputs["m0_artifact__session_inventory"] != inventory_path
    ):
        raise RuntimeError("current inventory path differs from the bound M0 artifact")
    m0_task_path = _authenticate_manifest_file_record(
        m0_artifacts["task_parameters_audit"],
        root=root,
        base=m0_base,
        label="bound M0 task-parameters audit",
    )
    if validated_inputs["m0_artifact__task_parameters_audit"] != m0_task_path:
        raise RuntimeError("D05 task-audit input is not the bound M0 artifact")
    m0_anatomy_path = _authenticate_manifest_file_record(
        m0_artifacts["unit_session_anatomy_coverage"],
        root=root,
        base=m0_base,
        label="bound M0 unit-session anatomy coverage",
    )
    if validated_inputs["m0_artifact__unit_session_anatomy_coverage"] != m0_anatomy_path:
        raise RuntimeError("D05 anatomy input is not the bound M0 artifact")

    behavior = _read_stable_json_object(
        validated_inputs["behavior_analysis_run"],
        label="bound behavior manifest",
    )
    _authenticate_bound_behavior_manifest(
        behavior,
        behavior_manifest_path=validated_inputs["behavior_analysis_run"],
        validated_inputs=validated_inputs,
        m0=m0,
        m0_manifest_path=validated_inputs["m0_audit_run"],
        inventory_path=inventory_path,
        qc_path=qc_path,
        root=root,
        expected_dandiset_version=expected_dandiset_version,
    )

    try:
        assignments = pl.read_parquet(assignments_path)
        mouse_assignments = pl.read_parquet(validated_outputs["mouse_assignments_parquet"])
        inventory = pl.read_parquet(inventory_path)
        qc = pl.read_parquet(qc_path)
        mouse_coverage = pl.read_parquet(validated_outputs["mouse_major_division_coverage_parquet"])
    except (OSError, pl.exceptions.PolarsError) as error:
        raise RuntimeError(f"could not read authenticated D05 Parquet inputs: {error}") from error
    normalized_inventory, normalized_qc = _validate_and_select_inputs(inventory, qc)
    expected_allocation = allocate_mouse_cohorts(
        normalized_inventory,
        normalized_qc,
        mouse_major_division_coverage=mouse_coverage,
        discovery_mice=DEFAULT_DISCOVERY_MICE,
        confirmation_mice=DEFAULT_CONFIRMATION_MICE,
        seed=DEFAULT_SPLIT_SEED,
    )
    if mouse_assignments.columns != expected_allocation.mice.columns:
        raise RuntimeError("authenticated mouse assignments have an unexpected schema")
    if mouse_assignments.schema != expected_allocation.mice.schema or not mouse_assignments.sort(
        "subject_id"
    ).equals(expected_allocation.mice.sort("subject_id")):
        raise RuntimeError(
            "authenticated mouse assignments differ from the recomputed exact D05 split"
        )
    if assignments.columns != expected_allocation.sessions.columns:
        raise RuntimeError("authenticated assignments have an unexpected schema")
    if assignments.schema != expected_allocation.sessions.schema or not assignments.sort(
        SESSION_KEY
    ).equals(expected_allocation.sessions.sort(SESSION_KEY)):
        raise RuntimeError("authenticated assignments differ from the recomputed exact D05 split")
    _validate_authenticated_assignment_content(
        assignments,
        normalized_inventory,
        normalized_qc,
        manifest=manifest,
    )
    return _select_discovery_session_sources_from_validated_table(
        assignments,
        expected_session_sources=normalized_inventory.get_column(SESSION_KEY).to_list(),
    )


def _authenticate_bound_behavior_manifest(
    behavior: Mapping[str, Any],
    *,
    behavior_manifest_path: pathlib.Path,
    validated_inputs: Mapping[str, pathlib.Path],
    m0: Mapping[str, Any],
    m0_manifest_path: pathlib.Path,
    inventory_path: pathlib.Path,
    qc_path: pathlib.Path,
    root: pathlib.Path,
    expected_dandiset_version: str,
) -> None:
    trial_input_mode = behavior.get("trial_input_mode")
    expected_status = (
        {
            "remote_nwb": "created_from_remote_nwb",
            "validated_existing_behavior_trials": "validated_existing_not_refreshed",
        }.get(trial_input_mode)
        if isinstance(trial_input_mode, str)
        else None
    )
    checkpoint_reference = behavior.get("behavior_trials_checkpoint")
    if (
        behavior.get("run_status") != "complete"
        or behavior.get("authoritative") is not True
        or behavior.get("analysis_status") != "pass"
        or behavior.get("pending_behavior_decisions") != []
        or behavior.get("dandiset_version") != expected_dandiset_version
        or expected_status is None
        or not isinstance(checkpoint_reference, dict)
        or checkpoint_reference.get("status") != expected_status
        or checkpoint_reference.get("output_matches_trusted_checkpoint") is not True
    ):
        raise RuntimeError("bound behavior manifest is not authoritative canonical behavior QC")

    generator_path = _authenticate_manifest_file_record(
        behavior.get("generator"),
        root=root,
        label="bound behavior generator",
    )
    _require_path_bound_in_d05_inputs(
        generator_path,
        validated_inputs=validated_inputs,
        label="behavior generator",
    )
    behavior_sources = behavior.get("local_sources")
    if not isinstance(behavior_sources, list) or not behavior_sources:
        raise RuntimeError("bound behavior manifest lacks local-source hashes")
    behavior_source_paths = []
    for index, source in enumerate(behavior_sources):
        source_path = _authenticate_manifest_file_record(
            source,
            root=root,
            label=f"bound behavior local source {index}",
        )
        _require_path_bound_in_d05_inputs(
            source_path,
            validated_inputs=validated_inputs,
            label=f"behavior local source {index}",
        )
        behavior_source_paths.append(source_path)

    behavior_m0_record = _authenticate_manifest_file_record(
        behavior.get("input_audit_run_manifest"),
        root=root,
        label="bound behavior M0 manifest",
    )
    if behavior_m0_record != m0_manifest_path or behavior.get("input_audit_run") != m0:
        raise RuntimeError("bound behavior manifest does not embed the current M0 parent")
    if validated_inputs.get("behavior_input_m0_manifest") != behavior_m0_record:
        raise RuntimeError("named D05 behavior-M0 input differs from the behavior parent")
    _require_path_bound_in_d05_inputs(
        behavior_m0_record,
        validated_inputs=validated_inputs,
        label="behavior M0 parent",
    )
    if behavior_manifest_path != validated_inputs["behavior_analysis_run"]:
        raise RuntimeError("bound behavior-manifest path differs from the D05 input chain")

    behavior_inventory_value = behavior.get("session_inventory_path")
    if not isinstance(behavior_inventory_value, str) or not behavior_inventory_value:
        raise RuntimeError("bound behavior manifest lacks its inventory path")
    behavior_inventory_path = _resolve_authenticated_path(
        behavior_inventory_value,
        root=root,
    )
    if (
        behavior_inventory_path != inventory_path
        or behavior.get("session_inventory_sha256")
        != hashlib.sha256(_read_stable_bytes(inventory_path, label="current inventory")).hexdigest()
    ):
        raise RuntimeError("bound behavior inventory differs from the current M0 inventory")

    outputs = behavior.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("bound behavior manifest lacks its output hash chain")
    behavior_qc = outputs.get("session_qc")
    behavior_trials = outputs.get("behavior_trials")
    behavior_qc_path = _authenticate_manifest_file_record(
        behavior_qc,
        root=root,
        label="bound behavior session QC",
    )
    current_trials_path = _authenticate_manifest_file_record(
        behavior_trials,
        root=root,
        label="bound current behavior trials",
    )
    if behavior_qc_path != qc_path:
        raise RuntimeError("current session-QC path differs from the bound behavior artifact")
    if (
        validated_inputs.get("behavior_current_session_qc") != behavior_qc_path
        or validated_inputs.get("behavior_current_trials") != current_trials_path
    ):
        raise RuntimeError("named D05 behavior outputs differ from the behavior parent")
    _require_path_bound_in_d05_inputs(
        behavior_qc_path,
        validated_inputs=validated_inputs,
        label="behavior session QC",
    )
    _require_path_bound_in_d05_inputs(
        current_trials_path,
        validated_inputs=validated_inputs,
        label="current behavior trials",
    )

    task_audit_value = behavior.get("task_parameters_audit_path")
    task_audit_sha = behavior.get("task_parameters_audit_sha256")
    if not isinstance(task_audit_value, str) or not task_audit_value:
        raise RuntimeError("bound behavior manifest lacks its task-parameters-audit path")
    task_audit_path = _resolve_authenticated_path(task_audit_value, root=root)
    task_audit_payload = _read_stable_bytes(task_audit_path, label="behavior task audit")
    if hashlib.sha256(task_audit_payload).hexdigest() != task_audit_sha:
        raise RuntimeError("bound behavior task audit differs from its recorded digest")
    m0_task_audit_path = _authenticate_manifest_file_record(
        m0.get("artifacts", {}).get("task_parameters_audit"),
        root=root,
        base=m0_manifest_path.parent.parent,
        label="bound M0 task audit",
    )
    if task_audit_path != m0_task_audit_path:
        raise RuntimeError("bound behavior task audit differs from its M0 parent")
    if validated_inputs.get("behavior_task_parameters_audit") != task_audit_path:
        raise RuntimeError("named D05 behavior task audit differs from the behavior parent")
    _require_path_bound_in_d05_inputs(
        task_audit_path,
        validated_inputs=validated_inputs,
        label="behavior task audit",
    )

    lock_value = behavior.get("analysis_lock_snapshot")
    if not isinstance(lock_value, str) or not lock_value:
        raise RuntimeError("bound behavior manifest lacks its lock snapshot path")
    lock_snapshot_path = _resolve_authenticated_path(lock_value, root=root)
    lock_payload = _read_stable_bytes(lock_snapshot_path, label="behavior lock snapshot")
    if hashlib.sha256(lock_payload).hexdigest() != behavior.get("analysis_lock_sha256"):
        raise RuntimeError("bound behavior lock snapshot differs from its recorded digest")
    current_lock_path = validated_inputs.get("analysis_lock")
    if (
        current_lock_path is None
        or _read_stable_bytes(
            current_lock_path,
            label="current analysis lock",
        )
        != lock_payload
    ):
        raise RuntimeError("bound behavior lock differs from the current D05 analysis lock")
    _require_path_bound_in_d05_inputs(
        lock_snapshot_path,
        validated_inputs=validated_inputs,
        label="behavior lock snapshot",
    )

    checkpoint_path = _authenticate_manifest_file_record(
        checkpoint_reference,
        root=root,
        label="bound behavior-trials checkpoint manifest",
    )
    if validated_inputs.get("behavior_trials_checkpoint") != checkpoint_path:
        raise RuntimeError("behavior checkpoint is not bound into the D05 input chain")
    checkpoint = _read_stable_json_object(
        checkpoint_path,
        label="bound behavior-trials checkpoint",
    )
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("checkpoint_kind") != "direct_remote_nwb_behavior_trials"
        or checkpoint.get("trial_input_mode") != "remote_nwb"
        or checkpoint.get("dandiset_version") != expected_dandiset_version
        or checkpoint.get("dandiset_id") != behavior.get("dandiset_id")
    ):
        raise RuntimeError("bound behavior-trials checkpoint is not canonical remote-NWB output")

    checkpoint_inventory_path = _authenticate_manifest_file_record(
        checkpoint.get("session_inventory"),
        root=root,
        label="bound behavior-checkpoint inventory",
    )
    trusted_trials_path = _authenticate_manifest_file_record(
        checkpoint.get("behavior_trials"),
        root=root,
        label="bound trusted behavior trials",
    )
    if checkpoint_inventory_path != inventory_path or trusted_trials_path != current_trials_path:
        raise RuntimeError(
            "behavior checkpoint data files differ from current authoritative inputs"
        )
    checkpoint_trials = checkpoint.get("behavior_trials")
    if not isinstance(checkpoint_trials, dict) or not isinstance(behavior_trials, dict):
        raise RuntimeError("behavior trial provenance records are malformed")
    trusted_sha = checkpoint_trials.get("sha256")
    if (
        checkpoint_reference.get("trusted_behavior_trials_sha256") != trusted_sha
        or behavior_trials.get("sha256") != trusted_sha
    ):
        raise RuntimeError("behavior trusted/current/checkpoint trial digests differ")
    reused_sha = behavior.get("reused_behavior_trials_sha256")
    if trial_input_mode == "remote_nwb" and reused_sha is not None:
        raise RuntimeError("direct remote behavior run unexpectedly reports reused trials")
    if trial_input_mode == "validated_existing_behavior_trials" and reused_sha != trusted_sha:
        raise RuntimeError("reused behavior trials differ from their trusted checkpoint")

    _require_path_bound_in_d05_inputs(
        checkpoint_inventory_path,
        validated_inputs=validated_inputs,
        label="behavior-checkpoint inventory",
    )
    _require_path_bound_in_d05_inputs(
        trusted_trials_path,
        validated_inputs=validated_inputs,
        label="trusted behavior trials",
    )

    checkpoint_generator_path = _authenticate_manifest_file_record(
        checkpoint.get("generator"),
        root=root,
        label="bound behavior-checkpoint generator",
    )
    _require_path_bound_in_d05_inputs(
        checkpoint_generator_path,
        validated_inputs=validated_inputs,
        label="behavior-checkpoint generator",
    )
    checkpoint_sources = checkpoint.get("local_sources")
    if not isinstance(checkpoint_sources, list) or not checkpoint_sources:
        raise RuntimeError("bound behavior checkpoint lacks local-source hashes")
    checkpoint_source_paths = []
    for index, source in enumerate(checkpoint_sources):
        source_path = _authenticate_manifest_file_record(
            source,
            root=root,
            label=f"bound behavior-checkpoint local source {index}",
        )
        _require_path_bound_in_d05_inputs(
            source_path,
            validated_inputs=validated_inputs,
            label=f"behavior-checkpoint local source {index}",
        )
        checkpoint_source_paths.append(source_path)
    if (
        checkpoint_generator_path != generator_path
        or checkpoint_source_paths != behavior_source_paths
    ):
        raise RuntimeError("bound behavior run and checkpoint source chains differ")

    companion_path = _authenticate_manifest_file_record(
        checkpoint.get("companion_trials"),
        root=root,
        label="bound behavior-checkpoint companion trials",
    )
    companion = behavior.get("companion_trials_provenance")
    if not isinstance(companion, dict):
        raise RuntimeError("bound behavior manifest lacks companion-trial provenance")
    cached_path = companion.get("cached_path")
    if not isinstance(cached_path, str) or not cached_path:
        raise RuntimeError("bound behavior companion provenance lacks cached_path")
    if (
        _resolve_authenticated_path(cached_path, root=root) != companion_path
        or companion.get("sha256") != checkpoint.get("companion_trials", {}).get("sha256")
        or companion.get("content_size_bytes")
        != checkpoint.get("companion_trials", {}).get("size_bytes")
    ):
        raise RuntimeError("bound behavior companion provenance differs from its checkpoint")
    for field in ("source_url", "repository_commit"):
        if companion.get(field) != checkpoint.get("companion_trials", {}).get(field):
            raise RuntimeError(f"bound behavior companion {field} differs from its checkpoint")
    _require_path_bound_in_d05_inputs(
        companion_path,
        validated_inputs=validated_inputs,
        label="behavior companion trials",
    )


def _resolve_authenticated_path(value: str, *, root: pathlib.Path) -> pathlib.Path:
    candidate = pathlib.Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not path.is_relative_to(root):
        raise RuntimeError("authenticated provenance path escapes the repository root")
    return path


def _require_path_bound_in_d05_inputs(
    path: pathlib.Path,
    *,
    validated_inputs: Mapping[str, pathlib.Path],
    label: str,
) -> None:
    if path not in validated_inputs.values():
        raise RuntimeError(f"{label} is not bound into the D05 input hash chain")


def _validate_authenticated_assignment_content(
    assignments: pl.DataFrame,
    inventory: pl.DataFrame,
    qc: pl.DataFrame,
    *,
    manifest: Mapping[str, Any],
) -> None:
    required = {
        SESSION_KEY,
        "subject_id",
        "cohort_assignment",
        "session_behavior_eligible",
        "mouse_behavior_eligible",
    }
    missing = sorted(required.difference(assignments.columns))
    if missing:
        raise RuntimeError(f"authenticated session assignments lack columns: {missing}")
    if assignments.height != inventory.height:
        raise RuntimeError("authenticated assignments do not preserve the inventory row count")
    expected_pairs = set(inventory.select(SESSION_KEY, "subject_id").iter_rows())
    observed_pairs = set(assignments.select(SESSION_KEY, "subject_id").iter_rows())
    if observed_pairs != expected_pairs or len(observed_pairs) != assignments.height:
        raise RuntimeError("authenticated assignments do not preserve inventory session keys")

    qc_check = assignments.select(
        SESSION_KEY,
        "subject_id",
        "session_behavior_eligible",
    ).join(
        qc.select(
            SESSION_KEY,
            "subject_id",
            pl.col("is_good_session").alias("expected_session_behavior_eligible"),
        ),
        on=(SESSION_KEY, "subject_id"),
        how="inner",
        validate="1:1",
    )
    if (
        qc_check.height != assignments.height
        or qc_check.filter(
            pl.col("session_behavior_eligible") != pl.col("expected_session_behavior_eligible")
        ).height
    ):
        raise RuntimeError("assignment behavior-eligibility flags differ from current QC")

    selected_by_mouse = {
        row["subject_id"]: bool(row["mouse_behavior_eligible"])
        for row in qc.group_by("subject_id")
        .agg(pl.col("is_good_session").any().alias("mouse_behavior_eligible"))
        .iter_rows(named=True)
    }
    assignment_rows = assignments.select(
        "subject_id", "cohort_assignment", "mouse_behavior_eligible"
    ).unique()
    if assignment_rows.get_column("subject_id").n_unique() != assignment_rows.height:
        raise RuntimeError("authenticated assignments split at least one mouse across cohorts")
    for row in assignment_rows.iter_rows(named=True):
        expected_eligible = selected_by_mouse.get(row["subject_id"])
        if expected_eligible is None or row["mouse_behavior_eligible"] is not expected_eligible:
            raise RuntimeError("assignment mouse eligibility differs from current QC")
        expected_cohort_valid = (
            row["cohort_assignment"] in {"discovery", "confirmation"}
            if expected_eligible
            else row["cohort_assignment"] == "excluded"
        )
        if not expected_cohort_valid:
            raise RuntimeError("assignment cohort differs from behavior eligibility")

    cohort_ids = {
        cohort: sorted(
            assignment_rows.filter(pl.col("cohort_assignment") == cohort)
            .get_column("subject_id")
            .to_list()
        )
        for cohort in ("discovery", "confirmation", "excluded")
    }
    if len(cohort_ids["discovery"]) != DEFAULT_DISCOVERY_MICE:
        raise RuntimeError("authenticated assignment does not contain 12 discovery mice")
    if len(cohort_ids["confirmation"]) != DEFAULT_CONFIRMATION_MICE:
        raise RuntimeError("authenticated assignment does not contain 6 confirmation mice")
    identity = manifest.get("assignment_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("D05 cohort manifest lacks assignment identity")
    for cohort, ids in cohort_ids.items():
        if identity.get(f"{cohort}_mouse_ids") != ids:
            raise RuntimeError(f"authenticated {cohort} mouse identity differs from manifest")

    counts = manifest.get("cohort_counts")
    if not isinstance(counts, dict):
        raise RuntimeError("D05 cohort manifest lacks cohort counts")
    expected_counts = {
        "inventory_mice": assignment_rows.height,
        "eligible_mice": DEFAULT_DISCOVERY_MICE + DEFAULT_CONFIRMATION_MICE,
        "excluded_mice": len(cohort_ids["excluded"]),
        "discovery_mice": len(cohort_ids["discovery"]),
        "confirmation_mice": len(cohort_ids["confirmation"]),
        "inventory_sessions": assignments.height,
        "threshold_selected_sessions": assignments.filter(
            pl.col("session_behavior_eligible")
        ).height,
        "discovery_threshold_selected_sessions": assignments.filter(
            (pl.col("cohort_assignment") == "discovery") & pl.col("session_behavior_eligible")
        ).height,
        "confirmation_threshold_selected_sessions": assignments.filter(
            (pl.col("cohort_assignment") == "confirmation") & pl.col("session_behavior_eligible")
        ).height,
    }
    for name, expected in expected_counts.items():
        if counts.get(name) != expected:
            raise RuntimeError(f"D05 manifest cohort count {name} differs from assignments")


def _authenticate_manifest_file_record(
    record: Any,
    *,
    root: pathlib.Path,
    label: str,
    base: pathlib.Path | None = None,
) -> pathlib.Path:
    if not isinstance(record, dict):
        raise RuntimeError(f"D05 {label} provenance must be an object")
    path_value = record.get("path")
    digest = record.get("sha256")
    size = record.get("size_bytes")
    if not isinstance(path_value, str) or not path_value:
        raise RuntimeError(f"D05 {label} provenance lacks a path")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise RuntimeError(f"D05 {label} provenance has an invalid SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise RuntimeError(f"D05 {label} provenance has an invalid size")
    relative_or_absolute = pathlib.Path(path_value)
    resolution_base = root if base is None else base.resolve()
    path = (
        relative_or_absolute.resolve()
        if relative_or_absolute.is_absolute()
        else (resolution_base / relative_or_absolute).resolve()
    )
    if not path.is_relative_to(root):
        raise RuntimeError(f"D05 {label} path escapes the repository root")
    payload = _read_stable_bytes(path, label=f"D05 {label}")
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
        raise RuntimeError(f"D05 {label} content differs from its recorded hash or size")
    return path


def _read_stable_bytes(path: pathlib.Path, *, label: str) -> bytes:
    try:
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
    except OSError as error:
        raise RuntimeError(f"could not read {label} {path}: {error}") from error
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"{label} changed while it was read")
    return payload


def _read_stable_json_object(path: pathlib.Path, *, label: str) -> dict[str, Any]:
    payload = _read_stable_bytes(path, label=label)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def _collect(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    return frame.collect() if isinstance(frame, pl.LazyFrame) else frame


def _validate_and_select_inputs(
    inventory: pl.DataFrame,
    qc: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    _require_columns(inventory, _REQUIRED_INVENTORY_COLUMNS, table="session inventory")
    _require_columns(qc, _REQUIRED_QC_COLUMNS, table="session QC")

    inventory_columns = [*_REQUIRED_INVENTORY_COLUMNS]
    inventory_columns.extend(
        column for column in _OPTIONAL_SESSION_ID_COLUMNS if column in inventory.columns
    )
    inventory = inventory.select(inventory_columns)
    qc = qc.select(_REQUIRED_QC_COLUMNS)
    _validate_unique_nonnull_key(inventory, table="session inventory")
    _validate_unique_nonnull_key(qc, table="session QC")
    _validate_subject_ids(inventory, table="session inventory")
    _validate_subject_ids(qc, table="session QC")

    inventory_keys = set(inventory.get_column(SESSION_KEY).to_list())
    qc_keys = set(qc.get_column(SESSION_KEY).to_list())
    if inventory_keys != qc_keys:
        missing_qc = sorted(inventory_keys - qc_keys)
        unexpected_qc = sorted(qc_keys - inventory_keys)
        raise ValueError(
            "session QC coverage differs from inventory: "
            f"missing={missing_qc[:5]}, unexpected={unexpected_qc[:5]}"
        )

    subject_check = inventory.select(SESSION_KEY, "subject_id").join(
        qc.select(SESSION_KEY, pl.col("subject_id").alias("qc_subject_id")),
        on=SESSION_KEY,
        how="inner",
        validate="1:1",
    )
    if subject_check.height != inventory.height:
        raise RuntimeError("subject identity audit lost session rows")
    conflicts = subject_check.filter(~pl.col("subject_id").eq_missing(pl.col("qc_subject_id")))
    if conflicts.height:
        raise ValueError("subject_id differs between session inventory and session QC")
    if qc.get_column("is_good_session").dtype != pl.Boolean:
        raise TypeError("session QC is_good_session must be Boolean")
    if qc.get_column("is_good_session").null_count():
        raise ValueError("session QC is_good_session must not contain nulls")

    normalized_rows = []
    for row in inventory.sort(SESSION_KEY).iter_rows(named=True):
        normalized = dict(row)
        normalized["subject_id"] = row["subject_id"]
        for column in CATEGORICAL_METADATA:
            normalized[column] = _normalize_category(row[column])
        normalized["recording_day_number"] = _recording_day_number(
            row["recording_day"], row["session_number"]
        )
        normalized_rows.append(normalized)
    inventory = pl.DataFrame(normalized_rows, infer_schema_length=None)
    return inventory, qc.sort(SESSION_KEY)


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], *, table: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {missing}")


def _validate_unique_nonnull_key(frame: pl.DataFrame, *, table: str) -> None:
    if frame.get_column(SESSION_KEY).dtype != pl.String:
        raise TypeError(f"{table} {SESSION_KEY} must be String")
    if frame.get_column(SESSION_KEY).null_count():
        raise ValueError(f"{table} {SESSION_KEY} contains nulls")
    if frame.filter(pl.col(SESSION_KEY).str.strip_chars() == "").height:
        raise ValueError(f"{table} {SESSION_KEY} contains empty values")
    if frame.get_column(SESSION_KEY).n_unique() != frame.height:
        raise ValueError(f"{table} {SESSION_KEY} must be unique")


def _validate_subject_ids(frame: pl.DataFrame, *, table: str) -> None:
    subjects = frame.get_column("subject_id")
    if subjects.dtype != pl.String:
        raise TypeError(
            f"{table} subject_id must be String; use the typed Parquet assignment artifact "
            "or an explicit CSV schema override"
        )
    if subjects.null_count():
        raise ValueError(f"{table} subject_id must be non-null")
    invalid = frame.filter(pl.col("subject_id").str.strip_chars() == "")
    if invalid.height:
        raise ValueError(f"{table} subject_id must be non-empty")
    noncanonical = frame.filter(pl.col("subject_id") != pl.col("subject_id").str.strip_chars())
    if noncanonical.height:
        raise ValueError(f"{table} subject_id must not contain surrounding whitespace")


def _normalize_category(value: Any) -> str:
    if value is None:
        return MISSING_CATEGORY
    normalized = str(value).strip()
    return normalized or MISSING_CATEGORY


def _recording_day_number(recording_day: Any, session_number: Any) -> int:
    parsed_day = None
    if recording_day is not None:
        match = re.fullmatch(r"(?:EPHYS[_ -]?)?(\d+)", str(recording_day).strip(), re.I)
        if match:
            parsed_day = int(match.group(1))
    parsed_session = None
    if session_number is not None and not isinstance(session_number, bool):
        numeric = float(session_number)
        if math.isfinite(numeric) and numeric.is_integer():
            parsed_session = int(numeric)
    if parsed_day is None and parsed_session is None:
        raise ValueError(
            f"could not derive recording day from {recording_day!r}, {session_number!r}"
        )
    if parsed_day is not None and parsed_session is not None and parsed_day != parsed_session:
        raise ValueError(
            "recording_day and session_number disagree: "
            f"{recording_day!r} versus {session_number!r}"
        )
    result = parsed_day if parsed_day is not None else parsed_session
    if result is None or result <= 0:
        raise ValueError("recording day must be a positive integer")
    return result


def _summarize_mice(inventory: pl.DataFrame, qc: pl.DataFrame) -> pl.DataFrame:
    sessions = inventory.join(
        qc.select(SESSION_KEY, "is_good_session"),
        on=SESSION_KEY,
        how="inner",
        validate="1:1",
    )
    if sessions.height != inventory.height:
        raise RuntimeError("mouse summarization lost one or more inventory sessions")
    rows = []
    for subject_id, mouse_sessions in sessions.partition_by(
        "subject_id", as_dict=True, maintain_order=False
    ).items():
        mouse_id = subject_id[0] if isinstance(subject_id, tuple) else subject_id
        categorical: dict[str, str] = {}
        for column in CATEGORICAL_METADATA:
            values = sorted(set(mouse_sessions.get_column(column).to_list()))
            if len(values) != 1:
                raise ValueError(f"{column} is inconsistent within mouse {mouse_id!r}: {values}")
            categorical[column] = values[0]
        days = [int(value) for value in mouse_sessions["recording_day_number"].to_list()]
        selected_days = [
            int(day)
            for day, is_selected in zip(
                days, mouse_sessions["is_good_session"].to_list(), strict=True
            )
            if is_selected
        ]
        selected = len(selected_days)
        rows.append(
            {
                "subject_id": str(mouse_id),
                **categorical,
                "project_cohort": categorical["project_code"],
                "n_inventory_sessions": mouse_sessions.height,
                "n_threshold_selected_sessions": selected,
                "n_excluded_inventory_sessions": mouse_sessions.height - selected,
                "n_recording_days": len(set(days)),
                "recording_day_sum": sum(days),
                "recording_day_min": min(days),
                "recording_day_mean": sum(days) / len(days),
                "recording_day_max": max(days),
                "recording_day_span": max(days) - min(days),
                "n_threshold_selected_recording_days": len(set(selected_days)),
                "threshold_selected_recording_day_sum": (
                    sum(selected_days) if selected_days else None
                ),
                "threshold_selected_recording_day_min": (
                    min(selected_days) if selected_days else None
                ),
                "threshold_selected_recording_day_mean": (
                    sum(selected_days) / len(selected_days) if selected_days else None
                ),
                "threshold_selected_recording_day_max": (
                    max(selected_days) if selected_days else None
                ),
                "threshold_selected_recording_day_span": (
                    max(selected_days) - min(selected_days) if selected_days else None
                ),
                "mouse_behavior_eligible": selected > 0,
                "mouse_exclusion_reason": (
                    None if selected > 0 else "no_threshold_selected_sessions"
                ),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort("subject_id")


def _build_features(
    eligible: pl.DataFrame,
    mouse_major_division_coverage: pl.DataFrame | None = None,
) -> tuple[_Feature, ...]:
    mouse_ids = tuple(eligible.get_column("subject_id").to_list())
    raw_features: list[tuple[str, str, str, dict[str, fractions.Fraction]]] = []
    categorical_families = {
        "sex": "sex",
        "genotype": "genotype",
        "project_code": "project_cohort",
    }
    for column, family in categorical_families.items():
        values = eligible.get_column(column).to_list()
        for level in sorted(set(values)):
            raw_features.append(
                (
                    family,
                    f"{column}={level}",
                    "categorical_indicator",
                    {
                        mouse: fractions.Fraction(value == level)
                        for mouse, value in zip(mouse_ids, values, strict=True)
                    },
                )
            )

    numeric_specs = (
        ("recording_day", "inventory_recording_day_mean", "recording_day_mean"),
        ("recording_day", "inventory_recording_day_span", "recording_day_span"),
        (
            "recording_day",
            "threshold_selected_recording_day_mean",
            "threshold_selected_recording_day_mean",
        ),
        (
            "recording_day",
            "threshold_selected_recording_day_span",
            "threshold_selected_recording_day_span",
        ),
        ("session_count", "n_inventory_sessions", "n_inventory_sessions"),
        (
            "session_count",
            "n_threshold_selected_sessions",
            "n_threshold_selected_sessions",
        ),
    )
    rows = {row["subject_id"]: row for row in eligible.iter_rows(named=True)}
    for family, name, source in numeric_specs:
        values = {}
        for mouse in mouse_ids:
            row = rows[mouse]
            if name == "inventory_recording_day_mean":
                value = fractions.Fraction(
                    int(row["recording_day_sum"]), int(row["n_inventory_sessions"])
                )
            elif name == "threshold_selected_recording_day_mean":
                value = fractions.Fraction(
                    int(row["threshold_selected_recording_day_sum"]),
                    int(row["n_threshold_selected_sessions"]),
                )
            else:
                value = fractions.Fraction(int(row[source]))
            values[mouse] = value
        raw_features.append((family, name, "numeric_summary", values))

    if mouse_major_division_coverage is not None:
        raw_features.extend(_major_division_raw_features(eligible, mouse_major_division_coverage))

    features = []
    for family, name, feature_type, values in raw_features:
        mean, variance = _mean_and_sample_variance(values.values())
        if variance == 0:
            continue
        features.append(
            _Feature(
                family=family,
                name=name,
                feature_type=feature_type,
                values=values,
                eligible_mean=mean,
                eligible_variance=variance,
            )
        )
    family_rank = {family: index for index, family in enumerate(FAMILY_ORDER)}
    return tuple(sorted(features, key=lambda item: (family_rank[item.family], item.name)))


def _major_division_raw_features(
    eligible: pl.DataFrame,
    coverage: pl.DataFrame,
) -> list[tuple[str, str, str, dict[str, fractions.Fraction]]]:
    required = (
        "subject_id",
        "major_division_id",
        "major_division_acronym",
        "n_threshold_selected_sessions",
        "n_threshold_selected_sessions_with_division",
        "threshold_selected_session_fraction_exact",
        "n_eligible_mice_represented",
        "eligible_mouse_fraction_variance_exact",
        "nonzero_eligible_mouse_variance",
        "include_in_balance",
        "minimum_represented_mice_for_balance",
        "feature_family",
        "coverage_definition",
        "unit_population",
        "count_policy",
        "neural_activity_outcomes_used",
    )
    _require_columns(coverage, required, table="mouse major-division coverage")
    if coverage.is_empty():
        raise ValueError("mouse major-division coverage is empty")
    _validate_subject_ids(coverage, table="mouse major-division coverage")
    mouse_ids = tuple(eligible.get_column("subject_id").to_list())
    observed_mice = set(coverage.get_column("subject_id").to_list())
    if observed_mice != set(mouse_ids):
        raise ValueError(
            "mouse major-division coverage must contain exactly the behavior-eligible mice"
        )
    if coverage.select("subject_id", "major_division_id").n_unique() != coverage.height:
        raise ValueError("mouse major-division coverage must be unique by mouse/division")
    division_definitions = coverage.select("major_division_id", "major_division_acronym").unique()
    if division_definitions.height != 12:
        raise ValueError("mouse major-division coverage must contain exactly 12 divisions")
    if (
        division_definitions.get_column("major_division_id").n_unique() != 12
        or division_definitions.get_column("major_division_acronym").n_unique() != 12
    ):
        raise ValueError("mouse major-division IDs and acronyms must be one-to-one")
    expected_rows = len(mouse_ids) * 12
    if coverage.height != expected_rows:
        raise ValueError(
            "mouse major-division coverage must be a complete eligible-mouse grid: "
            f"expected={expected_rows}, observed={coverage.height}"
        )
    for column in (
        "nonzero_eligible_mouse_variance",
        "include_in_balance",
        "neural_activity_outcomes_used",
    ):
        series = coverage.get_column(column)
        if series.dtype != pl.Boolean or series.null_count():
            raise TypeError(f"mouse major-division coverage {column} must be non-null Boolean")
    if coverage.get_column("neural_activity_outcomes_used").any():
        raise ValueError("regional balance covariates must not use neural activity outcomes")
    exact_policies = {
        "feature_family": "coarse_regional_coverage",
        "coverage_definition": ("fraction_of_d03_selected_sessions_with_any_raw_located_unit"),
        "unit_population": "all_units_presence_only",
        "count_policy": "unit_count_magnitude_not_used",
    }
    for column, expected in exact_policies.items():
        if set(coverage.get_column(column).unique().to_list()) != {expected}:
            raise ValueError(f"mouse major-division coverage {column} does not match D05 contract")
    if set(coverage.get_column("minimum_represented_mice_for_balance").unique().to_list()) != {
        MINIMUM_REGION_REPRESENTED_MICE
    }:
        raise ValueError(
            "mouse major-division coverage representation threshold must equal "
            f"{MINIMUM_REGION_REPRESENTED_MICE}"
        )

    selected_counts = {
        row["subject_id"]: int(row["n_threshold_selected_sessions"])
        for row in eligible.select("subject_id", "n_threshold_selected_sessions").iter_rows(
            named=True
        )
    }
    raw_features = []
    for division in division_definitions.sort("major_division_id").iter_rows(named=True):
        identifier = division["major_division_id"]
        acronym = division["major_division_acronym"]
        rows = coverage.filter(pl.col("major_division_id") == identifier).sort("subject_id")
        values: dict[str, fractions.Fraction] = {}
        for row in rows.iter_rows(named=True):
            mouse = row["subject_id"]
            denominator = row["n_threshold_selected_sessions"]
            numerator = row["n_threshold_selected_sessions_with_division"]
            if (
                isinstance(denominator, bool)
                or not isinstance(denominator, int)
                or denominator != selected_counts[mouse]
                or denominator <= 0
            ):
                raise ValueError("regional denominator differs from D03-selected session count")
            if (
                isinstance(numerator, bool)
                or not isinstance(numerator, int)
                or numerator < 0
                or numerator > denominator
            ):
                raise ValueError("regional numerator is not a valid selected-session count")
            value = fractions.Fraction(numerator, denominator)
            if row["threshold_selected_session_fraction_exact"] != str(value):
                raise ValueError("regional exact fraction conflicts with numerator/denominator")
            values[mouse] = value
        mean, variance = _mean_and_sample_variance(values.values())
        represented = sum(value > 0 for value in values.values())
        include = represented >= MINIMUM_REGION_REPRESENTED_MICE and variance > 0
        row_represented = set(rows.get_column("n_eligible_mice_represented").to_list())
        row_variances = set(rows.get_column("eligible_mouse_fraction_variance_exact").to_list())
        row_nonzero = set(rows.get_column("nonzero_eligible_mouse_variance").to_list())
        row_include = set(rows.get_column("include_in_balance").to_list())
        if row_represented != {represented}:
            raise ValueError("regional represented-mouse diagnostic is inconsistent")
        if row_variances != {str(variance)}:
            raise ValueError("regional exact variance diagnostic is inconsistent")
        if row_nonzero != {variance > 0} or row_include != {include}:
            raise ValueError("regional balance-inclusion diagnostic is inconsistent")
        if include:
            raw_features.append(
                (
                    "coarse_regional_coverage",
                    f"major_division_selected_session_fraction={acronym}",
                    "selected_session_presence_fraction",
                    values,
                )
            )
    if not raw_features:
        raise ValueError(
            "no major division is represented in at least five eligible mice with nonzero variance"
        )
    return raw_features


def _mean_and_sample_variance(
    values: Iterable[fractions.Fraction],
) -> tuple[fractions.Fraction, fractions.Fraction]:
    values = tuple(values)
    if not values:
        raise ValueError("cannot summarize an empty feature")
    mean = sum(values, fractions.Fraction()) / len(values)
    if len(values) == 1:
        return mean, fractions.Fraction()
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance


def _score_allocation(
    features: Sequence[_Feature],
    discovery_ids: Sequence[str],
    confirmation_ids: Sequence[str],
) -> _Score:
    squared_smds: dict[str, fractions.Fraction] = {}
    family_values: dict[str, list[fractions.Fraction]] = {}
    for feature in features:
        discovery_mean = sum(feature.values[mouse] for mouse in discovery_ids) / len(discovery_ids)
        confirmation_mean = sum(feature.values[mouse] for mouse in confirmation_ids) / len(
            confirmation_ids
        )
        squared_smd = (discovery_mean - confirmation_mean) ** 2 / feature.eligible_variance
        squared_smds[feature.name] = squared_smd
        family_values.setdefault(feature.family, []).append(squared_smd)
    family_objectives = {
        family: sum(values, fractions.Fraction()) / len(values)
        for family, values in family_values.items()
    }
    objective = sum(family_objectives.values(), fractions.Fraction()) / len(family_objectives)
    return _Score(
        objective=objective,
        maximum_squared_smd=max(squared_smds.values()),
        squared_smds=squared_smds,
        family_objectives=family_objectives,
    )


def _exhaustive_confirmation_search(
    eligible_ids: Sequence[str],
    features: Sequence[_Feature],
    *,
    confirmation_mice: int,
    seed: int,
) -> tuple[
    tuple[str, ...],
    _Score,
    int,
    int,
    int,
    str,
]:
    all_ids = tuple(sorted(eligible_ids))
    best_key = None
    best_confirmation = None
    best_score = None
    best_tie_hash = None
    candidates = 0
    primary_ties = 0
    secondary_ties = 0
    minimum_objective = None
    minimum_secondary = None
    for confirmation in itertools.combinations(all_ids, confirmation_mice):
        candidates += 1
        confirmation_set = frozenset(confirmation)
        discovery = tuple(mouse for mouse in all_ids if mouse not in confirmation_set)
        score = _score_allocation(features, discovery, confirmation)
        confirmation_key = "\0".join(confirmation)
        tie_hash = hashlib.sha256(f"{seed}\0{confirmation_key}".encode()).hexdigest()
        if minimum_objective is None or score.objective < minimum_objective:
            minimum_objective = score.objective
            minimum_secondary = score.maximum_squared_smd
            primary_ties = 1
            secondary_ties = 1
        elif score.objective == minimum_objective:
            primary_ties += 1
            if minimum_secondary is None or score.maximum_squared_smd < minimum_secondary:
                minimum_secondary = score.maximum_squared_smd
                secondary_ties = 1
            elif score.maximum_squared_smd == minimum_secondary:
                secondary_ties += 1

        key = (score.objective, score.maximum_squared_smd, tie_hash, confirmation)
        if best_key is None or key < best_key:
            best_key = key
            best_confirmation = confirmation
            best_score = score
            best_tie_hash = tie_hash

    if best_confirmation is None or best_score is None or best_tie_hash is None:
        raise RuntimeError("exhaustive allocation search evaluated no candidates")
    expected_candidates = math.comb(len(all_ids), confirmation_mice)
    if candidates != expected_candidates:
        raise RuntimeError("exhaustive allocation search did not cover every candidate")
    return (
        best_confirmation,
        best_score,
        candidates,
        primary_ties,
        secondary_ties,
        best_tie_hash,
    )


def _add_mouse_assignments(
    mice: pl.DataFrame,
    allocation_lookup: Mapping[str, str],
    *,
    seed: int,
    allocation_status: str,
) -> pl.DataFrame:
    is_final = allocation_status == FINAL_ALLOCATION_STATUS
    rows = []
    for row in mice.iter_rows(named=True):
        allocation = allocation_lookup.get(row["subject_id"], "excluded")
        row = dict(row)
        row.update(
            {
                "cohort_assignment": allocation,
                "allocation_status": allocation_status,
                "split_seed": seed,
                "confirmation_access_policy": (
                    "sealed_until_one_time_confirmation_gate"
                    if allocation == "confirmation"
                    else (
                        (
                            "discovery_activated_confirmation_sealed"
                            if is_final
                            else "provisional_discovery_not_activated"
                        )
                        if allocation == "discovery"
                        else "not_applicable"
                    )
                ),
                "ready_for_neural_discovery": is_final,
                "discovery_mouse_included": is_final and allocation == "discovery",
                "confirmation_mouse_included": False,
                "neural_activity_used_for_allocation": False,
            }
        )
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None).sort("subject_id")


def _build_session_assignments(
    inventory: pl.DataFrame,
    qc: pl.DataFrame,
    mice: pl.DataFrame,
    *,
    allocation_status: str,
) -> pl.DataFrame:
    mouse_columns = (
        "subject_id",
        "mouse_behavior_eligible",
        "cohort_assignment",
        "allocation_status",
        "split_seed",
        "confirmation_access_policy",
        "ready_for_neural_discovery",
        "discovery_mouse_included",
        "confirmation_mouse_included",
        "neural_activity_used_for_allocation",
    )
    sessions = (
        inventory.join(qc, on=(SESSION_KEY, "subject_id"), how="inner", validate="1:1")
        .join(mice.select(mouse_columns), on="subject_id", how="left", validate="m:1")
        .rename({"is_good_session": "session_behavior_eligible"})
    )
    if sessions.height != inventory.height:
        raise RuntimeError("session assignment joins lost one or more inventory sessions")
    split_reasons = []
    for row in sessions.iter_rows(named=True):
        qc_reason = row["session_exclusion_reasons"]
        qc_reason = None if qc_reason is None or not str(qc_reason).strip() else str(qc_reason)
        if row["session_behavior_eligible"]:
            reason = None
        elif not row["mouse_behavior_eligible"]:
            reason = "mouse_has_no_threshold_selected_sessions"
            if qc_reason:
                reason += f"; session_qc={qc_reason}"
        else:
            reason = "session_failed_behavior_qc"
            if qc_reason:
                reason += f"; session_qc={qc_reason}"
        split_reasons.append(reason)
    is_final = allocation_status == FINAL_ALLOCATION_STATUS
    discovery_included = (
        (pl.col("cohort_assignment") == "discovery")
        & pl.col("session_behavior_eligible")
        & pl.lit(is_final)
    )
    sessions = sessions.with_columns(
        discovery_included.alias("discovery_analysis_included"),
        pl.lit(False).alias("confirmation_analysis_included"),
        pl.when(discovery_included)
        .then(pl.lit(None, dtype=pl.String))
        .otherwise(
            pl.when(pl.lit(is_final))
            .then(pl.lit("not_selected_for_discovery_analysis"))
            .otherwise(pl.lit(PROVISIONAL_ALLOCATION_STATUS))
        )
        .alias("analysis_activation_block_reason"),
        pl.Series("session_split_exclusion_reason", split_reasons, dtype=pl.String),
    )
    first = [
        SESSION_KEY,
        "subject_id",
        "cohort_assignment",
        "allocation_status",
        "session_behavior_eligible",
        "mouse_behavior_eligible",
        "discovery_analysis_included",
        "confirmation_analysis_included",
        "analysis_activation_block_reason",
        "session_split_exclusion_reason",
        "session_exclusion_reasons",
        "split_seed",
        "confirmation_access_policy",
        "ready_for_neural_discovery",
        "discovery_mouse_included",
        "confirmation_mouse_included",
        "neural_activity_used_for_allocation",
    ]
    return sessions.select(*first, *[c for c in sessions.columns if c not in first]).sort(
        "subject_id", "recording_day_number", SESSION_KEY
    )


def _build_balance_table(
    features: Sequence[_Feature],
    *,
    discovery_ids: Sequence[str],
    confirmation_ids: Sequence[str],
    score: _Score,
) -> pl.DataFrame:
    family_sizes: dict[str, int] = {}
    for feature in features:
        family_sizes[feature.family] = family_sizes.get(feature.family, 0) + 1
    n_families = len(family_sizes)
    rows = []
    for feature in features:
        discovery_mean = sum(feature.values[mouse] for mouse in discovery_ids) / len(discovery_ids)
        confirmation_mean = sum(feature.values[mouse] for mouse in confirmation_ids) / len(
            confirmation_ids
        )
        difference = discovery_mean - confirmation_mean
        standard_deviation = math.sqrt(float(feature.eligible_variance))
        squared_smd = score.squared_smds[feature.name]
        signed_smd = math.copysign(math.sqrt(float(squared_smd)), float(difference))
        feature_weight = fractions.Fraction(1, n_families * family_sizes[feature.family])
        rows.append(
            {
                "metadata_family": feature.family,
                "feature": feature.name,
                "feature_type": feature.feature_type,
                "standardizer": "eligible_mouse_sample_sd",
                "eligible_mean": float(feature.eligible_mean),
                "eligible_standard_deviation": standard_deviation,
                "discovery_mean": float(discovery_mean),
                "confirmation_mean": float(confirmation_mean),
                "standardized_difference": signed_smd,
                "absolute_standardized_difference": abs(signed_smd),
                "squared_standardized_difference": float(squared_smd),
                "squared_standardized_difference_exact": str(squared_smd),
                "family_mean_squared_standardized_difference": float(
                    score.family_objectives[feature.family]
                ),
                "feature_weight_in_objective": float(feature_weight),
                "weighted_objective_contribution": float(feature_weight * squared_smd),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _build_summary_table(
    mice: pl.DataFrame,
    sessions: pl.DataFrame,
    *,
    score: _Score,
    seed: int,
    discovery_mice: int,
    confirmation_mice: int,
    candidates_evaluated: int,
    primary_ties: int,
    secondary_ties: int,
    tie_break_sha256: str,
    n_features: int,
    n_families: int,
    allocation_status: str,
) -> pl.DataFrame:
    eligible_sessions = sessions.filter(pl.col("session_behavior_eligible"))
    cohort_counts = {
        cohort: {
            "threshold_sessions": eligible_sessions.filter(
                pl.col("cohort_assignment") == cohort
            ).height,
            "inventory_sessions": sessions.filter(pl.col("cohort_assignment") == cohort).height,
        }
        for cohort in ("discovery", "confirmation")
    }
    return pl.DataFrame(
        {
            "split_seed": [seed],
            "search_method": ["exhaustive"],
            "objective_arithmetic": ["exact_fraction_squared_smd"],
            "n_inventory_mice": [mice.height],
            "n_eligible_mice": [discovery_mice + confirmation_mice],
            "n_excluded_mice": [mice.filter(~pl.col("mouse_behavior_eligible")).height],
            "n_discovery_mice": [discovery_mice],
            "n_confirmation_mice": [confirmation_mice],
            "n_inventory_sessions": [sessions.height],
            "n_threshold_selected_sessions": [eligible_sessions.height],
            "n_discovery_inventory_sessions": [cohort_counts["discovery"]["inventory_sessions"]],
            "n_confirmation_inventory_sessions": [
                cohort_counts["confirmation"]["inventory_sessions"]
            ],
            "n_discovery_threshold_selected_sessions": [
                cohort_counts["discovery"]["threshold_sessions"]
            ],
            "n_confirmation_threshold_selected_sessions": [
                cohort_counts["confirmation"]["threshold_sessions"]
            ],
            "n_balance_families": [n_families],
            "n_balance_features": [n_features],
            "n_candidates_evaluated": [candidates_evaluated],
            "n_primary_objective_ties": [primary_ties],
            "n_secondary_objective_ties": [secondary_ties],
            "mean_family_mean_squared_smd": [float(score.objective)],
            "mean_family_mean_squared_smd_exact": [str(score.objective)],
            "root_mean_family_mean_squared_smd": [math.sqrt(float(score.objective))],
            "maximum_absolute_smd": [math.sqrt(float(score.maximum_squared_smd))],
            "maximum_squared_smd_exact": [str(score.maximum_squared_smd)],
            "selected_tie_break_sha256": [tie_break_sha256],
            "allocation_status": [allocation_status],
            "ready_for_neural_discovery": [allocation_status == FINAL_ALLOCATION_STATUS],
            "confirmation_policy": ["sealed_until_one_time_confirmation_gate"],
            "neural_activity_used_for_allocation": [False],
        }
    )


def _validate_outputs(
    mice: pl.DataFrame,
    sessions: pl.DataFrame,
    *,
    inventory: pl.DataFrame,
    discovery_mice: int,
    confirmation_mice: int,
    allocation_status: str,
    seed: int,
) -> None:
    for table_name, frame in (("mouse", mice), ("session", sessions)):
        statuses = frame.get_column("allocation_status").unique().to_list()
        if statuses != [allocation_status]:
            raise RuntimeError(
                f"{table_name} assignment status differs from requested allocation status"
            )
        if frame.get_column("neural_activity_used_for_allocation").any():
            raise RuntimeError(f"{table_name} assignment claims neural activity was used")
        if frame.get_column("split_seed").unique().to_list() != [seed]:
            raise RuntimeError(f"{table_name} assignment split seed differs from requested seed")
    expected_mouse_ids = set(inventory.get_column("subject_id").to_list())
    observed_mouse_ids = set(mice.get_column("subject_id").to_list())
    if mice.height != len(expected_mouse_ids) or observed_mouse_ids != expected_mouse_ids:
        raise RuntimeError("mouse assignment keys differ from the session inventory")
    counts = mice.group_by("cohort_assignment").len()
    count_lookup = dict(zip(counts["cohort_assignment"], counts["len"], strict=True))
    if count_lookup.get("discovery") != discovery_mice:
        raise RuntimeError("discovery mouse count differs from requested size")
    if count_lookup.get("confirmation") != confirmation_mice:
        raise RuntimeError("confirmation mouse count differs from requested size")
    expected_session_rows = inventory.height
    if sessions.height != expected_session_rows:
        raise RuntimeError(
            "session assignment row count differs from inventory: "
            f"expected={expected_session_rows}, observed={sessions.height}"
        )
    expected_sessions = set(inventory.get_column(SESSION_KEY).to_list())
    observed_sessions = set(sessions.get_column(SESSION_KEY).to_list())
    if observed_sessions != expected_sessions or len(observed_sessions) != sessions.height:
        missing = sorted(expected_sessions - observed_sessions)
        unexpected = sorted(observed_sessions - expected_sessions)
        raise RuntimeError(
            "session assignment keys differ from inventory: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    expected_pairs = set(inventory.select(SESSION_KEY, "subject_id").iter_rows())
    observed_pairs = set(sessions.select(SESSION_KEY, "subject_id").iter_rows())
    if observed_pairs != expected_pairs:
        raise RuntimeError("session assignment subject/session pairs differ from inventory")
    per_mouse = sessions.group_by("subject_id").agg(
        pl.col("cohort_assignment").n_unique().alias("n_assignments")
    )
    if per_mouse.filter(pl.col("n_assignments") != 1).height:
        raise RuntimeError("at least one mouse was split across cohorts")
    expected_discovery = (
        (pl.col("cohort_assignment") == "discovery")
        & pl.col("session_behavior_eligible")
        & pl.lit(allocation_status == FINAL_ALLOCATION_STATUS)
    )
    if sessions.filter(pl.col("discovery_analysis_included") != expected_discovery).height:
        raise RuntimeError("discovery analysis flags differ from allocation readiness")
    if sessions.get_column("confirmation_analysis_included").any():
        raise RuntimeError("allocation activated confirmation analysis rows")
    expected_ready = allocation_status == FINAL_ALLOCATION_STATUS
    if set(sessions.get_column("ready_for_neural_discovery").unique().to_list()) != {
        expected_ready
    }:
        raise RuntimeError("allocation readiness flags differ from allocation status")
    expected_discovery_mice = (pl.col("cohort_assignment") == "discovery") & pl.lit(expected_ready)
    if mice.filter(pl.col("discovery_mouse_included") != expected_discovery_mice).height:
        raise RuntimeError("discovery mouse flags differ from allocation readiness")
    if mice.get_column("confirmation_mouse_included").any():
        raise RuntimeError("allocation activated confirmation mice")
