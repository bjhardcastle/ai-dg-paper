"""Efficient access to the Dynamic Gating NWB dataset.

The helpers in this module keep DANDI asset discovery separate from NWB table
queries.  All table access returns a Polars ``LazyFrame`` so filters and column
selection can be pushed down into the remote NWB read.
"""

from __future__ import annotations

import collections.abc
import os
import re
from typing import Any, Literal

import lazynwb
import polars as pl

import dg.quality

PathLike = str | os.PathLike[str]

DANDISET_ID = "001051"
DANDISET_VERSION = "0.260825.2232"
DANDISET_DOI = "10.48324/dandi.001051/0.260825.2232"
COMPANION_REPOSITORY_COMMIT = "3c09a0dc2972381f06393ddf5c000e16462c7037"
COMPANION_TRIALS_URL = (
    "https://raw.githubusercontent.com/AllenInstitute/"
    "SHIELD_Dynamic_Gating_Analysis/"
    f"{COMPANION_REPOSITORY_COMMIT}/metadata_tables/master_stim_trials_table.csv"
)

UNITS_PATH = "/units"
TRIALS_PATH = "/intervals/trials"
FLASH_PRESENTATIONS_PATH = "/intervals/flash_250ms_presentations"
ELECTRODES_PATH = "/general/extracellular_ephys/electrodes"
TASK_PARAMETERS_PATH = "/general/task_parameters"

KNOWN_IMAGE_PRESENTATION_PATHS = (
    "/intervals/dynamic_routing_image_set_presentations",
    "/intervals/image-set-im104_r_presentations",
    "/intervals/image-set-im114_r_presentations",
    "/intervals/image-set-im005_r_presentations",
    "/intervals/image-set-im087_r_presentations",
)

BEHAVIOR_TIMESERIES_PATHS = {
    "licks": "/processing/licking/licks",
    "reward_volume": "/processing/rewards/volume",
    "autorewarded": "/processing/rewards/autorewarded",
    "running_speed": "/processing/running/speed",
    "running_speed_unfiltered": "/processing/running/speed_unfiltered",
}

DEFAULT_UNIT_COLUMNS = (
    "id",
    "peak_channel_id",
    "isi_violations",
    "amplitude_cutoff",
    "firing_rate",
    "presence_ratio",
    "quality",
    "structure_layer",
)
DEFAULT_TRIAL_COLUMNS = (
    "id",
    "start_time",
    "stop_time",
    "change_time",
    "initial_image_name",
    "change_image_name",
    "is_change",
    "is_sham_change",
    "go",
    "catch",
    "aborted",
    "auto_rewarded",
    "hit",
    "false_alarm",
    "no_reward_epoch",
    "omitted_reward",
    "response_latency",
)
OPTIONAL_TRIAL_COLUMN_DTYPES = {
    "is_sham_change": pl.Boolean,
    "no_reward_epoch": pl.Boolean,
    "omitted_reward": pl.Boolean,
}
DEFAULT_IMAGE_PRESENTATION_COLUMNS = (
    "id",
    "start_time",
    "stop_time",
    "image_name",
    "stimulus_name",
    "stimulus_block",
    "active",
    "is_change",
    "is_image_novel",
    "flashes_since_change",
    "rewarded",
    "omitted",
)
DEFAULT_FLASH_PRESENTATION_COLUMNS = (
    "id",
    "start_time",
    "stop_time",
    "stimulus_name",
    "stimulus_index",
    "stimulus_block",
    "active",
    "is_change",
    "is_image_novel",
    "contrast",
    "rewarded",
    "omitted",
)
DEFAULT_ELECTRODE_COLUMNS = (
    "id",
    "location",
    "x",
    "y",
    "z",
    "probe_id",
    "probe_channel_number",
    "probe_horizontal_position",
    "probe_vertical_position",
    "valid_data",
)

_SESSION_ASSET_PATTERN = re.compile(
    r"^sub-(?P<subject>[^/]+)/sub-(?P=subject)_ses-\d{8}T\d{6}\.nwb$"
)
_DEFAULT_UNIT_QUALITY_THRESHOLDS = dg.quality.DEFAULT_UNIT_QUALITY_THRESHOLDS


def is_session_nwb_asset(asset: collections.abc.Mapping[str, Any]) -> bool:
    """Return whether a DANDI asset is a session NWB rather than a probe LFP NWB."""

    path = asset.get("path")
    return isinstance(path, str) and _SESSION_ASSET_PATTERN.fullmatch(path) is not None


def get_session_nwb_sources(
    *,
    subject_ids: str | int | collections.abc.Iterable[str | int] | None = None,
    dandiset_id: str = DANDISET_ID,
    version: str = DANDISET_VERSION,
    max_sessions: int | None = None,
    order: Literal["path", "created", "modified", "-path", "-created", "-modified"] = "path",
) -> list[str]:
    """Resolve session-level NWB assets to remote S3 sources.

    The published Dandiset contains both one session NWB per recording and
    separate probe-level LFP NWBs.  This function returns only session files.
    ``version`` defaults to an immutable published version for reproducibility.
    """

    if max_sessions is not None and max_sessions < 1:
        raise ValueError("max_sessions must be positive or None")

    subjects = None
    if subject_ids is not None:
        subject_values = (subject_ids,) if isinstance(subject_ids, (str, int)) else subject_ids
        subjects = {str(subject).removeprefix("sub-") for subject in subject_values}
        if not subjects:
            return []

    def asset_filter(asset: dict[str, Any]) -> bool:
        if not is_session_nwb_asset(asset):
            return False
        if subjects is None:
            return True
        match = _SESSION_ASSET_PATTERN.fullmatch(asset["path"])
        return match is not None and match.group("subject") in subjects

    lazynwb.config.anon = True
    return lazynwb.get_dandi_sources(
        dandiset_id,
        version=version,
        order=order,
        asset_filter=asset_filter,
        max_assets=max_sessions,
    )


def scan_nwb_table(
    sources: PathLike | collections.abc.Iterable[PathLike],
    table_path: str,
    *,
    columns: collections.abc.Iterable[str] | None = None,
    include_provenance: bool = True,
    infer_schema_length: int | None = None,
    schema_overrides: collections.abc.Mapping[str, Any] | None = None,
) -> pl.LazyFrame:
    """Lazily scan an exact table path from one or more NWB sources."""

    if not table_path.startswith("/"):
        raise ValueError("table_path must be an absolute NWB path beginning with '/'")

    lazynwb.config.anon = True
    frame = lazynwb.scan_nwb(
        sources,
        table_path,
        raise_on_missing=True,
        infer_schema_length=infer_schema_length,
        schema_overrides=schema_overrides,
    )
    return _select_columns(frame, columns, include_provenance=include_provenance)


def scan_trials(
    sources: PathLike | collections.abc.Iterable[PathLike],
    *,
    columns: collections.abc.Iterable[str] | None = DEFAULT_TRIAL_COLUMNS,
    infer_schema_length: int | None = None,
) -> pl.LazyFrame:
    """Lazily scan trials while normalizing three schema-optional flags.

    Twenty-two published sessions omit ``is_sham_change``,
    ``no_reward_epoch``, and ``omitted_reward``. Requested missing optional
    columns are represented as typed nulls rather than causing a scan failure.
    Use :func:`add_no_reward_epoch_from_companion` before behavioral QC.
    """

    frame = scan_nwb_table(
        sources,
        TRIALS_PATH,
        columns=None,
        infer_schema_length=infer_schema_length,
        schema_overrides=OPTIONAL_TRIAL_COLUMN_DTYPES,
    )
    if columns is None:
        return frame

    selected_columns = list(columns)
    schema_names = set(frame.collect_schema().names())
    missing_required = (
        set(selected_columns).difference(schema_names).difference(OPTIONAL_TRIAL_COLUMN_DTYPES)
    )
    if missing_required:
        raise ValueError(f"trials table is missing required columns: {sorted(missing_required)}")
    missing_optional = (
        set(selected_columns).difference(schema_names).intersection(OPTIONAL_TRIAL_COLUMN_DTYPES)
    )
    if missing_optional:
        frame = frame.with_columns(
            *(
                pl.lit(None, dtype=OPTIONAL_TRIAL_COLUMN_DTYPES[column]).alias(column)
                for column in sorted(missing_optional)
            )
        )
    return _select_columns(frame, selected_columns, include_provenance=True)


def scan_units(
    sources: PathLike | collections.abc.Iterable[PathLike],
    *,
    columns: collections.abc.Iterable[str] | None = DEFAULT_UNIT_COLUMNS,
    include_spike_times: bool = False,
    well_isolated: bool = False,
    thresholds: dg.quality.UnitQualityThresholds = _DEFAULT_UNIT_QUALITY_THRESHOLDS,
    infer_schema_length: int | None = None,
) -> pl.LazyFrame:
    """Lazily scan units with optional pushdown of isolation-quality filters."""

    lazynwb.config.anon = True
    frame = lazynwb.scan_nwb(
        sources,
        UNITS_PATH,
        raise_on_missing=True,
        infer_schema_length=infer_schema_length,
    )
    if well_isolated:
        frame = dg.quality.filter_well_isolated_units(frame, thresholds=thresholds)

    if columns is None:
        selected_columns = None
        if not include_spike_times:
            frame = frame.select(pl.exclude("spike_times"))
    else:
        selected_columns = list(columns)
        if include_spike_times:
            selected_columns.append("spike_times")
    return _select_columns(frame, selected_columns, include_provenance=True)


def scan_image_presentations(
    sources: PathLike | collections.abc.Iterable[PathLike],
    *,
    columns: collections.abc.Iterable[str] | None = DEFAULT_IMAGE_PRESENTATION_COLUMNS,
    table_paths: collections.abc.Mapping[str, str] | None = None,
    infer_schema_length: int | None = None,
) -> pl.LazyFrame:
    """Scan task images after discovering each session's variable table path."""

    source_list = _normalize_sources(sources)
    if not source_list:
        raise ValueError("at least one NWB source is required")
    if table_paths is None:
        table_paths = discover_image_presentation_paths(source_list)

    frames = []
    for source in source_list:
        source_key = str(source)
        if source_key not in table_paths:
            raise ValueError(f"no image-presentation table path for {source_key!r}")
        frames.append(
            scan_nwb_table(
                source,
                table_paths[source_key],
                columns=columns,
                infer_schema_length=infer_schema_length,
            )
        )
    if len(frames) == 1:
        return frames[0]
    return pl.concat(frames, how="diagonal_relaxed")


def discover_image_presentation_paths(
    sources: PathLike | collections.abc.Iterable[PathLike],
) -> dict[str, str]:
    """Find the task-image presentation table in every session NWB.

    The published files use five different internal table names. Discovery is
    based on the required task-image columns rather than on a single name, and
    explicitly excludes passive flash, gabor, and spontaneous tables.
    """

    lazynwb.config.anon = True
    result: dict[str, str] = {}
    required_children = {"image_name", "is_change", "active"}
    excluded_terms = {"flash", "gabor", "spontaneous"}
    for source in _normalize_sources(sources):
        internal_paths = lazynwb.get_internal_paths(
            source,
            include_table_columns=True,
        )
        children_by_parent: dict[str, set[str]] = {}
        for internal_path in internal_paths:
            if not internal_path.startswith("/intervals/") or "/" not in internal_path[1:]:
                continue
            parent, child = internal_path.rsplit("/", maxsplit=1)
            children_by_parent.setdefault(parent, set()).add(child)
        candidates = [
            parent
            for parent, children in children_by_parent.items()
            if parent.endswith("_presentations")
            and required_children.issubset(children)
            and not any(term in parent.lower() for term in excluded_terms)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"expected one task-image presentation table in {str(source)!r}; found {candidates}"
            )
        result[str(source)] = candidates[0]
    return result


def add_image_name_factors(
    presentations: pl.DataFrame | pl.LazyFrame,
    *,
    image_name_column: str = "image_name",
) -> pl.DataFrame | pl.LazyFrame:
    """Parse image identity and relative contrast from names such as ``im115_r-0.7``."""

    image_name = pl.col(image_name_column)
    return presentations.with_columns(
        image_name.str.extract(r"^(im\d+)_r-", group_index=1).alias("image_id"),
        image_name.str.extract(r"_r-([0-9]+(?:\.[0-9]+)?)$", group_index=1)
        .cast(pl.Float64, strict=False)
        .alias("image_relative_contrast"),
    )


def add_physical_image_change_flag(
    trials: pl.DataFrame | pl.LazyFrame,
    *,
    initial_image_column: str = "initial_image_name",
    change_image_column: str = "change_image_name",
) -> pl.DataFrame | pl.LazyFrame:
    """Label physical identity changes without conflating go and catch trials.

    In this dataset, the NWB ``is_change`` flag can be true for both rewarded
    go changes and unrewarded catch/sham events. This helper compares the image
    identities directly and leaves the task's ``go``, ``catch``, and
    ``is_sham_change`` labels intact as separate experimental factors.
    """

    initial_image = pl.col(initial_image_column)
    change_image = pl.col(change_image_column)
    return trials.with_columns(
        pl.when(initial_image.is_null() | change_image.is_null())
        .then(pl.lit(None, dtype=pl.Boolean))
        .otherwise(initial_image != change_image)
        .alias("physical_image_change")
    )


def scan_flash_presentations(
    sources: PathLike | collections.abc.Iterable[PathLike],
    *,
    columns: collections.abc.Iterable[str] | None = DEFAULT_FLASH_PRESENTATION_COLUMNS,
    infer_schema_length: int | None = None,
) -> pl.LazyFrame:
    """Scan the later passive 250-ms flash block.

    This table's numeric ``contrast`` is not the task's image-contrast
    perturbation; task contrast is encoded in the behavior ``image_name``.
    """

    return scan_nwb_table(
        sources,
        FLASH_PRESENTATIONS_PATH,
        columns=columns,
        infer_schema_length=infer_schema_length,
    )


def scan_electrodes(
    sources: PathLike | collections.abc.Iterable[PathLike],
    *,
    columns: collections.abc.Iterable[str] | None = DEFAULT_ELECTRODE_COLUMNS,
    infer_schema_length: int | None = None,
) -> pl.LazyFrame:
    """Scan electrodes carrying probe and CCF location metadata."""

    return scan_nwb_table(
        sources,
        ELECTRODES_PATH,
        columns=columns,
        infer_schema_length=infer_schema_length,
    )


def join_unit_locations(
    units: pl.LazyFrame,
    electrodes: pl.LazyFrame,
) -> pl.LazyFrame:
    """Join units to their peak-channel CCF locations within each NWB session."""

    source = lazynwb.NWB_PATH_COLUMN_NAME
    locations = electrodes.select(
        source,
        pl.col(lazynwb.TABLE_PATH_COLUMN_NAME).alias("electrode_table_path"),
        pl.col(lazynwb.TABLE_INDEX_COLUMN_NAME).alias("electrode_table_index"),
        pl.col("id").alias("peak_channel_id"),
        pl.col("location").alias("structure_acronym"),
        pl.col("x").alias("anterior_posterior_ccf_coordinate"),
        pl.col("y").alias("dorsal_ventral_ccf_coordinate"),
        pl.col("z").alias("left_right_ccf_coordinate"),
        "probe_id",
        "probe_channel_number",
        "probe_horizontal_position",
        "probe_vertical_position",
        "valid_data",
    )
    return units.join(
        locations,
        on=[source, "peak_channel_id"],
        how="left",
        validate="m:1",
    )


def get_session_metadata(
    sources: PathLike | collections.abc.Iterable[PathLike],
) -> pl.DataFrame:
    """Read cheap session/subject metadata for session NWB sources."""

    lazynwb.config.anon = True
    metadata = lazynwb.get_metadata_df(_normalize_sources(sources), as_polars=True)
    if not isinstance(metadata, pl.DataFrame):
        raise TypeError("lazynwb.get_metadata_df did not return a Polars DataFrame")
    if "identifier" not in metadata.columns:
        raise ValueError("session metadata is missing the NWB identifier")

    identifier = pl.col("identifier").cast(pl.Int64, strict=False)
    session_id = (
        pl.col("session_id").cast(pl.Int64, strict=False)
        if "session_id" in metadata.columns
        else pl.lit(None, dtype=pl.Int64)
    )
    return metadata.with_columns(
        identifier.alias("ecephys_session_id"),
        identifier.is_not_null().alias("ecephys_session_id_valid"),
        (identifier.is_not_null() & session_id.is_not_null() & (identifier != session_id))
        .fill_null(False)
        .alias("session_identifier_conflict"),
    )


def scan_companion_trial_reward_epochs(
    source: PathLike = COMPANION_TRIALS_URL,
) -> pl.LazyFrame:
    """Scan the authors' pinned trial-to-reward-epoch fallback table.

    The source has one row per stimulus presentation, so this function reduces
    it to one auditable row per ``(session_id, trials_id)`` and flags any
    within-trial disagreement rather than silently selecting a value.
    """

    return (
        pl.scan_csv(source)
        .with_row_index("_companion_csv_row")
        .select(
            "_companion_csv_row",
            pl.col("session_id").cast(pl.Int64),
            pl.col("trials_id").cast(pl.Int64),
            pl.col("no_reward_epoch").cast(pl.Boolean, strict=False),
            pl.lit(str(source)).alias("companion_source"),
        )
        .filter(pl.col("trials_id") >= 0)
        .group_by("session_id", "trials_id")
        .agg(
            pl.col("no_reward_epoch").drop_nulls().first().alias("companion_no_reward_epoch"),
            pl.col("no_reward_epoch")
            .drop_nulls()
            .n_unique()
            .alias("n_companion_reward_epoch_values"),
            pl.col("_companion_csv_row").min().alias("companion_csv_first_row"),
            pl.col("_companion_csv_row").max().alias("companion_csv_last_row"),
            pl.len().alias("n_companion_csv_rows"),
            pl.col("companion_source").first(),
        )
        .with_columns(
            (pl.col("n_companion_reward_epoch_values") > 1).alias("companion_reward_epoch_conflict")
        )
    )


def add_no_reward_epoch_from_companion(
    trials: pl.DataFrame | pl.LazyFrame,
    session_metadata: pl.DataFrame | pl.LazyFrame,
    companion_epochs: pl.DataFrame | pl.LazyFrame,
    *,
    source_column: str = lazynwb.NWB_PATH_COLUMN_NAME,
) -> pl.LazyFrame:
    """Fill missing NWB reward-epoch flags from the pinned companion table.

    Existing NWB values remain authoritative. The output records the source of
    every flag and explicitly marks disagreements or unresolved rows.
    """

    frame = trials.lazy() if isinstance(trials, pl.DataFrame) else trials
    metadata = (
        session_metadata.lazy() if isinstance(session_metadata, pl.DataFrame) else session_metadata
    )
    companion = (
        companion_epochs.lazy() if isinstance(companion_epochs, pl.DataFrame) else companion_epochs
    )
    metadata_columns = [source_column, "ecephys_session_id"]
    metadata_schema_names = set(metadata.collect_schema().names())
    metadata_columns.extend(
        column
        for column in ("ecephys_session_id_valid", "session_identifier_conflict")
        if column in metadata_schema_names
    )

    if "no_reward_epoch" in frame.collect_schema().names():
        frame = frame.rename({"no_reward_epoch": "nwb_no_reward_epoch"})
    else:
        frame = frame.with_columns(pl.lit(None, dtype=pl.Boolean).alias("nwb_no_reward_epoch"))

    frame = frame.join(
        metadata.select(*metadata_columns),
        on=source_column,
        how="left",
        validate="m:1",
    ).join(
        companion,
        left_on=["ecephys_session_id", "id"],
        right_on=["session_id", "trials_id"],
        how="left",
        validate="m:1",
    )
    both_present = (
        pl.col("nwb_no_reward_epoch").is_not_null()
        & pl.col("companion_no_reward_epoch").is_not_null()
    )
    return frame.with_columns(
        pl.coalesce("nwb_no_reward_epoch", "companion_no_reward_epoch").alias("no_reward_epoch"),
        (both_present & (pl.col("nwb_no_reward_epoch") != pl.col("companion_no_reward_epoch")))
        .fill_null(False)
        .alias("no_reward_epoch_conflict"),
        pl.when(pl.col("nwb_no_reward_epoch").is_not_null())
        .then(pl.lit("nwb"))
        .when(pl.col("companion_no_reward_epoch").is_not_null())
        .then(pl.lit("companion"))
        .otherwise(pl.lit("missing"))
        .alias("no_reward_epoch_source"),
    )


def scan_trials_with_reward_epochs(
    sources: PathLike | collections.abc.Iterable[PathLike],
    *,
    companion_source: PathLike = COMPANION_TRIALS_URL,
    infer_schema_length: int | None = None,
) -> pl.LazyFrame:
    """Scan trials and fill the 22 sessions missing NWB reward-epoch flags."""

    source_list = _normalize_sources(sources)
    trials = scan_trials(source_list, infer_schema_length=infer_schema_length)
    metadata = get_session_metadata(source_list)
    companion = scan_companion_trial_reward_epochs(companion_source)
    return add_no_reward_epoch_from_companion(trials, metadata, companion)


def get_task_parameters(source: PathLike) -> dict[str, Any]:
    """Read task attributes such as response window and programmed NR times.

    Programmed ``no_reward`` times are not exact trial boundaries and must not
    replace trial-level ``no_reward_epoch`` values in confirmatory analyses.
    """

    lazynwb.config.anon = True
    return lazynwb.get_attrs(source, TASK_PARAMETERS_PATH)


def get_behavior_timeseries(
    source: PathLike,
    name: Literal[
        "licks",
        "reward_volume",
        "autorewarded",
        "running_speed",
        "running_speed_unfiltered",
    ],
) -> Any:
    """Return a lazy behavior TimeSeries from one session NWB source."""

    lazynwb.config.anon = True
    return lazynwb.get_timeseries(
        source,
        BEHAVIOR_TIMESERIES_PATHS[name],
        exact_path=True,
    )


def _select_columns(
    frame: pl.LazyFrame,
    columns: collections.abc.Iterable[str] | None,
    *,
    include_provenance: bool,
) -> pl.LazyFrame:
    if columns is None:
        if include_provenance:
            return frame
        return frame.select(
            pl.exclude(
                lazynwb.NWB_PATH_COLUMN_NAME,
                lazynwb.TABLE_PATH_COLUMN_NAME,
                lazynwb.TABLE_INDEX_COLUMN_NAME,
            )
        )

    selected = list(columns)
    if include_provenance:
        selected.extend(
            [
                lazynwb.NWB_PATH_COLUMN_NAME,
                lazynwb.TABLE_PATH_COLUMN_NAME,
                lazynwb.TABLE_INDEX_COLUMN_NAME,
            ]
        )
    return frame.select(*dict.fromkeys(selected))


def _normalize_sources(
    sources: PathLike | collections.abc.Iterable[PathLike],
) -> list[PathLike]:
    if isinstance(sources, (str, os.PathLike)):
        return [sources]
    return list(sources)
