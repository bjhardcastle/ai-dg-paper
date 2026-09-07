"""Regression tests using identifiers and metrics from DANDI:001051.

The values below come from published version 0.260825.2232; no simulated
neuroscience observations are used.
"""

import polars as pl

import dg.data

PUBLIC_SESSION_SOURCE = (
    "https://dandiarchive.s3.amazonaws.com/blobs/585/411/5854110b-2b31-4bbf-881e-3280f7b87b95"
)


def test_session_asset_filter_excludes_probe_lfp_files() -> None:
    session = {
        "path": "sub-604914/sub-604914_ses-20220427T041046.nwb",
    }
    probe_lfp = {
        "path": "sub-607660/sub-607660_ses-None_probe-10_ecephys.nwb",
    }

    assert dg.data.is_session_nwb_asset(session)
    assert not dg.data.is_session_nwb_asset(probe_lfp)


def test_source_resolution_pins_version_and_filters_subject(monkeypatch) -> None:
    assets = [
        {"path": "sub-604914/sub-604914_ses-20220427T041046.nwb"},
        {"path": "sub-607660/sub-607660_ses-20220607T212534.nwb"},
        {"path": "sub-607660/sub-607660_ses-None_probe-10_ecephys.nwb"},
    ]
    call = {}

    def fake_get_dandi_sources(dandiset_id, **kwargs):
        call["dandiset_id"] = dandiset_id
        call.update(kwargs)
        return [asset["path"] for asset in assets if kwargs["asset_filter"](asset)][
            : kwargs["max_assets"]
        ]

    monkeypatch.setattr(
        dg.data.lazynwb,
        "get_dandi_sources",
        fake_get_dandi_sources,
    )

    sources = dg.data.get_session_nwb_sources(
        subject_ids=["sub-607660"],
        max_sessions=1,
    )

    assert call["dandiset_id"] == "001051"
    assert call["version"] == "0.260825.2232"
    assert sources == ["sub-607660/sub-607660_ses-20220607T212534.nwb"]


def test_scan_units_pushes_isolation_filter_before_selection(monkeypatch) -> None:
    # First three units in the published sub-604914 session NWB.
    actual_units = pl.DataFrame(
        {
            "id": [219487, 219488, 219489],
            "isi_violations": [
                4.042293115676158,
                0.5289306117865576,
                0.213029325911995,
            ],
            "amplitude_cutoff": [0.5, 0.5, 0.0464355489869127],
            "_nwb_path": [PUBLIC_SESSION_SOURCE] * 3,
            "_table_path": ["/units"] * 3,
            "_table_index": [0, 1, 2],
        }
    )

    def fake_scan_nwb(*args, **kwargs):
        return actual_units.lazy()

    monkeypatch.setattr(dg.data.lazynwb, "scan_nwb", fake_scan_nwb)

    result = dg.data.scan_units(
        PUBLIC_SESSION_SOURCE,
        columns=("id",),
        well_isolated=True,
    ).collect()

    assert result.get_column("id").to_list() == [219489]
    assert result.columns == [
        "id",
        "_nwb_path",
        "_table_path",
        "_table_index",
    ]


def test_table_paths_must_be_absolute() -> None:
    try:
        dg.data.scan_nwb_table(PUBLIC_SESSION_SOURCE, "units")
    except ValueError as error:
        assert "absolute NWB path" in str(error)
    else:
        raise AssertionError("relative NWB table path did not raise")


def test_missing_optional_trial_columns_become_auditable_nulls(monkeypatch) -> None:
    # Session 1173189336 uses id 130 at this table position; the missing-column
    # shape mirrors the 22 older published NWBs without reward-epoch flags.
    trials = pl.DataFrame(
        {
            "id": [130],
            "_nwb_path": [PUBLIC_SESSION_SOURCE],
            "_table_path": ["/intervals/trials"],
            "_table_index": [130],
        }
    )
    call = {}

    def fake_scan_nwb(*args, **kwargs):
        call.update(kwargs)
        return trials.lazy()

    monkeypatch.setattr(dg.data.lazynwb, "scan_nwb", fake_scan_nwb)

    result = dg.data.scan_trials(
        PUBLIC_SESSION_SOURCE,
        columns=("id", "no_reward_epoch"),
    ).collect()

    assert result.schema["no_reward_epoch"] == pl.Boolean
    assert result.get_column("no_reward_epoch").null_count() == 1
    assert call["schema_overrides"] == dg.data.OPTIONAL_TRIAL_COLUMN_DTYPES


def test_reward_epoch_fallback_preserves_nwb_authority_and_audits_conflicts() -> None:
    source = PUBLIC_SESSION_SOURCE
    trials = pl.DataFrame(
        {
            "_nwb_path": [source, source, source],
            "id": [294, 298, 299],
            "no_reward_epoch": [None, True, False],
        },
        schema_overrides={"no_reward_epoch": pl.Boolean},
    )
    metadata = pl.DataFrame(
        {
            "_nwb_path": [source],
            "ecephys_session_id": [1173189336],
            "ecephys_session_id_valid": [True],
            "session_identifier_conflict": [False],
        }
    )
    companion = pl.DataFrame(
        {
            "session_id": [1173189336] * 3,
            "trials_id": [294, 298, 299],
            "companion_no_reward_epoch": [True, True, True],
            "companion_reward_epoch_conflict": [False, False, False],
        }
    )

    result = dg.data.add_no_reward_epoch_from_companion(
        trials,
        metadata,
        companion,
    ).collect()

    assert result.get_column("no_reward_epoch").to_list() == [True, True, False]
    assert result.get_column("no_reward_epoch_source").to_list() == [
        "companion",
        "nwb",
        "nwb",
    ]
    assert result.get_column("no_reward_epoch_conflict").to_list() == [False, False, True]


def test_physical_change_and_image_contrast_are_parsed_from_names() -> None:
    presentations = pl.DataFrame({"image_name": ["im115_r-0.7", "im115_r-1.0", "omitted"]})
    parsed = dg.data.add_image_name_factors(presentations)

    assert parsed.get_column("image_id").to_list() == ["im115", "im115", None]
    assert parsed.get_column("image_relative_contrast").to_list() == [0.7, 1.0, None]

    trials = pl.DataFrame(
        {
            "initial_image_name": ["im115_r-1.0", "im115_r-1.0", None],
            "change_image_name": ["im012_r-1.0", "im115_r-1.0", "im012_r-1.0"],
        }
    )
    labeled = dg.data.add_physical_image_change_flag(trials)
    assert labeled.get_column("physical_image_change").to_list() == [True, False, None]


def test_session_identifier_is_authoritative_and_conflicts_are_reported(monkeypatch) -> None:
    metadata = pl.DataFrame(
        {
            "_nwb_path": [PUBLIC_SESSION_SOURCE],
            "identifier": ["1173189336"],
            "session_id": [999],
        }
    )
    monkeypatch.setattr(
        dg.data.lazynwb,
        "get_metadata_df",
        lambda *args, **kwargs: metadata,
    )

    result = dg.data.get_session_metadata(PUBLIC_SESSION_SOURCE)

    assert result.get_column("ecephys_session_id").item() == 1173189336
    assert result.get_column("ecephys_session_id_valid").item()
    assert result.get_column("session_identifier_conflict").item()


def test_unit_location_join_keeps_both_table_rows_traceable() -> None:
    units = pl.DataFrame(
        {
            "_nwb_path": [PUBLIC_SESSION_SOURCE],
            "_table_path": ["units"],
            "_table_index": [2],
            "id": [219489],
            "peak_channel_id": [62],
        }
    ).lazy()
    electrodes = pl.DataFrame(
        {
            "_nwb_path": [PUBLIC_SESSION_SOURCE],
            "_table_path": ["general/extracellular_ephys/electrodes"],
            "_table_index": [62],
            "id": [62],
            "location": ["LSc"],
            "x": [7599.0],
            "y": [2299.0],
            "z": [1099.0],
            "probe_id": [1122200385],
            "probe_channel_number": [62],
            "probe_horizontal_position": [27],
            "probe_vertical_position": [300],
            "valid_data": [True],
        }
    ).lazy()

    result = dg.data.join_unit_locations(units, electrodes).collect()

    assert result.get_column("_table_index").item() == 2
    assert result.get_column("electrode_table_index").item() == 62
    assert result.get_column("electrode_table_path").item().endswith("/electrodes")
    assert result.get_column("structure_acronym").item() == "LSc"


def test_all_unit_columns_can_exclude_large_spike_arrays(monkeypatch) -> None:
    units = pl.DataFrame(
        {
            "id": [219489],
            "spike_times": [[347.9157456367911, 348.56824274737454]],
            "_nwb_path": [PUBLIC_SESSION_SOURCE],
            "_table_path": ["units"],
            "_table_index": [2],
        }
    )
    monkeypatch.setattr(dg.data.lazynwb, "scan_nwb", lambda *args, **kwargs: units.lazy())

    without_spikes = dg.data.scan_units(
        PUBLIC_SESSION_SOURCE,
        columns=None,
        include_spike_times=False,
    ).collect()
    with_spikes = dg.data.scan_units(
        PUBLIC_SESSION_SOURCE,
        columns=None,
        include_spike_times=True,
    ).collect()

    assert "spike_times" not in without_spikes.columns
    assert "spike_times" in with_spikes.columns


def test_generic_all_column_scan_can_drop_provenance(monkeypatch) -> None:
    table = pl.DataFrame(
        {
            "id": [130],
            "_nwb_path": [PUBLIC_SESSION_SOURCE],
            "_table_path": ["intervals/trials"],
            "_table_index": [130],
        }
    )
    monkeypatch.setattr(dg.data.lazynwb, "scan_nwb", lambda *args, **kwargs: table.lazy())

    result = dg.data.scan_nwb_table(
        PUBLIC_SESSION_SOURCE,
        dg.data.TRIALS_PATH,
        columns=None,
        include_provenance=False,
    ).collect()

    assert result.columns == ["id"]
