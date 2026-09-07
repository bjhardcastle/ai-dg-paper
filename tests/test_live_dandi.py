"""Optional read-only integration tests against immutable DANDI assets."""

import importlib.metadata
import os

import polars as pl
import pytest

import dg.audit
import dg.data

EXPLICIT_FLAG_SOURCE = (
    "https://dandiarchive.s3.amazonaws.com/blobs/585/411/5854110b-2b31-4bbf-881e-3280f7b87b95"
)
MISSING_FLAG_SOURCE = (
    "https://dandiarchive.s3.amazonaws.com/blobs/324/6c5/3246c508-0142-4e96-b1f4-13ab63ee636e"
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("DG_RUN_LIVE_DANDI_TESTS") != "1",
        reason="set DG_RUN_LIVE_DANDI_TESTS=1 to access immutable DANDI assets",
    ),
]


def test_exact_dev8_combines_both_published_trial_schemas() -> None:
    """Ensure heterogeneous files retain explicit and missing reward flags."""

    assert importlib.metadata.version("lazynwb") == "1.0.0.dev8"

    counts = (
        dg.data.scan_trials(
            [EXPLICIT_FLAG_SOURCE, MISSING_FLAG_SOURCE],
            infer_schema_length=32,
        )
        .group_by("_nwb_path")
        .agg(
            pl.len().alias("n_trials"),
            pl.col("no_reward_epoch").null_count().alias("n_missing_reward_flags"),
        )
        .collect()
    )
    by_source = {row["_nwb_path"]: row for row in counts.iter_rows(named=True)}

    assert by_source[EXPLICIT_FLAG_SOURCE]["n_trials"] == 798
    assert by_source[EXPLICIT_FLAG_SOURCE]["n_missing_reward_flags"] == 0
    assert by_source[MISSING_FLAG_SOURCE]["n_trials"] == 643
    assert by_source[MISSING_FLAG_SOURCE]["n_missing_reward_flags"] == 643


def test_published_unit_filter_includes_author_good_label() -> None:
    """Check the complete predicate against a pinned published units table."""

    units = dg.data.scan_units(
        EXPLICIT_FLAG_SOURCE,
        columns=("id", "isi_violations", "amplitude_cutoff", "quality"),
        well_isolated=True,
    ).collect()

    assert units.height > 0
    assert units.get_column("quality").unique().to_list() == ["good"]
    assert units.get_column("isi_violations").max() < 0.5
    assert units.get_column("amplitude_cutoff").max() < 0.1


def test_published_electrode_columns_pass_schema_audit() -> None:
    """Guard the direct schema path needed for tables below NWB ``/general``."""

    session_manifest = pl.DataFrame(
        {
            "asset_id": ["c8015581-fc08-4a4f-a39a-ea3910b0d425"],
            "path": ["sub-604914/sub-604914_ses-20220427T041046.nwb"],
            "s3_url": [EXPLICIT_FLAG_SOURCE],
            "subject_id": ["604914"],
        }
    )

    audit = dg.audit.audit_nwb_schemas(session_manifest, max_workers=1)
    electrodes = audit.filter(pl.col("table_path") == dg.data.ELECTRODES_PATH)

    assert set(electrodes.get_column("column_name")) == set(
        dg.audit.REQUIRED_TABLE_COLUMNS[dg.data.ELECTRODES_PATH]
    )
    assert electrodes.get_column("present").all()
    assert electrodes.get_column("audit_error").null_count() == electrodes.height
