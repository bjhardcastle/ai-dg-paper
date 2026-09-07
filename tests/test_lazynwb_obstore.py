"""Tests for the lazynwb custom-reader bridge."""

import struct
import types

import dg.lazynwb_obstore


class _FakeRangeReader:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def read_range(self, start, *, length):
        self.calls.append((start, length))
        return self.payload


class _FakeScanner:
    async def bootstrap(self):
        return types.SimpleNamespace(offset_size=8)

    async def _read_global_heap_string(self, reference):
        return {1: "good", 2: "noise"}[reference.index]


class _FakeReader:
    def __init__(self, payload):
        self._range_reader = _FakeRangeReader(payload)
        self.scanner = _FakeScanner()
        self.closed = False

    async def get_source_identity(self):
        return types.SimpleNamespace(content_length=10_000)

    async def read_table_schema_snapshot(self, table_path):
        assert table_path == "units"
        dataset = types.SimpleNamespace(
            read_capabilities=("metadata", "direct_contiguous"),
            hdf5_data_offset=400,
            hdf5_storage_size=32,
        )
        column = types.SimpleNamespace(
            name="quality",
            dtype=types.SimpleNamespace(kind="vlen_string"),
            dataset=dataset,
            shape=(2,),
            ndim=1,
        )
        return types.SimpleNamespace(columns=(column,))

    def _get_scanner(self, content_length):
        assert content_length == 10_000
        return self.scanner

    async def close(self):
        self.closed = True


def test_vlen_string_column_uses_range_reader_and_preserves_requested_order(monkeypatch) -> None:
    payload = b"".join(
        (
            struct.pack("<IQI", 4, 1_000, 1),
            struct.pack("<IQI", 5, 1_000, 2),
        )
    )
    reader = _FakeReader(payload)
    monkeypatch.setattr(
        dg.lazynwb_obstore.lazynwb._hdf5.reader,
        "_default_hdf5_backend_reader",
        lambda source: reader,
    )

    result = dg.lazynwb_obstore.read_vlen_string_column(
        "https://example.test/session.nwb",
        "/units",
        "quality",
        row_indices=[1, 0],
    )

    assert result.get_column("_table_index").to_list() == [1, 0]
    assert result.get_column("quality").to_list() == ["noise", "good"]
    assert result.get_column("_nwb_path").to_list() == [
        "https://example.test/session.nwb",
        "https://example.test/session.nwb",
    ]
    assert reader._range_reader.calls == [(400, 32)]
    assert reader.closed


def test_vlen_string_column_rejects_invalid_rows_before_reading() -> None:
    for rows, error_type in (([-1], ValueError), ([True], TypeError), ([1.5], TypeError)):
        try:
            dg.lazynwb_obstore.read_vlen_string_column(
                "unused",
                "/units",
                "quality",
                row_indices=rows,
            )
        except error_type:
            pass
        else:
            raise AssertionError(f"row_indices={rows!r} did not raise {error_type.__name__}")


def test_indexed_numeric_mapper_reuses_reader_and_bounds_batches(monkeypatch) -> None:
    data_column = types.SimpleNamespace(
        name="spike_times",
        index_column_name="spike_times_index",
        shape=(8,),
    )
    index_column = types.SimpleNamespace(name="spike_times_index")

    class FakeIndexedReader:
        def __init__(self):
            self._range_reader = object()
            self.closed = False

        async def read_table_schema_snapshot(self, table_path):
            assert table_path == "units"
            return types.SimpleNamespace(columns=(data_column, index_column))

        async def close(self):
            self.closed = True

    reader = FakeIndexedReader()
    indexed_plan = types.SimpleNamespace(data_column=data_column)
    monkeypatch.setattr(
        dg.lazynwb_obstore.lazynwb._hdf5.reader,
        "_default_hdf5_backend_reader",
        lambda source: reader,
    )
    monkeypatch.setattr(
        dg.lazynwb_obstore.lazynwb.tables,
        "_plan_direct_hdf5_table_reads",
        lambda columns: types.SimpleNamespace(
            indexed_columns=(indexed_plan,),
            fallback_columns=(),
        ),
    )
    read_batches = []

    async def fake_read(range_reader, plan, rows):
        assert range_reader is reader._range_reader
        assert plan is indexed_plan
        read_batches.append(tuple(rows))
        return [[float(row), float(row) + 0.5] for row in rows]

    monkeypatch.setattr(
        dg.lazynwb_obstore.lazynwb.tables,
        "_read_direct_hdf5_indexed_column",
        fake_read,
    )
    callbacks = []

    dg.lazynwb_obstore.map_indexed_numeric_column_batches(
        "https://example.test/session.nwb",
        "/units",
        "spike_times",
        [1, 3, 4, 7],
        lambda rows, values: callbacks.append((rows, values)),
        batch_size=2,
    )

    assert read_batches == [(1, 3), (4, 7)]
    assert callbacks == [
        ((1, 3), [[1.0, 1.5], [3.0, 3.5]]),
        ((4, 7), [[4.0, 4.5], [7.0, 7.5]]),
    ]
    assert reader.closed
