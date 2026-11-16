from multiprocessing import get_context
from pathlib import Path

import pytest

from riichienv.kar.contracts import content_hash


def _replay_source(paths):
    from riichienv.kar.audit import replay_game  # noqa: PLC0415
    from riichienv.kar.mjsoul import decode_record_bytes, load_protocol  # noqa: PLC0415

    source, proto = paths
    game = decode_record_bytes(Path(source).read_bytes(), load_protocol(Path(proto)), container="majsoul_download")
    return {"game_id": game.game_id, "source_sha256": game.source_sha256, "replay": replay_game(game)}


def test_single_and_two_process_replays_are_byte_deterministic():
    assets = Path(__file__).parents[3] / "kar-m0-assets"
    root = assets / "jade-probe-20261115"
    sources = [
        root / "200331-1117d4b9-2b59-4ef0-88d0-b1da573e4412",
        root / "200401-005d7f6e-37f9-4402-ad80-0e48d9254601",
    ]
    if not all(source.is_file() for source in sources):
        pytest.skip("Private raw sources are not distributed with the code")
    arguments = [(str(source), str(assets / "mahjongsoul.proto")) for source in sources]
    serial = [_replay_source(pair) for pair in arguments]
    with get_context("spawn").Pool(processes=2) as workers:
        parallel = workers.map(_replay_source, arguments)
    assert all(row["replay"]["status"] == "PASS" for row in serial)
    assert content_hash(serial) == content_hash(parallel)
    assert [row["replay"]["input_encoding"]["prefix_sha256"] for row in serial] == [
        row["replay"]["input_encoding"]["prefix_sha256"] for row in parallel
    ]
