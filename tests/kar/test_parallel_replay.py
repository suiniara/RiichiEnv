import hashlib
import json
from itertools import groupby, islice
from multiprocessing import get_context
from pathlib import Path

import pytest

from riichienv.kar.contracts import content_hash


def _replay_source(paths):
    from riichienv.kar.audit import replay_game  # noqa: PLC0415
    from riichienv.kar.mjsoul import decode_record_bytes, load_protocol  # noqa: PLC0415

    source, proto = paths
    game = decode_record_bytes(Path(source).read_bytes(), load_protocol(Path(proto)), container="majsoul_download")
    decisions, early_prefixes = [], []

    def capture(item):
        decisions.append(
            {
                "trigger": item.trigger_event_id,
                "observer": item.snapshot.observer_player_id,
                "target": item.action_target_global,
                "provenance": item.action_provenance,
                "prefix_sha256": hashlib.sha256(item.prefix.to_bytes()).hexdigest(),
            }
        )
        if len(early_prefixes) < 8:
            early_prefixes.append(item.prefix.to_record())

    replay = replay_game(game, on_decision=capture)
    return {
        "game_id": game.game_id,
        "source_sha256": game.source_sha256,
        "replay": replay,
        "decision_inputs": decisions,
        "early_prefixes": early_prefixes,
    }


def _differences(left, right, path="root"):
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                yield path + "/" + key, "missing key"
            else:
                yield from _differences(left[key], right[key], path + "/" + key)
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            yield path, (len(left), len(right))
        for index, (a, b) in enumerate(zip(left, right)):
            yield from _differences(a, b, path + "/" + str(index))
    elif left != right:
        yield path, (left, right)


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
    for result in serial + parallel:
        inferred = (
            item for item in result["decision_inputs"] if item["provenance"] == "inferred_unique_next_draw_pass"
        )
        for _, group in groupby(inferred, key=lambda item: item["trigger"]):
            seats = [item["observer"] for item in group]
            assert seats == sorted(seats), "Pass export order must follow stable initial seats, not a native hash map"
    differences = list(islice(_differences(serial, parallel), 8))
    if differences:
        print("DETERMINISM_DIFFERENCES=" + json.dumps(differences, ensure_ascii=False))
    assert content_hash(serial) == content_hash(parallel), differences
    assert [row["replay"]["input_encoding"]["prefix_sha256"] for row in serial] == [
        row["replay"]["input_encoding"]["prefix_sha256"] for row in parallel
    ]
