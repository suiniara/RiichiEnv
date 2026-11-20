"""Exercise the replay datasets with different RNG streams in each worker."""

import random
from collections import Counter
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy", reason="Install the optional kar-ml dependency group for ML data tests")
pytest.importorskip("torch", reason="Install the optional kar-ml dependency group for ML data tests")

import numpy as np
import torch
from riichienv_ml.datasets import grp_dataset, mjai_logs
from torch.utils.data import DataLoader


class _SyntheticObservation:
    def __init__(self, file_id, player_id):
        self.file_id = file_id
        self.player_id = player_id

    def mask(self):
        return b"\x01"


class _SyntheticAction:
    def encode(self):
        return 0


class _SyntheticKyoku:
    def __init__(self, file_id):
        self.file_id = file_id

    def take_grp_features(self):
        return {
            "round_initial_scores": [25000] * 4,
            "round_end_scores": [25000] * 4,
            "round_delta_scores": [0] * 4,
            "chang": 0,
            "ju": 0,
            "ben": self.file_id * 4,
            "liqibang": 0,
        }

    def steps(self, player_id):
        yield _SyntheticObservation(self.file_id, player_id), _SyntheticAction()


class _SyntheticReplay:
    def __init__(self, file_id):
        self.file_id = file_id

    @classmethod
    def from_jsonl(cls, file_path, *, rule):
        assert rule == "mjsoul"
        return cls(int(Path(file_path).stem.split("-")[1]))

    def take_kyokus(self):
        return [_SyntheticKyoku(self.file_id)]


class _SyntheticEncoder:
    def encode(self, observation):
        return np.array([observation.file_id, observation.player_id], dtype=np.int64)


class _SyntheticRewards:
    def calc_all_player_rewards(self, features):
        return [0.0] * 4


def _initialize_worker(worker_id):
    # Spawn workers do not inherit the parent's monkeypatches. Replace only
    # parsing; iteration, feature construction and DataLoader sharding are real.
    mjai_logs.MjaiReplay = _SyntheticReplay
    grp_dataset.MjaiReplay = _SyntheticReplay
    random.seed(1009 + 997 * worker_id)


@pytest.fixture(autouse=True)
def _restore_rng():
    state = random.getstate()
    yield
    random.setstate(state)


@pytest.fixture
def replay_files(tmp_path):
    files = []
    for index in (8, 2, 10, 1, 9, 0, 6, 4, 11, 3, 7, 5):
        path = tmp_path / f"replay-{index:02d}.jsonl"
        path.write_text("synthetic replay parser fixture\n", encoding="utf-8")
        files.append(str(path))
    return files


def _dataset(kind, files, *, is_train=True):
    pattern = str(Path(files[0]).parent / "*.jsonl")
    if kind == "grp":
        return grp_dataset.GrpReplayDataset(pattern, is_train=is_train)
    return mjai_logs.MCDataset(
        files if kind == "mc_list" else pattern,
        reward_predictor=_SyntheticRewards(),
        encoder=_SyntheticEncoder(),
        is_train=is_train,
    )


def _sample_key(kind, sample):
    features = sample[0]
    if kind == "grp":
        # The real GRP encoder stores ben / 4 at 14 and seat one-hot at 16:20.
        return round(float(features[14])), int(features[16:20].argmax())
    return int(features[0]), int(features[1])


def _expected(files):
    return Counter((int(Path(path).stem.split("-")[1]), player) for path in files for player in range(4))


def _patch_parser(monkeypatch):
    monkeypatch.setattr(mjai_logs, "MjaiReplay", _SyntheticReplay)
    monkeypatch.setattr(grp_dataset, "MjaiReplay", _SyntheticReplay)


@pytest.mark.parametrize("kind", ["mc_list", "mc_glob", "grp"])
@pytest.mark.parametrize("workers", [2, 3])
def test_worker_rngs_cover_every_replay_exactly_once(monkeypatch, replay_files, kind, workers):
    _patch_parser(monkeypatch)
    original = list(replay_files)
    samples = []
    for worker_id in range(workers):
        monkeypatch.setattr(
            torch.utils.data,
            "get_worker_info",
            lambda worker_id=worker_id: SimpleNamespace(id=worker_id, num_workers=workers),
        )
        random.seed(1009 + 997 * worker_id)
        samples.extend(_sample_key(kind, sample) for sample in _dataset(kind, replay_files))
    assert Counter(samples) == _expected(replay_files)
    assert replay_files == original


@pytest.mark.parametrize("is_train", [False, True])
def test_single_worker_does_not_modify_the_supplied_file_list(monkeypatch, replay_files, is_train):
    _patch_parser(monkeypatch)
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
    original = list(replay_files)
    random.seed(83)
    samples = [_sample_key("mc_list", sample) for sample in _dataset("mc_list", replay_files, is_train=is_train)]
    assert replay_files == original
    assert Counter(samples) == _expected(original)
    if not is_train:
        assert samples == sorted(_expected(original))


def test_list_and_glob_assign_the_same_files_to_each_worker(monkeypatch, replay_files):
    _patch_parser(monkeypatch)
    # A glob's directory enumeration need not match a caller-provided list.
    monkeypatch.setattr(mjai_logs.glob, "glob", lambda *args, **kwargs: list(reversed(replay_files)))
    for worker_id in range(3):
        monkeypatch.setattr(
            torch.utils.data,
            "get_worker_info",
            lambda worker_id=worker_id: SimpleNamespace(id=worker_id, num_workers=3),
        )
        samples = []
        for kind in ("mc_list", "mc_glob"):
            random.seed(1009 + 997 * worker_id)
            samples.append([_sample_key(kind, sample) for sample in _dataset(kind, replay_files)])
        assert samples[0] == samples[1]


@pytest.mark.parametrize("kind", ["mc_list", "mc_glob", "grp"])
def test_spawn_dataloader_covers_every_replay_once(replay_files, kind):
    original = list(replay_files)
    loader = DataLoader(
        _dataset(kind, replay_files),
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
        worker_init_fn=_initialize_worker,
        # Coverage needs only Python identity keys. Keep actual replay iteration
        # and encoding in the worker without requiring Tensor-sharing sockets.
        collate_fn=partial(_sample_key, kind),
        generator=torch.Generator().manual_seed(47),
        timeout=30,
    )
    samples = list(loader)
    assert Counter(samples) == _expected(original)
    assert replay_files == original
