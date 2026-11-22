"""Stage-4 RL Training Contract Primitives for KAR V2 (Contract v2.5).

Implements the three mandatory Stage-4 stability & anti-collapse mechanisms:
1. [`CriticWarmupGate`]: Blocks Actor PPO updates until the independent Oracle Critic
   reaches Explained Variance >= threshold for consecutive evaluations, preventing
   an untrained Critic from destroying the Stage-3 BC policy.
2. [`PFSPOpponentPool`]: Prioritized Fictitious Self-Play opponent pool with a
   non-evictable frozen `bc_anchor` sampling floor, preventing cyclical self-play
   drift and forgetting against strong human-like play.
3. [`FrozenGAERolloutBatch`]: Enforces Contract v2.4.1 / v2.5 GAE immutability:
   computes GAE exactly once from rollout-frozen Critic predictions under a bound
   [`RolloutVersionTag`], stores detached (`requires_grad=False`) targets, and
   cryptographically verifies target immutability across multiple PPO epochs.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .grp import RolloutTruncation, RoundBoundary, compute_hanchan_gae
from .model import RolloutVersionTag


def compute_explained_variance(v_pred: torch.Tensor, v_target: torch.Tensor) -> float:
    """Compute Explained Variance `1 - Var(y - y_hat) / Var(y)` in `fp32`."""
    pred = v_pred.detach().float().reshape(-1)
    tgt = v_target.detach().float().reshape(-1)
    if pred.shape != tgt.shape:
        raise ValueError(f"Shape mismatch: v_pred={tuple(pred.shape)} vs v_target={tuple(tgt.shape)}")
    if pred.numel() < 2:
        return 0.0
    var_y = float(torch.var(tgt, unbiased=False).item())
    if var_y < 1e-12:
        return 0.0
    var_err = float(torch.var(tgt - pred, unbiased=False).item())
    return 1.0 - (var_err / var_y)


class CriticWarmupGate:
    """Blocks Actor updates until the independent Oracle Critic passes Explained Variance warmup.

    Contract v2.5 Stage-4 safeguard:
    - During Critic warmup, ONLY `OracleCriticV2` parameters may step.
    - Actor PPO optimizer steps MUST call `gate.assert_actor_allowed()` first.
    - Gate opens permanently once `step >= min_critic_steps` AND `ev >= ev_threshold`
      for `required_consecutive` consecutive evaluation checks.
    """

    def __init__(
        self,
        *,
        ev_threshold: float = 0.15,
        required_consecutive: int = 2,
        min_critic_steps: int = 0,
    ) -> None:
        if required_consecutive < 1:
            raise ValueError("required_consecutive must be >= 1")
        if min_critic_steps < 0:
            raise ValueError("min_critic_steps must be >= 0")
        self.ev_threshold = float(ev_threshold)
        self.required_consecutive = int(required_consecutive)
        self.min_critic_steps = int(min_critic_steps)
        self._consecutive_passes = 0
        self._unlocked = False
        self._history: list[tuple[int, float]] = []

    @property
    def actor_updates_allowed(self) -> bool:
        return self._unlocked

    @property
    def consecutive_passes(self) -> int:
        return self._consecutive_passes

    @property
    def history(self) -> tuple[tuple[int, float], ...]:
        return tuple(self._history)

    def record_eval(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        *,
        step: int,
    ) -> float:
        """Evaluate Critic EV at `step` and update gate state."""
        ev = compute_explained_variance(v_pred, v_target)
        self._history.append((int(step), ev))
        if not self._unlocked:
            if step >= self.min_critic_steps and ev >= self.ev_threshold:
                self._consecutive_passes += 1
                if self._consecutive_passes >= self.required_consecutive:
                    self._unlocked = True
            else:
                self._consecutive_passes = 0
        return ev

    def assert_actor_allowed(self) -> None:
        """Raise `RuntimeError` if an Actor update is attempted before warmup completes."""
        if not self._unlocked:
            last_ev = self._history[-1][1] if self._history else float("nan")
            raise RuntimeError(
                "Contract v2.5 CriticWarmupGate CLOSED: Actor PPO updates are blocked until "
                f"Oracle Critic achieves EV >= {self.ev_threshold:.3f} for "
                f"{self.required_consecutive} consecutive evals "
                f"(current passes={self._consecutive_passes}, last_ev={last_ev:.4f})."
            )

    def state_dict(self) -> dict[str, object]:
        return {
            "ev_threshold": self.ev_threshold,
            "required_consecutive": self.required_consecutive,
            "min_critic_steps": self.min_critic_steps,
            "consecutive_passes": self._consecutive_passes,
            "unlocked": self._unlocked,
            "history": list(self._history),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._consecutive_passes = int(state["consecutive_passes"])  # type: ignore[arg-type]
        self._unlocked = bool(state["unlocked"])
        raw_hist = state.get("history", [])
        self._history = [(int(s), float(e)) for s, e in raw_hist]  # type: ignore[misc]


@dataclass
class PFSPOpponentEntry:
    """Tracked opponent policy inside [`PFSPOpponentPool`]."""

    opponent_id: str
    policy_version: int
    is_bc_anchor: bool = False
    games_played: int = 0
    ema_learner_score: float = 0.5


class PFSPOpponentPool:
    """Prioritized Fictitious Self-Play opponent pool with mandatory frozen BC anchor.

    Table matchmaking distribution for each of the 3 opponent seats:
    - `latest_self_prob`: current learner (`LATEST_SELF_ID`)
    - `bc_anchor_prob`: frozen Stage-3 BC anchor (non-evictable, floor enforced)
    - `1 - latest_self_prob - bc_anchor_prob`: PFSP-weighted over historical RL snapshots
      with weight `w_j = (1 - ema_learner_score_j + eps) ** pfsp_power`
      (prioritizing opponents against whom the learner currently underperforms).
    """

    LATEST_SELF_ID = "__latest__"

    def __init__(
        self,
        bc_anchor_id: str = "bc_anchor_v0",
        *,
        latest_self_prob: float = 0.25,
        bc_anchor_prob: float = 0.20,
        max_snapshots: int = 32,
        pfsp_power: float = 2.0,
        ema_alpha: float = 0.1,
        smooth_eps: float = 0.05,
    ) -> None:
        if not bc_anchor_id or bc_anchor_id == self.LATEST_SELF_ID:
            raise ValueError("Invalid bc_anchor_id")
        if bc_anchor_prob <= 0.0:
            raise ValueError("Contract v2.5 violation: bc_anchor_prob MUST be > 0 to prevent self-play forgetting")
        if latest_self_prob < 0.0 or latest_self_prob + bc_anchor_prob > 1.0 + 1e-9:
            raise ValueError(f"Invalid probability split: latest={latest_self_prob}, bc_anchor={bc_anchor_prob}")
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be >= 1")

        self.bc_anchor_id = bc_anchor_id
        self.latest_self_prob = float(latest_self_prob)
        self.bc_anchor_prob = float(bc_anchor_prob)
        self.max_snapshots = int(max_snapshots)
        self.pfsp_power = float(pfsp_power)
        self.ema_alpha = float(ema_alpha)
        self.smooth_eps = float(smooth_eps)

        self._anchor = PFSPOpponentEntry(
            opponent_id=bc_anchor_id,
            policy_version=0,
            is_bc_anchor=True,
        )
        self._snapshots: list[PFSPOpponentEntry] = []

    @property
    def snapshots(self) -> tuple[PFSPOpponentEntry, ...]:
        return tuple(self._snapshots)

    @property
    def anchor(self) -> PFSPOpponentEntry:
        return self._anchor

    def add_snapshot(self, opponent_id: str, policy_version: int) -> None:
        """Register a new frozen historical snapshot (FIFO eviction; NEVER evicts `bc_anchor`)."""
        if opponent_id in (self.bc_anchor_id, self.LATEST_SELF_ID):
            raise ValueError(f"Reserved opponent_id: {opponent_id!r}")
        if any(e.opponent_id == opponent_id for e in self._snapshots):
            raise ValueError(f"Duplicate snapshot opponent_id: {opponent_id!r}")
        self._snapshots.append(
            PFSPOpponentEntry(
                opponent_id=opponent_id,
                policy_version=int(policy_version),
                is_bc_anchor=False,
            )
        )
        while len(self._snapshots) > self.max_snapshots:
            self._snapshots.pop(0)

    def record_match_result(self, opponent_id: str, learner_score: float) -> None:
        """Update EMA learner score in `[0.0, 1.0]` against `opponent_id`.

        For 4-player Riichi Mahjong, normalized placement score `(4 - rank) / 3.0`
        (1st=1.0, 2nd=2/3, 3rd=1/3, 4th=0.0) or pairwise win rate in `[0, 1]`
        gives a well-scaled hardness signal.
        """
        if not 0.0 <= learner_score <= 1.0:
            raise ValueError(f"learner_score must be in [0, 1], got {learner_score}")
        if opponent_id == self.LATEST_SELF_ID:
            return
        entry: PFSPOpponentEntry | None = None
        if opponent_id == self.bc_anchor_id:
            entry = self._anchor
        else:
            for snap in self._snapshots:
                if snap.opponent_id == opponent_id:
                    entry = snap
                    break
        if entry is None:
            raise KeyError(f"Unknown opponent_id {opponent_id!r}")
        entry.games_played += 1
        if entry.games_played == 1:
            entry.ema_learner_score = float(learner_score)
        else:
            entry.ema_learner_score = (1.0 - self.ema_alpha) * entry.ema_learner_score + self.ema_alpha * float(
                learner_score
            )

    def sampling_probabilities(self) -> dict[str, float]:
        """Return exact normalized opponent sampling probabilities across all pool members."""
        probs: dict[str, float] = {}
        if not self._snapshots:
            # Before any RL snapshot is saved, split between latest self-play and BC anchor
            # while strictly keeping `p(bc_anchor) >= self.bc_anchor_prob`.
            p_latest = min(self.latest_self_prob, 1.0 - self.bc_anchor_prob)
            p_anchor = 1.0 - p_latest
            if p_latest > 0.0:
                probs[self.LATEST_SELF_ID] = p_latest
            probs[self.bc_anchor_id] = p_anchor
            return probs

        if self.latest_self_prob > 0.0:
            probs[self.LATEST_SELF_ID] = self.latest_self_prob
        probs[self.bc_anchor_id] = self.bc_anchor_prob

        pfsp_mass = max(0.0, 1.0 - self.latest_self_prob - self.bc_anchor_prob)
        raw_weights = [
            (max(0.0, 1.0 - snap.ema_learner_score) + self.smooth_eps) ** self.pfsp_power for snap in self._snapshots
        ]
        total_w = sum(raw_weights)
        for snap, w in zip(self._snapshots, raw_weights, strict=True):
            probs[snap.opponent_id] = pfsp_mass * (w / total_w)
        return probs

    def sample_table_opponents(self, rng: random.Random | None = None) -> tuple[str, str, str]:
        """Sample 3 opponent IDs for a 4-player Riichi table."""
        generator = rng or random.Random()
        dist = self.sampling_probabilities()
        ids = list(dist.keys())
        weights = [dist[k] for k in ids]
        chosen = generator.choices(ids, weights=weights, k=3)
        return (chosen[0], chosen[1], chosen[2])


@dataclass(frozen=True)
class HanchanTrajectoryInput:
    """Single-seat Hanchan rollout trajectory with rollout-frozen Critic values."""

    steps: Sequence[RoundBoundary]
    round_start_potentials: Sequence[float]
    terminated: bool
    terminal_reward: float | None = None
    truncation: RolloutTruncation | None = None
    truncated_bootstrap_v_shaped: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "round_start_potentials": [float(p) for p in self.round_start_potentials],
            "terminated": bool(self.terminated),
            "terminal_reward": (float(self.terminal_reward) if self.terminal_reward is not None else None),
            "truncation": (self.truncation.to_dict() if self.truncation is not None else None),
            "truncated_bootstrap_v_shaped": (
                float(self.truncated_bootstrap_v_shaped) if self.truncated_bootstrap_v_shaped is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HanchanTrajectoryInput:
        raw_trunc = data.get("truncation")
        return cls(
            steps=[RoundBoundary.from_dict(s) for s in data["steps"]],
            round_start_potentials=[float(p) for p in data["round_start_potentials"]],
            terminated=bool(data["terminated"]),
            terminal_reward=(float(data["terminal_reward"]) if data.get("terminal_reward") is not None else None),
            truncation=(RolloutTruncation.from_dict(raw_trunc) if raw_trunc is not None else None),
            truncated_bootstrap_v_shaped=(
                float(data["truncated_bootstrap_v_shaped"])
                if data.get("truncated_bootstrap_v_shaped") is not None
                else None
            ),
        )


def _extract_trajectory_boundary_metadata(traj: HanchanTrajectoryInput) -> dict[str, Any]:
    last_k = traj.steps[-1].round_index if traj.steps else None
    first_k = traj.steps[0].round_index if traj.steps else None
    if traj.terminated:
        boundary_type = "terminal"
        trunc_dict: dict[str, Any] | None = None
    elif traj.truncation is not None:
        boundary_type = traj.truncation.kind
        trunc_dict = traj.truncation.to_dict()
    else:
        assert traj.truncated_bootstrap_v_shaped is not None and last_k is not None
        boundary_type = "intra_round"
        trunc_dict = RolloutTruncation(
            kind="intra_round",
            bootstrap_v_shaped=float(traj.truncated_bootstrap_v_shaped),
            next_round_index=last_k,
            next_round_phi=float(traj.round_start_potentials[last_k]),
            bootstrap_round_index=last_k,
        ).to_dict()
    return {
        "num_steps": len(traj.steps),
        "first_round_index": first_k,
        "last_round_index": last_k,
        "boundary_type": boundary_type,
        "terminated": bool(traj.terminated),
        "terminal_reward": (float(traj.terminal_reward) if traj.terminal_reward is not None else None),
        "truncation": trunc_dict,
    }


class FrozenGAERolloutBatch:
    """Immutable single-pass GAE rollout batch bound to [`RolloutVersionTag`].

    Enforces Contract v2.4.1 / v2.5:
    - GAE advantages and shaped returns are computed ONCE at rollout finalization
      from frozen rollout-time Critic predictions (`version_tag.critic_version`).
    - Stored tensors (`step_rewards`, `advantages`, `shaped_returns`, `rollout_v_shaped`)
      are detached (`requires_grad == False`), retain explicit per-trajectory boundary
      metadata (`trajectory_boundaries`), and are fingerprinted with SHA-256 so multi-epoch
      PPO loops and checkpoint serialization/resume CANNOT silently mutate targets or drop
      `RolloutTruncation` boundary semantics.
    """

    def __init__(
        self,
        *,
        version_tag: RolloutVersionTag,
        step_rewards: torch.Tensor,
        advantages: torch.Tensor,
        shaped_returns: torch.Tensor,
        rollout_v_shaped: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        trajectory_boundaries: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        self.version_tag = version_tag
        self.step_rewards = step_rewards.detach().clone().float()
        self.advantages = advantages.detach().clone().float()
        self.shaped_returns = shaped_returns.detach().clone().float()
        self.rollout_v_shaped = rollout_v_shaped.detach().clone().float()
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.trajectory_boundaries: tuple[dict[str, Any], ...] = tuple(dict(b) for b in trajectory_boundaries)
        self._target_sha256 = self._compute_target_digest()

    def _compute_target_digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.step_rewards.detach().cpu().contiguous().numpy().tobytes())
        h.update(self.advantages.detach().cpu().contiguous().numpy().tobytes())
        h.update(self.shaped_returns.detach().cpu().contiguous().numpy().tobytes())
        h.update(self.rollout_v_shaped.detach().cpu().contiguous().numpy().tobytes())
        meta = {
            "policy_version": self.version_tag.policy_version,
            "critic_version": self.version_tag.critic_version,
            "grp_sha256": self.version_tag.grp_sha256,
            "ruleset_sha256": self.version_tag.ruleset_sha256,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "trajectory_boundaries": list(self.trajectory_boundaries),
        }
        h.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return h.hexdigest()

    @property
    def num_steps(self) -> int:
        return int(self.advantages.shape[0])

    @property
    def target_sha256(self) -> str:
        return self._target_sha256

    def state_dict(self) -> dict[str, Any]:
        """Serialize frozen rollout batch including full boundary metadata and SHA-256 digest."""
        return {
            "version_tag": {
                "policy_version": int(self.version_tag.policy_version),
                "critic_version": int(self.version_tag.critic_version),
                "grp_sha256": str(self.version_tag.grp_sha256),
                "ruleset_sha256": str(self.version_tag.ruleset_sha256),
            },
            "step_rewards": self.step_rewards.detach().cpu().clone(),
            "advantages": self.advantages.detach().cpu().clone(),
            "shaped_returns": self.shaped_returns.detach().cpu().clone(),
            "rollout_v_shaped": self.rollout_v_shaped.detach().cpu().clone(),
            "gamma": float(self.gamma),
            "gae_lambda": float(self.gae_lambda),
            "trajectory_boundaries": [dict(b) for b in self.trajectory_boundaries],
            "target_sha256": self._target_sha256,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> FrozenGAERolloutBatch:
        """Restore a serialized [`FrozenGAERolloutBatch`] and verify cryptographic integrity."""
        vt_raw = state["version_tag"]
        tag = RolloutVersionTag(
            policy_version=int(vt_raw["policy_version"]),
            critic_version=int(vt_raw["critic_version"]),
            grp_sha256=str(vt_raw["grp_sha256"]),
            ruleset_sha256=str(vt_raw["ruleset_sha256"]),
        )
        raw_boundaries = state.get("trajectory_boundaries", ())
        normalized_boundaries: list[dict[str, Any]] = []
        for entry in raw_boundaries:
            item = dict(entry)
            if item.get("truncation") is not None:
                # Re-validate RolloutTruncation schema on restore
                item["truncation"] = RolloutTruncation.from_dict(item["truncation"]).to_dict()
            normalized_boundaries.append(item)

        batch = cls(
            version_tag=tag,
            step_rewards=torch.as_tensor(state["step_rewards"], dtype=torch.float32),
            advantages=torch.as_tensor(state["advantages"], dtype=torch.float32),
            shaped_returns=torch.as_tensor(state["shaped_returns"], dtype=torch.float32),
            rollout_v_shaped=torch.as_tensor(state["rollout_v_shaped"], dtype=torch.float32),
            gamma=float(state["gamma"]),
            gae_lambda=float(state["gae_lambda"]),
            trajectory_boundaries=normalized_boundaries,
        )
        expected_sha = str(state["target_sha256"])
        if batch.target_sha256 != expected_sha:
            raise RuntimeError(
                f"Contract v2.4.1 violation: Restored FrozenGAERolloutBatch SHA-256 mismatch "
                f"(expected={expected_sha}, got={batch.target_sha256})"
            )
        return batch

    @classmethod
    def from_hanchan_trajectories(
        cls,
        trajectories: Sequence[HanchanTrajectoryInput],
        *,
        version_tag: RolloutVersionTag,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
    ) -> FrozenGAERolloutBatch:
        """Compute GAE once over rollout-frozen trajectories and lock targets."""
        all_rewards: list[float] = []
        all_adv: list[float] = []
        all_ret: list[float] = []
        all_v_shaped: list[float] = []
        boundaries: list[dict[str, Any]] = []

        for traj in trajectories:
            rewards, adv, ret = compute_hanchan_gae(
                traj.steps,
                traj.round_start_potentials,
                terminated=traj.terminated,
                terminal_reward=traj.terminal_reward,
                truncation=traj.truncation,
                truncated_bootstrap_v_shaped=traj.truncated_bootstrap_v_shaped,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            all_rewards.extend(rewards)
            all_adv.extend(adv)
            all_ret.extend(ret)
            all_v_shaped.extend(s.v_shaped for s in traj.steps)
            if traj.steps:
                boundaries.append(_extract_trajectory_boundary_metadata(traj))

        if not all_adv:
            raise ValueError("Cannot construct FrozenGAERolloutBatch with 0 steps")

        return cls(
            version_tag=version_tag,
            step_rewards=torch.tensor(all_rewards, dtype=torch.float32),
            advantages=torch.tensor(all_adv, dtype=torch.float32),
            shaped_returns=torch.tensor(all_ret, dtype=torch.float32),
            rollout_v_shaped=torch.tensor(all_v_shaped, dtype=torch.float32),
            gamma=gamma,
            gae_lambda=gae_lambda,
            trajectory_boundaries=boundaries,
        )

    def assert_ready_for_ppo_epoch(
        self,
        *,
        current_policy_version: int,
        expected_grp_sha256: str,
        expected_ruleset_sha256: str,
        max_staleness: int = 2,
    ) -> None:
        """Verify version provenance, detached state, and cryptographic immutability before each PPO epoch."""
        self.version_tag.validate_compatible(
            current_policy_version=current_policy_version,
            expected_grp_sha256=expected_grp_sha256,
            expected_ruleset_sha256=expected_ruleset_sha256,
            max_staleness=max_staleness,
        )
        if (
            self.step_rewards.requires_grad
            or self.advantages.requires_grad
            or self.shaped_returns.requires_grad
            or self.rollout_v_shaped.requires_grad
        ):
            raise RuntimeError("Contract v2.4.1 violation: GAE rollout targets MUST remain detached")
        if self._compute_target_digest() != self._target_sha256:
            raise RuntimeError("Contract v2.4.1 violation: Frozen GAE targets were mutated across PPO epochs!")
