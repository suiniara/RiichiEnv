"""Pure-State Seat-Permutation-Equivariant GRP & Full-Hanchan Segmented GAE (Contract v2.4).

Implements Contract Sections 3.1 and 3.3:
- Pure-state input canonicalized relative to observer seat `i` (NO `end_score` or `delta_score`)
- Exact 4-seat cyclic permutation equivariance by construction
- Frozen version & checksum binding `(grp_version, grp_sha256, ruleset_id, ruleset_sha256)`
- Exact telescoping PBRS round rewards: sum_{k=0}^{K-1} r_k = R_terminal - Phi_0
- Full-Hanchan continuous GAE (`gamma=1.0`, `lambda_round == lambda`) handling:
  * normal intra-round and cross-round transitions
  * zero-decision skipped rounds (potential diff merged onto previous active decision)
  * Hanchan termination (`terminated=True` -> `next_value=0.0`)
  * mid-Hanchan rollout truncation (`truncated=True` -> bootstrap `V_shaped`, NEVER `0.0`)
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

PURE_STATE_GRP_INPUT_DIM = 16


def canonicalize_grp_state(
    scores: Sequence[int | float],
    round_wind: int,
    oya: int,
    honba: int,
    kyoutaku: int,
    seat: int,
    rank_payoffs: Sequence[float],
    is_tonpuusen: bool = False,
) -> torch.Tensor:
    """Canonicalize round-start public state relative to seat `seat ∈ {0,1,2,3}`.

    By rotating scores into `[seat, (seat+1)%4, (seat+2)%4, (seat+3)%4]` and
    encoding `oya_rel = (oya + 4 - seat) % 4`, the resulting 16-dim feature
    vector is strictly invariant under any cyclic permutation of seats.
    """
    if len(scores) != 4:
        raise ValueError(f"Expected 4 player scores, got {len(scores)}")
    if not 0 <= seat < 4:
        raise ValueError(f"Expected seat in [0, 3], got {seat}")
    if len(rank_payoffs) != 4:
        raise ValueError(f"Expected 4 rank payoffs, got {len(rank_payoffs)}")

    rel_scores = [float(scores[(seat + rel) % 4] - 25_000) / 25_000.0 for rel in range(4)]
    oya_rel = (int(oya) + 4 - int(seat)) % 4
    oya_one_hot = [1.0 if oya_rel == rel else 0.0 for rel in range(4)]

    # Pure start-of-round public state ONLY (no end_score, no delta_score!)
    feats = [
        *rel_scores,  # 0..3: relative scores
        *oya_one_hot,  # 4..7: relative dealer seat
        float(min(max(round_wind, 0), 2)) / 2.0,  # 8: round wind
        float(min(max(honba, 0), 8)) / 8.0,  # 9: honba
        float(min(max(kyoutaku, 0), 8)) / 8.0,  # 10: kyoutaku
        1.0 if is_tonpuusen else 0.0,  # 11: east-only flag
        *[float(w) for w in rank_payoffs],  # 12..15: rule payoff vector w(R)
    ]
    return torch.tensor(feats, dtype=torch.float32)


class PureStateGRP(nn.Module):
    """Pure-state Global Reward Predictor predicting P_GRP(r | s_k, i, R)."""

    def __init__(self, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PURE_STATE_GRP_INPUT_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, state_feats: torch.Tensor) -> torch.Tensor:
        """Return 4-placement logits `[..., 4]`."""
        return self.net(state_feats)

    def predict_probs(self, state_feats: torch.Tensor) -> torch.Tensor:
        """Return 4-placement probability distribution `P_GRP(r | s_k, i, R)`."""
        return torch.softmax(self.forward(state_feats).float(), dim=-1)

    def predict_potential(
        self,
        state_feats: torch.Tensor,
        rank_payoffs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar potential `Phi(s_k, i, R) = sum_r w_r * P_GRP(r | s_k, i, R)`."""
        probs = self.predict_probs(state_feats)
        return (probs * rank_payoffs.float()).sum(dim=-1)

    def parameter_sha256(self) -> str:
        """Deterministic SHA-256 checksum over all float32 parameters."""
        h = hashlib.sha256()
        for name, param in sorted(self.named_parameters()):
            h.update(name.encode("utf-8"))
            h.update(param.detach().cpu().float().contiguous().numpy().tobytes())
        return h.hexdigest()


class FrozenGRP:
    """Read-only frozen GRP wrapper bound to `(grp_version, grp_sha256, ruleset_id, ruleset_sha256)`.

    Enforces Contract v2.4 Section 3.1:
    - All parameters frozen (`requires_grad=False`, `eval()` mode)
    - Checksum verified on construction and available to workers/learners
    - Terminal potential strictly defined as `Phi_terminal = 0.0`
    """

    def __init__(
        self,
        model: PureStateGRP,
        *,
        grp_version: str,
        ruleset_id: str,
        ruleset_sha256: str,
        rank_payoffs: Sequence[float],
        is_tonpuusen: bool = False,
        expected_grp_sha256: str | None = None,
    ) -> None:
        if len(rank_payoffs) != 4:
            raise ValueError("rank_payoffs must have length 4")
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        actual_sha = model.parameter_sha256()
        if expected_grp_sha256 is not None and actual_sha != expected_grp_sha256:
            raise ValueError(f"GRP checksum mismatch: expected {expected_grp_sha256}, got {actual_sha}")
        self.model = model
        self.grp_version = grp_version
        self.grp_sha256 = actual_sha
        self.ruleset_id = ruleset_id
        self.ruleset_sha256 = ruleset_sha256
        self.rank_payoffs = tuple(float(w) for w in rank_payoffs)
        self.is_tonpuusen = bool(is_tonpuusen)

    @torch.inference_mode()
    def predict_all_seats(
        self,
        scores: Sequence[int | float],
        round_wind: int,
        oya: int,
        honba: int,
        kyoutaku: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(probs[4, 4], potentials[4])` for all 4 absolute seats."""
        device = next(self.model.parameters()).device
        batch = torch.stack(
            [
                canonicalize_grp_state(
                    scores=scores,
                    round_wind=round_wind,
                    oya=oya,
                    honba=honba,
                    kyoutaku=kyoutaku,
                    seat=seat,
                    rank_payoffs=self.rank_payoffs,
                    is_tonpuusen=self.is_tonpuusen,
                )
                for seat in range(4)
            ],
            dim=0,
        ).to(device)
        w = torch.tensor(self.rank_payoffs, dtype=torch.float32, device=device)
        probs = self.model.predict_probs(batch)
        potentials = (probs * w).sum(dim=-1)
        return probs, potentials


def compute_pbrs_round_rewards(
    round_start_potentials: Sequence[float],
    terminal_reward: float,
) -> list[float]:
    """Compute exact telescoping PBRS round rewards for a K-round Hanchan (gamma=1).

    Contract v2.4 Section 3.1:
      r_k     = Phi_{k+1} - Phi_k              for 0 <= k <= K-2
      r_{K-1} = R_terminal + 0 - Phi_{K-1}     for final round K-1 (Phi_terminal == 0)

    Guarantees: sum_{k=0}^{K-1} r_k == R_terminal - Phi_0.
    """
    k_rounds = len(round_start_potentials)
    if k_rounds == 0:
        raise ValueError("Hanchan must contain at least 1 played round")
    rewards: list[float] = []
    for k in range(k_rounds - 1):
        rewards.append(float(round_start_potentials[k + 1]) - float(round_start_potentials[k]))
    rewards.append(float(terminal_reward) - float(round_start_potentials[k_rounds - 1]))
    return rewards


def _require_finite(value: float, name: str) -> float:
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} must be finite, got {val}")
    return val


@dataclass(frozen=True)
class RoundBoundary:
    """One decision step inside a Hanchan trajectory for a single seat."""

    round_index: int
    phi_k: float
    v_raw: float

    @property
    def v_shaped(self) -> float:
        return self.v_raw - self.phi_k

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": int(self.round_index),
            "phi_k": float(self.phi_k),
            "v_raw": float(self.v_raw),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoundBoundary:
        for key in ("round_index", "phi_k", "v_raw"):
            if key not in data or data[key] is None:
                raise ValueError(f"RoundBoundary missing required field {key!r}")
        return cls(
            round_index=int(data["round_index"]),
            phi_k=_require_finite(float(data["phi_k"]), "RoundBoundary.phi_k"),
            v_raw=_require_finite(float(data["v_raw"]), "RoundBoundary.v_raw"),
        )


@dataclass(frozen=True)
class RolloutTruncation:
    """Explicit non-terminal rollout truncation boundary (Contract v2.5.1).

    Distinguishes:
    - `kind="intra_round"`: Rollout hit step cap while still inside `steps[-1].round_index`
      (`step_rewards[-1] == 0.0`, bootstrap `V_shaped(z_trunc)` evaluated in round `last_k`).
    - `kind="cross_round"`: Round `steps[-1].round_index` (and optionally one or more
      zero-decision skipped rounds) already settled and round `next_round_index > steps[-1].round_index`
      started with potential `next_round_phi` before any decision in `next_round_index`
      was recorded (`step_rewards[-1] == next_round_phi - Phi_{last_k}`, bootstrap
      `V_shaped(z_trunc)` evaluated in round `next_round_index` using `next_round_phi`).

    Optional `bootstrap_round_index` and `bootstrap_v_raw` allow strict fail-closed
    verification that `bootstrap_v_shaped` was evaluated in the declared boundary round
    with the matching round potential (never stale `Phi_k` after crossing rounds).
    """

    kind: Literal["intra_round", "cross_round"]
    bootstrap_v_shaped: float
    next_round_index: int | None = None
    next_round_phi: float | None = None
    bootstrap_round_index: int | None = None
    bootstrap_v_raw: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "bootstrap_v_shaped": float(self.bootstrap_v_shaped),
            "next_round_index": (int(self.next_round_index) if self.next_round_index is not None else None),
            "next_round_phi": (float(self.next_round_phi) if self.next_round_phi is not None else None),
            "bootstrap_round_index": (
                int(self.bootstrap_round_index) if self.bootstrap_round_index is not None else None
            ),
            "bootstrap_v_raw": (float(self.bootstrap_v_raw) if self.bootstrap_v_raw is not None else None),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RolloutTruncation:
        for key in ("kind", "bootstrap_v_shaped"):
            if key not in data or data[key] is None:
                raise ValueError(f"RolloutTruncation missing required field {key!r}")
        raw_kind = str(data["kind"])
        if raw_kind not in ("intra_round", "cross_round"):
            raise ValueError(f"Unknown truncation.kind={raw_kind!r}")
        kind: Literal["intra_round", "cross_round"] = "intra_round" if raw_kind == "intra_round" else "cross_round"
        if kind == "cross_round":
            if data.get("next_round_index") is None:
                raise ValueError("RolloutTruncation(kind='cross_round') missing required field 'next_round_index'")
            if data.get("next_round_phi") is None:
                raise ValueError("RolloutTruncation(kind='cross_round') missing required field 'next_round_phi'")
        return cls(
            kind=kind,
            bootstrap_v_shaped=_require_finite(float(data["bootstrap_v_shaped"]), "bootstrap_v_shaped"),
            next_round_index=(int(data["next_round_index"]) if data.get("next_round_index") is not None else None),
            next_round_phi=(
                _require_finite(float(data["next_round_phi"]), "next_round_phi")
                if data.get("next_round_phi") is not None
                else None
            ),
            bootstrap_round_index=(
                int(data["bootstrap_round_index"]) if data.get("bootstrap_round_index") is not None else None
            ),
            bootstrap_v_raw=(
                _require_finite(float(data["bootstrap_v_raw"]), "bootstrap_v_raw")
                if data.get("bootstrap_v_raw") is not None
                else None
            ),
        )


def _validate_bootstrap_consistency(
    truncation: RolloutTruncation,
    *,
    expected_round_index: int,
    expected_phi: float,
    bootstrap_val: float,
) -> None:
    if truncation.bootstrap_round_index is not None and truncation.bootstrap_round_index != expected_round_index:
        raise ValueError(
            f"{truncation.kind} truncation bootstrap_round_index={truncation.bootstrap_round_index} "
            f"disagrees with boundary round {expected_round_index}"
        )
    if truncation.bootstrap_v_raw is not None:
        v_raw = _require_finite(truncation.bootstrap_v_raw, "bootstrap_v_raw")
        expected_v_shaped = v_raw - expected_phi
        if abs(expected_v_shaped - bootstrap_val) > 1e-5:
            raise ValueError(
                f"{truncation.kind} truncation bootstrap_v_shaped={bootstrap_val} disagrees with "
                f"bootstrap_v_raw - expected_phi ({v_raw} - {expected_phi} = {expected_v_shaped}); "
                "ensure bootstrap V_shaped uses the target round potential"
            )


def _resolve_truncation_terminal_step(
    *,
    last_k: int,
    round_start_potentials: Sequence[float],
    truncation: RolloutTruncation | None,
    truncated_bootstrap_v_shaped: float | None,
) -> tuple[float, float]:
    """Resolve `(last_step_reward, bootstrap_v_shaped)` for a non-terminal truncated rollout."""
    n_potentials = len(round_start_potentials)
    if truncation is not None and truncated_bootstrap_v_shaped is not None:
        raise ValueError(
            "Pass either `truncation=RolloutTruncation(...)` or legacy `truncated_bootstrap_v_shaped`, not both"
        )
    if truncation is None and truncated_bootstrap_v_shaped is None:
        raise ValueError(
            "truncation (or truncated_bootstrap_v_shaped) is required when terminated=False "
            "(Contract v2.5.1: truncated rollouts MUST NOT bootstrap with 0.0)"
        )

    if truncation is None:
        if n_potentials > last_k + 1:
            raise ValueError(
                f"round_start_potentials has {n_potentials} entries beyond last_k={last_k}; "
                "pass explicit `truncation=RolloutTruncation(kind='cross_round', ...)` "
                "so cross-round potential rewards are never silently dropped"
            )
        assert truncated_bootstrap_v_shaped is not None
        truncation = RolloutTruncation(
            kind="intra_round",
            bootstrap_v_shaped=_require_finite(truncated_bootstrap_v_shaped, "truncated_bootstrap_v_shaped"),
            next_round_index=last_k,
            next_round_phi=float(round_start_potentials[last_k]),
            bootstrap_round_index=last_k,
        )

    bootstrap_val = _require_finite(truncation.bootstrap_v_shaped, "bootstrap_v_shaped")
    phi_last = float(round_start_potentials[last_k])

    if truncation.kind == "intra_round":
        if truncation.next_round_index is not None and truncation.next_round_index != last_k:
            raise ValueError(
                f"intra_round truncation requires next_round_index == last_k ({last_k}), "
                f"got {truncation.next_round_index}"
            )
        if truncation.next_round_phi is not None:
            phi_intra = _require_finite(truncation.next_round_phi, "intra_round next_round_phi")
            if abs(phi_intra - phi_last) > 1e-5:
                raise ValueError(
                    f"intra_round next_round_phi={phi_intra} disagrees with round_start_potentials[{last_k}]={phi_last}"
                )
        _validate_bootstrap_consistency(
            truncation,
            expected_round_index=last_k,
            expected_phi=phi_last,
            bootstrap_val=bootstrap_val,
        )
        return 0.0, bootstrap_val

    if truncation.kind == "cross_round":
        nxt_k = truncation.next_round_index
        if nxt_k is None or nxt_k <= last_k:
            raise ValueError(
                f"cross_round truncation requires next_round_index > last_k ({last_k}), got {nxt_k} "
                "(round already entered in trajectory; refusing duplicate or regressive potential reward)"
            )
        if truncation.next_round_phi is None:
            raise ValueError(
                f"cross_round truncation at next_round_index={nxt_k} requires explicit `next_round_phi` "
                "(fail-closed: never infer or zero-fill missing cross-round potential)"
            )
        phi_next = _require_finite(truncation.next_round_phi, "cross_round next_round_phi")
        if nxt_k < n_potentials and abs(phi_next - float(round_start_potentials[nxt_k])) > 1e-5:
            raise ValueError(
                f"cross_round next_round_phi={phi_next} disagrees with "
                f"round_start_potentials[{nxt_k}]={round_start_potentials[nxt_k]}"
            )
        _validate_bootstrap_consistency(
            truncation,
            expected_round_index=nxt_k,
            expected_phi=phi_next,
            bootstrap_val=bootstrap_val,
        )
        return phi_next - phi_last, bootstrap_val

    raise ValueError(f"Unknown truncation.kind={truncation.kind!r}")


def _validate_steps_and_potentials(
    steps: Sequence[RoundBoundary],
    round_start_potentials: Sequence[float],
) -> None:
    n_potentials = len(round_start_potentials)
    for idx, phi_val in enumerate(round_start_potentials):
        _require_finite(phi_val, f"round_start_potentials[{idx}]")
    for idx, step in enumerate(steps):
        if not 0 <= step.round_index < n_potentials:
            raise ValueError(
                f"step.round_index={step.round_index} out of bounds for {n_potentials} round_start_potentials"
            )
        step_phi = _require_finite(step.phi_k, f"steps[{idx}].phi_k")
        _require_finite(step.v_raw, f"steps[{idx}].v_raw")
        if abs(step_phi - float(round_start_potentials[step.round_index])) > 1e-5:
            raise ValueError(
                f"step.phi_k={step_phi} disagrees with "
                f"round_start_potentials[{step.round_index}]={round_start_potentials[step.round_index]}"
            )


def compute_hanchan_gae(
    steps: Sequence[RoundBoundary],
    round_start_potentials: Sequence[float],
    *,
    terminated: bool,
    terminal_reward: float | None = None,
    truncation: RolloutTruncation | None = None,
    truncated_bootstrap_v_shaped: float | None = None,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float], list[float]]:
    """Compute full-Hanchan continuous GAE (`lambda_round == gae_lambda`) over shaped values.

    Handles ALL Contract v2.5.1 Section 3.3 boundaries:
    1. Intra-round consecutive steps: r_t = 0.0, bootstrap V_shaped(z_{t+1}).
    2. Cross-round normal transitions (k -> k+1): r_t = Phi_{k+1} - Phi_k.
    3. Zero-decision skipped rounds (seat had 0 decisions in round k+1, e.g. opponent Tenhou):
       potential jump Phi_{next_round} - Phi_{prev_round} is cleanly merged onto the last
       active decision before the gap, preserving exact telescoping.
    4. Hanchan termination (`terminated=True`): last step gets `R_terminal - Phi_{last_round}`
       and `next_value = 0.0` (the ONLY place `next_value == 0.0`).
    5. Mid-Hanchan intra-round truncation (`RolloutTruncation(kind="intra_round", ...)`):
       last step gets `r_t = 0.0` and bootstraps `bootstrap_v_shaped` (NEVER `0.0`!).
    6. Cross-round / round-end truncation (`RolloutTruncation(kind="cross_round", ...)`):
       when round `last_k` (and any skipped zero-decision rounds) settled before rollout cap
       and round `next_round_index > last_k` started with potential `Phi_{next_round}`,
       last step gets `r_t = Phi_{next_round} - Phi_{last_k}` AND bootstraps
       `bootstrap_v_shaped`, preserving exact multi-chunk telescoping!

    Returns `(step_rewards, advantages, shaped_gae_returns)`.
    """
    gamma = _require_finite(gamma, "gamma")
    gae_lambda = _require_finite(gae_lambda, "gae_lambda")
    if terminated and (truncation is not None or truncated_bootstrap_v_shaped is not None):
        raise ValueError("truncation / truncated_bootstrap_v_shaped must not be supplied when terminated=True")
    if not terminated and terminal_reward is not None:
        raise ValueError("terminal_reward must not be supplied when terminated=False")
    n_steps = len(steps)
    if n_steps == 0:
        return [], [], []

    _validate_steps_and_potentials(steps, round_start_potentials)

    step_rewards = [0.0] * n_steps
    for t in range(n_steps - 1):
        cur_k = steps[t].round_index
        nxt_k = steps[t + 1].round_index
        if nxt_k < cur_k:
            raise ValueError(f"Non-monotone round indices: {cur_k} -> {nxt_k}")
        if nxt_k > cur_k:
            step_rewards[t] = float(round_start_potentials[nxt_k]) - float(round_start_potentials[cur_k])

    last_k = steps[-1].round_index
    if terminated:
        if terminal_reward is None:
            raise ValueError("terminal_reward is required when terminated=True")
        if truncation is not None or truncated_bootstrap_v_shaped is not None:
            raise ValueError("truncation / truncated_bootstrap_v_shaped must not be supplied when terminated=True")
        r_term = _require_finite(terminal_reward, "terminal_reward")
        step_rewards[-1] = r_term - float(round_start_potentials[last_k])
        bootstrap_val = 0.0
    else:
        if terminal_reward is not None:
            raise ValueError("terminal_reward must not be supplied when terminated=False")
        step_rewards[-1], bootstrap_val = _resolve_truncation_terminal_step(
            last_k=last_k,
            round_start_potentials=round_start_potentials,
            truncation=truncation,
            truncated_bootstrap_v_shaped=truncated_bootstrap_v_shaped,
        )

    advantages = [0.0] * n_steps
    shaped_returns = [0.0] * n_steps
    gae = 0.0

    for t in reversed(range(n_steps)):
        v_t = steps[t].v_shaped
        if t == n_steps - 1:
            next_val = 0.0 if terminated else bootstrap_val
            non_terminal = 0.0 if terminated else 1.0
        else:
            next_val = steps[t + 1].v_shaped
            non_terminal = 1.0

        delta = step_rewards[t] + gamma * next_val - v_t
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae
        shaped_returns[t] = gae + v_t

    return step_rewards, advantages, shaped_returns


@dataclass(frozen=True)
class GRPCalibrationReport:
    """Contract v2.5 Stage-1/Stage-4 PureStateGRP calibration verification metrics."""

    n_samples: int
    nll: float
    brier_score: float
    ece: float
    classwise_ece: float
    rank_probs_mean: tuple[float, float, float, float]

    def passes_gate(
        self,
        *,
        max_ece: float = 0.03,
        max_classwise_ece: float = 0.03,
        max_brier: float = 0.68,
    ) -> bool:
        """Check whether calibrated GRP meets the pre-RL freeze quality gate."""
        return (
            self.n_samples > 0
            and self.ece <= max_ece
            and self.classwise_ece <= max_classwise_ece
            and self.brier_score <= max_brier
        )


@torch.no_grad()
def evaluate_grp_calibration(
    probs: torch.Tensor,
    final_ranks: torch.Tensor,
    *,
    n_bins: int = 10,
) -> GRPCalibrationReport:
    """Evaluate 4-class GRP calibration (NLL, Brier Score, top-1 ECE, and classwise ECE).

    Args:
        probs: Predicted 4-placement probabilities `[N, 4]`.
        final_ranks: True final placement indices `[N]` in `{0, 1, 2, 3}` (0=1st, 3=4th).
        n_bins: Number of uniform probability bins in `[0, 1]`.
    """
    if probs.ndim != 2 or probs.shape[1] != 4:
        raise ValueError(f"Expected probs shape [N, 4], got {tuple(probs.shape)}")
    if final_ranks.ndim != 1 or final_ranks.shape[0] != probs.shape[0]:
        raise ValueError(f"Expected final_ranks shape [{probs.shape[0]}], got {tuple(final_ranks.shape)}")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    n = int(probs.shape[0])
    if n == 0:
        raise ValueError("Cannot evaluate GRP calibration on empty sample set")

    p = probs.detach().float().clamp(min=1e-7, max=1.0)
    p = p / p.sum(dim=-1, keepdim=True)
    y = final_ranks.detach().long()
    if bool((y < 0).any()) or bool((y > 3).any()):
        raise ValueError("final_ranks must be in {0, 1, 2, 3}")

    # 1. Negative Log-Likelihood (CE)
    nll = float(-torch.log(p[torch.arange(n, device=p.device), y]).mean().item())

    # 2. 4-class Brier Score: mean_i sum_r (p_{i,r} - 1[y_i == r])^2
    one_hot = F.one_hot(y, num_classes=4).float()
    brier_score = float(((p - one_hot) ** 2).sum(dim=-1).mean().item())

    # 3. Top-1 Expected Calibration Error (ECE)
    conf, pred = p.max(dim=-1)
    correct = (pred == y).float()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=p.device)
    ece_val = 0.0
    for b in range(n_bins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]
        in_bin = (conf >= lo) & (conf <= hi) if b == 0 else (conf > lo) & (conf <= hi)
        cnt = int(in_bin.sum().item())
        if cnt > 0:
            avg_conf = float(conf[in_bin].mean().item())
            avg_acc = float(correct[in_bin].mean().item())
            ece_val += (cnt / n) * abs(avg_conf - avg_acc)

    # 4. 4-class marginal Classwise ECE (essential because Phi = sum_r w_r P(r))
    cw_ece_sum = 0.0
    for cls in range(4):
        p_c = p[:, cls]
        y_c = (y == cls).float()
        cls_ece = 0.0
        for b in range(n_bins):
            lo = bin_edges[b]
            hi = bin_edges[b + 1]
            in_bin = (p_c >= lo) & (p_c <= hi) if b == 0 else (p_c > lo) & (p_c <= hi)
            cnt = int(in_bin.sum().item())
            if cnt > 0:
                cls_ece += (cnt / n) * abs(float(p_c[in_bin].mean().item()) - float(y_c[in_bin].mean().item()))
        cw_ece_sum += cls_ece
    classwise_ece = cw_ece_sum / 4.0

    means = p.mean(dim=0).tolist()
    return GRPCalibrationReport(
        n_samples=n,
        nll=nll,
        brier_score=brier_score,
        ece=ece_val,
        classwise_ece=classwise_ece,
        rank_probs_mean=(float(means[0]), float(means[1]), float(means[2]), float(means[3])),
    )


def train_pure_state_grp_step(
    model: PureStateGRP,
    optimizer: torch.optim.Optimizer,
    state_feats: torch.Tensor,
    final_ranks: torch.Tensor,
    *,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    """Perform one supervised 4-class cross-entropy update step on [`PureStateGRP`]."""
    if state_feats.ndim != 2 or state_feats.shape[1] != PURE_STATE_GRP_INPUT_DIM:
        raise ValueError(f"Expected state_feats shape [B, {PURE_STATE_GRP_INPUT_DIM}], got {tuple(state_feats.shape)}")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(state_feats.float())
    loss = F.cross_entropy(logits.float(), final_ranks.long())
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm).item())
    optimizer.step()
    return {"loss": float(loss.item()), "grad_norm": grad_norm}
