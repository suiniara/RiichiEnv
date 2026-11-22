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
from collections.abc import Sequence
from dataclasses import dataclass

import torch
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
            raise ValueError(
                f"GRP checksum mismatch: expected {expected_grp_sha256}, got {actual_sha}"
            )
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


@dataclass(frozen=True)
class RoundBoundary:
    """One decision step inside a Hanchan trajectory for a single seat."""

    round_index: int
    phi_k: float
    v_raw: float

    @property
    def v_shaped(self) -> float:
        return self.v_raw - self.phi_k


def compute_hanchan_gae(
    steps: Sequence[RoundBoundary],
    round_start_potentials: Sequence[float],
    *,
    terminated: bool,
    terminal_reward: float | None = None,
    truncated_bootstrap_v_shaped: float | None = None,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> tuple[list[float], list[float], list[float]]:
    """Compute full-Hanchan continuous GAE (`lambda_round == gae_lambda`) over shaped values.

    Handles ALL Contract v2.4 Section 3.3 boundaries:
    1. Intra-round consecutive steps: r_t = 0.0, bootstrap V_shaped(z_{t+1}).
    2. Cross-round normal transitions (k -> k+1): r_t = Phi_{k+1} - Phi_k.
    3. Zero-decision skipped rounds (seat had 0 decisions in round k+1, e.g. opponent Tenhou):
       potential jump Phi_{next_round} - Phi_{prev_round} is cleanly merged onto the last
       active decision before the gap, preserving exact telescoping.
    4. Hanchan termination (`terminated=True`): last step gets `R_terminal - Phi_{last_round}`
       and `next_value = 0.0` (the ONLY place `next_value == 0.0`).
    5. Mid-Hanchan rollout truncation (`terminated=False`, `truncated=True`):
       MUST supply `truncated_bootstrap_v_shaped` (NEVER silently zeroed!).

    Returns `(step_rewards, advantages, shaped_gae_returns)`.
    """
    n_steps = len(steps)
    if n_steps == 0:
        return [], [], []
    if terminated and terminal_reward is None:
        raise ValueError("terminal_reward is required when terminated=True")
    if not terminated and truncated_bootstrap_v_shaped is None:
        raise ValueError(
            "truncated_bootstrap_v_shaped is required when terminated=False "
            "(Contract v2.4: truncated rollouts MUST NOT bootstrap with 0.0)"
        )

    # Validate monotone round indices and assign exact telescoping step rewards.
    step_rewards = [0.0] * n_steps
    for t in range(n_steps - 1):
        cur_k = steps[t].round_index
        nxt_k = steps[t + 1].round_index
        if nxt_k < cur_k:
            raise ValueError(f"Non-monotone round indices: {cur_k} -> {nxt_k}")
        if nxt_k > cur_k:
            # Automatically merges any zero-decision skipped rounds between cur_k and nxt_k!
            step_rewards[t] = float(round_start_potentials[nxt_k]) - float(
                round_start_potentials[cur_k]
            )

    last_k = steps[-1].round_index
    if terminated:
        assert terminal_reward is not None
        # Also merges any trailing zero-decision rounds before Hanchan end:
        # (Phi_{K-1} - Phi_{last_k}) + (R_terminal - Phi_{K-1}) == R_terminal - Phi_{last_k}!
        step_rewards[-1] = float(terminal_reward) - float(round_start_potentials[last_k])
    else:
        # Mid-Hanchan truncation inside round `last_k` (no round-end jump unless caller ended at boundary)
        step_rewards[-1] = 0.0

    advantages = [0.0] * n_steps
    shaped_returns = [0.0] * n_steps
    gae = 0.0

    for t in reversed(range(n_steps)):
        v_t = steps[t].v_shaped
        if t == n_steps - 1:
            if terminated:
                next_val = 0.0
                non_terminal = 0.0
            else:
                assert truncated_bootstrap_v_shaped is not None
                next_val = float(truncated_bootstrap_v_shaped)
                non_terminal = 1.0
        else:
            next_val = steps[t + 1].v_shaped
            non_terminal = 1.0

        delta = step_rewards[t] + gamma * next_val - v_t
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[t] = gae
        shaped_returns[t] = gae + v_t

    return step_rewards, advantages, shaped_returns
