"""KAR V2 Hybrid Architecture, Pure-State GRP, and Contract Verification (v2.4)."""

from .grp import (
    FrozenGRP,
    PureStateGRP,
    RoundBoundary,
    compute_hanchan_gae,
    compute_pbrs_round_rewards,
)
from .model import (
    CriticModelConfig,
    KARV2Actor,
    OracleCriticV2,
    RolloutVersionTag,
    V2ModelConfig,
    masked_entropy_and_kl_fp32,
    masked_log_softmax_fp32,
    oracle_critic_loss,
    ppo_actor_loss,
)

__all__ = [
    "CriticModelConfig",
    "FrozenGRP",
    "KARV2Actor",
    "OracleCriticV2",
    "PureStateGRP",
    "RolloutVersionTag",
    "RoundBoundary",
    "V2ModelConfig",
    "compute_hanchan_gae",
    "compute_pbrs_round_rewards",
    "masked_entropy_and_kl_fp32",
    "masked_log_softmax_fp32",
    "oracle_critic_loss",
    "ppo_actor_loss",
]
