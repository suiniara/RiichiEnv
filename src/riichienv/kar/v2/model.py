"""KAR V2 Hybrid Architecture & Completely Independent Oracle Critic (Contract v2.4).

Supports all four Stage-2/3/4 experimental presets via `V2ModelConfig`:
- `spatial_only` (Model B): 12 Spatial ResBlocks, 0 Event layers, Factorized Head
- `v2_base`      (Model C1): 6 Spatial, 4 Event, 2 Fusion, Factorized Head
- `v2_base_flat` (Model C1-Flat): 6 Spatial, 4 Event, 2 Fusion, Flat 546 Head
- `v2_large`     (Model C2): 8 Spatial, 6 Event, 2 Fusion, Factorized Head

Contract v2.4 guarantees enforced in code:
- `NULL_TILE` (index 34) and `NULL_SOURCE` (index 4) learnable embeddings + `has_slot_flags`
- `OracleCriticV2` receives ONLY raw observations + oracle features (ZERO Actor hidden states!)
- `fp32` upcast for all masked log-softmax, KL, entropy, PPO ratio, and Critic value math
- `K == 1` forced steps excluded from Actor PPO/KL/Entropy gradients via `has_choice = (K >= 2)`
- Critic RL loss uses `CE + alpha_td * MSE(V_shaped, target)` supporting both `gae_mse` and `mc_mse`
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from riichienv.kar.actions import ACTION_COUNT, ACTION_MANIFEST, ACTION_MANIFEST_SHA256

TILE_DIM = 34
TILE_FEATS_DIM = 64
EVENT_BYTES = 10
MAX_EVENT_LEN = 256
GLOBAL_FEATS_DIM = 48
PLAYER_FEATS_DIM = 32
MAX_CANDIDATES = 32
CAND_PAD_ID = 546
CAND_MODE_COUNT = 7
CAND_DYN_DIM = 8
ORACLE_TILE_DIM = 32
ORACLE_PLAYER_DIM = 12
NULL_TILE_IDX = 34
NULL_SOURCE_IDX = 4

_ACTION_TYPE_TO_ID: dict[str, int] = {
    "DISCARD": 0,
    "ANKAN": 1,
    "KAKAN": 2,
    "TSUMO": 3,
    "KYUSHU_KYUHAI": 4,
    "PASS": 5,
    "CHI": 6,
    "PON": 7,
    "DAIMINKAN": 8,
    "RON": 9,
}


def _is_red5_tile37(tile37: int | None) -> bool:
    return tile37 in (0, 10, 20)


def _tile37_to_base34(tile37: int) -> int:
    if tile37 >= 30:
        return tile37 - 3
    suit, rank = divmod(tile37, 10)
    return suit * 9 + (4 if rank == 0 else rank - 1)


def build_static_manifest_buffers() -> dict[str, torch.Tensor]:
    """Build the 547-row static ACTION_MANIFEST lookup tensors (row 546 = PAD)."""
    action_type = torch.full((ACTION_COUNT + 1,), 10, dtype=torch.long)
    target_base34 = torch.full((ACTION_COUNT + 1,), NULL_TILE_IDX, dtype=torch.long)
    c0_base34 = torch.full((ACTION_COUNT + 1,), NULL_TILE_IDX, dtype=torch.long)
    c1_base34 = torch.full((ACTION_COUNT + 1,), NULL_TILE_IDX, dtype=torch.long)
    source_rel = torch.full((ACTION_COUNT + 1,), NULL_SOURCE_IDX, dtype=torch.long)
    has_slot_flags = torch.zeros((ACTION_COUNT + 1, 4), dtype=torch.float32)
    static_flags = torch.zeros((ACTION_COUNT + 1, 4), dtype=torch.float32)

    for spec in ACTION_MANIFEST:
        gid = spec.global_id
        action_type[gid] = _ACTION_TYPE_TO_ID[spec.action_type]
        if spec.base34 is not None:
            target_base34[gid] = spec.base34
            has_slot_flags[gid, 0] = 1.0
        if len(spec.consumed_tile37) >= 1:
            c0_base34[gid] = _tile37_to_base34(spec.consumed_tile37[0])
            has_slot_flags[gid, 1] = 1.0
        if len(spec.consumed_tile37) >= 2:
            c1_base34[gid] = _tile37_to_base34(spec.consumed_tile37[1])
            has_slot_flags[gid, 2] = 1.0
        if spec.source is not None:
            source_rel[gid] = spec.source
            has_slot_flags[gid, 3] = 1.0

        static_flags[gid, 0] = 1.0 if _is_red5_tile37(spec.tile37) else 0.0
        static_flags[gid, 1] = (
            1.0 if any(_is_red5_tile37(t) for t in spec.consumed_tile37) else 0.0
        )
        static_flags[gid, 2] = 1.0 if spec.tsumogiri else 0.0
        static_flags[gid, 3] = 1.0 if spec.riichi else 0.0

    return {
        "action_type": action_type,
        "target_base34": target_base34,
        "c0_base34": c0_base34,
        "c1_base34": c1_base34,
        "source_rel": source_rel,
        "has_slot_flags": has_slot_flags,
        "static_flags": static_flags,
    }


@dataclass(frozen=True)
class V2ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    n_spatial_layers: int = 6
    n_event_layers: int = 4
    n_fusion_layers: int = 2
    head_type: Literal["factorized", "flat546"] = "factorized"
    enable_belief_heads: bool = True
    dropout: float = 0.0

    @classmethod
    def spatial_only(cls) -> V2ModelConfig:
        """Model B: Pure 34-tile Spatial ResNet + Factorized Head (0 Event layers)."""
        return cls(
            d_model=256,
            n_heads=8,
            n_spatial_layers=12,
            n_event_layers=0,
            n_fusion_layers=1,
            head_type="factorized",
        )

    @classmethod
    def v2_base(cls) -> V2ModelConfig:
        """Model C1: V2-Base Hybrid (6 Spatial, 4 Event, 2 Fusion)."""
        return cls(
            d_model=256,
            n_heads=8,
            n_spatial_layers=6,
            n_event_layers=4,
            n_fusion_layers=2,
            head_type="factorized",
        )

    @classmethod
    def v2_base_flat(cls) -> V2ModelConfig:
        """Model C1-Flat: Ablation with flat 546 head."""
        return cls(
            d_model=256,
            n_heads=8,
            n_spatial_layers=6,
            n_event_layers=4,
            n_fusion_layers=2,
            head_type="flat546",
        )

    @classmethod
    def v2_large(cls) -> V2ModelConfig:
        """Model C2: V2-Large Hybrid (8 Spatial, 6 Event, 2 Fusion)."""
        return cls(
            d_model=384,
            n_heads=8,
            n_spatial_layers=8,
            n_event_layers=6,
            n_fusion_layers=2,
            head_type="factorized",
        )


@dataclass(frozen=True)
class CriticModelConfig:
    """Independent Oracle Critic configuration (zero shared weights/hidden states with Actor)."""

    d_model: int = 192
    n_heads: int = 6
    n_spatial_layers: int = 4
    n_event_layers: int = 2
    n_fusion_layers: int = 1
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(math.ceil((8 * dim / 3) / 16) * 16)
        self.w_gate_up = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.drop(self.w_down(F.silu(gate) * up))


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, L, D], cos/sin: [1, 1, L, D//2]
    x1, x2 = x[..., ::2], x[..., 1::2]
    rot = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rot.flatten(-2)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self,
        q_in: torch.Tensor,
        kv_in: torch.Tensor,
        *,
        is_causal: bool = False,
        key_valid_mask: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, q_len, dim = q_in.shape
        kv_len = kv_in.shape[1]
        q = self.q_proj(q_in).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_in).view(bsz, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_in).view(bsz, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            q = _apply_rope(q, rope_cos[:, :, :q_len, :], rope_sin[:, :, :q_len, :])
            k = _apply_rope(k, rope_cos[:, :, :kv_len, :], rope_sin[:, :, :kv_len, :])

        attn_mask: torch.Tensor | None = None
        if key_valid_mask is not None:
            # key_valid_mask: [B, kv_len] bool (True = valid key)
            # Build explicit boolean mask [B, 1, q_len, kv_len]
            valid = key_valid_mask[:, None, None, :].expand(bsz, 1, q_len, kv_len)
            if is_causal:
                causal = torch.ones((q_len, kv_len), dtype=torch.bool, device=q.device).tril()
                valid = valid & causal[None, None, :, :]
            # Guard against any fully-padded query row producing NaN: unmask key 0 for padded slots
            row_has_any = valid.any(dim=-1, keepdim=True)
            safe_valid = torch.where(row_has_any, valid, torch.zeros_like(valid))
            if not bool(row_has_any.all()):
                fallback = torch.zeros_like(safe_valid)
                fallback[..., 0] = True
                safe_valid = torch.where(row_has_any, safe_valid, fallback)
            attn_mask = safe_valid
            use_causal = False
        else:
            use_causal = is_causal

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=use_causal,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, dim)
        return self.o_proj(out)


class IntraSuitResBlock(nn.Module):
    """Mahjong-specific spatial block: zero cross-suit convolution bleed + Sanshoku/Flush pool + FiLM."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.film = nn.Linear(d_model, d_model * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        # Shared intra-suit convolutions on [B*3, d, 9] (never crosses 9m->1p->9p->1s->E!)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)
        self.suit_mix = nn.Linear(d_model * 4, d_model, bias=False)

        # Dedicated honor branch [B, 7, d] (winds 4 + dragons 3 group interaction)
        self.honor_attn = MultiHeadAttention(d_model, max(1, n_heads // 2), dropout=dropout)
        self.honor_proj = nn.Linear(d_model * 2, d_model, bias=False)

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, 34, d], cond: [B, d]
        bsz, _, dim = x.shape
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = self.norm1(x) * (1.0 + gamma[:, None, :]) + beta[:, None, :]

        # Suited tiles [B, 27, d] -> [B, 3, 9, d]
        suited = h[:, :27, :].view(bsz, 3, 9, dim)
        suited_c = suited.reshape(bsz * 3, 9, dim).transpose(1, 2)  # [B*3, d, 9]
        c3 = self.conv3(suited_c).transpose(1, 2).view(bsz, 3, 9, dim)
        c5 = self.conv5(suited_c).transpose(1, 2).view(bsz, 3, 9, dim)
        # Same-rank cross-suit pool (Sanshoku bias) & per-suit pool (Honitsu/Chinitsu bias)
        same_rank = suited.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        per_suit = suited.mean(dim=2, keepdim=True).expand(-1, -1, 9, -1)
        suited_out = self.suit_mix(torch.cat((c3, c5, same_rank, per_suit), dim=-1)).view(
            bsz, 27, dim
        )

        # Honor tiles [B, 7, d] (winds[0..4] + dragons[4..7])
        honors = h[:, 27:, :]
        h_attn = self.honor_attn(honors, honors)
        winds_pool = honors[:, :4, :].mean(dim=1, keepdim=True).expand(-1, 4, -1)
        dragons_pool = honors[:, 4:, :].mean(dim=1, keepdim=True).expand(-1, 3, -1)
        group_pool = torch.cat((winds_pool, dragons_pool), dim=1)
        honor_out = self.honor_proj(torch.cat((h_attn, group_pool), dim=-1))

        x = x + torch.cat((suited_out, honor_out), dim=1)
        x = x + self.ffn(self.norm2(x))
        return x


class IntraSuitSpatialEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_channels, d_model)
        self.suit_emb = nn.Embedding(4, d_model)
        self.rank_emb = nn.Embedding(16, d_model)

        suit_ids = [0] * 9 + [1] * 9 + [2] * 9 + [3] * 7
        rank_ids = list(range(9)) * 3 + list(range(9, 16))
        self.register_buffer("tile_suit_ids", torch.tensor(suit_ids, dtype=torch.long))
        self.register_buffer("tile_rank_ids", torch.tensor(rank_ids, dtype=torch.long))

        self.blocks = nn.ModuleList(
            [IntraSuitResBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.out_norm = RMSNorm(d_model)

    def forward(self, tile_feats: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        pos = self.suit_emb(self.tile_suit_ids) + self.rank_emb(self.tile_rank_ids)
        h = self.in_proj(tile_feats) + pos[None, :, :]
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out_norm(h)


class EventTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            self.norm1(x),
            is_causal=True,
            key_valid_mask=valid_mask,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        x = x + self.ffn(self.norm2(x))
        return x


class CausalEventTransformer(nn.Module):
    """Causal 1D-RoPE Event Transformer over 10-byte observer-visible round events (`1 <= L_ev <= 256`)."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.emb_kind = nn.Embedding(12, d_model)
        self.emb_actor = nn.Embedding(5, d_model)
        self.emb_source = nn.Embedding(5, d_model)
        self.emb_tile = nn.Embedding(38, d_model)
        self.emb_c0 = nn.Embedding(38, d_model)
        self.emb_c1 = nn.Embedding(38, d_model)
        self.proj_bits = nn.Linear(13, d_model, bias=False)  # 5 flags + 4 sub_mask + 2 norm indices + 2 sin/cos
        self.in_norm = RMSNorm(d_model)

        head_dim = d_model // n_heads
        inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(MAX_EVENT_LEN).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("rope_cos", freqs.cos()[None, None, :, :], persistent=False)
        self.register_buffer("rope_sin", freqs.sin()[None, None, :, :], persistent=False)

        self.layers = nn.ModuleList(
            [EventTransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
        )
        self.out_norm = RMSNorm(d_model)

    def forward(self, events: torch.Tensor, event_len: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # events: [B, L, 10] uint8/long, event_len: [B] int
        bsz, seq_len, n_bytes = events.shape
        if n_bytes != EVENT_BYTES:
            raise ValueError(f"Expected {EVENT_BYTES} bytes per event, got {n_bytes}")
        if seq_len < 1 or seq_len > MAX_EVENT_LEN:
            raise ValueError(
                f"Contract v2.4 violation: event sequence length {seq_len} outside [1, {MAX_EVENT_LEN}]"
            )
        if bool((event_len < 1).any()) or bool((event_len > MAX_EVENT_LEN).any()):
            raise ValueError("Contract v2.4 violation: event_len must satisfy 1 <= L_ev <= 256")

        ev = events.long()
        h = (
            self.emb_kind(ev[..., 0].clamp(0, 11))
            + self.emb_actor(ev[..., 1].clamp(0, 4))
            + self.emb_source(ev[..., 2].clamp(0, 4))
            + self.emb_tile(ev[..., 3].clamp(0, 37))
            + self.emb_c0(ev[..., 4].clamp(0, 37))
            + self.emb_c1(ev[..., 5].clamp(0, 37))
        )

        flags = ev[..., 6]
        sub_mask = ev[..., 7]
        turn_norm = ev[..., 8].float() / 100.0
        seq_norm = ev[..., 9].float() / 24.0
        bit_feats = torch.stack(
            [
                ((flags >> 0) & 1).float(),
                ((flags >> 1) & 1).float(),
                ((flags >> 2) & 1).float(),
                ((flags >> 3) & 1).float(),
                ((flags >> 4) & 1).float(),
                ((sub_mask >> 0) & 1).float(),
                ((sub_mask >> 1) & 1).float(),
                ((sub_mask >> 2) & 1).float(),
                ((sub_mask >> 3) & 1).float(),
                turn_norm,
                seq_norm,
                torch.sin(turn_norm * math.pi),
                torch.cos(turn_norm * math.pi),
            ],
            dim=-1,
        ).to(h.dtype)
        h = self.in_norm(h + self.proj_bits(bit_feats))

        pos_idx = torch.arange(seq_len, device=events.device)[None, :]
        valid_mask = pos_idx < event_len[:, None]

        cos = self.rope_cos.to(dtype=h.dtype)
        sin = self.rope_sin.to(dtype=h.dtype)
        for layer in self.layers:
            h = layer(h, valid_mask, cos, sin)
        return self.out_norm(h), valid_mask


class FusionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, has_events: bool, dropout: float = 0.0) -> None:
        super().__init__()
        self.has_events = has_events
        if has_events:
            self.cross_norm_q = RMSNorm(d_model)
            self.cross_norm_kv = RMSNorm(d_model)
            self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.self_norm = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)

    def forward(
        self,
        core: torch.Tensor,
        events_h: torch.Tensor | None = None,
        events_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.has_events and events_h is not None:
            core = core + self.cross_attn(
                self.cross_norm_q(core),
                self.cross_norm_kv(events_h),
                key_valid_mask=events_mask,
            )
        n_core = self.self_norm(core)
        core = core + self.self_attn(n_core, n_core)
        core = core + self.ffn(self.ffn_norm(core))
        return core


class FactorizedCandidateHead(nn.Module):
    """Structured 546-action candidate head with learnable NULL_TILE / NULL_SOURCE embeddings."""

    manifest_action_type: torch.Tensor
    manifest_target_base34: torch.Tensor
    manifest_c0_base34: torch.Tensor
    manifest_c1_base34: torch.Tensor
    manifest_source_rel: torch.Tensor
    manifest_has_slot_flags: torch.Tensor
    manifest_static_flags: torch.Tensor

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.manifest_sha256 = ACTION_MANIFEST_SHA256
        buffers = build_static_manifest_buffers()
        for name, tensor in buffers.items():
            self.register_buffer(f"manifest_{name}", tensor)

        # Contract v2.4 P1-3: explicit learnable NULL slots (index 34 for tile, 4 for source)
        self.null_tile_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        self.null_source_emb = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.null_tile_emb, std=0.02)
        nn.init.normal_(self.null_source_emb, std=0.02)

        self.emb_action_type = nn.Embedding(11, 32)
        self.emb_cand_mode = nn.Embedding(CAND_MODE_COUNT, 16)

        in_dim = d_model * 4 + 32 + 16 + 4 + 4 + CAND_DYN_DIM
        self.cand_in = nn.Linear(in_dim, d_model)
        self.cross_norm_q = RMSNorm(d_model)
        self.cross_norm_kv = RMSNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.self_norm = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout=dropout)
        self.out_norm = RMSNorm(d_model)
        self.logit_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        core: torch.Tensor,
        h_player: torch.Tensor,
        h_tile: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_mask: torch.Tensor,
        cand_mode: torch.Tensor,
        cand_dyn: torch.Tensor,
    ) -> torch.Tensor:
        bsz, k_max = cand_ids.shape
        safe_ids = torch.where(cand_mask, cand_ids.long().clamp(0, ACTION_COUNT), CAND_PAD_ID)

        # Append NULL_TILE (idx 34) & NULL_SOURCE (idx 4) -> zero out-of-bounds access!
        h_tile_ext = torch.cat(
            (h_tile, self.null_tile_emb.expand(bsz, 1, -1).to(h_tile.dtype)), dim=1
        )  # [B, 35, d]
        h_player_ext = torch.cat(
            (h_player, self.null_source_emb.expand(bsz, 1, -1).to(h_player.dtype)), dim=1
        )  # [B, 5, d]

        t_idx = self.manifest_target_base34[safe_ids]  # [B, K] in [0..34]
        c0_idx = self.manifest_c0_base34[safe_ids]  # [B, K] in [0..34]
        c1_idx = self.manifest_c1_base34[safe_ids]  # [B, K] in [0..34]
        s_idx = self.manifest_source_rel[safe_ids]  # [B, K] in [0..4]

        batch_idx = torch.arange(bsz, device=cand_ids.device)[:, None]
        h_t = h_tile_ext[batch_idx, t_idx]
        h_c0 = h_tile_ext[batch_idx, c0_idx]
        h_c1 = h_tile_ext[batch_idx, c1_idx]
        h_src = h_player_ext[batch_idx, s_idx]

        e_type = self.emb_action_type(self.manifest_action_type[safe_ids])
        e_mode = self.emb_cand_mode(cand_mode.long().clamp(0, CAND_MODE_COUNT - 1))
        has_slots = self.manifest_has_slot_flags[safe_ids].to(h_t.dtype)
        s_flags = self.manifest_static_flags[safe_ids].to(h_t.dtype)

        q = self.cand_in(
            torch.cat(
                (h_t, h_c0, h_c1, h_src, e_type, e_mode, has_slots, s_flags, cand_dyn.to(h_t.dtype)),
                dim=-1,
            )
        )
        q = q + self.cross_attn(self.cross_norm_q(q), self.cross_norm_kv(core))
        n_q = self.self_norm(q)
        q = q + self.self_attn(n_q, n_q, key_valid_mask=cand_mask)
        q = q + self.ffn(self.ffn_norm(q))
        logits = self.logit_proj(self.out_norm(q)).squeeze(-1)  # [B, K]
        return logits


class Flat546CandidateHead(nn.Module):
    """Experiment C ablation head: flat 546-class classifier gathered at legal `cand_ids`."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.proj = nn.Linear(d_model, ACTION_COUNT + 1)

    def forward(
        self,
        core: torch.Tensor,
        h_player: torch.Tensor,
        h_tile: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_mask: torch.Tensor,
        cand_mode: torch.Tensor,
        cand_dyn: torch.Tensor,
    ) -> torch.Tensor:
        del h_player, h_tile, cand_mode, cand_dyn
        pooled = self.norm(core[:, 0, :])
        all_logits = self.proj(pooled)  # [B, 547]
        safe_ids = torch.where(cand_mask, cand_ids.long().clamp(0, ACTION_COUNT), CAND_PAD_ID)
        return all_logits.gather(dim=-1, index=safe_ids)


class OpponentBeliefHeads(nn.Module):
    """Stage-3 multi-task opponent belief auxiliary heads (strictly optional at inference)."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.tenpai_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.wait_p = nn.Linear(d_model, d_model, bias=False)
        self.wait_t = nn.Linear(d_model, d_model, bias=False)
        self.dealin_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 1),
        )
        self.hand_cnt_head = nn.Linear(d_model, 5)
        self.scale = 1.0 / math.sqrt(d_model)

    def forward(
        self, h_global: torch.Tensor, h_player: torch.Tensor, h_tile: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        opps = h_player[:, 1:4, :]  # [B, 3, d]
        g_exp = h_global[:, None, :].expand(-1, 3, -1)
        tenpai_logits = self.tenpai_head(torch.cat((opps, g_exp), dim=-1)).squeeze(-1)  # [B, 3]

        wait_logits = (
            torch.einsum("bpd,btd->bpt", self.wait_p(opps), self.wait_t(h_tile)) * self.scale
        )  # [B, 3, 34]

        opp_pool = opps.mean(dim=1, keepdim=True).expand(-1, TILE_DIM, -1)
        dealin_logits = self.dealin_head(torch.cat((h_tile, opp_pool), dim=-1)).squeeze(-1)  # [B, 34]

        pair = opps[:, :, None, :] + h_tile[:, None, :, :]  # [B, 3, 34, d]
        hand_cnt_logits = self.hand_cnt_head(F.silu(pair))  # [B, 3, 34, 5]
        return {
            "tenpai_logits": tenpai_logits,
            "wait_logits": wait_logits,
            "dealin_logits": dealin_logits,
            "hand_cnt_logits": hand_cnt_logits,
        }


class KARV2Actor(nn.Module):
    """KAR V2.4 Hybrid Policy Actor (pure legal-observation forward path, zero Oracle leakage)."""

    def __init__(self, config: V2ModelConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model

        self.global_proj = nn.Linear(GLOBAL_FEATS_DIM, d)
        self.player_proj = nn.Linear(PLAYER_FEATS_DIM, d)
        self.cond_proj = nn.Linear(GLOBAL_FEATS_DIM + PLAYER_FEATS_DIM, d)

        self.spatial = IntraSuitSpatialEncoder(
            TILE_FEATS_DIM, d, config.n_heads, config.n_spatial_layers, dropout=config.dropout
        )
        self.event_encoder = (
            CausalEventTransformer(d, config.n_heads, config.n_event_layers, dropout=config.dropout)
            if config.n_event_layers > 0
            else None
        )
        self.fusion_layers = nn.ModuleList(
            [
                FusionBlock(
                    d,
                    config.n_heads,
                    has_events=self.event_encoder is not None,
                    dropout=config.dropout,
                )
                for _ in range(config.n_fusion_layers)
            ]
        )

        if config.head_type == "factorized":
            self.candidate_head: nn.Module = FactorizedCandidateHead(
                d, config.n_heads, dropout=config.dropout
            )
        else:
            self.candidate_head = Flat546CandidateHead(d)

        self.belief_heads = OpponentBeliefHeads(d) if config.enable_belief_heads else None

    def forward(
        self,
        *,
        tile_feats: torch.Tensor,
        events: torch.Tensor,
        event_len: torch.Tensor,
        global_feats: torch.Tensor,
        player_feats: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_mask: torch.Tensor,
        cand_mode: torch.Tensor,
        cand_dyn: torch.Tensor,
        return_belief: bool = False,
    ) -> dict[str, torch.Tensor]:
        cond = self.cond_proj(torch.cat((global_feats, player_feats[:, 0, :]), dim=-1))
        h_tile = self.spatial(tile_feats, cond)
        h_global = self.global_proj(global_feats)[:, None, :]
        h_player = self.player_proj(player_feats)
        core = torch.cat((h_global, h_player, h_tile), dim=1)  # [B, 39, d]

        events_h: torch.Tensor | None = None
        events_mask: torch.Tensor | None = None
        if self.event_encoder is not None:
            events_h, events_mask = self.event_encoder(events, event_len)

        for layer in self.fusion_layers:
            core = layer(core, events_h, events_mask)

        f_global = core[:, 0, :]
        f_player = core[:, 1:5, :]
        f_tile = core[:, 5:, :]

        logits = self.candidate_head(
            core, f_player, f_tile, cand_ids, cand_mask, cand_mode, cand_dyn
        )
        out: dict[str, torch.Tensor] = {"logits": logits}
        if return_belief and self.belief_heads is not None:
            out.update(self.belief_heads(f_global, f_player, f_tile))
        return out


class OracleCriticV2(nn.Module):
    """Completely Independent Asymmetric Oracle Critic (Contract v2.4 P0-2).

    STRICT INVARIANT:
    - Receives ONLY raw public tensors (`tile_feats`, `events`, `event_len`,
      `global_feats`, `player_feats`) + training-only Oracle tensors
      (`oracle_tile`, `oracle_player`) + frozen GRP round-start prior `grp_prior_probs`.
    - ZERO dependency on Actor parameters or Actor dynamic hidden representations!
    """

    def __init__(self, config: CriticModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or CriticModelConfig()
        self.config = cfg
        d = cfg.d_model

        self.global_proj = nn.Linear(GLOBAL_FEATS_DIM, d)
        self.player_proj = nn.Linear(PLAYER_FEATS_DIM + ORACLE_PLAYER_DIM, d)
        self.cond_proj = nn.Linear(GLOBAL_FEATS_DIM + PLAYER_FEATS_DIM, d)

        self.spatial = IntraSuitSpatialEncoder(
            TILE_FEATS_DIM + ORACLE_TILE_DIM,
            d,
            cfg.n_heads,
            cfg.n_spatial_layers,
            dropout=cfg.dropout,
        )
        self.event_encoder = (
            CausalEventTransformer(d, cfg.n_heads, cfg.n_event_layers, dropout=cfg.dropout)
            if cfg.n_event_layers > 0
            else None
        )
        self.fusion_layers = nn.ModuleList(
            [
                FusionBlock(
                    d,
                    cfg.n_heads,
                    has_events=self.event_encoder is not None,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.n_fusion_layers)
            ]
        )
        self.out_norm = RMSNorm(d)
        self.rank_residual_in = nn.Linear(d * 2, d)
        self.rank_residual_out = nn.Linear(d, 4)
        # Initialize residual head near zero so initial P_phi starts near frozen GRP prior!
        nn.init.zeros_(self.rank_residual_out.weight)
        nn.init.zeros_(self.rank_residual_out.bias)

    def forward(
        self,
        *,
        tile_feats: torch.Tensor,
        oracle_tile: torch.Tensor,
        events: torch.Tensor,
        event_len: torch.Tensor,
        global_feats: torch.Tensor,
        player_feats: torch.Tensor,
        oracle_player: torch.Tensor,
        grp_prior_probs: torch.Tensor,
        rank_payoffs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz = tile_feats.shape[0]
        cond = self.cond_proj(torch.cat((global_feats, player_feats[:, 0, :]), dim=-1))

        # Concatenate raw public + oracle spatial planes -> [B, 34, 96]
        tile_in = torch.cat((tile_feats, oracle_tile), dim=-1)
        h_tile = self.spatial(tile_in, cond)

        # Pad Self (seat 0) with zeros for oracle_player -> [B, 4, 32 + 12]
        zero_self_oracle = torch.zeros(
            (bsz, 1, ORACLE_PLAYER_DIM), dtype=oracle_player.dtype, device=oracle_player.device
        )
        opp_oracle_full = torch.cat((zero_self_oracle, oracle_player), dim=1)
        h_player = self.player_proj(torch.cat((player_feats, opp_oracle_full), dim=-1))
        h_global = self.global_proj(global_feats)[:, None, :]
        core = torch.cat((h_global, h_player, h_tile), dim=1)

        events_h: torch.Tensor | None = None
        events_mask: torch.Tensor | None = None
        if self.event_encoder is not None:
            events_h, events_mask = self.event_encoder(events, event_len)

        for layer in self.fusion_layers:
            core = layer(core, events_h, events_mask)

        pooled = torch.cat((self.out_norm(core[:, 0, :]), self.out_norm(core[:, 5:, :].mean(dim=1))), dim=-1)
        res_logits = self.rank_residual_out(F.silu(self.rank_residual_in(pooled))).float()

        prior_probs_f32 = grp_prior_probs.float().clamp(min=1e-6)
        prior_logits = torch.log(prior_probs_f32)
        rank_logits = prior_logits + res_logits
        rank_probs = torch.softmax(rank_logits, dim=-1)

        w = rank_payoffs.float()
        if w.ndim == 1:
            w = w[None, :]
        phi_k = (prior_probs_f32 * w).sum(dim=-1)
        v_raw = (rank_probs * w).sum(dim=-1)
        v_shaped = v_raw - phi_k
        return {
            "rank_logits": rank_logits,
            "rank_probs": rank_probs,
            "v_raw": v_raw,
            "phi_k": phi_k,
            "v_shaped": v_shaped,
        }


# =========================================================================
# Contract v2.4 Section 3.3 & 3.4: fp32 Masked Policy / KL / PPO / Critic Losses
# =========================================================================


def masked_log_softmax_fp32(logits: torch.Tensor, cand_mask: torch.Tensor) -> torch.Tensor:
    """Numerically safe `fp32` masked log-softmax over valid candidates (`~cand_mask -> -inf`)."""
    logits_f32 = logits.float()
    masked = logits_f32.masked_fill(~cand_mask, float("-inf"))
    return F.log_softmax(masked, dim=-1)


def masked_entropy_and_kl_fp32(
    logits: torch.Tensor,
    cand_mask: torch.Tensor,
    ref_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute exact `fp32` masked entropy & optional KL(pi || pi_ref) with zero `0 * -inf = NaN`."""
    log_p = masked_log_softmax_fp32(logits, cand_mask)
    p = torch.where(cand_mask, torch.exp(log_p), torch.zeros_like(log_p))
    ent_terms = torch.where(cand_mask, -p * log_p, torch.zeros_like(log_p))
    entropy = ent_terms.sum(dim=-1)

    if ref_logits is None:
        return entropy, None

    log_q = masked_log_softmax_fp32(ref_logits, cand_mask)
    kl_terms = torch.where(cand_mask, p * (log_p - log_q), torch.zeros_like(log_p))
    kl = kl_terms.sum(dim=-1)
    return entropy, kl


def ppo_actor_loss(
    *,
    logits: torch.Tensor,
    cand_ids: torch.Tensor,
    cand_mask: torch.Tensor,
    chosen_cand_idx: torch.Tensor,
    chosen_global_id: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    is_learner_seat: torch.Tensor,
    ref_bc_logits: torch.Tensor | None = None,
    clip_eps: float = 0.2,
    beta_kl: float = 0.02,
    beta_ent: float = 0.005,
) -> dict[str, torch.Tensor]:
    """Contract v2.4 Section 3.4 PPO actor loss with strict index check & `has_choice (K >= 2)` guard."""
    bsz = logits.shape[0]
    batch_idx = torch.arange(bsz, device=logits.device)
    selected_gid = cand_ids[batch_idx, chosen_cand_idx]
    selected_valid = cand_mask[batch_idx, chosen_cand_idx]
    if not bool((selected_gid == chosen_global_id).all()):
        raise ValueError("Contract v2.4 violation: cand_ids[chosen_cand_idx] != chosen_global_id")
    if not bool(selected_valid.all()):
        raise ValueError("Contract v2.4 violation: chosen_cand_idx points to masked candidate")

    log_p = masked_log_softmax_fp32(logits, cand_mask)
    new_log_prob = log_p[batch_idx, chosen_cand_idx]
    entropy, kl = masked_entropy_and_kl_fp32(logits, cand_mask, ref_bc_logits)

    has_choice = cand_mask.sum(dim=-1) >= 2
    active = is_learner_seat.bool() & has_choice
    active_f = active.float()
    denom = active_f.sum().clamp(min=1.0)

    # Normalize advantages strictly within active M = {is_learner_seat & has_choice}
    adv = advantages.float()
    if int(active.sum().item()) > 1:
        adv_active = adv[active]
        adv = (adv - adv_active.mean()) / (adv_active.std(unbiased=False) + 1e-8)

    log_ratio = (new_log_prob - old_log_prob.float()).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_term = -torch.minimum(surr1, surr2)

    kl_term = kl if kl is not None else torch.zeros_like(entropy)
    per_step = policy_term + beta_kl * kl_term - beta_ent * entropy
    loss = (per_step * active_f).sum() / denom
    return {
        "loss": loss,
        "policy_loss": (policy_term * active_f).sum() / denom,
        "kl": (kl_term * active_f).sum() / denom,
        "entropy": (entropy * active_f).sum() / denom,
        "active_count": active_f.sum(),
    }


@dataclass(frozen=True)
class RolloutVersionTag:
    """Contract v2.4.1 version provenance bound to every PPO rollout trajectory.

    GAE targets MUST be computed once using frozen rollout-time Critic predictions
    under `critic_version`, never recomputed across PPO mini-batch epochs.
    """

    policy_version: int
    critic_version: int
    grp_sha256: str
    ruleset_sha256: str

    def validate_compatible(
        self,
        *,
        current_policy_version: int,
        expected_critic_version: int | None = None,
        expected_grp_sha256: str,
        expected_ruleset_sha256: str,
        max_staleness: int = 2,
    ) -> None:
        if self.grp_sha256 != expected_grp_sha256:
            raise ValueError(f"GRP mismatch: rollout={self.grp_sha256} vs learner={expected_grp_sha256}")
        if self.ruleset_sha256 != expected_ruleset_sha256:
            raise ValueError(
                f"Ruleset mismatch: rollout={self.ruleset_sha256} vs learner={expected_ruleset_sha256}"
            )
        if expected_critic_version is not None and self.critic_version != expected_critic_version:
            raise ValueError(
                f"Mixed critic_version within trajectory: got {self.critic_version}, expected {expected_critic_version}"
            )
        if current_policy_version - self.policy_version > max_staleness:
            raise ValueError(
                f"Stale rollout policy_version={self.policy_version} "
                f"(current={current_policy_version}, max_staleness={max_staleness})"
            )


def oracle_critic_loss(
    *,
    rank_logits: torch.Tensor,
    v_shaped: torch.Tensor,
    final_rank_idx: torch.Tensor,
    shaped_target: torch.Tensor,
    alpha_td: float = 0.5,
    td_mode: Literal["gae_mse", "mc_mse"] = "gae_mse",
) -> dict[str, torch.Tensor]:
    """Contract v2.4.1 Section 3.3 Critic loss: CE(P_phi, r_final) + alpha_td * MSE(V_shaped, stopgrad(target)).

    - Strictly detaches `shaped_target` (`stopgrad`) so gradients NEVER propagate
      through GAE target value predictions.
    - `td_mode == "gae_mse"`: `shaped_target` is rollout-frozen GAE shaped return `G_hat^{GAE,shaped}`.
    - `td_mode == "mc_mse"`:  `shaped_target` is Monte-Carlo shaped return `R_terminal - Phi_k`.
    """
    if td_mode not in ("gae_mse", "mc_mse"):
        raise ValueError(f"Invalid critic td_mode {td_mode!r}")
    ce_loss = F.cross_entropy(rank_logits.float(), final_rank_idx.long())
    detached_target = shaped_target.detach().float()
    mse_loss = F.mse_loss(v_shaped.float(), detached_target)
    total = ce_loss + alpha_td * mse_loss
    return {"loss": total, "ce_loss": ce_loss, "mse_loss": mse_loss}
