"""Query gate: per-token weights from ten features and a small MLP, normalised to sum to L."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

NUM_FEATURES = 10
_NEIGHBOUR_OFFSETS = (-8, -4, -2, 2, 4, 8)


def _zscore(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return torch.zeros_like(x)
    std = x.std(unbiased=False)
    if float(std) < 1e-6:
        return torch.zeros_like(x)
    return ((x - x.mean()) / (std + 1e-5)).clamp(-3.0, 3.0)


def query_features(raw: torch.Tensor, unit: torch.Tensor, vision_mask: torch.Tensor,
                   aug_mask: torch.Tensor, has_text: bool) -> torch.Tensor:
    """[L, 10] features; ``raw`` is the model output and ``unit`` its per-token normalisation."""
    length = unit.shape[0]
    device = unit.device
    z_norm = _zscore(raw.float().norm(dim=-1))

    mean_dir = F.normalize(unit.mean(dim=0, keepdim=True), dim=-1)
    cos_dev = (1.0 - (unit * mean_dir).sum(dim=-1)).clamp(0.0, 2.0)
    z_dev = _zscore(cos_dev)

    if length > 1:
        step = (1.0 - (unit[1:] * unit[:-1]).sum(dim=-1)).clamp(0.0, 2.0)
        jump = torch.zeros(length, device=device)
        jump[:-1] += step
        jump[1:] += step
        jump[1:-1] *= 0.5
        z_jump = _zscore(jump)
        index = torch.arange(length, device=device)
        similarity = torch.zeros(length, device=device)
        for offset in _NEIGHBOUR_OFFSETS:
            neighbour = (index + offset).clamp(0, length - 1)
            similarity += (unit * unit[neighbour]).sum(dim=-1)
        local_unique = (1.0 - similarity / len(_NEIGHBOUR_OFFSETS)).clamp(0.0, 2.0)
    else:
        z_jump = torch.zeros(length, device=device)
        local_unique = torch.zeros(length, device=device)

    vision = vision_mask.to(device).float()
    aug = aug_mask.to(device).float()
    f_vis_mixed = vision if has_text else torch.zeros_like(vision)
    f_vis_pure = torch.zeros_like(vision) if has_text else vision
    f_aug = aug * (1.0 - vision).clamp_min(0.0)
    rel_pos = torch.linspace(0.0, 1.0, length, device=device) if length > 1 else torch.zeros(length, device=device)
    log_len = torch.full((length,), max(-1.5, min(2.0, math.log(max(length, 1) / 32.0) / 3.0)), device=device)
    return torch.stack([z_norm, cos_dev, z_dev, z_jump, local_unique,
                        f_vis_mixed, f_vis_pure, f_aug, rel_pos, log_len], dim=-1)


DEFAULT_GATE_FILENAME = "query_gate.safetensors"


class QueryGate:
    def __init__(self, mlp_w1, mlp_b1, mlp_w2, mlp_b2, dir_w, tau: float):
        self.mlp_w1, self.mlp_b1 = mlp_w1.float(), mlp_b1.float()
        self.mlp_w2, self.mlp_b2 = mlp_w2.float(), mlp_b2.float()
        self.dir_w = dir_w.float()
        self.tau = float(tau)

    @classmethod
    def load(cls, path: str | Path) -> "QueryGate":
        with safe_open(str(path), framework="pt") as handle:
            meta = handle.metadata() or {}
        if meta.get("kind") != "salience_v1":
            raise ValueError(f"unsupported gate kind in {path}: {meta.get('kind')!r}")
        t = load_file(str(path))
        return cls(t["mlp_w1"], t["mlp_b1"], t["mlp_w2"], t["mlp_b2"], t["dir_w"], float(meta["tau"]))

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path | None = None,
        *,
        filename: str = DEFAULT_GATE_FILENAME,
    ) -> "QueryGate":
        """Load the query gate from a local file, local directory, or Hugging Face repo ID."""
        if model_name_or_path is None:
            from .encoder import DEFAULT_MODEL_ID
            model_name_or_path = DEFAULT_MODEL_ID
        candidate = Path(model_name_or_path)
        if candidate.is_file():
            return cls.load(candidate)
        if candidate.is_dir():
            return cls.load(candidate / filename)
        from huggingface_hub import hf_hub_download

        return cls.load(hf_hub_download(repo_id=str(model_name_or_path), filename=filename))

    @torch.no_grad()
    def weights(self, raw: torch.Tensor, vision_mask: torch.Tensor, aug_mask: torch.Tensor,
                has_text: bool) -> torch.Tensor:
        """Per-token weights [L] that sum to L."""
        raw = raw.float()
        unit = F.normalize(raw, dim=-1)
        feats = query_features(raw, unit, vision_mask, aug_mask, has_text)
        hidden = F.silu(F.linear(feats, self.mlp_w1, self.mlp_b1))
        logit = F.linear(hidden, self.mlp_w2, self.mlp_b2).squeeze(-1) + F.linear(unit, self.dir_w).squeeze(-1)
        mult = torch.exp(self.tau * torch.tanh(logit)).clamp_min(0.05)
        return mult * (unit.shape[0] / mult.sum().clamp_min(1e-6))
