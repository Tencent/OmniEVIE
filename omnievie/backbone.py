"""Qwen3.5 backbone with a prefix-MRL multi-vector head.

The model structure follows ColQwen3.5 from ColPali-engine (Apache-2.0).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3_5 import Qwen3_5Model

_DENSE_ATTENTION_CLASSES = ("Qwen3_5Attention", "Qwen3Attention")


class OmniEVIEBackbone(Qwen3_5Model):
    """Emits one unit vector per token; a head of width d uses the first d projection rows."""

    def __init__(self, config):
        super().__init__(config=config)
        hidden = getattr(config, "hidden_size", None)
        if hidden is None:
            hidden = config.text_config.hidden_size
        self.max_dim = int(max(config.head_dims))
        self.custom_text_proj = nn.Linear(hidden, self.max_dim)
        self.head_dim = self.max_dim
        self.post_init()

    def set_head_dim(self, dim: int) -> None:
        if int(dim) not in [int(d) for d in self.config.head_dims]:
            raise ValueError(f"head {dim} is not in {list(self.config.head_dims)}")
        self.head_dim = int(dim)

    def enable_bidirectional_attention(self) -> None:
        """Turn full-attention layers into bidirectional encoders; linear-attention layers are unchanged."""
        self.config.is_causal = False
        text_config = getattr(self.config, "text_config", None)
        if text_config is not None:
            text_config.is_causal = False
        for module in self.modules():
            if type(module).__name__ in _DENSE_ATTENTION_CLASSES and hasattr(module, "is_causal"):
                module.is_causal = False

    def forward(self, *args, **kwargs):
        if "pixel_values" in kwargs:
            # The processor pads per-image patch rows; keep only the real rows of each image.
            grid = kwargs["image_grid_thw"]
            counts = (grid[:, 1] * grid[:, 2]).tolist()
            kwargs["pixel_values"] = torch.cat(
                [pixels[:n] for pixels, n in zip(kwargs["pixel_values"], counts)], dim=0
            )
        attention_mask = kwargs["attention_mask"]
        for key in ("return_dict", "output_hidden_states", "use_cache"):
            kwargs.pop(key, None)
        hidden = super().forward(*args, **kwargs, use_cache=False, return_dict=True).last_hidden_state
        vectors = self.custom_text_proj(hidden)[..., : self.head_dim]
        vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return vectors * attention_mask.unsqueeze(-1)
