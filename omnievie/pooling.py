"""Ward clustering into at most 128 vectors. Position codes guide the clustering only; each
cluster stores the mean of its original token vectors."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

BUDGET = 128
MERGE_SIZE = 2
IMAGE_POSITION_WEIGHT = 0.10
TEXT_POSITION_WEIGHT = 0.10
VIDEO_SPACE_WEIGHT = 0.07
VIDEO_TIME_WEIGHT = 0.03


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def grid_positions(grid_thw, merge: int = MERGE_SIZE) -> torch.Tensor:
    """Normalised (x, y) positions of the merged tokens of one image grid, in raster order."""
    _, height, width = (int(v) for v in grid_thw)
    rows, cols = height // merge, width // merge
    yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, rows), torch.linspace(0.0, 1.0, cols), indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), dim=-1)


def sinusoidal_2d(positions: torch.Tensor, dim: int) -> torch.Tensor:
    quarter = max(1, dim // 4)
    freqs = torch.exp(torch.arange(quarter, dtype=torch.float32) * (-math.log(10000.0) / max(1, quarter - 1)))
    x = positions[:, 0:1].float() / freqs.view(1, -1)
    y = positions[:, 1:2].float() / freqs.view(1, -1)
    enc = torch.cat((torch.sin(x), torch.cos(x), torch.sin(y), torch.cos(y)), dim=1)
    if enc.shape[1] < dim:
        enc = F.pad(enc, (0, dim - enc.shape[1]))
    return _unit(enc[:, :dim])


def sinusoidal_1d_monotonic(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Time code whose inner product decreases monotonically with the gap between two times."""
    half = max(1, dim // 2)
    freqs = torch.linspace(0.25 * math.pi, math.pi, half, dtype=torch.float32)
    angles = t[:, None].float() * freqs[None, :]
    enc = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
    if enc.shape[1] < dim:
        enc = F.pad(enc, (0, dim - enc.shape[1]))
    return _unit(enc[:, :dim])


def _ward_labels(guide: torch.Tensor, budget: int) -> torch.Tensor:
    from sklearn.cluster import AgglomerativeClustering

    labels = AgglomerativeClustering(n_clusters=budget, linkage="ward").fit_predict(guide.numpy())
    return torch.from_numpy(labels).long()


def _mean_pool(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    budget = int(labels.max()) + 1
    sums = torch.zeros(budget, values.shape[1]).index_add_(0, labels, values)
    counts = torch.bincount(labels, minlength=budget).clamp_min(1).unsqueeze(1).float()
    return _unit(sums / counts)


def image(tokens: torch.Tensor, grid_thw, budget: int = BUDGET,
          position_weight: float = IMAGE_POSITION_WEIGHT, *,
          return_labels: bool = False):
    """Visual tokens [T, D] of one image -> [min(T, budget), D] unit vectors."""
    values = tokens.float()
    if values.shape[0] <= budget:
        out = _unit(values)
        return (out, torch.arange(values.shape[0], dtype=torch.long)) if return_labels else out
    positions = grid_positions(grid_thw)
    guide = _unit((1.0 - position_weight) * _unit(values) + position_weight * sinusoidal_2d(positions, values.shape[1]))
    labels = _ward_labels(guide, budget)
    out = _mean_pool(values, labels)
    return (out, labels) if return_labels else out


def text_window(tokens: torch.Tensor, budget: int = BUDGET,
                position_weight: float = TEXT_POSITION_WEIGHT) -> torch.Tensor:
    values = tokens.float()
    if values.shape[0] <= budget:
        return _unit(values)
    n = values.shape[0]
    t = torch.arange(n, dtype=torch.float32) / max(1, n - 1)
    guide = _unit((1.0 - position_weight) * _unit(values) + position_weight * sinusoidal_1d_monotonic(t, values.shape[1]))
    return _mean_pool(values, _ward_labels(guide, budget))


def video_segment(tokens: torch.Tensor, grids, budget: int = BUDGET,
                  space_weight: float = VIDEO_SPACE_WEIGHT,
                  time_weight: float = VIDEO_TIME_WEIGHT, *,
                  return_labels: bool = False):
    """Video tokens of one segment, with one [1, h, w] grid per temporal group."""
    values = tokens.float()
    if values.shape[0] <= budget:
        out = _unit(values)
        return (out, torch.arange(values.shape[0], dtype=torch.long)) if return_labels else out
    frames = len(grids)
    positions = torch.cat([grid_positions(g) for g in grids], dim=0)
    dim = values.shape[1]
    normed = _unit(values)
    if frames <= 1:
        w = space_weight + time_weight
        guide = _unit((1.0 - w) * normed + w * sinusoidal_2d(positions, dim))
    else:
        times = torch.cat([
            torch.full((int(g[1]) // MERGE_SIZE * (int(g[2]) // MERGE_SIZE),), i / (frames - 1))
            for i, g in enumerate(grids)
        ])
        guide = _unit(
            (1.0 - space_weight - time_weight) * normed
            + space_weight * sinusoidal_2d(positions, dim)
            + time_weight * sinusoidal_1d_monotonic(times, dim)
        )
    labels = _ward_labels(guide, budget)
    out = _mean_pool(values, labels)
    return (out, labels) if return_labels else out


def merge(parts: list[torch.Tensor], budget: int = BUDGET, time_weight: float = VIDEO_TIME_WEIGHT) -> torch.Tensor:
    """Combine per-window or per-segment vectors, ordered by their position in the document."""
    parts = [p.float() for p in parts if p is not None and p.shape[0] > 0]
    if not parts:
        raise ValueError("no vectors to merge")
    if len(parts) == 1:
        return _unit(parts[0])
    values = torch.cat(parts, dim=0)
    if values.shape[0] <= budget:
        return _unit(values)
    count = len(parts)
    times = torch.cat([torch.full((p.shape[0],), i / (count - 1)) for i, p in enumerate(parts)])
    guide = _unit((1.0 - time_weight) * _unit(values) + time_weight * sinusoidal_1d_monotonic(times, values.shape[1]))
    return _mean_pool(values, _ward_labels(guide, budget))
