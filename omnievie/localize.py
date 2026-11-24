"""Direct local evidence localization over ST-HAC compressed clusters (page regions and video spans)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from . import pooling
from .index import NOISE_THETA, SCALE


@dataclass
class ImageLocalization:
    """Per-token cluster hits and aggregated spatial evidence heat map on an image or document page."""

    grid_hw: tuple[int, int]
    heat_map: torch.Tensor                          # [gh, gw] float32
    top_box: tuple[float, float, float, float] | None  # normalized (x0, y0, x1, y1) in [0, 1]
    token_hits: list[dict]


@dataclass
class VideoLocalization:
    """Temporal evidence heat curve, peak timestamp, and returned interval [start_sec, end_sec]."""

    duration: float
    peak_sec: float
    span_sec: tuple[float, float]
    heat: torch.Tensor                              # [num_groups] float32
    group_times: list[tuple[float, float]]


def _quantize_centroids(vectors: torch.Tensor) -> torch.Tensor:
    unit = F.normalize(vectors.float(), dim=-1)
    return torch.round(unit * SCALE).clamp(-127, 127) / SCALE


def _prepare_query(
    query_vectors: torch.Tensor,
    weights: torch.Tensor | None,
    token_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    u = F.normalize(query_vectors.float(), dim=-1)
    w = torch.ones(u.shape[0], dtype=torch.float32) if weights is None else weights.float().cpu()
    if token_mask is None:
        keep = list(range(u.shape[0]))
    else:
        keep = torch.nonzero(token_mask.bool(), as_tuple=False).flatten().tolist()
    if not keep:
        keep = list(range(u.shape[0]))
    return u[keep], w[keep], keep


def localize_image(
    query_vectors: torch.Tensor,
    image_tokens: torch.Tensor,
    grid_thw: Sequence[int],
    *,
    weights: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    budget: int = pooling.BUDGET,
    max_area_ratio: float | None = 0.05,
) -> ImageLocalization:
    """Map each query token to its matched ST-HAC cluster bounding box and accumulate a page heat map.

    Clusters whose bounding box covers more than ``max_area_ratio`` of the page are skipped to
    exclude full-page background clusters (Section 7.5 of the technical report).
    """
    centroids, labels = pooling.image(image_tokens, grid_thw, budget=budget, return_labels=True)
    mu = _quantize_centroids(centroids)
    u, w, keep = _prepare_query(query_vectors, weights, token_mask)

    sim, arg = (u @ mu.T).max(dim=-1)
    contrib = w * F.relu(sim - NOISE_THETA)

    gh = max(1, int(grid_thw[1]) // pooling.MERGE_SIZE)
    gw = max(1, int(grid_thw[2]) // pooling.MERGE_SIZE)
    heat = torch.zeros(gh, gw, dtype=torch.float32)

    hits: list[dict] = []
    best_box: tuple[float, float, float, float] | None = None
    best_contrib = -1.0

    for idx, orig_idx in enumerate(keep):
        k = int(arg[idx])
        c = float(contrib[idx])
        members = torch.nonzero(labels == k, as_tuple=False).flatten().tolist()
        rows = [p // gw for p in members]
        cols = [p % gw for p in members]
        box_grid = (min(cols), min(rows), max(cols) + 1, max(rows) + 1)
        box_norm = (box_grid[0] / gw, box_grid[1] / gh, box_grid[2] / gw, box_grid[3] / gh)
        area_ratio = (box_norm[2] - box_norm[0]) * (box_norm[3] - box_norm[1])
        if max_area_ratio is not None and area_ratio > max_area_ratio:
            continue
        if c > 0.0:
            for r, col in zip(rows, cols):
                heat[r, col] += c
        if c > best_contrib:
            best_contrib = c
            best_box = box_norm
        hits.append({
            "token_idx": orig_idx,
            "cluster": k,
            "score": round(float(sim[idx]), 4),
            "weight": round(float(w[idx]), 4),
            "contribution": round(c, 4),
            "box_grid": box_grid,
            "box_norm": tuple(round(v, 4) for v in box_norm),
        })

    return ImageLocalization(grid_hw=(gh, gw), heat_map=heat, top_box=best_box, token_hits=hits)


def localize_video(
    query_vectors: torch.Tensor,
    segments: Sequence[tuple[torch.Tensor, Sequence[Sequence[int]]]],
    duration: float,
    *,
    weights: torch.Tensor | None = None,
    token_mask: torch.Tensor | None = None,
    budget: int = pooling.BUDGET,
    span_ratio: float = 0.8,
) -> VideoLocalization:
    """Accumulate per-temporal-group evidence heat across video segments and return peak and span in seconds."""
    if not segments:
        raise ValueError("segments must not be empty")
    duration = float(duration)
    all_centroids: list[torch.Tensor] = []
    cluster_spans: list[tuple[int, int]] = []
    total_groups = 0

    for tokens, grids in segments:
        centroids, labels = pooling.video_segment(tokens, grids, budget=budget, return_labels=True)
        group_ids = torch.cat([
            torch.full(
                (int(g[1]) // pooling.MERGE_SIZE * (int(g[2]) // pooling.MERGE_SIZE),),
                total_groups + i,
                dtype=torch.long,
            )
            for i, g in enumerate(grids)
        ])
        for k in range(centroids.shape[0]):
            g = group_ids[labels == k]
            cluster_spans.append((int(g.min()), int(g.max())))
        all_centroids.append(centroids)
        total_groups += len(grids)

    mu = _quantize_centroids(torch.cat(all_centroids, dim=0))
    u, w, _ = _prepare_query(query_vectors, weights, token_mask)
    sim, arg = (u @ mu.T).max(dim=-1)
    contrib = w * F.relu(sim - NOISE_THETA)

    heat = torch.zeros(total_groups, dtype=torch.float32)
    for k, c in zip(arg.tolist(), contrib.tolist()):
        if c <= 0.0:
            continue
        g0, g1 = cluster_spans[k]
        heat[g0:g1 + 1] += c

    step = duration / max(total_groups, 1)
    group_times = [(i * step, (i + 1) * step) for i in range(total_groups)]
    peak_idx = int(torch.argmax(heat))
    peak_val = float(heat[peak_idx])
    peak_sec = (peak_idx + 0.5) * step

    if peak_val <= 0.0:
        span_sec = (group_times[peak_idx][0], group_times[peak_idx][1])
    else:
        thresh = peak_val * span_ratio
        lo = hi = peak_idx
        while lo > 0 and float(heat[lo - 1]) >= thresh:
            lo -= 1
        while hi + 1 < total_groups and float(heat[hi + 1]) >= thresh:
            hi += 1
        span_sec = (group_times[lo][0], group_times[hi][1])

    return VideoLocalization(
        duration=duration,
        peak_sec=round(peak_sec, 2),
        span_sec=(round(span_sec[0], 2), round(span_sec[1], 2)),
        heat=heat,
        group_times=group_times,
    )
