"""Video frame selection and decoding (ffmpeg), frame resizing, text chunking."""

from __future__ import annotations

import io
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

FRAMES_PER_MINUTE = 16
SEGMENT_SECONDS = 60.0
FRAME_MAX_PIXELS = 1024 * 1024
CHUNK_MAX_TOKENS = 128
DOC_MAX_TOKENS = 32768

FRAME_MIN_SIDE = 336          # short clips (<= 16 frames) are upscaled to this short side
SHORT_CLIP_MAX_FRAMES = 16
VIDEO_TOKEN_CAP = 3600        # per segment; larger segments are downscaled, never frame-dropped
MIN_SIDE_FLOOR = 224
PATCH = 32

MMEB_SAVED_FRAMES = 64
MMEB_FRAMES = 32
MMEB_TOKEN_CAP = 14400
PROTOCOLS = {
    "deploy": {"frames": None, "token_cap": VIDEO_TOKEN_CAP},
    "mmeb": {"frames": MMEB_FRAMES, "token_cap": MMEB_TOKEN_CAP},
}

_SENTENCE_BREAK = re.compile(r"(?<=[。！？!?；;…])\s*|\n+")


class TokenizerAdapter:
    """Minimal encode/decode view of a Hugging Face tokenizer, without special tokens."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: Sequence[int]) -> str:
        return self._tokenizer.decode(list(ids))


def _run(cmd: list[str]) -> bytes:
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"{cmd[0]} failed on {cmd[-1]}: {proc.stderr.decode('utf-8', 'ignore')[:200]}")
    return proc.stdout


def probe_duration(path: str | Path) -> float:
    out = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)])
    return float(json.loads(out)["format"]["duration"])


def keyframe_times(path: str | Path) -> list[float]:
    """Timestamps of all keyframes; non-key frames are skipped without decoding."""
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
                "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path)])
    times = []
    for line in out.decode().splitlines():
        line = line.strip().rstrip(",")
        if line:
            try:
                times.append(float(line))
            except ValueError:
                continue
    return sorted(times)


def select_frame_times(path: str | Path, duration: float) -> list[float]:
    budget = max(1, math.ceil(duration / 60.0)) * FRAMES_PER_MINUTE
    keys = keyframe_times(path)
    selected: list[float] = []
    if keys:
        take = min(budget, len(keys))
        idx = [round(i * (len(keys) - 1) / max(1, take - 1)) for i in range(take)] if take > 1 else [0]
        selected = [keys[i] for i in sorted(set(idx))]
    if len(selected) < budget:
        gap = max(0.05, duration / (budget * 2))
        for i in range(budget * 2):
            if len(selected) >= budget:
                break
            t = duration * (i + 0.5) / (budget * 2)
            if all(abs(t - s) > gap for s in selected):
                selected.append(t)
    return sorted(selected)


def extract_frame(path: str | Path, t: float) -> Image.Image:
    data = _run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1",
                 "-f", "image2pipe", "-vcodec", "mjpeg", "-"])
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def scale_frame(image: Image.Image, max_pixels: int = FRAME_MAX_PIXELS) -> Image.Image:
    """Keep the aspect ratio, cap the pixel count, and snap both sides to multiples of 32."""
    w, h = image.size
    if w * h <= max_pixels:
        nw, nh = max(PATCH, w // PATCH * PATCH), max(PATCH, h // PATCH * PATCH)
    else:
        s = math.sqrt(max_pixels / (w * h))
        nw, nh = max(PATCH, int(w * s) // PATCH * PATCH), max(PATCH, int(h * s) // PATCH * PATCH)
    return image if (nw, nh) == (w, h) else image.resize((nw, nh), Image.LANCZOS)


def iter_segments(path: str | Path, segment_seconds: float = SEGMENT_SECONDS):
    """Yield the frames of each 60-second block that contains sampled frames."""
    grouped: dict[int, list[float]] = {}
    for t in select_frame_times(path, probe_duration(path)):
        grouped.setdefault(int(t // segment_seconds), []).append(t)
    for index in sorted(grouped):
        yield [scale_frame(extract_frame(path, t)) for t in grouped[index]]


def prepare_frames(frames: Sequence[Image.Image], token_cap: int = VIDEO_TOKEN_CAP) -> list[Image.Image]:
    """Harmonise one segment's frames for the native video encoder (fixed size, even count, token cap)."""
    frames = [im.convert("RGB") for im in frames]
    if not frames:
        raise ValueError("video segment has no frames")
    size0 = frames[0].size
    frames = [im if im.size == size0 else im.resize(size0, Image.BICUBIC) for im in frames]
    if len(frames) <= SHORT_CLIP_MAX_FRAMES and min(size0) < FRAME_MIN_SIDE:
        factor = FRAME_MIN_SIDE / min(size0)
        frames = [_resize(im, factor) for im in frames]
    if len(frames) % 2:
        frames.append(frames[-1])  # temporal patches merge frame pairs
    w, h = frames[0].size
    tokens = (len(frames) / 2) * max(1, w // PATCH) * max(1, h // PATCH)
    if tokens > token_cap:
        factor = max((token_cap / tokens) ** 0.5, MIN_SIDE_FLOOR / min(w, h))
        if factor < 1.0:
            frames = [_resize(im, factor) for im in frames]
    return frames


def protocol_settings(protocol: str) -> dict:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown video protocol {protocol!r}; choose from {sorted(PROTOCOLS)}")
    return PROTOCOLS[protocol]


def uniform_frames(frames: Sequence, count: int | None) -> list:
    """``count`` evenly spaced items of ``frames``. The whole list is kept if ``count`` is None or larger."""
    frames = list(frames)
    if count is None or count > len(frames):
        return frames
    picks = np.linspace(0, len(frames) - 1, count, dtype=int)
    return [frames[i] for i in picks]


def saved_frame_indices(total: int, saved: int = MMEB_SAVED_FRAMES) -> list[int]:
    """Source frame numbers that the MMEB loader saves: all frames when there are few, else ``saved`` spread evenly."""
    if total <= saved:
        return list(range(total))
    step = total / saved
    return [math.floor(i * step) for i in range(saved)]


def count_frames(path: str | Path) -> int:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path)])
    return int(out.decode().strip())


def mmeb_frames(path: str | Path) -> list[Image.Image]:
    """The 32 frames the MMEB protocol reads from a video file (decoded straight from the source)."""
    chosen = uniform_frames(saved_frame_indices(count_frames(path)), MMEB_FRAMES)
    select = "+".join(f"eq(n\\,{i})" for i in chosen)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select='{select}'",
                        "-vsync", "vfr", os.path.join(tmp, "frame_%04d.jpg")], check=True)
        names = sorted(os.listdir(tmp))
        frames = []
        for name in names:
            with Image.open(os.path.join(tmp, name)) as image:
                frames.append(image.convert("RGB"))
    if len(frames) != len(chosen):
        raise RuntimeError(f"expected {len(chosen)} frames from {path}, decoded {len(frames)}")
    return frames


def _resize(image: Image.Image, factor: float) -> Image.Image:
    w, h = image.size
    return image.resize((max(PATCH, round(w * factor)), max(PATCH, round(h * factor))), Image.BICUBIC)


def chunk_text(text: str, tokenizer: TokenizerAdapter, max_tokens: int = CHUNK_MAX_TOKENS,
               doc_max_tokens: int = DOC_MAX_TOKENS) -> list[str]:
    """Split a document into sentence-packed chunks of at most ``max_tokens`` tokens.

    The document is truncated to ``doc_max_tokens`` first. A sentence longer than the
    limit is cut into fixed token windows.
    """
    ids = tokenizer.encode(text or " ")
    if len(ids) > doc_max_tokens:
        text = tokenizer.decode(ids[:doc_max_tokens])
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for sentence in _SENTENCE_BREAK.split(text):
        if not sentence or not sentence.strip():
            continue
        sentence_ids = tokenizer.encode(sentence)
        if len(sentence_ids) > max_tokens:
            flush()
            for start in range(0, len(sentence_ids), max_tokens):
                chunks.append(tokenizer.decode(sentence_ids[start:start + max_tokens]).strip())
            continue
        candidate = f"{current} {sentence}" if current else sentence
        if current and len(tokenizer.encode(candidate)) > max_tokens:
            flush()
            candidate = sentence
        current = candidate
    flush()
    return [c for c in chunks if c] or [" "]
