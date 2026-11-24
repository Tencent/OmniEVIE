"""OmniEVIE encoder: documents and queries in text, image and video -> multi-vector embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from . import localize, media, pooling, prompts
from .backbone import OmniEVIEBackbone
from .processing import OmniEVIEProcessor

DEFAULT_MODEL_ID = "tencent/OmniEVIE-4B"
TEXT_WINDOW_CHUNKS = 8   # document text is encoded in windows of 8 chunks (about 1K tokens)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
TEXT_EXTENSIONS = {".txt", ".md"}


@dataclass
class EncodedQuery:
    """Per-token query output plus the masks used by the query gate."""

    vectors: torch.Tensor        # [L, head_dim] float32, model output (unit length per token)
    vision_mask: torch.Tensor    # [L] bool, visual tokens of the query image
    aug_mask: torch.Tensor       # [L] bool, trailing augmentation tokens
    has_text: bool               # whether the query carries question text


def _as_rgb(image) -> Image.Image:
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    return image.convert("RGB")


def _text_or_blank(text: str) -> str:
    return text if text and text.strip() else " "


class OmniEVIE:
    """Multi-vector encoder with a 64-dimensional head; BF16 inference."""

    def __init__(self, backbone: OmniEVIEBackbone, processor: OmniEVIEProcessor, device: str):
        self.backbone = backbone
        self.processor = processor
        self.device = device
        self.tokenizer = media.TokenizerAdapter(processor.tokenizer)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str | Path = DEFAULT_MODEL_ID, *,
                        head_dim: int = 64, max_image_tokens: int = 4096,
                        device: str = "cuda") -> "OmniEVIE":
        """Load OmniEVIE-4B from a Hugging Face repo ID (e.g. ``tencent/OmniEVIE-4B``) or local directory."""
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("flash-attn is required (attn_implementation=flash_attention_2)") from exc
        from transformers.models.qwen3_5 import Qwen3_5Config

        source = str(model_name_or_path)
        config = Qwen3_5Config.from_pretrained(source)
        if head_dim not in [int(d) for d in config.head_dims]:
            raise ValueError(f"head_dim {head_dim} is not in {list(config.head_dims)}")
        backbone = OmniEVIEBackbone.from_pretrained(
            source, config=config, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).eval()
        backbone.enable_bidirectional_attention()
        backbone.set_head_dim(head_dim)
        backbone.to(device)
        processor = OmniEVIEProcessor(source, max_image_tokens=max_image_tokens)
        return cls(backbone, processor, device)

    # ------------------------------------------------------------------ primitives

    @torch.inference_mode()
    def _run(self, batch) -> torch.Tensor:
        if hasattr(self.backbone, "rope_deltas"):
            self.backbone.rope_deltas = None  # rotary state must not leak between batches
        out = self.backbone(**{k: v.to(self.device) for k, v in batch.items()})
        return out.float().cpu()

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> list[torch.Tensor]:
        """One forward pass per text: padding changes the linear-attention outputs."""
        outputs: list[torch.Tensor] = []
        for text in texts:
            batch = self.processor.texts([_text_or_blank(text)])
            keep = batch["attention_mask"][0].bool()
            outputs.append(self._run(batch)[0][keep].contiguous())
        return outputs

    @torch.inference_mode()
    def image_tokens(self, images: Sequence[Image.Image]):
        """Raw visual tokens of each image as (tokens [T, D], grid_thw); one forward pass per image."""
        results = []
        for image in images:
            batch = self.processor.document_images([image])
            keep = batch["attention_mask"][0].bool()
            emb = self._run(batch)[0][keep]
            ids = batch["input_ids"][0][keep]
            positions = torch.nonzero(ids == self.processor.image_token_id).flatten()
            grid = [int(v) for v in batch["image_grid_thw"][0].tolist()]
            start_pos, end_pos = int(positions[0]), int(positions[-1]) + 1
            if end_pos - start_pos != (grid[1] // 2) * (grid[2] // 2):
                raise RuntimeError(f"visual token count does not match grid {grid}")
            results.append((emb[start_pos:end_pos].contiguous(), grid))
        return results

    @torch.inference_mode()
    def video_tokens(self, frames: Sequence[Image.Image], *, token_cap: int = media.VIDEO_TOKEN_CAP):
        """Native video encoding of one segment: (tokens [T, D], one [1, h, w] grid per temporal group)."""
        batch = self.processor.video([media.prepare_frames(frames, token_cap)])
        emb = self._run(batch)
        selected = batch["input_ids"][0] == self.processor.video_token_id
        tokens = emb[0][selected]
        t, h, w = (int(v) for v in batch["video_grid_thw"][0].tolist())
        if tokens.shape[0] != t * (h // 2) * (w // 2):
            raise RuntimeError(f"video token count does not match grid {[t, h, w]}")
        return tokens.contiguous(), [[1, h, w]] * t

    # ------------------------------------------------------------------ documents

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """Text document -> [<=128, D] unit vectors."""
        chunks = media.chunk_text(text, self.tokenizer)
        windows = [" ".join(chunks[i:i + TEXT_WINDOW_CHUNKS]) for i in range(0, len(chunks), TEXT_WINDOW_CHUNKS)]
        parts = [pooling.text_window(F.normalize(v, dim=-1)) for v in self.encode_texts(windows)]
        return pooling.merge(parts)

    @torch.no_grad()
    def encode_image(self, image) -> torch.Tensor:
        """Image document -> [<=128, D] unit vectors."""
        [(tokens, grid)] = self.image_tokens([_as_rgb(image)])
        return pooling.image(tokens, grid)

    @torch.no_grad()
    def encode_clip(self, frames: Sequence[Image.Image], *, protocol: str = "deploy") -> torch.Tensor:
        """One video clip (a list of frames) -> [<=128, D] unit vectors."""
        settings = media.protocol_settings(protocol)
        clip = media.uniform_frames(frames, settings["frames"])
        tokens, grids = self.video_tokens(clip, token_cap=settings["token_cap"])
        return pooling.video_segment(tokens, grids)

    @torch.no_grad()
    def encode_video(self, path: str | Path, *, protocol: str = "deploy") -> torch.Tensor:
        """Video document -> [<=128, D] unit vectors."""
        if protocol == "mmeb":
            return self.encode_clip(media.mmeb_frames(path), protocol="mmeb")
        media.protocol_settings(protocol)
        segments = []
        for frames in media.iter_segments(path):
            tokens, grids = self.video_tokens(frames)
            segments.append(pooling.video_segment(tokens, grids))
        if not segments:
            raise ValueError(f"no decodable frames in {path}")
        return pooling.merge(segments)

    def encode_document(self, path: str | Path, *, protocol: str = "deploy") -> tuple[str, torch.Tensor]:
        """Dispatch by file suffix; returns (modality, vectors)."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            return "text", self.encode_text(path.read_text(encoding="utf-8", errors="replace"))
        if suffix in IMAGE_EXTENSIONS:
            return "image", self.encode_image(path)
        if suffix in VIDEO_EXTENSIONS:
            return "video", self.encode_video(path, protocol=protocol)
        raise ValueError(f"unsupported document type: {path}")

    # ------------------------------------------------------------------ queries

    @torch.no_grad()
    def encode_query(self, text: str | None = None, image=None, target: str = "image", *,
                     video: Sequence[Image.Image] | None = None,
                     instruction: str | None = None,
                     max_chars: int | None = prompts.MAX_QUERY_CHARS,
                     protocol: str = "deploy") -> EncodedQuery:
        """Text, image (+ text), or video frames (+ text). ``instruction`` replaces the scene
        instruction; ``max_chars=None`` keeps the whole text."""
        raw = text or ""
        body = (raw if max_chars is None else raw[:max_chars]).strip()
        has_text = bool(body)
        if video is None and image is None and not has_text:
            raise ValueError("a query needs text, an image or a video")
        if video is not None:
            settings = media.protocol_settings(protocol)
            clip = media.prepare_frames(media.uniform_frames(video, settings["frames"]), settings["token_cap"])
            batch = self.processor.video_queries([clip], [instruction or ""], [body])
        elif image is None:
            if instruction is not None:
                body = f"{instruction} {body}".strip()
            elif target == "video":
                body = prompts.text_to_video_query(body)
            batch = self.processor.texts([_text_or_blank(body)])
        elif has_text:
            scene = prompts.INSTR_IMAGE_TEXT if instruction is None else instruction
            batch = self.processor.image_queries([_as_rgb(image)], [body], [scene])
        else:
            scene = prompts.INSTR_IMAGE_PAIR if instruction is None else instruction
            batch = self.processor.image_queries([_as_rgb(image)], [prompts.IMAGE_PAIR_TEXT], [scene])
        emb = self._run(batch)[0]
        keep = batch["attention_mask"][0].bool()
        ids = batch["input_ids"][0][keep]
        vision = (ids == self.processor.image_token_id) | (ids == self.processor.video_token_id)
        return EncodedQuery(
            vectors=emb[keep].contiguous(),
            vision_mask=vision,
            aug_mask=ids == self.processor.aug_token_id,
            has_text=has_text,
        )

    # ------------------------------------------------------------------ localization

    @torch.no_grad()
    def localize_image(self, query: EncodedQuery, image, *,
                       weights: torch.Tensor | None = None,
                       budget: int = pooling.BUDGET,
                       max_area_ratio: float | None = 0.05) -> localize.ImageLocalization:
        """Locate query-matched ST-HAC clusters on a document page or image."""
        [(tokens, grid)] = self.image_tokens([_as_rgb(image)])
        return localize.localize_image(
            query.vectors, tokens, grid, weights=weights, token_mask=~query.aug_mask,
            budget=budget, max_area_ratio=max_area_ratio,
        )

    @torch.no_grad()
    def localize_video(self, query: EncodedQuery, frames: Sequence[Image.Image], duration: float, *,
                       weights: torch.Tensor | None = None, num_windows: int = 4,
                       budget: int = pooling.BUDGET, span_ratio: float = 0.8) -> localize.VideoLocalization:
        """Locate the matched timestamp span [start_sec, end_sec] in a video frame sequence."""
        frames = list(frames)
        if not frames:
            raise ValueError("frames must not be empty")
        n_win = max(1, min(int(num_windows), len(frames)))
        step = max(1, len(frames) // n_win)
        windows = [frames[i * step:(len(frames) if i == n_win - 1 else (i + 1) * step)] for i in range(n_win)]
        segments = [self.video_tokens(w) for w in windows if w]
        return localize.localize_video(
            query.vectors, segments, duration,
            weights=weights, token_mask=~query.aug_mask, budget=budget, span_ratio=span_ratio,
        )
