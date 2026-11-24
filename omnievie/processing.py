"""Batch construction: text, document images, image queries and video clips."""

from __future__ import annotations

from typing import Sequence

import torch
from PIL import Image
from transformers.models.qwen3_vl import Qwen3VLProcessor

from . import prompts

PATCH_TILE = 32  # one visual token covers 32 x 32 pixels (16-px patch x 2 x 2 merge)


def _rgb(images: Sequence[Image.Image]) -> list[Image.Image]:
    return [image.convert("RGB") for image in images]


def _pad_pixels(batch):
    """Split the flattened patch rows per image and pad them into a [N, max_rows, D] tensor."""
    grid = batch["image_grid_thw"]
    counts = (grid[:, 1] * grid[:, 2]).tolist()
    pieces = list(torch.split(batch["pixel_values"], counts))
    batch["pixel_values"] = torch.nn.utils.rnn.pad_sequence(pieces, batch_first=True)
    return batch


class OmniEVIEProcessor:
    def __init__(self, weights_dir: str, max_image_tokens: int = 4096):
        processor = Qwen3VLProcessor.from_pretrained(weights_dir)
        image_processor = processor.image_processor
        image_processor.max_pixels = max_image_tokens * PATCH_TILE * PATCH_TILE
        image_processor.size["longest_edge"] = image_processor.max_pixels
        processor.tokenizer.padding_side = "left"
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.image_token_id = int(processor.image_token_id)
        self.video_token_id = int(processor.video_token_id)
        self.aug_token_id = int(self.tokenizer.convert_tokens_to_ids(prompts.AUG_TOKEN))

    def texts(self, texts: Sequence[str]):
        """Text documents and text queries; the augmentation suffix is appended to each."""
        suffix = prompts.AUG_TOKEN * prompts.AUG_COUNT
        return self.processor(text=[t + suffix for t in texts], return_tensors="pt", padding="longest")

    def document_images(self, images: Sequence[Image.Image]):
        batch = self.processor(
            text=[prompts.DOC_IMAGE_PROMPT] * len(images),
            images=_rgb(images),
            padding="longest",
            return_tensors="pt",
        )
        return _pad_pixels(batch)

    def image_queries(self, images: Sequence[Image.Image], texts: Sequence[str], instructions: Sequence[str]):
        """Each row is instruction + vision block + question + augmentation tokens."""
        suffix = prompts.AUG_TOKEN * prompts.AUG_COUNT
        prompts_full = [
            ins + prompts.VISION_BLOCK + text + suffix for ins, text in zip(instructions, texts)
        ]
        batch = self.processor(
            text=prompts_full,
            images=_rgb(images),
            padding="longest",
            return_tensors="pt",
        )
        return _pad_pixels(batch)

    def video(self, clips: Sequence[Sequence[Image.Image]]):
        """One video per clip; frames are passed through untouched (no extra sampling)."""
        return self._video_batch([prompts.DOC_VIDEO_PROMPT] * len(clips), clips)

    def video_queries(self, clips: Sequence[Sequence[Image.Image]], instructions: Sequence[str],
                      texts: Sequence[str]):
        """Video queries: instruction, video block and optional question per clip."""
        return self._video_batch(
            [prompts.video_query_prompt(ins, text) for ins, text in zip(instructions, texts)], clips)

    def _video_batch(self, prompt_list: Sequence[str], clips: Sequence[Sequence[Image.Image]]):
        metas = [
            {
                "fps": 1.0,
                "total_num_frames": len(frames),
                "frames_indices": list(range(len(frames))),
                "duration": float(len(frames)),
            }
            for frames in clips
        ]
        return self.processor(
            text=list(prompt_list),
            videos=[list(frames) for frames in clips],
            video_metadata=metas,
            do_sample_frames=False,
            padding="longest",
            return_tensors="pt",
        )
