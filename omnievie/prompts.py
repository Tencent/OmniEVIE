"""Prompt templates for documents and queries. They must match the training-time strings exactly."""

AUG_TOKEN = "<|endoftext|>"
AUG_COUNT = 10            # trailing augmentation tokens appended to every text sequence
MAX_QUERY_CHARS = 512

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
DOC_IMAGE_PROMPT = "<|im_start|>user\n" + VISION_BLOCK + "<|im_end|>" + AUG_TOKEN
DOC_VIDEO_PROMPT = "<|im_start|>user\n<|video_pad|><|im_end|>" + AUG_TOKEN

INSTR_IMAGE_PAIR = "Retrieve the image showing the same object as the given image."
INSTR_IMAGE_TEXT = "Retrieve the image that answers the question about the given image."
INSTR_TEXT_TO_VIDEO = "Retrieve the most relevant video for the user's request."
IMAGE_PAIR_TEXT = ""      # image-to-image queries carry no question text
VIDEO_VISION_BLOCK = "<|vision_start|><|video_pad|><|vision_end|>"


def video_query_prompt(instruction: str | None, text: str) -> str:
    """Instruction, video block, question, then a single augmentation token."""
    lead = f"{instruction} " if instruction else ""
    return f"<|im_start|>user\n{lead}{VIDEO_VISION_BLOCK}{text}<|im_end|>" + AUG_TOKEN


def text_to_video_query(text: str) -> str:
    return f"{INSTR_TEXT_TO_VIDEO} {text}"
