"""OmniEVIE: multi-vector retrieval and evidence localization over text, image, document and video."""

from .encoder import DEFAULT_MODEL_ID, EncodedQuery, OmniEVIE
from .gate import QueryGate
from .index import ThreeStageIndex
from .localize import ImageLocalization, VideoLocalization, localize_image, localize_video

__all__ = [
    "DEFAULT_MODEL_ID",
    "EncodedQuery",
    "ImageLocalization",
    "OmniEVIE",
    "QueryGate",
    "ThreeStageIndex",
    "VideoLocalization",
    "localize_image",
    "localize_video",
]
__version__ = "1.0.0"
