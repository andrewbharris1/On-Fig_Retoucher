"""
Canada Goose Photo Retouching Pipeline

A modular pipeline for automated photo retouching:
- Ingestion & pairing of images with retouch notes
- Red marker mask extraction using OpenCV
- Background segmentation with shadow preservation
- Deterministic geometric cropping and placement
- Generative AI inpainting integration
"""

from .config import Config
from .pipeline import RetouchingPipeline

__version__ = "1.0.0"
__all__ = ["Config", "RetouchingPipeline"]
