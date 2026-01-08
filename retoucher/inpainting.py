"""
Inpainting module for generative AI retouching.

Provides a placeholder interface for various inpainting APIs:
- Stable Diffusion XL Inpainting
- OpenAI DALL-E 3 Editing
- Custom/local models

The module handles prompt selection based on mask location and
provides a unified interface for different backends.
"""

import base64
import io
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import Config

logger = logging.getLogger(__name__)


class InpaintingBackend(Enum):
    """Available inpainting backends."""
    PLACEHOLDER = "placeholder"
    STABLE_DIFFUSION = "stable_diffusion"
    OPENAI_DALLE = "openai_dalle"
    REPLICATE = "replicate"


class RetouchArea(Enum):
    """Types of areas that can be retouched."""
    FABRIC = "fabric"
    SKIN = "skin"
    GENERIC = "generic"


class InpaintingProvider(ABC):
    """Abstract base class for inpainting providers."""

    @abstractmethod
    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str
    ) -> np.ndarray:
        """
        Perform inpainting on the masked area.

        Args:
            image: BGR image to edit.
            mask: Binary mask (white = area to inpaint).
            prompt: Text prompt describing desired result.

        Returns:
            Inpainted BGR image.
        """
        pass


class PlaceholderProvider(InpaintingProvider):
    """
    Placeholder provider for testing pipeline without API calls.

    This provider simulates inpainting by applying a subtle blur
    to the masked region. Replace with actual API implementation.
    """

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str
    ) -> np.ndarray:
        """
        Placeholder inpainting using OpenCV inpaint.

        Uses Navier-Stokes based inpainting for basic fill.
        This is NOT generative AI - just a placeholder.
        """
        logger.info(f"Placeholder inpainting with prompt: {prompt[:50]}...")

        # Ensure mask is single channel
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Use OpenCV's built-in inpainting (not AI, but fills gaps)
        result = cv2.inpaint(
            image,
            mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA
        )

        return result


class StableDiffusionProvider(InpaintingProvider):
    """
    Stable Diffusion XL Inpainting provider.

    Requires a running SD API endpoint (e.g., Automatic1111 or ComfyUI).
    """

    def __init__(
        self,
        api_url: str = "http://localhost:7860",
        model_name: str = "sd_xl_inpainting_1.0"
    ):
        """
        Initialize the SD provider.

        Args:
            api_url: URL of the Stable Diffusion API.
            model_name: Name of the inpainting model to use.
        """
        self.api_url = api_url
        self.model_name = model_name

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str
    ) -> np.ndarray:
        """
        Perform inpainting using Stable Diffusion API.

        TODO: Implement actual API call.
        """
        try:
            import requests
        except ImportError:
            raise ImportError("requests is required for SD API calls")

        # Convert images to base64
        image_b64 = self._numpy_to_base64(image)
        mask_b64 = self._numpy_to_base64(mask)

        payload = {
            "init_images": [image_b64],
            "mask": mask_b64,
            "prompt": prompt,
            "negative_prompt": "blurry, distorted, unnatural",
            "steps": 30,
            "cfg_scale": 7.5,
            "denoising_strength": 0.75,
            "width": image.shape[1],
            "height": image.shape[0],
            "inpaint_full_res": True,
            "inpaint_full_res_padding": 32
        }

        response = requests.post(
            f"{self.api_url}/sdapi/v1/img2img",
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        result_data = response.json()
        result_b64 = result_data["images"][0]
        result_image = self._base64_to_numpy(result_b64)

        return result_image

    def _numpy_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array to base64 string."""
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def _base64_to_numpy(self, b64_string: str) -> np.ndarray:
        """Convert base64 string to numpy array."""
        image_data = base64.b64decode(b64_string)
        pil_image = Image.open(io.BytesIO(image_data))
        numpy_image = np.array(pil_image)

        if len(numpy_image.shape) == 3 and numpy_image.shape[2] >= 3:
            return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        return numpy_image


class OpenAIProvider(InpaintingProvider):
    """
    OpenAI DALL-E image editing provider.

    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: OpenAI API key. Uses env var if not provided.
        """
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var."
            )

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: str
    ) -> np.ndarray:
        """
        Perform inpainting using OpenAI's image editing API.

        TODO: Implement actual API call.
        """
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required for DALL-E editing")

        client = openai.OpenAI(api_key=self.api_key)

        # Convert to PIL and prepare for API
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)

        # API expects RGBA with transparent regions to edit
        image_rgba = image_pil.convert("RGBA")
        mask_array = np.array(mask_pil)

        # Set alpha to 0 where mask is white (area to edit)
        image_np = np.array(image_rgba)
        image_np[:, :, 3] = np.where(mask_array > 127, 0, 255)
        image_rgba = Image.fromarray(image_np)

        # Save to buffer
        image_buffer = io.BytesIO()
        image_rgba.save(image_buffer, format="PNG")
        image_buffer.seek(0)

        response = client.images.edit(
            model="dall-e-2",  # DALL-E 3 doesn't support editing yet
            image=image_buffer,
            prompt=prompt,
            n=1,
            size=f"{image.shape[1]}x{image.shape[0]}"
        )

        # Download result
        import requests
        result_url = response.data[0].url
        result_response = requests.get(result_url)
        result_pil = Image.open(io.BytesIO(result_response.content))
        result_np = np.array(result_pil)

        return cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)


class InpaintingEngine:
    """
    Main inpainting engine that orchestrates providers and prompt selection.

    This class:
    - Selects appropriate prompts based on mask location
    - Routes to the configured inpainting provider
    - Handles preprocessing and postprocessing
    """

    PROVIDERS = {
        InpaintingBackend.PLACEHOLDER: PlaceholderProvider,
        InpaintingBackend.STABLE_DIFFUSION: StableDiffusionProvider,
        InpaintingBackend.OPENAI_DALLE: OpenAIProvider,
    }

    def __init__(
        self,
        config: Optional[Config] = None,
        backend: InpaintingBackend = InpaintingBackend.PLACEHOLDER,
        **provider_kwargs
    ):
        """
        Initialize the inpainting engine.

        Args:
            config: Pipeline configuration.
            backend: Which inpainting backend to use.
            **provider_kwargs: Additional args for the provider.
        """
        self.config = config or Config()
        self.backend = backend

        provider_class = self.PROVIDERS.get(backend)
        if provider_class is None:
            raise ValueError(f"Unknown backend: {backend}")

        self.provider = provider_class(**provider_kwargs)

    def get_prompt_for_area(self, area_type: RetouchArea) -> str:
        """
        Get the appropriate prompt for a retouch area type.

        Args:
            area_type: Type of area (fabric, skin, generic).

        Returns:
            The prompt string.
        """
        if area_type == RetouchArea.FABRIC:
            return self.config.fabric_prompt
        elif area_type == RetouchArea.SKIN:
            return self.config.skin_prompt
        else:
            return self.config.generic_prompt

    def detect_area_type(
        self,
        mask: np.ndarray,
        image_height: int,
        segmentation_mask: Optional[np.ndarray] = None
    ) -> RetouchArea:
        """
        Detect the type of area based on mask position.

        Simple heuristic: upper 20% of image is likely skin (face),
        rest is likely fabric. Can be enhanced with body part detection.

        Args:
            mask: The retouch mask.
            image_height: Height of the image.
            segmentation_mask: Optional model segmentation mask.

        Returns:
            Detected area type.
        """
        # Find centroid of mask
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return RetouchArea.GENERIC

        center_y = int(moments["m01"] / moments["m00"])
        relative_y = center_y / image_height

        # Upper 20% = likely face/skin
        if relative_y < 0.20:
            return RetouchArea.SKIN

        return RetouchArea.FABRIC

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        area_type: Optional[RetouchArea] = None,
        custom_prompt: Optional[str] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Perform inpainting on the image.

        Args:
            image: BGR image to edit.
            mask: Binary mask (white = area to inpaint).
            area_type: Type of area. Auto-detected if not provided.
            custom_prompt: Override prompt. Uses area-based if not provided.

        Returns:
            Tuple of (inpainted_image, metadata).
        """
        # Auto-detect area type if not provided
        if area_type is None:
            area_type = self.detect_area_type(mask, image.shape[0])

        # Get or use custom prompt
        prompt = custom_prompt or self.get_prompt_for_area(area_type)

        logger.info(f"Inpainting area type: {area_type.value}")
        logger.info(f"Using prompt: {prompt}")

        # Perform inpainting
        result = self.provider.inpaint(image, mask, prompt)

        metadata = {
            "backend": self.backend.value,
            "area_type": area_type.value,
            "prompt": prompt,
            "mask_coverage_percent": float(np.sum(mask > 0) / mask.size * 100)
        }

        return result, metadata

    def inpaint_multiple_regions(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """
        Inpaint multiple regions with potentially different prompts.

        Splits the mask into connected components and processes each
        with the appropriate prompt.

        Args:
            image: BGR image to edit.
            mask: Binary mask with potentially multiple regions.

        Returns:
            Tuple of (inpainted_image, metadata).
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        result = image.copy()
        regions_processed = []

        # Skip label 0 (background)
        for i in range(1, num_labels):
            # Create mask for this region
            region_mask = (labels == i).astype(np.uint8) * 255

            # Detect area type for this specific region
            center_y = centroids[i][1]
            relative_y = center_y / image.shape[0]

            if relative_y < 0.20:
                area_type = RetouchArea.SKIN
            else:
                area_type = RetouchArea.FABRIC

            # Inpaint this region
            result, _ = self.inpaint(result, region_mask, area_type)

            regions_processed.append({
                "region_id": i,
                "area_type": area_type.value,
                "centroid": {"x": float(centroids[i][0]), "y": float(centroids[i][1])},
                "area_pixels": int(stats[i, cv2.CC_STAT_AREA])
            })

        metadata = {
            "backend": self.backend.value,
            "regions_processed": len(regions_processed),
            "regions": regions_processed
        }

        return result, metadata
