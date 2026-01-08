"""
Segmentation module for separating model from background.

Uses rembg for foreground extraction and implements shadow preservation
using the multiply blend mode technique (standard Photoshop approach).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

from .config import Config


class ModelSegmenter:
    """
    Handles model segmentation and background replacement with shadow preservation.

    The shadow preservation uses a grayscale + multiply blend approach,
    which is the standard technique used in Photoshop for keeping shadows
    when replacing backgrounds.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the segmenter.

        Args:
            config: Pipeline configuration.
        """
        self.config = config or Config()
        self._session = None

        if not REMBG_AVAILABLE:
            raise ImportError(
                "rembg is required for segmentation. "
                "Install with: pip install rembg"
            )

    @property
    def session(self):
        """Lazy-load rembg session for better performance."""
        if self._session is None:
            # Use u2net model for best quality
            self._session = new_session("u2net")
        return self._session

    def segment_model(
        self,
        image_path: str | Path,
        return_mask: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Extract the model/foreground from the image.

        Args:
            image_path: Path to the original image.
            return_mask: If True, also return the segmentation mask.

        Returns:
            RGBA image with transparent background, or tuple of (image, mask).
        """
        image_path = Path(image_path)

        # Load with PIL for rembg compatibility
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Remove background
            result = remove(img, session=self.session, alpha_matting=True)

        # Convert to numpy array (RGBA)
        result_np = np.array(result)

        if return_mask:
            # Extract alpha channel as mask
            mask = result_np[:, :, 3]
            return result_np, mask

        return result_np

    def extract_shadow(
        self,
        original_image: np.ndarray,
        segmentation_mask: np.ndarray
    ) -> np.ndarray:
        """
        Extract shadow from the original image.

        Uses grayscale conversion and thresholding to isolate shadow regions
        below the model (floor shadows from 3/4 angle lighting).

        Args:
            original_image: Original BGR image.
            segmentation_mask: Binary mask of the model.

        Returns:
            Shadow mask (grayscale, 0-255).
        """
        # Convert to grayscale
        if len(original_image.shape) == 3:
            gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = original_image.copy()

        # Invert the grayscale (shadows become bright)
        inverted = 255 - gray

        # Create a floor region mask (bottom 30% of image, outside model)
        h, w = gray.shape[:2]
        floor_mask = np.zeros((h, w), dtype=np.uint8)
        floor_start = int(h * 0.7)
        floor_mask[floor_start:, :] = 255

        # Exclude the model from the floor region
        model_dilated = cv2.dilate(
            segmentation_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)),
            iterations=2
        )
        floor_mask = cv2.bitwise_and(floor_mask, cv2.bitwise_not(model_dilated))

        # Threshold to isolate darker areas (shadows)
        _, shadow_thresh = cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)

        # Combine with floor region
        shadow_mask = cv2.bitwise_and(shadow_thresh, floor_mask)

        # Smooth the shadow edges
        shadow_mask = cv2.GaussianBlur(
            shadow_mask,
            (self.config.shadow_blur_kernel * 2 + 1,
             self.config.shadow_blur_kernel * 2 + 1),
            0
        )

        return shadow_mask

    def apply_shadow_multiply(
        self,
        background: np.ndarray,
        shadow_layer: np.ndarray,
        opacity: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply shadow to background using multiply blend mode.

        This replicates the Photoshop technique of using a multiply layer
        to keep shadows on a new background.

        Args:
            background: Solid color background (BGR).
            shadow_layer: Grayscale shadow layer.
            opacity: Shadow opacity (0-1). Uses config default if not provided.

        Returns:
            Background with shadow applied.
        """
        opacity = opacity or self.config.shadow_opacity

        # Ensure shadow is 3 channel
        if len(shadow_layer.shape) == 2:
            shadow_rgb = cv2.cvtColor(shadow_layer, cv2.COLOR_GRAY2BGR)
        else:
            shadow_rgb = shadow_layer

        # Normalize shadow to 0-1 range for multiply
        shadow_normalized = shadow_rgb.astype(float) / 255.0

        # Invert shadow (we want dark areas to darken the background)
        shadow_normalized = 1.0 - shadow_normalized

        # Multiply blend: result = background * shadow
        background_float = background.astype(float) / 255.0
        multiplied = background_float * shadow_normalized

        # Blend with original background based on opacity
        result = (1 - opacity) * background_float + opacity * multiplied

        # Convert back to uint8
        result = (result * 255).clip(0, 255).astype(np.uint8)

        return result

    def create_background_with_shadow(
        self,
        original_image: np.ndarray,
        segmentation_mask: np.ndarray,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Create a new background canvas with the original shadow preserved.

        Args:
            original_image: Original BGR image.
            segmentation_mask: Binary mask of the model.
            width: Canvas width.
            height: Canvas height.

        Returns:
            New background canvas with shadow.
        """
        # Create solid background
        bg_color = self.config.background_color_bgr
        background = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Extract shadow from original
        shadow = self.extract_shadow(original_image, segmentation_mask)

        # Resize shadow to match canvas if needed
        if shadow.shape[:2] != (height, width):
            shadow = cv2.resize(shadow, (width, height))

        # Apply shadow to background
        background_with_shadow = self.apply_shadow_multiply(background, shadow)

        return background_with_shadow

    def composite_model_on_background(
        self,
        model_rgba: np.ndarray,
        background: np.ndarray,
        position: Tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        """
        Composite the segmented model onto a background.

        Args:
            model_rgba: RGBA model image with transparent background.
            background: BGR background image.
            position: (x, y) position to place the model.

        Returns:
            Composited BGR image.
        """
        x, y = position
        model_h, model_w = model_rgba.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # Ensure we don't go out of bounds
        x_end = min(x + model_w, bg_w)
        y_end = min(y + model_h, bg_h)
        model_w_actual = x_end - x
        model_h_actual = y_end - y

        # Extract alpha channel
        alpha = model_rgba[:model_h_actual, :model_w_actual, 3].astype(float) / 255.0
        alpha = alpha[:, :, np.newaxis]

        # Extract RGB from model
        model_rgb = model_rgba[:model_h_actual, :model_w_actual, :3]

        # Convert model from RGBA (RGB order) to BGR for OpenCV
        model_bgr = cv2.cvtColor(model_rgb, cv2.COLOR_RGB2BGR)

        # Get the background region
        bg_region = background[y:y_end, x:x_end].astype(float)

        # Alpha composite
        composite = alpha * model_bgr.astype(float) + (1 - alpha) * bg_region

        # Place back on background
        result = background.copy()
        result[y:y_end, x:x_end] = composite.astype(np.uint8)

        return result

    def process_with_shadow_preservation(
        self,
        image_path: str | Path,
        canvas_width: int,
        canvas_height: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full segmentation pipeline with shadow preservation.

        Args:
            image_path: Path to the original image.
            canvas_width: Output canvas width.
            canvas_height: Output canvas height.

        Returns:
            Tuple of (model_rgba, segmentation_mask, background_with_shadow).
        """
        # Load original image
        original = cv2.imread(str(image_path))
        if original is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Segment the model
        model_rgba, seg_mask = self.segment_model(image_path, return_mask=True)

        # Create background with shadow
        background = self.create_background_with_shadow(
            original, seg_mask, canvas_width, canvas_height
        )

        return model_rgba, seg_mask, background
