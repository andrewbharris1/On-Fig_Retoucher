"""
Geometry module for deterministic cropping and placement.

This module uses ONLY pixel math - no AI involved.
All calculations follow Canada Goose brand standards exactly.

Canvas: 9000px height
Top padding: 203px (head to top edge)
Bottom padding: 203px (feet to bottom edge)
Safe model height: 9000 - 203 - 203 = 8594px
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from .config import Config


class GeometryProcessor:
    """
    Handles all geometric transformations for the pipeline.

    Key operations:
    - Resize model to exact 8594px height (maintaining aspect ratio)
    - Center horizontally on canvas
    - Place at Y=203 (top padding)

    NO AI is used in this module - all operations are deterministic pixel math.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the geometry processor.

        Args:
            config: Pipeline configuration with canvas specifications.
        """
        self.config = config or Config()

    def calculate_resize_dimensions(
        self,
        original_width: int,
        original_height: int,
        target_height: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Calculate new dimensions maintaining aspect ratio.

        Args:
            original_width: Current width in pixels.
            original_height: Current height in pixels.
            target_height: Target height (defaults to safe_model_height).

        Returns:
            Tuple of (new_width, new_height).
        """
        target_height = target_height or self.config.safe_model_height

        # Calculate scale factor based on height
        scale = target_height / original_height

        # Apply scale to width (maintaining aspect ratio)
        new_width = int(original_width * scale)
        new_height = target_height

        return new_width, new_height

    def resize_to_brand_standard(
        self,
        image: np.ndarray,
        interpolation: int = cv2.INTER_LANCZOS4
    ) -> np.ndarray:
        """
        Resize image to brand standard height (8594px).

        Uses Lanczos interpolation for highest quality downscaling/upscaling.

        Args:
            image: Input image (any channel count).
            interpolation: OpenCV interpolation method.

        Returns:
            Resized image with height exactly 8594px.
        """
        h, w = image.shape[:2]
        new_w, new_h = self.calculate_resize_dimensions(w, h)

        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        return resized

    def calculate_canvas_placement(
        self,
        model_width: int,
        model_height: int,
        canvas_width: Optional[int] = None,
        canvas_height: Optional[int] = None
    ) -> Tuple[int, int]:
        """
        Calculate the (x, y) position to place the model on the canvas.

        Model is centered horizontally and placed at top_padding vertically.

        Args:
            model_width: Width of the resized model.
            model_height: Height of the resized model.
            canvas_width: Canvas width (defaults to config).
            canvas_height: Canvas height (defaults to config).

        Returns:
            Tuple of (x, y) coordinates for top-left corner placement.
        """
        canvas_width = canvas_width or self.config.canvas_width
        canvas_height = canvas_height or self.config.canvas_height

        # Horizontal centering
        x = (canvas_width - model_width) // 2

        # Vertical placement at top padding
        y = self.config.top_padding

        return x, y

    def get_bounding_box_from_alpha(
        self,
        rgba_image: np.ndarray
    ) -> Tuple[int, int, int, int]:
        """
        Get the tight bounding box of non-transparent pixels.

        Args:
            rgba_image: RGBA image with alpha channel.

        Returns:
            Tuple of (x, y, width, height) of the bounding box.
        """
        if rgba_image.shape[2] != 4:
            raise ValueError("Image must have alpha channel (4 channels)")

        alpha = rgba_image[:, :, 3]

        # Find non-zero (non-transparent) pixels
        rows = np.any(alpha > 0, axis=1)
        cols = np.any(alpha > 0, axis=0)

        if not rows.any() or not cols.any():
            # No non-transparent pixels found
            return 0, 0, rgba_image.shape[1], rgba_image.shape[0]

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        width = x_max - x_min + 1
        height = y_max - y_min + 1

        return x_min, y_min, width, height

    def crop_to_content(self, rgba_image: np.ndarray) -> np.ndarray:
        """
        Crop RGBA image to tight bounding box around content.

        Args:
            rgba_image: RGBA image with transparent areas.

        Returns:
            Cropped image containing only the content.
        """
        x, y, w, h = self.get_bounding_box_from_alpha(rgba_image)
        return rgba_image[y:y+h, x:x+w]

    def place_on_canvas(
        self,
        model_rgba: np.ndarray,
        background: np.ndarray,
        auto_resize: bool = True
    ) -> Tuple[np.ndarray, dict]:
        """
        Place the model on the canvas following brand standards.

        This is the main function that implements the 203px sizing logic:
        1. Crop model to content
        2. Resize to 8594px height
        3. Center horizontally
        4. Place at Y=203

        Args:
            model_rgba: RGBA model image.
            background: BGR background canvas.
            auto_resize: If True, automatically resize to brand standard.

        Returns:
            Tuple of (composited_image, placement_metadata).
        """
        canvas_h, canvas_w = background.shape[:2]

        # Step 1: Crop to content (remove empty transparent areas)
        cropped = self.crop_to_content(model_rgba)
        crop_h, crop_w = cropped.shape[:2]

        # Step 2: Resize to brand standard height
        if auto_resize:
            new_w, new_h = self.calculate_resize_dimensions(crop_w, crop_h)
            resized = cv2.resize(
                cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
            )
        else:
            resized = cropped
            new_w, new_h = crop_w, crop_h

        # Step 3 & 4: Calculate centered position at Y=203
        x, y = self.calculate_canvas_placement(new_w, new_h, canvas_w, canvas_h)

        # Ensure model fits on canvas
        if x < 0:
            # Model is wider than canvas - need to resize down
            scale = canvas_w / new_w
            new_w = canvas_w
            new_h = int(new_h * scale)
            resized = cv2.resize(
                cropped, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
            )
            x = 0
            y = self.config.top_padding

        # Composite the model onto the background
        result = self._alpha_composite(resized, background, x, y)

        metadata = {
            "original_size": {"width": model_rgba.shape[1], "height": model_rgba.shape[0]},
            "cropped_size": {"width": crop_w, "height": crop_h},
            "final_size": {"width": new_w, "height": new_h},
            "placement": {"x": x, "y": y},
            "canvas_size": {"width": canvas_w, "height": canvas_h},
            "top_padding": self.config.top_padding,
            "bottom_padding": self.config.bottom_padding,
            "safe_height": self.config.safe_model_height
        }

        return result, metadata

    def _alpha_composite(
        self,
        foreground_rgba: np.ndarray,
        background_bgr: np.ndarray,
        x: int,
        y: int
    ) -> np.ndarray:
        """
        Composite RGBA foreground onto BGR background at position (x, y).

        Args:
            foreground_rgba: RGBA image to composite.
            background_bgr: BGR background.
            x: X coordinate for placement.
            y: Y coordinate for placement.

        Returns:
            Composited BGR image.
        """
        fg_h, fg_w = foreground_rgba.shape[:2]
        bg_h, bg_w = background_bgr.shape[:2]

        # Calculate actual region to composite
        x_end = min(x + fg_w, bg_w)
        y_end = min(y + fg_h, bg_h)
        fg_w_actual = x_end - max(x, 0)
        fg_h_actual = y_end - max(y, 0)

        # Handle negative x or y
        fg_x_start = max(0, -x)
        fg_y_start = max(0, -y)
        x = max(x, 0)
        y = max(y, 0)

        # Extract region
        fg_region = foreground_rgba[fg_y_start:fg_y_start+fg_h_actual,
                                    fg_x_start:fg_x_start+fg_w_actual]

        # Get alpha channel
        alpha = fg_region[:, :, 3].astype(float) / 255.0
        alpha = alpha[:, :, np.newaxis]

        # Convert RGBA to BGR
        fg_bgr = cv2.cvtColor(fg_region[:, :, :3], cv2.COLOR_RGB2BGR)

        # Get background region
        bg_region = background_bgr[y:y_end, x:x_end].astype(float)

        # Alpha composite
        composite = alpha * fg_bgr.astype(float) + (1 - alpha) * bg_region

        # Create result
        result = background_bgr.copy()
        result[y:y_end, x:x_end] = composite.astype(np.uint8)

        return result

    def create_canvas(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
        color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Create a blank canvas with the specified or default dimensions.

        Args:
            width: Canvas width (defaults to config).
            height: Canvas height (defaults to config).
            color: BGR color tuple (defaults to config background).

        Returns:
            BGR numpy array of the canvas.
        """
        width = width or self.config.canvas_width
        height = height or self.config.canvas_height
        color = color or self.config.background_color_bgr

        return np.full((height, width, 3), color, dtype=np.uint8)

    def resize_mask_to_match(
        self,
        mask: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Resize a mask to match a target shape.

        Args:
            mask: Binary mask to resize.
            target_shape: (height, width) target dimensions.

        Returns:
            Resized mask.
        """
        return cv2.resize(
            mask,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    def transform_mask_to_canvas(
        self,
        mask: np.ndarray,
        original_size: Tuple[int, int],
        placement_metadata: dict
    ) -> np.ndarray:
        """
        Transform a mask from original image space to canvas space.

        This accounts for cropping, resizing, and placement.

        Args:
            mask: Original mask from the R1 file.
            original_size: (width, height) of original image.
            placement_metadata: Metadata from place_on_canvas().

        Returns:
            Transformed mask on canvas coordinates.
        """
        canvas_w = placement_metadata["canvas_size"]["width"]
        canvas_h = placement_metadata["canvas_size"]["height"]
        final_w = placement_metadata["final_size"]["width"]
        final_h = placement_metadata["final_size"]["height"]
        place_x = placement_metadata["placement"]["x"]
        place_y = placement_metadata["placement"]["y"]

        # Create empty canvas-sized mask
        canvas_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        # Resize mask to match final model size
        # This uses the same scale as the model resize
        orig_h, orig_w = mask.shape[:2]
        scale_x = final_w / original_size[0]
        scale_y = final_h / original_size[1]

        # Use average scale (should be same if aspect ratio maintained)
        scale = min(scale_x, scale_y)

        new_mask_w = int(orig_w * scale)
        new_mask_h = int(orig_h * scale)

        resized_mask = cv2.resize(
            mask, (new_mask_w, new_mask_h), interpolation=cv2.INTER_NEAREST
        )

        # Place resized mask on canvas
        end_x = min(place_x + new_mask_w, canvas_w)
        end_y = min(place_y + new_mask_h, canvas_h)
        actual_w = end_x - place_x
        actual_h = end_y - place_y

        canvas_mask[place_y:end_y, place_x:end_x] = resized_mask[:actual_h, :actual_w]

        return canvas_mask
