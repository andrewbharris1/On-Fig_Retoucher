"""
Mask extraction module for detecting red markers in retouch note images.

Uses OpenCV to detect bright red markers drawn by human editors,
converting them into binary masks for inpainting operations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from .config import Config


class MaskExtractor:
    """
    Extracts binary masks from retouch note images by detecting red markers.

    Red markers drawn by human editors indicate areas that need retouching.
    This class converts those visual markers into machine-readable masks.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the mask extractor.

        Args:
            config: Pipeline configuration with HSV ranges for red detection.
        """
        self.config = config or Config()

    def extract_mask(self, retouch_note_path: str | Path) -> np.ndarray:
        """
        Extract a binary mask from a retouch note image.

        Args:
            retouch_note_path: Path to the _R1 image with red markers.

        Returns:
            Binary mask (numpy array) where white=retouch area, black=protect.
        """
        retouch_note_path = Path(retouch_note_path)
        if not retouch_note_path.exists():
            raise FileNotFoundError(f"Retouch note not found: {retouch_note_path}")

        # Load the image
        image = cv2.imread(str(retouch_note_path))
        if image is None:
            raise ValueError(f"Failed to load image: {retouch_note_path}")

        # Detect red markers
        mask = self._detect_red_markers(image)

        # Dilate to cover marker edges
        mask = self._dilate_mask(mask)

        return mask

    def _detect_red_markers(self, image: np.ndarray) -> np.ndarray:
        """
        Detect red pixels in the image using HSV color space.

        Red wraps around the HSV hue spectrum, so we check two ranges:
        - Low red: H=0-10
        - High red: H=160-180

        Args:
            image: BGR image from OpenCV.

        Returns:
            Binary mask of detected red pixels.
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create masks for both red ranges
        lower1 = np.array(self.config.red_hsv_lower_1)
        upper1 = np.array(self.config.red_hsv_upper_1)
        mask1 = cv2.inRange(hsv, lower1, upper1)

        lower2 = np.array(self.config.red_hsv_lower_2)
        upper2 = np.array(self.config.red_hsv_upper_2)
        mask2 = cv2.inRange(hsv, lower2, upper2)

        # Combine both masks
        combined_mask = cv2.bitwise_or(mask1, mask2)

        return combined_mask

    def _dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Dilate the mask to ensure red marker edges are fully covered.

        This is crucial because the inpainting needs to cover the entire
        marked area plus a small buffer to avoid edge artifacts.

        Args:
            mask: Binary mask to dilate.

        Returns:
            Dilated binary mask.
        """
        kernel_size = self.config.mask_dilation_kernel_size
        iterations = self.config.mask_dilation_iterations

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )

        dilated = cv2.dilate(mask, kernel, iterations=iterations)

        return dilated

    def extract_mask_with_metadata(
        self,
        retouch_note_path: str | Path
    ) -> Tuple[np.ndarray, dict]:
        """
        Extract mask and return metadata about the detected regions.

        Args:
            retouch_note_path: Path to the _R1 image.

        Returns:
            Tuple of (mask, metadata_dict) with region information.
        """
        mask = self.extract_mask(retouch_note_path)

        # Find contours to analyze regions
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            center_x = x + w // 2
            center_y = y + h // 2

            regions.append({
                "id": i,
                "bounding_box": {"x": x, "y": y, "width": w, "height": h},
                "area_pixels": int(area),
                "center": {"x": center_x, "y": center_y}
            })

        metadata = {
            "total_regions": len(regions),
            "total_mask_pixels": int(np.sum(mask > 0)),
            "mask_coverage_percent": float(np.sum(mask > 0) / mask.size * 100),
            "regions": regions
        }

        return mask, metadata

    def get_region_type(
        self,
        mask: np.ndarray,
        segmentation_mask: Optional[np.ndarray] = None,
        image_height: Optional[int] = None
    ) -> str:
        """
        Determine the type of region (fabric/skin) based on mask location.

        This is a simplified heuristic. A more sophisticated approach would
        use body part segmentation.

        Args:
            mask: The retouch mask.
            segmentation_mask: Optional foreground/model segmentation mask.
            image_height: Height of the image for vertical position analysis.

        Returns:
            "skin" or "fabric" based on heuristics.
        """
        if image_height is None:
            image_height = mask.shape[0]

        # Find the centroid of the mask
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return "fabric"  # Default to fabric if no mask

        center_y = int(moments["m01"] / moments["m00"])
        relative_y = center_y / image_height

        # Simple heuristic: upper 20% likely face/skin, rest likely fabric
        # This can be refined with actual body segmentation
        if relative_y < 0.20:
            return "skin"

        return "fabric"

    def save_mask(
        self,
        mask: np.ndarray,
        output_path: str | Path,
        as_alpha: bool = False
    ) -> Path:
        """
        Save the mask to a file.

        Args:
            mask: Binary mask to save.
            output_path: Destination path.
            as_alpha: If True, save as RGBA with mask as alpha channel.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)

        if as_alpha:
            # Create RGBA image with white foreground and mask as alpha
            rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            rgba[:, :, :3] = 255  # White
            rgba[:, :, 3] = mask  # Alpha from mask
            cv2.imwrite(str(output_path), rgba)
        else:
            cv2.imwrite(str(output_path), mask)

        return output_path

    def visualize_mask_overlay(
        self,
        original_path: str | Path,
        mask: np.ndarray,
        output_path: Optional[str | Path] = None,
        color: Tuple[int, int, int] = (0, 255, 0),  # Green
        opacity: float = 0.5
    ) -> np.ndarray:
        """
        Create a visualization of the mask overlaid on the original image.

        Useful for verifying red circle detection before processing.

        Args:
            original_path: Path to the original image.
            mask: Binary mask to overlay.
            output_path: Optional path to save the visualization.
            color: BGR color for the mask overlay.
            opacity: Opacity of the overlay (0-1).

        Returns:
            The visualization image.
        """
        original = cv2.imread(str(original_path))
        if original is None:
            raise ValueError(f"Failed to load original: {original_path}")

        # Resize mask if needed
        if mask.shape[:2] != original.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]))

        # Create colored overlay
        overlay = original.copy()
        overlay[mask > 0] = color

        # Blend with original
        result = cv2.addWeighted(original, 1 - opacity, overlay, opacity, 0)

        if output_path:
            cv2.imwrite(str(output_path), result)

        return result
