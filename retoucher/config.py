"""
Configuration constants for the Canada Goose retouching pipeline.

All measurements are in pixels. Canvas specifications follow brand standards.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    """Pipeline configuration with Canada Goose brand standards."""

    # Canvas specifications
    canvas_height: int = 9000
    canvas_width: int = 6000  # Default width, can be adjusted

    # Padding (deterministic pixel math - no AI)
    top_padding: int = 203
    bottom_padding: int = 203

    # Calculated safe height for model placement
    # safe_height = canvas_height - (top_padding + bottom_padding) = 8594px
    @property
    def safe_model_height(self) -> int:
        """Maximum height for the model after cropping."""
        return self.canvas_height - (self.top_padding + self.bottom_padding)

    # Background color (Canada Goose standard)
    background_color_hex: str = "#edeef0"

    @property
    def background_color_rgb(self) -> Tuple[int, int, int]:
        """Convert hex to RGB tuple."""
        hex_color = self.background_color_hex.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @property
    def background_color_bgr(self) -> Tuple[int, int, int]:
        """Convert hex to BGR tuple for OpenCV."""
        rgb = self.background_color_rgb
        return (rgb[2], rgb[1], rgb[0])

    # Red marker detection HSV range
    # Bright red markers typically fall in these ranges
    red_hsv_lower_1: Tuple[int, int, int] = (0, 100, 100)
    red_hsv_upper_1: Tuple[int, int, int] = (10, 255, 255)
    red_hsv_lower_2: Tuple[int, int, int] = (160, 100, 100)
    red_hsv_upper_2: Tuple[int, int, int] = (180, 255, 255)

    # Mask dilation kernel size (to cover marker edges)
    mask_dilation_kernel_size: int = 15
    mask_dilation_iterations: int = 2

    # File naming patterns
    original_suffix: str = "_a.jpg"
    retouch_note_suffix: str = "_a_R1.jpg"

    # Inpainting prompts
    fabric_prompt: str = (
        "Remove wrinkles and hard crease in fabric, "
        "maintain the texture, pattern and colour of the fabric."
    )
    skin_prompt: str = (
        "Keep skin natural, use dodge and burn technique to even out skin. "
        "No frequency separation. Remove red zits, keep dark beauty marks or moles."
    )
    generic_prompt: str = (
        "Clean up and retouch the marked area naturally, "
        "maintaining consistency with surrounding pixels."
    )

    # Shadow preservation settings
    shadow_opacity: float = 0.6
    shadow_blur_kernel: int = 5

    # Output settings
    output_quality: int = 95  # JPEG quality
    output_suffix: str = "_retouched.jpg"
