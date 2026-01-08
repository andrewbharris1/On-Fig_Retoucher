"""
Tests for the geometry module.

Verifies the deterministic pixel math for:
- Canvas dimensions: 9000px height
- Top padding: 203px
- Bottom padding: 203px
- Safe model height: 8594px
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from retoucher.config import Config
from retoucher.geometry import GeometryProcessor


class TestConfig:
    """Test configuration values."""

    def test_safe_model_height(self):
        """Verify safe model height calculation: 9000 - 203 - 203 = 8594"""
        config = Config()
        assert config.canvas_height == 9000
        assert config.top_padding == 203
        assert config.bottom_padding == 203
        assert config.safe_model_height == 8594

    def test_background_color_conversion(self):
        """Test hex to RGB/BGR conversion."""
        config = Config(background_color_hex="#edeef0")
        assert config.background_color_rgb == (237, 238, 240)
        assert config.background_color_bgr == (240, 238, 237)


class TestGeometryProcessor:
    """Test geometry calculations."""

    @pytest.fixture
    def processor(self):
        return GeometryProcessor()

    def test_resize_dimensions_maintains_aspect_ratio(self, processor):
        """Verify aspect ratio is maintained during resize."""
        # Original 4500x6750 image (2:3 ratio)
        orig_w, orig_h = 4500, 6750

        new_w, new_h = processor.calculate_resize_dimensions(orig_w, orig_h)

        assert new_h == 8594  # Target height
        # Check aspect ratio is preserved
        original_ratio = orig_w / orig_h
        new_ratio = new_w / new_h
        assert abs(original_ratio - new_ratio) < 0.001

    def test_resize_dimensions_exact_height(self, processor):
        """Verify output height is exactly 8594px."""
        test_cases = [
            (4000, 6000),
            (5000, 7500),
            (6000, 9000),
            (3000, 4500),
        ]

        for orig_w, orig_h in test_cases:
            _, new_h = processor.calculate_resize_dimensions(orig_w, orig_h)
            assert new_h == 8594, f"Failed for input {orig_w}x{orig_h}"

    def test_canvas_placement_vertical(self, processor):
        """Verify model is placed at Y=203 (top padding)."""
        model_w, model_h = 5730, 8594

        x, y = processor.calculate_canvas_placement(model_w, model_h)

        assert y == 203  # Top padding

    def test_canvas_placement_horizontal_centering(self, processor):
        """Verify model is horizontally centered."""
        canvas_w = 6000
        model_w = 4000
        model_h = 8594

        x, y = processor.calculate_canvas_placement(
            model_w, model_h, canvas_width=canvas_w
        )

        # Should be centered: (6000 - 4000) // 2 = 1000
        assert x == 1000

    def test_create_canvas_dimensions(self, processor):
        """Verify canvas creation with correct dimensions."""
        canvas = processor.create_canvas()

        assert canvas.shape == (9000, 6000, 3)

    def test_create_canvas_color(self, processor):
        """Verify canvas has correct background color."""
        canvas = processor.create_canvas()

        # Check a pixel matches the expected BGR color
        expected_bgr = processor.config.background_color_bgr
        assert tuple(canvas[0, 0]) == expected_bgr

    def test_bounding_box_from_alpha(self, processor):
        """Test bounding box extraction from alpha channel."""
        # Create a 100x100 RGBA image with content in center
        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        rgba[20:80, 30:70, :3] = 255  # White content
        rgba[20:80, 30:70, 3] = 255   # Opaque alpha

        x, y, w, h = processor.get_bounding_box_from_alpha(rgba)

        assert x == 30
        assert y == 20
        assert w == 40
        assert h == 60


class TestBrandStandards:
    """Verify brand standard compliance."""

    def test_203px_math(self):
        """
        Verify the fundamental brand standard calculation:
        9000 - 203 - 203 = 8594
        """
        canvas_height = 9000
        top_padding = 203
        bottom_padding = 203
        safe_height = canvas_height - top_padding - bottom_padding

        assert safe_height == 8594

    def test_model_placement_fills_safe_zone(self):
        """
        Verify a model resized to 8594px exactly fills the safe zone.
        """
        config = Config()
        processor = GeometryProcessor(config)

        # Model exactly 8594px tall should fit perfectly
        model_h = 8594

        x, y = processor.calculate_canvas_placement(5000, model_h)

        # Should be placed at top padding
        assert y == config.top_padding

        # Bottom should align with canvas_height - bottom_padding
        model_bottom = y + model_h
        expected_bottom = config.canvas_height - config.bottom_padding
        assert model_bottom == expected_bottom


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
