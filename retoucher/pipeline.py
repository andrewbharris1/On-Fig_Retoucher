"""
Main pipeline orchestration for the Canada Goose photo retouching workflow.

This module brings together all components:
- Ingestion: File scanning and pairing
- Mask Extraction: Red marker detection
- Segmentation: Model extraction with shadow preservation
- Geometry: Deterministic cropping and placement
- Inpainting: Generative AI retouching
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable

import cv2
import numpy as np

from .config import Config
from .ingestion import ImageIngestion, ImagePair, SKUGroup
from .mask_extraction import MaskExtractor
from .segmentation import ModelSegmenter
from .geometry import GeometryProcessor
from .inpainting import InpaintingEngine, InpaintingBackend

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single image pair."""

    pair: ImagePair
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"ProcessingResult({self.pair.base_name}, {status})"


@dataclass
class SKUProcessingResult:
    """Result of processing an entire SKU group."""

    sku: str
    results: List[ProcessingResult]

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def all_success(self) -> bool:
        return all(r.success for r in self.results)


class RetouchingPipeline:
    """
    Main orchestration class for the photo retouching pipeline.

    Workflow:
    1. Scan directory for image pairs
    2. For each pair with retouch notes:
       a. Extract red marker mask from R1 file
       b. Segment model from original
       c. Create background with shadow preservation
       d. Resize and place model (203px logic)
       e. Transform mask to canvas coordinates
       f. Apply generative inpainting
       g. Save result
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        inpainting_backend: InpaintingBackend = InpaintingBackend.PLACEHOLDER,
        **inpainting_kwargs
    ):
        """
        Initialize the retouching pipeline.

        Args:
            config: Pipeline configuration.
            inpainting_backend: Which inpainting API to use.
            **inpainting_kwargs: Additional args for inpainting provider.
        """
        self.config = config or Config()

        # Initialize all components
        self.ingestion = ImageIngestion(self.config)
        self.mask_extractor = MaskExtractor(self.config)
        self.segmenter = ModelSegmenter(self.config)
        self.geometry = GeometryProcessor(self.config)
        self.inpainting = InpaintingEngine(
            self.config,
            backend=inpainting_backend,
            **inpainting_kwargs
        )

        # Progress callback
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable[[str, int, int], None]):
        """
        Set a callback for progress updates.

        Args:
            callback: Function(message, current, total) called on progress.
        """
        self._progress_callback = callback

    def _report_progress(self, message: str, current: int = 0, total: int = 0):
        """Report progress if callback is set."""
        logger.info(message)
        if self._progress_callback:
            self._progress_callback(message, current, total)

    def scan_directory(self, input_dir: str | Path) -> Dict[str, SKUGroup]:
        """
        Scan a directory and return grouped images.

        Args:
            input_dir: Path to directory with images.

        Returns:
            Dictionary of SKU -> SKUGroup.
        """
        return self.ingestion.scan_directory(input_dir)

    def validate_directory(self, input_dir: str | Path) -> Dict:
        """
        Validate a directory and return statistics.

        Args:
            input_dir: Path to validate.

        Returns:
            Validation results dictionary.
        """
        return self.ingestion.validate_directory(input_dir)

    def process_single_image(
        self,
        pair: ImagePair,
        output_dir: Path,
        skip_inpainting: bool = False
    ) -> ProcessingResult:
        """
        Process a single image pair through the full pipeline.

        Args:
            pair: ImagePair to process.
            output_dir: Directory to save output.
            skip_inpainting: If True, skip the inpainting step.

        Returns:
            ProcessingResult with status and metadata.
        """
        try:
            self._report_progress(f"Processing {pair.base_name}...")

            # Step 1: Extract mask from R1 file
            if not pair.has_retouch_note:
                return ProcessingResult(
                    pair=pair,
                    success=False,
                    error="No retouch note file found"
                )

            self._report_progress(f"  Extracting mask from {pair.retouch_note_path.name}")
            mask, mask_metadata = self.mask_extractor.extract_mask_with_metadata(
                pair.retouch_note_path
            )

            # Step 2: Segment model from original
            self._report_progress(f"  Segmenting model from {pair.original_path.name}")
            model_rgba, seg_mask, _ = self.segmenter.process_with_shadow_preservation(
                pair.original_path,
                self.config.canvas_width,
                self.config.canvas_height
            )

            # Step 3: Create background canvas
            self._report_progress("  Creating background with shadow preservation")
            original = cv2.imread(str(pair.original_path))
            background = self.segmenter.create_background_with_shadow(
                original,
                seg_mask,
                self.config.canvas_width,
                self.config.canvas_height
            )

            # Step 4: Place model on canvas (203px sizing logic)
            self._report_progress("  Applying geometry transformations")
            composited, placement_metadata = self.geometry.place_on_canvas(
                model_rgba,
                background
            )

            # Step 5: Transform mask to canvas coordinates
            original_size = (original.shape[1], original.shape[0])
            canvas_mask = self.geometry.transform_mask_to_canvas(
                mask,
                original_size,
                placement_metadata
            )

            # Step 6: Apply inpainting
            if not skip_inpainting and np.any(canvas_mask > 0):
                self._report_progress("  Applying generative inpainting")
                inpainted, inpaint_metadata = self.inpainting.inpaint_multiple_regions(
                    composited,
                    canvas_mask
                )
            else:
                inpainted = composited
                inpaint_metadata = {"skipped": True}

            # Step 7: Save result
            output_filename = f"{pair.base_name}{self.config.output_suffix}"
            output_path = output_dir / output_filename

            cv2.imwrite(
                str(output_path),
                inpainted,
                [cv2.IMWRITE_JPEG_QUALITY, self.config.output_quality]
            )

            self._report_progress(f"  Saved to {output_path}")

            return ProcessingResult(
                pair=pair,
                success=True,
                output_path=output_path,
                metadata={
                    "mask": mask_metadata,
                    "placement": placement_metadata,
                    "inpainting": inpaint_metadata
                }
            )

        except Exception as e:
            logger.exception(f"Error processing {pair.base_name}")
            return ProcessingResult(
                pair=pair,
                success=False,
                error=str(e)
            )

    def process_sku(
        self,
        sku_group: SKUGroup,
        output_dir: str | Path,
        skip_inpainting: bool = False
    ) -> SKUProcessingResult:
        """
        Process all images in a SKU group.

        Args:
            sku_group: SKUGroup to process.
            output_dir: Directory to save outputs.
            skip_inpainting: If True, skip inpainting step.

        Returns:
            SKUProcessingResult with all individual results.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        pairs_to_process = sku_group.pairs_with_notes

        self._report_progress(
            f"Processing SKU {sku_group.sku}: {len(pairs_to_process)} images",
            0, len(pairs_to_process)
        )

        for i, pair in enumerate(pairs_to_process):
            result = self.process_single_image(pair, output_dir, skip_inpainting)
            results.append(result)
            self._report_progress(
                f"Completed {i + 1}/{len(pairs_to_process)}",
                i + 1, len(pairs_to_process)
            )

        return SKUProcessingResult(sku=sku_group.sku, results=results)

    def process_directory(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        skip_inpainting: bool = False,
        sku_filter: Optional[List[str]] = None
    ) -> List[SKUProcessingResult]:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory with input images.
            output_dir: Directory for output images.
            skip_inpainting: If True, skip inpainting step.
            sku_filter: Optional list of SKUs to process. Process all if None.

        Returns:
            List of SKUProcessingResult for each SKU processed.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Scan and group images
        sku_groups = self.scan_directory(input_dir)

        if sku_filter:
            sku_groups = {
                sku: group for sku, group in sku_groups.items()
                if sku in sku_filter
            }

        self._report_progress(f"Found {len(sku_groups)} SKUs to process")

        results = []
        for sku, group in sku_groups.items():
            sku_result = self.process_sku(group, output_dir, skip_inpainting)
            results.append(sku_result)

        # Summary
        total_processed = sum(r.success_count + r.failure_count for r in results)
        total_success = sum(r.success_count for r in results)
        total_failure = sum(r.failure_count for r in results)

        self._report_progress(
            f"\nProcessing complete: {total_success}/{total_processed} successful, "
            f"{total_failure} failures"
        )

        return results

    def verify_red_detection(
        self,
        retouch_note_path: str | Path,
        original_path: str | Path,
        output_path: Optional[str | Path] = None
    ) -> Path:
        """
        Create a visualization to verify red circle detection.

        This is useful for testing the red marker detection before
        running on thousands of images.

        Args:
            retouch_note_path: Path to the R1 file.
            original_path: Path to the original file.
            output_path: Where to save visualization. Auto-generates if None.

        Returns:
            Path to the saved visualization.
        """
        retouch_note_path = Path(retouch_note_path)

        if output_path is None:
            output_path = retouch_note_path.parent / f"{retouch_note_path.stem}_mask_preview.jpg"

        mask = self.mask_extractor.extract_mask(retouch_note_path)

        self.mask_extractor.visualize_mask_overlay(
            original_path,
            mask,
            output_path,
            color=(0, 255, 0),  # Green overlay
            opacity=0.5
        )

        return output_path

    def process_with_custom_prompt(
        self,
        pair: ImagePair,
        output_dir: Path,
        custom_prompt: str
    ) -> ProcessingResult:
        """
        Process an image with a custom inpainting prompt.

        Args:
            pair: ImagePair to process.
            output_dir: Directory to save output.
            custom_prompt: Custom prompt for inpainting.

        Returns:
            ProcessingResult.
        """
        # Store original prompts
        original_fabric = self.config.fabric_prompt
        original_skin = self.config.skin_prompt
        original_generic = self.config.generic_prompt

        # Override all prompts with custom
        self.config.fabric_prompt = custom_prompt
        self.config.skin_prompt = custom_prompt
        self.config.generic_prompt = custom_prompt

        try:
            result = self.process_single_image(pair, output_dir)
        finally:
            # Restore original prompts
            self.config.fabric_prompt = original_fabric
            self.config.skin_prompt = original_skin
            self.config.generic_prompt = original_generic

        return result


def create_pipeline(
    canvas_width: int = 6000,
    canvas_height: int = 9000,
    top_padding: int = 203,
    bottom_padding: int = 203,
    background_color: str = "#edeef0",
    inpainting_backend: str = "placeholder"
) -> RetouchingPipeline:
    """
    Factory function to create a configured pipeline.

    Args:
        canvas_width: Output canvas width.
        canvas_height: Output canvas height.
        top_padding: Pixels from head to top edge.
        bottom_padding: Pixels from feet to bottom edge.
        background_color: Background hex color.
        inpainting_backend: Which inpainting backend to use.

    Returns:
        Configured RetouchingPipeline instance.
    """
    config = Config(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        top_padding=top_padding,
        bottom_padding=bottom_padding,
        background_color_hex=background_color
    )

    backend = InpaintingBackend(inpainting_backend)

    return RetouchingPipeline(config=config, inpainting_backend=backend)
