#!/usr/bin/env python3
"""
Canada Goose Photo Retouching Pipeline - CLI Entry Point

Usage:
    python main.py process <input_dir> <output_dir> [options]
    python main.py validate <input_dir>
    python main.py verify-mask <r1_file> <original_file> [--output <path>]

Examples:
    # Process all images in a directory
    python main.py process ./input ./output

    # Process specific SKUs only
    python main.py process ./input ./output --sku 5520MB --sku 6612LB

    # Validate directory structure
    python main.py validate ./input

    # Test red circle detection on a single image
    python main.py verify-mask ./input/5520MB_9061_a_R1.jpg ./input/5520MB_9061_a.jpg
"""

import argparse
import logging
import sys
from pathlib import Path

from retoucher import Config, RetouchingPipeline
from retoucher.inpainting import InpaintingBackend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def progress_callback(message: str, current: int, total: int):
    """Print progress to console."""
    if total > 0:
        print(f"[{current}/{total}] {message}")
    else:
        print(message)


def cmd_process(args):
    """Process images through the retouching pipeline."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Create config
    config = Config(
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        top_padding=args.top_padding,
        bottom_padding=args.bottom_padding,
        background_color_hex=args.background_color
    )

    # Select inpainting backend
    try:
        backend = InpaintingBackend(args.inpainting_backend)
    except ValueError:
        print(f"Error: Unknown inpainting backend: {args.inpainting_backend}")
        print(f"Available: {[b.value for b in InpaintingBackend]}")
        sys.exit(1)

    # Create pipeline
    pipeline = RetouchingPipeline(config=config, inpainting_backend=backend)
    pipeline.set_progress_callback(progress_callback)

    # Process
    print(f"\nCanada Goose Photo Retouching Pipeline")
    print(f"=" * 40)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Canvas: {config.canvas_width}x{config.canvas_height}")
    print(f"Padding: top={config.top_padding}px, bottom={config.bottom_padding}px")
    print(f"Safe model height: {config.safe_model_height}px")
    print(f"Background: {config.background_color_hex}")
    print(f"Inpainting: {backend.value}")
    print()

    sku_filter = args.sku if args.sku else None

    results = pipeline.process_directory(
        input_dir,
        output_dir,
        skip_inpainting=args.skip_inpainting,
        sku_filter=sku_filter
    )

    # Print summary
    print(f"\n{'=' * 40}")
    print("PROCESSING SUMMARY")
    print(f"{'=' * 40}")

    for sku_result in results:
        status = "✓" if sku_result.all_success else "⚠"
        print(f"{status} SKU {sku_result.sku}: "
              f"{sku_result.success_count} success, {sku_result.failure_count} failed")

        for result in sku_result.results:
            if not result.success:
                print(f"    ✗ {result.pair.base_name}: {result.error}")

    total_success = sum(r.success_count for r in results)
    total_failed = sum(r.failure_count for r in results)
    print(f"\nTotal: {total_success} successful, {total_failed} failed")


def cmd_validate(args):
    """Validate a directory and show statistics."""
    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)

    pipeline = RetouchingPipeline()
    stats = pipeline.validate_directory(input_dir)

    print(f"\nDirectory Validation: {input_dir}")
    print(f"{'=' * 40}")
    print(f"Valid: {'Yes' if stats['valid'] else 'No'}")
    print(f"SKU Count: {stats['sku_count']}")
    print(f"Total Images: {stats['total_images']}")
    print(f"Images with Retouch Notes: {stats['images_with_notes']}")
    print(f"Images without Retouch Notes: {stats['images_without_notes']}")

    if stats['orphan_retouch_notes']:
        print(f"\nOrphan Retouch Notes (no matching original):")
        for orphan in stats['orphan_retouch_notes']:
            print(f"  - {orphan}")

    print(f"\nSKUs Found:")
    for sku in stats['skus']:
        print(f"  - {sku}")


def cmd_verify_mask(args):
    """Verify red circle detection on a single image."""
    r1_path = Path(args.r1_file)
    original_path = Path(args.original_file)

    if not r1_path.exists():
        print(f"Error: R1 file not found: {r1_path}")
        sys.exit(1)

    if not original_path.exists():
        print(f"Error: Original file not found: {original_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else None

    pipeline = RetouchingPipeline()
    result_path = pipeline.verify_red_detection(r1_path, original_path, output_path)

    print(f"\nRed Circle Detection Verification")
    print(f"{'=' * 40}")
    print(f"R1 File: {r1_path}")
    print(f"Original: {original_path}")
    print(f"Output: {result_path}")
    print(f"\nVisualization saved. Green overlay shows detected red markers.")
    print("Review this file to verify detection accuracy before bulk processing.")


def main():
    parser = argparse.ArgumentParser(
        description="Canada Goose Photo Retouching Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process images through the retouching pipeline"
    )
    process_parser.add_argument("input_dir", help="Input directory with images")
    process_parser.add_argument("output_dir", help="Output directory for processed images")
    process_parser.add_argument(
        "--sku",
        action="append",
        help="Process only specific SKU(s). Can be repeated."
    )
    process_parser.add_argument(
        "--canvas-width",
        type=int,
        default=6000,
        help="Canvas width in pixels (default: 6000)"
    )
    process_parser.add_argument(
        "--canvas-height",
        type=int,
        default=9000,
        help="Canvas height in pixels (default: 9000)"
    )
    process_parser.add_argument(
        "--top-padding",
        type=int,
        default=203,
        help="Top padding in pixels (default: 203)"
    )
    process_parser.add_argument(
        "--bottom-padding",
        type=int,
        default=203,
        help="Bottom padding in pixels (default: 203)"
    )
    process_parser.add_argument(
        "--background-color",
        default="#edeef0",
        help="Background color hex code (default: #edeef0)"
    )
    process_parser.add_argument(
        "--inpainting-backend",
        choices=["placeholder", "stable_diffusion", "openai_dalle"],
        default="placeholder",
        help="Inpainting backend to use (default: placeholder)"
    )
    process_parser.add_argument(
        "--skip-inpainting",
        action="store_true",
        help="Skip the inpainting step (for testing)"
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a directory and show statistics"
    )
    validate_parser.add_argument("input_dir", help="Directory to validate")

    # Verify mask command
    verify_parser = subparsers.add_parser(
        "verify-mask",
        help="Verify red circle detection on a single image"
    )
    verify_parser.add_argument("r1_file", help="Path to the R1 (retouch note) file")
    verify_parser.add_argument("original_file", help="Path to the original file")
    verify_parser.add_argument(
        "--output", "-o",
        help="Output path for visualization (auto-generated if not provided)"
    )

    args = parser.parse_args()

    if args.command == "process":
        cmd_process(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "verify-mask":
        cmd_verify_mask(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
