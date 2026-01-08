"""
Ingestion module for scanning directories and pairing images with retouch notes.

Handles the file discovery and pairing logic based on naming conventions:
- Original: {SKU}_{ID}_a.jpg (e.g., 5520MB_9061_a.jpg)
- Retouch Note: {SKU}_{ID}_a_R1.jpg (e.g., 5520MB_9061_a_R1.jpg)
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from .config import Config


@dataclass
class ImagePair:
    """Represents a paired original image and its retouch note."""

    original_path: Path
    retouch_note_path: Optional[Path]
    sku: str
    image_id: str

    @property
    def base_name(self) -> str:
        """Get the base filename without extension."""
        return f"{self.sku}_{self.image_id}_a"

    @property
    def has_retouch_note(self) -> bool:
        """Check if this pair has a retouch note file."""
        return self.retouch_note_path is not None

    def __repr__(self) -> str:
        note_status = "✓" if self.has_retouch_note else "✗"
        return f"ImagePair({self.base_name}, retouch_note={note_status})"


@dataclass
class SKUGroup:
    """A group of images belonging to the same SKU."""

    sku: str
    pairs: List[ImagePair]

    @property
    def image_count(self) -> int:
        """Total number of images in this SKU group."""
        return len(self.pairs)

    @property
    def pairs_with_notes(self) -> List[ImagePair]:
        """Get only pairs that have retouch notes."""
        return [p for p in self.pairs if p.has_retouch_note]

    def __repr__(self) -> str:
        notes = len(self.pairs_with_notes)
        return f"SKUGroup({self.sku}, images={self.image_count}, with_notes={notes})"


class ImageIngestion:
    """
    Handles scanning directories and pairing images with their retouch notes.

    The pairing logic identifies files by their naming convention:
    - Original images end with '_a.jpg'
    - Retouch notes end with '_a_R1.jpg'
    """

    # Pattern to extract SKU and ID from filename
    # Matches: {SKU}_{ID}_a.jpg or {SKU}_{ID}_a_R1.jpg
    FILENAME_PATTERN = re.compile(
        r'^(?P<sku>[A-Za-z0-9]+)_(?P<id>[A-Za-z0-9]+)_a(?P<suffix>_R1)?\.(?P<ext>jpg|jpeg|png)$',
        re.IGNORECASE
    )

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the ingestion handler.

        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or Config()

    def scan_directory(self, directory: Union[str, Path]) -> Dict[str, SKUGroup]:
        """
        Scan a directory and group images by SKU.

        Args:
            directory: Path to the directory containing images.

        Returns:
            Dictionary mapping SKU codes to SKUGroup objects.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Collect all matching files
        originals: Dict[str, Path] = {}
        retouch_notes: Dict[str, Path] = {}

        for file_path in directory.iterdir():
            if not file_path.is_file():
                continue

            match = self.FILENAME_PATTERN.match(file_path.name)
            if not match:
                continue

            sku = match.group('sku')
            image_id = match.group('id')
            is_retouch_note = match.group('suffix') is not None
            key = f"{sku}_{image_id}"

            if is_retouch_note:
                retouch_notes[key] = file_path
            else:
                originals[key] = file_path

        # Create pairs and group by SKU
        sku_groups: Dict[str, List[ImagePair]] = {}

        for key, original_path in originals.items():
            sku, image_id = key.rsplit('_', 1)
            retouch_path = retouch_notes.get(key)

            pair = ImagePair(
                original_path=original_path,
                retouch_note_path=retouch_path,
                sku=sku,
                image_id=image_id
            )

            if sku not in sku_groups:
                sku_groups[sku] = []
            sku_groups[sku].append(pair)

        # Convert to SKUGroup objects
        return {
            sku: SKUGroup(sku=sku, pairs=pairs)
            for sku, pairs in sku_groups.items()
        }

    def get_sku_list(self, directory: Union[str, Path]) -> List[str]:
        """
        Get a list of all SKU codes found in a directory.

        Args:
            directory: Path to scan.

        Returns:
            Sorted list of SKU codes.
        """
        groups = self.scan_directory(directory)
        return sorted(groups.keys())

    def get_processing_queue(
        self,
        directory: Union[str, Path],
        require_retouch_note: bool = True
    ) -> List[ImagePair]:
        """
        Get a flat list of image pairs ready for processing.

        Args:
            directory: Path to scan.
            require_retouch_note: If True, only include pairs with retouch notes.

        Returns:
            List of ImagePair objects to process.
        """
        groups = self.scan_directory(directory)
        queue = []

        for sku_group in groups.values():
            for pair in sku_group.pairs:
                if require_retouch_note and not pair.has_retouch_note:
                    continue
                queue.append(pair)

        return queue

    def validate_directory(self, directory: Union[str, Path]) -> Dict[str, any]:
        """
        Validate a directory and return statistics about its contents.

        Args:
            directory: Path to validate.

        Returns:
            Dictionary with validation results and statistics.
        """
        directory = Path(directory)
        groups = self.scan_directory(directory)

        total_originals = sum(g.image_count for g in groups.values())
        total_with_notes = sum(len(g.pairs_with_notes) for g in groups.values())
        orphan_notes = self._find_orphan_notes(directory, groups)

        return {
            "valid": True,
            "directory": str(directory),
            "sku_count": len(groups),
            "total_images": total_originals,
            "images_with_notes": total_with_notes,
            "images_without_notes": total_originals - total_with_notes,
            "orphan_retouch_notes": orphan_notes,
            "skus": list(groups.keys())
        }

    def _find_orphan_notes(
        self,
        directory: Path,
        groups: Dict[str, SKUGroup]
    ) -> List[str]:
        """Find retouch notes that don't have matching originals."""
        all_pair_keys = set()
        for group in groups.values():
            for pair in group.pairs:
                all_pair_keys.add(f"{pair.sku}_{pair.image_id}")

        orphans = []
        for file_path in directory.iterdir():
            if not file_path.is_file():
                continue

            match = self.FILENAME_PATTERN.match(file_path.name)
            if match and match.group('suffix'):
                key = f"{match.group('sku')}_{match.group('id')}"
                if key not in all_pair_keys:
                    orphans.append(file_path.name)

        return orphans
