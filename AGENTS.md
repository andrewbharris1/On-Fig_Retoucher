# AGENTS.md - Codebase Patterns for AI Agents

This file contains patterns, conventions, and learnings for AI agents working on this codebase.

## Project Overview

**Canada Goose Photo Retouching Pipeline** - Automated photo retouching for product images following brand standards.

## Code Patterns

### Configuration

- All config in `retoucher/config.py` using `@dataclass`
- Use properties for computed values (e.g., `safe_model_height`)
- HSV color ranges for OpenCV detection stored as tuples

### Module Structure

```
retoucher/
├── config.py          # Dataclass-based configuration
├── ingestion.py       # File scanning, regex-based pairing
├── mask_extraction.py # OpenCV HSV-based red detection
├── segmentation.py    # rembg integration (lazy-loaded)
├── geometry.py        # Deterministic pixel math
├── inpainting.py      # Multi-backend AI integration
└── pipeline.py        # Orchestration with lazy loading
```

### Lazy Loading Pattern

Heavy dependencies (rembg, AI APIs) are lazy-loaded:
```python
@property
def segmenter(self) -> ModelSegmenter:
    if self._segmenter is None:
        self._segmenter = ModelSegmenter(self.config)
    return self._segmenter
```

### Result Dataclasses

Use `@dataclass` for result objects with computed properties:
```python
@dataclass
class ProcessingResult:
    pair: ImagePair
    success: bool
    output_path: Optional[Path] = None
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
```

### Type Hints

- Use `Union[str, Path]` for path parameters (Python 3.9 compatibility)
- Use `Optional[X]` for nullable types
- Use `Tuple[int, int, int]` for color values

### Error Handling

- Catch exceptions in processing methods
- Return result objects with `success: bool` and `error: str`
- Log exceptions with `logger.exception()`

## Testing

- Tests in `tests/` directory
- Use `pytest` for testing
- Run syntax check: `python -m py_compile main.py retoucher/*.py`

## CLI

- CLI in `main.py` using standard argparse
- Commands: `process`, `validate`, `verify-mask`
- Use `--skip-inpainting` for testing without AI

## Gotchas

1. **OpenCV uses BGR, not RGB** - Convert with `cv2.cvtColor()` or swap channels
2. **rembg is slow to load** - Always use lazy loading pattern
3. **Path handling** - Always convert to `Path` objects early with `Path(input_dir)`
4. **JPEG quality** - Use `cv2.IMWRITE_JPEG_QUALITY` constant for save quality

## Quality Checks

Before marking any story as complete:

```bash
# Tests
pytest tests/ -v

# Syntax check
python -m py_compile main.py retoucher/*.py

# CLI sanity check
python main.py --help
```

## File Naming

- Original: `{SKU}_{ID}_a.jpg`
- Retouch note: `{SKU}_{ID}_a_R1.jpg`
- Output: `{basename}_retouched.jpg`
