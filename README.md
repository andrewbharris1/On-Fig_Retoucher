# Canada Goose Photo Retouching Pipeline

Automated photo retouching pipeline for product images following Canada Goose brand standards.

## Features

- **Ingestion & Pairing**: Automatically pairs original images with their retouch notes (`_R1` files)
- **Red Marker Detection**: Uses OpenCV to detect red circles drawn by human editors
- **Model Segmentation**: Separates model from background using `rembg` with shadow preservation
- **Deterministic Geometry**: Exact 203px padding logic with no AI guessing
- **Generative Inpainting**: Placeholder for Stable Diffusion/DALL-E integration

## Brand Standards

| Parameter | Value |
|-----------|-------|
| Canvas Height | 9000px |
| Top Padding | 203px |
| Bottom Padding | 203px |
| Safe Model Height | 8594px |
| Background Color | #edeef0 |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Process Images

```bash
# Process all images in a directory
python main.py process ./input ./output

# Process specific SKUs
python main.py process ./input ./output --sku 5520MB --sku 6612LB

# Skip inpainting (for testing)
python main.py process ./input ./output --skip-inpainting
```

### Validate Directory

```bash
python main.py validate ./input
```

### Verify Red Circle Detection

Before bulk processing, verify the red marker detection:

```bash
python main.py verify-mask ./input/5520MB_9061_a_R1.jpg ./input/5520MB_9061_a.jpg
```

## File Naming Convention

- **Original**: `{SKU}_{ID}_a.jpg` (e.g., `5520MB_9061_a.jpg`)
- **Retouch Note**: `{SKU}_{ID}_a_R1.jpg` (e.g., `5520MB_9061_a_R1.jpg`)

## Module Structure

```
retoucher/
├── __init__.py          # Package exports
├── config.py            # Configuration constants
├── ingestion.py         # File scanning and pairing
├── mask_extraction.py   # Red marker detection (OpenCV)
├── segmentation.py      # Model segmentation (rembg)
├── geometry.py          # Deterministic cropping/placement
├── inpainting.py        # Generative AI integration
└── pipeline.py          # Main orchestration
```

## Inpainting Backends

The pipeline supports multiple inpainting backends:

- `placeholder`: OpenCV-based (for testing, no AI)
- `stable_diffusion`: Stable Diffusion XL API
- `openai_dalle`: OpenAI DALL-E editing API

## Prompts

The system uses context-aware prompts based on mask location:

**Fabric** (body area):
> "Remove wrinkles and hard crease in fabric, maintain the texture, pattern and colour of the fabric."

**Skin** (upper 20% of image):
> "Keep skin natural, use dodge and burn technique to even out skin. No frequency separation. Remove red zits, keep dark beauty marks or moles."

## Python API

```python
from retoucher import RetouchingPipeline, Config

# Create pipeline with custom config
config = Config(
    canvas_width=6000,
    canvas_height=9000,
    top_padding=203,
    bottom_padding=203
)

pipeline = RetouchingPipeline(config=config)

# Process a directory
results = pipeline.process_directory("./input", "./output")

# Check results
for sku_result in results:
    print(f"SKU {sku_result.sku}: {sku_result.success_count} success")
```

## Testing

```bash
pytest tests/ -v
```
