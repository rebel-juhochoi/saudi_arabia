# Person Detection Model

This directory contains the person detection model files.

## Expected Files

- `yolo11n-seg.pt` - YOLO11 segmentation model (5.9MB)
- `yolo11n-seg.rbln` - Compiled RBLN model for person detection (20MB)
- `model.py` - Model implementation code
- `compile.py` - Model compilation script

## Model Information

- **Type**: YOLO11 + RBLN models
- **Purpose**: Person detection and segmentation
- **Sizes**: 
  - `.pt` file: ~5.9MB (gitignored due to size)
  - `.rbln` file: ~20MB (gitignored due to size)
- **Format**: PyTorch `.pt` and RBLN `.rbln` files

## Usage

1. Download the YOLO11 model weights
2. Run `compile.py` to compile the RBLN model
3. Use `model.py` to load and run inference

## Note

The model files (`.pt` and `.rbln`) are gitignored due to their large sizes. You need to download/compile these models locally.
