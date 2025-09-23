# Gender Classification Model

This directory contains the gender classification model files.

## Expected Files

- `deepface_gender.rbln` - Compiled RBLN model for gender classification (371MB)
- `model.py` - Model implementation code
- `compile.py` - Model compilation script

## Model Information

- **Type**: RBLN (Rapid Binary Learning Network) model
- **Purpose**: Gender classification from face images
- **Size**: ~371MB (gitignored due to size)
- **Format**: `.rbln` binary model file

## Usage

1. Run `compile.py` to compile the model
2. The compiled model will be saved as `deepface_gender.rbln`
3. Use `model.py` to load and run inference

## Note

The compiled model file (`deepface_gender.rbln`) is gitignored due to its large size. You need to compile the model locally to generate this file.
