#!/usr/bin/env python3
"""
Compile gender classification model from deepface repository using rebel-compiler
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Set TensorFlow environment to fix deepface compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from deepface.models.demography.Gender import load_model
import rebel

def compile_gender_model():
    """
    Compile the gender classification model from deepface using rebel-compiler
    """
    print("Loading gender classification model from deepface...")
    
    # Import deepface modules with proper error handling
    
    # Load the gender model
    model = load_model()
    
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Test the model with a sample input
    print("Testing model with sample input...")
    sample_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    prediction = model(sample_input)
    print(f"Sample prediction shape: {prediction.shape}")
    print(f"Sample prediction: {prediction}")
    
    # Convert model to tf.function for compilation
    print("Converting model to tf.function...")
    func = tf.function(lambda input_img: model(input_img))
    
    # Compile the model using rebel-compiler
    print("Compiling with rebel-compiler...")   
        
    # Define input info for compilation (224x224 as expected by deepface model)
    input_info = [('input_img', [1, 224, 224, 3], tf.float32)]
    
    # Compile the model using the official RBLN approach
    compiled_model = rebel.compile_from_tf_function(
        func,
        input_info,
    )
    
    # Save the compiled model
    compiled_model_path = Path(__file__).parent / "deepface_gender.rbln"
    compiled_model.save(str(compiled_model_path))
    
    print(f"Compiled model saved to {compiled_model_path}")
    return True


if __name__ == "__main__":
    success = compile_gender_model()
    if not success:
        print("Compilation failed!")
        sys.exit(1)
    else:
        print("Compilation completed successfully!")
