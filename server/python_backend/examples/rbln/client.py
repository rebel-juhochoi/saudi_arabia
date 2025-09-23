#!/usr/bin/env python3

import numpy as np
import tritonclient.http as httpclient
import sys
import os
import cv2

def test_person_detection():
    """Test person detection model"""
    print("Testing person detection model...")
    
    # Create client
    triton_client = httpclient.InferenceServerClient(url="localhost:8000")
    
    # Load a test image
    test_image_path = "/workspace/projects/global/saudi_arabia/gender_detection/data/inputs/01_man.mp4"
    cap = cv2.VideoCapture(test_image_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not load test image")
        return False
    
    # Prepare input
    input_tensor = httpclient.InferInput("INPUT__0", frame.shape, "UINT8")
    input_tensor.set_data_from_numpy(frame)
    
    # Prepare outputs
    outputs = [
        httpclient.InferRequestedOutput("OUTPUT__0"),  # boxes
        httpclient.InferRequestedOutput("OUTPUT__1"),  # confidences
        httpclient.InferRequestedOutput("OUTPUT__2"),  # class_ids
        httpclient.InferRequestedOutput("OUTPUT__3"),  # masks
    ]
    
    try:
        # Run inference
        response = triton_client.infer(
            model_name="person_detection",
            inputs=[input_tensor],
            outputs=outputs
        )
        
        # Get results
        boxes = response.as_numpy("OUTPUT__0")
        confidences = response.as_numpy("OUTPUT__1")
        class_ids = response.as_numpy("OUTPUT__2")
        masks = response.as_numpy("OUTPUT__3")
        
        print(f"✅ Person detection successful!")
        print(f"   Detected {len(boxes)} objects")
        print(f"   Boxes shape: {boxes.shape}")
        print(f"   Confidences shape: {confidences.shape}")
        print(f"   Class IDs shape: {class_ids.shape}")
        print(f"   Masks shape: {masks.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Person detection failed: {e}")
        return False

def test_gender_classification():
    """Test gender classification model"""
    print("Testing gender classification model...")
    
    # Create client
    triton_client = httpclient.InferenceServerClient(url="localhost:8000")
    
    # Create a dummy person image (224x224x3)
    person_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Prepare input
    input_tensor = httpclient.InferInput("INPUT__0", person_image.shape, "UINT8")
    input_tensor.set_data_from_numpy(person_image)
    
    # Prepare outputs
    outputs = [
        httpclient.InferRequestedOutput("OUTPUT__0"),  # gender
        httpclient.InferRequestedOutput("OUTPUT__1"),  # confidence
    ]
    
    try:
        # Run inference
        response = triton_client.infer(
            model_name="gender_classification",
            inputs=[input_tensor],
            outputs=outputs
        )
        
        # Get results
        gender = response.as_numpy("OUTPUT__0")
        confidence = response.as_numpy("OUTPUT__1")
        
        print(f"✅ Gender classification successful!")
        print(f"   Gender: {gender}")
        print(f"   Confidence: {confidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gender classification failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing RBLN Triton Models")
    print("=" * 40)
    
    # Test person detection
    person_success = test_person_detection()
    print()
    
    # Test gender classification
    gender_success = test_gender_classification()
    print()
    
    if person_success and gender_success:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
