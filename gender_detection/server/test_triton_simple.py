#!/usr/bin/env python3
"""
Simple test for Triton-compatible server
"""

import requests
import numpy as np
import cv2
import json

def test_triton_server():
    """Test the Triton-compatible server"""
    base_url = "http://localhost:8000"
    
    # Test health check
    print("1. Testing health check...")
    try:
        response = requests.get(f"{base_url}/v2/health/ready")
        print(f"   Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"   Health check failed: {e}")
        return False
    
    # Test person detection
    print("2. Testing person detection...")
    try:
        # Load a test image
        test_image_path = "/workspace/projects/global/saudi_arabia/gender_detection/data/inputs/01_man.mp4"
        cap = cv2.VideoCapture(test_image_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("   Error: Could not load test image")
            return False
        
        # Prepare request
        request_data = {
            "inputs": [
                {
                    "name": "INPUT__0",
                    "shape": frame.shape,
                    "data": frame.tolist()
                }
            ]
        }
        
        print(f"   Image shape: {frame.shape}")
        print(f"   Request data size: {len(json.dumps(request_data))} bytes")
        
        # Send request
        response = requests.post(
            f"{base_url}/v2/models/person_detection/infer",
            json=request_data,
            timeout=30
        )
        
        print(f"   Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Response text: {response.text}")
            return False
        
        result = response.json()
        print(f"   Response keys: {result.keys()}")
        if "outputs" in result:
            print(f"   Number of outputs: {len(result['outputs'])}")
            for i, output in enumerate(result['outputs']):
                print(f"   Output {i}: {output['name']} - shape {output['shape']}")
        
        print("   ✅ Person detection successful!")
        
    except Exception as e:
        print(f"   ❌ Person detection failed: {e}")
        return False
    
    # Test gender classification
    print("3. Testing gender classification...")
    try:
        # Create a dummy person image
        person_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Prepare request
        request_data = {
            "inputs": [
                {
                    "name": "INPUT__0",
                    "shape": person_image.shape,
                    "data": person_image.tolist()
                }
            ]
        }
        
        # Send request
        response = requests.post(
            f"{base_url}/v2/models/gender_classification/infer",
            json=request_data,
            timeout=30
        )
        
        print(f"   Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"   Response text: {response.text}")
            return False
        
        result = response.json()
        print(f"   Response: {result}")
        
        print("   ✅ Gender classification successful!")
        
    except Exception as e:
        print(f"   ❌ Gender classification failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Triton-compatible server")
    print("=" * 40)
    
    success = test_triton_server()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
