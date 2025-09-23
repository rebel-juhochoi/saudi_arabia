import requests
import numpy as np
from typing import Tuple
import json

class TritonHTTPClient:
    """HTTP client for Triton-compatible server"""
    
    def __init__(self, triton_url: str = "localhost:8002"):
        self.triton_url = triton_url
        self.base_url = f"http://{triton_url}"
    
    def detect_persons(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Detect persons in image using Triton person detection model"""
        try:
            # Prepare request
            request_data = {
                "inputs": [
                    {
                        "name": "INPUT__0",
                        "shape": image.shape,
                        "data": image.tolist()
                    }
                ]
            }
            
            # Run inference
            response = requests.post(
                f"{self.base_url}/v2/models/person_detection/infer",
                json=request_data,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            result = response.json()
            outputs = result["outputs"]
            
            # Extract results
            boxes = np.array(outputs[0]["data"]).reshape(outputs[0]["shape"])
            confidences = np.array(outputs[1]["data"]).reshape(outputs[1]["shape"])
            class_ids = np.array(outputs[2]["data"]).reshape(outputs[2]["shape"])
            masks = np.array(outputs[3]["data"]).reshape(outputs[3]["shape"])
            
            return boxes, confidences, class_ids, masks
            
        except Exception as e:
            print(f"Error in person detection: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def classify_gender(self, person_image: np.ndarray) -> Tuple[str, float]:
        """Classify gender from person image using Triton gender classification model"""
        try:
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
            
            # Run inference
            response = requests.post(
                f"{self.base_url}/v2/models/gender_classification/infer",
                json=request_data,
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            
            result = response.json()
            outputs = result["outputs"]
            
            # Extract results
            gender = outputs[0]["data"][0]
            confidence = float(outputs[1]["data"][0])
            
            return gender, confidence
            
        except Exception as e:
            print(f"Error in gender classification: {e}")
            return "Unknown", 0.0
    
    def is_server_ready(self) -> bool:
        """Check if Triton server is ready"""
        try:
            response = requests.get(f"{self.base_url}/v2/health/ready", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_model_status(self, model_name: str) -> bool:
        """Check if specific model is ready"""
        try:
            response = requests.get(f"{self.base_url}/v2/models/{model_name}", timeout=5)
            return response.status_code == 200
        except:
            return False
