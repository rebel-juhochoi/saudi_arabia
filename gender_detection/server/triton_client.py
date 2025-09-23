import tritonclient.http as httpclient
import numpy as np
from typing import Tuple
import asyncio

class TritonInferenceClient:
    """Client for Triton Inference Server"""
    
    def __init__(self, triton_url: str = "localhost:8000"):
        self.triton_url = triton_url
        self.client = httpclient.InferenceServerClient(url=f"http://{triton_url}")
    
    def detect_persons(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Detect persons in image using Triton person detection model"""
        try:
            # Prepare input
            input_tensor = httpclient.InferInput("INPUT__0", image.shape, "UINT8")
            input_tensor.set_data_from_numpy(image)
            
            # Prepare outputs
            outputs = [
                httpclient.InferRequestedOutput("OUTPUT__0"),  # boxes
                httpclient.InferRequestedOutput("OUTPUT__1"),  # confidences
                httpclient.InferRequestedOutput("OUTPUT__2"),  # class_ids
                httpclient.InferRequestedOutput("OUTPUT__3")   # masks
            ]
            
            # Run inference
            response = self.client.infer(
                model_name="person_detection",
                inputs=[input_tensor],
                outputs=outputs
            )
            
            # Extract results
            boxes = response.as_numpy("OUTPUT__0")
            confidences = response.as_numpy("OUTPUT__1")
            class_ids = response.as_numpy("OUTPUT__2")
            masks = response.as_numpy("OUTPUT__3")
            
            return boxes, confidences, class_ids, masks
            
        except Exception as e:
            print(f"Error in person detection: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def classify_gender(self, person_image: np.ndarray) -> Tuple[str, float]:
        """Classify gender from person image using Triton gender classification model"""
        try:
            # Prepare input
            input_tensor = httpclient.InferInput("INPUT__0", person_image.shape, "UINT8")
            input_tensor.set_data_from_numpy(person_image)
            
            # Prepare outputs
            outputs = [
                httpclient.InferRequestedOutput("OUTPUT__0"),  # gender
                httpclient.InferRequestedOutput("OUTPUT__1")   # confidence
            ]
            
            # Run inference
            response = self.client.infer(
                model_name="gender_classification",
                inputs=[input_tensor],
                outputs=outputs
            )
            
            # Extract results
            gender = response.as_numpy("OUTPUT__0")[0].decode('utf-8')
            confidence = float(response.as_numpy("OUTPUT__1")[0])
            
            return gender, confidence
            
        except Exception as e:
            print(f"Error in gender classification: {e}")
            return "Unknown", 0.0
    
    def is_server_ready(self) -> bool:
        """Check if Triton server is ready"""
        try:
            return self.client.is_server_ready()
        except:
            return False
    
    def get_model_status(self, model_name: str) -> bool:
        """Check if specific model is ready"""
        try:
            return self.client.is_model_ready(model_name)
        except:
            return False
