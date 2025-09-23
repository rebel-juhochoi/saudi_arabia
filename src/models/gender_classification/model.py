import cv2
import numpy as np
import rebel
from pathlib import Path
import sys

# Add deepface to path for preprocessing
deepface_path = Path(__file__).parent / "deepface"
sys.path.insert(0, str(deepface_path))

from deepface.commons import package_utils

class GenderClassifier:
    """
    Gender classification using deepface model compiled with rebel-compiler
    """
    def __init__(self, conf_threshold=0.5):
        """
        Initialize gender classifier
        """
        self.conf_threshold = conf_threshold
        self.model_name = "deepface_gender"
        
        # Load compiled model
        model_root = Path(__file__).parent
        compiled_model_path = model_root / f"{self.model_name}.rbln"
        
        if not compiled_model_path.exists():
            raise FileNotFoundError(f"Compiled model not found at {compiled_model_path}")
        
        self.module = rebel.Runtime(str(compiled_model_path))
        
        # Gender labels
        self.labels = ["Woman", "Man"]
        
        print(f"GenderClassifier initialized with model: {self.model_name}")
    
    def preprocess_person(self, person_image):
        """
        Preprocess person image for gender classification
        The deepface model will handle face detection internally
        Args:
            person_image: Person image (BGR format)
        Returns:
            Preprocessed image ready for model input
        """
        # Convert BGR to RGB
        person_rgb = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224 (required by deepface model)
        person_resized = cv2.resize(person_rgb, (224, 224))
        
        # Normalize to [0, 1]
        person_normalized = person_resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        person_batch = np.expand_dims(person_normalized, axis=0)
        
        return person_batch
    
    def postprocess_gender(self, prediction):
        """
        Postprocess gender prediction
        Args:
            prediction: Raw model output
        Returns:
            Gender string ("Man" or "Woman") and confidence
        """
        # Get probabilities for each gender
        probabilities = prediction[0]  # Shape: (2,)
        
        # Get the predicted gender index
        gender_idx = np.argmax(probabilities)
        confidence = float(probabilities[gender_idx])
        
        # Get gender label
        gender = self.labels[gender_idx]
        
        return gender, confidence
    
    def classify_gender(self, person_image):
        """
        Classify gender from person image
        The deepface model will handle face detection internally
        Args:
            person_image: Person image (BGR format)
        Returns:
            Tuple of (gender, confidence) where gender is "Man" or "Woman"
        """
        try:
            # Preprocess the person image
            processed_person = self.preprocess_person(person_image)
            
            # Run inference (deepface model handles face detection internally)
            prediction = self.module.run(processed_person)
            
            # Postprocess results
            gender, confidence = self.postprocess_gender(prediction)
            
            return gender, confidence
            
        except Exception as e:
            print(f"Error in gender classification: {e}")
            return "Unknown", 0.0
    
    def extract_person_from_bbox(self, frame, bbox, mask=None):
        """
        Extract person region from frame using bounding box and optional segmentation mask
        Args:
            frame: Full frame image
            bbox: Bounding box [x1, y1, x2, y2]
            mask: Optional segmentation mask for the person
        Returns:
            Cropped person image or None if extraction fails
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            # Extract person region
            person_crop = frame[y1:y2, x1:x2]
            
            # If mask is provided, apply it to get only the segmented person area
            if mask is not None:
                # Resize mask to match crop size
                mask_resized = cv2.resize(mask, (person_crop.shape[1], person_crop.shape[0]))
                
                # Ensure mask is uint8 and binary
                if mask_resized.dtype != np.uint8:
                    mask_resized = (mask_resized * 255).astype(np.uint8)
                
                # Ensure mask is binary (0 or 255)
                mask_resized = (mask_resized > 0.5).astype(np.uint8) * 255
                
                # Apply mask to person crop
                person_crop = cv2.bitwise_and(person_crop, person_crop, mask=mask_resized)
            
            # Check if crop is valid
            if person_crop.size == 0 or person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
                return None
            
            return person_crop
            
        except Exception as e:
            print(f"Error extracting person: {e}")
            return None