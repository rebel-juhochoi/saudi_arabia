import cv2
import numpy as np
from pathlib import Path
import rebel
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression


class PersonDetector:
    """
    Person detector with tracking capabilities
    """
    def __init__(self, conf=0.25, enable_color_detection=False):
        """
        Initialize detector 
        """
        self.model_name = "yolo11n-seg"
        self.conf = conf
        self.enable_color_detection = enable_color_detection
        
        # Load YOLO model for configuration
        model_root = Path(__file__).parent
        pt_path = model_root / f"{self.model_name}.pt"
        self.cfg = self._load_model_config(pt_path, conf)
        
        # Load compiled model           
        compiled_model_path = model_root / f"{self.model_name}.rbln"
        self.module = rebel.Runtime(str(compiled_model_path))
        
        print(f"PersonDetector initialized: {self.model_name}")
    
    def _load_model_config(self, pt_path, conf):
        """Load model configuration"""
        model = YOLO(pt_path)
        num_classes = len(model.names)
        box_cls_dim = num_classes + 4
        return dict(
            num_classes=num_classes,
            box_cls_dim=box_cls_dim,
            conf=model.overrides.get("conf", conf),
            iou=model.overrides.get("iou", 0.45),
            max_det=model.overrides.get("max_det", 1000),
            names=model.names,
            agnostic_nms=model.overrides.get("agnostic_nms", False),
            classes=model.overrides.get("classes", [0]),
            retina_masks=False,
        )
    
    def preprocess(self, frame):
        """Preprocess frame for model input"""
        frame = LetterBox(new_shape=(640, 640))(image=frame)
        frame = frame.transpose((2, 0, 1))[::-1]
        return (frame[None] / 255).astype(np.float32).copy()
    
    def postprocess(self, preds, batch, orig_shape):
        """Postprocess raw tensor outputs to dictionary format per frame"""
        box_cls = torch.from_numpy(preds[0][:, : self.cfg["box_cls_dim"], :])
        masks = torch.from_numpy(preds[4])
        new_preds = [torch.cat((box_cls, masks), dim=1), tuple(torch.from_numpy(p) for p in preds[1:6])]
        p = non_max_suppression(
            new_preds[0],
            self.cfg["conf"],
            self.cfg["iou"],
            agnostic=self.cfg["agnostic_nms"],
            max_det=self.cfg["max_det"],
            nc=self.cfg["num_classes"],
            classes=self.cfg["classes"],
        )
        
        proto = new_preds[1][-1] if isinstance(new_preds[1], tuple) else new_preds[1]
        results = []
        # Handle single prediction (batch size 1)
        for pred in p:
            if not len(pred):
                # No detections
                result_dict = {
                    'boxes': None,
                    'conf': None,
                    'cls': None,
                    'masks': None,
                    'orig_shape': orig_shape,
                    'names': self.cfg["names"]
                }
            else:
                # Process detections
                if self.cfg["retina_masks"]:
                    pred[:, :4] = ops.scale_boxes(batch.shape[2:], pred[:, :4], orig_shape)
                    masks = ops.process_mask_native(proto[0], pred[:, 6:], pred[:, :4], orig_shape[:2])
                else:
                    masks = ops.process_mask(
                        proto[0], pred[:, 6:], pred[:, :4], batch.shape[2:], upsample=True
                    )
                    pred[:, :4] = ops.scale_boxes(batch.shape[2:], pred[:, :4], orig_shape)
                
                # Create result dictionary with same structure as Results object
                result_dict = {
                    'boxes': pred[:, :4],  # xyxy coordinates
                    'conf': pred[:, 4],    # confidence scores
                    'cls': pred[:, 5],     # class ids
                    'masks': masks,        # segmentation masks
                    'orig_shape': orig_shape,
                    'names': self.cfg["names"]
                }
            
            results.append(result_dict)
        return results
    
    def detect_color(self, frame, bbox, mask):
        """
        Detect color percentages in the segmentation area, excluding skin color
        Args:
            frame: Original frame
            bbox: Bounding box [x1, y1, x2, y2]
            mask: Segmentation mask
        Returns:
            Dictionary with color percentages: {'white': 0.3, 'red': 0.2, 'black': 0.5, ...}
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
            
            if person_crop.size == 0:
                return {'unknown': 1.0}
            
            # Resize mask to match crop size
            mask_resized = cv2.resize(mask, (person_crop.shape[1], person_crop.shape[0]))
            
            # Ensure mask is binary
            if mask_resized.dtype != np.uint8:
                mask_resized = (mask_resized * 255).astype(np.uint8)
            
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # Apply mask to get only the segmented person area
            person_masked = cv2.bitwise_and(person_crop, person_crop, mask=mask_binary)
            
            # Get only the masked pixels
            masked_pixels = person_masked[mask_binary == 1]
            
            if len(masked_pixels) == 0:
                return {'unknown': 1.0}
            
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(person_masked, cv2.COLOR_BGR2HSV)
            masked_hsv = hsv[mask_binary == 1]
            
            if len(masked_hsv) == 0:
                return {'unknown': 1.0}
            
            # Detect skin color and create exclusion mask
            skin_mask = self._detect_skin_color(masked_hsv)
            
            # Exclude skin pixels from clothing analysis
            clothing_hsv = masked_hsv[~skin_mask]
            
            if len(clothing_hsv) == 0:
                return {'unknown': 1.0}
            
            # Calculate color percentages for clothing only
            color_percentages = self._calculate_color_percentages(clothing_hsv)
            
            return color_percentages
                
        except Exception as e:
            print(f"Error in color detection: {e}")
            return {'unknown': 1.0}
    
    def _classify_color_hsv(self, mean_hue, mean_saturation, mean_brightness, median_hue, has_pattern):
        """
        Classify color based on HSV values
        """
        # Black: low brightness, low saturation
        if mean_brightness < 50 and mean_saturation < 30:
            return 'black'
        
        # White: high brightness, low saturation
        elif mean_brightness > 200 and mean_saturation < 30:
            return 'white'
        
        # Gray: medium brightness, low saturation
        elif 50 <= mean_brightness <= 200 and mean_saturation < 30:
            return 'gray'
        
        # Red: hue around 0 or 179 (red range)
        elif (0 <= mean_hue <= 10 or 170 <= mean_hue <= 179) and mean_saturation > 50:
            # Check for patterns in red
            if has_pattern and mean_brightness > 150:
                return 'red_pattern'  # Red with stripes/checks
            return 'red'
        
        # Pink: red hue with high brightness and medium saturation
        elif (0 <= mean_hue <= 10 or 170 <= mean_hue <= 179) and mean_saturation > 30 and mean_brightness > 150:
            return 'pink'
        
        # Brown: red-orange hue with low brightness and medium saturation
        elif (10 <= mean_hue <= 25) and mean_saturation > 40 and mean_brightness < 120:
            return 'brown'
        
        # Orange: hue around 15-25
        elif 15 <= mean_hue <= 25 and mean_saturation > 50:
            return 'orange'
        
        # Yellow: hue around 25-35
        elif 25 <= mean_hue <= 35 and mean_saturation > 50:
            return 'yellow'
        
        # Green: hue around 35-85
        elif 35 <= mean_hue <= 85 and mean_saturation > 50:
            return 'green'
        
        # Blue: hue around 85-130
        elif 85 <= mean_hue <= 130 and mean_saturation > 50:
            return 'blue'
        
        # Purple: hue around 130-170
        elif 130 <= mean_hue <= 170 and mean_saturation > 50:
            return 'purple'
        
        # Check for white patterns
        elif mean_brightness > 180 and mean_saturation < 40 and has_pattern:
            return 'white_pattern'  # White with stripes/checks
        
        # Default to unknown if no clear classification
        return 'unknown'
    
    def _detect_pattern(self, person_masked, mask_binary):
        """
        Detect stripes or check patterns in the clothing
        """
        try:
            # Convert to grayscale for pattern detection
            gray = cv2.cvtColor(person_masked, cv2.COLOR_BGR2GRAY)
            masked_gray = gray[mask_binary == 1]
            
            if len(masked_gray) == 0:
                return False
            
            # Reshape to 2D for analysis
            h, w = person_masked.shape[:2]
            masked_2d = masked_gray.reshape(-1, 1)
            
            # Simple pattern detection using edge detection
            edges = cv2.Canny(gray, 50, 150)
            masked_edges = edges[mask_binary == 1]
            
            # Count edge pixels
            edge_ratio = np.sum(masked_edges > 0) / len(masked_edges) if len(masked_edges) > 0 else 0
            
            # Pattern threshold - adjust based on testing
            pattern_threshold = 0.1  # 10% of pixels are edges
            
            return edge_ratio > pattern_threshold
            
        except Exception as e:
            print(f"Error in pattern detection: {e}")
            return False
    
    def _detect_skin_color(self, hsv_pixels):
        """
        Detect skin color pixels in HSV space
        Args:
            hsv_pixels: Array of HSV pixel values
        Returns:
            Boolean mask where True indicates skin pixels
        """
        try:
            h_values = hsv_pixels[:, 0]  # Hue
            s_values = hsv_pixels[:, 1]  # Saturation
            v_values = hsv_pixels[:, 2]  # Value
            
            # Skin color ranges in HSV
            # Hue: 0-20 (red-orange) and 160-179 (red-purple)
            # Saturation: 20-255 (not too desaturated)
            # Value: 20-255 (not too dark)
            
            skin_mask = (
                ((h_values >= 0) & (h_values <= 20)) | 
                ((h_values >= 160) & (h_values <= 179))
            ) & (
                (s_values >= 20) & (s_values <= 255)
            ) & (
                (v_values >= 20) & (v_values <= 255)
            )
            
            return skin_mask
            
        except Exception as e:
            print(f"Error in skin detection: {e}")
            return np.zeros(len(hsv_pixels), dtype=bool)
    
    def _calculate_color_percentages(self, hsv_pixels):
        """
        Calculate percentage of each color in the clothing pixels
        Args:
            hsv_pixels: Array of HSV pixel values (clothing only, no skin)
        Returns:
            Dictionary with color percentages
        """
        try:
            h_values = hsv_pixels[:, 0]  # Hue
            s_values = hsv_pixels[:, 1]  # Saturation
            v_values = hsv_pixels[:, 2]  # Value
            
            total_pixels = len(hsv_pixels)
            color_counts = {
                'white': 0, 'black': 0, 'red': 0, 'brown': 0, 'pink': 0,
                'blue': 0, 'green': 0, 'yellow': 0, 'orange': 0, 'purple': 0, 'gray': 0
            }
            
            for i in range(total_pixels):
                h, s, v = h_values[i], s_values[i], v_values[i]
                
                # Classify each pixel
                color = self._classify_single_pixel_hsv(h, s, v)
                if color in color_counts:
                    color_counts[color] += 1
            
            # Convert counts to percentages
            color_percentages = {}
            for color, count in color_counts.items():
                color_percentages[color] = count / total_pixels if total_pixels > 0 else 0
            
            return color_percentages
            
        except Exception as e:
            print(f"Error in color percentage calculation: {e}")
            return {'unknown': 1.0}
    
    def _classify_single_pixel_hsv(self, h, s, v):
        """
        Classify a single pixel based on HSV values
        Args:
            h, s, v: Hue, Saturation, Value of a single pixel
        Returns:
            Color category string
        """
        # Black: low brightness, low saturation
        if v < 50 and s < 30:
            return 'black'
        
        # White: high brightness, low saturation
        elif v > 200 and s < 30:
            return 'white'
        
        # Gray: medium brightness, low saturation
        elif 50 <= v <= 200 and s < 30:
            return 'gray'
        
        # Red: hue around 0 or 179 (red range)
        elif (0 <= h <= 10 or 170 <= h <= 179) and s > 50:
            return 'red'
        
        # Pink: red hue with high brightness and medium saturation
        elif (0 <= h <= 10 or 170 <= h <= 179) and s > 30 and v > 150:
            return 'pink'
        
        # Brown: red-orange hue with low brightness and medium saturation
        elif (10 <= h <= 25) and s > 40 and v < 120:
            return 'brown'
        
        # Orange: hue around 15-25
        elif 15 <= h <= 25 and s > 50:
            return 'orange'
        
        # Yellow: hue around 25-35
        elif 25 <= h <= 35 and s > 50:
            return 'yellow'
        
        # Green: hue around 35-85
        elif 35 <= h <= 85 and s > 50:
            return 'green'
        
        # Blue: hue around 85-130
        elif 85 <= h <= 130 and s > 50:
            return 'blue'
        
        # Purple: hue around 130-170
        elif 130 <= h <= 170 and s > 50:
            return 'purple'
        
        # Default to unknown
        return 'unknown'
    
    def infer(self, frame, orig_shape):
        """Run complete inference pipeline: preprocess -> model.run -> postprocess
        
        Args:
            frame: Input frame for inference
            orig_shape: Original image shape (height, width, channels) or (height, width)
        """
        # Preprocess
        batch = self.preprocess(frame)
        
        # Run inference
        preds = self.module.run(batch)
        
        # Postprocess
        results = self.postprocess(preds, batch, orig_shape)
        
        return results