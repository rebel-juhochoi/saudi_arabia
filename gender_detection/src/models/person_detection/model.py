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
        Detect the most dominant color (black or white) in the segmentation area
        Args:
            frame: Original frame
            bbox: Bounding box [x1, y1, x2, y2]
            mask: Segmentation mask
        Returns:
            'black' or 'white' based on dominant color
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
                return 'unknown'
            
            # Resize mask to match crop size
            mask_resized = cv2.resize(mask, (person_crop.shape[1], person_crop.shape[0]))
            
            # Ensure mask is binary
            if mask_resized.dtype != np.uint8:
                mask_resized = (mask_resized * 255).astype(np.uint8)
            
            mask_binary = (mask_resized > 0.5).astype(np.uint8)
            
            # Apply mask to get only the segmented person area
            person_masked = cv2.bitwise_and(person_crop, person_crop, mask=mask_binary)
            
            # Convert to grayscale for color analysis
            gray = cv2.cvtColor(person_masked, cv2.COLOR_BGR2GRAY)
            
            # Get only the masked pixels
            masked_pixels = gray[mask_binary == 1]
            
            if len(masked_pixels) == 0:
                return 'unknown'
            
            # Calculate mean brightness
            mean_brightness = np.mean(masked_pixels)
            
            # Threshold to determine if it's more black or white
            # Values closer to 0 are black, closer to 255 are white
            threshold = 127  # Middle point
            
            if mean_brightness < threshold:
                return 'black'
            else:
                return 'white'
                
        except Exception as e:
            print(f"Error in color detection: {e}")
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