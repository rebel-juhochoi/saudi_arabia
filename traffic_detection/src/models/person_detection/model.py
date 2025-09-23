import cv2
import numpy as np
from pathlib import Path
import rebel
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression


class VehicleDetector:
    """
    Vehicle detector with tracking capabilities
    """
    def __init__(self, conf=0.25):
        """
        Initialize detector 
        """
        self.model_name = "yolo11n-seg"
        self.conf = conf
        # Vehicle classes: bicycle=1, car=2, motorcycle=3, bus=5, truck=7
        self.vehicle_classes = [1, 2, 3, 5, 7]
        
        # Load YOLO model for configuration
        model_root = Path(__file__).parent
        pt_path = model_root / f"{self.model_name}.pt"
        self.cfg = self._load_model_config(pt_path, conf)
        
        # Load compiled model           
        compiled_model_path = model_root / f"{self.model_name}.rbln"
        self.module = rebel.Runtime(str(compiled_model_path))
        
        print(f"VehicleDetector initialized: {self.model_name} - detecting classes {self.vehicle_classes}")
    
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
            classes=model.overrides.get("classes", self.vehicle_classes),
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
    
    def get_vehicle_type(self, class_id):
        """
        Get vehicle type name from class ID
        Args:
            class_id: COCO class ID
        Returns:
            Vehicle type name
        """
        vehicle_names = {
            1: 'bicycle',
            2: 'car', 
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        return vehicle_names.get(class_id, 'unknown')
    
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