import cv2
import numpy as np


class Renderer:
    """
    Renderer class for annotating and displaying detection results
    """
    
    def __init__(self, show_segmentation=False):
        """
        Initialize renderer
        """
        self.show_segmentation = show_segmentation
    
    def annotate(self, frame, detections, tracked_objects):
        """
        Annotate frame with bounding boxes, track IDs, and segmentation masks
        """
        annotated_frame = frame.copy()
        original_height, original_width = frame.shape[:2]
        
        # Draw bounding boxes and track IDs (only for Man and Woman)
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, gender = obj
            
            # Only draw if gender is determined (Man or Woman)
            if gender == 1 or gender == 'Man':
                color = [255, 0, 0]  # Blue for man
                gender_label = "Man"
            elif gender == 0 or gender == 'Woman':
                color = [203, 192, 255]  # Pink for woman
                gender_label = "Woman"
            else:
                continue  # Skip unknown or undetermined gender
            
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
            label = f"ID: {track_id} ({gender_label})"
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw segmentation masks from original detections (if enabled)
        if self.show_segmentation:
            for result in detections:
                if result['boxes'] is not None and result['masks'] is not None:
                    # Convert torch tensors to numpy if needed
                    if hasattr(result['boxes'], 'cpu'):
                        boxes = result['boxes'].cpu().numpy()
                        confidences = result['conf'].cpu().numpy()
                        class_ids = result['cls'].cpu().numpy()
                        masks = result['masks'].cpu().numpy()
                    else:
                        boxes = result['boxes']
                        confidences = result['conf']
                        class_ids = result['cls']
                        masks = result['masks']
                    
                    for box, conf, class_id, mask in zip(boxes, confidences, class_ids, masks):
                        if int(class_id) == 0:  # Person class
                            # Proper reverse sizing from 640x640 to original frame size
                            mask_height, mask_width = mask.shape
                            scale = min(640 / original_width, 640 / original_height)
                            new_width = int(original_width * scale)
                            new_height = int(original_height * scale)
                            pad_x = (640 - new_width) // 2
                            pad_y = (640 - new_height) // 2
                            
                            if pad_x > 0 or pad_y > 0:
                                content_mask = mask[pad_y:pad_y + new_height, pad_x:pad_x + new_width]
                            else:
                                content_mask = mask
                            
                            if content_mask.size > 0:
                                mask_resized = cv2.resize(content_mask, (original_width, original_height))
                            else:
                                mask_resized = np.zeros((original_height, original_width), dtype=mask.dtype)
                            
                            mask_binary = (mask_resized > 0.5).astype(np.uint8)
                            color_mask = np.zeros_like(annotated_frame)
                            
                            # Use gender-based color for segmentation mask
                            # Find corresponding tracked object for this detection
                            mask_color = [0, 255, 0]  # Default green
                            for obj in tracked_objects:
                                obj_x1, obj_y1, obj_x2, obj_y2 = obj[:4]
                                # Check if this detection matches a tracked object
                                if (abs(box[0] - obj_x1) < 10 and abs(box[1] - obj_y1) < 10 and 
                                    abs(box[2] - obj_x2) < 10 and abs(box[3] - obj_y2) < 10):
                                    gender = obj[5] if len(obj) > 5 else 'Unknown'
                                    if gender == 1:
                                        mask_color = [255, 0, 0]  # Blue for man
                                    elif gender == 0:
                                        mask_color = [203, 192, 255]  # Pink for woman
                                    break
                            
                            # Create a more subtle overlay that doesn't darken the video
                            color_mask[mask_binary == 1] = mask_color
                            
                            # Use a lighter overlay approach - only add color where mask exists
                            mask_indices = mask_binary == 1
                            if np.any(mask_indices):
                                # Blend the color mask more subtly
                                annotated_frame[mask_indices] = cv2.addWeighted(
                                    annotated_frame[mask_indices], 0.7, 
                                    color_mask[mask_indices], 0.3, 0
                                )
        
        return annotated_frame
    
    def display(self, frame, window_name="Gender Detection"):
        """
        Display frame in a window
        """
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1) & 0xFF
