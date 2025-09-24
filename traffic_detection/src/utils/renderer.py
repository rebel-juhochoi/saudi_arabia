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
    
    def annotate(self, frame, detections, tracked_objects, traffic_counts=None, counting_lines=None):
        """
        Annotate frame with bounding boxes, track IDs, and segmentation masks
        """
        annotated_frame = frame.copy()
        original_height, original_width = frame.shape[:2]
        
        # Draw bounding boxes and track IDs for vehicles
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, vehicle_type = obj
            
            # Unified white color for all vehicle types
            color = [255, 255, 255]  # White
            
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID: {track_id}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Note: Segmentation display removed as requested
        # Segmentation masks are still used internally for IoU calculations
        
        # Draw traffic counts
        if traffic_counts:
            annotated_frame = self._draw_traffic_counts(annotated_frame, traffic_counts)
        
        # Counting lines are now invisible (not drawn)
        
        return annotated_frame
    
    def _draw_traffic_counts(self, frame, traffic_counts):
        """
        Draw traffic count information on the frame
        
        Args:
            frame: Input frame
            traffic_counts: Traffic count data dictionary
            
        Returns:
            Frame with traffic counts overlaid
        """
        if not traffic_counts or 'road_counts' not in traffic_counts:
            return frame
        
        # Draw background for count display
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Calculate panel size based on number of roads
        num_roads = len(traffic_counts['road_counts'])
        panel_height = max(120, 40 + num_roads * 60)
        panel_width = 280
        
        # Draw semi-transparent background panel
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw title
        cv2.putText(frame, "TRAFFIC COUNTS", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw total count
        total_vehicles = traffic_counts.get('total_vehicles', 0)
        cv2.putText(frame, f"Total: {total_vehicles}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw counts per road
        y_offset = 85
        for road_id, road_data in traffic_counts['road_counts'].items():
            total = road_data.get('total', 0)
            
            # Road title
            cv2.putText(frame, f"Road {road_id}: {total}", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Vehicle types breakdown
            by_type = road_data.get('by_type', {})
            type_text = ", ".join([f"{vtype}: {count}" for vtype, count in by_type.items() if count > 0])
            if type_text:
                cv2.putText(frame, type_text, (30, y_offset + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            y_offset += 45
        
        # Draw session info
        session_duration = traffic_counts.get('session_duration', 0)
        vehicles_per_min = total_vehicles / max(session_duration / 60, 0.1)
        cv2.putText(frame, f"Rate: {vehicles_per_min:.1f}/min", (20, panel_height - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return frame
    
    def _draw_counting_lines(self, frame, counting_lines):
        """
        Draw counting lines on the frame
        
        Args:
            frame: Input frame
            counting_lines: Dictionary mapping road_id to line coordinates
            
        Returns:
            Frame with counting lines drawn
        """
        colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]  # Yellow, Magenta, Cyan
        
        for road_id, (x1, y1, x2, y2) in counting_lines.items():
            color = colors[road_id % len(colors)]
            
            # Draw counting line
            cv2.line(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw road ID label
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            cv2.putText(frame, f"R{road_id}", (mid_x - 10, mid_y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def display(self, frame, window_name="Vehicle Detection"):
        """
        Display frame in a window
        """
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1) & 0xFF
