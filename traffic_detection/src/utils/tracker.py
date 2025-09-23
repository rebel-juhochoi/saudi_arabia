import numpy as np
from models import VehicleDetector, RoadAreaDetector
from .traffic_counter import TrafficCounter


class Tracker:
    """
    Tracker that handles vehicle detection, tracking, and visualization
    """
    def __init__(self, vehicle_detector, iou_threshold=0.5, road_detector=None, enable_traffic_counting=True):
        """
        Initialize tracker with provided models
        """
        self.vehicle_detector = vehicle_detector
        self.iou_threshold = iou_threshold
        self.road_detector = road_detector
        self.enable_traffic_counting = enable_traffic_counting
        
        # Initialize traffic counter
        self.traffic_counter = TrafficCounter(road_detector) if enable_traffic_counting else None
        
        # Track data structure: {track_id: {'bbox': [x1,y1,x2,y2], 'conf': conf, 'age': age, 'last_seen': frame, 'vehicle_type': 'unknown'}}
        self.tracks = {}
        self.next_track_id = 1
        self.track_history = {}
        self.frame_count = 0  # Counter for merge interval
        self.road_initialized = False
    
    def detect_and_track(self, frame, orig_shape, is_first_frame=False):
        """
        Perform detection and tracking on a single frame
        """
        # Increment frame counter
        self.frame_count += 1
        
        # Step 0: Initialize road areas on first frame
        if is_first_frame and self.road_detector and not self.road_initialized:
            print("Initializing road area detection...")
            self.road_detector.detect_road_areas(frame, save_debug=True)
            self.road_initialized = True
            
            # Initialize traffic counter with detected road areas
            if self.traffic_counter:
                self.traffic_counter.initialize_road_areas(self.road_detector.road_masks)
        
        # Ensure traffic counter is always initialized (fallback)
        if self.traffic_counter and (not hasattr(self.traffic_counter, 'road_counts') or not self.traffic_counter.road_counts):
            print("⚠️  TrafficCounter not initialized, initializing with empty road masks...")
            self.traffic_counter.initialize_road_areas([])
        
        # Step 1: Detection with compiled model
        detections = self.vehicle_detector.infer(frame, orig_shape)
        
        # Step 2: Update tracker with detections
        tracked_objects = self._update_tracker(detections, frame, orig_shape)
        
        # Step 3: Merge overlapping objects every 10 frames
        if self.frame_count % 10 == 0:
            self.merge_overlapping_objects()
        
        return tracked_objects
    
    def _update_tracker(self, detections, frame, orig_shape):
        """
        Update tracker with new detections
        """
        if not detections or len(detections) == 0:
            # Age existing tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > 10:  # Remove old tracks
                    del self.tracks[track_id]
            return []
        
        # Convert detections to format
        current_detections = self._format_detections(detections, frame)
        
        if len(current_detections) == 0:
            return []
        
        # Match detections to existing tracks
        matched_tracks, unmatched_detections = self._match_tracks(current_detections)
        
        # Update matched tracks
        for track_id, detection in matched_tracks.items():
            self.tracks[track_id]['bbox'] = detection[:4]
            self.tracks[track_id]['conf'] = detection[4]
            self.tracks[track_id]['age'] = 0
            self.tracks[track_id]['last_seen'] = 0
            if len(detection) > 6:  # If mask is available
                self.tracks[track_id]['mask'] = detection[6]
            
            # Update vehicle type for new track
            self.tracks[track_id]['vehicle_type'] = self.vehicle_detector.get_vehicle_type(int(detection[5]))
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = {
                'bbox': detection[:4],
                'conf': detection[4],
                'age': 0,
                'last_seen': 0,
                'vehicle_type': self.vehicle_detector.get_vehicle_type(int(detection[5]))
            }
            if len(detection) > 6:  # If mask is available
                self.tracks[track_id]['mask'] = detection[6]
            
            # Update vehicle type for new track
            self.tracks[track_id]['vehicle_type'] = self.vehicle_detector.get_vehicle_type(int(detection[5]))
        
        # Age unmatched tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > 10:  # Remove old tracks
                    del self.tracks[track_id]
        
        # Update track history
        self._update_track_history()
        
        # Return tracked objects with vehicle type
        tracked_objects = []
        for track_id, track in self.tracks.items():
            if track['age'] < 5:  # Only return recent tracks
                bbox = track['bbox']
                vehicle_type = track.get('vehicle_type', 'unknown')
                tracked_objects.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id, vehicle_type])
        
        # Update traffic counter with current tracked objects
        if self.traffic_counter and tracked_objects:
            self.traffic_counter.update_tracking(tracked_objects, orig_shape)
            
            # Filter out counted vehicles from display
            filtered_objects = []
            for obj in tracked_objects:
                track_id = obj[4]
                # Only include vehicles that haven't been counted yet
                if not self.traffic_counter.track_histories.get(track_id, {}).get('counted', False):
                    filtered_objects.append(obj)
            tracked_objects = filtered_objects
        
        return tracked_objects
    
    def _format_detections(self, detections, frame=None):
        """
        Format detections from VehicleDetector to tracker format
        """
        formatted_detections = []
        
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
                
                for box, conf, cls_id, mask in zip(boxes, confidences, class_ids, masks):
                    if int(cls_id) in self.vehicle_detector.vehicle_classes and conf >= self.vehicle_detector.conf:  # Vehicle classes
                        # Check if detection is in road area (if road detection is enabled)
                        if self.road_detector and self.road_initialized:
                            if not self.road_detector.is_in_road_area((box[0], box[1], box[2], box[3])):
                                continue  # Skip detections outside road areas
                        
                        detection = [box[0], box[1], box[2], box[3], float(conf), int(cls_id), mask]
                        formatted_detections.append(detection)
        
        return formatted_detections
    
    def _match_tracks(self, detections):
        """
        Match detections to existing tracks using IoU
        """
        matched_tracks = {}
        unmatched_detections = []
        
        if not self.tracks:
            return matched_tracks, detections
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        track_ids = list(self.tracks.keys())
        
        for i, detection in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track_bbox = self.tracks[track_id]['bbox']
                iou_matrix[i, j] = self._calculate_iou(detection[:4], track_bbox)
        
        # Match based on IoU threshold
        for i, detection in enumerate(detections):
            best_match_idx = np.argmax(iou_matrix[i])
            best_iou = iou_matrix[i, best_match_idx]
            
            if best_iou > self.iou_threshold:
                track_id = track_ids[best_match_idx]
                matched_tracks[track_id] = detection
            else:
                unmatched_detections.append(detection)
        
        return matched_tracks, unmatched_detections
    
    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union (IoU) of two bounding boxes
        """
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    
    def _update_track_history(self):
        """
        Update track history for visualization
        """
        for track_id, track in self.tracks.items():
            if track['age'] < 5:  # Only update recent tracks
                bbox = track['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                
                self.track_history[track_id].append((center_x, center_y))
                
                # Keep only last 30 points
                if len(self.track_history[track_id]) > 30:
                    self.track_history[track_id].pop(0)
    
    def get_track_history(self, track_id):
        """
        Get track history for a specific track ID
        """
        return self.track_history.get(track_id, [])
    
    def merge_overlapping_objects(self):
        """
        Merge overlapping tracked objects with IoU > iou_threshold
        Runs every 10 frames to avoid performance issues
        """
        if len(self.tracks) < 2:
            return
        
        # Get all track IDs and their bounding boxes
        track_ids = list(self.tracks.keys())
        merged_tracks = set()
        
        for i in range(len(track_ids)):
            if track_ids[i] in merged_tracks:
                continue
                
            track1_id = track_ids[i]
            bbox1 = self.tracks[track1_id]['bbox']
            
            for j in range(i + 1, len(track_ids)):
                if track_ids[j] in merged_tracks:
                    continue
                    
                track2_id = track_ids[j]
                bbox2 = self.tracks[track2_id]['bbox']
                
                # Calculate IoU (Intersection over Union)
                iou = self._calculate_iou(bbox1, bbox2)
                
                if iou > self.iou_threshold:  # Use configured IoU threshold
                    # Merge track2 into track1 (keep lower ID as priority)
                    if track1_id < track2_id:
                        self._merge_tracks(track1_id, track2_id)
                        merged_tracks.add(track2_id)
                    else:
                        self._merge_tracks(track2_id, track1_id)
                        merged_tracks.add(track1_id)
                        break  # track1_id is now merged, skip remaining comparisons
    
    
    def _merge_tracks(self, keep_id, merge_id):
        """
        Merge merge_id track into keep_id track
        """
        if keep_id not in self.tracks or merge_id not in self.tracks:
            return
        
        keep_track = self.tracks[keep_id]
        merge_track = self.tracks[merge_id]
        
        # Keep the vehicle type from the more recent track
        if merge_track['age'] < keep_track['age']:  # merge_track is newer
            keep_track['vehicle_type'] = merge_track['vehicle_type']
        
        # Use the more recent bounding box and confidence
        if merge_track['age'] < keep_track['age']:  # merge_track is newer
            keep_track['bbox'] = merge_track['bbox']
            keep_track['conf'] = merge_track['conf']
            keep_track['age'] = merge_track['age']
            keep_track['last_seen'] = merge_track['last_seen']
        
        # Merge track history
        if merge_id in self.track_history:
            if keep_id not in self.track_history:
                self.track_history[keep_id] = []
            self.track_history[keep_id].extend(self.track_history[merge_id])
            del self.track_history[merge_id]
        
        # Remove the merged track
        del self.tracks[merge_id]
    
    def get_traffic_counts(self):
        """
        Get current traffic counts from the traffic counter
        
        Returns:
            Dictionary with traffic count data or None if counting disabled
        """
        if self.traffic_counter:
            return self.traffic_counter.get_traffic_counts()
        return None
    
    def get_road_summary(self, road_id: int):
        """
        Get summary for a specific road
        
        Args:
            road_id: ID of the road area
            
        Returns:
            Dictionary with road-specific data or None if counting disabled
        """
        if self.traffic_counter:
            return self.traffic_counter.get_road_summary(road_id)
        return None
    
    def get_analytics_data(self):
        """
        Get comprehensive analytics data
        
        Returns:
            Dictionary with full analytics or None if counting disabled
        """
        if self.traffic_counter:
            return self.traffic_counter.get_analytics_data()
        return None
    
    def get_counting_lines_for_visualization(self):
        """
        Get counting line coordinates for visualization
        
        Returns:
            Dictionary mapping road_id to line coordinates or empty dict if counting disabled
        """
        if self.traffic_counter:
            return self.traffic_counter.get_counting_lines_for_visualization()
        return {}
    
    def reset_traffic_counters(self):
        """
        Reset all traffic counters
        """
        if self.traffic_counter:
            self.traffic_counter.reset_counters()
    
    def save_traffic_analytics(self, filepath: str):
        """
        Save traffic analytics to file
        
        Args:
            filepath: Path to save analytics JSON
        """
        if self.traffic_counter:
            self.traffic_counter.save_analytics(filepath)
    
