import numpy as np
from models import PersonDetector, GenderClassifier


class Tracker:
    """
    Tracker that handles person detection, gender classification, tracking, and visualization
    """
    def __init__(self, person_detector, gender_classifier, iou_threshold=0.5, enable_color_heuristic=False):
        """
        Initialize tracker with provided models
        """
        self.person_detector = person_detector
        self.gender_classifier = gender_classifier
        self.iou_threshold = iou_threshold
        self.enable_color_heuristic = enable_color_heuristic
        
        # Track data structure: {track_id: {'bbox': [x1,y1,x2,y2], 'conf': conf, 'age': age, 'last_seen': frame, 'gender_counts': {'Man': 0, 'Woman': 0}, 'gender': 'Unknown', 'color_percentages': {'black': 0.0, 'white': 0.0, 'red': 0.0, 'brown': 0.0, 'pink': 0.0, 'blue': 0.0, 'green': 0.0, 'yellow': 0.0, 'orange': 0.0, 'purple': 0.0, 'gray': 0.0}, 'color_counts': {'black': 0, 'white': 0}, 'dominant_color': 'Unknown', 'top_colors': []}}
        # Color heuristic: Priority 1 = black OR pink in top 2 → Woman, Priority 2 = white OR red in top 2 → Man, else → Woman
        self.tracks = {}
        self.next_track_id = 1
        self.track_history = {}
        self.frame_count = 0  # Counter for merge interval
    
    def _update_top_colors(self, track_id):
        """
        Update the top 2 colors for a track based on color percentages
        """
        percentages = self.tracks[track_id]['color_percentages']
        
        # Sort colors by percentage (descending)
        sorted_colors = sorted(percentages.items(), key=lambda x: x[1], reverse=True)
        
        # Get top 2 colors
        top_colors = [color for color, percentage in sorted_colors[:2] if percentage > 0]
        
        self.tracks[track_id]['top_colors'] = top_colors
    
    def detect_and_track(self, frame, orig_shape):
        """
        Perform detection and tracking on a single frame
        """
        # Increment frame counter
        self.frame_count += 1
        
        # Step 1: Detection with compiled model
        detections = self.person_detector.infer(frame, orig_shape)
        
        # Step 2: Update tracker with detections
        tracked_objects = self._update_tracker(detections, frame)
        
        # Step 3: Merge overlapping objects every 10 frames
        if self.frame_count % 10 == 0:
            self.merge_overlapping_objects()
        
        # Step 4: Clean up old tracks periodically to prevent memory buildup
        if self.frame_count % 30 == 0:  # Every 30 frames (less frequent)
            self._cleanup_old_tracks()
        
        return detections, tracked_objects
    
    def _cleanup_old_tracks(self):
        """Clean up old tracks to prevent memory buildup"""
        tracks_to_remove = []
        
        for track_id, track_data in self.tracks.items():
            # Remove tracks that are very old or have been inactive
            if track_data['age'] > 15:  # Less aggressive cleanup
                tracks_to_remove.append(track_id)
            elif track_data['age'] > 5 and track_data.get('last_seen', 0) > 10:
                # Remove tracks that are old and haven't been seen recently
                tracks_to_remove.append(track_id)
        
        # Remove old tracks
        for track_id in tracks_to_remove:
            if track_id in self.tracks:
                del self.tracks[track_id]
        
        if tracks_to_remove:
            print(f"Cleaned up {len(tracks_to_remove)} old tracks. Remaining: {len(self.tracks)}")
    
    def reset_tracker(self):
        """Complete tracker reset - only called during loop restart or video switch"""
        print(f"Resetting tracker completely ({len(self.tracks)} tracks)...")
        self.tracks.clear()
        self.track_history.clear()
        self.next_track_id = 1
        self.frame_count = 0
        print("Tracker reset complete")
    
    def _update_tracker(self, detections, frame):
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
            
            # Update color information if available
            if len(detection) > 7:  # If color percentages are available
                color_percentages = detection[7]
                if isinstance(color_percentages, dict):
                    # Update color percentages (average with previous values)
                    for color, percentage in color_percentages.items():
                        if color in self.tracks[track_id]['color_percentages']:
                            # Simple moving average (could be improved with weighted average)
                            current = self.tracks[track_id]['color_percentages'][color]
                            self.tracks[track_id]['color_percentages'][color] = (current + percentage) / 2
                    
                    # Update top colors ranking
                    self._update_top_colors(track_id)
            
            # Perform gender classification for matched tracks
            self._classify_gender_for_track(track_id, frame)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = {
                'bbox': detection[:4],
                'conf': detection[4],
                'age': 0,
                'last_seen': 0,
                'gender_counts': {'Man': 0, 'Woman': 0},
                'gender': 'Unknown',
                'color_percentages': {'black': 0.0, 'white': 0.0, 'red': 0.0, 'brown': 0.0, 'pink': 0.0, 'blue': 0.0, 'green': 0.0, 'yellow': 0.0, 'orange': 0.0, 'purple': 0.0, 'gray': 0.0},
                'color_counts': {'black': 0, 'white': 0},
                'dominant_color': 'Unknown',
                'top_colors': []
            }
            if len(detection) > 6:  # If mask is available
                self.tracks[track_id]['mask'] = detection[6]
            
            # Update color information if available
            if len(detection) > 7:  # If color percentages are available
                color_percentages = detection[7]
                if isinstance(color_percentages, dict):
                    # Update color percentages (average with previous values)
                    for color, percentage in color_percentages.items():
                        if color in self.tracks[track_id]['color_percentages']:
                            # Simple moving average (could be improved with weighted average)
                            current = self.tracks[track_id]['color_percentages'][color]
                            self.tracks[track_id]['color_percentages'][color] = (current + percentage) / 2
                    
                    # Update top colors ranking
                    self._update_top_colors(track_id)
            
            # Perform gender classification for new tracks
            self._classify_gender_for_track(track_id, frame)
        
        # Age unmatched tracks
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > 10:  # Remove old tracks
                    del self.tracks[track_id]
        
        # Update track history
        self._update_track_history()
        
        # Return tracked objects with gender information (only Man and Woman)
        tracked_objects = []
        for track_id, track in self.tracks.items():
            if track['age'] < 5:  # Only return recent tracks
                bbox = track['bbox']
                gender = track.get('gender', 'Unknown')
                # Only include objects with determined gender (Man or Woman)
                if gender in [0, 1, 'Man', 'Woman']:  # 0=Woman, 1=Man
                    tracked_objects.append([bbox[0], bbox[1], bbox[2], bbox[3], track_id, gender])
        
        return tracked_objects
    
    def _format_detections(self, detections, frame=None):
        """
        Format detections from PersonDetector to tracker format
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
                    if int(cls_id) == 0 and conf >= self.person_detector.conf:  # Person class
                        detection = [box[0], box[1], box[2], box[3], float(conf), int(cls_id), mask]
                        
                        # Add color detection if enabled
                        if self.enable_color_heuristic and frame is not None:
                            color_percentages = self.person_detector.detect_color(frame, box, mask)
                            detection.append(color_percentages)
                        else:
                            detection.append({'unknown': 1.0})
                        
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
    
    def _classify_gender_for_track(self, track_id, frame):
        """
        Classify gender for a specific track and update gender counts
        Returns the predicted gender if confidence is high enough, None otherwise
        """
        if track_id not in self.tracks:
            return None
        
        try:
            # Extract person from bounding box and segmentation mask
            bbox = self.tracks[track_id]['bbox']
            mask = self.tracks[track_id].get('mask', None)
            person_crop = self.gender_classifier.extract_person_from_bbox(frame, bbox, mask)
            
            if person_crop is not None:
                # Classify gender (deepface model handles face detection internally)
                gender, confidence = self.gender_classifier.classify_gender(person_crop)
                
                # Check if confidence meets the threshold
                if confidence >= self.gender_classifier.conf_threshold:
                    if self.enable_color_heuristic:
                        # Color heuristic mode: top 2 colors analysis
                        # Safety check to ensure track still exists and has required keys
                        if track_id not in self.tracks:
                            return None
                        top_colors = self.tracks[track_id].get('top_colors', [])
                        
                        # Priority check: If black OR pink is in top 2, classify as woman (overrides other logic)
                        if 'black' in top_colors or 'pink' in top_colors:
                            # Black or pink in top 2: increment woman count (overrides white/red logic)
                            if 'gender_counts' in self.tracks[track_id]:
                                self.tracks[track_id]['gender_counts']['Woman'] += 1
                            predicted_gender = 'Woman'
                        # Secondary check: Male condition: white OR red (or both) in top 2 colors
                        elif 'white' in top_colors or 'red' in top_colors:
                            # Either white or red (or both) in top 2: increment man count
                            if 'gender_counts' in self.tracks[track_id]:
                                self.tracks[track_id]['gender_counts']['Man'] += 1
                            predicted_gender = 'Man'
                        else:
                            # Neither black, white, nor red in top 2: default to female
                            if 'gender_counts' in self.tracks[track_id]:
                                self.tracks[track_id]['gender_counts']['Woman'] += 1
                            predicted_gender = 'Woman'
                    else:
                        # Normal mode: use deepface only
                        if 'gender_counts' in self.tracks[track_id] and gender in self.tracks[track_id]['gender_counts']:
                            self.tracks[track_id]['gender_counts'][gender] += 1
                            predicted_gender = gender
                        else:
                            predicted_gender = None
                    
                    # Update the most likely gender based on counts
                    if 'gender_counts' in self.tracks[track_id]:
                        counts = self.tracks[track_id]['gender_counts']
                        if counts['Man'] > counts['Woman']:
                            self.tracks[track_id]['gender'] = 'Man'
                        elif counts['Woman'] > counts['Man']:
                            self.tracks[track_id]['gender'] = 'Woman'
                        # Keep 'Unknown' if counts are equal
                    
                    return predicted_gender
                    
        except Exception as e:
            # Silently handle errors to avoid disrupting tracking
            pass
        
        return None
    
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
        
        # Merge gender counts
        for gender in ['Man', 'Woman']:
            keep_track['gender_counts'][gender] += merge_track['gender_counts'][gender]
        
        # Merge color counts (with safety check)
        if 'color_counts' not in keep_track:
            keep_track['color_counts'] = {'black': 0, 'white': 0}
        if 'color_counts' not in merge_track:
            merge_track['color_counts'] = {'black': 0, 'white': 0}
        
        for color in ['black', 'white']:
            keep_track['color_counts'][color] += merge_track['color_counts'][color]
        
        # Update dominant color based on merged counts
        counts = keep_track['color_counts']
        if counts['black'] > counts['white']:
            keep_track['dominant_color'] = 'black'
        elif counts['white'] > counts['black']:
            keep_track['dominant_color'] = 'white'
        else:
            keep_track['dominant_color'] = 'Unknown'
        
        # Update gender based on merged counts
        gender_counts = keep_track['gender_counts']
        if gender_counts['Man'] > gender_counts['Woman']:
            keep_track['gender'] = 'Man'
        elif gender_counts['Woman'] > gender_counts['Man']:
            keep_track['gender'] = 'Woman'
        else:
            keep_track['gender'] = 'Unknown'
        
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
    
