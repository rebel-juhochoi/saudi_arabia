import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import json
import time
from pathlib import Path


class TrafficCounter:
    """
    Traffic counter for cumulative vehicle counting in road areas
    """
    
    def __init__(self, counting_line_position=0.85, min_track_history=5):
        """
        Initialize traffic counter
        Args:
            counting_line_position: Y-position of the counting line as a ratio of frame height (0.0 to 1.0)
            min_track_history: Minimum number of frames a track must exist before being considered for counting
        """
        self.counting_line_position = counting_line_position
        self.min_track_history = min_track_history
        self.road_counts: Dict[int, Dict] = {}
        self.counting_lines: Dict[int, Tuple[int, int, int, int]] = {}
        self.track_histories: Dict[int, Dict] = {} # Stores positions and status for each track_id
        self.frame_count = 0
        self.start_time = time.time()
        self.total_vehicles_seen = 0

        # Direction detection
        self.direction_sensitivity = 10  # pixels minimum movement to determine direction

        # Simple bottom line counting
        self.frame_height = None
        self.frame_width = None
        
    def initialize_road_areas(self, road_masks: List[np.ndarray]):
        """
        Initialize counting lines and counters - using simple bottom line approach
        
        Args:
            road_masks: List of binary masks for each road area (ignored, using simple approach)
        """
        self.road_counts = {}
        self.counting_lines = {}
        
        # Use single counting line at bottom of frame for all vehicles
        self.road_counts[0] = {
            'total': 0,
            'by_type': defaultdict(int),
            'directions': {'up': 0, 'down': 0, 'left': 0, 'right': 0},
            'hourly_counts': defaultdict(int)  # For time-based analytics
        }
        
        # Simple horizontal line at bottom (will be set when first frame is processed)
        self.counting_lines[0] = None  # Will be set in update_tracking
        self.frame_height = None
        self.frame_width = None
        
        print(f"Initialized simple traffic counting with bottom line approach")
    
    def _create_counting_line(self, road_mask: np.ndarray, road_id: int) -> Tuple[int, int, int, int]:
        """
        Create a counting line across the road area
        
        Args:
            road_mask: Binary mask of the road area
            road_id: ID of the road area
            
        Returns:
            Tuple of (x1, y1, x2, y2) for the counting line
        """
        # Find road boundaries
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0, 0, road_mask.shape[1], road_mask.shape[0])
        
        # Get bounding box of the road area
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # Create horizontal counting line at specified position
        line_y = int(y + h * self.counting_line_position)
        
        # Extend line across the width of the road area
        x1, x2 = x, x + w
        y1 = y2 = line_y
        
        print(f"Road {road_id}: Counting line at y={line_y} from x={x1} to x={x2}")
        return (x1, y1, x2, y2)
    
    def update_tracking(self, tracked_objects: List, frame_shape: Tuple[int, int]):
        """
        Update traffic counting based on tracked objects using simple bottom line approach
        
        Args:
            tracked_objects: List of tracked objects [x1, y1, x2, y2, track_id, vehicle_type]
            frame_shape: Shape of the current frame
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Set frame dimensions and counting line on first call
        if self.frame_height is None:
            self.frame_height, self.frame_width = frame_shape
            # Create horizontal counting line at 90% of frame height (near bottom)
            # Start line at 1/5 from left edge to avoid parked cars
            line_y = int(self.frame_height * 0.90)
            line_start_x = int(self.frame_width * 0.2)  # Start at 1/5 from left
            self.counting_lines[0] = (line_start_x, line_y, self.frame_width, line_y)
            print(f"Created counting line at y={line_y} from x={line_start_x} to x={self.frame_width} (90% height, avoiding left 20%)")
        
        # Update existing tracks and add new ones
        current_track_ids = set()
        
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id, vehicle_type = obj
            current_track_ids.add(track_id)
            
            # Calculate right bottom corner (bottom-right of bounding box)
            corner_x = x2  # Right edge
            corner_y = y2  # Bottom edge
            
            # Initialize track if new
            if track_id not in self.track_histories:
                self.track_histories[track_id] = {
                    'positions': deque(maxlen=10),  # Keep last 10 positions
                    'counted': False,
                    'vehicle_type': vehicle_type,
                    'first_seen': current_time,
                    'last_seen': current_time
                }
            
            # Update track
            track = self.track_histories[track_id]
            track['positions'].append((corner_x, corner_y, current_time))
            track['last_seen'] = current_time
            track['vehicle_type'] = vehicle_type  # Update in case it changed
            
            # Check if vehicle should be counted (crossed bottom line)
            if not track['counted'] and len(track['positions']) >= 2:
                if self._should_count_vehicle_simple(track_id):
                    self._count_vehicle_simple(track_id, vehicle_type)
        
        # Clean up old tracks
        self._cleanup_old_tracks(current_track_ids, current_time)
    
    def _get_road_id_for_position(self, x: float, y: float) -> Optional[int]:
        """
        Determine which road area a position belongs to
        
        Args:
            x, y: Position coordinates
            
        Returns:
            Road ID or None if not in any road area
        """
        # With simple counting, all vehicles are considered part of road 0
        return 0
    
    def _should_count_vehicle(self, track_id: int, road_id: int) -> bool:
        """
        Determine if a vehicle should be counted based on crossing counting line
        
        Args:
            track_id: ID of the track
            road_id: ID of the road area
            
        Returns:
            True if vehicle should be counted
        """
        track = self.track_histories[track_id]
        positions = list(track['positions'])
        
        if len(positions) < 2:
            return False
        
        # Get counting line for this road
        if road_id not in self.counting_lines:
            return False
        
        x1, y1, x2, y2 = self.counting_lines[road_id]
        line_y = y1  # Horizontal line
        
        # Check if vehicle crossed the counting line
        crossed_line = False
        
        # Look at recent positions to see if line was crossed
        for i in range(1, len(positions)):
            prev_y = positions[i-1][1]
            curr_y = positions[i][1]
            
            # Check if vehicle crossed the horizontal line
            if (prev_y <= line_y <= curr_y) or (curr_y <= line_y <= prev_y):
                crossed_line = True
                break
        
        # Additional validation: ensure vehicle spent enough time in the area
        if crossed_line:
            track_duration = positions[-1][2] - positions[0][2]
            if track_duration >= 0.5:  # At least 0.5 seconds in area
                return True
        
        return False
    
    def _should_count_vehicle_simple(self, track_id: int) -> bool:
        """
        Simple counting logic: check if vehicle touches the bottom line
        
        Args:
            track_id: ID of the track
            
        Returns:
            True if vehicle should be counted
        """
        track = self.track_histories[track_id]
        positions = list(track['positions'])
        
        if len(positions) < 1:
            return False
        
        # Get counting line coordinates
        if 0 not in self.counting_lines or self.counting_lines[0] is None:
            return False
        
        line_start_x, line_y, line_end_x, _ = self.counting_lines[0]
        
        # Check if vehicle's right bottom corner is within counting line area
        current_x, current_y = positions[-1][:2]
        
        # Vehicle counted if: corner is at/past line Y AND within line X range
        if current_y >= line_y and line_start_x <= current_x <= line_end_x:
            return True
        
        return False
    
    def _count_vehicle_simple(self, track_id: int, vehicle_type: str):
        """
        Count a vehicle using simple approach (single road area)
        
        Args:
            track_id: ID of the track
            vehicle_type: Type of vehicle
        """
        # Mark as counted
        self.track_histories[track_id]['counted'] = True
        
        # Update counters (using road_id = 0 for all vehicles)
        road_id = 0
        self.road_counts[road_id]['total'] += 1
        self.road_counts[road_id]['by_type'][vehicle_type] += 1
        self.total_vehicles_seen += 1
        
        print(f"🚗 Counted {vehicle_type} (ID: {track_id}) - Total: {self.total_vehicles_seen}")
        
        # Determine direction (simplified)
        direction = self._get_vehicle_direction_simple(track_id)
        if direction:
            self.road_counts[road_id]['directions'][direction] += 1
        
        # Update hourly counts
        current_hour = int(time.time() // 3600)
        self.road_counts[road_id]['hourly_counts'][current_hour] += 1
        
        print(f"🚗 Counted {vehicle_type} (ID: {track_id}) - Total: {self.road_counts[road_id]['total']}")
    
    def _get_vehicle_direction_simple(self, track_id: int) -> Optional[str]:
        """
        Simple direction detection - since we're counting vehicles going down, return 'down'
        
        Args:
            track_id: ID of the track
            
        Returns:
            Direction string
        """
        return 'down'  # All counted vehicles are moving down towards bottom line
    
    def _count_vehicle(self, track_id: int, road_id: int, vehicle_type: str):
        """
        Count a vehicle and update statistics
        
        Args:
            track_id: ID of the track
            road_id: ID of the road area  
            vehicle_type: Type of vehicle
        """
        # Mark as counted
        self.track_histories[track_id]['counted'] = True
        
        # Update counters
        self.road_counts[road_id]['total'] += 1
        self.road_counts[road_id]['by_type'][vehicle_type] += 1
        self.total_vehicles_seen += 1
        
        # Determine direction
        direction = self._get_vehicle_direction(track_id)
        if direction:
            self.road_counts[road_id]['directions'][direction] += 1
        
        # Update hourly counts
        current_hour = int(time.time() // 3600)
        self.road_counts[road_id]['hourly_counts'][current_hour] += 1
        
        print(f"🚗 Counted {vehicle_type} (ID: {track_id}) in road {road_id} - Total: {self.road_counts[road_id]['total']}")
    
    def _get_vehicle_direction(self, track_id: int) -> Optional[str]:
        """
        Determine the direction of vehicle movement
        
        Args:
            track_id: ID of the track
            
        Returns:
            Direction string ('up', 'down', 'left', 'right') or None
        """
        if track_id not in self.track_histories:
            return None
        
        positions = list(self.track_histories[track_id]['positions'])
        if len(positions) < 3:
            return None
        
        # Calculate average movement direction
        dx_total = 0
        dy_total = 0
        
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dx_total += dx
            dy_total += dy
        
        # Determine primary direction
        if abs(dx_total) > abs(dy_total):
            return 'right' if dx_total > self.direction_sensitivity else 'left'
        else:
            return 'down' if dy_total > self.direction_sensitivity else 'up'
    
    def _cleanup_old_tracks(self, current_track_ids: set, current_time: float):
        """
        Clean up tracks that are no longer active
        
        Args:
            current_track_ids: Set of currently active track IDs
            current_time: Current timestamp
        """
        tracks_to_remove = []
        
        for track_id, track in self.track_histories.items():
            # Remove if track hasn't been seen for 5 seconds
            if track_id not in current_track_ids:
                if current_time - track['last_seen'] > 5.0:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_histories[track_id]
    
    def get_traffic_counts(self) -> Dict:
        """
        Get current traffic counts for all roads
        
        Returns:
            Dictionary with traffic count data
        """
        result = {
            'road_counts': dict(self.road_counts),
            'total_vehicles': self.total_vehicles_seen,
            'session_duration': time.time() - self.start_time,
            'frame_count': self.frame_count,
            'active_tracks': len(self.track_histories)
        }
        
        return result
    
    def get_road_summary(self, road_id: int) -> Dict:
        """
        Get detailed summary for a specific road
        
        Args:
            road_id: ID of the road area
            
        Returns:
            Dictionary with road-specific data
        """
        if road_id not in self.road_counts:
            return {}
        
        road_data = self.road_counts[road_id]
        session_duration = time.time() - self.start_time
        
        return {
            'road_id': road_id,
            'total_vehicles': road_data['total'],
            'vehicles_by_type': dict(road_data['by_type']),
            'directions': dict(road_data['directions']),
            'vehicles_per_minute': road_data['total'] / max(session_duration / 60, 0.1),
            'most_common_vehicle': max(road_data['by_type'].items(), key=lambda x: x[1])[0] if road_data['by_type'] else 'none'
        }
    
    def get_analytics_data(self) -> Dict:
        """
        Get comprehensive analytics data
        
        Returns:
            Dictionary with full analytics
        """
        session_duration = time.time() - self.start_time
        
        # Calculate overall statistics
        total_by_type = defaultdict(int)
        total_by_direction = defaultdict(int)
        
        for road_data in self.road_counts.values():
            for vehicle_type, count in road_data['by_type'].items():
                total_by_type[vehicle_type] += count
            for direction, count in road_data['directions'].items():
                total_by_direction[direction] += count
        
        return {
            'session': {
                'duration_seconds': session_duration,
                'duration_minutes': session_duration / 60,
                'start_time': self.start_time,
                'frames_processed': self.frame_count
            },
            'totals': {
                'vehicles': self.total_vehicles_seen,
                'by_type': dict(total_by_type),
                'by_direction': dict(total_by_direction),
                'vehicles_per_minute': self.total_vehicles_seen / max(session_duration / 60, 0.1)
            },
            'roads': {road_id: self.get_road_summary(road_id) for road_id in self.road_counts.keys()}
        }
    
    def save_analytics(self, filepath: str):
        """
        Save analytics data to file
        
        Args:
            filepath: Path to save analytics JSON
        """
        analytics = self.get_analytics_data()
        
        with open(filepath, 'w') as f:
            json.dump(analytics, f, indent=2)
        
        print(f"Analytics saved to {filepath}")
    
    def reset_counters(self):
        """
        Reset all traffic counters
        """
        self.road_counts = {}
        self.counting_lines = {}
        self.track_histories = {}
        self.frame_count = 0
        self.start_time = time.time()
        self.total_vehicles_seen = 0
        self.frame_height = None
        self.frame_width = None
        print("Traffic counters reset.")
    
    def get_counting_lines_for_visualization(self) -> Dict:
        """
        Get counting line coordinates for visualization
        
        Returns:
            Dictionary mapping road_id to line coordinates
        """
        # Return empty dict if frame dimensions haven't been set yet
        if self.frame_height is None:
            return {}
        return dict(self.counting_lines)
