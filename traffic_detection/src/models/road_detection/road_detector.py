import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json


class RoadAreaDetector:
    """
    Road area detector using computer vision techniques to identify road regions
    in the first frame for vehicle detection ROI
    """
    
    def __init__(self, min_road_area=10000, max_road_areas=2):
        """
        Initialize road area detector
        
        Args:
            min_road_area: Minimum area (pixels) for a valid road region
            max_road_areas: Maximum number of road areas to detect
        """
        self.min_road_area = min_road_area
        self.max_road_areas = max_road_areas
        self.road_masks = []
        self.road_areas = []
        
    def detect_road_areas(self, frame: np.ndarray, save_debug=False) -> List[np.ndarray]:
        """
        Detect road areas in the first frame using computer vision techniques
        
        Args:
            frame: Input frame (BGR format)
            save_debug: Save debug images for analysis
            
        Returns:
            List of binary masks for detected road areas
        """
        print("Detecting road areas in first frame...")
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Method 1: Road detection using color and texture analysis
        road_mask1 = self._detect_by_color_texture(frame, gray, hsv)
        
        # Method 2: Road detection using edge analysis and morphology
        road_mask2 = self._detect_by_edges_morphology(gray)
        
        # Method 3: Road detection using bottom region analysis
        road_mask3 = self._detect_by_bottom_region(frame, gray)
        
        # Combine masks using voting
        combined_mask = self._combine_masks([road_mask1, road_mask2, road_mask3])
        
        # Find distinct road areas
        road_areas = self._find_road_regions(combined_mask)
        
        # Save debug images if requested
        if save_debug:
            self._save_debug_images(frame, [road_mask1, road_mask2, road_mask3], 
                                  combined_mask, road_areas)
        
        # If no roads detected, fall back to simple split
        if len(road_areas) == 0:
            print("No roads detected with computer vision, falling back to simple split...")
            road_areas = self._create_simple_road_split(frame)
        
        # Store results
        self.road_masks = road_areas
        self.road_areas = [self._get_bounding_box(mask) for mask in road_areas]
        
        print(f"Detected {len(road_areas)} road area(s)")
        return road_areas
    
    def _detect_by_color_texture(self, frame: np.ndarray, gray: np.ndarray, 
                                hsv: np.ndarray) -> np.ndarray:
        """
        Detect road areas using color and texture analysis
        """
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Road color ranges (asphalt/concrete)
        # Dark gray to light gray range
        lower_gray = np.array([0, 0, 20])
        upper_gray = np.array([180, 50, 120])
        
        # Create color mask
        color_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # Texture analysis using Local Binary Pattern approximation
        # Calculate standard deviation in local neighborhoods
        kernel = np.ones((15, 15), np.float32) / 225
        mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_mean = cv2.filter2D((gray.astype(np.float32))**2, -1, kernel)
        texture = np.sqrt(sqr_mean - mean**2)
        
        # Road typically has moderate texture (not too smooth, not too rough)
        texture_mask = ((texture > 5) & (texture < 25)).astype(np.uint8) * 255
        
        # Combine color and texture
        combined = cv2.bitwise_and(color_mask, texture_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        return combined
    
    def _detect_by_edges_morphology(self, gray: np.ndarray) -> np.ndarray:
        """
        Detect road areas using edge analysis and morphological operations
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find areas with low edge density (roads typically have fewer edges)
        kernel_large = np.ones((31, 31), np.float32) / (31*31)
        edge_density = cv2.filter2D(edges_dilated.astype(np.float32), -1, kernel_large)
        
        # Road areas have moderate edge density
        road_mask = ((edge_density > 0.02) & (edge_density < 0.15)).astype(np.uint8) * 255
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        
        return road_mask
    
    def _detect_by_bottom_region(self, frame: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """
        Detect road areas by analyzing bottom regions of the image
        """
        h, w = gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Focus on bottom 70% of image where roads typically appear
        roi_top = int(h * 0.3)
        roi = gray[roi_top:, :]
        
        # Use adaptive thresholding to find uniform regions
        adaptive = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 21, 10)
        
        # Invert to get dark regions (roads are typically darker)
        adaptive_inv = cv2.bitwise_not(adaptive)
        
        # Find large connected components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        processed = cv2.morphologyEx(adaptive_inv, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        
        # Place back in full image
        mask[roi_top:, :] = processed
        
        return mask
    
    def _combine_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """
        Combine multiple masks using voting
        """
        if not masks:
            return np.zeros_like(masks[0])
        
        # Convert to binary and sum
        combined = np.zeros_like(masks[0], dtype=np.float32)
        for mask in masks:
            combined += (mask > 127).astype(np.float32)
        
        # Require at least 2 out of 3 methods to agree
        threshold = len(masks) / 2
        result = (combined >= threshold).astype(np.uint8) * 255
        
        return result
    
    def _find_road_regions(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Find distinct road regions from combined mask
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        
        road_masks = []
        
        # Sort by area (largest first)
        areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        sorted_indices = np.argsort(areas)[::-1]
        
        for i in sorted_indices[:self.max_road_areas]:
            area = areas[i]
            if area >= self.min_road_area:
                # Create mask for this region
                region_mask = (labels == (i + 1)).astype(np.uint8) * 255
                
                # Additional filtering: check aspect ratio and shape
                if self._validate_road_shape(region_mask, stats[i + 1]):
                    road_masks.append(region_mask)
        
        return road_masks
    
    def _validate_road_shape(self, mask: np.ndarray, stats: np.ndarray) -> bool:
        """
        Validate if the detected region has road-like characteristics
        """
        x, y, w, h, area = stats
        
        # Road should have reasonable aspect ratio
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 10:  # Too elongated
            return False
        
        # Road should not be too small relative to image
        image_area = mask.shape[0] * mask.shape[1]
        if area < image_area * 0.05:  # Less than 5% of image
            return False
        
        # Road should not cover entire image
        if area > image_area * 0.8:  # More than 80% of image
            return False
        
        return True
    
    def _create_simple_road_split(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Create simple road areas by splitting frame into left and right halves
        as a fallback when computer vision detection fails
        """
        h, w = frame.shape[:2]
        road_masks = []
        
        # Left road area (left half of frame)
        left_mask = np.zeros((h, w), dtype=np.uint8)
        left_mask[:, :w//2] = 255
        road_masks.append(left_mask)
        
        # Right road area (right half of frame) 
        right_mask = np.zeros((h, w), dtype=np.uint8)
        right_mask[:, w//2:] = 255
        road_masks.append(right_mask)
        
        print(f"Created {len(road_masks)} simple road areas (left/right split)")
        return road_masks
    
    def _get_bounding_box(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Get bounding box coordinates for a road mask
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            return (x, y, x + w, y + h)
        return (0, 0, mask.shape[1], mask.shape[0])
    
    def _save_debug_images(self, original: np.ndarray, individual_masks: List[np.ndarray],
                          combined_mask: np.ndarray, final_masks: List[np.ndarray]):
        """
        Save debug images for analysis
        """
        debug_dir = Path("debug_road_detection")
        debug_dir.mkdir(exist_ok=True)
        
        # Save original
        cv2.imwrite(str(debug_dir / "01_original.jpg"), original)
        
        # Save individual masks
        for i, mask in enumerate(individual_masks):
            cv2.imwrite(str(debug_dir / f"02_method_{i+1}.jpg"), mask)
        
        # Save combined mask
        cv2.imwrite(str(debug_dir / "03_combined.jpg"), combined_mask)
        
        # Save final road areas
        for i, mask in enumerate(final_masks):
            cv2.imwrite(str(debug_dir / f"04_road_area_{i+1}.jpg"), mask)
        
        # Save overlay
        overlay = original.copy()
        for i, mask in enumerate(final_masks):
            color = [(0, 255, 0), (0, 0, 255), (255, 0, 0)][i % 3]
            overlay[mask > 127] = color
        
        result = cv2.addWeighted(original, 0.7, overlay, 0.3, 0)
        cv2.imwrite(str(debug_dir / "05_overlay.jpg"), result)
        
        print(f"Debug images saved to {debug_dir}/")
    
    def is_in_road_area(self, bbox: Tuple[float, float, float, float]) -> bool:
        """
        Check if a bounding box intersects with any detected road area
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            True if bbox intersects with road area
        """
        if not self.road_masks:
            return True  # If no road detection, allow all detections
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Check intersection with any road mask
        for mask in self.road_masks:
            # Create bbox mask
            bbox_mask = np.zeros_like(mask)
            bbox_mask[y1:y2, x1:x2] = 255
            
            # Check intersection
            intersection = cv2.bitwise_and(mask, bbox_mask)
            if np.sum(intersection > 0) > 0:
                return True
        
        return False
    
    def get_road_info(self) -> dict:
        """
        Get information about detected road areas
        """
        return {
            "num_roads": len(self.road_areas),
            "road_areas": self.road_areas,
            "total_road_pixels": sum(np.sum(mask > 0) for mask in self.road_masks)
        }
    
    def save_road_data(self, filepath: str):
        """
        Save road detection data to file
        """
        data = {
            "road_areas": self.road_areas,
            "num_roads": len(self.road_areas)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save masks as numpy arrays
        mask_dir = Path(filepath).parent / "road_masks"
        mask_dir.mkdir(exist_ok=True)
        
        for i, mask in enumerate(self.road_masks):
            np.save(mask_dir / f"road_mask_{i}.npy", mask)
    
    def load_road_data(self, filepath: str):
        """
        Load road detection data from file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.road_areas = data["road_areas"]
        
        # Load masks
        mask_dir = Path(filepath).parent / "road_masks"
        self.road_masks = []
        
        for i in range(data["num_roads"]):
            mask_path = mask_dir / f"road_mask_{i}.npy"
            if mask_path.exists():
                mask = np.load(mask_path)
                self.road_masks.append(mask)
