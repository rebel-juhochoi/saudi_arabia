import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import OrderedDict
from utils import Tracker, Renderer
from models import VehicleDetector, RoadAreaDetector


class VideoStreamProcessor:
    """
    VideoStreamProcessor with multiple video support and continuous streaming
    """
    def __init__(self, vehicle_conf=0.7, iou_threshold=0.5, show_segmentation=True, enable_road_detection=True):
        """
        Initialize VideoStreamProcessor with shared models and video data
        """
        # Store configuration
        self.vehicle_conf = vehicle_conf
        self.iou_threshold = iou_threshold
        self.show_segmentation = show_segmentation
        self.enable_road_detection = enable_road_detection
        
        # Initialize shared models (separate from trackers)
        print("Initializing shared models...")
        self.vehicle_detector = VehicleDetector(conf=vehicle_conf)
        
        # Initialize road detector if enabled
        self.road_detector = RoadAreaDetector() if enable_road_detection else None
        
        # Initialize renderer
        self.renderer = Renderer(show_segmentation=show_segmentation)
        
        # Load all videos into ordered dictionary
        self.videos = self._load_all_videos()
        
        # Initialize trackers for each video (but don't start them yet)
        self.trackers = {}
        self._initialize_trackers()
        
        print(f"VideoStreamProcessor initialized with {len(self.videos)} videos")
    
    def _load_all_videos(self):
        """
        Load all 5 input videos into an ordered dictionary
        Returns: OrderedDict with video numbers as keys and video data as values
        """
        videos = OrderedDict()
        project_root = Path(__file__).parent.parent
        
        # Video numbers to load
        video_numbers = ["01", "02", "03", "04", "05"]
        
        for video_num in video_numbers:
            input_path = project_root / "data" / "inputs" / f"{video_num}_*.mp4"
            
            # Find the actual video file (since we don't know the exact name after the number)
            video_files = list(project_root.glob(f"data/inputs/{video_num}_*.mp4"))
            
            if not video_files:
                print(f"Warning: No video found for number {video_num}")
                continue
            
            # Use the first matching file
            video_path = video_files[0]
            video_name = video_path.stem  # Get name without extension
            
            # Open video
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                continue
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Loaded video {video_num}: {video_name} - {width}x{height}, {fps} FPS, {total_frames} frames")
            
            videos[video_num] = {
                'name': video_name,
                'cap': cap,
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames,
                'path': video_path
            }
        
        return videos
    
    def _initialize_trackers(self):
        """
        Initialize a tracker for each video
        """
        for video_num in self.videos.keys():
            # Create tracker with shared models
            tracker = Tracker(
                vehicle_detector=self.vehicle_detector,
                iou_threshold=self.iou_threshold,
                road_detector=self.road_detector,
                enable_traffic_counting=self.enable_road_detection
            )
            
            self.trackers[video_num] = tracker
            print(f"Initialized tracker for video {video_num}")
    
    def process_video_stream(self, video_num):
        """
        Process a specific video in streaming mode with looping
        """
        if video_num not in self.videos:
            print(f"Error: Video {video_num} not found")
            return
        
        video_data = self.videos[video_num]
        tracker = self.trackers[video_num]
        cap = video_data['cap']
        fps = video_data['fps']
        
        # Calculate frames to skip (1 second) - only for first loop
        frames_to_skip = fps
        is_first_loop = True
        
        print(f"Starting stream for video {video_num}: {video_data['name']}")
        print("Press Ctrl+C to stop streaming")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    # Video ended, restart from beginning
                    print(f"Video {video_num} ended, restarting...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    is_first_loop = False  # Disable skip for subsequent loops
                    continue
                
                # Skip first second only on first loop
                if is_first_loop and cap.get(cv2.CAP_PROP_POS_FRAMES) <= frames_to_skip:
                    continue
                
                # Check if this is the first processed frame for road detection
                current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                is_first_frame = (current_frame_num == frames_to_skip + 1) and is_first_loop
                
                # Process frame through tracker
                tracked_objects = tracker.detect_and_track(
                    frame, frame.shape[:2], is_first_frame=is_first_frame
                )
                
                # Get traffic count data for display
                traffic_counts = tracker.get_traffic_counts()
                counting_lines = tracker.get_counting_lines_for_visualization()
                
                # Annotate frame
                annotated_frame = self.renderer.annotate(
                    frame, [], tracked_objects,
                    traffic_counts=traffic_counts,
                    counting_lines=counting_lines
                )
                
                # Display frame
                cv2.imshow(f"Video {video_num} - {video_data['name']}", annotated_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(f"Stopping stream for video {video_num}")
                    break
                
        except KeyboardInterrupt:
            print(f"\nStream interrupted for video {video_num}")
        
        finally:
            cv2.destroyAllWindows()
    
    def restart_tracker(self, video_num):
        """
        Restart the tracker for a specific video
        """
        if video_num not in self.trackers:
            print(f"Error: Tracker for video {video_num} not found")
            return
        
        # Reset tracker state
        tracker = self.trackers[video_num]
        tracker.tracks = {}
        tracker.next_track_id = 1
        tracker.track_history = {}
        tracker.frame_count = 0
        
        print(f"Restarted tracker for video {video_num}")
    
    def list_videos(self):
        """
        List all available videos
        """
        print("Available videos:")
        for video_num, video_data in self.videos.items():
            print(f"  {video_num}: {video_data['name']}")
    
    def cleanup(self):
        """
        Clean up all video captures
        """
        for video_data in self.videos.values():
            video_data['cap'].release()
        cv2.destroyAllWindows()
        print("Cleanup completed")


def parsing_argument():
    parser = argparse.ArgumentParser(description="Stream person detection and tracking on videos")
    parser.add_argument(
        "--video_num",
        required=True,
        help="Video number to stream (01, 02, 03, 04, or 05)"
    )
    parser.add_argument(
        "--vehicle_conf",
        type=float,
        default=0.25,
        help="Confidence threshold for vehicle detection"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IOU threshold for tracking (default: 0.5)"
    )
    parser.add_argument(
        "--pure",
        action="store_true",
        help="Use pure deepface model without color-based heuristic (default: use color heuristic)"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Disable segmentation mask display (default: show segmentation)"
    )
    parser.add_argument(
        "--no_road_detection",
        action="store_true",
        help="Disable road area detection (default: enabled)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available videos and exit"
    )
    return parser.parse_args()


def main():
    """
    Run video streaming with detection and tracking
    """
    args = parsing_argument()
    
    # Parse arguments
    video_num = args.video_num
    vehicle_conf = args.vehicle_conf
    iou_threshold = args.iou
    show_segmentation = not args.simple     # Invert: --simple means disable segmentation
    enable_road_detection = not args.no_road_detection
    
    # Initialize VideoStreamProcessor
    processor = VideoStreamProcessor(
        vehicle_conf=vehicle_conf,
        iou_threshold=iou_threshold,
        show_segmentation=show_segmentation,
        enable_road_detection=enable_road_detection
    )
    
    # List videos if requested
    if args.list:
        processor.list_videos()
        processor.cleanup()
        return
    
    # Process video stream
    try:
        processor.process_video_stream(video_num)
    finally:
        processor.cleanup()


if __name__ == "__main__":
    main()
