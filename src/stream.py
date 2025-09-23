import argparse
import cv2
import numpy as np
from pathlib import Path
from collections import OrderedDict
from utils import Tracker, Renderer
from models import PersonDetector, GenderClassifier


class VideoStreamProcessor:
    """
    VideoStreamProcessor with multiple video support and continuous streaming
    """
    def __init__(self, person_conf=0.7, iou_threshold=0.5, gender_conf=0.5, enable_color_heuristic=True, show_segmentation=True):
        """
        Initialize VideoStreamProcessor with shared models and video data
        """
        # Store configuration
        self.person_conf = person_conf
        self.iou_threshold = iou_threshold
        self.gender_conf = gender_conf
        self.enable_color_heuristic = enable_color_heuristic
        self.show_segmentation = show_segmentation
        
        # Initialize shared models (separate from trackers)
        print("Initializing shared models...")
        self.person_detector = PersonDetector(conf=person_conf, enable_color_detection=enable_color_heuristic)
        self.gender_classifier = GenderClassifier(conf_threshold=gender_conf)
        
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
                person_detector=self.person_detector,
                gender_classifier=self.gender_classifier,
                iou_threshold=self.iou_threshold,
                enable_color_heuristic=self.enable_color_heuristic
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
                
                # Process frame through tracker
                detections, tracked_objects = tracker.detect_and_track(frame)
                
                # Annotate frame
                annotated_frame = self.renderer.annotate(frame, detections, tracked_objects)
                
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
        "--person_conf",
        type=float,
        default=0.25,
        help="Confidence threshold for person detection"
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IOU threshold for tracking (default: 0.5)"
    )
    parser.add_argument(
        "--gender_conf",
        type=float,
        default=0.5,
        help="Confidence threshold for gender classification"
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
    person_conf = args.person_conf
    iou_threshold = args.iou
    gender_conf = args.gender_conf
    enable_color_heuristic = not args.pure  # Invert: --pure means disable color heuristic
    show_segmentation = not args.simple     # Invert: --simple means disable segmentation
    
    # Initialize VideoStreamProcessor
    processor = VideoStreamProcessor(
        person_conf=person_conf,
        iou_threshold=iou_threshold,
        gender_conf=gender_conf,
        enable_color_heuristic=enable_color_heuristic,
        show_segmentation=show_segmentation
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
