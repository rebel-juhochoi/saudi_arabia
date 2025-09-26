import argparse
import cv2
import numpy as np
from pathlib import Path
from utils import Tracker, Renderer
from models import VehicleDetector, RoadAreaDetector


class VideoProcessor:
    """
    VideoProcessor with tracking capabilities
    """
    def __init__(self, vehicle_conf=0.7, iou_threshold=0.3, show_segmentation=True, enable_road_detection=True, count_display_delay=0.5):
        """
        Initialize VideoProcessor with shared models and tracker
        """
        # Store configuration
        self.vehicle_conf = vehicle_conf
        self.iou_threshold = iou_threshold
        self.show_segmentation = show_segmentation
        self.enable_road_detection = enable_road_detection
        
        # Initialize shared models
        print("Initializing shared models...")
        self.vehicle_detector = VehicleDetector(conf=vehicle_conf)
        
        # Initialize road detector if enabled
        self.road_detector = RoadAreaDetector() if enable_road_detection else None
        
        # Initialize tracker with shared models
        self.tracker = Tracker(
            vehicle_detector=self.vehicle_detector,
            iou_threshold=iou_threshold,
            road_detector=self.road_detector,
            enable_traffic_counting=enable_road_detection,  # Enable counting when road detection is enabled
            count_display_delay=count_display_delay
        )
        
        # Initialize renderer
        self.renderer = Renderer(show_segmentation=show_segmentation)
        self.video_writer = None  # Will be initialized in setup_input_video
    
    def setup_input_video(self, video_name):
        """
        Setup input video, validate it exists, create output directory, get properties, and initialize video writer
        Returns: (cap, fps, width, height, total_frames, output_path) or None if error
        """
        # Setup paths
        project_root = Path(__file__).parent.parent
        input_path = project_root / "data" / "inputs" / f"{video_name}.mp4"
        output_path = project_root / "data" / "outputs" / f"{video_name}_output.mp4"
        
        # Validate input
        if not input_path.exists():
            print(f"Error: Video file {input_path} not found")
            return None
        
        # Setup output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Print video properties
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        return cap, fps, width, height, total_frames, output_path
    
    def process_frame(self, frame, frame_count):
        """Process a single frame through the unified tracker"""
        # Check if this is the first frame for road detection
        is_first_frame = (frame_count == 1)
        
        # Delegate everything to the unified tracker
        tracked_objects = self.tracker.detect_and_track(
            frame, frame.shape[:2], is_first_frame=is_first_frame
        )
        detections = []  # Detections are handled internally by tracker
        
        # Get traffic count data for display
        traffic_counts = self.tracker.get_traffic_counts()
        counting_lines = self.tracker.get_counting_lines_for_visualization()
        
        annotated_frame = self.renderer.annotate(
            frame, detections, tracked_objects, 
            traffic_counts=traffic_counts, 
            counting_lines=counting_lines
        )
        
        return annotated_frame, len(tracked_objects)
    
    def update_progress(self, frame_count, total_frames, last_progress, active_tracks):
        """Update and display progress with tracking info"""
        current_progress = int((frame_count / total_frames) * 100)
        if current_progress >= last_progress + 10 and current_progress <= 100:
            print(f"Progress: {current_progress}% ({frame_count}/{total_frames} frames) - Active tracks: {active_tracks}")
            return current_progress
        return last_progress
    
    def process_video(self, video_name):
        """Main video processing pipeline with tracking"""
        # Setup input video and get all necessary data
        result = self.setup_input_video(video_name)
        if result is None:
            return
        
        cap, fps, width, height, total_frames, output_path = result
        
        # Process frames
        frame_count = 0
        last_progress = 0
        total_tracks = 0
        
        print(f"Processing video with tracking...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                
                # Process frame
                annotated_frame, active_tracks = self.process_frame(frame, frame_count)
                total_tracks = max(total_tracks, active_tracks)
                
                # Write frame to output video
                self.video_writer.write(annotated_frame)
                
                # Update progress
                last_progress = self.update_progress(frame_count, total_frames, last_progress, active_tracks)
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            # Clean up
            cap.release()
            self.video_writer.release()
            cv2.destroyAllWindows()
            
            print(f"Processing complete! Output saved to: {output_path}")
            print(f"Maximum active tracks: {total_tracks}")


def parsing_argument():
    parser = argparse.ArgumentParser(description="Run person detection and tracking on video")
    # parser.add_argument(
    #     "--input",
    #     required=True,
    #     help="Name of the video file in data/inputs directory (without extension)"
    # )
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
        help="IOU threshold for tracking (default: 0.3)"
    )
    parser.add_argument(
        "--no_road_detection",
        action="store_true",
        help="Disable road area detection (default: enabled)"
    )
    # parser.add_argument(
    #     "--pure",
    #     action="store_true",
    #     help="Use pure deepface model without color-based heuristic (default: use color heuristic)"
    # )
    # parser.add_argument(
    #     "--simple",
    #     action="store_false",
    #     help="Disable segmentation mask display (default: show segmentation)"
    # )
    return parser.parse_args()


def main():
    """
    Run person detection and tracking
    """
    args = parsing_argument()
    # video_name = args.input
    vehicle_conf = args.vehicle_conf
    iou_threshold = args.iou
    enable_road_detection = not args.no_road_detection
    # show_segmentation = not args.simple     # Invert: --simple means disable segmentation

    # Initialize VideoProcessor with tracking
    processor = VideoProcessor(
        vehicle_conf=vehicle_conf, 
        iou_threshold=iou_threshold, 
        show_segmentation=True,
        enable_road_detection=enable_road_detection
    )
    
    # Process video
    processor.process_video("01_man")


    # Initialize VideoProcessor with tracking
    processor = VideoProcessor(
        vehicle_conf=vehicle_conf, 
        iou_threshold=iou_threshold, 
        show_segmentation=True,
        enable_road_detection=enable_road_detection
    )
    
    # Process video
    processor.process_video("02_woman")


    # Initialize VideoProcessor with tracking
    processor = VideoProcessor(
        vehicle_conf=vehicle_conf, 
        iou_threshold=iou_threshold, 
        show_segmentation=True,
        enable_road_detection=enable_road_detection
    )
    
    # Process video
    processor.process_video("03_family")


    # Initialize VideoProcessor with tracking
    processor = VideoProcessor(
        vehicle_conf=vehicle_conf, 
        iou_threshold=iou_threshold, 
        show_segmentation=True,
        enable_road_detection=enable_road_detection
    )
    
    # Process video
    processor.process_video("04_group")


    # Initialize VideoProcessor with tracking
    processor = VideoProcessor(person_conf=0.2, iou_threshold=iou_threshold, gender_conf=gender_conf, enable_color_heuristic=True, show_segmentation=True)
    
    # Process video
    processor.process_video("05_office")


if __name__ == "__main__":
    main()
