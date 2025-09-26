import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
import shutil
import asyncio
import cv2
import numpy as np
from collections import deque
import threading
import time
import gc
import weakref

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import json
import base64
import io

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import the VideoProcessor and related components
from process import VideoProcessor


app = FastAPI(
    title="Vehicle Detection Video Processing API",
    description="FastAPI server for processing videos with vehicle detection and tracking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount data directory for video files first (more specific path)
data_inputs_path = Path(__file__).parent / "data" / "inputs"
if data_inputs_path.exists():
    app.mount("/static/inputs", StaticFiles(directory=str(data_inputs_path)), name="inputs")

# Mount static files for frontend
frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Global storage for single demo processor
demo_processor: Optional[VideoProcessor] = None
processing_status: Dict[str, Dict[str, Any]] = {}

# Single demo configuration for traffic detection
DEMO_CONFIG = {
    "vehicle_conf": 0.25,  # Standard confidence for demo
    "iou_threshold": 0.5,
    "show_segmentation": False,
    "enable_road_detection": True
}


def initialize_demo_processor():
    """Initialize single demo processor"""
    global demo_processor
    
    print("Initializing demo processor...")
    demo_processor = VideoProcessor(
        vehicle_conf=DEMO_CONFIG["vehicle_conf"],
        iou_threshold=DEMO_CONFIG["iou_threshold"],
        show_segmentation=DEMO_CONFIG["show_segmentation"],
        enable_road_detection=DEMO_CONFIG["enable_road_detection"],
        count_display_delay=4.0  # 2 second delay for better visual synchronization
    )
    print("Demo processor initialized!")


@app.on_event("startup")
async def startup_event():
    """Initialize demo processor when the server starts"""
    initialize_demo_processor()


class UltraFastStreamingProcessor(VideoProcessor):
    """Ultra-fast streaming video processor with asynchronous pre-processing pipeline and zero-lag display"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_streaming = False
        self.show_segmentation = kwargs.get('show_segmentation', False)
        self.streaming_speed = 1.0  # Normal speed
        
        # Asynchronous pipeline optimization
        self.frame_queue = asyncio.Queue(maxsize=30)  # Frame buffer
        self.detection_queue = asyncio.Queue(maxsize=50)  # Detection results buffer
        self.processed_frames = {}  # Frame cache with detection results
        self.frame_workers = 2  # Reduced workers to prevent memory issues
        self.preload_seconds = 2  # Pre-load 2 seconds of detections
        
        # Thread safety
        self._frame_lock = asyncio.Lock()  # Lock for processed_frames access
        self._cleanup_lock = asyncio.Lock()  # Lock for cleanup operations
        
        # Video FPS synchronization
        self.video_fps = 30  # Will be detected from video
        self.target_fps = 30  # Target FPS (same as video FPS)
        self.frame_interval = 1.0 / 30  # Time between frames in seconds
        
        # Adaptive FPS optimization (now based on video FPS)
        self.processing_times = deque(maxlen=10)  # Track last 10 frames
        self.adaptive_fps = 30  # Start with video FPS
        self.min_fps = 15
        self.max_fps = 60  # Maximum FPS (2x video FPS for smoothness)
        self.fps_adjustment_threshold = 0.01  # 10ms threshold
        
        # Frame skipping optimization
        self.frames_skipped = 0
        self.max_skip_ratio = 0.1  # Reduced skip ratio for better quality
        self.skip_threshold = 0.03  # Skip if processing takes > 30ms
        self.consecutive_skips = 0
        self.max_consecutive_skips = 3
        
        # Binary transmission optimization
        self.use_binary_transmission = True
        self.compression_quality = 90  # Higher quality for better detection
    
    def _detect_video_fps(self, cap):
        """Detect video FPS and update synchronization settings"""
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            self.video_fps = fps
            self.target_fps = fps
            self.frame_interval = 1.0 / fps
            self.adaptive_fps = min(fps, 60)  # Cap at 60 FPS for smoothness
            self.max_fps = min(fps * 2, 120)  # Max 2x video FPS
            print(f"Detected video FPS: {fps}, Target FPS: {self.target_fps}")
        else:
            print("Could not detect video FPS, using default 30 FPS")
            self.video_fps = 30
            self.target_fps = 30
            self.frame_interval = 1.0 / 30
            self.adaptive_fps = 30
        
        # Pipeline state
        self.pipeline_started = False
        self.detection_workers = []
        self.current_frame_id = 0
        
    async def stream_video(self, video_path: str, websocket: WebSocket):
        """Ultra-fast streaming with asynchronous pre-processing pipeline"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            await websocket.send_text(json.dumps({"type": "error", "message": "Could not open video"}))
            return
        
        # Detect video FPS and update synchronization settings
        self._detect_video_fps(cap)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Send video info to frontend
        await websocket.send_text(json.dumps({
            "type": "video_info",
            "fps": self.video_fps,
            "target_fps": self.target_fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "adaptive_fps": self.adaptive_fps,
            "binary_transmission": self.use_binary_transmission,
            "ultra_fast_mode": True,
            "preload_seconds": self.preload_seconds
        }))
        
        self.is_streaming = True
        self.current_frame_id = 0
        
        # Reset optimization counters
        self.frames_skipped = 0
        self.consecutive_skips = 0
        self.processing_times.clear()
        self.processed_frames.clear()
        
        # Start asynchronous pipeline
        await self._start_async_pipeline(cap, websocket)
    
    async def _start_async_pipeline(self, cap, websocket):
        """Start the asynchronous processing pipeline"""
        detection_tasks = []
        try:
            # Start parallel detection workers
            for i in range(self.frame_workers):
                task = asyncio.create_task(self._detection_worker(f"worker-{i}"))
                detection_tasks.append(task)
            
            # Start frame reader task
            reader_task = asyncio.create_task(self._frame_reader(cap, websocket))
            
            # Start frame sender task
            sender_task = asyncio.create_task(self._frame_sender(websocket))
            
            # Start traffic data updater task for real-time synchronization
            traffic_updater_task = asyncio.create_task(self._traffic_data_updater(websocket))
            
            # Wait for all tasks
            await asyncio.gather(reader_task, sender_task, traffic_updater_task, *detection_tasks)
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        finally:
            # Cancel all tasks
            for task in detection_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            
            # Clean up resources
            await self._cleanup_resources()
            cap.release()
            self.is_streaming = False
    
    async def _cleanup_resources(self):
        """Clean up resources to prevent memory corruption"""
        try:
            async with self._cleanup_lock:
                # Clear processed frames
                self.processed_frames.clear()
                
                # Clear queues
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                
                while not self.detection_queue.empty():
                    try:
                        self.detection_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                
                # Force garbage collection
                import gc
                gc.collect()
                
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    async def _frame_reader(self, cap, websocket):
        """Continuously read frames and add to queue with looping support"""
        frame_id = 0
        loop_count = 0
        
        while self.is_streaming:
            ret, frame = cap.read()
            if not ret:
                # End of video - restart for looping
                loop_count += 1
                print(f"Video ended, restarting loop #{loop_count}")
                
                # Notify frontend about loop restart
                try:
                    await websocket.send_text(json.dumps({
                        "type": "loop_restart",
                        "message": f"Video restarting (loop #{loop_count})",
                        "loop_count": loop_count
                    }))
                except Exception as e:
                    print(f"Failed to send loop restart notification: {e}")
                
                # Reset video to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_id = 0
                
                # Clear processed frames cache for new loop
                async with self._frame_lock:
                    self.processed_frames.clear()
                
                await asyncio.sleep(0.1)
                continue
            
            frame_id += 1
            
            # Add frame to processing queue
            try:
                await self.frame_queue.put({
                    'frame_id': frame_id,
                    'frame': frame.copy(),
                    'timestamp': time.time(),
                    'loop_count': loop_count
                })
            except asyncio.QueueFull:
                # Skip frame if queue is full
                self.frames_skipped += 1
                continue
    
    async def _detection_worker(self, worker_name):
        """Worker that processes frames for detection"""
        while self.is_streaming:
            try:
                # Get frame from queue with timeout
                frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
                frame_id = frame_data['frame_id']
                frame = frame_data['frame']
                
                # Process frame for detection
                loop = asyncio.get_event_loop()
                processed_frame, active_tracks, traffic_data = await loop.run_in_executor(
                    None, self._process_frame_safe, frame, frame_id
                )
                
                # Store processed frame with thread safety
                async with self._frame_lock:
                    self.processed_frames[frame_id] = {
                        'processed_frame': processed_frame,
                        'active_tracks': active_tracks,
                        'traffic_data': traffic_data,
                        'timestamp': time.time(),
                        'loop_count': frame_data.get('loop_count', 0)
                    }
                    
                    # Clean up old frames (keep last 50 to reduce memory usage)
                    if len(self.processed_frames) > 50:
                        oldest_key = min(self.processed_frames.keys())
                        del self.processed_frames[oldest_key]
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Detection worker {worker_name} error: {e}")
                # Add small delay to prevent rapid error loops
                await asyncio.sleep(0.1)
                continue
    
    async def _frame_sender(self, websocket):
        """Send frames to client with pre-loaded detections at video FPS"""
        loop = asyncio.get_event_loop()
        last_sent_frame = 0
        last_frame_time = time.time()
        current_loop_count = 0
        
        while self.is_streaming:
            try:
                # Calculate time since last frame
                current_time = time.time()
                time_since_last_frame = current_time - last_frame_time
                
                # Only send frame if enough time has passed (respect video FPS)
                if time_since_last_frame < self.frame_interval:
                    await asyncio.sleep(self.frame_interval - time_since_last_frame)
                    continue
                
                # Look for next frame to send with thread safety
                frame_to_send = None
                async with self._frame_lock:
                    # Check if we need to reset for a new loop
                    if self.processed_frames:
                        # Get the loop count from the first available frame
                        first_frame_id = min(self.processed_frames.keys())
                        frame_data = self.processed_frames[first_frame_id]
                        if isinstance(frame_data, dict) and frame_data.get('loop_count', 0) > current_loop_count:
                            # New loop started, reset frame counter
                            current_loop_count = frame_data.get('loop_count', 0)
                            last_sent_frame = 0
                            print(f"Frame sender: New loop detected (#{current_loop_count}), resetting frame counter")
                    
                    for frame_id in range(last_sent_frame + 1, last_sent_frame + 10):
                        if frame_id in self.processed_frames:
                            frame_to_send = self.processed_frames[frame_id]
                            last_sent_frame = frame_id
                            break
                
                if frame_to_send is None:
                    # No processed frame available, send raw frame
                    try:
                        frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=0.1)
                        frame_to_send = {
                            'processed_frame': frame_data['frame'],
                            'active_tracks': 0,
                            'traffic_data': None,
                            'timestamp': time.time()
                        }
                    except asyncio.TimeoutError:
                        await asyncio.sleep(0.01)  # Short wait
                        continue
                
                # Update timing
                last_frame_time = current_time
                
                # Get current traffic data for real-time synchronization
                current_traffic_data = None
                if hasattr(self, 'tracker') and self.tracker:
                    current_traffic_data = self.tracker.get_traffic_counts()
                
                # Use current traffic data if available, otherwise use frame data
                traffic_data_to_send = current_traffic_data if current_traffic_data else frame_to_send.get('traffic_data')
                
                # Encode and send frame
                if self.use_binary_transmission:
                    encoded_frame = await loop.run_in_executor(
                        None, self._encode_frame_binary, frame_to_send['processed_frame']
                    )
                    
                    if encoded_frame:
                        await websocket.send_bytes(encoded_frame)
                        
                        metadata = {
                            "type": "frame_metadata",
                            "frame_count": last_sent_frame,
                            "active_tracks": frame_to_send['active_tracks'],
                            "traffic_counts": traffic_data_to_send,
                            "adaptive_fps": self.adaptive_fps,
                            "frames_skipped": self.frames_skipped,
                            "pipeline_mode": "ultra_fast",
                            "timestamp": time.time()
                        }
                        await websocket.send_text(json.dumps(metadata))
                else:
                    # Fallback to base64
                    encoded_frame = await loop.run_in_executor(
                        None, self._encode_frame, frame_to_send['processed_frame']
                    )
                    
                    if encoded_frame:
                        message = {
                            "type": "frame",
                            "frame": encoded_frame,
                            "frame_count": last_sent_frame,
                            "active_tracks": frame_to_send['active_tracks'],
                            "traffic_counts": traffic_data_to_send,
                            "adaptive_fps": self.adaptive_fps,
                            "frames_skipped": self.frames_skipped,
                            "pipeline_mode": "ultra_fast",
                            "timestamp": time.time()
                        }
                        await websocket.send_text(json.dumps(message))
                
                # Dynamic frame delay
                frame_delay = (1.0 / self.adaptive_fps) / self.streaming_speed
                await asyncio.sleep(frame_delay)
                
            except Exception as e:
                print(f"Frame sender error: {e}")
                await asyncio.sleep(0.1)
    
    async def _traffic_data_updater(self, websocket):
        """Send periodic traffic data updates for real-time synchronization"""
        last_traffic_data = None
        update_interval = 0.1  # Update every 100ms for real-time feel
        
        while self.is_streaming:
            try:
                # Get current traffic data
                current_traffic_data = None
                if hasattr(self, 'tracker') and self.tracker:
                    current_traffic_data = self.tracker.get_traffic_counts()
                
                # Only send if data has changed
                if current_traffic_data and current_traffic_data != last_traffic_data:
                    traffic_update = {
                        "type": "traffic_update",
                        "traffic_counts": current_traffic_data,
                        "timestamp": time.time()
                    }
                    await websocket.send_text(json.dumps(traffic_update))
                    last_traffic_data = current_traffic_data
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                print(f"Traffic data updater error: {e}")
                await asyncio.sleep(0.5)  # Wait longer on error
                continue
    
    def _process_frame_safe(self, frame, frame_count):
        """Safely process frame with error handling and memory management"""
        try:
            # Create a copy of the frame to prevent memory issues
            frame_copy = frame.copy()
            
            # Update renderer segmentation setting
            if hasattr(self, 'renderer') and self.renderer is not None:
                self.renderer.show_segmentation = self.show_segmentation
            
            # Process frame
            processed_frame, active_tracks = self.process_frame(frame_copy, frame_count)
            
            # Get traffic count data
            traffic_data = None
            if hasattr(self, 'tracker') and self.tracker:
                try:
                    traffic_counts = self.tracker.get_traffic_counts()
                    if traffic_counts:
                        traffic_data = {
                            'total_vehicles': traffic_counts.get('total_vehicles', 0),
                            'road_counts': {str(k): v for k, v in traffic_counts.get('road_counts', {}).items()},
                            'session_duration': traffic_counts.get('session_duration', 0)
                        }
                except Exception as e:
                    print(f"Traffic count error: {e}")
                    traffic_data = None
            
            # Ensure we return a copy to prevent memory corruption
            return processed_frame.copy() if processed_frame is not None else frame_copy, active_tracks, traffic_data
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            # Return original frame as fallback
            return frame.copy() if frame is not None else None, 0, None
    
    def _encode_frame(self, frame):
        """Encode frame to base64"""
        try:
            _, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, 80,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Frame encoding error: {e}")
            return None
    
    def stop_streaming(self):
        """Stop streaming"""
        self.is_streaming = False
    
    def set_speed(self, speed: float):
        """Set streaming speed"""
        self.streaming_speed = max(0.1, min(3.0, speed))
    
    def _should_skip_frame(self, frame_count: int) -> bool:
        """Determine if current frame should be skipped for performance"""
        # Don't skip if we've already skipped too many frames
        if self.frames_skipped >= frame_count * self.max_skip_ratio:
            return False
        
        # Don't skip if we've skipped too many consecutive frames
        if self.consecutive_skips >= self.max_consecutive_skips:
            return False
        
        # Skip if processing is taking too long
        if len(self.processing_times) > 0:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            if avg_processing_time > self.skip_threshold:
                return True
        
        return False
    
    def _adjust_adaptive_fps(self):
        """Adjust FPS based on processing performance"""
        if len(self.processing_times) < 3:  # Need at least 3 samples
            return
        
        avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        target_frame_time = 1.0 / self.adaptive_fps
        
        if avg_processing_time > target_frame_time * 0.8:  # If processing takes > 80% of frame time
            # Reduce FPS
            self.adaptive_fps = max(self.min_fps, self.adaptive_fps - 2)
        elif avg_processing_time < target_frame_time * 0.4:  # If processing takes < 40% of frame time
            # Increase FPS
            self.adaptive_fps = min(self.max_fps, self.adaptive_fps + 1)
    
    def _encode_frame_binary(self, frame):
        """Encode frame as binary data for efficient transmission"""
        try:
            # Compress frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [
                cv2.IMWRITE_JPEG_QUALITY, self.compression_quality,
                cv2.IMWRITE_JPEG_OPTIMIZE, 1
            ])
            return buffer.tobytes()
        except Exception as e:
            print(f"Binary frame encoding error: {e}")
            return None
    
    def toggle_segmentation(self, enabled: bool):
        """Toggle segmentation display"""
        self.show_segmentation = enabled
        print(f"Segmentation {'enabled' if enabled else 'disabled'}")


class CustomVideoProcessor(VideoProcessor):
    """Extended VideoProcessor for handling uploaded files"""
    
    def setup_uploaded_video(self, input_path: str, output_path: str):
        """
        Setup video from uploaded file path
        Returns: (cap, fps, width, height, total_frames, output_path) or None if error
        """
        # Validate input
        if not Path(input_path).exists():
            print(f"Error: Video file {input_path} not found")
            return None
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
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
    
    def process_uploaded_video(self, input_path: str, output_path: str, job_id: str):
        """Process an uploaded video file"""
        global processing_status
        
        try:
            # Update status
            processing_status[job_id]["status"] = "processing"
            processing_status[job_id]["message"] = "Setting up video..."
            
            # Setup video
            result = self.setup_uploaded_video(input_path, output_path)
            if result is None:
                processing_status[job_id]["status"] = "error"
                processing_status[job_id]["message"] = "Failed to setup video"
                return
            
            cap, fps, width, height, total_frames, _ = result
            
            # Process frames
            frame_count = 0
            last_progress = 0
            total_tracks = 0
            
            processing_status[job_id]["message"] = f"Processing video..."
            
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
                current_progress = int((frame_count / total_frames) * 100)
                if current_progress >= last_progress + 10 and current_progress <= 100:
                    processing_status[job_id]["progress"] = current_progress
                    processing_status[job_id]["message"] = f"Progress: {current_progress}% ({frame_count}/{total_frames} frames) - Active tracks: {active_tracks}"
                    last_progress = current_progress
            
            # Clean up
            cap.release()
            self.video_writer.release()
            cv2.destroyAllWindows()
            
            # Update final status
            processing_status[job_id]["status"] = "completed"
            processing_status[job_id]["progress"] = 100
            processing_status[job_id]["message"] = f"Processing complete! Maximum active tracks: {total_tracks}"
            processing_status[job_id]["output_path"] = output_path
            
        except Exception as e:
            processing_status[job_id]["status"] = "error"
            processing_status[job_id]["message"] = f"Error during processing: {str(e)}"
            
            # Clean up on error
            try:
                cap.release()
                self.video_writer.release()
                cv2.destroyAllWindows()
            except:
                pass



async def process_video_async(processor: CustomVideoProcessor, input_path: str, output_path: str, job_id: str):
    """Async wrapper for video processing"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, processor.process_uploaded_video, input_path, output_path, job_id)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend web application"""
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if frontend_path.exists():
        with open(frontend_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return {
            "message": "Vehicle Detection Demo API",
            "version": "1.0.0",
            "available_endpoints": [
                "/ws/demo-stream",
                "/traffic-analytics",
                "/traffic-summary",
                "/reset-counters",
                "/demo-config"
            ],
            "note": "Frontend not found. API endpoints available at /docs"
        }

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Vehicle Detection Demo API",
        "version": "1.0.0",
        "available_endpoints": [
            "/ws/demo-stream",
            "/traffic-analytics",
            "/traffic-summary",
            "/reset-counters",
            "/demo-config"
        ]
    }


@app.get("/demo-config")
async def get_demo_config():
    """Get demo configuration"""
    return {
        "demo_config": DEMO_CONFIG,
        "status": "ready" if demo_processor else "initializing"
    }


# Old video processing endpoints removed for simplified demo mode


# @app.post("/process-custom-video")
async def process_custom_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    vehicle_conf: float = Form(0.25),
    iou_threshold: float = Form(0.5),
    show_segmentation: bool = Form(False),
    enable_road_detection: bool = Form(True)
):
    """
    Process an uploaded video with custom configuration parameters
    """
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files are supported")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create temporary directory for this job
    temp_dir = Path(tempfile.gettempdir()) / "video_processing" / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded file
    input_path = temp_dir / f"input_{file.filename}"
    output_path = temp_dir / f"output_{file.filename}"
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    
    # Initialize processing status
    processing_status[job_id] = {
        "status": "queued",
        "progress": 0,
        "message": "Video queued for processing",
        "config_name": "custom",
        "filename": file.filename,
        "output_path": None
    }
    
    # Create custom processor with the provided configuration
    processor = CustomVideoProcessor(
        vehicle_conf=vehicle_conf,
        iou_threshold=iou_threshold,
        show_segmentation=show_segmentation,
        enable_road_detection=enable_road_detection
    )
    
    # Start processing in background
    background_tasks.add_task(
        process_video_async, 
        processor, 
        str(input_path), 
        str(output_path), 
        job_id
    )
    
    custom_config = {
        "vehicle_conf": vehicle_conf,
        "iou_threshold": iou_threshold,
        "show_segmentation": show_segmentation,
        "enable_road_detection": enable_road_detection
    }
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video processing started with custom configuration",
        "config_used": custom_config
    }


@app.get("/status/{job_id}")
async def get_processing_status(job_id: str):
    """Get the processing status of a job"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_status[job_id]


@app.get("/download/{job_id}")
async def download_processed_video(job_id: str):
    """Download the processed video"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_status = processing_status[job_id]
    
    if job_status["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Video processing not completed. Current status: {job_status['status']}"
        )
    
    output_path = job_status.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Processed video file not found")
    
    filename = f"processed_{job_status['filename']}"
    
    return FileResponse(
        path=output_path,
        filename=filename,
        media_type="video/mp4"
    )


@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up temporary files for a job"""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove temporary directory
    temp_dir = Path(tempfile.gettempdir()) / "video_processing" / job_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # Remove from processing status
    del processing_status[job_id]
    
    return {"message": f"Job {job_id} cleaned up successfully"}


@app.websocket("/ws/demo-stream")
async def websocket_demo_stream(websocket: WebSocket):
    """WebSocket endpoint for demo video streaming"""
    await websocket.accept()
    
    if not demo_processor:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": "Demo processor not initialized"
        }))
        await websocket.close()
        return
    
    # Use the demo video file
    video_file_path = str(Path(__file__).parent / "data" / "inputs" / "traffic.mp4")
    
    # Check if video file exists
    if not Path(video_file_path).exists():
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": "Demo video file not found: traffic.mp4"
        }))
        await websocket.close()
        return
    
    processor = None
    
    try:
        # Create simple streaming processor with demo config
        processor = UltraFastStreamingProcessor(
            vehicle_conf=DEMO_CONFIG["vehicle_conf"],
            iou_threshold=DEMO_CONFIG["iou_threshold"],
            show_segmentation=DEMO_CONFIG["show_segmentation"],
            enable_road_detection=DEMO_CONFIG["enable_road_detection"]
        )
        
        print("Starting demo video stream")
        
        # Start streaming and control handling concurrently
        streaming_task = asyncio.create_task(processor.stream_video(video_file_path, websocket))
        control_task = asyncio.create_task(handle_stream_controls(websocket, processor))
        
        # Wait for either task to complete
        done, pending = await asyncio.wait(
            [streaming_task, control_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except:
            pass
    
    finally:
        if processor:
            processor.stop_streaming()
        print("WebSocket connection closed")


async def handle_stream_controls(websocket: WebSocket, processor: UltraFastStreamingProcessor):
    """Handle control messages from client - clean and simple"""
    try:
        while processor.is_streaming:
            try:
                # Wait for control messages with timeout
                message = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                data = json.loads(message)
                
                if data.get("type") == "control":
                    command = data.get("command")
                    
                    if command == "stop":
                        processor.stop_streaming()
                        await websocket.send_text(json.dumps({
                            "type": "control_response", 
                            "message": "Streaming stopped"
                        }))
                        break
                        
                    elif command == "set_speed":
                        speed = float(data.get("value", 1.0))
                        processor.set_speed(speed)
                        await websocket.send_text(json.dumps({
                            "type": "control_response", 
                            "message": f"Speed set to {speed}x"
                        }))
                        
                    elif command == "toggle_segmentation":
                        enabled = bool(data.get("value", True))
                        processor.toggle_segmentation(enabled)
                        await websocket.send_text(json.dumps({
                            "type": "control_response", 
                            "message": f"Segmentation {'enabled' if enabled else 'disabled'}"
                        }))
                        
            except asyncio.TimeoutError:
                continue  # Keep listening
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                continue  # Skip invalid messages
            except Exception as e:
                print(f"Control error: {e}")
                continue
                
    except Exception as e:
        print(f"Control handler error: {e}")
    finally:
        print("Control handler stopped")


@app.get("/traffic-analytics")
async def get_traffic_analytics():
    """Get traffic analytics for demo"""
    if not demo_processor:
        raise HTTPException(status_code=404, detail="Demo processor not initialized")
    
    if hasattr(demo_processor, 'tracker') and demo_processor.tracker:
        analytics = demo_processor.tracker.get_analytics_data()
        if analytics:
            return analytics
    
    return {"message": "No analytics data available"}


@app.get("/traffic-summary")
async def get_traffic_summary():
    """Get traffic summary for demo"""
    if not demo_processor:
        raise HTTPException(status_code=404, detail="Demo processor not initialized")
    
    if hasattr(demo_processor, 'tracker') and demo_processor.tracker:
        traffic_counts = demo_processor.tracker.get_traffic_counts()
        if traffic_counts:
            return {
                "total_vehicles": traffic_counts.get('total_vehicles', 0),
                "road_counts": traffic_counts.get('road_counts', {}),
                "session_duration": traffic_counts.get('session_duration', 0),
                "active_tracks": traffic_counts.get('active_tracks', 0)
            }
    
    return {"message": "No traffic data available"}


@app.post("/reset-counters")
async def reset_traffic_counters():
    """Reset traffic counters for demo"""
    if not demo_processor:
        raise HTTPException(status_code=404, detail="Demo processor not initialized")
    
    if hasattr(demo_processor, 'tracker') and demo_processor.tracker:
        demo_processor.tracker.reset_traffic_counters()
        return {"message": "Traffic counters reset"}
    
    return {"message": "No traffic counters to reset"}

@app.get("/debug/counted-vehicles")
async def get_counted_vehicles():
    """Get list of counted vehicle IDs for debugging"""
    try:
        if demo_processor and hasattr(demo_processor, 'tracker'):
            counted_ids = demo_processor.tracker.get_counted_vehicle_ids()
            traffic_counts = demo_processor.tracker.get_traffic_counts()
            return {
                "counted_vehicle_ids": counted_ids,
                "total_counted": len(counted_ids),
                "traffic_counts": traffic_counts
            }
        return {"message": "No processor available"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "demo_processor_ready": demo_processor is not None,
        "active_jobs": len(processing_status)
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
