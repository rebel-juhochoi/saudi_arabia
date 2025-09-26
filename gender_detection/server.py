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
import subprocess

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
    title="Gender Detection Video Processing API",
    description="FastAPI server for processing videos with gender detection and tracking",
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

# Global storage for video processors (one for each configuration)
video_processors: Dict[str, VideoProcessor] = {}
streaming_processors: Dict[str, Any] = {}  # Will be UltraFastStreamingProcessor instances
processing_status: Dict[str, Dict[str, Any]] = {}

# Connection state management to prevent race conditions
connection_states: Dict[str, Dict[str, Any]] = {}  # {config_name: {"status": "ready|cleaning|streaming", "last_cleanup": timestamp}}

# Memory corruption detection and auto-restart
MEMORY_CORRUPTION_DETECTED = False
RESTART_COUNT = 0
MAX_RESTARTS = 3

def detect_memory_corruption():
    """Detect memory corruption and trigger restart if needed"""
    global MEMORY_CORRUPTION_DETECTED, RESTART_COUNT
    
    if MEMORY_CORRUPTION_DETECTED:
        return True
    
    # Check for common memory corruption indicators
    try:
        # Try to allocate and free memory to test heap integrity
        test_data = [0] * 1000
        del test_data
        gc.collect()
        return False
    except Exception as e:
        print(f"Memory corruption detected: {e}")
        MEMORY_CORRUPTION_DETECTED = True
        return True

async def restart_server():
    """Restart the server automatically when memory corruption is detected"""
    global RESTART_COUNT
    
    if RESTART_COUNT >= MAX_RESTARTS:
        print(f"Maximum restart attempts ({MAX_RESTARTS}) reached. Manual intervention required.")
        return
    
    RESTART_COUNT += 1
    print(f"Memory corruption detected. Restarting server (attempt {RESTART_COUNT}/{MAX_RESTARTS})...")
    
    # Send restart notification to all connected clients
    for config_name in connection_states:
        if connection_states[config_name].get("status") == "streaming":
            try:
                # This will be handled by the WebSocket cleanup
                pass
            except Exception as e:
                print(f"Error notifying client {config_name}: {e}")
    
    # Wait a moment for cleanup
    await asyncio.sleep(2)
    
    # Restart the server
    try:
        # Get the current script path
        script_path = os.path.abspath(__file__)
        print(f"Restarting server: python {script_path}")
        
        # Start new server process
        subprocess.Popen([
            sys.executable, script_path
        ], cwd=os.path.dirname(script_path))
        
        # Exit current process
        os._exit(0)
        
    except Exception as e:
        print(f"Error restarting server: {e}")
        print("Manual restart required.")

async def set_connection_state(config_name: str, status: str):
    """Set connection state for a config to prevent race conditions"""
    if config_name not in connection_states:
        connection_states[config_name] = {}
    
    connection_states[config_name]["status"] = status
    connection_states[config_name]["last_update"] = time.time()
    
    print(f"Connection state for {config_name}: {status}")

async def wait_for_connection_ready(config_name: str, max_wait: float = 5.0):
    """Wait for connection to be ready, with timeout and stale connection handling"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        if config_name in connection_states:
            state = connection_states[config_name]
            current_time = time.time()
            last_update = state.get("last_update", 0)
            
            # If connection is ready, allow it
            if state.get("status") == "ready":
                return True
            
            # If connection is cleaning, wait for it
            elif state.get("status") == "cleaning":
                print(f"Waiting for {config_name} cleanup to complete...")
                await asyncio.sleep(0.1)
                continue
            
            # If connection is streaming but stale (older than 10 seconds), allow new connection
            elif state.get("status") == "streaming" and (current_time - last_update) > 10.0:
                print(f"Connection for {config_name} appears stale (last update: {current_time - last_update:.1f}s ago) - allowing new connection")
                await set_connection_state(config_name, "ready")
                return True
            
            # If connection is streaming and recent, wait a bit
            elif state.get("status") == "streaming":
                print(f"Connection for {config_name} is active - waiting...")
                await asyncio.sleep(0.5)
                continue
        
        await asyncio.sleep(0.1)
    
    print(f"Timeout waiting for {config_name} to be ready - forcing ready state")
    await set_connection_state(config_name, "ready")
    return True

# Video processor configurations matching the original process.py
PROCESSOR_CONFIGS = {
    "01": {
        "person_conf": 0.25,
        "iou_threshold": 0.5,
        "gender_conf": 0.5,
        "enable_color_heuristic": True,
        "show_segmentation": False
    },
    "02": {
        "person_conf": 0.25,
        "iou_threshold": 0.5,
        "gender_conf": 0.5,
        "enable_color_heuristic": True,
        "show_segmentation": False
    },
    "03": {
        "person_conf": 0.25,
        "iou_threshold": 0.5,
        "gender_conf": 0.5,
        "enable_color_heuristic": True,
        "show_segmentation": False
    },
    "04": {
        "person_conf": 0.25,
        "iou_threshold": 0.5,
        "gender_conf": 0.5,
        "enable_color_heuristic": True,
        "show_segmentation": False
    }
}


def initialize_processors():
    """Initialize all video processors with their specific configurations"""
    global video_processors, streaming_processors
    
    print("Initializing video processors...")
    for config_name, config in PROCESSOR_CONFIGS.items():
        print(f"Initializing processor for {config_name}...")
        
        # Create streaming processor (can handle both streaming and batch processing)
        streaming_processors[config_name] = UltraFastStreamingProcessor(
            person_conf=config["person_conf"],
            iou_threshold=config["iou_threshold"],
            gender_conf=config["gender_conf"],
            enable_color_heuristic=config["enable_color_heuristic"],
            show_segmentation=config["show_segmentation"]
        )
        
        # Use the same processor for both streaming and batch processing
        video_processors[config_name] = streaming_processors[config_name]
    print("All video processors initialized!")


@app.on_event("startup")
async def startup_event():
    """Initialize processors when the server starts"""
    initialize_processors()


class UltraFastStreamingProcessor(VideoProcessor):
    """Ultra-fast streaming processor with asynchronous pre-processing pipeline and zero-lag display"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_streaming = False
        self.show_segmentation = kwargs.get('show_segmentation', False)
        self.streaming_speed = 1.0  # Normal speed
        self.man_enabled = True  # Default both enabled
        self.woman_enabled = True
        
        # Asynchronous pipeline optimization
        self.frame_queue = asyncio.Queue(maxsize=20)  # Increased buffer for single worker
        self.detection_queue = asyncio.Queue(maxsize=30)  # Increased detection buffer
        self.processed_frames = {}  # Frame cache with detection results
        self.frame_workers = 1  # Single worker to prevent race conditions
        
        # Thread safety locks
        self._frame_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        
        # Cleanup control flag
        self._cleanup_disabled = False
        self._detection_disabled = False
        
        # Config name for connection state management
        self.config_name = None
        self.preload_seconds = 2  # Pre-load 2 seconds of detections
        
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
        
        # Frame preloading system
        self.preloaded_frames = {}  # {video_path: [frame1, frame2, ...]}
        self.preloaded_frame_count = {}  # {video_path: count}
        self.preloaded_fps = {}  # {video_path: fps}
    
    async def preload_video_frames(self, video_path: str):
        """Preload all video frames into memory for efficient looping"""
        if video_path in self.preloaded_frames:
            print(f"Video {video_path} already preloaded")
            return True
            
        print(f"Preloading video frames for {video_path}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video {video_path}")
            return False
        
        frames = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Store frame copy in memory
                frames.append(frame.copy())
                frame_count += 1
                
                # Progress indicator for long videos
                if frame_count % 100 == 0:
                    print(f"Preloaded {frame_count} frames...")
            
            # Store preloaded data
            self.preloaded_frames[video_path] = frames
            self.preloaded_frame_count[video_path] = frame_count
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.preloaded_fps[video_path] = fps
            
            print(f"Successfully preloaded {frame_count} frames at {fps} FPS for {video_path}")
            return True
            
        except Exception as e:
            print(f"Error preloading video {video_path}: {e}")
            return False
        finally:
            cap.release()
    
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
        """Ultra-fast streaming with preloaded frames"""
        # Check if video is preloaded globally
        if video_path not in GLOBAL_PRELOADED_FRAMES:
            await websocket.send_text(json.dumps({
                "type": "status", 
                "message": "Video not preloaded, please wait..."
            }))
            
            try:
                success = await preload_video_frames_global(video_path)
                if not success:
                    await websocket.send_text(json.dumps({"type": "error", "message": "Could not preload video"}))
                    return
            except Exception as e:
                print(f"Error preloading video {video_path}: {e}")
                await websocket.send_text(json.dumps({"type": "error", "message": f"Error preloading video: {str(e)}"}))
                return
        
        # Get preloaded video data from global storage
        frames = GLOBAL_PRELOADED_FRAMES[video_path]
        total_frames = GLOBAL_PRELOADED_FRAME_COUNT[video_path]
        fps = GLOBAL_PRELOADED_FPS[video_path]
        
        # Set up video properties from preloaded data
        self.video_fps = fps
        self.target_fps = fps
        self.frame_interval = 1.0 / fps
        self.adaptive_fps = min(fps, 60)  # Cap at 60 FPS for smoothness
        self.max_fps = min(fps * 2, 120)  # Max 2x video FPS
        
        # Get frame dimensions from first frame
        if frames:
            height, width = frames[0].shape[:2]
        else:
            await websocket.send_text(json.dumps({"type": "error", "message": "No frames available"}))
            return
        
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
        
        # Only clean up if there are existing resources to clean (video switch scenario)
        if self.is_streaming or len(self.processed_frames) > 0 or not self.frame_queue.empty():
            await self._comprehensive_cleanup(reset_tracker=True, config_name=self.config_name)
        
        self.is_streaming = True
        self.current_frame_id = 0
        
        # Reset optimization counters
        self.frames_skipped = 0
        self.consecutive_skips = 0
        self.processing_times.clear()
        self.processed_frames.clear()
        
        # Start asynchronous pipeline
        await self._start_async_pipeline(frames, websocket)
    
    async def _start_async_pipeline(self, frames, websocket):
        """Start the asynchronous processing pipeline"""
        try:
            # Start parallel detection workers
            detection_tasks = []
            for i in range(self.frame_workers):
                task = asyncio.create_task(self._detection_worker(f"worker-{i}"))
                detection_tasks.append(task)
            
            # Start frame reader task with preloaded frames
            reader_task = asyncio.create_task(self._frame_reader_preloaded(frames, websocket))
            
            # Start frame sender task
            sender_task = asyncio.create_task(self._frame_sender(websocket))
            
            # Wait for all tasks (no periodic cleanup needed - comprehensive cleanup handles everything)
            await asyncio.gather(reader_task, sender_task, *detection_tasks)
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        finally:
            self.is_streaming = False
            await self._comprehensive_cleanup(reset_tracker=False, config_name=self.config_name)
    
    async def _frame_reader(self, cap, websocket):
        """Continuously read frames and add to queue with looping support"""
        frame_id = 0
        loop_count = 0
        
        consecutive_failures = 0
        max_consecutive_failures = 10
        
        while self.is_streaming:
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > max_consecutive_failures:
                    print(f"Too many consecutive frame failures ({consecutive_failures}), restarting video...")
                    # Reset video to beginning
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    consecutive_failures = 0
                    continue
                # Check if this is actually the end of video or a corrupted frame
                current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                
                # If we're near the end of the video (within 5 frames), consider it end of video
                if current_pos >= total_frames - 5 or total_frames <= 0:
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
                else:
                    # This might be a corrupted frame, try to skip it
                    print(f"Warning: Corrupted frame at position {current_pos}/{total_frames}, skipping...")
                    # Try to skip to next frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos + 1)
                    
                    # Add a small delay to prevent rapid retries
                    await asyncio.sleep(0.01)
                    continue
                
                # Clear processed frames cache for new loop with proper cleanup
                await self._comprehensive_cleanup(reset_tracker=True, config_name=self.config_name)
                
                await asyncio.sleep(0.1)
                continue
            
            frame_id += 1
            consecutive_failures = 0  # Reset failure counter on successful read
            
            # Validate frame before processing
            if frame is None or frame.size == 0:
                print(f"Warning: Invalid frame at position {frame_id}, skipping...")
                continue
            
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
    
    async def _frame_reader_preloaded(self, frames, websocket):
        """Read frames from preloaded memory instead of file"""
        try:
            frame_id = 0
            loop_count = 0
            total_frames = len(frames)
            
            while self.is_streaming:
                # Get frame from preloaded memory
                if frame_id >= total_frames:
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
                    
                    # Reset frame counter (no need to reload frames!)
                    frame_id = 0
                    
                    # Clean up old processed frames and reset tracker for new loop
                    await self._comprehensive_cleanup(reset_tracker=True, config_name=self.config_name)
                    
                    # Check for memory corruption after cleanup
                    if detect_memory_corruption():
                        print("Memory corruption detected during loop restart - triggering server restart")
                        await restart_server()
                        return
                    
                    await asyncio.sleep(0.1)
                    continue
            
                # Get frame from preloaded memory
                frame = frames[frame_id]
                frame_id += 1
                
                # Validate frame before processing
                if frame is None or frame.size == 0:
                    print(f"Warning: Invalid frame at position {frame_id}, skipping...")
                    continue
                
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
                
        except Exception as e:
            print(f"Frame reader error: {e}")
            print(f"Error type: {type(e).__name__}")
            # Check if this is a memory corruption error
            if "corrupted" in str(e).lower() or "double free" in str(e).lower() or "malloc" in str(e).lower():
                print("Memory corruption detected in frame reader - triggering server restart")
                detect_memory_corruption()  # Set the flag
            raise
    
    async def _detection_worker(self, worker_name):
        """Worker that processes frames for detection"""
        try:
            while self.is_streaming:
                try:
                    # Skip processing if detection is disabled (during video switches)
                    if self._detection_disabled:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Get frame from queue with timeout
                    frame_data = await asyncio.wait_for(self.frame_queue.get(), timeout=1.0)
                    frame_id = frame_data['frame_id']
                    frame = frame_data['frame']
                    
                    # Process frame for detection
                    loop = asyncio.get_event_loop()
                    processed_frame, active_tracks, detected_genders = await loop.run_in_executor(
                        None, self._process_frame_safe, frame, frame_id
                    )
                    
                    # Clean up the original frame data immediately after processing
                    del frame_data['frame']
                
                    # Store processed frame with thread safety
                    async with self._frame_lock:
                        self.processed_frames[frame_id] = {
                            'processed_frame': processed_frame,
                            'active_tracks': active_tracks,
                            'detected_genders': detected_genders,
                            'timestamp': time.time(),
                            'loop_count': frame_data.get('loop_count', 0),
                            'sent': False
                        }
                        
                        # Clean up old frames (keep last 30) with proper memory cleanup
                        # Skip cleanup if disabled (during transitions)
                        if not self._cleanup_disabled and len(self.processed_frames) > 30:
                            # Clean up multiple old frames at once
                            frames_to_remove = []
                            for frame_id, frame_data in self.processed_frames.items():
                                if isinstance(frame_data, dict):
                                    # Remove frames that have been sent or are too old
                                    if (frame_data.get('sent', False) or 
                                        (time.time() - frame_data.get('timestamp', 0)) > 5.0):
                                        frames_to_remove.append(frame_id)
                            
                            # Remove old frames
                            for frame_id in frames_to_remove[:10]:  # Remove up to 10 frames at once
                                if frame_id in self.processed_frames:
                                    frame_data = self.processed_frames[frame_id]
                                    if isinstance(frame_data, dict):
                                        if 'processed_frame' in frame_data:
                                            del frame_data['processed_frame']
                                        if 'active_tracks' in frame_data:
                                            del frame_data['active_tracks']
                                        if 'detected_genders' in frame_data:
                                            del frame_data['detected_genders']
                                    del self.processed_frames[frame_id]
                
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"Detection worker {worker_name} error: {e}")
                    await asyncio.sleep(0.1)
                    continue
                    
        except Exception as e:
            print(f"Detection worker {worker_name} outer error: {e}")
    
    async def _comprehensive_cleanup(self, reset_tracker=True, config_name=None):
        """Single comprehensive cleanup method for both loop restart and video switch - non-blocking approach"""
        try:
            # Set connection state to cleaning if config_name provided
            if config_name:
                await set_connection_state(config_name, "cleaning")
            
            # Check if there's anything to clean up
            has_processed_frames = len(self.processed_frames) > 0
            has_frame_queue = not self.frame_queue.empty()
            has_detection_queue = not self.detection_queue.empty()
            has_tracks = hasattr(self, 'tracker') and self.tracker and len(self.tracker.tracks) > 0
            
            if not (has_processed_frames or has_frame_queue or has_detection_queue or has_tracks):
                print("No resources to clean up, skipping cleanup")
                return
            
            # Disable detection worker cleanup to prevent race conditions
            self._cleanup_disabled = True
            
            # For video switches, also disable detection workers to prevent new tracks
            if config_name and reset_tracker:
                self._detection_disabled = True
            
            print("Starting comprehensive cleanup...")
            
            # 1. Reset tracker state (tracks, track_history, next_track_id) only if there are tracks to reset
            if reset_tracker and has_tracks:
                self.tracker.reset_tracker()
            
            # 2. Clear processed frames and free all frame copies (non-blocking)
            if has_processed_frames:
                print(f"Clearing {len(self.processed_frames)} processed frames...")
                # Clear frames without acquiring locks to avoid blocking
                for frame_id, frame_data in self.processed_frames.items():
                    if isinstance(frame_data, dict):
                        # Clear all frame copies to free memory
                        if 'processed_frame' in frame_data:
                            del frame_data['processed_frame']
                        if 'active_tracks' in frame_data:
                            del frame_data['active_tracks']
                        if 'detected_genders' in frame_data:
                            del frame_data['detected_genders']
                self.processed_frames.clear()
            
            # 3. Clear frame queue and free all frame copies (non-blocking)
            if has_frame_queue:
                print("Clearing frame queue...")
                frames_cleared = 0
                while not self.frame_queue.empty():
                    try:
                        frame_data = self.frame_queue.get_nowait()
                        if isinstance(frame_data, dict) and 'frame' in frame_data:
                            del frame_data['frame']  # Free frame copy
                            frames_cleared += 1
                    except asyncio.QueueEmpty:
                        break
                print(f"Cleared {frames_cleared} frames from queue")
            
            # 4. Clear detection queue (non-blocking)
            if has_detection_queue:
                print("Clearing detection queue...")
                detections_cleared = 0
                while not self.detection_queue.empty():
                    try:
                        self.detection_queue.get_nowait()
                        detections_cleared += 1
                    except asyncio.QueueEmpty:
                        break
                print(f"Cleared {detections_cleared} detections from queue")
            
            # 5. Force garbage collection
            import gc
            gc.collect()
            
            print("Comprehensive cleanup completed successfully")
            
        except Exception as e:
            print(f"Error during comprehensive cleanup: {e}")
            import gc
            gc.collect()
        finally:
            # Re-enable detection worker cleanup
            self._cleanup_disabled = False
            
            # Re-enable detection workers
            self._detection_disabled = False
            
            # Set connection state to ready if config_name provided
            if config_name:
                await set_connection_state(config_name, "ready")
    
    async def _cleanup_sent_frame(self, frame_id):
        """Clean up frame data after it has been sent to prevent memory accumulation"""
        async with self._frame_lock:
            if frame_id in self.processed_frames:
                frame_data = self.processed_frames[frame_id]
                if isinstance(frame_data, dict):
                    # Clear frame data to free memory
                    if 'processed_frame' in frame_data:
                        del frame_data['processed_frame']
                    if 'active_tracks' in frame_data:
                        del frame_data['active_tracks']
                    if 'detected_genders' in frame_data:
                        del frame_data['detected_genders']
                    # Keep metadata but clear heavy data
                    frame_data['sent'] = True
                    frame_data['sent_timestamp'] = time.time()
    
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
                
                # Look for next frame to send
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
                    
                    # Look for the next frame in sequence, but also check for frame 1 of new loop
                    next_frame_id = last_sent_frame + 1
                    if next_frame_id in self.processed_frames:
                        frame_to_send = self.processed_frames[next_frame_id]
                        last_sent_frame = next_frame_id
                    elif current_loop_count > 0 and 1 in self.processed_frames:
                        # If we're in a new loop and frame 1 is available, send it
                        frame_to_send = self.processed_frames[1]
                        last_sent_frame = 1
                
                if frame_to_send is None:
                    # No processed frame available, wait a bit
                    await asyncio.sleep(0.01)
                    
                    # Check for memory corruption periodically
                    if detect_memory_corruption():
                        print("Memory corruption detected in frame sender - triggering server restart")
                        await restart_server()
                        return
                    
                    continue
                
                # Update timing
                last_frame_time = current_time
                
                # Get current detection data for real-time synchronization
                current_detection_data = None
                if hasattr(self, 'tracker') and self.tracker:
                    # For gender detection, we can get current active tracks
                    current_detection_data = {
                        'active_tracks': len([t for t in self.tracker.tracks.values() if t['age'] > 0]),
                        'detected_genders': frame_to_send.get('detected_genders', [])
                    }
                
                # Use current detection data if available, otherwise use frame data
                active_tracks_to_send = current_detection_data['active_tracks'] if current_detection_data else frame_to_send['active_tracks']
                detected_genders_to_send = current_detection_data['detected_genders'] if current_detection_data else frame_to_send.get('detected_genders', [])
                
                # Encode and send frame
                encoded_frame = None
                try:
                    if self.use_binary_transmission:
                        encoded_frame = await loop.run_in_executor(
                            None, self._encode_frame_binary, frame_to_send['processed_frame']
                        )
                        
                        if encoded_frame:
                            await websocket.send_bytes(encoded_frame)
                            
                            metadata = {
                                "type": "frame_metadata",
                                "frame_count": last_sent_frame,
                                "active_tracks": active_tracks_to_send,
                                "segmentation": self.show_segmentation,
                                "detected_genders": detected_genders_to_send,
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
                                "segmentation": self.show_segmentation,
                                "detected_genders": frame_to_send['detected_genders'],
                                "adaptive_fps": self.adaptive_fps,
                                "frames_skipped": self.frames_skipped,
                                "pipeline_mode": "ultra_fast"
                            }
                            await websocket.send_text(json.dumps(message))
                finally:
                    # Explicitly clean up encoded frame data
                    if encoded_frame is not None:
                        del encoded_frame
                
                # Clean up the frame data after sending to prevent memory accumulation
                await self._cleanup_sent_frame(last_sent_frame)
                
                # Dynamic frame delay
                frame_delay = (1.0 / self.adaptive_fps) / self.streaming_speed
                await asyncio.sleep(frame_delay)
                
            except Exception as e:
                print(f"Frame sender error: {e}")
                await asyncio.sleep(0.1)
    
    def _process_frame_safe(self, frame, frame_count):
        """Safely process frame with error handling and memory management"""
        try:
            # Validate input frame
            if frame is None or frame.size == 0:
                return None, 0, []
            
            # Create a copy to prevent memory corruption
            frame_copy = frame.copy()
            
            # Update renderer segmentation setting
            if hasattr(self, 'renderer') and self.renderer is not None:
                self.renderer.show_segmentation = self.show_segmentation
            
            # Process frame with error handling
            try:
                # Check if tracker is in a valid state before processing
                if not hasattr(self, 'tracker') or self.tracker is None:
                    print(f"Frame processing error: Tracker not available at count {frame_count}")
                    del frame_copy
                    return frame.copy() if frame is not None else None, 0, []
                
                processed_frame, active_tracks = self.process_frame(frame_copy, frame_count)
            except KeyError as e:
                print(f"Frame processing KeyError in process_frame: {e}")
                print(f"Frame count: {frame_count}")
                print(f"Frame shape: {frame.shape if frame is not None else 'None'}")
                print(f"Tracker tracks count: {len(self.tracker.tracks) if hasattr(self, 'tracker') and self.tracker else 'N/A'}")
                
                # Check if the missing key is a track ID
                missing_key = str(e).strip("'\"")
                if missing_key.isdigit():
                    track_id = int(missing_key)
                    print(f"Missing track ID: {track_id}")
                    if hasattr(self, 'tracker') and self.tracker:
                        # Check if track was recently deleted
                        if track_id not in self.tracker.tracks:
                            print(f"Track {track_id} was deleted - this is a race condition")
                            # Try to continue without this track
                            del frame_copy
                            return frame.copy() if frame is not None else None, 0, []
                
                # Try to identify which track is causing the issue
                if hasattr(self, 'tracker') and self.tracker:
                    track_ids = sorted(self.tracker.tracks.keys())
                    print(f"Available track IDs: {track_ids[:10]}...")  # Show first 10
                    if len(track_ids) > 10:
                        print(f"Total tracks: {len(track_ids)}")
                
                del frame_copy
                return frame.copy() if frame is not None else None, 0, []
            except Exception as e:
                print(f"Frame processing error in process_frame: {e}")
                print(f"Error type: {type(e).__name__}")
                print(f"Frame count: {frame_count}")
                print(f"Frame shape: {frame.shape if frame is not None else 'None'}")
                del frame_copy
                return frame.copy() if frame is not None else None, 0, []
            
            # Extract detected genders with error handling
            try:
                detected_genders = self._extract_detected_genders()
            except Exception as e:
                print(f"Frame processing error in _extract_detected_genders: {e}")
                detected_genders = []
            
            # Ensure we return a copy to prevent memory corruption
            result_frame = processed_frame.copy() if processed_frame is not None else frame_copy.copy()
            
            # Clean up local references
            del frame_copy
            if processed_frame is not None:
                del processed_frame
            
            return result_frame, active_tracks, detected_genders
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            
            # Check if this is a memory corruption error
            if "corrupted" in str(e).lower() or "double free" in str(e).lower() or "malloc" in str(e).lower():
                print("Memory corruption detected in frame processing - triggering server restart")
                detect_memory_corruption()  # Set the flag
                # Note: We can't await restart_server() here as this is not an async context
                # The restart will be triggered from the main loop
            
            # Return original frame as fallback
            return frame.copy() if frame is not None else None, 0, []
    
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
    
    def _extract_detected_genders(self):
        """Extract unique detected genders from current tracked objects"""
        if not hasattr(self, 'tracker') or not self.tracker:
            return []
        
        detected_genders = set()
        try:
            # Create a copy of track IDs to avoid modification during iteration
            track_ids = list(self.tracker.tracks.keys())
            for track_id in track_ids:
                try:
                    if track_id in self.tracker.tracks:
                        track_data = self.tracker.tracks[track_id]
                        if isinstance(track_data, dict) and 'gender' in track_data and track_data['gender'] != 'Unknown':
                            detected_genders.add(track_data['gender'])
                except KeyError:
                    # Track was deleted during iteration - skip it
                    print(f"Track {track_id} was deleted during gender extraction - skipping")
                    continue
        except Exception as e:
            print(f"Error extracting detected genders: {e}")
            return []
        
        return list(detected_genders)
    
    def update_gender_switches(self, man_enabled: bool, woman_enabled: bool):
        """Update gender switch settings"""
        self.man_enabled = man_enabled
        self.woman_enabled = woman_enabled
        print(f"Gender switches updated - Man: {man_enabled}, Woman: {woman_enabled}")


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
            "message": "Gender Detection Video Processing API",
            "version": "1.0.0",
            "available_endpoints": [
                "/process-video/{config_name}",
                "/process-custom-video",
                "/status/{job_id}",
                "/download/{job_id}",
                "/configs"
            ],
            "note": "Frontend not found. API endpoints available at /docs"
        }

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Gender Detection Video Processing API",
        "version": "1.0.0",
        "available_endpoints": [
            "/process-video/{config_name}",
            "/process-custom-video",
            "/status/{job_id}",
            "/download/{job_id}",
            "/configs"
        ]
    }


@app.get("/configs")
async def get_configs():
    """Get available processor configurations"""
    return {
        "available_configs": list(PROCESSOR_CONFIGS.keys()),
        "configurations": PROCESSOR_CONFIGS
    }


@app.post("/process-video/{config_name}")
async def process_video_with_config(
    config_name: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Process an uploaded video using a specific processor configuration
    """
    if config_name not in PROCESSOR_CONFIGS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid config name. Available configs: {list(PROCESSOR_CONFIGS.keys())}"
        )
    
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
        "config_name": config_name,
        "filename": file.filename,
        "output_path": None
    }
    
    # Create custom processor with the specified configuration
    config = PROCESSOR_CONFIGS[config_name]
    processor = CustomVideoProcessor(
        person_conf=config["person_conf"],
        iou_threshold=config["iou_threshold"],
        gender_conf=config["gender_conf"],
        enable_color_heuristic=config["enable_color_heuristic"],
        show_segmentation=config["show_segmentation"]
    )
    
    # Start processing in background
    background_tasks.add_task(
        process_video_async, 
        processor, 
        str(input_path), 
        str(output_path), 
        job_id
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Video processing started with config '{config_name}'",
        "config_used": config
    }


@app.post("/process-custom-video")
async def process_custom_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    person_conf: float = Form(0.25),
    iou_threshold: float = Form(0.5),
    gender_conf: float = Form(0.5),
    enable_color_heuristic: bool = Form(True),
    show_segmentation: bool = Form(True)
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
        person_conf=person_conf,
        iou_threshold=iou_threshold,
        gender_conf=gender_conf,
        enable_color_heuristic=enable_color_heuristic,
        show_segmentation=show_segmentation
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
        "person_conf": person_conf,
        "iou_threshold": iou_threshold,
        "gender_conf": gender_conf,
        "enable_color_heuristic": enable_color_heuristic,
        "show_segmentation": show_segmentation
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


@app.websocket("/ws/stream-video/{config_name}")
async def websocket_stream_video(websocket: WebSocket, config_name: str):
    """Clean WebSocket endpoint for video streaming with connection state management"""
    await websocket.accept()
    
    if config_name not in PROCESSOR_CONFIGS:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": f"Invalid config name. Available: {list(PROCESSOR_CONFIGS.keys())}"
        }))
        await websocket.close()
        return
    
    # Wait for connection to be ready to prevent race conditions
    
    if not await wait_for_connection_ready(config_name, max_wait=3.0):
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": f"Server busy - please try again in a moment"
        }))
        await websocket.close()
        return
    
    # Set connection state to streaming
    await set_connection_state(config_name, "streaming")
    
    # Start periodic state update to keep connection fresh
    async def update_connection_state():
        while True:
            try:
                await asyncio.sleep(5.0)  # Update every 5 seconds
                if config_name in connection_states and connection_states[config_name].get("status") == "streaming":
                    await set_connection_state(config_name, "streaming")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error updating connection state: {e}")
                break
    
    state_update_task = asyncio.create_task(update_connection_state())
    
    # Connection cleanup will be handled in the main task completion
    
    # Video file mapping
    video_mapping = {
        '01': 'data/inputs/01.mp4',
        '02': 'data/inputs/02.mp4',
        '03': 'data/inputs/03.mp4',
        '04': 'data/inputs/04.mp4'
    }
    
    if config_name not in video_mapping:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": f"Video not found for config: {config_name}"
        }))
        await websocket.close()
        return
    
    video_file_path = video_mapping[config_name]  # Use relative path to match startup preloading
    processor = None
    
    try:
        # Reuse existing streaming processor instead of creating new one
        if config_name not in streaming_processors:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": f"Streaming processor not initialized for config: {config_name}"
            }))
            await websocket.close()
            return
        
        processor = streaming_processors[config_name]
        
        # Set config name for connection state management
        processor.config_name = config_name
        
        # Check if video file exists
        if not os.path.exists(video_file_path):
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": f"Video file not found: {video_file_path}"
            }))
            await websocket.close()
            return
        
        # Start streaming and control handling concurrently
        try:
            # Add timeout to stream_video to prevent hanging
            async def stream_video_with_timeout():
                try:
                    await asyncio.wait_for(processor.stream_video(video_file_path, websocket), timeout=60.0)
                except asyncio.TimeoutError:
                    await websocket.close()
                    raise
            
            streaming_task = asyncio.create_task(stream_video_with_timeout())
            control_task = asyncio.create_task(handle_stream_controls(websocket, processor))
            
            # Add a timeout to detect hanging tasks
            done, pending = await asyncio.wait(
                [streaming_task, control_task, state_update_task],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=30.0  # 30 second timeout for single worker
            )
            
            if not done:
                # Cancel all tasks
                for task in [streaming_task, control_task, state_update_task]:
                    task.cancel()
                await websocket.close()
                return
            
            # Check if streaming task failed
            if streaming_task in done:
                try:
                    await streaming_task
                except Exception as e:
                    pass
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error", 
                "message": f"Failed to start streaming: {str(e)}"
            }))
            await websocket.close()
            return
        
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
        
        # Clean up connection state
        print(f"WebSocket closed for {config_name} - cleaning up connection state")
        await set_connection_state(config_name, "ready")
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
                        
                    elif command == "update_gender_switches":
                        switch_data = data.get("value", {})
                        man_enabled = bool(switch_data.get("man_enabled", True))
                        woman_enabled = bool(switch_data.get("woman_enabled", True))
                        processor.update_gender_switches(man_enabled, woman_enabled)
                        await websocket.send_text(json.dumps({
                            "type": "control_response", 
                            "message": f"Gender switches updated - Man: {man_enabled}, Woman: {woman_enabled}"
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "processors_loaded": len(video_processors),
        "streaming_processors_loaded": len(streaming_processors),
        "active_jobs": len(processing_status)
    }


# Global preloaded frames storage
GLOBAL_PRELOADED_FRAMES = {}
GLOBAL_PRELOADED_FRAME_COUNT = {}
GLOBAL_PRELOADED_FPS = {}

@app.on_event("startup")
async def startup_event():
    """Preload all video frames at startup for better performance"""
    print("=" * 50)
    print("SERVER STARTUP: Starting video preloading...")
    print("=" * 50)
    
    # Get all video paths
    video_mapping = {
        '01': 'data/inputs/01.mp4',
        '02': 'data/inputs/02.mp4', 
        '03': 'data/inputs/03.mp4',
        '04': 'data/inputs/04.mp4'
    }
    
    # Preload all videos sequentially to avoid memory issues
    for config_name, video_path in video_mapping.items():
        print(f"\n--- Preloading video {config_name} ({video_path}) ---")
        
        # Check if file exists with absolute path
        abs_path = os.path.abspath(video_path)
        print(f"Checking path: {abs_path}")
        
        if os.path.exists(video_path):
            print(f"✅ Video file found: {video_path}")
            try:
                success = await preload_video_frames_global(video_path)
                if success:
                    print(f"✅ Successfully preloaded video {config_name}")
                else:
                    print(f"❌ Failed to preload video {config_name}")
            except Exception as e:
                print(f"❌ Error preloading video {config_name}: {e}")
        else:
            print(f"❌ Video file not found: {video_path}")
            print(f"❌ Absolute path: {abs_path}")
            print(f"❌ Current working directory: {os.getcwd()}")
        
        # Small delay between videos to prevent memory pressure
        await asyncio.sleep(0.5)
    
    print("\n" + "=" * 50)
    print("SERVER STARTUP: Video preloading completed!")
    print(f"Preloaded videos: {list(GLOBAL_PRELOADED_FRAMES.keys())}")
    print("=" * 50)
    
    # Initialize connection states for all configs
    for config_name in PROCESSOR_CONFIGS.keys():
        await set_connection_state(config_name, "ready")
        print(f"Initialized connection state for {config_name}: ready")

async def preload_video_frames_global(video_path: str):
    """Preload video frames into global storage with improved memory management"""
    if video_path in GLOBAL_PRELOADED_FRAMES:
        print(f"Video {video_path} already preloaded")
        return True
        
    print(f"Preloading video frames for {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video {video_path}")
        return False
    
    frames = []
    frame_count = 0
    
    try:
        # Get video properties first
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_estimate = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Pre-allocate list with estimated size to reduce memory fragmentation
        if total_frames_estimate > 0:
            frames = [None] * total_frames_estimate
        
        # Reset to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Validate frame before storing
            if frame is not None and frame.size > 0:
                # Store frame copy in memory
                frames[frame_count] = frame.copy()
                frame_count += 1
                
                # Progress indicator for long videos
                if frame_count % 100 == 0:
                    print(f"Preloaded {frame_count} frames...")
                    
                # Force garbage collection every 500 frames to prevent memory buildup
                if frame_count % 500 == 0:
                    import gc
                    gc.collect()
            else:
                print(f"Warning: Skipping invalid frame at position {frame_count}")
        
        # Trim the list to actual size
        frames = frames[:frame_count]
        
        # Store preloaded data globally
        GLOBAL_PRELOADED_FRAMES[video_path] = frames
        GLOBAL_PRELOADED_FRAME_COUNT[video_path] = frame_count
        GLOBAL_PRELOADED_FPS[video_path] = fps
        
        # Final garbage collection
        import gc
        gc.collect()
        
        print(f"Successfully preloaded {frame_count} frames at {fps} FPS for {video_path}")
        return True
        
    except Exception as e:
        print(f"Error preloading video {video_path}: {e}")
        # Clean up on error
        if frames:
            for frame in frames:
                if frame is not None:
                    del frame
            del frames
        import gc
        gc.collect()
        return False
    finally:
        cap.release()

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
