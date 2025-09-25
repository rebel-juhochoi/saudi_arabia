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
processing_status: Dict[str, Dict[str, Any]] = {}

# Video processor configurations matching the original process.py
PROCESSOR_CONFIGS = {
    "01": {
        "person_conf": 0.25,
        "iou_threshold": 0.5,
        "gender_conf": 0.5,
        "enable_color_heuristic": True,
        "show_segmentation": True
    },
    "02": {
        "person_conf": 0.25,
        "iou_threshold": 0.5,
        "gender_conf": 0.5,
        "enable_color_heuristic": True,
        "show_segmentation": True
    },
    "03": {
        "person_conf": 0.25,
        "iou_threshold": 0.5,
        "gender_conf": 0.5,
        "enable_color_heuristic": True,
        "show_segmentation": True
    },
    "04": {
        "person_conf": 0.25,
        "iou_threshold": 0.5,
        "gender_conf": 0.5,
        "enable_color_heuristic": True,
        "show_segmentation": True
    }
}


def initialize_processors():
    """Initialize all video processors with their specific configurations"""
    global video_processors
    
    print("Initializing video processors...")
    for config_name, config in PROCESSOR_CONFIGS.items():
        print(f"Initializing processor for {config_name}...")
        video_processors[config_name] = VideoProcessor(
            person_conf=config["person_conf"],
            iou_threshold=config["iou_threshold"],
            gender_conf=config["gender_conf"],
            enable_color_heuristic=config["enable_color_heuristic"],
            show_segmentation=config["show_segmentation"]
        )
    print("All video processors initialized!")


@app.on_event("startup")
async def startup_event():
    """Initialize processors when the server starts"""
    initialize_processors()


class SimpleStreamingProcessor(VideoProcessor):
    """Clean, simple streaming video processor"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_streaming = False
        self.show_segmentation = kwargs.get('show_segmentation', False)
        self.streaming_speed = 1.0  # Normal speed
        self.man_enabled = True  # Default both enabled
        self.woman_enabled = True
        
    async def stream_video(self, video_path: str, websocket: WebSocket):
        """Main streaming method - simple and clean"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            await websocket.send_text(json.dumps({"type": "error", "message": "Could not open video"}))
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Send video info to frontend
        await websocket.send_text(json.dumps({
            "type": "video_info",
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames
        }))
        
        self.is_streaming = True
        frame_count = 0
        frames_to_skip = fps  # Skip first second
        
        # Calculate frame timing for controlled playback
        target_fps = 25  # Target 25 FPS for smooth real-time viewing
        frame_delay = (1.0 / target_fps) / self.streaming_speed
        
        try:
            while self.is_streaming:
                ret, frame = cap.read()
                
                if not ret:
                    # End of video - restart for looping
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    await asyncio.sleep(0.1)  # Brief pause before restart
                    continue
                
                frame_count += 1
                
                # Get event loop once
                loop = asyncio.get_event_loop()
                
                # Process frame if past skip period
                if frame_count > frames_to_skip:
                    # Run inference in thread pool to avoid blocking
                    processed_frame, active_tracks, detected_genders = await loop.run_in_executor(
                        None, self._process_frame_safe, frame, frame_count, frames_to_skip
                    )
                else:
                    processed_frame = frame
                    active_tracks = 0
                    detected_genders = []
                
                # Encode frame to base64
                encoded_frame = await loop.run_in_executor(
                    None, self._encode_frame, processed_frame
                )
                
                if encoded_frame:
                    # Send frame to frontend
                    message = {
                        "type": "frame",
                        "frame": encoded_frame,
                        "frame_count": frame_count,
                        "active_tracks": active_tracks,
                        "segmentation": self.show_segmentation,
                        "detected_genders": detected_genders
                    }
                    
                    await websocket.send_text(json.dumps(message))
                
                # Control playback speed
                await asyncio.sleep(frame_delay)
                
        except Exception as e:
            print(f"Streaming error: {e}")
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        
        finally:
            self.is_streaming = False
            cap.release()
            print("Video streaming stopped")
    
    def _process_frame_safe(self, frame, frame_count, frames_to_skip):
        """Safely process frame with error handling"""
        try:
            # Update renderer segmentation setting
            if hasattr(self, 'renderer') and self.renderer is not None:
                self.renderer.show_segmentation = self.show_segmentation
            
            # Process frame
            processed_frame, active_tracks = self.process_frame(frame, frame_count, frames_to_skip)
            
            # Extract detected genders from tracked objects
            detected_genders = self._extract_detected_genders()
            
            return processed_frame, active_tracks, detected_genders
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame, 0, []
    
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
    
    def toggle_segmentation(self, enabled: bool):
        """Toggle segmentation display"""
        self.show_segmentation = enabled
        print(f"Segmentation {'enabled' if enabled else 'disabled'}")
    
    def _extract_detected_genders(self):
        """Extract unique detected genders from current tracked objects"""
        if not hasattr(self, 'tracker') or not self.tracker:
            return []
        
        detected_genders = set()
        for track_id, track_data in self.tracker.tracks.items():
            if 'gender' in track_data and track_data['gender'] != 'Unknown':
                detected_genders.add(track_data['gender'])
        
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
            
            # Calculate frames to skip (1 second)
            frames_to_skip = fps
            
            # Process frames
            frame_count = 0
            last_progress = 0
            total_tracks = 0
            
            processing_status[job_id]["message"] = f"Processing video... (skipping first {frames_to_skip} frames)"
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                
                # Process frame
                annotated_frame, active_tracks = self.process_frame(frame, frame_count, frames_to_skip)
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
    """Clean WebSocket endpoint for video streaming"""
    await websocket.accept()
    
    if config_name not in PROCESSOR_CONFIGS:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": f"Invalid config name. Available: {list(PROCESSOR_CONFIGS.keys())}"
        }))
        await websocket.close()
        return
    
    # Video file mapping
    video_mapping = {
        '01': 'inputs/01.mp4',
        '02': 'inputs/02.mp4',
        '03': 'inputs/03.mp4',
        '04': 'inputs/04.mp4'
    }
    
    if config_name not in video_mapping:
        await websocket.send_text(json.dumps({
            "type": "error", 
            "message": f"Video not found for config: {config_name}"
        }))
        await websocket.close()
        return
    
    video_file_path = str(Path(__file__).parent / "data" / video_mapping[config_name])
    processor = None
    
    try:
        # Create simple streaming processor
        config = PROCESSOR_CONFIGS[config_name]
        processor = SimpleStreamingProcessor(
            person_conf=config["person_conf"],
            iou_threshold=config["iou_threshold"],
            gender_conf=config["gender_conf"],
            enable_color_heuristic=config["enable_color_heuristic"],
            show_segmentation=config["show_segmentation"]
        )
        
        print(f"Starting video stream for {config_name}")
        
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


async def handle_stream_controls(websocket: WebSocket, processor: SimpleStreamingProcessor):
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
        "active_jobs": len(processing_status)
    }


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
