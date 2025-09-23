from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional
import time
from contextlib import asynccontextmanager
import sys
import os

# Add the src directory to the path
sys.path.append('/workspace/projects/global/saudi_arabia/gender_detection/src')

from triton_client_http import TritonHTTPClient
from utils import Renderer

# Global variables
triton_client = None
video_cache = {}
current_video = None
video_processors = {}
is_processing = False
renderer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global triton_client, video_cache, renderer
    print("🚀 Starting Gender Detection Backend with Triton...")
    
    # Initialize Triton client
    try:
        triton_client = TritonHTTPClient("localhost:8002")
        
        # Wait for server to be ready
        max_retries = 30
        for i in range(max_retries):
            if triton_client.is_server_ready():
                print("✅ Triton server is ready")
                break
            print(f"⏳ Waiting for Triton server... ({i+1}/{max_retries})")
            await asyncio.sleep(2)
        else:
            print("❌ Triton server not ready after 60 seconds")
            triton_client = None
    except Exception as e:
        print(f"❌ Failed to initialize Triton client: {e}")
        triton_client = None
    
    # Initialize renderer
    renderer = Renderer(show_segmentation=True)
    
    # Pre-load video files
    video_cache = {}
    video_dir = Path("/workspace/projects/global/saudi_arabia/gender_detection/data/inputs")
    for video_file in video_dir.glob("*.mp4"):
        video_name = video_file.stem
        video_cache[video_name] = str(video_file)
        print(f"📁 Loaded video: {video_name}")
    
    print("🎬 Backend ready!")
    yield
    
    # Shutdown
    print("🛑 Shutting down backend...")

app = FastAPI(
    title="Gender Detection API",
    description="Real-time gender detection with video streaming using Triton Inference Server",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Gender Detection API is running!", "triton_ready": triton_client is not None and triton_client.is_server_ready()}

@app.get("/api/videos")
async def get_available_videos():
    """Get list of available video files"""
    videos = []
    for video_name in video_cache.keys():
        # Extract video number and clean name
        if video_name.startswith(('01_', '02_', '03_', '04_', '05_')):
            video_num = video_name[:2]
            clean_name = video_name[3:]
            videos.append({
                "value": video_num,
                "label": clean_name.title(),
                "full_name": video_name
            })
    
    return {"videos": videos}

@app.post("/api/start-tracking")
async def start_tracking(request: Dict):
    """Start tracking for a specific video"""
    global current_video, is_processing
    
    video_num = request.get("video_num")
    if not video_num:
        raise HTTPException(status_code=400, detail="video_num is required")
    
    # Find video by number
    video_name = None
    for name in video_cache.keys():
        if name.startswith(f"{video_num}_"):
            video_name = name
            break
    
    if not video_name:
        raise HTTPException(status_code=404, detail=f"Video {video_num} not found")
    
    current_video = video_name
    is_processing = True
    
    print(f"🎬 Started tracking for video {video_num}: {video_name}")
    
    return {
        "message": f"Started tracking for {video_name}",
        "video_num": video_num,
        "video_name": video_name
    }

@app.post("/api/pause-tracking")
async def pause_tracking():
    """Pause current tracking"""
    global is_processing
    
    is_processing = False
    print("⏸️ Tracking paused")
    
    return {"message": "Tracking paused"}

@app.post("/api/resume-tracking")
async def resume_tracking():
    """Resume current tracking"""
    global is_processing
    
    is_processing = True
    print("▶️ Tracking resumed")
    
    return {"message": "Tracking resumed"}

@app.get("/api/video-stream/{video_num}")
@app.head("/api/video-stream/{video_num}")
async def stream_video(video_num: str, request: Request):
    """Fast video stream - sends original frames without processing"""
    global current_video, is_processing, triton_client, renderer
    
    if not triton_client:
        raise HTTPException(status_code=503, detail="Triton server not available")
    
    # Find video by number
    video_name = None
    for name in video_cache.keys():
        if name.startswith(f"{video_num}_"):
            video_name = name
            break
    
    if not video_name:
        raise HTTPException(status_code=404, detail=f"Video {video_num} not found")
    
    # Handle HEAD request
    if request.method == "HEAD":
        return Response(
            status_code=200,
            headers={
                "Content-Type": "multipart/x-mixed-replace; boundary=frame",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    def generate_fast_frames():
        # Open video
        cap = cv2.VideoCapture(video_cache[video_name])
        if not cap.isOpened():
            print(f"Error: Could not open video {video_name}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Fast streaming video {video_name}: {width}x{height}, {fps} FPS")
        
        frame_count = 0
        frames_to_skip = fps  # Skip first second
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Video ended, restart
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    continue
                
                frame_count += 1
                
                # Skip first second
                if frame_count <= frames_to_skip:
                    continue
                
                # Encode frame (fast, no processing)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_bytes = buffer.tobytes()
                
                # Yield frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Frame rate control (target 30 FPS for smooth video)
                time.sleep(1.0 / 30.0)
                
        except Exception as e:
            print(f"Error in fast video streaming: {e}")
        finally:
            cap.release()
    
    return StreamingResponse(
        generate_fast_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/inference-stream/{video_num}")
@app.head("/api/inference-stream/{video_num}")
async def stream_inference(video_num: str, request: Request):
    """Stream inference data (tracking results) using batch processing"""
    global current_video, is_processing, triton_client, renderer
    
    if not triton_client:
        raise HTTPException(status_code=503, detail="Triton server not available")
    
    # Find video by number
    video_name = None
    for name in video_cache.keys():
        if name.startswith(f"{video_num}_"):
            video_name = name
            break
    
    if not video_name:
        raise HTTPException(status_code=404, detail=f"Video {video_num} not found")
    
    # Handle HEAD request
    if request.method == "HEAD":
        return Response(
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    
    def generate_inference_batches():
        # Open video
        cap = cv2.VideoCapture(video_cache[video_name])
        if not cap.isOpened():
            print(f"Error: Could not open video {video_name}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Batch inference streaming {video_name}: {width}x{height}, {fps} FPS")
        
        # Tracking variables
        tracks = {}
        next_track_id = 1
        frame_count = 0
        frames_to_skip = fps  # Skip first second
        
        # Batch processing variables
        batch_frames = []
        batch_size = fps  # Process 1 second of frames at a time
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    # Video ended, restart
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    batch_frames = []
                    continue
                
                frame_count += 1
                
                # Skip first second
                if frame_count <= frames_to_skip:
                    continue
                
                # Add frame to batch
                batch_frames.append((frame_count, frame))
                
                # Process batch when we have enough frames
                if len(batch_frames) >= batch_size:
                    # Process the entire batch
                    batch_results = []
                    
                    for frame_num, frame in batch_frames:
                        if not is_processing:
                            # When paused, send empty results
                            batch_results.append({
                                "frame": frame_num,
                                "tracks": [],
                                "timestamp": time.time()
                            })
                            continue
                        
                        try:
                            # Detect persons
                            boxes, confidences, class_ids, masks = triton_client.detect_persons(frame)
                            
                            # Process detections and update tracking
                            tracked_objects = []
                            person_detections = []
                            
                            # Filter person detections first
                            for i, (box, conf, cls_id, mask) in enumerate(zip(boxes, confidences, class_ids, masks)):
                                if int(cls_id) == 0 and conf >= 0.25:  # Person class
                                    x1, y1, x2, y2 = box.astype(int)
                                    person_detections.append((x1, y1, x2, y2, conf))
                            
                            # Process only if we have detections
                            if person_detections:
                                # Simple tracking - find closest existing track
                                for x1, y1, x2, y2, conf in person_detections:
                                    track_id = None
                                    min_distance = float('inf')
                                    
                                    # Only check tracks that are not too old
                                    for tid, track in tracks.items():
                                        if 'last_bbox' in track and track['age'] < 10:
                                            tx1, ty1, tx2, ty2 = track['last_bbox']
                                            # Calculate IoU (simplified)
                                            iou = calculate_iou([x1, y1, x2, y2], [tx1, ty1, tx2, ty2])
                                            if iou > 0.3:  # IoU threshold
                                                distance = abs(x1 - tx1) + abs(y1 - ty1)
                                                if distance < min_distance:
                                                    min_distance = distance
                                                    track_id = tid
                                    
                                    if track_id is None:
                                        track_id = next_track_id
                                        next_track_id += 1
                                        tracks[track_id] = {
                                            'gender_counts': {'Man': 0, 'Woman': 0},
                                            'gender': 'Unknown',
                                            'age': 0
                                        }
                                    
                                    # Update track
                                    tracks[track_id]['last_bbox'] = [x1, y1, x2, y2]
                                    tracks[track_id]['age'] = 0
                                    
                                    # Classify gender (only every 5 frames for performance)
                                    if frame_num % 5 == 0:
                                        person_crop = frame[y1:y2, x1:x2]
                                        if person_crop.size > 0:
                                            gender, gender_conf = triton_client.classify_gender(person_crop)
                                            if gender_conf >= 0.5:  # Confidence threshold
                                                if gender in tracks[track_id]['gender_counts']:
                                                    tracks[track_id]['gender_counts'][gender] += 1
                                                
                                                # Update most likely gender
                                                counts = tracks[track_id]['gender_counts']
                                                if counts['Man'] > counts['Woman']:
                                                    tracks[track_id]['gender'] = 'Man'
                                                elif counts['Woman'] > counts['Man']:
                                                    tracks[track_id]['gender'] = 'Woman'
                                    
                                    tracked_objects.append({
                                        'id': track_id,
                                        'bbox': [x1, y1, x2, y2],
                                        'gender': tracks[track_id]['gender'],
                                        'confidence': conf
                                    })
                            
                            # Age tracks and remove old ones
                            tracks_to_remove = []
                            for tid, track in tracks.items():
                                track['age'] += 1
                                if track['age'] > 30:  # Remove tracks older than 30 frames
                                    tracks_to_remove.append(tid)
                            
                            for tid in tracks_to_remove:
                                del tracks[tid]
                            
                            batch_results.append({
                                "frame": frame_num,
                                "tracks": tracked_objects,
                                "timestamp": time.time()
                            })
                            
                        except Exception as e:
                            print(f"Error processing frame {frame_num}: {e}")
                            batch_results.append({
                                "frame": frame_num,
                                "tracks": [],
                                "timestamp": time.time()
                            })
                    
                    # Send the entire batch as JSON
                    yield f"data: {json.dumps({'batch': batch_results})}\n\n"
                    
                    # Clear batch for next iteration
                    batch_frames = []
                
        except Exception as e:
            print(f"Error in batch inference streaming: {e}")
        finally:
            cap.release()
    
    return StreamingResponse(
        generate_inference_batches(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)
    
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)