#!/usr/bin/env python3
"""
Triton-compatible server using RBLN SDK directly
This mimics the Triton Inference Server API but uses RBLN models directly
"""

import sys
import os
from pathlib import Path
import asyncio
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import time
from contextlib import asynccontextmanager

# Add the src directory to the path
sys.path.append('/workspace/projects/global/saudi_arabia/gender_detection/src')

from models import PersonDetector, GenderClassifier
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

class TritonCompatibleServer:
    """Triton-compatible server using RBLN SDK"""
    
    def __init__(self, person_conf=0.25, gender_conf=0.5, enable_color_heuristic=False):
        self.person_conf = person_conf
        self.gender_conf = gender_conf
        self.enable_color_heuristic = enable_color_heuristic
        
        # Initialize models
        print("Loading RBLN models...")
        self.person_detector = PersonDetector(conf=person_conf, enable_color_detection=enable_color_heuristic)
        self.gender_classifier = GenderClassifier(conf_threshold=gender_conf)
        print("✅ RBLN models loaded successfully")
    
    def detect_persons(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Detect persons in image using RBLN PersonDetector"""
        try:
            results = self.person_detector.infer(image)
            if not results or len(results) == 0:
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            boxes = []
            confidences = []
            class_ids = []
            masks = []
            
            for result in results:
                if result.boxes is not None and result.masks is not None:
                    boxes_tensor = result.boxes.xyxy.cpu().numpy()
                    confidences_tensor = result.boxes.conf.cpu().numpy()
                    class_ids_tensor = result.boxes.cls.cpu().numpy()
                    masks_tensor = result.masks.data.cpu().numpy()
                    
                    for box, conf, cls_id, mask in zip(boxes_tensor, confidences_tensor, class_ids_tensor, masks_tensor):
                        if int(cls_id) == 0 and conf >= self.person_conf:
                            boxes.append([box[0], box[1], box[2], box[3]])
                            confidences.append(float(conf))
                            class_ids.append(int(cls_id))
                            masks.append(mask)
            
            return np.array(boxes), np.array(confidences), np.array(class_ids), np.array(masks)
        except Exception as e:
            print(f"Error in person detection: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def classify_gender(self, person_image: np.ndarray) -> Tuple[str, float]:
        """Classify gender from person image using RBLN GenderClassifier"""
        try:
            gender, confidence = self.gender_classifier.classify_gender(person_image)
            return gender, confidence
        except Exception as e:
            print(f"Error in gender classification: {e}")
            return "Unknown", 0.0

# Global server instance
triton_server: Optional[TritonCompatibleServer] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global triton_server
    print("🚀 Starting Triton-compatible server with RBLN...")
    
    try:
        triton_server = TritonCompatibleServer()
        print("✅ Triton-compatible server ready!")
    except Exception as e:
        print(f"❌ Failed to initialize server: {e}")
        triton_server = None
    
    yield
    
    # Shutdown
    print("🛑 Shutting down server...")

# Create FastAPI app
app = FastAPI(
    title="Triton-compatible RBLN Server",
    description="Triton Inference Server compatible API using RBLN SDK",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/v2/health/ready")
async def health_ready():
    """Triton health check endpoint"""
    if triton_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    return JSONResponse(content={"status": "ready"})

@app.get("/v2/health/live")
async def health_live():
    """Triton liveness check endpoint"""
    return JSONResponse(content={"status": "live"})

@app.get("/v2/models")
async def list_models():
    """List available models"""
    return JSONResponse(content={
        "models": [
            {
                "name": "person_detection",
                "platform": "python",
                "backend": "rbln"
            },
            {
                "name": "gender_classification", 
                "platform": "python",
                "backend": "rbln"
            }
        ]
    })

@app.get("/v2/models/{model_name}")
async def get_model(model_name: str):
    """Get model information"""
    if model_name not in ["person_detection", "gender_classification"]:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return JSONResponse(content={
        "name": model_name,
        "platform": "python",
        "backend": "rbln",
        "status": "ready"
    })

@app.post("/v2/models/{model_name}/infer")
async def infer_model(model_name: str, request: dict):
    """Run inference on a model"""
    if triton_server is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    try:
        # Extract input data
        inputs = request.get("inputs", [])
        if not inputs:
            raise HTTPException(status_code=400, detail="No inputs provided")
        
        # Get the first input (assuming single input for simplicity)
        input_data = inputs[0]
        input_name = input_data.get("name")
        input_shape = input_data.get("shape")
        input_data_raw = input_data.get("data")
        
        if not input_data_raw:
            raise HTTPException(status_code=400, detail="No input data provided")
        
        # Convert input data to numpy array
        image_array = np.array(input_data_raw, dtype=np.uint8).reshape(input_shape)
        
        if model_name == "person_detection":
            # Run person detection
            boxes, confidences, class_ids, masks = triton_server.detect_persons(image_array)
            
            # Format outputs
            outputs = [
                {
                    "name": "OUTPUT__0",
                    "shape": boxes.shape,
                    "data": boxes.tolist()
                },
                {
                    "name": "OUTPUT__1", 
                    "shape": confidences.shape,
                    "data": confidences.tolist()
                },
                {
                    "name": "OUTPUT__2",
                    "shape": class_ids.shape,
                    "data": class_ids.tolist()
                },
                {
                    "name": "OUTPUT__3",
                    "shape": masks.shape,
                    "data": masks.tolist()
                }
            ]
            
        elif model_name == "gender_classification":
            # Run gender classification
            gender, confidence = triton_server.classify_gender(image_array)
            
            # Format outputs
            outputs = [
                {
                    "name": "OUTPUT__0",
                    "shape": [1],
                    "data": [gender]
                },
                {
                    "name": "OUTPUT__1",
                    "shape": [1], 
                    "data": [confidence]
                }
            ]
        else:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return JSONResponse(content={
            "model_name": model_name,
            "outputs": outputs
        })
        
    except Exception as e:
        print(f"Error in inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Video cache for batch processing
video_cache = {}

def load_videos():
    """Load all videos into cache"""
    video_dir = Path("/workspace/projects/global/saudi_arabia/gender_detection/data/inputs")
    if not video_dir.exists():
        print("❌ Videos directory not found")
        return
    
    for video_file in video_dir.glob("*.mp4"):
        video_name = video_file.stem
        video_cache[video_name] = str(video_file)
        print(f"📁 Loaded video: {video_name}")

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    x1 = max(x1_min, x2_min)
    y1 = max(y1_min, y2_min)
    x2 = min(x1_max, x2_max)
    y2 = min(y1_max, y2_max)
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

@app.get("/api/batch-inference/{video_num}")
async def batch_inference(video_num: str):
    """Batch inference endpoint that processes frames in batches"""
    global triton_server
    
    # Find video by number
    video_name = None
    for name in video_cache.keys():
        if name.startswith(f"{video_num}_"):
            video_name = name
            break
    
    if not video_name:
        raise HTTPException(status_code=404, detail=f"Video {video_num} not found")
    
    def generate_batch_results():
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
                        try:
                            # Detect persons using direct model calls (no HTTP)
                            boxes, confidences, class_ids, masks = triton_server.detect_persons(frame)
                            
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
                                            gender, gender_conf = triton_server.classify_gender(person_crop)
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
                                        'id': int(track_id),
                                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                        'gender': tracks[track_id]['gender'],
                                        'confidence': float(conf)
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
        generate_batch_results(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

if __name__ == "__main__":
    # Load videos on startup
    load_videos()
    uvicorn.run(app, host="0.0.0.0", port=8002)
