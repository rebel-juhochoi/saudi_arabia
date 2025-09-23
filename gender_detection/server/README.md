# Gender Detection Server

A FastAPI-based server that provides real-time gender detection using RBLN SDK with YOLO11n-seg and DeepFace models.

## Architecture

- **RBLN SDK**: Direct model inference using rebel-compiler without Docker
- **FastAPI Server**: Provides REST API endpoints for video streaming and control
- **Real-time Processing**: Processes video frames with person detection and gender classification

## Quick Start

### Start the Server
```bash
./start_server.sh
```

### Access the API
- **API Base**: `http://localhost:8001`
- **Health Check**: `http://localhost:8001/`
- **API Docs**: `http://localhost:8001/docs`

## API Endpoints

### Health Check
- `GET /` - Check if API is running and RBLN models are ready

### Video Management
- `GET /api/videos` - Get list of available videos
- `POST /api/start-tracking` - Start tracking for a specific video
- `POST /api/pause-tracking` - Pause current tracking
- `POST /api/resume-tracking` - Resume tracking

### Video Streaming
- `GET /api/video-stream/{video_num}` - Stream processed video with tracking

## Video Numbers
- `01` - 01_man
- `02` - 02_woman  
- `03` - 03_family
- `04` - 04_group
- `05` - 05_office

## Files

- `main.py` - FastAPI server with RBLN integration
- `rbln_server.py` - RBLN model server using rebel-compiler
- `start_server.sh` - Server startup script
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.8+
- RBLN SDK (rebel-compiler)
- OpenCV
- FastAPI
- NumPy < 2.0

## Usage

1. **Start the server**:
   ```bash
   ./start_server.sh
   ```

2. **Test the API**:
   ```bash
   curl http://localhost:8001/
   ```

3. **Access video stream**:
   Open `http://localhost:8001/api/video-stream/01` in browser

## Development

The server uses the existing `stream.py` architecture with shared models and individual trackers per video, but integrates with RBLN for direct model inference without requiring Docker or Triton.