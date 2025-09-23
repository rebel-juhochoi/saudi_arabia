# Traffic Detection Video Processing System

A real-time traffic monitoring system that detects, tracks, and counts vehicles using computer vision and machine learning.

## Features

- **Real-time Vehicle Detection**: Stream video with live vehicle detection and tracking
- **Intelligent Vehicle Counting**: Count vehicles using invisible horizontal line detection
- **Multi-Vehicle Support**: Detect cars, trucks, buses, bicycles, and motorcycles
- **WebSocket Streaming**: Real-time video streaming to web frontend
- **Smart Counting Algorithm**: Uses right bottom corner detection to avoid false positives
- **Interactive Web Interface**: Modern web UI with live traffic dashboard
- **Automatic Video Looping**: Continuous playback with seamless restart
- **FastAPI Backend**: RESTful API with WebSocket support
- **Clean Visual Output**: Invisible counting line for professional appearance

## System Architecture

```
┌─────────────────┐    WebSocket     ┌─────────────────┐
│   Web Frontend  │ ←──────────────→ │  FastAPI Server │
│   (HTML/JS)     │    25 FPS        │   (Python)      │
└─────────────────┘                  └─────────────────┘
                                              │
                                              ▼
                                    ┌─────────────────┐
                                    │ Video Processor │
                                    │ - Vehicle Det.  │
                                    │ - Road Area Det.│
                                    │ - Vehicle Track.│
                                    └─────────────────┘
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Start the Server**:
   ```bash
   ./start.sh
   ```

3. **Open Web Interface**:
   Navigate to `http://localhost:8001`

4. **Start Traffic Counting**:
   - Click "Start Detection" to begin processing
   - View live vehicle counts in the traffic dashboard
   - Watch vehicles disappear after being counted

## Project Structure

```
traffic_detection/
├── src/                    # Core processing modules
│   ├── models/            # AI model implementations
│   └── utils/             # Utility functions
├── frontend/              # Web interface
│   ├── index.html         # Main web page
│   ├── script.js          # Frontend JavaScript
│   └── styles.css         # CSS styling
├── data/                  # Video files and model data
│   └── inputs/           # Input video files
├── docs/                  # Documentation
├── server.py             # Main FastAPI server
├── start_server.py       # Server startup script
└── requirements.txt      # Python dependencies
```

## Demo Configuration

The system uses a single optimized configuration for the traffic detection demo:

- **Vehicle Confidence**: 0.25 (balanced detection sensitivity)
- **IoU Threshold**: 0.5 (optimal tracking accuracy)  
- **Road Detection**: Enabled (automatic road area identification)
- **Traffic Counting**: Enabled (cumulative vehicle counting per road area)
- **Segmentation Display**: Disabled (masks used internally for tracking)

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Server health check
- `GET /configs` - Available video configurations
- `WebSocket /ws/stream-video/{config_name}` - Real-time video streaming

## Technologies Used

- **Backend**: FastAPI, OpenCV, AsyncIO
- **Frontend**: HTML5, JavaScript, WebSockets
- **AI/ML**: Custom vehicle detection and road area detection models
- **Video Processing**: OpenCV with real-time frame processing
- **Streaming**: WebSocket-based real-time communication

## Development

The system uses a clean, modular architecture:

- **VehicleDetector**: YOLO-based vehicle detection for 5 vehicle classes
- **RoadAreaDetector**: Computer vision-based road area detection using color, texture, and edge analysis
- **ROI Filtering**: Vehicle detection limited to road areas for improved accuracy
- **SimpleStreamingProcessor**: Handles video streaming and processing
- **WebSocket Management**: Clean connection handling with automatic reconnection
- **Thread Pool Processing**: Non-blocking AI inference
- **Automatic Resource Management**: Efficient memory and connection cleanup

## Road Area Detection

The system automatically detects road areas in the first frame using:

- **Color Analysis**: Identifies typical road colors (asphalt/concrete)
- **Texture Analysis**: Detects road surface texture patterns
- **Edge Detection**: Analyzes edge density for road identification
- **Morphological Processing**: Cleans and refines detected areas
- **Multi-Road Support**: Can detect up to 2 separate road areas (e.g., different directions)

### Road Detection Features:
- Automatic ROI generation from first frame
- Debug image output for analysis
- Configurable minimum road area thresholds
- Support for complex road geometries
- Intersection filtering for vehicle detections

## Performance

- **Streaming Rate**: 25 FPS for smooth real-time viewing
- **Processing**: Asynchronous frame processing with thread pools
- **Memory Efficient**: Automatic cleanup and resource management
- **Scalable**: WebSocket-based architecture supports multiple clients

---

**Status**: Production Ready ✅
**Last Updated**: September 2025