# Gender Detection Video Processing System

A real-time video processing system that detects and tracks people with gender classification using computer vision and machine learning.

## Features

- **Real-time Video Processing**: Stream video with live gender detection and tracking
- **WebSocket Streaming**: Real-time video streaming to web frontend at 25 FPS
- **Multiple Video Configurations**: Pre-configured settings for different scenarios (man, woman, family, group, office)
- **Interactive Web Interface**: Modern web UI with video selection and controls
- **Segmentation Toggle**: Show/hide segmentation masks during streaming
- **Automatic Video Looping**: Continuous playback with seamless restart
- **FastAPI Backend**: RESTful API with WebSocket support

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
                                    │ - Person Det.   │
                                    │ - Gender Class. │
                                    │ - Object Track. │
                                    └─────────────────┘
```

## Quick Start

1. **Install Dependencies**:
   ```bash
   uv sync
   ```

2. **Start the Server**:
   ```bash
   uv run start_server.py
   ```

3. **Open Web Interface**:
   Navigate to `http://localhost:8000`

4. **Select and Stream**:
   - Choose a video from the dropdown
   - Toggle segmentation masks if desired
   - Video will stream automatically with gender detection

## Project Structure

```
gender_detection/
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

## Video Configurations

The system includes pre-configured settings for different scenarios:

- **01_man**: Single person detection with basic settings
- **02_woman**: Optimized for individual woman detection
- **03_family**: Family group detection with color heuristics
- **04_group**: Multi-person group scenarios
- **05_office**: Office environment with lower confidence thresholds

## API Endpoints

- `GET /` - Web interface
- `GET /health` - Server health check
- `GET /configs` - Available video configurations
- `WebSocket /ws/stream-video/{config_name}` - Real-time video streaming

## Technologies Used

- **Backend**: FastAPI, OpenCV, AsyncIO
- **Frontend**: HTML5, JavaScript, WebSockets
- **AI/ML**: Custom person detection and gender classification models
- **Video Processing**: OpenCV with real-time frame processing
- **Streaming**: WebSocket-based real-time communication

## Development

The system uses a clean, modular architecture:

- **SimpleStreamingProcessor**: Handles video streaming and processing
- **WebSocket Management**: Clean connection handling with automatic reconnection
- **Thread Pool Processing**: Non-blocking AI inference
- **Automatic Resource Management**: Efficient memory and connection cleanup

## Performance

- **Streaming Rate**: 25 FPS for smooth real-time viewing
- **Processing**: Asynchronous frame processing with thread pools
- **Memory Efficient**: Automatic cleanup and resource management
- **Scalable**: WebSocket-based architecture supports multiple clients

---

**Status**: Production Ready ✅
**Last Updated**: September 2025