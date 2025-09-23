# 🌐 Gender Detection Web Frontend

A modern, responsive web interface for the Gender Detection Video Processing system. Upload videos, configure processing parameters, and view results with real-time progress tracking.

## 🚀 Quick Start

### 1. Start the FastAPI Server
```bash
python start_server.py
```

### 2. Open the Web Interface
Navigate to: **http://localhost:8000**

The web frontend will automatically load, showing:
- ✅ Server status indicator
- 🎛️ Configuration options (predefined or custom)
- 📤 Video upload area
- 📊 Real-time processing progress
- 🎥 Side-by-side video comparison

## 🎯 Features

### 🔧 **Configuration Options**
- **Predefined Configs**: Choose from 5 optimized configurations
  - `01_MAN`: Basic detection (no color heuristic)
  - `02_WOMAN`: Standard detection with color heuristic
  - `03_FAMILY`: Family scenario optimization
  - `04_GROUP`: Group detection settings
  - `05_OFFICE`: Office environment (lower confidence)

- **Custom Config**: Fine-tune parameters
  - Person Detection Confidence (0.1 - 0.9)
  - Gender Classification Confidence (0.1 - 0.9)
  - IoU Threshold (0.1 - 0.9)
  - Color Heuristic (Enable/Disable)
  - Segmentation Display (Show/Hide)

### 📤 **File Upload**
- **Drag & Drop**: Simply drag video files onto the upload area
- **Browse**: Click to select files from your computer
- **Supported Formats**: MP4, AVI, MOV, MKV
- **File Validation**: Automatic format checking and size display

### 📊 **Real-Time Processing**
- **Progress Bar**: Visual progress indicator (0-100%)
- **Status Updates**: Live status messages during processing
- **Job Tracking**: Unique job ID for each processing task
- **Configuration Display**: Shows which settings are being used

### 🎥 **Video Comparison**
- **Side-by-Side View**: Original vs. Processed video
- **Synchronized Playback**: Both videos play in sync
- **Download Option**: Save processed video to your device
- **Process Another**: Easily start a new processing job

### 🎨 **Modern UI/UX**
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Glass Morphism**: Modern translucent design elements
- **Smooth Animations**: Polished transitions and interactions
- **Dark/Light Theme**: Automatic adaptation based on system preference
- **Accessibility**: Keyboard navigation and screen reader support

## 📱 Screenshots

### Main Interface
```
┌─────────────────────────────────────────────────────────────┐
│  🎥 Gender Detection Video Processor                        │
│  Upload a video to detect and track people with gender...   │
├─────────────────────────────────────────────────────────────┤
│  🟢 Server Status: Online (5 processors loaded)            │
├─────────────────────────────────────────────────────────────┤
│  ⚙️ Processing Configuration                                │
│  ◉ Use Predefined Configuration                            │
│  ○ Custom Configuration                                     │
│                                                             │
│  Choose Configuration: [02_WOMAN ▼]                        │
│  📋 Configuration Details:                                  │
│  • Person Confidence: 0.25                                 │
│  • Color Heuristic: Enabled                                │
├─────────────────────────────────────────────────────────────┤
│  📤 Upload Video                                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ☁️                                                  │   │
│  │  Drag & drop your video here or browse files       │   │
│  │  Supported formats: MP4, AVI, MOV, MKV             │   │
│  └─────────────────────────────────────────────────────┘   │
│  ▶️ Start Processing                                        │
└─────────────────────────────────────────────────────────────┘
```

### Processing View
```
┌─────────────────────────────────────────────────────────────┐
│  ⏳ Processing Video                                        │
│  ████████████░░░░░░░░░░░░ 65%                              │
│  Status: Processing frame 780/1200 - Active tracks: 3      │
│                                                             │
│  Job ID: abc123-def456-ghi789                              │
│  Configuration: 02_woman                                    │
└─────────────────────────────────────────────────────────────┘
```

### Results View
```
┌─────────────────────────────────────────────────────────────┐
│  👁️ Results                                                 │
│  ┌─────────────────────┐  ┌─────────────────────────────┐   │
│  │   Original Video    │  │ Processed Video (with       │   │
│  │   ▶️ [============] │  │ Detections)                 │   │
│  │                     │  │ ▶️ [============]           │   │
│  └─────────────────────┘  └─────────────────────────────┘   │
│                                                             │
│  💾 Download Processed Video    ➕ Process Another Video   │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Workflow

1. **Server Check**: Frontend automatically checks if FastAPI server is running
2. **Config Load**: Fetches available configurations from the server
3. **File Selection**: User uploads a video file
4. **Configuration**: Choose predefined or custom settings
5. **Processing**: Video is sent to server for processing
6. **Progress Tracking**: Real-time updates via polling
7. **Results Display**: Original and processed videos shown side-by-side
8. **Download**: Save processed video locally

## 🛠️ Technical Details

### Frontend Stack
- **HTML5**: Semantic markup with modern features
- **CSS3**: Advanced styling with animations and responsive design
- **Vanilla JavaScript**: No framework dependencies for maximum compatibility
- **Font Awesome**: Professional icons throughout the interface

### API Integration
- **RESTful API**: Full integration with FastAPI backend
- **Async Operations**: Non-blocking file uploads and processing
- **Error Handling**: Comprehensive error messages and recovery options
- **Progress Polling**: Real-time status updates every 2 seconds

### Browser Support
- ✅ Chrome 80+
- ✅ Firefox 75+
- ✅ Safari 13+
- ✅ Edge 80+

## 🎛️ Configuration Comparison

| Config | Person Conf | Gender Conf | IoU | Color Heuristic | Best For |
|--------|-------------|-------------|-----|-----------------|----------|
| **01_MAN** | 0.25 | 0.5 | 0.5 | ❌ | Single person videos |
| **02_WOMAN** | 0.25 | 0.5 | 0.5 | ✅ | General purpose |
| **03_FAMILY** | 0.25 | 0.5 | 0.5 | ✅ | Family gatherings |
| **04_GROUP** | 0.25 | 0.5 | 0.5 | ✅ | Group scenarios |
| **05_OFFICE** | 0.2 | 0.5 | 0.5 | ✅ | Office environments |

## 🚨 Troubleshooting

### Server Connection Issues
```
❌ Server Status: Offline
```
**Solution**: 
1. Ensure FastAPI server is running: `python start_server.py`
2. Check if port 8000 is available
3. Refresh the page after server starts

### Upload Failures
```
❌ Please select a valid video file (MP4, AVI, MOV, MKV)
```
**Solution**: 
1. Check file format is supported
2. Ensure file is not corrupted
3. Try a smaller file size first

### Processing Errors
```
❌ Processing failed: Server timeout
```
**Solution**:
1. Check server logs for errors
2. Ensure sufficient system resources
3. Try with a shorter video
4. Use lower confidence settings

## 🎯 Performance Tips

### For Best Results:
- **Video Length**: Under 2 minutes for faster processing
- **Resolution**: 720p or 1080p recommended
- **Format**: MP4 with H.264 encoding works best
- **Content**: Well-lit scenes with clear person visibility

### Server Resources:
- **RAM**: 8GB+ recommended for larger videos
- **CPU**: Multi-core processor for faster processing
- **GPU**: Not required but can improve performance if available

## 🔗 API Endpoints Used

The frontend interacts with these FastAPI endpoints:

- `GET /health` - Server status checking
- `GET /configs` - Load available configurations
- `POST /process-video/{config_name}` - Process with predefined config
- `POST /process-custom-video` - Process with custom config
- `GET /status/{job_id}` - Check processing status
- `GET /download/{job_id}` - Download processed video
- `DELETE /cleanup/{job_id}` - Clean up temporary files

## 🎉 Enjoy!

The web frontend provides a user-friendly way to access all the power of your gender detection system. No command-line knowledge required - just drag, drop, and watch the magic happen!
