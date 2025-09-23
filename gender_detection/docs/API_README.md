# Gender Detection Video Processing API

This FastAPI server transforms the VideoProcessor into a web service that can process videos with gender detection and tracking capabilities.

## Features

- **Multiple Processor Configurations**: Pre-configured processors matching the original 5 video scenarios
- **Custom Configuration**: Upload videos with custom detection parameters
- **Asynchronous Processing**: Non-blocking video processing with job status tracking
- **File Upload/Download**: Upload videos and download processed results
- **Real-time Progress**: Track processing progress and status
- **Automatic Cleanup**: Clean up temporary files after processing

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python start_server.py
```

The server will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs (Interactive Swagger UI)
- **ReDoc**: http://localhost:8000/redoc (Alternative documentation)

## API Endpoints

### Core Endpoints

#### `GET /`
Get API information and available endpoints.

#### `GET /configs`
Get all available processor configurations and their parameters.

#### `POST /process-video/{config_name}`
Process a video using one of the pre-configured processors.

**Parameters:**
- `config_name`: One of `01_man`, `02_woman`, `03_family`, `04_group`, `05_office`
- `file`: Video file to upload

**Available Configurations:**
- `01_man`: Basic detection without color heuristic
- `02_woman`: Standard detection with color heuristic
- `03_family`: Family scenario with color heuristic
- `04_group`: Group detection with color heuristic  
- `05_office`: Office scenario with lower person confidence (0.2)

#### `POST /process-custom-video`
Process a video with custom configuration parameters.

**Parameters:**
- `file`: Video file to upload
- `person_conf`: Person detection confidence (default: 0.25)
- `iou_threshold`: IoU threshold for tracking (default: 0.5)
- `gender_conf`: Gender classification confidence (default: 0.5)
- `enable_color_heuristic`: Enable color-based heuristic (default: true)
- `show_segmentation`: Show segmentation masks (default: true)

#### `GET /status/{job_id}`
Get the current processing status of a job.

**Response includes:**
- `status`: queued, processing, completed, error
- `progress`: Percentage complete (0-100)
- `message`: Current status message
- `config_name`: Configuration used
- `filename`: Original filename

#### `GET /download/{job_id}`
Download the processed video file.

#### `DELETE /cleanup/{job_id}`
Clean up temporary files for a completed job.

#### `GET /health`
Health check endpoint showing server status.

## Usage Examples

### Using cURL

#### 1. Process video with pre-configured settings:
```bash
curl -X POST "http://localhost:8000/process-video/02_woman" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_video.mp4"
```

#### 2. Process video with custom settings:
```bash
curl -X POST "http://localhost:8000/process-custom-video" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_video.mp4" \
     -F "person_conf=0.3" \
     -F "gender_conf=0.6" \
     -F "enable_color_heuristic=true"
```

#### 3. Check processing status:
```bash
curl -X GET "http://localhost:8000/status/{job_id}"
```

#### 4. Download processed video:
```bash
curl -X GET "http://localhost:8000/download/{job_id}" --output processed_video.mp4
```

### Using Python requests

```python
import requests
import time

# Upload and process video
with open('your_video.mp4', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process-video/02_woman',
        files={'file': f}
    )

job_data = response.json()
job_id = job_data['job_id']
print(f"Job started: {job_id}")

# Poll for completion
while True:
    status_response = requests.get(f'http://localhost:8000/status/{job_id}')
    status_data = status_response.json()
    
    print(f"Status: {status_data['status']} - {status_data['message']}")
    
    if status_data['status'] == 'completed':
        break
    elif status_data['status'] == 'error':
        print("Processing failed!")
        break
        
    time.sleep(5)  # Wait 5 seconds before checking again

# Download result
if status_data['status'] == 'completed':
    download_response = requests.get(f'http://localhost:8000/download/{job_id}')
    with open('processed_video.mp4', 'wb') as f:
        f.write(download_response.content)
    print("Downloaded processed video!")
```

## Processor Configurations

The server includes 5 pre-configured processors that match the original processing script:

| Config Name | Person Conf | IoU Threshold | Gender Conf | Color Heuristic | Description |
|-------------|-------------|---------------|-------------|-----------------|-------------|
| `01_man` | 0.25 | 0.5 | 0.5 | False | Basic man detection |
| `02_woman` | 0.25 | 0.5 | 0.5 | True | Woman detection with color |
| `03_family` | 0.25 | 0.5 | 0.5 | True | Family scenario |
| `04_group` | 0.25 | 0.5 | 0.5 | True | Group detection |
| `05_office` | 0.2 | 0.5 | 0.5 | True | Office (lower confidence) |

## Supported Video Formats

- MP4 (.mp4)
- AVI (.avi)
- MOV (.mov)
- MKV (.mkv)

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid file format or configuration
- **404 Not Found**: Job ID not found
- **500 Internal Server Error**: Processing errors

## Performance Notes

- Video processing runs asynchronously in the background
- Multiple jobs can be processed simultaneously
- Temporary files are stored in the system temp directory
- Use the cleanup endpoint to remove temporary files after downloading results
- Processing time depends on video length, resolution, and detection parameters

## Development

To run in development mode with auto-reload:
```bash
python start_server.py
```

To run with custom host/port:
```python
uvicorn server:app --host 0.0.0.0 --port 8080 --reload
```

## Architecture

The server extends the original `VideoProcessor` class with:
- `CustomVideoProcessor`: Handles uploaded files and async processing
- Background task processing using FastAPI's BackgroundTasks
- Job status tracking with unique job IDs
- Temporary file management
- RESTful API endpoints for all operations

Each processor configuration creates its own instance with specific parameters, ensuring optimal performance for different video scenarios.
