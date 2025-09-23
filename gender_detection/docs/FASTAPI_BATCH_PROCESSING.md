# FastAPI Batch Video Processing Guide

This guide shows how to process all 5 videos using the FastAPI server with their predefined configurations.

## 🚀 Quick Start

### 1. Start the FastAPI Server
```bash
python start_server.py
```
Wait for the message: "All video processors initialized!"

### 2. Process All Videos with Predefined Configs
```bash
python example_client.py --all
```

This will:
- Process all 5 videos (`01_man.mp4`, `02_woman.mp4`, `03_family.mp4`, `04_group.mp4`, `05_office.mp4`)
- Use their corresponding predefined configurations from `PROCESSOR_CONFIGS`
- Save outputs to `data/outputs/` with `_fastapi_output.mp4` suffix

## 📁 Expected Output Files

After processing, you'll have these files in `data/outputs/`:

```
data/outputs/
├── 01_man_fastapi_output.mp4      # Uses config: enable_color_heuristic=False
├── 02_woman_fastapi_output.mp4    # Uses config: enable_color_heuristic=True
├── 03_family_fastapi_output.mp4   # Uses config: enable_color_heuristic=True
├── 04_group_fastapi_output.mp4    # Uses config: enable_color_heuristic=True
└── 05_office_fastapi_output.mp4   # Uses config: person_conf=0.2 (lower confidence)
```

## 🎯 Configuration Mapping

| Video | Config Used | Key Settings |
|-------|-------------|--------------|
| `01_man.mp4` | `01_man` | `enable_color_heuristic: False` |
| `02_woman.mp4` | `02_woman` | `enable_color_heuristic: True` |
| `03_family.mp4` | `03_family` | `enable_color_heuristic: True` |
| `04_group.mp4` | `04_group` | `enable_color_heuristic: True` |
| `05_office.mp4` | `05_office` | `person_conf: 0.2` (lower confidence) |

## 📊 Sample Output

```
Gender Detection Video Processing API - Example Client
============================================================
✅ Server is running!
Available configurations: ['01_man', '02_woman', '03_family', '04_group', '05_office']

🎬 Processing 5 videos with predefined configurations...
======================================================================

📹 Processing 01_man.mp4 with config '01_man'...
   Job ID: abc123-def456-ghi789
   Configuration: {'person_conf': 0.25, 'enable_color_heuristic': False, ...}
   Status: processing - Progress: 10% (120/1200 frames) - Active tracks: 1
   Status: processing - Progress: 20% (240/1200 frames) - Active tracks: 1
   ...
   Status: completed - Processing complete! Maximum active tracks: 1
   ✅ Downloaded to: data/outputs/01_man_fastapi_output.mp4
   🧹 Cleaned up temporary files

📹 Processing 02_woman.mp4 with config '02_woman'...
   ...

======================================================================
📊 PROCESSING SUMMARY
======================================================================
✅ Successfully processed 5 videos:
   • 01_man.mp4 → data/outputs/01_man_fastapi_output.mp4 (config: 01_man)
   • 02_woman.mp4 → data/outputs/02_woman_fastapi_output.mp4 (config: 02_woman)
   • 03_family.mp4 → data/outputs/03_family_fastapi_output.mp4 (config: 03_family)
   • 04_group.mp4 → data/outputs/04_group_fastapi_output.mp4 (config: 04_group)
   • 05_office.mp4 → data/outputs/05_office_fastapi_output.mp4 (config: 05_office)

🎉 Processing complete! Check the data/outputs/ directory for results.
```

## 🔧 Alternative Usage

### Process Single Video
```bash
python example_client.py 02_woman
```

### Check Available Configs
```bash
curl http://localhost:8000/configs
```

## 🆚 Comparison with Original Processing

| Method | Original `process.py` | FastAPI `example_client.py` |
|--------|----------------------|---------------------------|
| **Execution** | Sequential, blocking | Asynchronous, non-blocking |
| **Output Files** | `{video}_output.mp4` | `{video}_fastapi_output.mp4` |
| **Progress** | Console prints | Real-time API status |
| **Error Handling** | Script stops on error | Continues with other videos |
| **Scalability** | Single instance | Multiple concurrent jobs |

## 🎯 Key Benefits

1. **Same Processing Logic**: Uses identical AI models and configurations
2. **Better Error Handling**: Failed videos don't stop the entire batch
3. **Progress Tracking**: Real-time status updates for each video
4. **Parallel Processing**: Server can handle multiple videos simultaneously
5. **API Integration**: Easy to integrate with web applications or other services
6. **Clean Separation**: FastAPI outputs are tagged separately from original outputs

## 🐛 Troubleshooting

### Server Not Running
```
❌ Server is not running. Please start it with: python start_server.py
```
**Solution**: Start the FastAPI server first.

### Video File Not Found
```
❌ Video file data/inputs/01_man.mp4 not found. Skipping...
```
**Solution**: Ensure all input videos exist in `data/inputs/` directory.

### Processing Failed
```
❌ Error processing 02_woman: Connection timeout
```
**Solution**: Check server logs and ensure sufficient system resources.

## 📈 Performance Notes

- Processing time depends on video length and system resources
- Each video processes independently (no shared state between jobs)
- Progress updates every 10% completion
- Temporary files are automatically cleaned up after download
- Server can handle multiple concurrent requests
