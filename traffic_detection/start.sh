#!/bin/bash

echo "============================================================"
echo "🚗 TRAFFIC DETECTION VIDEO PROCESSING API SERVER"
echo "============================================================"

# Use the dedicated port killer script
./kill_ports.sh

echo "🚀 Starting Traffic Detection API Server..."
echo "📡 Server will be available at: http://localhost:8001"
echo "📚 API documentation: http://localhost:8001/docs"
echo "🌐 Frontend interface: http://localhost:8001/"
echo "⚠️  Press Ctrl+C to stop the server"
echo "------------------------------------------------------------"

# Run the server directly
cd /workspace/global/saudi_arabia/traffic_detection
uv run server.py
