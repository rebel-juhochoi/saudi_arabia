#!/bin/bash

echo "🚀 Starting Gender Detection Server..."
echo "====================================="

# Change to server directory
cd /workspace/projects/global/saudi_arabia/gender_detection/server

# Install dependencies
echo "📦 Installing dependencies..."
uv pip install -r ../requirements.txt

# Start Triton server
echo "📡 Starting Triton server on port 8002..."
uv run python triton_server.py
