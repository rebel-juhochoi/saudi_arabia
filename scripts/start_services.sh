#!/bin/bash

echo "🚀 Starting Complete Gender Detection System..."
echo "=============================================="

# Get the project root directory
PROJECT_ROOT="/workspace/projects/global/saudi_arabia/gender_detection"

# Kill any existing services
echo "🛑 Stopping existing services..."
pkill -f "python.*triton_server" 2>/dev/null
pkill -f "python.*main.py" 2>/dev/null
pkill -f "node.*server.js" 2>/dev/null
sleep 3

# Start Triton server
echo "📡 Starting Triton server on port 8002..."
cd "$PROJECT_ROOT/server"
nohup uv run python triton_server.py > triton.log 2>&1 &
TRITON_PID=$!
echo "   Triton server PID: $TRITON_PID"

# Wait for Triton to be ready
echo "⏳ Waiting for Triton server to be ready..."
sleep 10

# Start FastAPI server
echo "🌐 Starting FastAPI server on port 8001..."
nohup uv run python main.py > fastapi.log 2>&1 &
API_PID=$!
echo "   FastAPI server PID: $API_PID"

# Wait for FastAPI to be ready
echo "⏳ Waiting for FastAPI server to be ready..."
sleep 5

# Start Frontend server
echo "🎨 Starting Frontend server on port 8000..."
cd "$PROJECT_ROOT/client"
nohup node server.js > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "   Frontend server PID: $FRONTEND_PID"

# Wait for Frontend to be ready
echo "⏳ Waiting for Frontend server to be ready..."
sleep 3

# Check all services
echo ""
echo "🔍 Checking service status..."

# Check Triton
if curl -s http://localhost:8002/v2/health/ready > /dev/null; then
    echo "✅ Triton server is running on port 8002"
else
    echo "❌ Triton server failed to start"
fi

# Check FastAPI
if curl -s http://localhost:8001/ > /dev/null; then
    echo "✅ FastAPI server is running on port 8001"
else
    echo "❌ FastAPI server failed to start"
fi

# Check Frontend
if curl -s http://localhost:8000 > /dev/null; then
    echo "✅ Frontend server is running on port 8000"
else
    echo "❌ Frontend server failed to start"
fi

echo ""
echo "🎉 All services started successfully!"
echo "=============================================="
echo "📊 Triton server:    http://localhost:8002 (PID: $TRITON_PID)"
echo "🔗 FastAPI server:   http://localhost:8001 (PID: $API_PID)"
echo "🌐 Frontend:         http://localhost:8000 (PID: $FRONTEND_PID)"
echo ""
echo "📝 Logs:"
echo "   Triton:  tail -f $PROJECT_ROOT/server/triton.log"
echo "   FastAPI: tail -f $PROJECT_ROOT/server/fastapi.log"
echo "   Frontend: tail -f $PROJECT_ROOT/client/frontend.log"
echo ""
echo "🛑 To stop all services:"
echo "   $PROJECT_ROOT/scripts/stop_services.sh"
echo ""
echo "🔍 To monitor services:"
echo "   $PROJECT_ROOT/scripts/monitor_services.sh"
echo ""
echo "🎬 Open your browser and go to: http://localhost:8000"
