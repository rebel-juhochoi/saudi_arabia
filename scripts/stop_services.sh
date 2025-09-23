#!/bin/bash

echo "🛑 Stopping Gender Detection System Services..."
echo "=============================================="

# Get the project root directory
PROJECT_ROOT="/workspace/projects/global/saudi_arabia/gender_detection"

# Function to check if a process is running
check_process() {
    local pattern="$1"
    local name="$2"
    if pgrep -f "$pattern" > /dev/null; then
        echo "✅ $name is running"
        return 0
    else
        echo "❌ $name is not running"
        return 1
    fi
}

# Function to stop a process gracefully
stop_process() {
    local pattern="$1"
    local name="$2"
    local port="$3"
    
    echo "🛑 Stopping $name..."
    
    # First try graceful shutdown
    if pgrep -f "$pattern" > /dev/null; then
        echo "   Sending SIGTERM to $name..."
        pkill -f "$pattern"
        sleep 3
        
        # Check if still running
        if pgrep -f "$pattern" > /dev/null; then
            echo "   Process still running, sending SIGKILL..."
            pkill -9 -f "$pattern"
            sleep 2
        fi
        
        # Final check
        if pgrep -f "$pattern" > /dev/null; then
            echo "   ❌ Failed to stop $name"
            return 1
        else
            echo "   ✅ $name stopped successfully"
        fi
    else
        echo "   ℹ️  $name was not running"
    fi
}

# Check what's currently running
echo "🔍 Checking current service status..."
echo ""

check_process "python.*triton_server" "Triton Server"
check_process "python.*main.py" "FastAPI Server" 
check_process "node.*server.js" "Frontend Server"

echo ""
echo "🛑 Stopping all services..."
echo ""

# Stop all three daemons
stop_process "python.*triton_server" "Triton Server" "8002"
stop_process "python.*main.py" "FastAPI Server" "8001"
stop_process "node.*server.js" "Frontend Server" "8000"

echo ""
echo "🔍 Final status check..."
echo ""

# Final verification
TRITON_RUNNING=$(pgrep -f "python.*triton_server" | wc -l)
FASTAPI_RUNNING=$(pgrep -f "python.*main.py" | wc -l)
FRONTEND_RUNNING=$(pgrep -f "node.*server.js" | wc -l)

if [ $TRITON_RUNNING -eq 0 ] && [ $FASTAPI_RUNNING -eq 0 ] && [ $FRONTEND_RUNNING -eq 0 ]; then
    echo "✅ All services stopped successfully!"
    echo ""
    echo "📊 Service Status:"
    echo "   Triton Server:  Stopped"
    echo "   FastAPI Server: Stopped" 
    echo "   Frontend Server: Stopped"
    echo ""
    echo "🎉 Gender Detection System is now offline"
else
    echo "⚠️  Some services may still be running:"
    [ $TRITON_RUNNING -gt 0 ] && echo "   - Triton Server ($TRITON_RUNNING processes)"
    [ $FASTAPI_RUNNING -gt 0 ] && echo "   - FastAPI Server ($FASTAPI_RUNNING processes)"
    [ $FRONTEND_RUNNING -gt 0 ] && echo "   - Frontend Server ($FRONTEND_RUNNING processes)"
    echo ""
    echo "💡 You may need to manually kill remaining processes:"
    [ $TRITON_RUNNING -gt 0 ] && echo "   pkill -9 -f 'python.*triton_server'"
    [ $FASTAPI_RUNNING -gt 0 ] && echo "   pkill -9 -f 'python.*main.py'"
    [ $FRONTEND_RUNNING -gt 0 ] && echo "   pkill -9 -f 'node.*server.js'"
fi

echo ""
echo "=============================================="
echo "🛑 Stop script completed"
