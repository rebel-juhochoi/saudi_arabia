#!/bin/bash

echo "🔍 Gender Detection System Monitoring"
echo "====================================="

# Get the project root directory
PROJECT_ROOT="/workspace/projects/global/saudi_arabia/gender_detection"

# Function to show logs with timestamps
show_logs() {
    echo "📊 Monitoring all servers (last 20 lines each)..."
    echo ""
    
    echo "=== TRITON SERVER (Port 8002) ==="
    tail -n 20 "$PROJECT_ROOT/server/triton.log" 2>/dev/null || echo "No logs yet"
    echo ""
    
    echo "=== FASTAPI SERVER (Port 8001) ==="
    tail -n 20 "$PROJECT_ROOT/server/fastapi.log" 2>/dev/null || echo "No logs yet"
    echo ""
    
    echo "=== FRONTEND SERVER (Port 8000) ==="
    tail -n 20 "$PROJECT_ROOT/client/frontend.log" 2>/dev/null || echo "No logs yet"
    echo ""
}

# Function to follow logs in real-time
follow_logs() {
    echo "🔄 Following logs in real-time..."
    echo "Press Ctrl+C to stop"
    echo ""
    
    # Use tail -f to follow all log files
    tail -f "$PROJECT_ROOT/server/triton.log" \
          "$PROJECT_ROOT/server/fastapi.log" \
          "$PROJECT_ROOT/client/frontend.log" 2>/dev/null
}

# Function to check server health
check_health() {
    echo "🏥 Checking server health..."
    echo ""
    
    # Check Triton
    if curl -s http://localhost:8002/v2/health/ready > /dev/null; then
        echo "✅ Triton server is healthy (port 8002)"
    else
        echo "❌ Triton server is not responding (port 8002)"
    fi
    
    # Check FastAPI
    if curl -s http://localhost:8001/ > /dev/null; then
        echo "✅ FastAPI server is healthy (port 8001)"
    else
        echo "❌ FastAPI server is not responding (port 8001)"
    fi
    
    # Check Frontend
    if curl -s http://localhost:8000 > /dev/null; then
        echo "✅ Frontend server is healthy (port 8000)"
    else
        echo "❌ Frontend server is not responding (port 8000)"
    fi
    echo ""
}

# Function to check if servers are running
check_servers() {
    echo "🔍 Checking server process status..."
    echo ""
    
    # Check Triton
    if ps aux | grep -q "triton_server.py" | grep -v grep; then
        echo "✅ Triton server process is running"
    else
        echo "❌ Triton server process is not running"
    fi
    
    # Check FastAPI
    if ps aux | grep -q "main.py" | grep -v grep; then
        echo "✅ FastAPI server process is running"
    else
        echo "❌ FastAPI server process is not running"
    fi
    
    # Check Frontend
    if ps aux | grep -q "node server.js" | grep -v grep; then
        echo "✅ Frontend server process is running"
    else
        echo "❌ Frontend server process is not running"
    fi
    echo ""
}

# Function to show system resources
show_resources() {
    echo "💻 System Resources"
    echo "=================="
    echo ""
    
    echo "Memory Usage:"
    free -h
    echo ""
    
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print "CPU Usage: " 100 - $1 "%"}'
    echo ""
    
    echo "Disk Usage:"
    df -h | grep -E "(Filesystem|/workspace)"
    echo ""
}

# Function to show port usage
show_ports() {
    echo "🌐 Port Usage"
    echo "============="
    echo ""
    
    echo "Port 8000 (Frontend):"
    netstat -tlnp | grep :8000 || echo "  Not in use"
    echo ""
    
    echo "Port 8001 (FastAPI):"
    netstat -tlnp | grep :8001 || echo "  Not in use"
    echo ""
    
    echo "Port 8002 (Triton):"
    netstat -tlnp | grep :8002 || echo "  Not in use"
    echo ""
}

# Main menu
case "${1:-menu}" in
    "status")
        check_servers
        check_health
        ;;
    "health")
        check_health
        ;;
    "logs")
        show_logs
        ;;
    "follow")
        follow_logs
        ;;
    "resources")
        show_resources
        ;;
    "ports")
        show_ports
        ;;
    "full")
        check_servers
        check_health
        show_resources
        show_ports
        ;;
    "menu"|*)
        echo "Usage: $0 [status|health|logs|follow|resources|ports|full]"
        echo ""
        echo "Commands:"
        echo "  status    - Check if servers are running and healthy"
        echo "  health    - Check server health endpoints"
        echo "  logs      - Show recent logs from all servers"
        echo "  follow    - Follow logs in real-time"
        echo "  resources - Show system resource usage"
        echo "  ports     - Show port usage information"
        echo "  full      - Show complete system status"
        echo ""
        echo "Or run without arguments to see this menu"
        ;;
esac
