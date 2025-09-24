#!/bin/bash

echo "============================================================"
echo "🚀 STARTING BOTH VIDEO PROCESSING API SERVERS"
echo "============================================================"

# Use the dedicated port killer script
./kill_ports.sh

echo "🚀 Starting both servers in separate terminals..."
echo "📡 Gender Detection Server: http://localhost:8000"
echo "📡 Traffic Detection Server: http://localhost:8001"
echo "📚 API documentation available at /docs endpoints"
echo "⚠️  Use Ctrl+C in each terminal to stop individual servers"
echo "------------------------------------------------------------"

# Check if gnome-terminal is available
if command -v gnome-terminal &> /dev/null; then
    echo "🖥️  Opening servers in divided terminal view..."
    
    # Start both servers in a split terminal view
    gnome-terminal --title="Saudi Arabia Video Processing Servers" \
        --window-with-profile=Default \
        -- bash -c "
            echo '🚀 Starting Gender Detection Server (Port 8000)...';
            cd /workspace/global/saudi_arabia/gender_detection;
            uv run start_server.py;
            echo 'Press Enter to close this terminal...';
            read
        " \
        --tab --title="Traffic Detection Server" \
        -- bash -c "
            echo '🚀 Starting Traffic Detection Server (Port 8001)...';
            cd /workspace/global/saudi_arabia/traffic_detection;
            uv run server.py;
            echo 'Press Enter to close this terminal...';
            read
        "
    
    echo "✅ Both servers started in separate terminal tabs!"
    echo "🌐 Gender Detection: http://localhost:8000"
    echo "🌐 Traffic Detection: http://localhost:8001"
    
elif command -v xterm &> /dev/null; then
    echo "🖥️  Opening servers in separate xterm windows..."
    
    # Start Gender Detection server
    xterm -title "Gender Detection Server (Port 8000)" -e "
        echo '🚀 Starting Gender Detection Server (Port 8000)...';
        cd /workspace/global/saudi_arabia/gender_detection;
        uv run start_server.py;
        echo 'Press Enter to close this terminal...';
        read
    " &
    
    # Start Traffic Detection server
    xterm -title "Traffic Detection Server (Port 8001)" -e "
        echo '🚀 Starting Traffic Detection Server (Port 8001)...';
        cd /workspace/global/saudi_arabia/traffic_detection;
        uv run server.py;
        echo 'Press Enter to close this terminal...';
        read
    " &
    
    echo "✅ Both servers started in separate xterm windows!"
    echo "🌐 Gender Detection: http://localhost:8000"
    echo "🌐 Traffic Detection: http://localhost:8001"
    
else
    echo "❌ Neither gnome-terminal nor xterm found. Falling back to sequential execution..."
    echo "⚠️  Note: This will run servers sequentially, not in separate terminals."
    echo ""
    
    echo "🚀 Starting Gender Detection Server (Port 8000)..."
    echo "📡 Server will be available at: http://localhost:8000"
    echo "📚 API documentation: http://localhost:8000/docs"
    echo "⚠️  Press Ctrl+C to stop the server"
    echo "------------------------------------------------------------"
    
    # Run Gender Detection server
    cd /workspace/global/saudi_arabia/gender_detection
    uv run start_server.py &
    GENDER_PID=$!
    
    # Wait a moment for the first server to start
    sleep 3
    
    echo ""
    echo "🚀 Starting Traffic Detection Server (Port 8001)..."
    echo "📡 Server will be available at: http://localhost:8001"
    echo "📚 API documentation: http://localhost:8001/docs"
    echo "⚠️  Press Ctrl+C to stop the server"
    echo "------------------------------------------------------------"
    
    # Run Traffic Detection server
    cd /workspace/global/saudi_arabia/traffic_detection
    uv run server.py &
    TRAFFIC_PID=$!
    
    echo ""
    echo "✅ Both servers started in background!"
    echo "🌐 Gender Detection: http://localhost:8000"
    echo "🌐 Traffic Detection: http://localhost:8001"
    echo "⚠️  Use 'kill $GENDER_PID $TRAFFIC_PID' to stop both servers"
    
    # Wait for both processes
    wait $GENDER_PID $TRAFFIC_PID
fi
