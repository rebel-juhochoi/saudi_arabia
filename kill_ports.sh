#!/bin/bash

echo "🔪 Killing processes on ports 8000-8001..."

for port in 8000 8001; do
    echo "🔍 Checking port $port..."
    
    # Convert port to hex for /proc/net/tcp lookup
    hex_port=$(printf '%X' $port)
    
    # Find and kill processes using this port (multiple attempts)
    for attempt in 1 2 3; do
        killed_any=false
        
        cat /proc/net/tcp | grep ":$hex_port" | while read line; do
            # Extract inode from the line
            inode=$(echo $line | awk '{print $10}')
            
            # Skip if inode is 0 (no socket)
            if [ "$inode" != "0" ]; then
                # Find PID that owns this socket inode
                pid=$(find /proc/*/fd -ls 2>/dev/null | grep "socket:\[$inode\]" | head -1 | awk -F'/' '{print $3}')
                
                if [ -n "$pid" ]; then
                    echo "🔪 Killing PID $pid on port $port (attempt $attempt)"
                    kill -9 $pid 2>/dev/null || true
                    killed_any=true
                fi
            fi
        done
        
        # Wait a bit between attempts
        sleep 1
        
        # Check if port is now free
        if python3 -c "import socket; s=socket.socket(); s.bind(('localhost', $port)); s.close()" 2>/dev/null; then
            break
        fi
    done
done

# Also kill common server processes
pkill -9 -f 'uvicorn.*server:app' 2>/dev/null || true
pkill -9 -f 'python.*server.py' 2>/dev/null || true

# Wait for processes to die
sleep 2

echo "✅ Port cleanup completed!"

# Verify ports are free
echo "🔍 Verifying ports are free..."
for port in 8000 8001 8002 8003; do
    if python3 -c "import socket; s=socket.socket(); s.bind(('localhost', $port)); s.close()" 2>/dev/null; then
        echo "✅ Port $port is free"
    else
        echo "❌ Port $port is still in use"
    fi
done
