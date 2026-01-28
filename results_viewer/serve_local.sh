#!/bin/bash
# Script to serve the results viewer on your local network

PORT=8000

# Find local IP address
echo "Finding your local IP address..."
if command -v hostname &> /dev/null; then
    LOCAL_IP=$(hostname -I | awk '{print $1}')
elif command -v ip &> /dev/null; then
    LOCAL_IP=$(ip route get 8.8.8.8 2>/dev/null | grep -oP 'src \K\S+')
else
    LOCAL_IP=$(ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1' | head -1)
fi

if [ -z "$LOCAL_IP" ]; then
    echo "Could not determine local IP. Using 0.0.0.0 (accessible from all interfaces)"
    LOCAL_IP="0.0.0.0"
fi

echo ""
echo "=========================================="
echo "Starting local server..."
echo "=========================================="
echo "Local access:    http://localhost:$PORT/results_viewer/"
echo "Network access:  http://$LOCAL_IP:$PORT/results_viewer/"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

# Change to repository root (parent of results_viewer)
cd "$(dirname "$0")/.." || exit 1

# Start Python HTTP server
if command -v python3 &> /dev/null; then
    python3 -m http.server "$PORT" --bind 0.0.0.0
elif command -v python &> /dev/null; then
    python -m http.server "$PORT" --bind 0.0.0.0
else
    echo "Error: Python not found. Please install Python 3."
    exit 1
fi
