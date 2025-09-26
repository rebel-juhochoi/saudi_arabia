// Configuration - Auto-detect API base URL based on current location
const API_BASE_URL = `${window.location.protocol}//${window.location.host}`;
const POLL_INTERVAL = 2000; // 2 seconds

// Global state
let websocket = null;
let canvasContext = null;
let isStreaming = false;
let isDetectionActive = false;
let reconnectAttempts = 0;
let maxReconnectAttempts = 5;
let reconnectDelay = 2000; // 2 seconds

// Simple frame display control
let isDisplaying = false;

// Demo configuration
const DEMO_VIDEO_FILE = 'traffic.mp4';

// DOM Elements
const elements = {
    serverStatus: document.getElementById('server-status'),
    statusText: document.getElementById('status-text'),
    startDemoBtn: document.getElementById('start-demo-btn'),
    stopDemoBtn: document.getElementById('stop-demo-btn'),
    videoPanel: document.getElementById('video-panel'),
    videoCanvas: document.getElementById('video-canvas'),
    streamStatus: document.getElementById('stream-status'), // Hidden for cleaner display
    segmentationToggle: document.getElementById('segmentation-toggle'),
    errorPanel: document.getElementById('error-panel'),
    errorMessage: document.getElementById('error-message'),
    retryBtn: document.getElementById('retry-btn'),
    trafficDashboard: document.getElementById('traffic-dashboard'),
    totalVehicles: document.getElementById('total-vehicles'),
    vehiclesPerMinute: document.getElementById('vehicles-per-minute'),
    roadCounts: document.getElementById('road-counts'), // Hidden for cleaner display
    resetCountersBtn: document.getElementById('reset-counters-btn'),
    exportAnalyticsBtn: document.getElementById('export-analytics-btn')
};

// Utility Functions
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorPanel.style.display = 'block';
    elements.processingPanel.style.display = 'none';
}

function hideError() {
    elements.errorPanel.style.display = 'none';
}

function resetUI() {
    elements.processingPanel.style.display = 'none';
    elements.videoPanel.style.display = 'none';
    elements.errorPanel.style.display = 'none';
    elements.progressFill.style.width = '0%';
    elements.progressText.textContent = '0%';
    currentJobId = null;
}

// API Functions
async function checkServerStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            elements.serverStatus.className = 'status-indicator online';
            elements.statusText.textContent = 'Online';
            return true;
        } else {
            throw new Error('Server responded with error');
        }
    } catch (error) {
        elements.serverStatus.className = 'status-indicator offline';
        elements.statusText.textContent = 'Offline';
        return false;
    }
}


function startDemoStream() {
    reconnectAttempts = 0;
    isDetectionActive = true;
    
    // Update UI
    elements.startDemoBtn.style.display = 'none';
    elements.stopDemoBtn.style.display = 'inline-flex';
    elements.videoPanel.style.display = 'block';
    showTrafficDashboard();
    
    connectDemoWebSocket();
}

function connectDemoWebSocket() {
    // Create WebSocket connection
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/demo-stream`;
    
    console.log('Attempting to connect to demo:', wsUrl);
    if (elements.streamStatus) {
        elements.streamStatus.textContent = 'Connecting...';
        elements.streamStatus.className = 'stream-status';
    }
    
    // Clean up any existing connection
    if (websocket) {
        console.log('Closing existing WebSocket connection');
        websocket.onclose = null; // Prevent close handler from firing
        websocket.onerror = null;
        websocket.onmessage = null;
        try {
            websocket.close(1000, 'New connection');
        } catch (e) {
            console.log('Error closing old websocket:', e);
        }
        websocket = null;
    }
    
    // Clean up previous state
    cleanupStreamingState();
    
    // Wait a moment before creating new connection
    setTimeout(() => {
        websocket = new WebSocket(wsUrl);
        websocket.binaryType = 'arraybuffer'; // Ensure binary data is received as ArrayBuffer
        canvasContext = elements.videoCanvas.getContext('2d');
        isStreaming = true;
        
        websocket.onopen = function(event) {
        console.log('WebSocket connected to:', wsUrl);
        reconnectAttempts = 0; // Reset reconnect attempts on successful connection
        if (elements.streamStatus) {
            elements.streamStatus.textContent = 'Connected - Starting video stream...';
            elements.streamStatus.className = 'stream-status processing';
        }
    };
    
    websocket.onmessage = function(event) {
        try {
            console.log('WebSocket message received, type:', typeof event.data, 'constructor:', event.data.constructor.name);
            
            if (event.data instanceof ArrayBuffer) {
                // Handle binary frame data (ArrayBuffer)
                console.log('Processing ArrayBuffer frame, size:', event.data.byteLength);
                handleBinaryFrame(event.data);
            } else if (event.data instanceof Blob) {
                // Handle binary frame data (Blob) - convert to ArrayBuffer
                console.log('Processing Blob frame, size:', event.data.size);
                const reader = new FileReader();
                reader.onload = function() {
                    handleBinaryFrame(reader.result);
                };
                reader.readAsArrayBuffer(event.data);
            } else {
                // Handle JSON messages
                console.log('Processing JSON message:', event.data);
                const data = JSON.parse(event.data);
                handleStreamMessage(data);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
            console.error('Event data type:', typeof event.data);
            console.error('Event data:', event.data);
        }
    };
    
    websocket.onclose = function(event) {
        console.log('WebSocket closed, code:', event.code, 'reason:', event.reason);
        
        // Clean up state regardless of close reason
        cleanupStreamingState();
        
        if (isStreaming && !websocket?._manualClose) {
            // Only attempt one reconnection for unexpected closures (not manual stops)
            if (reconnectAttempts === 0 && event.code !== 1000 && event.code !== 1001) {
                reconnectAttempts++;
                if (elements.streamStatus) {
                    elements.streamStatus.textContent = 'Connection lost - Reconnecting...';
                    elements.streamStatus.className = 'stream-status';
                }
                
                setTimeout(() => {
                    if (isStreaming) {
                        console.log('Attempting reconnection...');
                        connectDemoWebSocket();
                    }
                }, 2000);
            } else {
            if (elements.streamStatus) {
                elements.streamStatus.textContent = 'Connection closed';
                elements.streamStatus.className = 'stream-status error';
            }
                stopDemoStream();
            }
        } else {
        if (elements.streamStatus) {
            elements.streamStatus.textContent = 'Stream stopped';
            elements.streamStatus.className = 'stream-status';
        }
        }
    };
    
        websocket.onerror = function(error) {
            console.error('WebSocket error:', error);
        if (elements.streamStatus) {
            elements.streamStatus.textContent = 'Connection error - Retrying...';
            elements.streamStatus.className = 'stream-status error';
        }
        };
    }, 100); // Small delay to ensure clean connection
}

function handleBinaryFrame(binaryData) {
    // Convert ArrayBuffer to Blob and create image
    const blob = new Blob([binaryData], { type: 'image/jpeg' });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    
    img.onload = function() {
        // Draw the image to canvas
        if (canvasContext && elements.videoCanvas) {
            canvasContext.drawImage(img, 0, 0, elements.videoCanvas.width, elements.videoCanvas.height);
        }
        // Clean up the object URL
        URL.revokeObjectURL(url);
    };
    
    img.onerror = function() {
        console.error('Error loading binary frame image');
        URL.revokeObjectURL(url);
    };
    
    img.src = url;
}

function handleStreamMessage(data) {
    switch (data.type) {
        case 'video_info':
            // Set canvas dimensions based on video
            const aspectRatio = data.width / data.height;
            const maxWidth = 800;
            const canvasWidth = Math.min(maxWidth, data.width);
            const canvasHeight = canvasWidth / aspectRatio;
            
            elements.videoCanvas.width = canvasWidth;
            elements.videoCanvas.height = canvasHeight;
            
            const videoFpsText = data.fps ? ` (Video: ${data.fps} FPS)` : '';
            const targetFpsText = data.target_fps ? ` (Target: ${data.target_fps} FPS)` : '';
            const streamingFpsText = data.adaptive_fps ? ` (Streaming: ${data.adaptive_fps} FPS)` : '';
            const binaryText = data.binary_transmission ? ' [Binary]' : '';
            const ultraFastText = data.ultra_fast_mode ? ' [Ultra-Fast Pipeline]' : '';
            const preloadText = data.preload_seconds ? ` [Preload: ${data.preload_seconds}s]` : '';
            if (elements.streamStatus) {
                elements.streamStatus.textContent = `Processing ${data.width}x${data.height} video${videoFpsText}${targetFpsText}${streamingFpsText}${binaryText}${ultraFastText}${preloadText}`;
                elements.streamStatus.className = 'stream-status processing';
            }
            break;
            
        case 'frame_metadata':
            // Handle metadata for binary frames
            const skipText = data.frames_skipped > 0 ? ` [Skipped: ${data.frames_skipped}]` : '';
            const adaptiveText = data.adaptive_fps ? ` [FPS: ${data.adaptive_fps}]` : '';
            const pipelineText = data.pipeline_mode === 'ultra_fast' ? ' [Pipeline]' : '';
            if (elements.streamStatus) {
                elements.streamStatus.textContent = `Streaming live video${skipText}${adaptiveText}${pipelineText}`;
                elements.streamStatus.className = 'stream-status processing';
            }
            
            // Update traffic counts if available
            if (data.traffic_counts) {
                updateTrafficDashboard(data.traffic_counts);
                showTrafficDashboard();
            }
            break;
            
        case 'frame':
            // Display single frame on canvas
            displayFrame(data.frame);
            
            // Update status
            if (elements.streamStatus) {
                elements.streamStatus.textContent = 'Streaming live video';
                elements.streamStatus.className = 'stream-status processing';
            }
            
            // Update traffic counts if available
            if (data.traffic_counts) {
                updateTrafficDashboard(data.traffic_counts);
                showTrafficDashboard();
            }
            break;
            
        case 'complete':
        case 'stream_ended':
            if (elements.streamStatus) {
                elements.streamStatus.textContent = 'Stream ended - Video will restart shortly';
                elements.streamStatus.className = 'stream-status processing';
            }
            // Don't set isStreaming to false - allow reconnection
            break;
            
        case 'error':
            showError(data.message);
            if (elements.streamStatus) {
                elements.streamStatus.textContent = 'Error occurred';
                elements.streamStatus.className = 'stream-status error';
            }
            isStreaming = false;
            break;
            
        case 'control_response':
            console.log('Control response:', data.message);
            break;
            
        case 'loop_restart':
            console.log('Video looping:', data.message);
            const loopCount = data.loop_count || 0;
            if (elements.streamStatus) {
                elements.streamStatus.textContent = `Video restarting... (Loop #${loopCount})`;
                elements.streamStatus.className = 'stream-status processing';
            }
            break;
            
        case 'traffic_update':
            // Handle real-time traffic data updates
            if (data.traffic_counts) {
                updateTrafficDashboard(data.traffic_counts);
                showTrafficDashboard();
            }
            break;
    }
}

function displayFrame(frameData) {
    if (!canvasContext || !frameData) return;
    
    const img = new Image();
    img.onload = function() {
        if (canvasContext && isStreaming) {
            // Clear canvas and draw frame
            canvasContext.clearRect(0, 0, elements.videoCanvas.width, elements.videoCanvas.height);
            canvasContext.drawImage(img, 0, 0, elements.videoCanvas.width, elements.videoCanvas.height);
        }
    };
    img.src = 'data:image/jpeg;base64,' + frameData;
}


function cleanupStreamingState() {
    console.log('Cleaning up streaming state...');
    
    // Clear canvas
    if (canvasContext) {
        canvasContext.clearRect(0, 0, elements.videoCanvas.width, elements.videoCanvas.height);
    }
    
    console.log('Streaming state cleanup completed');
}

function stopDemoStream() {
    console.log('Stopping demo stream...');
    isStreaming = false;
    isDetectionActive = false;
    reconnectAttempts = 0;
    
    // Update UI
    elements.startDemoBtn.style.display = 'inline-flex';
    elements.stopDemoBtn.style.display = 'none';
    hideTrafficDashboard();
    
    // Clean up display state
    cleanupStreamingState();
    
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        // Send stop command to server
        try {
            websocket.send(JSON.stringify({
                type: 'control',
                command: 'stop'
            }));
        } catch (e) {
            console.log('Error sending stop command:', e);
        }
    }
    
    // Close websocket connection
    if (websocket) {
        // Set a flag to prevent reconnection attempts
        websocket._manualClose = true;
        websocket.close(1000, 'Manual stop');
        websocket = null;
    }
    
    if (elements.streamStatus) {
        elements.streamStatus.textContent = 'Stopped';
        elements.streamStatus.className = 'stream-status';
    }
}

function sendSegmentationToggle(enabled) {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({
            type: 'control',
            command: 'toggle_segmentation',
            value: enabled
        }));
        console.log(`Segmentation ${enabled ? 'enabled' : 'disabled'}`);
    }
}

async function checkJobStatus(jobId) {
    try {
        const response = await fetch(`${API_BASE_URL}/status/${jobId}`);
        if (!response.ok) throw new Error('Failed to check status');
        return await response.json();
    } catch (error) {
        throw error;
    }
}

async function downloadVideo(jobId) {
    try {
        const response = await fetch(`${API_BASE_URL}/download/${jobId}`);
        if (!response.ok) throw new Error('Failed to download video');
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `processed_video_${jobId}.mp4`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch (error) {
        showError('Failed to download video: ' + error.message);
    }
}

async function cleanupJob(jobId) {
    try {
        await fetch(`${API_BASE_URL}/cleanup/${jobId}`, { method: 'DELETE' });
    } catch (error) {
        console.warn('Failed to cleanup job:', error);
    }
}

// Processing Functions
async function pollJobStatus(jobId) {
    try {
        const status = await checkJobStatus(jobId);
        
        // Update progress
        if (status.progress !== undefined) {
            elements.progressFill.style.width = `${status.progress}%`;
            elements.progressText.textContent = `${status.progress}%`;
        }
        
        elements.statusMessage.textContent = status.message;
        
        if (status.status === 'completed') {
            // Processing completed
            elements.statusMessage.textContent = 'Processing completed! Loading video...';
            
            // Download and display the processed video
            const downloadResponse = await fetch(`${API_BASE_URL}/download/${jobId}`);
            if (downloadResponse.ok) {
                const blob = await downloadResponse.blob();
                const videoUrl = URL.createObjectURL(blob);
                elements.processedVideo.src = videoUrl;
                
                // Show video panel
                elements.processingPanel.style.display = 'none';
                elements.videoPanel.style.display = 'block';
                
                // Set up download button
                elements.downloadBtn.onclick = () => downloadVideo(jobId);
                
            } else {
                throw new Error('Failed to load processed video');
            }
            
        } else if (status.status === 'error') {
            throw new Error(status.message);
        } else {
            // Continue polling
            setTimeout(() => pollJobStatus(jobId), POLL_INTERVAL);
        }
        
    } catch (error) {
        showError('Processing failed: ' + error.message);
        if (jobId) {
            cleanupJob(jobId);
        }
    }
}

// Event Handlers
function setupEventListeners() {
    // Start demo button
    if (elements.startDemoBtn) {
        elements.startDemoBtn.addEventListener('click', () => {
            console.log('Starting demo...');
            hideError();
            try {
                startDemoStream();
            } catch (error) {
                showError('Failed to start demo: ' + error.message);
            }
        });
    }
    
    // Stop demo button
    if (elements.stopDemoBtn) {
        elements.stopDemoBtn.addEventListener('click', () => {
            console.log('Stopping demo...');
            stopDemoStream();
        });
    }
    
    // Segmentation toggle
    elements.segmentationToggle.addEventListener('change', (e) => {
        const enabled = e.target.checked;
        sendSegmentationToggle(enabled);
    });
    
    // Traffic dashboard controls
    if (elements.resetCountersBtn) {
        elements.resetCountersBtn.addEventListener('click', resetCounters);
    }
    
    if (elements.exportAnalyticsBtn) {
        elements.exportAnalyticsBtn.addEventListener('click', exportAnalytics);
    }
    
    // Retry button
    if (elements.retryBtn) {
        elements.retryBtn.addEventListener('click', () => {
            hideError();
            if (isDetectionActive) {
                startDemoStream();
            }
        });
    }
}


// Removed updateProcessButton since we no longer have a process button

// Initialization
async function initialize() {
    console.log('Initializing Vehicle Detection Video Processor...');
    
    setupEventListeners();
    
    // Check server status
    const serverOnline = await checkServerStatus();
    
    if (!serverOnline) {
        showError('Server is offline. Please start the FastAPI server and refresh the page.');
    }
    
    // Set up periodic server status checks
    setInterval(checkServerStatus, 30000); // Check every 30 seconds
}

// Traffic Dashboard Functions
function showTrafficDashboard() {
    if (elements.trafficDashboard) {
        elements.trafficDashboard.style.display = 'block';
    }
}

function hideTrafficDashboard() {
    if (elements.trafficDashboard) {
        elements.trafficDashboard.style.display = 'none';
    }
}

function updateTrafficDashboard(trafficData) {
    if (!trafficData || !elements.trafficDashboard) return;
    
    // Update total statistics
    elements.totalVehicles.textContent = trafficData.total_vehicles || 0;
    
    // Calculate vehicles per minute
    const sessionDuration = trafficData.session_duration || 0;
    const vehiclesPerMin = sessionDuration > 0 ? 
        (trafficData.total_vehicles / (sessionDuration / 60)).toFixed(1) : '0.0';
    elements.vehiclesPerMinute.textContent = vehiclesPerMin;
    
    // Update road-specific counts
    updateRoadCounts(trafficData.road_counts || {});
    
    // Add real-time indicator
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    
    // Update or create real-time indicator
    let realTimeIndicator = document.getElementById('real-time-indicator');
    if (!realTimeIndicator) {
        realTimeIndicator = document.createElement('div');
        realTimeIndicator.id = 'real-time-indicator';
        realTimeIndicator.className = 'real-time-indicator';
        realTimeIndicator.innerHTML = '<i class="fas fa-circle"></i> Live';
        elements.trafficDashboard.appendChild(realTimeIndicator);
    }
    
    // Update timestamp
    realTimeIndicator.title = `Last updated: ${timeString}`;
}

function updateRoadCounts(roadCounts) {
    if (!elements.roadCounts) return;
    
    elements.roadCounts.innerHTML = '';
    
    Object.entries(roadCounts).forEach(([roadId, roadData]) => {
        const roadItem = document.createElement('div');
        roadItem.className = 'road-count-item';
        
        const total = roadData.total || 0;
        const byType = roadData.by_type || {};
        
        roadItem.innerHTML = `
            <h4>Road ${roadId}: ${total} vehicles</h4>
            <div class="vehicle-types">
                ${Object.entries(byType)
                    .filter(([type, count]) => count > 0)
                    .map(([type, count]) => 
                        `<span class="vehicle-type-badge">${type}: ${count}</span>`
                    ).join('')}
            </div>
        `;
        
        elements.roadCounts.appendChild(roadItem);
    });
}

async function resetCounters() {
    if (!isDetectionActive) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/reset-counters`, {
            method: 'POST'
        });
        
        if (response.ok) {
            console.log('Counters reset successfully');
            // Clear dashboard immediately
            elements.totalVehicles.textContent = '0';
            elements.vehiclesPerMinute.textContent = '0.0';
            elements.roadCounts.innerHTML = '';
        } else {
            console.error('Failed to reset counters');
        }
    } catch (error) {
        console.error('Error resetting counters:', error);
    }
}

async function exportAnalytics() {
    if (!isDetectionActive) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/traffic-analytics`);
        
        if (response.ok) {
            const analytics = await response.json();
            
            // Create and download JSON file
            const blob = new Blob([JSON.stringify(analytics, null, 2)], {
                type: 'application/json'
            });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `traffic-analytics-demo-${new Date().toISOString().slice(0, 19)}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            console.log('Analytics exported successfully');
        } else {
            console.error('Failed to export analytics');
        }
    } catch (error) {
        console.error('Error exporting analytics:', error);
    }
}

// Start the application
document.addEventListener('DOMContentLoaded', initialize);
