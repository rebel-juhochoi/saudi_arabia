// Configuration - Auto-detect API base URL based on current location
const API_BASE_URL = `${window.location.protocol}//${window.location.host}`;
const POLL_INTERVAL = 2000; // 2 seconds

// Global state
let currentJobId = null;
let availableConfigs = {};
let selectedVideo = null;
let websocket = null;
let canvasContext = null;
let isStreaming = false;
let reconnectAttempts = 0;
let maxReconnectAttempts = 5;
let reconnectDelay = 2000; // 2 seconds
let currentConfigName = null;

// Simple frame display control
let isDisplaying = false;

// Video mapping - maps display names to config names and file paths
const VIDEO_MAPPING = {
    '01': { config: '01', file: 'data/inputs/01.mp4' },
    '02': { config: '02', file: 'data/inputs/02.mp4' },
    '03': { config: '03', file: 'data/inputs/03.mp4' },
    '04': { config: '04', file: 'data/inputs/04.mp4' }
};

// DOM Elements
const elements = {
    serverStatus: document.getElementById('server-status'),
    statusText: document.getElementById('status-text'),
    videoSelect: document.getElementById('video-select'),
    processingPanel: document.getElementById('processing-panel'),
    progressFill: document.getElementById('progress-fill'),
    progressText: document.getElementById('progress-text'),
    statusMessage: document.getElementById('status-message'),
    jobId: document.getElementById('job-id'),
    usedConfig: document.getElementById('used-config'),
    videoPanel: document.getElementById('video-panel'),
    videoCanvas: document.getElementById('video-canvas'),
    streamStatus: document.getElementById('stream-status'),
    stopStreamBtn: document.getElementById('stop-stream-btn'),
    segmentationToggle: document.getElementById('segmentation-toggle'),
    errorPanel: document.getElementById('error-panel'),
    errorMessage: document.getElementById('error-message'),
    retryBtn: document.getElementById('retry-btn'),
    manSwitch: document.getElementById('man-switch'),
    womanSwitch: document.getElementById('woman-switch'),
    alarmStatus: document.getElementById('alarm-status'),
    alarmText: document.getElementById('alarm-text')
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
    resetAlarm();
}

// Alarm System Functions
function setAlarm(type, message) {
    elements.alarmStatus.className = `alarm-status ${type}`;
    elements.alarmText.textContent = message;
    
    // Update icon based on alarm type
    const icon = elements.alarmStatus.querySelector('i');
    if (type === 'normal') {
        icon.className = 'fas fa-check-circle';
    } else if (type === 'warning') {
        icon.className = 'fas fa-exclamation-triangle';
    } else if (type === 'alert') {
        icon.className = 'fas fa-exclamation-circle';
    }
}

function resetAlarm() {
    setAlarm('normal', 'Zone guidance - Ready');
}

function checkZoneAlarm(detectedGenders) {
    const manEnabled = elements.manSwitch.checked;
    const womanEnabled = elements.womanSwitch.checked;
    const currentVideo = elements.videoSelect.value;
    
    console.log('=== MAN VIDEO ALARM DEBUG ===');
    console.log('Video:', currentVideo);
    console.log('Man switch:', manEnabled);
    console.log('Woman switch:', womanEnabled);
    console.log('Detected zones:', detectedGenders);
    
    // Case 1: Both switches ON
    if (manEnabled && womanEnabled) {
        console.log('Both zones ON - checking if only single zone detected');
        if (detectedGenders.length === 1) {
            console.log('Only single zone detected - GUIDANCE NEEDED');
            setAlarm('warning', 'Zone guidance - Both access needed');
        } else if (detectedGenders.length === 2) {
            console.log('Multiple zones detected - GREEN');
            setAlarm('normal', 'Zone guidance - Both access active');
        } else {
            console.log('No zone detected - YELLOW');
            setAlarm('warning', 'Zone guidance - No activity detected');
        }
        return;
    }
    
    // Case 2: Only Man switch ON
    if (manEnabled && !womanEnabled) {
        console.log('Only Man switch ON - checking detected zones');
        // Check for Woman or multiple zones
        if (detectedGenders.length > 1 || detectedGenders.includes('Woman')) {
            console.log('Woman detected or multiple zones - GUIDANCE NEEDED');
            setAlarm('alert', 'Zone guidance needed');
        } else {
            console.log('Only Man detected - GREEN');
            setAlarm('normal', 'Zone guidance - Single zone active');
        }
        return;
    }
    
    // Case 3: Only Woman switch ON
    if (womanEnabled && !manEnabled) {
        console.log('Only Woman switch ON - checking detected zones');
        // Check for Man or multiple zones
        if (detectedGenders.length > 1 || detectedGenders.includes('Man')) {
            console.log('Man detected or multiple zones - GUIDANCE NEEDED');
            setAlarm('alert', 'Zone guidance needed');
        } else {
            console.log('Only Woman detected - GREEN');
            setAlarm('normal', 'Zone guidance - Single zone active');
        }
        return;
    }
    
    // Case 4: No switches ON
    if (!manEnabled && !womanEnabled) {
        console.log('No switches ON - should show normal');
        setAlarm('normal', 'Zone guidance - Standby');
        return;
    }
    
    console.log('=== END DEBUG ===');
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

async function loadVideoOptions() {
    try {
        const response = await fetch(`${API_BASE_URL}/configs`);
        if (!response.ok) throw new Error('Failed to load configurations');
        
        const data = await response.json();
        availableConfigs = data.configurations;
        
        // Populate video select with simplified names
        elements.videoSelect.innerHTML = '';
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select a video...';
        elements.videoSelect.appendChild(defaultOption);
        
        Object.keys(VIDEO_MAPPING).forEach(displayName => {
            const option = document.createElement('option');
            option.value = displayName;
            option.textContent = displayName;
            elements.videoSelect.appendChild(option);
        });
        
    } catch (error) {
        console.error('Failed to load video options:', error);
        elements.videoSelect.innerHTML = '<option value="">Failed to load videos</option>';
    }
}

function startVideoStreaming(configName) {
    if (!configName) {
        throw new Error('No video configuration provided');
    }
    
    currentConfigName = configName;
    reconnectAttempts = 0;
    connectWebSocket(configName);
}

function createFreshConnection(configName) {
    // Simple fresh connection without cleanup conflicts
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/stream-video/${configName}`;
    
    console.log('Creating fresh WebSocket connection to:', wsUrl);
    elements.streamStatus.textContent = 'Creating fresh connection...';
    elements.streamStatus.className = 'stream-status';
    
    // Create new WebSocket directly (no cleanup of existing)
    websocket = new WebSocket(wsUrl);
    websocket.binaryType = 'arraybuffer';
    canvasContext = elements.videoCanvas.getContext('2d');
    isStreaming = true;
    
    websocket.onopen = function(event) {
        console.log('WebSocket connected to:', wsUrl);
        reconnectAttempts = 0;
        elements.streamStatus.textContent = 'Connected - Starting video stream...';
        elements.streamStatus.className = 'stream-status processing';
    };
    
    websocket.onmessage = function(event) {
        try {
            if (event.data instanceof ArrayBuffer) {
                handleBinaryFrame(event.data);
            } else if (event.data instanceof Blob) {
                const reader = new FileReader();
                reader.onload = function() {
                    handleBinaryFrame(reader.result);
                };
                reader.readAsArrayBuffer(event.data);
            } else {
                const data = JSON.parse(event.data);
                handleStreamMessage(data);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };
    
    websocket.onclose = function(event) {
        console.log('WebSocket closed, code:', event.code, 'reason:', event.reason);
        cleanupStreamingState();
        
        if (isStreaming && currentConfigName && !websocket?._manualClose) {
            if (event.code === 1006) {
                console.log('WebSocket abnormally closed (1006) - Creating fresh connection...');
                elements.streamStatus.textContent = 'Connection lost - Creating fresh connection...';
                elements.streamStatus.className = 'stream-status';
                
                reconnectAttempts = 0;
                setTimeout(() => {
                    if (isStreaming && currentConfigName) {
                        createFreshConnection(currentConfigName);
                    }
                }, 1000);
            } else if (reconnectAttempts === 0 && event.code !== 1000 && event.code !== 1001) {
                reconnectAttempts++;
                elements.streamStatus.textContent = 'Connection lost - Reconnecting...';
                elements.streamStatus.className = 'stream-status';
                
                setTimeout(() => {
                    if (isStreaming && currentConfigName) {
                        connectWebSocket(currentConfigName);
                    }
                }, 2000);
            }
        }
    };
    
    websocket.onerror = function(error) {
        console.error('WebSocket error:', error);
        elements.streamStatus.textContent = 'Connection error - Will retry on close...';
        elements.streamStatus.className = 'stream-status error';
    };
}

function connectWebSocket(configName) {
    // Create WebSocket connection
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws/stream-video/${configName}`;
    
    console.log('Attempting to connect to:', wsUrl);
    elements.streamStatus.textContent = 'Connecting...';
    elements.streamStatus.className = 'stream-status';
    
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
        elements.streamStatus.textContent = 'Connected - Starting video stream...';
        elements.streamStatus.className = 'stream-status processing';
    };
    
    websocket.onmessage = function(event) {
        try {
            if (event.data instanceof ArrayBuffer) {
                // Handle binary frame data (ArrayBuffer)
                handleBinaryFrame(event.data);
            } else if (event.data instanceof Blob) {
                // Handle binary frame data (Blob) - convert to ArrayBuffer
                const reader = new FileReader();
                reader.onload = function() {
                    handleBinaryFrame(reader.result);
                };
                reader.readAsArrayBuffer(event.data);
            } else {
                // Handle JSON messages
                const data = JSON.parse(event.data);
                handleStreamMessage(data);
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };
    
    websocket.onclose = function(event) {
        console.log('WebSocket closed, code:', event.code, 'reason:', event.reason);
        console.log('Current streaming state:', { isStreaming, currentConfigName, manualClose: websocket?._manualClose });
        
        // Clean up state regardless of close reason
        cleanupStreamingState();
        
        if (isStreaming && currentConfigName && !websocket?._manualClose) {
            // Handle different closure codes
            if (event.code === 1006) {
                // Code 1006: Abnormal closure - create fresh connection instead of reconnecting
                console.log('WebSocket abnormally closed (1006) - Creating fresh connection...');
                elements.streamStatus.textContent = 'Connection lost - Creating fresh connection...';
                elements.streamStatus.className = 'stream-status';
                
                // Clean up current connection
                cleanupStreamingState();
                
                // Reset reconnect attempts for fresh connection
                reconnectAttempts = 0;
                
                // Create fresh connection after a short delay
                setTimeout(() => {
                    if (isStreaming && currentConfigName) {
                        console.log('Creating fresh WebSocket connection...');
                        createFreshConnection(currentConfigName);
                    }
                }, 1000);
            } else if (reconnectAttempts === 0 && event.code !== 1000 && event.code !== 1001) {
                // Other unexpected closures - attempt one reconnection
                reconnectAttempts++;
                elements.streamStatus.textContent = 'Connection lost - Reconnecting...';
                elements.streamStatus.className = 'stream-status';
                
                setTimeout(() => {
                    if (isStreaming && currentConfigName) {
                        console.log('Attempting reconnection...');
                        connectWebSocket(currentConfigName);
                    }
                }, 2000);
            } else {
                elements.streamStatus.textContent = 'Connection closed';
                elements.streamStatus.className = 'stream-status error';
                isStreaming = false;
                isDisplaying = false;
                currentConfigName = null;
            }
        } else {
            elements.streamStatus.textContent = 'Stream stopped';
            elements.streamStatus.className = 'stream-status';
        }
    };
    
        websocket.onerror = function(error) {
            console.error('WebSocket error:', error);
            elements.streamStatus.textContent = 'Connection error - Will retry on close...';
            elements.streamStatus.className = 'stream-status error';
            // Don't try to reconnect immediately on error - wait for close event
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
            elements.streamStatus.textContent = `Processing ${data.width}x${data.height} video${videoFpsText}${targetFpsText}${streamingFpsText}${binaryText}${ultraFastText}${preloadText}`;
            elements.streamStatus.className = 'stream-status processing';
            break;
            
        case 'frame_metadata':
            // Handle metadata for binary frames
            const metadataTrackText = data.active_tracks > 0 ? ` (${data.active_tracks} tracks)` : '';
            const skipText = data.frames_skipped > 0 ? ` [Skipped: ${data.frames_skipped}]` : '';
            const adaptiveText = data.adaptive_fps ? ` [FPS: ${data.adaptive_fps}]` : '';
            const pipelineText = data.pipeline_mode === 'ultra_fast' ? ' [Pipeline]' : '';
            elements.streamStatus.textContent = `Streaming live video${metadataTrackText}${skipText}${adaptiveText}${pipelineText}`;
            elements.streamStatus.className = 'stream-status processing';
            
            // Check for zone alarm if zone data is available
            if (data.detected_genders) {
                checkZoneAlarm(data.detected_genders);
            }
            break;
            
        case 'frame':
            // Display single frame on canvas
            displayFrame(data.frame);
            
            // Update status with track info
            const frameTrackText = data.active_tracks > 0 ? ` (${data.active_tracks} tracks)` : '';
            elements.streamStatus.textContent = `Streaming live video${frameTrackText}`;
            elements.streamStatus.className = 'stream-status processing';
            
            // Check for zone alarm if zone data is available
            if (data.detected_genders) {
                checkZoneAlarm(data.detected_genders);
            }
            break;
            
        case 'complete':
        case 'stream_ended':
            elements.streamStatus.textContent = 'Stream ended - Video will restart shortly';
            elements.streamStatus.className = 'stream-status processing';
            // Don't set isStreaming to false - allow reconnection
            break;
            
        case 'error':
            showError(data.message);
            elements.streamStatus.textContent = 'Error occurred';
            elements.streamStatus.className = 'stream-status error';
            
            // Handle "Server busy" error specially - retry after delay
            if (data.message && data.message.includes('Server busy')) {
                console.log('Server busy - will retry connection in 2 seconds...');
                elements.streamStatus.textContent = 'Server busy - Retrying in 2 seconds...';
                elements.streamStatus.className = 'stream-status processing';
                
                // Retry connection after delay
                setTimeout(() => {
                    if (currentConfigName) {
                        console.log('Retrying connection after server busy...');
                        connectWebSocket(currentConfigName);
                    }
                }, 2000);
            } else {
                isStreaming = false;
            }
            break;
            
        case 'control_response':
            console.log('Control response:', data.message);
            break;
            
        case 'loop_restart':
            console.log('Video looping:', data.message);
            const loopCount = data.loop_count || 0;
            elements.streamStatus.textContent = `Video restarting... (Loop #${loopCount})`;
            elements.streamStatus.className = 'stream-status processing';
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

function stopVideoStreaming() {
    console.log('Stopping video streaming...');
    isStreaming = false;
    currentConfigName = null;
    reconnectAttempts = 0;
    
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
    
    elements.streamStatus.textContent = 'Stopped';
    elements.streamStatus.className = 'stream-status';
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

function sendZoneSwitchUpdate() {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
        const manEnabled = elements.manSwitch.checked;
        const womanEnabled = elements.womanSwitch.checked;
        
        websocket.send(JSON.stringify({
            type: 'control',
            command: 'update_gender_switches',
            value: {
                man_enabled: manEnabled,
                woman_enabled: womanEnabled
            }
        }));
        console.log(`Zone switches updated - Man: ${manEnabled}, Woman: ${womanEnabled}`);
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
    // Video selection - automatically start streaming
    elements.videoSelect.addEventListener('change', (e) => {
        const videoName = e.target.value;
        selectedVideo = videoName;
        
        if (videoName && VIDEO_MAPPING[videoName]) {
            console.log(`Switching to video: ${videoName}`);
            
            // Stop any existing stream
            if (isStreaming) {
                stopVideoStreaming();
            }
            
            // Wait for cleanup to complete
            setTimeout(() => {
                // Aggressive cleanup for video switch
                cleanupStreamingState();
                
                // Additional cleanup
                reconnectAttempts = 0;
                currentConfigName = null;
                
                // Hide error panel and show video panel
                hideError();
                elements.processingPanel.style.display = 'none';
                elements.videoPanel.style.display = 'block';
            
                try {
                    const configName = VIDEO_MAPPING[videoName].config;
                    startVideoStreaming(configName);
                    
                } catch (error) {
                    showError('Failed to start streaming: ' + error.message);
                }
            }, 200); // Wait 200ms for cleanup to complete
        }
    });
    
    // Stop streaming button
    elements.stopStreamBtn.addEventListener('click', () => {
        stopVideoStreaming();
    });
    
    // Segmentation toggle
    elements.segmentationToggle.addEventListener('change', (e) => {
        const enabled = e.target.checked;
        sendSegmentationToggle(enabled);
    });
    
    // Gender switches
    elements.manSwitch.addEventListener('change', (e) => {
        sendZoneSwitchUpdate();
    });
    
    elements.womanSwitch.addEventListener('change', (e) => {
        sendZoneSwitchUpdate();
    });
    
    // Retry button
    elements.retryBtn.addEventListener('click', () => {
        hideError();
        if (selectedVideo) {
            // Trigger the change event to restart streaming
            elements.videoSelect.dispatchEvent(new Event('change'));
        }
    });
}


// Removed updateProcessButton since we no longer have a process button

// Initialization
async function initialize() {
    console.log('Initializing Zone Guidance Video Processor...');
    
    setupEventListeners();
    
    // Check server status
    const serverOnline = await checkServerStatus();
    
    if (serverOnline) {
        // Load video options
        await loadVideoOptions();
    } else {
        showError('Server is offline. Please start the FastAPI server and refresh the page.');
    }
    
    // Set up periodic server status checks
    setInterval(checkServerStatus, 30000); // Check every 30 seconds
}

// Start the application
document.addEventListener('DOMContentLoaded', initialize);
