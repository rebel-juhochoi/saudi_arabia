# Triton Inference Server Migration Plan

## Architecture Overview

### Current State
- **In-Process Inference**: Models loaded directly in FastAPI server
- **Rebel Runtime**: Using `.rbln` compiled models
- **5 Video Configs**: Different thresholds, same models
- **Memory Issues**: Models loaded per video instance

### Target State
- **Triton Inference Server**: Centralized model serving
- **HTTP/gRPC Client**: FastAPI communicates via network calls
- **Single Model Instance**: Shared across all video configs
- **Better Resource Management**: GPU memory optimization

## Implementation Phases

### Phase 1: Triton Server Setup (2-3 days)

#### 1.1 Model Repository Structure
```
triton_models/
├── yolo11n_seg/
│   ├── config.pbtxt
│   └── 1/
│       ├── model.rbln (or convert to ONNX)
│       └── metadata.json
├── deepface_gender/
│   ├── config.pbtxt
│   └── 1/
│       ├── model.rbln (or convert to ONNX)
│       └── metadata.json
└── ensemble_detector/  # Optional: Combined pipeline
    ├── config.pbtxt
    └── 1/
```

#### 1.2 Model Configuration Files
**yolo11n_seg/config.pbtxt:**
```protobuf
name: "yolo11n_seg"
platform: "python"  # or "onnxruntime_onnx" if converted
max_batch_size: 8
input [
  {
    name: "images"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }
]
output [
  {
    name: "output0"
    data_type: TYPE_FP32
    dims: [ 84, 8400 ]
  }
]
dynamic_batching {
  max_queue_delay_microseconds: 100
}
```

#### 1.3 Triton Server Deployment
```bash
# Docker deployment
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /path/to/triton_models:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### Phase 2: Client Implementation (3-4 days)

#### 2.1 Triton Client Interface
```python
# src/inference/triton_client.py
import tritonclient.http as httpclient
import numpy as np
from typing import Tuple, List, Optional

class TritonInferenceClient:
    def __init__(self, triton_url: str = "localhost:8001"):
        self.client = httpclient.InferenceServerClient(triton_url)
        self.person_model = "yolo11n_seg"
        self.gender_model = "deepface_gender"
    
    async def detect_persons(self, frame: np.ndarray, config: dict) -> List:
        """Person detection via Triton"""
        # Preprocess frame
        input_data = self._preprocess_yolo(frame)
        
        # Create inference request
        inputs = [httpclient.InferInput("images", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        
        outputs = [httpclient.InferRequestedOutput("output0")]
        
        # Run inference
        response = self.client.infer(self.person_model, inputs, outputs=outputs)
        result = response.as_numpy("output0")
        
        # Postprocess with config-specific thresholds
        return self._postprocess_yolo(result, frame.shape, config)
    
    async def classify_gender(self, person_crop: np.ndarray, config: dict) -> Tuple[str, float]:
        """Gender classification via Triton"""
        # Preprocess person crop
        input_data = self._preprocess_gender(person_crop)
        
        inputs = [httpclient.InferInput("input", input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)
        
        outputs = [httpclient.InferRequestedOutput("output")]
        
        # Run inference
        response = self.client.infer(self.gender_model, inputs, outputs=outputs)
        result = response.as_numpy("output")
        
        return self._postprocess_gender(result, config)
```

#### 2.2 Model Adapter Classes
```python
# src/inference/adapters.py
class TritonPersonDetector:
    """Adapter to maintain PersonDetector interface"""
    def __init__(self, triton_client: TritonInferenceClient, config: dict):
        self.client = triton_client
        self.config = config
    
    def infer(self, frame: np.ndarray, orig_shape: tuple) -> List:
        """Maintain original interface"""
        return asyncio.run(self.client.detect_persons(frame, self.config))

class TritonGenderClassifier:
    """Adapter to maintain GenderClassifier interface"""
    def __init__(self, triton_client: TritonInferenceClient, config: dict):
        self.client = triton_client
        self.config = config
    
    def classify_gender(self, person_crop: np.ndarray) -> Tuple[str, float]:
        """Maintain original interface"""
        return asyncio.run(self.client.classify_gender(person_crop, self.config))
```

### Phase 3: Integration (2-3 days)

#### 3.1 Server Modifications
```python
# In server.py - Modified VideoProcessor initialization
class VideoProcessor:
    def __init__(self, config_name: str, **kwargs):
        self.config_name = config_name
        self.config = PROCESSOR_CONFIGS[config_name]
        
        # Initialize Triton client (shared across all processors)
        if not hasattr(VideoProcessor, '_triton_client'):
            VideoProcessor._triton_client = TritonInferenceClient()
        
        # Create adapted models
        self.person_detector = TritonPersonDetector(
            VideoProcessor._triton_client, 
            self.config
        )
        self.gender_classifier = TritonGenderClassifier(
            VideoProcessor._triton_client, 
            self.config
        )
        
        # Initialize tracker with adapted models
        self.tracker = Tracker(
            person_detector=self.person_detector,
            gender_classifier=self.gender_classifier,
            **kwargs
        )
```

#### 3.2 Configuration Enhancement
```python
PROCESSOR_CONFIGS = {
    "01_man": {
        "person_conf": 0.25,
        "iou_threshold": 0.5,
        "gender_conf": 0.5,
        "enable_color_heuristic": False,
        "show_segmentation": False,
        "triton_config": {
            "person_model_name": "yolo11n_seg",
            "gender_model_name": "deepface_gender",
            "batch_size": 1,
            "timeout_ms": 1000
        }
    },
    # ... other configs
}
```

## Benefits of Migration

### 1. **Resource Efficiency**
- **Single Model Instance**: Instead of 5 copies, one shared instance
- **GPU Memory**: ~80% reduction in memory usage
- **CPU Efficiency**: Centralized inference processing

### 2. **Scalability**
- **Horizontal Scaling**: Multiple Triton replicas
- **Load Balancing**: Distribute inference load
- **Auto-scaling**: Scale based on demand

### 3. **Performance**
- **Dynamic Batching**: Batch multiple requests automatically
- **Model Optimization**: TensorRT optimization
- **Concurrent Requests**: Handle multiple videos simultaneously

### 4. **Maintenance**
- **Model Versioning**: Easy A/B testing and rollbacks
- **Health Monitoring**: Built-in metrics and health checks
- **Configuration Management**: Centralized model configs

## Migration Timeline

### Week 1: Setup & Preparation
- Day 1-2: Model conversion and Triton setup
- Day 3-4: Basic client implementation
- Day 5: Testing and validation

### Week 2: Integration & Testing
- Day 1-3: FastAPI integration
- Day 4-5: End-to-end testing and optimization

### Week 3: Deployment & Monitoring
- Day 1-2: Production deployment
- Day 3-5: Performance monitoring and tuning

## Risk Mitigation

### 1. **Backward Compatibility**
- Keep original models as fallback
- Feature flags for gradual rollout
- A/B testing framework

### 2. **Network Reliability**
- Connection pooling and retry logic
- Circuit breaker pattern
- Local caching for critical paths

### 3. **Performance Validation**
- Benchmark before/after migration
- Load testing with realistic traffic
- Memory usage monitoring

## Success Metrics

### 1. **Performance**
- Inference latency: <50ms increase
- Memory usage: >70% reduction
- Concurrent streams: 5x improvement

### 2. **Reliability**
- 99.9% uptime
- <1% error rate
- Graceful degradation

### 3. **Maintainability**
- Model deployment time: <5 minutes
- Configuration changes: Hot-swappable
- Monitoring coverage: 100%
