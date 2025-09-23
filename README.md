# Gender Detection System

A comprehensive AI-powered gender detection system built with FastAPI, Triton Inference Server, and a modern web frontend.

## 🏗️ Architecture

The system consists of three main components:

- **Triton Server** (Port 8002): AI model inference server
- **FastAPI Backend** (Port 8001): REST API and business logic
- **Frontend** (Port 8000): Web interface for video processing

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 16+
- UV package manager
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd gender_detection
```

2. Install Python dependencies:
```bash
uv pip install -r requirements.txt
```

3. Install Node.js dependencies:
```bash
cd client
npm install
cd ..
```

### Running the System

#### Option 1: Start All Services (Recommended)
```bash
./scripts/start_services.sh
```

#### Option 2: Start Individual Components

**Start Server (Triton + FastAPI):**
```bash
cd server
./start.sh
```

**Start Frontend:**
```bash
cd client
./start.sh
```

### Stopping the System

```bash
./scripts/stop_services.sh
```

### Monitoring

```bash
./scripts/monitor_services.sh
```

Available monitoring commands:
- `./scripts/monitor_services.sh status` - Check service status
- `./scripts/monitor_services.sh health` - Check health endpoints
- `./scripts/monitor_services.sh logs` - Show recent logs
- `./scripts/monitor_services.sh follow` - Follow logs in real-time
- `./scripts/monitor_services.sh resources` - Show system resources
- `./scripts/monitor_services.sh ports` - Show port usage

## 📁 Project Structure

```
gender_detection/
├── scripts/                    # Management scripts
│   ├── start_services.sh      # Start all services
│   ├── stop_services.sh       # Stop all services
│   └── monitor_services.sh    # Monitor services
├── server/                     # Backend services
│   ├── start.sh               # Start server components
│   ├── main.py                # FastAPI application
│   ├── triton_server.py       # Triton inference server
│   └── python_backend/        # Model implementations
├── client/                     # Frontend
│   ├── start.sh               # Start frontend
│   ├── server.js              # Express server
│   └── index.html             # Web interface
├── src/                        # Source code
│   ├── models/                # AI models
│   └── utils/                 # Utility functions
└── data/                       # Data directory
    ├── inputs/                # Input videos
    └── outputs/               # Processed videos
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Server Configuration
TRITON_PORT=8002
FASTAPI_PORT=8001
FRONTEND_PORT=8000

# Model Configuration
MODEL_REPOSITORY_PATH=/path/to/models
```

### Model Setup

1. Place your AI models in the appropriate directories
2. Update model configurations in `server/python_backend/`
3. Ensure models are compatible with Triton Inference Server

## 📊 API Endpoints

### FastAPI Server (Port 8001)

- `GET /` - Health check
- `POST /process` - Process video for gender detection
- `GET /status` - System status

### Triton Server (Port 8002)

- `GET /v2/health/ready` - Server readiness
- `POST /v2/models/{model_name}/infer` - Model inference

## 🐛 Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8000, 8001, and 8002 are available
2. **Model loading errors**: Check model paths and configurations
3. **GPU memory issues**: Reduce batch size or use CPU inference

### Logs

Check service logs:
```bash
# Triton server
tail -f server/triton.log

# FastAPI server
tail -f server/fastapi.log

# Frontend server
tail -f client/frontend.log
```

### Health Checks

```bash
# Check all services
curl http://localhost:8000  # Frontend
curl http://localhost:8001  # FastAPI
curl http://localhost:8002/v2/health/ready  # Triton
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs
3. Create an issue in the repository

## 🔄 Updates

To update the system:
1. Pull the latest changes: `git pull`
2. Update dependencies: `uv pip install -r requirements.txt`
3. Restart services: `./scripts/stop_services.sh && ./scripts/start_services.sh`