# RunPod Deployment Guide

## 🚀 Quick Setup

### 1. Clone Repository

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/Chatterbox-Multilingual-TTS.git
cd Chatterbox-Multilingual-TTS
```

### 2. Run Setup Script

```bash
bash setup_runpod.sh
```

This script will:
- Create virtual environment
- Install PyTorch with CUDA support
- Install all dependencies
- Verify installation

### 3. Start Server

```bash
source venv/bin/activate
python server.py
```

Or with uvicorn:
```bash
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

## 📋 Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements_fastapi.txt

# Create reference audio directory
mkdir -p reference_audio

# Start server
python server.py
```

## 🔧 RunPod Specific Configuration

### Port Configuration
The server runs on port 8000 by default. To change:
```bash
export PORT=8000
python server.py
```

### Running in Background (tmux)

```bash
# Start tmux session
tmux new-session -d -s chatterbox_tts

# Run server in tmux
tmux send-keys -t chatterbox_tts "source venv/bin/activate && cd /workspace/Chatterbox-Multilingual-TTS && python server.py" Enter

# Attach to session
tmux attach -t chatterbox_tts

# Detach: Ctrl+B, then D
```

## 🌐 API Endpoints

Once running, access:
- **API**: `http://localhost:8000`
- **Docs**: `http://localhost:8000/docs`
- **Health**: `http://localhost:8000/health`

## 📝 Example Usage

### Generate Speech
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "language_id": "en"
  }' \
  --output output.wav
```

### Upload Reference Audio
```bash
curl -X POST "http://localhost:8000/upload_reference" \
  -F "file=@my_voice.wav" \
  -F "name=my_voice"
```

## ⚠️ Troubleshooting

### Model Download
On first run, the model will download from Hugging Face. This may take time.

### CUDA Issues
If CUDA is not available, the model will use CPU (slower).

### Port Already in Use
Change port:
```bash
export PORT=8001
python server.py
```

## 📚 More Information

See `README.md` for detailed API documentation.

