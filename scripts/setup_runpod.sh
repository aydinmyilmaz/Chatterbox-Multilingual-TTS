#!/bin/bash
# Setup script for RunPod deployment

set -e

echo "🚀 Setting up Chatterbox Multilingual TTS FastAPI Server..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "📦 Installing dependencies..."
pip install -r requirements_fastapi.txt

# Create reference audio directory
mkdir -p reference_audio

echo "✅ Setup complete!"
echo ""
echo "📋 To start the server:"
echo "   source venv/bin/activate"
echo "   python server.py"
echo ""
echo "📋 Or with uvicorn:"
echo "   uvicorn server:app --host 0.0.0.0 --port 8000"

