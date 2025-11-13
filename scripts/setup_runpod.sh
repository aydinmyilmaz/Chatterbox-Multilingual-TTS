#!/bin/bash
# RunPod Setup Script for Chatterbox Multilingual TTS FastAPI Server
# This script sets up the environment and installs all dependencies

set -e  # Exit on error

echo "🚀 Setting up Chatterbox Multilingual TTS FastAPI Server on RunPod..."
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python: $python_version"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install PyTorch with CUDA support
echo ""
echo "🔥 Installing PyTorch with CUDA support..."
echo "   This may take a few minutes..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

# Install other dependencies
echo ""
echo "📦 Installing dependencies from requirements_fastapi.txt..."
echo "   This may take several minutes..."
pip install -r requirements_fastapi.txt --quiet

# Create reference audio directory
echo ""
echo "📁 Creating reference_audio directory..."
mkdir -p reference_audio
touch reference_audio/.gitkeep
echo -e "${GREEN}✅ Directory created${NC}"

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python3 -c "
import torch
import fastapi
from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS
print('✅ All imports successful')
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
print('✅ FastAPI: ' + fastapi.__version__)
print('✅ ChatterboxMultilingualTTS imported successfully')
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Setup completed successfully!${NC}"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Activate virtual environment: source venv/bin/activate"
    echo "   2. Start server: python server.py"
    echo "   3. Or with uvicorn: uvicorn server:app --host 0.0.0.0 --port 8000"
    echo ""
    echo "🌐 Server will be available at: http://localhost:8000"
    echo "📖 API docs: http://localhost:8000/docs"
else
    echo ""
    echo -e "${RED}❌ Setup verification failed${NC}"
    echo "   Please check the error messages above"
    exit 1
fi
