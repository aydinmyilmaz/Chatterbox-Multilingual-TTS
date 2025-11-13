#!/bin/bash
set -e

echo "🚀 Setting up Chatterbox Multilingual TTS - Gradio App"
echo "======================================================"

# Check if we're in the right directory
if [ ! -f "examples/app.py" ]; then
    echo "❌ Error: examples/app.py not found. Please run this script from the project root."
    exit 1
fi

# Install git-lfs if not available
if ! command -v git-lfs &> /dev/null; then
    echo "📦 Installing git-lfs..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y git-lfs
    elif command -v yum &> /dev/null; then
        sudo yum install -y git-lfs
    else
        echo "⚠️  Please install git-lfs manually: https://git-lfs.com"
    fi
    git lfs install
fi

# Install UV if not available
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Install Rust if not available (required for UV)
if ! command -v rustc &> /dev/null; then
    echo "📦 Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    export PATH="$HOME/.cargo/bin:$PATH"
fi

cd /workspace/Chatterbox-Multilingual-TTS

echo "🐍 Creating virtual environment with UV..."
uv venv --python 3.10 venv
source venv/bin/activate

echo "⚡ Installing setuptools first (required for pkg_resources)..."
uv pip install "setuptools>=65.0.0" --quiet

echo "⚡ Installing hf_transfer (for fast Hugging Face downloads)..."
uv pip install "hf_transfer>=0.1.0" --quiet

echo "⚡ Installing PyTorch with CUDA support (using UV pip)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet

echo "⚡ Installing Gradio and dependencies..."
uv pip install gradio --quiet

echo "⚡ Installing other dependencies (using UV pip)..."
uv pip install -r requirements_fastapi.txt --quiet

echo "📁 Creating reference_audio directory..."
mkdir -p reference_audio
touch reference_audio/.gitkeep

echo "✅ Installation complete!"
echo ""
echo "🔍 Verifying installation..."
python - <<'EOF'
import torch
try:
    from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    print(f'✅ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
    print(f'✅ Chatterbox Multilingual TTS ready!')
    print(f'✅ Supported languages: {len(SUPPORTED_LANGUAGES)} languages')
    import gradio as gr
    print(f'✅ Gradio: {gr.__version__}')
except ImportError as e:
    print(f'⚠️ Import error: {e}')
    print('⚠️ Some dependencies may be missing, but continuing...')
EOF

echo ""
echo "🌐 Detecting RunPod URLs..."
RUNPOD_PUBLIC_IP=$(curl -s http://checkip.amazonaws.com 2>/dev/null || echo "unknown")
RUNPOD_PUBLIC_PORT=${RUNPOD_PUBLIC_PORT:-8004}
echo "   Public IP: $RUNPOD_PUBLIC_IP"
echo "   Public Port: $RUNPOD_PUBLIC_PORT"

# Notify remote API if available
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "📡 Notifying RunPod API..."
    curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"mutation { podResume(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }\"}" \
        > /dev/null 2>&1 || echo "   (API notification skipped)"
fi

echo ""
echo "🚀 Starting Gradio app in tmux session..."
SESSION="chatterbox_gradio_app"
LOG_FILE="/workspace/gradio.log"

# Kill existing session if any
tmux kill-session -t "$SESSION" 2>/dev/null || true
sleep 2

# Start Gradio app in tmux
tmux new-session -d -s "$SESSION" bash -c "
    source '$PWD/venv/bin/activate'
    cd '$PWD'
    export HF_HUB_ENABLE_HF_TRANSFER=1
    export PORT=8004
    python examples/app.py 2>&1 | tee '$LOG_FILE'
"

sleep 5

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "✅ Gradio app started successfully!"
    echo ""
    echo "📋 Session info:"
    echo "   Session name: $SESSION"
    echo "   Log file: $LOG_FILE"
    echo "   Port: 8004"
    echo ""
    echo "🔍 To check logs:"
    echo "   tmux attach -t $SESSION"
    echo "   or: tail -f $LOG_FILE"
    echo ""
    echo "🛑 To stop:"
    echo "   tmux kill-session -t $SESSION"
    echo ""
    echo "🌐 Access the app at:"
    echo "   http://$RUNPOD_PUBLIC_IP:$RUNPOD_PUBLIC_PORT"
else
    echo "❌ Failed to start Gradio app. Check logs:"
    echo "   tail -f $LOG_FILE"
    exit 1
fi

