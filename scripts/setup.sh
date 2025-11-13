#!/bin/bash
set -e

# ============================================================
# 🚀 Chatterbox Multilingual TTS Server Setup Script
# ============================================================

echo "📦 Cloning Chatterbox-Multilingual-TTS repository..."
cd /workspace
if [ ! -d "Chatterbox-Multilingual-TTS" ]; then
    git clone https://github.com/aydinmyilmaz/Chatterbox-Multilingual-TTS.git
else
    echo "🔁 Repository already exists. Pulling latest changes..."
    cd Chatterbox-Multilingual-TTS
    git pull
    cd ..
fi

echo "🔧 Installing UV (super-fast Python package manager)..."
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "🦀 Installing Rust (required for UV build tools)..."
curl https://sh.rustup.rs -sSf | sh -s -- -y

# Load Rust environment
source $HOME/.cargo/env

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

echo "⚡ Installing dependencies (using UV pip)..."
uv pip install -r requirements_fastapi.txt --quiet

echo "📁 Creating reference_audio directory..."
mkdir -p reference_audio
touch reference_audio/.gitkeep

# ============================================================
# ✅ Verify Installation
# ============================================================

echo "🧪 Testing PyTorch and Chatterbox..."
python - <<'EOF'
import torch
try:
    from src.chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    print(f'✅ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')
    print(f'✅ Chatterbox Multilingual TTS ready!')
    print(f'✅ Supported languages: {len(SUPPORTED_LANGUAGES)} languages')
except ImportError as e:
    print(f'⚠️ Import error: {e}')
    print('⚠️ Some dependencies may be missing, but continuing...')
EOF

# ============================================================
# 🌐 RunPod Auto URL Detection
# ============================================================

PORT=8004
STREAMLIT_UPDATE_URL="http://194.163.145.174:8505/update_tts_url"
SESSION="chatterbox_multilingual_tts"
LOG_FILE="/workspace/server.log"

echo "🔍 Auto-detecting RunPod server URL..."

# Method 1: Use pod ID from hostname
POD_ID=$(hostname)
TTS_URL="https://${POD_ID}-${PORT}.proxy.runpod.net"

# Method 2: Fallback to env vars
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    POD_ID="${RUNPOD_POD_ID}"
    TTS_URL="https://${POD_ID}-${PORT}.proxy.runpod.net"
fi

API_URL="${TTS_URL}/tts"
GENERATE_URL="${TTS_URL}/generate"
UI_URL="${TTS_URL}"
DOCS_URL="${TTS_URL}/docs"

echo "✅ Auto-detected URLs:"
echo "   🖥️  Web UI: ${UI_URL}"
echo "   📚 API Docs: ${DOCS_URL}"
echo "   🔌 Legacy API Endpoint: ${API_URL}"
echo "   🆕 New API Endpoint: ${GENERATE_URL}"

echo "📡 Notifying remote API endpoint..."
curl -s -X POST -H 'Content-Type: application/json' \
     -d "{\"tts_url\":\"$API_URL\"}" \
     "$STREAMLIT_UPDATE_URL" \
     && echo "✅ Remote API endpoint notified successfully!" \
     || echo "⚠️ Failed to update external service"

# ============================================================
# 🧰 Server Launch via TMUX
# ============================================================

echo "🧹 Killing any existing tmux session..."
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "🎬 Starting Chatterbox Multilingual TTS server in tmux session: $SESSION"
tmux new-session -d -s "$SESSION" bash -c "
  echo '🔌 Activating virtual environment...'
  source '/workspace/Chatterbox-Multilingual-TTS/venv/bin/activate'
  echo '📂 Changing to server directory...'
  cd '/workspace/Chatterbox-Multilingual-TTS'
  echo '🚀 Starting Python server...'
  python server.py 2>&1 | tee '$LOG_FILE'
"

echo "⏳ Waiting for server startup..."
sleep 30

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "✅ Server session is running"
    echo ""
    echo "🎉 Setup complete! Server is running in tmux session '$SESSION'"
    echo ""
    echo "📋 Useful commands:"
    echo "   • View logs: tail -f $LOG_FILE"
    echo "   • Attach to session: tmux attach -t $SESSION"
    echo "   • Stop server: tmux kill-session -t $SESSION"
    echo ""
    echo "🌐 Server URLs:"
    echo "   • API Docs: ${DOCS_URL}"
    echo "   • Legacy /tts: ${API_URL}"
    echo "   • New /generate: ${GENERATE_URL}"
else
    echo "❌ Server session failed to start"
    echo "📝 Check logs: tail -f $LOG_FILE"
    exit 1
fi

