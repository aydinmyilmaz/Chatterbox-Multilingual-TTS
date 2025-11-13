#!/bin/bash
# RunPod Kurulum Script'i
# Bu script'i RunPod terminalinde çalıştırın

set -e

echo "🚀 Chatterbox Multilingual TTS RunPod Kurulumu"
echo ""

# Repo URL (GitHub username'inizi değiştirin)
GITHUB_USERNAME="YOUR_USERNAME"  # ← BURAYI DEĞİŞTİRİN
REPO_NAME="Chatterbox-Multilingual-TTS"
REPO_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

# Workspace directory
WORKSPACE_DIR="/workspace"
PROJECT_DIR="${WORKSPACE_DIR}/${REPO_NAME}"

echo "📋 Kurulum Bilgileri:"
echo "   Repo: ${REPO_URL}"
echo "   Hedef: ${PROJECT_DIR}"
echo ""

# Clone repository
if [ -d "${PROJECT_DIR}" ]; then
    echo "⚠️  Klasör zaten var. Güncelleniyor..."
    cd "${PROJECT_DIR}"
    git pull origin main
else
    echo "📥 Repository klonlanıyor..."
    cd "${WORKSPACE_DIR}"
    git clone "${REPO_URL}"
    cd "${PROJECT_DIR}"
fi

# Run setup script
echo ""
echo "🔧 Setup script çalıştırılıyor..."
bash setup_runpod.sh

echo ""
echo "✅ Kurulum tamamlandı!"
echo ""
echo "📋 Server'ı başlatmak için:"
echo "   cd ${PROJECT_DIR}"
echo "   source venv/bin/activate"
echo "   python server.py"
echo ""
echo "🌐 Server: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"
