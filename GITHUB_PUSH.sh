#!/bin/bash
# GitHub Push Script
# Kullanım: bash GITHUB_PUSH.sh YOUR_USERNAME

if [ -z "$1" ]; then
    echo "❌ Kullanım: bash GITHUB_PUSH.sh YOUR_USERNAME"
    echo "   Örnek: bash GITHUB_PUSH.sh aydinmyilmaz"
    exit 1
fi

USERNAME=$1
REPO_NAME="Chatterbox-Multilingual-TTS"

echo "🚀 GitHub'a push ediliyor..."
echo "   Repo: $USERNAME/$REPO_NAME"
echo ""

# Check if remote exists
if git remote get-url origin > /dev/null 2>&1; then
    echo "⚠️  Remote 'origin' zaten var. Güncelleniyor..."
    git remote set-url origin "https://github.com/$USERNAME/$REPO_NAME.git"
else
    echo "➕ Remote 'origin' ekleniyor..."
    git remote add origin "https://github.com/$USERNAME/$REPO_NAME.git"
fi

# Set branch to main
git branch -M main

# Push
echo ""
echo "📤 Push ediliyor..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Başarıyla push edildi!"
    echo "🌐 Repo: https://github.com/$USERNAME/$REPO_NAME"
else
    echo ""
    echo "❌ Push başarısız!"
    echo "   GitHub'da repo oluşturduğunuzdan emin olun:"
    echo "   https://github.com/new"
    echo ""
    echo "   Repo adı: $REPO_NAME"
    echo "   Public veya Private seçin"
fi

