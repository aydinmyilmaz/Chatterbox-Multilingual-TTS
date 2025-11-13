#!/bin/bash
echo "🛑 Stopping FastAPI server..."
SESSION="chatterbox_multilingual_tts"
tmux kill-session -t "$SESSION" 2>/dev/null && echo "✅ FastAPI server stopped" || echo "⚠️  No FastAPI session found"
sleep 2

