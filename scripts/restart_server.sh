#!/bin/bash
# Restart Chatterbox Multilingual TTS Server in tmux

set -e

SESSION="chatterbox_multilingual_tts"
LOG_FILE="/workspace/server.log"
PROJECT_DIR="/workspace/Chatterbox-Multilingual-TTS"

echo "🔄 Restarting Chatterbox Multilingual TTS Server..."
echo ""

# Kill existing session
echo "🧹 Stopping existing server..."
tmux kill-session -t "$SESSION" 2>/dev/null && echo "   ✅ Existing session stopped" || echo "   ℹ️  No existing session found"

# Wait a moment
sleep 2

# Start new session
echo "🚀 Starting server in tmux session: $SESSION"
tmux new-session -d -s "$SESSION" bash -c "
  echo '🔌 Activating virtual environment...'
  source '$PROJECT_DIR/venv/bin/activate'
  echo '📂 Changing to server directory...'
  cd '$PROJECT_DIR'
  echo '🚀 Starting Python server...'
  python server.py 2>&1 | tee '$LOG_FILE'
"

# Wait for startup
echo "⏳ Waiting for server startup..."
sleep 5

# Check status
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo ""
    echo "✅ Server restarted successfully!"
    echo ""
    echo "📋 Useful commands:"
    echo "   • View logs: tail -f $LOG_FILE"
    echo "   • Attach to session: tmux attach -t $SESSION"
    echo "   • Stop server: tmux kill-session -t $SESSION"
    echo "   • Check status: tmux has-session -t $SESSION && echo 'Running' || echo 'Stopped'"
else
    echo ""
    echo "❌ Server failed to start"
    echo "📝 Check logs: tail -f $LOG_FILE"
    exit 1
fi

