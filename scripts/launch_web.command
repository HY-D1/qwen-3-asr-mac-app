#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Qwen3-ASR Pro - Web UI Launcher                          ║
# ║         Works in browser - No tkinter needed                     ║
# ╚══════════════════════════════════════════════════════════════════╝

cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         🌐 Qwen3-ASR Pro - Web UI                          ║"
echo "║         Launches in your browser                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Suppress Ollama GIN logs (HTTP request logging)
export GIN_MODE=release

# Check if Ollama is running, start it quietly if not
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "🤖 Starting Ollama server (logs → ~/.ollama/logs/server.log)..."
    mkdir -p ~/.ollama/logs
    nohup ollama serve > ~/.ollama/logs/server.log 2>&1 &
    sleep 2
    # Verify it started
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "   ✅ Ollama is ready"
    else
        echo "   ⚠️  Ollama may still be starting..."
    fi
    echo ""
fi

# Kill any existing Gradio servers
echo "🧹 Cleaning up old servers..."
pkill -f "python.*web_ui.py" 2>/dev/null || true
sleep 1

# Try environments in order
if [ -d "$HOME/qwen_system" ]; then
    source ~/qwen_system/bin/activate
elif [ -d "$HOME/qwen_venv" ]; then
    source ~/qwen_venv/bin/activate
elif [ -d "backend/venv_system" ]; then
    source backend/venv_system/bin/activate
else
    echo "⚠️  No virtual environment found"
    echo "   Creating one now..."
    /usr/bin/python3 -m venv ~/qwen_cli
    source ~/qwen_cli/bin/activate
    pip install -q transformers==4.35.0 torch==2.1.0 numpy==1.26.0
fi

echo "Python: $(which python)"
echo ""

# Check if audio file provided
if [ -z "$1" ]; then
    echo "📦 Checking dependencies..."
    pip install -q gradio
    
    echo ""
    echo "🚀 Starting Web UI..."
    echo ""
    
    # Don't set port - let web_ui.py auto-detect
    # This avoids port conflicts
    unset GRADIO_SERVER_PORT
    
    echo "📱 The app will open in your browser automatically"
    echo ""
    echo "   Features:"
    echo "   • 🎤 Record audio from microphone"
    echo "   • 📁 Upload audio files"
    echo "   • 🤖 AI text reforming"
    echo "   • 💾 Copy results to clipboard"
    echo ""
    echo "   Press Ctrl+C to stop the server"
    echo ""
    
    python web_ui.py
else
    # Pass all arguments to CLI app
    python cli_app.py "$@"
fi
