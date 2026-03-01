#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Qwen3-ASR Pro - CLI Launcher                             ║
# ║         Command-line interface for transcription                 ║
# ╚══════════════════════════════════════════════════════════════════╝

cd "$(dirname "$0")/.."

# Suppress Ollama GIN logs (HTTP request logging)
export GIN_MODE=release

# Check if Ollama is running, start it quietly if not
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "🤖 Starting Ollama server (logs → ~/.ollama/logs/server.log)..."
    mkdir -p ~/.ollama/logs
    nohup ollama serve > ~/.ollama/logs/server.log 2>&1 &
    sleep 2
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         🖥️  Qwen3-ASR Pro - CLI Mode                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Try environments in order
if [ -d "$HOME/qwen_system" ]; then
    source ~/qwen_system/bin/activate
elif [ -d "$HOME/qwen_venv" ]; then
    source ~/qwen_venv/bin/activate
elif [ -d "backend/venv_system" ]; then
    source backend/venv_system/bin/activate
else
    echo "⚠️  No virtual environment found"
    echo "   Run ./scripts/setup.command first"
    exit 1
fi

echo "🐍 Python: $(python --version)"
echo ""

# Launch CLI
python cli_app.py "$@"
