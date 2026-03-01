#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Qwen3-ASR Pro - CLI Launcher                             ║
# ║         No GUI - Works on all macOS versions                     ║
# ╚══════════════════════════════════════════════════════════════════╝

cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Qwen3-ASR Pro - CLI Mode                           ║"
echo "║         (No GUI - Works with any Python)                   ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Try environments in order
if [ -d "$HOME/qwen_system" ]; then
    source ~/qwen_system/bin/activate
elif [ -d "$HOME/qwen_venv" ]; then
    source ~/qwen_venv/bin/activate
elif [ -d "backend/venv" ]; then
    source backend/venv/bin/activate
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
    echo "Usage:"
    echo "  ./scripts/launch_cli.command                    # Interactive mode"
    echo "  ./scripts/launch_cli.command audio.wav          # Transcribe file"
    echo "  ./scripts/launch_cli.command audio.wav -r clean # Transcribe + reform"
    echo ""
    echo "Starting interactive mode..."
    echo ""
    python cli_app.py --interactive
else
    # Pass all arguments to CLI app
    python cli_app.py "$@"
fi
