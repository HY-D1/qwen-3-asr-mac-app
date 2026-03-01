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

# Activate environment
if [ -d "$HOME/qwen_system" ]; then
    source ~/qwen_system/bin/activate
elif [ -d "$HOME/qwen_venv" ]; then
    source ~/qwen_venv/bin/activate
elif [ -d "backend/venv_system" ]; then
    source backend/venv_system/bin/activate
else
    echo "❌ No virtual environment found"
    exit 1
fi

echo "📦 Checking dependencies..."

# Install gradio if not present
pip install -q gradio

echo ""
echo "🚀 Starting Web UI..."
echo ""
echo "📱 The app will open in your browser at:"
echo "   http://localhost:7860"
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
