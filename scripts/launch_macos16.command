#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Qwen3-ASR Pro Launcher for macOS 16                      ║
# ║         Uses system Python to avoid tkinter issues               ║
# ╚══════════════════════════════════════════════════════════════════╝

cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Qwen3-ASR Pro for macOS 16                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Use system Python (has compatible tkinter)
SYSTEM_PYTHON="/usr/bin/python3"

if [ ! -f "$SYSTEM_PYTHON" ]; then
    echo "❌ System Python not found at $SYSTEM_PYTHON"
    exit 1
fi

echo "Using system Python: $SYSTEM_PYTHON"
echo ""

# Create system venv if not exists
if [ ! -d "backend/venv_system" ]; then
    echo "📦 Creating system Python virtual environment..."
    $SYSTEM_PYTHON -m venv backend/venv_system
fi

# Activate system venv
source backend/venv_system/bin/activate

echo "📦 Installing dependencies..."
pip install -q --upgrade pip

# Install packages (without problematic ones)
pip install -q transformers==4.35.0 torch==2.1.0 numpy==1.26.0 sounddevice

# Try to install mlx for transcription
pip install -q mlx-qwen3-asr 2>/dev/null || echo "⚠️  Note: mlx-qwen3-asr install had warnings"

echo ""
echo "🚀 Starting application..."
echo ""

# Run with system Python
python src/main.py
