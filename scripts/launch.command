#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Qwen3-ASR Pro v3.1.1 - Optimized Speech-to-Text            ║
# ║         Based on official Qwen3-ASR & mlx-qwen3-asr              ║
# ╚══════════════════════════════════════════════════════════════════╝

cd "$(dirname "$0")"

if [ ! -d "backend/venv" ]; then
    echo "❌ Setup required"
    echo ""
    echo "Please run SETUP.command first to install dependencies."
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
fi

source backend/venv/bin/activate

# Detect platform
ARCH=$(uname -m)
OS=$(uname -s)

if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    echo "🚀 Apple Silicon detected"
else
    echo "💻 Intel Mac detected"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Qwen3-ASR Pro v3.1.1                                 ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  Features:                                                 ║"
echo "║  • 🎤 Real-time recording with configurable auto-stop      ║"
echo "║  • 📁 File upload (WAV, MP3, M4A, FLAC, OGG)               ║"
echo "║  • ⚡ MLX optimized for Apple Silicon                      ║"
echo "║  • 🎚️ Adjustable silence detection (0.5s - 5s)            ║"
echo "║  • 📊 Real-time RTF performance metrics                    ║"
echo "║  • 🌍 50+ languages supported                              ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Starting application..."
echo ""

python src/main.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Application exited with error"
    read -n 1 -s -r -p "Press any key to close..."
fi
