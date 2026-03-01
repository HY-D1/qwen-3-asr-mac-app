#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Qwen3-ASR Pro Launcher                                   ║
# ╚══════════════════════════════════════════════════════════════════╝

cd "$(dirname "$0")/.."

# Detect platform
ARCH=$(uname -m)
OS=$(uname -s)

if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    echo "🚀 Apple Silicon detected"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Qwen3-ASR Pro v3.3.0                               ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  Features:                                                 ║"
echo "║  • 🎤 Real-time recording with configurable auto-stop      ║"
echo "║  • 📁 File upload (WAV, MP3, M4A, FLAC, OGG)               ║"
echo "║  • ⚡ MLX optimized for Apple Silicon                      ║"
echo "║  • 🤖 AI text refinement (optional)                        ║"
echo "║  • 🎚️ Adjustable silence detection (0.5s - 60s)           ║"
echo "║  • 📊 Real-time performance metrics                        ║"
echo "║  • 🌍 50+ languages supported                              ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check for working environment first
if [ -d "$HOME/qwen_venv" ]; then
    echo "🔧 Using working environment (qwen_venv)..."
    source ~/qwen_venv/bin/activate
elif [ -d "backend/venv" ]; then
    echo "📦 Using project virtual environment..."
    source backend/venv/bin/activate
elif command -v conda &> /dev/null; then
    # Try conda environments
    if conda env list | grep -q "^qwen\s"; then
        echo "🐍 Using conda environment (qwen)..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate qwen
    else
        echo "⚠️  No virtual environment found!"
        echo "   Please run: ./scripts/setup_local_llm_m1.command"
        read -n 1 -s -r -p "Press any key to exit..."
        exit 1
    fi
else
    echo "⚠️  No virtual environment found!"
    echo "   Please run: ./scripts/setup_local_llm_m1.command"
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
fi

echo "Starting application..."
echo ""

# Run
python src/main.py

# Check exit code
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Application exited with error"
    echo ""
    echo "If you see 'macOS 26 required', try:"
    echo "  source ~/qwen_venv/bin/activate"
    echo "  pip uninstall torch -y"
    echo "  pip install torch==2.1.0"
    echo "  python src/main.py"
    echo ""
    read -n 1 -s -r -p "Press any key to close..."
fi
