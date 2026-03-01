#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Qwen3-ASR Pro - macOS Setup                              ║
# ╚══════════════════════════════════════════════════════════════════╝

set -e

cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         Qwen3-ASR Pro Setup                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if command -v python3.12 &> /dev/null; then
    PYTHON=python3.12
elif command -v python3 &> /dev/null; then
    PYTHON=python3
else
    echo "❌ Python 3.12+ not found"
    echo "Please install Python from https://python.org"
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | cut -d' ' -f2)
echo "✅ Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "📦 Setting up virtual environment..."
mkdir -p backend
if [ ! -d "backend/venv" ]; then
    $PYTHON -m venv backend/venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment exists"
fi

source backend/venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip -q

# Detect platform
ARCH=$(uname -m)
OS=$(uname -s)

echo ""
echo "🔍 Platform: $OS $ARCH"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."

# Core dependencies
pip install sounddevice numpy -q

# Check if Apple Silicon
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    echo "🚀 Installing MLX packages for Apple Silicon..."
    
    # Try mlx-qwen3-asr first
    pip install mlx-qwen3-asr -q || echo "⚠️  Note: mlx-qwen3-asr install had warnings"
    
    # Also try mlx-audio
    pip install mlx-audio -q 2>/dev/null || echo "⚠️  Note: mlx-audio optional"
    
    echo "✅ MLX packages installed"
else
    echo "💻 Installing PyTorch backend for Intel Mac..."
    pip install qwen-asr torch -q
    echo "✅ PyTorch backend installed"
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."

if python -c "import sounddevice; import numpy" 2>/dev/null; then
    echo "✅ Audio libraries OK"
else
    echo "❌ Audio libraries failed"
    exit 1
fi

# Check which backend is available
if python -c "import mlx_audio" 2>/dev/null; then
    echo "✅ MLX-Audio backend available"
elif python -m mlx_qwen3_asr --version &>/dev/null; then
    echo "✅ MLX-CLI backend available"
elif python -c "import qwen_asr" 2>/dev/null; then
    echo "✅ PyTorch backend available"
else
    echo "⚠️  Warning: No transcription backend found"
    echo "   The app may not work correctly."
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ Setup Complete!                            ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  Launch: ./scripts/launch.command                          ║"
echo "║                                                            ║"
echo "║  Features:                                                 ║"
echo "║  • 🎤 Real-time recording with auto-stop                   ║"
echo "║  • 📁 File upload & drag-drop                              ║"
echo "║  • ⚡ Apple Silicon optimized (MLX)                        ║"
echo "║  • 🎚️ Adjustable silence detection                        ║"
echo "║  • 📊 Real-time performance metrics                        ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
read -n 1 -s -r -p "Press any key to close..."
