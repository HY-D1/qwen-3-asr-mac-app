#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Qwen3-ASR Pro - LLM Text Reformer Setup                  ║
# ║         Optional: For AI text refinement features                ║
# ╚══════════════════════════════════════════════════════════════════╝

set -e

cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         🤖 LLM Text Reformer Setup                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if venv exists
if [ ! -d "backend/venv" ]; then
    echo "❌ Virtual environment not found."
    echo "   Please run ./scripts/setup.command first"
    read -n 1 -s -r -p "Press any key to exit..."
    exit 1
fi

source backend/venv/bin/activate

# Detect platform
ARCH=$(uname -m)
OS=$(uname -s)

echo "🔍 Platform: $OS $ARCH"
echo ""

# Check Python environment
if command -v conda &> /dev/null && [[ "$PATH" == *"conda"* ]]; then
    echo "⚠️  Conda detected. This can cause build issues with llama-cpp."
    echo "   Recommendation: Use mlx-lm (Apple Silicon) or system Python."
    echo ""
fi

# Installation options
echo "📦 Installation Options:"
echo ""
echo "  [1] 🚀 MLX-LM (Recommended for Apple Silicon)"
echo "      Fastest option, uses Apple's MLX framework"
echo "      Model: Qwen2.5-3B-Instruct-4bit (~1.8GB download on first use)"
echo ""
echo "  [2] 🔧 llama-cpp-python (For Intel Macs or compatibility)"
echo "      Uses GGUF quantized models"
echo "      ⚠️  May require build tools and take time to compile"
echo ""
echo "  [3] ⚙️  Both (Install both backends)"
echo ""
echo "  [4] ❌ Skip (App works without LLM for transcription)"
echo ""

read -p "Select option [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "📥 Installing mlx-lm..."
        pip install --upgrade mlx-lm
        echo ""
        echo "✅ MLX-LM installed successfully!"
        echo "   The AI text reformer will use this backend."
        ;;
    2)
        echo ""
        echo "📥 Installing llama-cpp-python..."
        
        if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
            # Apple Silicon with Metal
            echo "   Configuring for Apple Silicon with Metal..."
            CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir
        else
            # Intel Mac
            echo "   Configuring for Intel Mac (no Metal)..."
            CMAKE_ARGS="-DLLAMA_METAL=off" pip install llama-cpp-python --no-cache-dir
        fi
        
        echo ""
        echo "✅ llama-cpp-python installed successfully!"
        ;;
    3)
        echo ""
        echo "📥 Installing both backends..."
        pip install --upgrade mlx-lm
        
        if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
            CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir || true
        else
            CMAKE_ARGS="-DLLAMA_METAL=off" pip install llama-cpp-python --no-cache-dir || true
        fi
        
        echo ""
        echo "✅ Both backends installed!"
        echo "   The app will use MLX-LM (preferred) with llama-cpp as fallback."
        ;;
    4)
        echo ""
        echo "⏭️  Skipping LLM installation."
        echo "   The transcription features will work without AI text refinement."
        echo ""
        echo "   You can install later by running this script again."
        exit 0
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ LLM Setup Complete!                        ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  Next steps:                                               ║"
echo "║  1. Run ./scripts/launch.command                           ║"
echo "║  2. Enable "🤖 AI Text Refiner" in the sidebar              ║"
echo "║  3. The model will download automatically (~2GB)           ║"
echo "║                                                            ║"
echo "║  Features available:                                       ║"
echo "║  • 📝 Punctuate & capitalize                               ║"
echo "║  • 📄 Structure paragraphs                                 ║"
echo "║  • 📋 Summarize content                                    ║"
echo "║  • 🔑 Extract key points                                   ║"
echo "║  • 📑 Format meeting notes                                 ║"
echo "║  • ✨ Clean filler words                                   ║"
echo "║  • 📊 Topic & sentiment analysis                           ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
read -n 1 -s -r -p "Press any key to close..."
