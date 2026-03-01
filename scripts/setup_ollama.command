#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Setup Ollama for Local LLM                               ║
# ║         Free, Open Source, Runs Locally                          ║
# ╚══════════════════════════════════════════════════════════════════╝

cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         🤖 Ollama Setup                                    ║"
echo "║         Free Local LLM (Qwen, Llama, Phi, etc.)            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found"
    echo ""
    echo "Installing Ollama..."
    echo ""
    
    # Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh
    
    if [ $? -ne 0 ]; then
        echo "❌ Ollama installation failed"
        echo ""
        echo "Please install manually from: https://ollama.com/download"
        exit 1
    fi
    
    echo "✅ Ollama installed"
else
    echo "✅ Ollama already installed"
fi

echo ""
echo "Starting Ollama server..."

# Start Ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait for server to start
sleep 3

echo ""
echo "═══════════════════════════════════════════════════════════"
echo " AVAILABLE MODELS"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Choose a model to download:"
echo ""
echo "  [1] qwen:1.8b (Recommended for 8GB RAM)"
echo "      • Size: ~1GB"
echo "      • Speed: Very fast"
echo "      • Quality: Good"
echo ""
echo "  [2] qwen:4b (Better quality)"
echo "      • Size: ~2.5GB"
echo "      • Speed: Fast"
echo "      • Quality: Better"
echo ""
echo "  [3] qwen:7b (Best quality)"
echo "      • Size: ~4GB"
echo "      • Speed: Moderate"
echo "      • Quality: Best"
echo ""
echo "  [4] llama3.2:3b (Meta)"
echo "      • Size: ~2GB"
echo "      • Good alternative"
echo ""
echo "  [5] phi3:3.8b (Microsoft)"
echo "      • Size: ~2GB"
echo "      • Good for summarization"
echo ""

read -p "Select model [1-5, default: 1]: " choice

case $choice in
    2) MODEL="qwen:4b" ;;
    3) MODEL="qwen:7b" ;;
    4) MODEL="llama3.2:3b" ;;
    5) MODEL="phi3:3.8b" ;;
    *) MODEL="qwen:1.8b" ;;
esac

echo ""
echo "📥 Downloading $MODEL..."
echo "   (This may take a few minutes depending on your internet)"
echo ""

ollama pull $MODEL

if [ $? -ne 0 ]; then
    echo "❌ Failed to download model"
    kill $OLLAMA_PID 2>/dev/null
    exit 1
fi

echo ""
echo "✅ Model downloaded successfully!"
echo ""

# Test the model
echo "🧪 Testing model..."
TEST_RESULT=$(curl -s http://localhost:11434/api/generate -d "{
  \"model\": \"$MODEL\",
  \"prompt\": \"Say hello\",
  \"stream\": false
}" | grep -o '"response":"[^"]*"' | cut -d'"' -f4)

if [ -n "$TEST_RESULT" ]; then
    echo "✅ Model test successful!"
    echo "   Response: $TEST_RESULT"
else
    echo "⚠️  Model test had issues, but should work"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ Setup Complete!                            ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  Model: $MODEL"
echo "║                                                            ║"
echo "║  Usage:                                                    ║"
echo "║    ./scripts/launch_web.command                            ║"
echo "║                                                            ║"
echo "║  The AI will automatically use Ollama!                     ║"
echo "║                                                            ║"
echo "║  To use a different model, run:                            ║"
echo "║    ollama pull <model-name>                                ║"
echo "║                                                            ║"
echo "║  Available models:                                         ║"
echo "║    ollama list                                             ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Keep server running
read -n 1 -s -r -p "Press any key to stop the server..."
echo ""

kill $OLLAMA_PID 2>/dev/null
echo "Server stopped"
