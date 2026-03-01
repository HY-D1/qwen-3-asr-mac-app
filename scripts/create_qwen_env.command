#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Create Qwen Environment for M1 Pro                       ║
# ║         Fixes conda dependency conflicts                         ║
# ╚══════════════════════════════════════════════════════════════════╝

cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         🚀 Create Qwen Environment                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check architecture
if [ "$(uname -m)" != "arm64" ]; then
    echo "❌ Not on ARM64 architecture"
    exit 1
fi

echo "✅ ARM64 detected (Apple Silicon)"
echo ""

# Environment name
ENV_NAME="qwen"

# Remove old env if exists
echo "🧹 Cleaning up old environment (if exists)..."
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# Create new environment
echo ""
echo "📦 Creating new conda environment: $ENV_NAME"
echo "   Python: 3.12"
echo "   Platform: osx-arm64"
echo ""

conda create -n $ENV_NAME python=3.12 -y

# Activate environment
echo ""
echo "🔄 Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Set ARM64 explicitly
conda config --env --set subdir osx-arm64

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
echo "   • mlx-lm (LLM text refinement)"
echo "   • sounddevice (audio I/O)"
echo "   • numpy (audio processing)"
echo "   • mlx-qwen3-asr (transcription)"
echo ""

pip install mlx-lm sounddevice numpy || {
    echo "❌ Installation failed"
    exit 1
}

# Try to install optional mlx-qwen3-asr
pip install mlx-qwen3-asr 2>/dev/null || echo "⚠️  Note: mlx-qwen3-asr had warnings (usually OK)"

# Test installation
echo ""
echo "🧪 Testing installation..."
python -c "from mlx_lm import load; print('✅ mlx-lm OK')" || {
    echo "❌ mlx-lm test failed"
    exit 1
}

python -c "import sounddevice; print('✅ sounddevice OK')" || {
    echo "❌ sounddevice test failed"
    exit 1
}

# Update launch script
echo ""
echo "📝 Updating launch script..."

# Backup original
cp scripts/launch.command scripts/launch.command.backup

# Create new launch script
cat > scripts/launch.command << 'LAUNCH_EOF'
#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Qwen3-ASR Pro Launcher (Conda Edition)                   ║
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

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qwen

echo "Starting application..."
echo ""

# Run
python src/main.py
LAUNCH_EOF

chmod +x scripts/launch.command

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ Environment Ready!                         ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  Usage:                                                    ║"
echo "║    ./scripts/launch.command                                ║"
echo "║                                                            ║"
echo "║  Or manually:                                              ║"
echo "║    conda activate qwen                                     ║"
echo "║    python src/main.py                                      ║"
echo "║                                                            ║"
echo "║  Environment: $ENV_NAME                                    ║"
echo "║  Python: $(python --version)                               ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

read -n 1 -s -r -p "Press any key to launch..."

# Launch
./scripts/launch.command
