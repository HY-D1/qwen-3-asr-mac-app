#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Setup Local LLM for Mac M1 8GB                           ║
# ║         Uses Qwen2.5-0.5B-Instruct (~1GB download)               ║
# ╚══════════════════════════════════════════════════════════════════╝

cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         🤖 Local LLM Setup for Mac M1 8GB                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check architecture
if [ "$(uname -m)" != "arm64" ]; then
    echo "❌ This setup is for Apple Silicon (M1/M2/M3/M4) only"
    exit 1
fi

# Check RAM
RAM_GB=$(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024))
echo "✅ Detected: Mac with $RAM_GB GB RAM"
echo ""

if [ $RAM_GB -lt 8 ]; then
    echo "⚠️  Warning: Less than 8GB RAM detected"
    echo "   The 0.5B model should still work but may be slower"
    echo ""
fi

# Model selection
echo "Select model size:"
echo ""
echo "  [1] 0.5B - Recommended for 8GB (fastest, ~1GB download)"
echo "  [2] 1.5B - Better quality, fits 8GB (~3GB download)"
echo "  [3] 3B - Best quality, risky on 8GB (~5GB download)"
echo ""
read -p "Choice [1-3, default: 1]: " model_choice

case $model_choice in
    2) MODEL_SIZE="1.5B" ;;
    3) MODEL_SIZE="3B" ;;
    *) MODEL_SIZE="0.5B" ;;
esac

echo ""
echo "Selected: $MODEL_SIZE model"
echo ""

# Create environment
echo "═══════════════════════════════════════════════════════════"
echo "Step 1: Creating Python Environment"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Try different methods to create environment
if command -v conda &> /dev/null; then
    echo "Using conda..."
    conda create -n qwen python=3.12 -y
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate qwen
else
    echo "Using system Python..."
    /usr/bin/python3 -m venv backend/venv_local
    source backend/venv_local/bin/activate
fi

echo ""
echo "✅ Environment ready"
echo ""

# Install dependencies
echo "═══════════════════════════════════════════════════════════"
echo "Step 2: Installing Dependencies"
echo "═══════════════════════════════════════════════════════════"
echo ""

pip install --upgrade pip setuptools wheel

echo "Installing packages..."
echo "  • transformers (for LLM)"
echo "  • torch (PyTorch backend)"
echo "  • sounddevice (audio)"
echo "  • numpy (processing)"
echo ""

pip install transformers torch sounddevice numpy

# Optional: Try to install MLX if available
if pip install mlx-lm 2>/dev/null; then
    echo "✅ Also installed mlx-lm (Apple Silicon optimized)"
fi

echo ""
echo "✅ Dependencies installed"
echo ""

# Test import
echo "═══════════════════════════════════════════════════════════"
echo "Step 3: Testing Installation"
echo "═══════════════════════════════════════════════════════════"
echo ""

python -c "
import sys
print(f'Python: {sys.version}')
print(f'Platform: {sys.platform}')

print('Testing imports...')
from transformers import AutoModelForCausalLM, AutoTokenizer
print('✅ transformers OK')

import torch
print(f'✅ PyTorch {torch.__version__}')
print(f'   MPS available: {torch.backends.mps.is_available()}')

import sounddevice
print('✅ sounddevice OK')

import numpy
print('✅ numpy OK')

print('')
print('All dependencies ready!')
"

if [ $? -ne 0 ]; then
    echo "❌ Installation test failed"
    exit 1
fi

# Download model
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Step 4: Downloading Model ($MODEL_SIZE)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "This will download the model (~1GB for 0.5B, ~3GB for 1.5B)"
echo "It only needs to be done once."
echo ""
read -n 1 -s -r -p "Press any key to start download..."
echo ""

python << PYTHON_EOF
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "mlx-community/Qwen2.5-${MODEL_SIZE}-Instruct-4bit" if "${MODEL_SIZE}" != "0.5B" else "Qwen/Qwen2.5-0.5B-Instruct"

print(f"Downloading: {model_name}")
print("")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✅ Tokenizer downloaded")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"✅ Model downloaded")
    print("")
    print("Testing model...")
    
    # Quick test
    inputs = tokenizer("Hello, this is a test.", return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=10)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Test output: {result}")
    print("")
    print("✅ Model working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
PYTHON_EOF

if [ $? -ne 0 ]; then
    echo "❌ Model download failed"
    exit 1
fi

# Update app to use local model
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Step 5: Configuring App"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Create a launcher script
cat > scripts/launch_local.command << LAUNCH_EOF
#!/bin/bash
cd "\$(dirname "\$0")/.."

# Activate environment
if [ -d "backend/venv_local" ]; then
    source backend/venv_local/bin/activate
else
    source "\$(conda info --base)/etc/profile.d/conda.sh"
    conda activate qwen
fi

echo "🚀 Starting Qwen3-ASR Pro with Local LLM"
echo "   Model: ${MODEL_SIZE}"
echo ""

# Set environment variable to use local model
export USE_LOCAL_LLM=true
export LOCAL_LLM_SIZE=${MODEL_SIZE}

python src/main.py
LAUNCH_EOF

chmod +x scripts/launch_local.command

echo "✅ Created scripts/launch_local.command"
echo ""

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              ✅ Setup Complete!                            ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  Model: Qwen2.5-${MODEL_SIZE}-Instruct                       ║"
echo "║  Size: $([ "$MODEL_SIZE" = "0.5B" ] && echo "~1GB" || ([ "$MODEL_SIZE" = "1.5B" ] && echo "~3GB" || echo "~5GB"))"
echo "║                                                            ║"
echo "║  Usage:                                                    ║"
echo "║    ./scripts/launch_local.command                          ║"
echo "║                                                            ║"
echo "║  Or manually:                                              ║"
echo "║    conda activate qwen  # or:                              ║"
echo "║    source backend/venv_local/bin/activate                  ║"
echo "║    python src/main.py                                      ║"
echo "║                                                            ║"
echo "║  In the app:                                               ║"
echo "║    1. Record or upload audio                               ║"
echo "║    2. Enable "🤖 AI Text Refiner"                          ║"
echo "║    3. Click "✨ Reform Text"                               ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

read -n 1 -s -r -p "Press any key to launch..."
echo ""

# Launch
./scripts/launch_local.command
