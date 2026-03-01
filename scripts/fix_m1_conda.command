#!/bin/bash

# ╔══════════════════════════════════════════════════════════════════╗
# ║         Fix M1 Pro Conda Setup                                   ║
# ║         Resolves x86_64/arm64 architecture mixing                ║
# ╚══════════════════════════════════════════════════════════════════╝

cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         🔧 M1 Pro Conda Fix                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if running on ARM64
if [ "$(uname -m)" != "arm64" ]; then
    echo "❌ Not running on ARM64 architecture"
    echo "   Please use a native Terminal (not Rosetta)"
    exit 1
fi

echo "✅ Running on ARM64 (Apple Silicon)"
echo ""

# Check conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found"
    exit 1
fi

echo "📊 Current conda info:"
conda info | grep -E "(platform|base environment)"
echo ""

# Option 1: Try conda-forge (easiest)
echo "═══════════════════════════════════════════════════════════"
echo " OPTION 1: Install via conda-forge (Recommended)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "This will install pre-built ARM64 packages (no compilation):"
echo ""
read -p "Install mlx-lm via conda-forge? [Y/n]: " choice

if [[ "$choice" =~ ^[Yy]$ ]] || [[ -z "$choice" ]]; then
    echo ""
    echo "📦 Installing from conda-forge..."
    conda install -c conda-forge -y mlx-lm || {
        echo "❌ conda-forge install failed"
        echo ""
        echo "Falling back to Option 2..."
    }
    
    if python -c "from mlx_lm import load; print('✅ mlx-lm working')" 2>/dev/null; then
        echo ""
        echo "✅ SUCCESS! mlx-lm is installed and working."
        echo ""
        read -n 1 -s -r -p "Press any key to close..."
        exit 0
    fi
fi

# Option 2: Create fresh ARM64 environment
echo ""
echo "═══════════════════════════════════════════════════════════"
echo " OPTION 2: Create Fresh ARM64 Environment"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "This creates a clean ARM64 conda environment:"
echo ""
read -p "Create new 'qwen_arm64' environment? [Y/n]: " choice

if [[ "$choice" =~ ^[Yy]$ ]] || [[ -z "$choice" ]]; then
    echo ""
    echo "🧹 Creating fresh ARM64 environment..."
    
    # Remove old env if exists
    conda env remove -n qwen_arm64 -y 2>/dev/null || true
    
    # Create new env with explicit ARM64
    conda create -n qwen_arm64 python=3.12 -y
    
    # Activate and configure
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate qwen_arm64
    conda config --env --set subdir osx-arm64
    
    echo ""
    echo "📦 Installing dependencies..."
    pip install --upgrade pip
    pip install mlx-lm mlx-qwen3-asr sounddevice numpy
    
    echo ""
    echo "🧪 Testing installation..."
    if python -c "from mlx_lm import load; print('✅ mlx-lm working')" 2>/dev/null; then
        echo ""
        echo "✅ SUCCESS! Environment 'qwen_arm64' is ready."
        echo ""
        echo "To use it in the future:"
        echo "  conda activate qwen_arm64"
        echo "  python src/main.py"
        echo ""
        
        # Update launch script temporarily
        echo "📝 Would you like to update the launch script to use this environment?"
        read -p "Update launch.command? [y/N]: " update_choice
        
        if [[ "$update_choice" =~ ^[Yy]$ ]]; then
            sed -i.bak 's|source backend/venv/bin/activate|conda activate qwen_arm64|' scripts/launch.command
            echo "✅ launch.command updated (backup saved as .bak)"
        fi
        
        read -n 1 -s -r -p "Press any key to close..."
        exit 0
    fi
fi

# Option 3: System Python
echo ""
echo "═══════════════════════════════════════════════════════════"
echo " OPTION 3: Use System Python (Bypass Conda)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "This creates a venv using system Python:"
echo ""
read -p "Create system Python venv? [Y/n]: " choice

if [[ "$choice" =~ ^[Yy]$ ]] || [[ -z "$choice" ]]; then
    echo ""
    echo "🧹 Creating system Python environment..."
    
    # Use system Python
    /usr/bin/python3 -m venv backend/venv_system
    source backend/venv_system/bin/activate
    
    echo ""
    echo "📦 Installing dependencies..."
    pip install --upgrade pip
    pip install mlx-lm mlx-qwen3-asr sounddevice numpy
    
    echo ""
    echo "🧪 Testing installation..."
    if python -c "from mlx_lm import load; print('✅ mlx-lm working')" 2>/dev/null; then
        echo ""
        echo "✅ SUCCESS! System Python environment is ready."
        echo ""
        echo "To use it, modify scripts/launch.command:"
        echo "  Change: source backend/venv/bin/activate"
        echo "  To:     source backend/venv_system/bin/activate"
        echo ""
    fi
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "              Troubleshooting Tips"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "If nothing worked, try:"
echo ""
echo "1. Reinstall Xcode Command Line Tools:"
echo "   sudo rm -rf /Library/Developer/CommandLineTools"
echo "   xcode-select --install"
echo ""
echo "2. Clear conda cache:"
echo "   conda clean --all"
echo ""
echo "3. Install miniforge (ARM64-only conda):"
echo "   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
echo "   bash Miniforge3-MacOSX-arm64.sh"
echo ""
echo "4. Skip LLM (app works without it):"
echo "   The transcription features work without AI text refinement."
echo ""

read -n 1 -s -r -p "Press any key to close..."
