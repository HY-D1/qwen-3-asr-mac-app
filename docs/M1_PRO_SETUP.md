# Mac M1 Pro Setup Guide for Qwen3-ASR Pro

This guide provides solutions for running LLM text refinement on Mac M1 Pro (and other Apple Silicon Macs).

## 🎯 Quick Solutions (Pick One)

### ✅ Option 1: Use MLX-LM (Recommended - Easiest)

**Best for:** All Apple Silicon Macs (M1/M2/M3/M4)
**Pros:** Native Apple framework, no compilation, fastest performance
**Cons:** None for Apple Silicon

```bash
# If using conda (you are)
conda install -c conda-forge mlx-lm

# Or with pip
pip install mlx-lm
```

The app will automatically use MLX-LM on Apple Silicon. **This is already working for you!**

---

### 🔧 Option 2: Pre-built Wheel for llama-cpp-python

**Best for:** When you need GGUF model support
**Pros:** No compilation needed
**Cons:** Slightly less optimized than MLX

```bash
# Pre-built wheel with Metal support (M1/M2/M3/M4)
pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
```

---

### 🛠️ Option 3: Build from Source (If Pre-built Fails)

```bash
# 1. Clean environment
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH SDKROOT CC CXX CFLAGS CXXFLAGS

# 2. Set compiler paths
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export SDKROOT="$(xcrun --show-sdk-path)"

# 3. Install with Metal support
CMAKE_ARGS="-DGGML_METAL=on -DCMAKE_OSX_ARCHITECTURES=arm64" \
  pip install --no-cache-dir --force-reinstall llama-cpp-python
```

---

## 🔥 Fixing Your Current Error

Your error shows conda mixing x86_64 and arm64 architectures. Here's the fix:

### Step 1: Verify Architecture
```bash
# Should output: arm64
uname -m

# Should output: 0 (not in Rosetta)
sysctl -n sysctl.proc_translated 2>/dev/null || echo "0"
```

### Step 2: Use Conda-Forge for ARM64
```bash
# Install miniforge for Apple Silicon (if not already)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh

# Or ensure your current conda is ARM64
conda info | grep platform
# Should show: platform : osx-arm64
```

### Step 3: Install MLX-LM via Conda
```bash
# Cleanest approach - use conda-forge
conda install -c conda-forge mlx-lm

# This avoids pip compilation entirely!
```

---

## 📊 Performance Comparison on M1 Pro

| Backend | Setup Difficulty | Speed | Memory | Best For |
|---------|-----------------|-------|--------|----------|
| **MLX-LM** | ⭐ Easy | 🚀 Fastest | 🟢 Efficient | Apple Silicon |
| llama-cpp (Metal) | ⭐⭐ Medium | 🚀 Fast | 🟡 Good | GGUF models |
| llama-cpp (CPU) | ⭐⭐⭐ Hard | 🐢 Slow | 🔴 Heavy | Fallback only |

---

## 🎓 Understanding the Architecture Issue

Your error happens because:

```
# Conda is trying to compile with x86_64 tools
# But your M1 Pro needs arm64 binaries
# This causes the "fatal error: 'mutex' file not found"
```

**Solutions:**

1. **Use pre-built packages** (avoid compilation)
   ```bash
   conda install -c conda-forge mlx-lm
   ```

2. **Use system Python instead of conda**
   ```bash
   /usr/bin/python3 -m venv ~/venv_qwen
   source ~/venv_qwen/bin/activate
   pip install mlx-lm
   ```

3. **Fix conda architecture**
   ```bash
   # Ensure you're using arm64 conda
   conda create -n qwen_arm64 python=3.12
   conda activate qwen_arm64
   conda config --env --set subdir osx-arm64
   pip install mlx-lm
   ```

---

## ✅ Verified Working Setup for M1 Pro

```bash
#!/bin/bash
# Save as: setup_m1_pro.sh

echo "🚀 Setting up Qwen3-ASR Pro for M1 Pro..."

# Check architecture
if [ "$(uname -m)" != "arm64" ]; then
    echo "❌ Not running on ARM64. Please use native Terminal (not Rosetta)"
    exit 1
fi

# Option 1: Use conda-forge (recommended if using conda)
if command -v conda &> /dev/null; then
    echo "📦 Using conda-forge for ARM64..."
    conda install -c conda-forge mlx-lm -y
    
# Option 2: Use pip with virtual environment
else
    echo "📦 Creating virtual environment..."
    python3 -m venv backend/venv_m1
    source backend/venv_m1/bin/activate
    pip install --upgrade pip
    pip install mlx-lm
fi

echo "✅ Setup complete!"
echo ""
echo "Launch with: ./scripts/launch.command"
```

---

## 🧪 Testing Your Installation

```python
# test_mlx.py
print("Testing MLX-LM installation...")

try:
    from mlx_lm import load, generate
    print("✅ MLX-LM imported successfully")
    
    # Try loading a small model
    print("🧪 Loading test model (this may download ~400MB)...")
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    print("✅ Model loaded successfully!")
    
    # Test generation
    response = generate(model, tokenizer, "Hello!", max_tokens=10, verbose=False)
    print(f"✅ Generation test: '{response}'")
    
    print("\n🎉 All tests passed! Ready to use AI text refinement.")
    
except ImportError as e:
    print(f"❌ MLX-LM not installed: {e}")
    print("Run: conda install -c conda-forge mlx-lm")
    
except Exception as e:
    print(f"❌ Error: {e}")
```

Run test:
```bash
python test_mlx.py
```

---

## 🚨 Common M1 Pro Issues & Fixes

### Issue 1: "'mutex' file not found"
**Cause:** Mixing x86_64 and arm64 toolchains
**Fix:**
```bash
# Reinstall Xcode Command Line Tools
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install
# Then use conda-forge: conda install -c conda-forge mlx-lm
```

### Issue 2: "Illegal instruction" crash
**Cause:** Running x86_64 binary on ARM64
**Fix:**
```bash
# Ensure Python is ARM64
file $(which python)
# Should show: Mach-O 64-bit executable arm64

# If not, reinstall Python
brew install python@3.12
```

### Issue 3: Slow performance
**Cause:** Not using Metal acceleration
**Fix:** MLX-LM uses Metal automatically on Apple Silicon

---

## 📚 References

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [llama-cpp-python Metal Guide](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/)
- [Apple Silicon Optimization Guide](https://www.reddit.com/r/LocalLLM/comments/1m8ajjr/apple_silicon_optimization_guide/)

---

## 💡 Recommendation for Your Setup

**Given your conda environment issues, I recommend:**

```bash
# Cleanest solution - bypass conda entirely for LLM dependencies
/usr/bin/python3 -m venv ~/qwen_venv
source ~/qwen_venv/bin/activate
pip install mlx-lm mlx-qwen3-asr sounddevice numpy

# Then run app with this environment
python src/main.py
```

Or simply use conda-forge which handles ARM64 properly:
```bash
conda install -c conda-forge mlx-lm
```
