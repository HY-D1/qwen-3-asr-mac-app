# 🎙️ Qwen3-ASR Pro

A professional speech-to-text application for macOS with real-time streaming transcription, AI-powered text refinement, and responsive UI.

![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)
![Architecture](https://img.shields.io/badge/arch-Apple%20Silicon%20%7C%20Intel-blue)
![Backend](https://img.shields.io/badge/backend-MLX%20%7C%20PyTorch-green)
![LLM](https://img.shields.io/badge/LLM-Qwen2.5--3B-purple)

## ✨ Features

- **🎓 Live Class Mode** - Real-time transcription with text appearing as you speak
- **🤖 AI Text Refinement** - LLM-powered text reformation and analysis (8GB RAM compatible)
- **⚡ Fast Mode** - Optimized batch processing for quick recordings
- **📁 Auto-Save** - Raw audio automatically saved for later review
- **📱 Responsive UI** - Adapts to any window size (desktop/compact/mobile)
- **🎚️ Smart Silence Detection** - Adjustable auto-stop (0.5s - 60s)
- **⚡ MLX Acceleration** - Optimized for Apple Silicon (M1/M2/M3/M4)
- **🌍 Multi-language** - Supports 50+ languages

## 🚀 Quick Start

```bash
# First-time setup
./scripts/setup.command

# Launch application
./scripts/launch.command
```

## 📁 Project Structure

```
qwen-3-asr-mac-app-main/
├── src/                       # Source code
│   ├── __init__.py
│   ├── main.py               # Entry point
│   ├── app.py                # Main application
│   └── constants.py          # Colors, settings
├── scripts/                   # Shell scripts
│   ├── launch.command        # Launch application
│   └── setup.command         # Installation script
├── assets/                    # Resources
│   ├── c-asr/                # C implementation
│   │   ├── qwen_asr         # Binary
│   │   ├── download_model.sh
│   │   └── samples/         # Test audio
│   └── models/              # ML models (downloaded)
├── backend/                   # Python virtual environment
├── docs/                      # Documentation
├── tests/                     # Test files
├── README.md                  # This file
├── LICENSE                    # MIT License
└── .gitignore
```

## 📱 Responsive Layout

The UI automatically adapts to your window size:

| Mode | Window Width | Layout |
|------|--------------|--------|
| **Desktop** | > 750px | Full sidebar with all controls |
| **Compact** | 550-750px | Collapsible sidebar |
| **Mobile** | < 550px | Bottom bar + slide-out settings |

**Manual Toggle**: Click ◀/▶ to collapse/expand the sidebar.

## 🎓 Recording Modes

### Live Class Mode (Recommended)
```
🎤 Microphone → [Stream] → 📺 Live Text + 💾 Raw File
```
- Text appears **word-by-word** as you speak (~2s delay)
- Raw audio saved to `~/Documents/Qwen3-ASR-Recordings/`
- Perfect for lectures and meetings

### Fast Mode
```
🎤 Microphone → [Save] → [Process] → 📄 Text
```
- Faster processing (0.02x RTF vs 0.46x)
- Best for quick voice memos

## 🤖 AI Text Refinement

Qwen3-ASR Pro includes an optional LLM-powered text refinement feature using **Qwen2.5-3B-Instruct** - optimized for 8GB RAM systems.

### Features
- **✨ Punctuate** - Add proper punctuation and capitalization
- **📄 Paragraph** - Structure text into readable paragraphs
- **📋 Summarize** - Create concise summaries
- **🔑 Key Points** - Extract important bullet points
- **📑 Meeting Notes** - Format as structured meeting notes
- **✨ Clean Up** - Remove filler words (um, uh, like)

### Analysis Tools
- **📊 Topic Extraction** - Identify main discussion topics
- **😊 Sentiment Analysis** - Detect positive/negative tone
- **🔑 Keyword Detection** - Find important keywords
- **👥 Entity Recognition** - Extract people, organizations, locations

### Requirements
```bash
# Install LLM dependencies (optional)
pip install mlx-lm          # For Apple Silicon (recommended)
pip install llama-cpp-python # For Intel Macs
```

The 3B parameter model (~2GB download) runs efficiently on 8GB RAM systems with MLX acceleration on Apple Silicon.

## 🎛️ Settings

### Model Selection
- **0.6B (Fast)** - Faster, good accuracy
- **1.7B (Accurate)** - Higher accuracy, slower (recommended)

### Language
- **Auto** - Automatic detection
- **English, Chinese, Japanese, Korean, Spanish, French, German**

### Auto-Stop Silence
| Preset | Duration | Best For |
|--------|----------|----------|
| Fast | 0.8s | Quick notes |
| Class | 30s | Lectures (recommended) |
| Max | 60s | Long pauses |

**Manual**: Use slider for any value 0.5s - 60s.

## 📁 File Organization

Recordings automatically saved to:
```
~/Documents/Qwen3-ASR-Recordings/
├── class_20240227_103000.wav    # Raw audio
├── class_20240227_113000.wav    # Next recording
└── ...
```

## 📊 Performance

| Model | Mode | Speed | Use Case |
|-------|------|-------|----------|
| 0.6B | MLX | ~0.02x RTF | Fast transcription |
| 1.7B | MLX | ~0.03x RTF | Best accuracy |
| 0.6B | Streaming | ~0.46x RTF | Live transcription |

*RTF (Real-Time Factor): Lower is faster*

## 🎯 Usage Tips

### During Class
1. Select **🎓 Live Class Mode**
2. Set **Auto-stop** to 30s (Class preset)
3. Click **Start Recording**
4. Watch text appear live as professor speaks!

### After Class
1. Click **📁 Open Folder** to find raw audio
2. Copy transcript from app to your notes
3. Re-listen to confusing sections using raw file

## 🔧 Troubleshooting

### No Audio Detected
```bash
# Check microphone permissions
System Preferences → Security & Privacy → Microphone → Terminal ✅
```

### C Binary Not Found
```bash
cd assets/c-asr
make blas
```

### Model Download Issues
```bash
# Download manually
cd assets/c-asr
./download_model.sh --model small  # 0.6B
./download_model.sh --model large  # 1.7B
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen3-ASR Pro                            │
├─────────────────────────────────────────────────────────────┤
│  🎓 Live Mode        │  ⚡ Fast Mode                        │
│  ┌──────────────────┴─────────────────────────────────┐   │
│  │ C Binary (Streaming)    │    MLX (Batch)          │   │
│  │ • Real-time output      │    • Fast processing    │   │
│  │ • Word-by-word          │    • High throughput    │   │
│  └─────────────────────────────────────────────────────┘   │
│                        ↓                                    │
│              Responsive Tkinter UI                          │
│         (Desktop / Compact / Mobile)                        │
└─────────────────────────────────────────────────────────────┘
```

## 🧑‍💻 Development

### Running from Source
```bash
# Activate virtual environment
source backend/venv/bin/activate

# Run directly
python src/main.py
```

### Building C Binary
```bash
cd assets/c-asr
make blas
```

## 🔗 References

- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [mlx-audio GitHub](https://github.com/Blaizzy/mlx-audio)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [C Implementation](https://github.com/antirez/qwen-asr)

## 📝 License

MIT License - Same as Qwen3-ASR and mlx-audio

---

**Made for macOS with ❤️**
