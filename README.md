# 🎙️ Qwen3-ASR Pro

A professional speech-to-text application for macOS, optimized for Apple Silicon with MLX acceleration.

![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)
![Architecture](https://img.shields.io/badge/arch-Apple%20Silicon%20%7C%20Intel-blue)
![Backend](https://img.shields.io/badge/backend-MLX%20%7C%20PyTorch-green)

## ✨ Features

- **🎤 Real-time Recording** - One-click recording with configurable auto-stop
- **📁 File Upload** - Support for WAV, MP3, M4A, FLAC, OGG
- **⚡ MLX Acceleration** - Optimized for Apple Silicon (M1/M2/M3/M4)
- **🎚️ Smart Silence Detection** - Adjustable pause duration (0.5s - 5s)
- **📊 Performance Monitoring** - Real-time RTF (Real-Time Factor) metrics
- **🌍 Multi-language** - Support for 50+ languages

## 🚀 Quick Start

```bash
cd macos-asr-app
./SETUP.command      # First-time setup
./Qwen3-ASR.command  # Launch application
```

## 🎛️ Settings Guide

### Auto-Stop Silence Duration
Control how long the app waits before auto-stopping recording:

| Preset | Duration | Best For |
|--------|----------|----------|
| **Fast** | 0.8s | Quick notes, commands |
| **Normal** | 2.0s | General purpose (default) |
| **Patient** | 3.5s | Natural speech with pauses |

**Manual Adjustment**: Use the slider to set any value from 0.5s to 5.0s.

### Model Selection
- **Qwen/Qwen3-ASR-0.6B** - Faster, good accuracy (recommended for most use cases)
- **Qwen/Qwen3-ASR-1.7B** - Higher accuracy, slower processing

### Language
- **Auto** - Automatic language detection (default)
- **English, Chinese, Japanese, Korean, Spanish, French, German** - Force specific language

## ⚡ Performance

| Model | Hardware | RTF | Speed |
|-------|----------|-----|-------|
| 0.6B | M4 Max | ~0.02x | ~50x real-time |
| 0.6B | M3 | ~0.03x | ~33x real-time |
| 1.7B | M4 Max | ~0.05x | ~20x real-time |

*RTF (Real-Time Factor): Lower is faster. 0.02x means processing is 50x faster than real-time.*

## 🎯 Tips

### Recording Stops Too Fast?
1. Increase **Auto-stop silence** duration using the slider
2. Click **Patient** preset (3.5s)
3. Speak continuously without long pauses

### Best Accuracy
1. Select **1.7B model**
2. Set language explicitly (not Auto)
3. Minimize background noise
4. Speak clearly at moderate pace

### Maximum Speed
1. Select **0.6B model**
2. Use default settings
3. Close other GPU-intensive applications

## 📁 File Structure

```
macos-asr-app/
├── Qwen3-ASR.command      # Main launcher
├── SETUP.command          # Installation script
├── qwen_asr_app.py        # Main application
└── README.md              # Documentation
```

## 🔧 Troubleshooting

### No Audio Detected
1. Check **System Preferences → Security & Privacy → Microphone**
2. Ensure Terminal (or your terminal app) has microphone permission
3. Try clicking **Reset** and record again

### Slow Performance
1. Check backend indicator in top-right (should show "⚡ MLX")
2. Use **0.6B model** instead of 1.7B
3. Close other applications using GPU
4. Restart the app

### Transcription Errors
1. Check audio file is not corrupted
2. Try converting to WAV format first
3. Ensure model files are downloaded (first run requires download)
4. Check internet connection for initial model download

### Backend Issues
The app automatically selects the best available backend:
1. **MLX-Audio** (Python API) - Best option, full features
2. **MLX-CLI** (Command line) - Reliable fallback
3. **PyTorch** - For Intel Macs or if MLX not available

## 🌐 Supported Languages

Qwen3-ASR supports 50+ languages including:
- **Chinese** (Mandarin, Cantonese, Sichuanese, + 19 dialects)
- **English** (US, UK, AU, + multiple accents)
- **European**: French, German, Spanish, Italian, Portuguese, Russian, etc.
- **Asian**: Japanese, Korean, Thai, Vietnamese, Indonesian, etc.

## 🔗 References

- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [mlx-audio GitHub](https://github.com/Blaizzy/mlx-audio)
- [MLX Framework](https://github.com/ml-explore/mlx)

## 📝 License

MIT License - Same as Qwen3-ASR and mlx-audio
