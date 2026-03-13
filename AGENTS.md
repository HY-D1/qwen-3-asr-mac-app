# Qwen3-ASR Pro - Agent Guide

## Project Overview

**Qwen3-ASR Pro** is a professional speech-to-text application for macOS with real-time streaming transcription, AI-powered text refinement, and multiple UI options (Tkinter GUI, Web UI, and CLI). It supports both Apple Silicon (MLX acceleration) and Intel Macs (PyTorch backend).

- **Version:** 3.3.0
- **License:** MIT (Copyright 2026 HY-D1)
- **Language:** English (documentation and comments)
- **Platform:** macOS only (Darwin)

### Key Features
- **🎓 Live Mode** - Real-time transcription with word-by-word output (~2s delay)
- **⚡ Fast Mode** - Batch processing for quick recordings
- **🤖 AI Text Refinement** - LLM-powered text reformation using Qwen2.5-3B or Ollama (8GB RAM compatible)
- **📁 Auto-Save** - Raw audio automatically saved to `~/Documents/Qwen3-ASR-Recordings/`
- **📱 Multiple Interfaces** - Tkinter GUI (main), Gradio Web UI, CLI
- **🎚️ Smart Silence Detection** - Adjustable auto-stop (0.5s - 60s)
- **⚡ MLX Acceleration** - Optimized for Apple Silicon (M1/M2/M3/M4)
- **🌍 Multi-language** - Supports 50+ languages with auto-detection

---

## Project Structure

```
qwen-3-asr-mac-app-main/
├── src/                       # Python source code
│   ├── __init__.py           # Package init, version 3.3.0
│   ├── main.py               # Entry point (imports from app.py)
│   ├── app.py                # Main Tkinter application (~2200 lines)
│   ├── constants.py          # Colors, settings, paths
│   ├── text_reformer.py      # LLM text refinement engine (MLX-based)
│   └── simple_llm.py         # Alternative LLM with Ollama/OpenAI backends
├── scripts/                   # Shell scripts (all .command files for macOS)
│   ├── launch.command        # Launch main Tkinter application
│   ├── launch_web.command    # Launch Gradio web interface
│   ├── launch_cli.command    # Launch CLI interactive mode
│   ├── setup.command         # Installation script
│   ├── install_llm.command   # Install LLM dependencies
│   ├── setup_ollama.command  # Setup Ollama for AI refinement
│   ├── kill_servers.command  # Kill running servers
│   └── stop_ollama.command   # Stop Ollama service
├── assets/                    # Resources
│   ├── c-asr/                # C implementation for live streaming
│   │   ├── qwen_asr         # Pre-built binary
│   │   ├── *.c, *.h         # C source files (C99)
│   │   ├── Makefile         # Build configuration (Apple Accelerate)
│   │   ├── download_model.sh # Model download script
│   │   ├── qwen3-asr-0.6b/  # 0.6B model directory
│   │   ├── qwen3-asr-1.7b/  # 1.7B model directory
│   │   └── samples/         # Test audio files (JFK speech, etc.)
│   └── models/              # Python ML models (downloaded)
├── backend/                   # Python virtual environment
│   └── venv/                 # Created by setup.command
├── tests/                     # Comprehensive test suite (40+ test files)
│   ├── test_ui.py            # UI component tests
│   ├── test_live_streaming_final.py
│   ├── test_models.py
│   ├── test_memory_leaks.py
│   ├── test_integration.py
│   ├── test_llm_reforming.py
│   ├── test_transcription_backends.py
│   ├── test_webui_integration.py
│   ├── test_cli_functionality.py
│   ├── test_performance_benchmarks.py
│   ├── test_edge_cases_real.py
│   ├── conftest.py           # Pytest configuration
│   ├── assets/               # Test audio samples
│   └── leak_reports/         # Memory leak test reports
├── docs/                      # Documentation
│   └── M1_PRO_SETUP.md       # Apple Silicon setup guide
├── web_ui.py                 # Gradio-based web interface
├── cli_app.py                # Command-line interface
├── README.md                 # User documentation
├── QUICK_START.md            # Quick start guide
├── PROJECT_DESIGN.md         # System architecture document
├── AGENTS.md                 # This file
├── LICENSE                   # MIT License
└── .gitignore                # Git ignore rules
```

---

## Technology Stack

### Core Technologies
- **Python 3.12+** - Main application language
- **Tkinter** - Main GUI framework (light theme, ~2200 lines)
- **Gradio** - Web UI framework
- **NumPy** - Audio processing
- **SoundDevice** - Audio I/O for recording
- **Wave** - Audio file handling

### ML Backends (Auto-detected in priority order)
1. **C Binary** (preferred for live streaming)
   - Pure C99 implementation at `assets/c-asr/qwen_asr`
   - Uses Apple Accelerate framework (BLAS)
   - Fastest for live streaming transcription
   
2. **MLX-Audio** (preferred on Apple Silicon for batch)
   - `mlx_audio.stt` module
   - Fastest batch processing
   
3. **MLX-CLI** (fallback)
   - `python -m mlx_qwen3_asr`
   - Subprocess-based
   
4. **PyTorch** (Intel Mac)
   - `qwen_asr` package
   - MPS acceleration on Apple Silicon

### LLM Backends for Text Refinement
1. **Ollama** (recommended, free)
   - Local Qwen models (1.8B, 4B, 7B)
   - Requires `ollama serve` running
   
2. **MLX-LM** (Apple Silicon)
   - `mlx-community/Qwen2.5-3B-Instruct-4bit`
   - ~1.8GB download, 8GB RAM compatible
   
3. **OpenAI API** (optional)
   - Cloud-based, requires API key
   
4. **Rule-based** (fallback)
   - Always works, no dependencies

### C Implementation
- **Language:** C99
- **Build:** GCC with Make
- **Acceleration:** Apple Accelerate (macOS) / OpenBLAS (Linux)
- **Binary:** `assets/c-asr/qwen_asr` (pre-built included)
- **Purpose:** Live streaming transcription with low latency

---

## Build and Run Commands

### First-time Setup
```bash
./scripts/setup.command
```
- Creates Python 3.12+ virtual environment in `backend/venv`
- Installs platform-specific dependencies
- Detects Apple Silicon vs Intel Mac

### Launch Main Application (Tkinter)
```bash
./scripts/launch.command
```

### Launch Web UI (Gradio)
```bash
./scripts/launch_web.command
```
- Opens browser at `http://localhost:7860`
- Auto-detects free ports if 7860 is in use

### Launch CLI
```bash
./scripts/launch_cli.command
# Or directly:
python cli_app.py -i  # Interactive mode
```

### Run from Source (Development)
```bash
source backend/venv/bin/activate
python src/main.py              # Tkinter GUI
python web_ui.py                # Web UI
python cli_app.py -i            # CLI interactive
```

### Build C Binary
```bash
cd assets/c-asr
make blas  # Uses Apple Accelerate on macOS
```

### Download Models
```bash
cd assets/c-asr
./download_model.sh --model small  # 0.6B
./download_model.sh --model large  # 1.7B
```

---

## Testing

### Run All Tests
```bash
# Run with Python unittest
python -m unittest discover tests/ -v

# Or with pytest
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# UI tests only
pytest tests/test_ui.py -v

# Live streaming tests
pytest tests/test_live_streaming_final.py -v

# LLM reforming tests
pytest tests/test_llm_reforming.py -v

# Performance benchmarks
pytest tests/test_performance_benchmarks.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

### Test Categories

| Test File | Purpose | Key Tests |
|-----------|---------|-----------|
| `test_ui.py` | UI constants | Colors, breakpoints, sidebar behavior |
| `test_live_streaming_final.py` | Live streaming | Threading, RTF, memory leaks |
| `test_models.py` | ML models | Model loading, transcription accuracy |
| `test_llm_reforming.py` | LLM features | Text reformation, analysis |
| `test_memory_leaks.py` | Memory stability | Temp file cleanup, process cleanup |
| `test_integration.py` | End-to-end | Full workflows, error recovery |
| `test_webui_integration.py` | Web UI | Gradio interface tests |
| `test_cli_functionality.py` | CLI | Command-line interface tests |
| `test_transcription_backends.py` | Backends | C-binary, MLX, PyTorch |
| `test_performance_benchmarks.py` | Performance | RTF measurements |
| `test_edge_cases_real.py` | Edge cases | Error handling, corner cases |

### Test Reports
- `tests/LIVE_STREAMING_FIX_REPORT.md` - Threading fix validation
- `tests/TRANSCRIPTION_TEST_REPORT.md` - Backend accuracy tests
- `LLM_TEST_REPORT.md` - LLM functionality tests
- `INTEGRATION_TEST_REPORT.md` - Integration test results
- `tests/PERFORMANCE_REPORT.md` - Benchmark results

---

## Code Organization

### Main Application (`src/app.py`)

#### Classes
1. **`QwenASRApp`** - Main application controller (~2200 lines)
   - UI setup and layout management
   - Recording control (start/stop)
   - File upload processing
   - LLM reformer integration
   - Event handling

2. **`CollapsibleSidebar`** - Settings sidebar
   - Recording controls
   - Language selection
   - Silence duration settings
   - LLM reformer controls
   - Collapsible (260px expanded, 60px compact)

3. **`SlideOutPanel`** - Mobile settings panel
   - Slides in from right
   - 300px width
   - Overlay for closing

4. **`BottomBar`** - Mobile control bar
   - 60px height
   - Record button, timer, settings

5. **`LiveStreamer`** - Live streaming transcription
   - 5-second chunk processing
   - ThreadPoolExecutor (max_workers=1)
   - subprocess.run() for C binary
   - Auto-saves raw audio

6. **`AudioRecorder`** - Audio capture with VAD
   - Configurable silence threshold
   - Auto-stop on silence
   - Real-time level callback

7. **`TranscriptionEngine`** - Upload transcription
   - Auto-detects backend (C-binary/MLX/PyTorch)
   - 1.7B model for best accuracy
   - Progress callbacks

8. **`WaveformVisualizer`** - Audio level display
   - 40-bar history
   - Color-coded levels (green/yellow/red)

### Text Reformer Module (`src/text_reformer.py`)

#### Classes
1. **`TextReformer`** - LLM-based text reformation engine
   - Uses Qwen2.5-3B-Instruct-4bit model (~1.8GB)
   - Supports MLX (Apple Silicon) and llama.cpp (Intel) backends
   - 8GB RAM compatible

2. **`SimpleLLM`** (`src/simple_llm.py`) - Alternative LLM
   - Ollama backend (recommended)
   - OpenAI API backend
   - Context-aware prompts
   - 1300+ lines with detailed prompt engineering

3. **`ReformMode`** (Enum) - Reformation modes
   - `PUNCTUATE` - Add punctuation and capitalization
   - `PARAGRAPH` - Structure into paragraphs
   - `SUMMARIZE` - Create summary
   - `KEY_POINTS` - Extract key points
   - `FORMAT` - Format as meeting notes
   - `CLEAN` - Remove filler words

### Key Constants (`src/constants.py`)
```python
APP_NAME = "Qwen3-ASR Pro"
VERSION = "3.3.0"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.05
MIN_WIDTH_COMPACT = 750  # px
MIN_WIDTH_MOBILE = 550   # px
COLORS = { ... }  # 17-color light theme
```

---

## Code Style Guidelines

### Naming Conventions
- **Classes:** `PascalCase` (e.g., `QwenASRApp`, `LiveStreamer`)
- **Methods/Variables:** `snake_case` (e.g., `transcribe_audio`, `chunk_duration`)
- **Constants:** `UPPER_CASE` (e.g., `SAMPLE_RATE`, `DEFAULT_LANGUAGE`)
- **Private:** Leading underscore (e.g., `_process_chunk`, `_on_frame_configure`)

### Documentation
- **Docstrings:** Triple-quote with description for all public classes and methods
- **Comments:** Inline for complex logic, use `#` for single-line comments
- **Type hints:** Used for public methods where appropriate

### Code Structure
```python
# Example from app.py
class LiveStreamer:
    """Live streaming transcription with 5-second chunks"""
    
    def __init__(self, app, model_dir: str):
        """
        Initialize live streamer.
        
        Args:
            app: Parent application instance
            model_dir: Path to model directory
        """
        self.app = app
        self.model_dir = model_dir
        # ...
```

---

## Responsive Layout

| Mode | Width | Layout |
|------|-------|--------|
| Desktop | > 750px | Full sidebar (260px) |
| Compact | 550-750px | Collapsed sidebar (60px) |
| Mobile | < 550px | Bottom bar + slide-out panel |

---

## Processing Modes

### Live Mode (🎓 Live)
```
Microphone → [5s chunks] → C binary (qwen_asr) → Live text + Raw file
```
- Always uses 0.6B model for stability
- Word-by-word output
- Raw audio saved to `~/Documents/Qwen3-ASR-Recordings/`

### Upload Mode (📁 File Upload)
```
Audio file → C-binary/MLX/PyTorch (1.7B) → Text
```
- 1.7B model for best accuracy
- Supports WAV, MP3, M4A, FLAC, OGG, AAC
- Batch processing with progress

---

## Development Conventions

### Thread Safety
- UI updates only from main thread via `root.after()`
- Background threads for transcription
- `queue.Queue` for thread communication
- `threading.Lock` for shared state

### Error Handling
```python
try:
    # Operation
except Exception as e:
    traceback.print_exc()
    self.status_queue.put(('error', str(e)))
```

### File Paths
- Always use `os.path.join()`
- Base directory: `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`
- Recordings: `os.path.expanduser("~/Documents/Qwen3-ASR-Recordings")`

### Audio Processing
- Format: 16-bit PCM, 16kHz, mono
- Normalized to float32 (-1.0 to 1.0)
- Conversion: `np.clip(audio * 32767, -32768, 32768).astype(np.int16)`

---

## Testing Strategy

### Unit Tests
- Test constants and configuration
- No GUI required (headless-friendly)
- Fast execution (< 1 second)

### Integration Tests
- Test full workflows
- Require audio samples
- Test file I/O

### Live Streaming Tests
- Test chunk processing
- Memory leak detection
- Process cleanup verification
- RTF performance measurement

### Test Data
- `tests/assets/` - Test audio files
- `assets/c-asr/samples/` - Sample recordings
- JFK speech sample for accuracy testing

---

## Security Considerations

### Microphone Permissions
- Requires Terminal/iTerm microphone access
- Settings: System Preferences → Security & Privacy → Microphone

### File System
- Creates recordings in user's Documents folder
- Temp files cleaned up after processing
- No network access required for core functionality

### Dependencies
- All dependencies from PyPI
- No external API keys required (optional OpenAI)
- Local ML inference only

---

## Known Issues and Fixes

### Threading Hang (Fixed)
**Issue:** Live streaming would hang due to PIPE buffer deadlocks  
**Fix:** Replaced `Popen` with `subprocess.run()`, added `ThreadPoolExecutor(max_workers=1)`

### Race Condition (Fixed)
**Issue:** Short audio (< 5s) not processed on stop  
**Fix:** Added `_process_chunk_sync()` for remaining audio processing

### Memory Leaks (Monitored)
- Temp file cleanup verified in `test_memory_leaks.py`
- Process cleanup verified (no zombies)
- Acceptable memory growth: ~20-30MB per session

---

## Performance Targets

| Model | Mode | Target RTF | Notes |
|-------|------|------------|-------|
| 0.6B | Upload | ~0.02x | Batch processing |
| 1.7B | Upload | ~0.03x | Batch processing |
| 0.6B | Streaming | < 3.0x | Live mode (includes overhead) |

*RTF (Real-Time Factor) < 1.0 means faster than real-time*

---

## Deployment

### Distribution
- Not packaged as .app bundle
- Run via `launch.command` scripts
- Requires Python 3.12+ installed

### User Data
- Recordings: `~/Documents/Qwen3-ASR-Recordings/`
- Naming: `live_YYYYMMDD_HHMMSS.wav`

---

## Version History

- **3.3.0** - Current: AI Text Refinement with Qwen2.5-3B LLM integration
- **3.2.0** - Simplified UI, auto language detection, model optimization
- **3.1.1** - Dual mode (Live/Fast), manual model selection

---

## References

- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [mlx-audio GitHub](https://github.com/Blaizzy/mlx-audio)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [C Implementation](https://github.com/antirez/qwen-asr)

---

**Last Updated:** 2026-03-13  
**Maintainer:** HY-D1
