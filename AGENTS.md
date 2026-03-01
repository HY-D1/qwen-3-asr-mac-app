# Qwen3-ASR Pro - Agent Guide

## Project Overview

**Qwen3-ASR Pro** is a professional speech-to-text application for macOS with real-time streaming transcription and a responsive Tkinter-based UI. It supports both Apple Silicon (MLX acceleration) and Intel Macs (PyTorch backend).

- **Version:** 3.1.1
- **License:** MIT (Copyright 2026 HY-D1)
- **Language:** English (documentation and comments)
- **Platform:** macOS only (Darwin)

### Key Features
- **🎓 Live Class Mode** - Real-time transcription with word-by-word output (~2s delay)
- **⚡ Fast Mode** - Optimized batch processing for quick recordings
- **📁 Auto-Save** - Raw audio automatically saved to `~/Documents/Qwen3-ASR-Recordings/`
- **📱 Responsive UI** - Adapts to any window size (desktop/compact/mobile)
- **🎚️ Smart Silence Detection** - Adjustable auto-stop (0.5s - 60s)
- **⚡ MLX Acceleration** - Optimized for Apple Silicon (M1/M2/M3/M4)
- **🌍 Multi-language** - Supports 50+ languages

## Project Structure

```
qwen-3-asr-mac-app-main/
├── src/                       # Python source code
│   ├── __init__.py           # Package init, version 3.1.1
│   ├── main.py               # Entry point
│   ├── app.py                # Main application (~1600 lines)
│   ├── constants.py          # Colors, settings, paths
│   ├── core/                 # (empty - reserved)
│   └── ui/                   # (empty - reserved)
├── scripts/                   # Shell scripts
│   ├── launch.command        # Launch application
│   └── setup.command         # Installation script
├── assets/                    # Resources
│   ├── c-asr/                # C implementation
│   │   ├── qwen_asr         # Pre-built binary
│   │   ├── *.c, *.h         # C source files
│   │   ├── Makefile         # Build configuration
│   │   ├── download_model.sh
│   │   ├── qwen3-asr-0.6b/  # Model directory
│   │   ├── qwen3-asr-1.7b/  # Model directory
│   │   └── samples/         # Test audio files
│   └── models/              # Python ML models (downloaded)
├── backend/                   # Python virtual environment
│   └── venv/                 # Created by setup.command
├── tests/                     # Comprehensive test suite
│   ├── test_ui.py            # UI component tests
│   ├── test_live_streaming.py
│   ├── test_live_streaming_final.py
│   ├── test_models.py
│   ├── test_memory_leaks.py
│   ├── test_integration.py
│   ├── test_error_handling.py
│   ├── test_file_io.py
│   ├── test_macos_compat.py
│   ├── test_recording_vad.py
│   ├── test_batch_mode.py
│   ├── test_user_workflows.py
│   ├── README.md             # Test documentation
│   ├── TEST_REPORT.md        # Live streaming test report
│   ├── LIVE_STREAMING_FIX_REPORT.md
│   └── assets/               # Test audio samples
├── docs/                      # (empty)
├── README.md                  # User documentation
├── LICENSE                    # MIT License
└── .gitignore                 # Git ignore rules
```

## Technology Stack

### Core Technologies
- **Python 3.12+** - Main application language
- **Tkinter** - GUI framework (light theme)
- **NumPy** - Audio processing
- **SoundDevice** - Audio I/O
- **Wave** - Audio file handling

### ML Backends (Auto-detected)
1. **MLX-Audio** (preferred on Apple Silicon)
   - `mlx_audio.stt` module
   - Fastest performance
2. **MLX-CLI** (fallback)
   - `python -m mlx_qwen3_asr`
   - Subprocess-based
3. **PyTorch** (Intel Mac)
   - `qwen_asr` package
   - MPS acceleration on Apple Silicon

### C Implementation
- **Language:** C99
- **Build:** GCC with Make
- **Acceleration:** Apple Accelerate (macOS) / OpenBLAS (Linux)
- **Binary:** `assets/c-asr/qwen_asr`
- **Purpose:** Live streaming transcription

### Dependencies (Runtime)
```python
# Core (always required)
- sounddevice
- numpy

# Apple Silicon
- mlx-qwen3-asr
- mlx-audio

# Intel Mac
- qwen-asr
- torch

# Optional
- librosa  # For audio duration
```

## Build and Run Commands

### First-time Setup
```bash
./scripts/setup.command
```
- Creates Python 3.12+ virtual environment in `backend/venv`
- Installs platform-specific dependencies
- Detects Apple Silicon vs Intel Mac

### Launch Application
```bash
./scripts/launch.command
```
- Activates virtual environment
- Runs `python src/main.py`

### Run from Source (Development)
```bash
source backend/venv/bin/activate
python src/main.py
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

## Testing

### Run All Tests
```bash
# Run with Python unittest
python -m unittest discover tests/ -v

# Or individual test files
python tests/test_ui.py
python tests/test_live_streaming_final.py
```

### Run with pytest
```bash
pytest tests/ -v

# Specific test class
pytest tests/test_ui.py::TestColorConstants -v

# Specific test method
pytest tests/test_ui.py::TestColorConstants::test_all_colors_defined -v
```

### Test Categories

| Test File | Purpose | Key Tests |
|-----------|---------|-----------|
| `test_ui.py` | UI constants | Colors, breakpoints, sidebar behavior |
| `test_live_streaming_final.py` | Live streaming | Threading, RTF, memory leaks |
| `test_models.py` | ML models | Model loading, transcription accuracy |
| `test_memory_leaks.py` | Memory stability | Temp file cleanup, process cleanup |
| `test_integration.py` | End-to-end | Full workflows, error recovery |
| `test_error_handling.py` | Error handling | Edge cases, recovery |
| `test_file_io.py` | File operations | Save/load, formats |
| `test_macos_compat.py` | macOS specific | Permissions, paths |
| `test_recording_vad.py` | Audio VAD | Silence detection |
| `test_batch_mode.py` | Batch processing | Fast mode |
| `test_user_workflows.py` | User scenarios | Complete workflows |

### Test Reports
- `tests/TEST_REPORT.md` - Live streaming initial test results
- `tests/LIVE_STREAMING_FIX_REPORT.md` - Threading fix validation

## Code Organization

### Main Application (`src/app.py`)

#### Classes
1. **`QwenASRApp`** - Main application controller
   - UI setup and layout management
   - Recording control (start/stop)
   - Mode switching (live/batch)
   - Event handling

2. **`CollapsibleSidebar`** - Settings sidebar
   - Recording controls
   - Model/language selection
   - Silence duration settings
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

7. **`TranscriptionEngine`** - Batch transcription
   - Auto-detects backend (MLX/PyTorch)
   - Model caching
   - Progress callbacks

8. **`WaveformVisualizer`** - Audio level display
   - 40-bar history
   - Color-coded levels (green/yellow/red)

#### Key Constants (`src/constants.py`)
```python
APP_NAME = "Qwen3-ASR Pro"
VERSION = "3.1.1"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.05
MIN_WIDTH_COMPACT = 750  # px
MIN_WIDTH_MOBILE = 550   # px
COLORS = { ... }  # 17-color light theme
```

### Responsive Layout

| Mode | Width | Layout |
|------|-------|--------|
| Desktop | > 750px | Full sidebar (260px) |
| Compact | 550-750px | Collapsed sidebar (60px) |
| Mobile | < 550px | Bottom bar + slide-out panel |

### Recording Modes

#### Live Mode (🎓 Live Class)
```
Microphone → [5s chunks] → C binary (qwen_asr) → Live text + Raw file
```
- Always uses 0.6B model for stability
- Word-by-word output
- Raw audio saved to `~/Documents/Qwen3-ASR-Recordings/`

#### Fast Mode (⚡ Fast)
```
Microphone → [Save] → MLX/PyTorch → Text
```
- User-selected model (0.6B or 1.7B)
- Faster processing (lower RTF)
- Best for quick voice memos

## Development Conventions

### Code Style
- **Docstrings:** Triple-quote with description
- **Comments:** Inline for complex logic
- **Naming:** 
  - Classes: `PascalCase`
  - Methods/variables: `snake_case`
  - Constants: `UPPER_CASE`
- **Type hints:** Used for public methods
- **Line length:** ~100 characters

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

## Security Considerations

### Microphone Permissions
- Requires Terminal/iTerm microphone access
- Settings: System Preferences → Security & Privacy → Microphone

### File System
- Creates recordings in user's Documents folder
- Temp files cleaned up after processing
- No network access required

### Dependencies
- All dependencies from PyPI
- No external API keys required
- Local ML inference only

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

## Performance Targets

| Model | Mode | Target RTF | Notes |
|-------|------|------------|-------|
| 0.6B | Fast | ~0.02x | Batch processing |
| 1.7B | Fast | ~0.03x | Batch processing |
| 0.6B | Streaming | < 1.0x | Live mode |

*RTF (Real-Time Factor) < 1.0 means faster than real-time*

## Deployment

### Distribution
- Not packaged as .app bundle
- Run via `launch.command`
- Requires Python 3.12+ installed

### User Data
- Recordings: `~/Documents/Qwen3-ASR-Recordings/`
- Naming: `class_YYYYMMDD_HHMMSS.wav`

## References

- [Qwen3-ASR GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [mlx-audio GitHub](https://github.com/Blaizzy/mlx-audio)
- [MLX Framework](https://github.com/ml-explore/mlx)
- [C Implementation](https://github.com/antirez/qwen-asr)

---

**Last Updated:** 2026-02-28  
**Maintainer:** HY-D1
