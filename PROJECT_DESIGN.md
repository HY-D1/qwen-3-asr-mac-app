# Qwen3-ASR Pro - Project Design Document
## Complete System Architecture & Phased Implementation

---

## 🎯 Project Vision

A **web-based speech-to-text application** with:
- **Live audio recording** → **Transcription** → **AI refinement** → **Display results**
- **File upload** → **Transcription** → **AI refinement** → **Display results**
- **Free, local AI models** (Ollama/Qwen) - no API keys needed
- **Modern web UI** - no tkinter issues

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND (Web UI)                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────────────────┐ │
│  │ Live Record  │  │ File Upload  │  │           Display Area              │ │
│  │  - Record    │  │  - Drag/Drop │  │  ┌────────────┐  ┌─────────────┐   │ │
│  │  - Stop      │  │  - Browse    │  │  │ Raw Text   │  │ Refined Text│   │ │
│  │  - Status    │  │  - Progress  │  │  │            │  │             │   │ │
│  └──────┬───────┘  └──────┬───────┘  │  └────────────┘  └─────────────┘   │ │
│         │                  │           │  ┌──────────────────────────────┐  │ │
│         └──────────────────┴───────────│  │  AI Settings & Controls      │  │ │
│                                        │  │  - Model Select              │  │ │
│                                        │  │  - Reform Mode               │  │ │
│                                        │  └──────────────────────────────┘  │ │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          BACKEND (Python)                                    │
│                                                                              │
│  ┌────────────────────────┐    ┌──────────────────────────────────────────┐ │
│  │   Audio Processor      │    │        Transcription Engine               │ │
│  │  - Web Audio API       │───▶│  ┌──────────────┐    ┌────────────────┐  │ │
│  │  - File Handling       │    │  │  MLX (ASR)   │ or │  MLX-CLI       │  │ │
│  │  - Format Conversion   │    │  │  0.6B Model  │    │  (Fallback)    │  │ │
│  └────────────────────────┘    │  └──────────────┘    └────────────────┘  │ │
│                                └────────────────────┬───────────────────────┘ │
│                                                     │                        │
│                                                     ▼                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    AI Text Refinement Engine                           │ │
│  │                                                                       │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                    Backend Priority                              │  │ │
│  │  │  1. OLLAMA (qwen:1.8b/4b/7b)  ← FREE, LOCAL, OPEN SOURCE ✅      │  │ │
│  │  │  2. OpenAI API (gpt-3.5-turbo)  ← Cloud-based (optional)         │  │ │
│  │  │  3. Transformers (local)  ← Direct HuggingFace                   │  │ │
│  │  │  4. Rule-based  ← Fallback (always works)                        │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                       │ │
│  │  Reform Modes:                                                        │ │
│  │  • PUNCTUATE: Add punctuation & capitalization                        │ │
│  │  • SUMMARIZE: Create concise summary                                  │ │
│  │  • CLEAN: Remove filler words (um, uh, like)                          │ │
│  │  • KEY_POINTS: Extract bullet points                                  │ │
│  │  • FORMAT: Structure as meeting notes                                 │ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Phase 1: Core Infrastructure
**Goal:** Get basic transcription working with CLI
**Duration:** Day 1
**Deliverables:**

### 1.1 Environment Setup
- [ ] Python virtual environment (compatible with macOS 16)
- [ ] Install core dependencies: `mlx`, `mlx-qwen3-asr`, `numpy`, `sounddevice`
- [ ] Test transcription on sample audio file

### 1.2 CLI Transcription
- [ ] Command: `python cli.py transcribe audio.wav`
- [ ] Output: Raw text to terminal
- [ ] Support for: WAV, MP3, M4A

### 1.3 Deliverable
```bash
# Should work:
./scripts/launch_cli.command audio.wav
# Output: Transcribed text
```

---

## 📋 Phase 2: Web UI Foundation
**Goal:** Browser-based UI with file upload
**Duration:** Days 2-3
**Deliverables:**

### 2.1 Web Framework
- [ ] Install Gradio: `pip install gradio`
- [ ] Create `web_ui.py` with basic layout
- [ ] Test: Open browser at `http://localhost:7860`

### 2.2 File Upload Feature
- [ ] Drag & drop zone
- [ ] File browser button
- [ ] Progress indicator during upload
- [ ] Display uploaded file name

### 2.3 Display Results
- [ ] Left panel: Raw transcription output
- [ ] Right panel: (placeholder for Phase 3)
- [ ] Copy buttons for text areas

### 2.4 Deliverable
```bash
./scripts/launch_web.command
# Browser opens with:
# - File upload working
# - Raw transcription displayed
```

---

## 📋 Phase 3: AI Integration
**Goal:** Add free, local LLM for text refinement
**Duration:** Days 4-5
**Deliverables:**

### 3.1 Ollama Setup
- [ ] Install Ollama: `curl -fsSL https://ollama.com/install.sh | sh`
- [ ] Download Qwen model: `ollama pull qwen:1.8b`
- [ ] Test: `ollama run qwen:1.8b "Say hello"`

### 3.2 AI Backend Module
- [ ] Create `src/ai_backend.py`
- [ ] Implement Ollama API client
- [ ] Test prompt: "Add punctuation to: 'hello world'"

### 3.3 Reform Modes
- [ ] PUNCTUATE: `ai.process(text, "punctuate")`
- [ ] SUMMARIZE: `ai.process(text, "summarize")`
- [ ] CLEAN: `ai.process(text, "clean")`

### 3.4 Web UI Integration
- [ ] Add dropdown: Reform mode selection
- [ ] Add panel: Refined text display
- [ ] Pipeline: Upload → Transcribe → Refine → Display

### 3.5 Deliverable
```bash
./scripts/launch_web.command
# Can now:
# 1. Upload audio file
# 2. Select "Punctuate" mode
# 3. See raw text (left) and refined text (right)
# AI uses free local Qwen model
```

---

## 📋 Phase 4: Live Recording
**Goal:** Record audio from browser microphone
**Duration:** Days 6-7
**Deliverables:**

### 4.1 Web Audio Integration
- [ ] Add "Record" button to UI
- [ ] Use Gradio's microphone component
- [ ] Show recording status (Recording... / Stopped)

### 4.2 Recording Pipeline
- [ ] Record → Save to temp file
- [ ] Transcribe temp file
- [ ] Refine with AI
- [ ] Display both outputs

### 4.3 Deliverable
```bash
./scripts/launch_web.command
# Can now:
# 1. Click "Record" button
# 2. Speak into microphone
# 3. Click "Stop"
# 4. See transcription and refined text
```

---

## 📋 Phase 5: Polish & Advanced Features
**Goal:** Production-ready with options
**Duration:** Days 8-10
**Deliverables:**

### 5.1 Model Selection
- [ ] UI dropdown: Select AI model
  - qwen:1.8b (fast, 1GB)
  - qwen:4b (balanced, 2.5GB)
  - qwen:7b (quality, 4GB)
  - llama3.2:3b (alternative)
- [ ] Auto-detect available models

### 5.2 Language Support
- [ ] UI dropdown: Language selection
  - Auto-detect
  - English, Chinese, Japanese, etc.
- [ ] Pass language to transcription

### 5.3 Export Features
- [ ] Download as TXT button
- [ ] Download as Markdown button
- [ ] Copy buttons for both panels

### 5.4 Session History
- [ ] Save recent transcriptions
- [ ] List previous sessions
- [ ] Re-open previous results

### 5.5 Deliverable
```bash
./scripts/launch_web.command
# Full featured:
# - Live recording
# - File upload  
# - Model selection
# - Multiple reform modes
# - Export options
# - Session history
```

---

## 🎯 Current Status (As of Today)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Core Infra | ✅ Done | CLI transcription works |
| Phase 2: Web UI Basic | ✅ Done | File upload + display |
| Phase 3: AI Integration | ✅ Done | Ollama/Qwen added |
| Phase 4: Live Recording | ✅ Done | Gradio mic component |
| Phase 5: Polish | 🔄 Partial | Model selection needed |

---

## 🚀 Quick Start (Current Working Version)

### Step 1: Install Ollama
```bash
# Install Ollama (one-time)
curl -fsSL https://ollama.com/install.sh | sh

# Download Qwen model (~1GB)
ollama pull qwen:1.8b

# Start Ollama server
ollama serve &
```

### Step 2: Run Web App
```bash
# Install Gradio if not present
pip install gradio

# Launch web UI
./scripts/launch_web.command

# Open browser: http://localhost:7860
```

### Step 3: Use the App
1. **Record Audio:**
   - Click "🎤 Record" tab
   - Click microphone button
   - Speak, then click "Stop"
   - Click "Transcribe"

2. **Upload File:**
   - Click "📁 Upload" tab
   - Drag audio file or click to browse
   - Select reform mode (Punctuate/Summarize/Clean)
   - Click "Transcribe & Reform"

3. **View Results:**
   - Left: Raw transcription
   - Right: AI-refined text
   - Click copy buttons to copy text

---

## 📁 Project Structure

```
qwen-3-asr-mac-app-main/
├── src/
│   ├── __init__.py          # Package init
│   ├── constants.py         # Configuration
│   ├── simple_llm.py        # AI backends (Ollama, OpenAI, etc.)
│   ├── ollama_backend.py    # Ollama-specific implementation
│   └── app.py               # Legacy tkinter app (not used)
├── scripts/
│   ├── launch_web.command   # ✅ USE THIS - Web UI launcher
│   ├── setup_ollama.command # Install Ollama + models
│   ├── launch_cli.command   # CLI version
│   └── ...
├── web_ui.py                # ✅ Main Web Application
├── cli_app.py               # CLI application
├── PROJECT_DESIGN.md        # This document
└── backend/
    └── venv_system/         # Virtual environment
```

---

## 🎨 UI Mockup

```
┌─────────────────────────────────────────────────────────────────┐
│ 🎙️ Qwen3-ASR Pro  v3.3.0                              [Help]   │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────┐  ┌─────────────────────────────────────┐│
│ │  🎤 Record Audio    │  │  AI Engine: ollama-qwen:1.8b        ││
│ │                     │  │  Status: Ready                      ││
│ │  [🔴 Record]        │  ├─────────────────────────────────────┤│
│ │  [⏹ Stop]           │  │  Language: [Auto ▼]                 ││
│ │                     │  │  AI Mode:  [Punctuate ▼]            ││
│ │  Recording: 00:00   │  │  Model:    [qwen:1.8b ▼]            ││
│ └─────────────────────┘  └─────────────────────────────────────┘│
│ ┌─────────────────────┐  ┌─────────────────────────────────────┐│
│ │  📁 Upload File     │  │                                     ││
│ │                     │  │  📝 Raw Transcript         [Copy]   ││
│ │  Drag & drop or     │  │  ─────────────────────────────      ││
│ │  click to browse    │  │  um so like we need to discuss      ││
│ │                     │  │  the project timeline uh...         ││
│ │  [Select File]      │  │                                     ││
│ │                     │  │  ✨ AI-Reformed Text       [Copy]   ││
│ └─────────────────────┘  │  ─────────────────────────────      ││
│                          │  So, we need to discuss the         ││
│  [🚀 Transcribe]         │  project timeline. First, we...     ││
│                          │                                     ││
│                          │  [💾 Save as TXT] [📋 Markdown]     ││
│                          └─────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│  ℹ️  Tips: Upload WAV/MP3/M4A or record. AI runs locally.      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Success Criteria

### Minimum Viable Product (MVP)
- [x] File upload → Transcription → Display
- [x] Basic AI refinement (rule-based)
- [x] Web UI (no tkinter issues)

### Full Product
- [x] Live recording from browser
- [x] Free local AI (Ollama/Qwen)
- [x] Multiple reform modes
- [ ] Model selection UI
- [ ] Export to file
- [ ] Session history

---

## 🔄 Next Steps

1. **Immediate:** Run `./scripts/launch_web.command` and test
2. **If Ollama not setup:** Run `./scripts/setup_ollama.command`
3. **To use better models:** Run `ollama pull qwen:4b`
4. **For development:** Edit `web_ui.py` and restart

---

## 📞 Troubleshooting

| Issue | Solution |
|-------|----------|
| "Ollama not found" | Run `./scripts/setup_ollama.command` |
| "Model not found" | Run `ollama pull qwen:1.8b` |
| "Port already in use" | Kill existing: `pkill -f "python web_ui.py"` |
| "Gradio not found" | Run `pip install gradio` |

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-28  
**Status:** Phase 4 Complete, Phase 5 In Progress
