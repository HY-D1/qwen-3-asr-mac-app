# Qwen3-ASR Pro - Quick Start Guide

## 🚀 Getting Started (3 Easy Steps)

### Step 1: Start Ollama (for AI text reforming)

**Option A: Use the launcher (recommended)**
```bash
./scripts/launch_web.command
```
This will start Ollama automatically in the background.

**Option B: Manual start**
```bash
# Terminal 1: Start Ollama (keep running)
ollama serve

# Terminal 2: Run the app
./scripts/launch_web.command
```

### Step 2: Choose Your Interface

| Interface | Best For | Command |
|-----------|----------|---------|
| **Web UI** | Recording, file upload, visual interface | `./scripts/launch_web.command` |
| **CLI** | Quick transcription, batch processing | `./scripts/launch_cli.command` or `python cli_app.py` |

### Step 3: Start Transcribing

---

## 🌐 Web UI (Browser-based)

### Launch
```bash
./scripts/launch_web.command
```

The app will open in your browser at `http://localhost:7860` (or auto-detected port).

### Features

#### Tab 1: Upload File
1. Click **"Upload File"** tab
2. Drag & drop an audio file OR click to browse
3. Select language (or leave as "Auto-detect")
4. Choose AI text reforming mode:
   - **None** - Raw transcription only
   - **Punctuate** - Add proper punctuation
   - **Summarize** - Create brief summary
   - **Clean up** - Remove filler words (um, uh)
   - **Key points** - Extract main points
5. Click **"Transcribe & Reform"**
6. Copy results with the **Copy** buttons

#### Tab 2: Record Audio
1. Click **"Record Audio"** tab
2. Click the microphone button to start recording
3. Speak clearly
4. Click stop when done
5. Select reform mode and click **"Transcribe & Reform"**

### Supported Audio Formats
- WAV (16kHz, mono preferred)
- MP3, M4A, OGG (auto-converted)

---

## 🖥️ CLI (Command Line)

### Quick Transcription
```bash
# Transcribe a file
python cli_app.py recording.wav

# Transcribe with AI reforming
python cli_app.py recording.wav --reform punctuate

# Save to output file
python cli_app.py recording.wav -o result.txt

# Specify language
python cli_app.py recording.wav --language zh
```

### Interactive Mode
```bash
python cli_app.py --interactive
# or
python cli_app.py -i
```

**Available Commands:**
```
> help                      # Show all commands
> status                    # Check system status
> transcribe file.wav       # Transcribe audio file
> reform "text" punctuate   # Reform text
> process file.wav summarize # Transcribe + reform
> quit                      # Exit
```

**Example Interactive Session:**
```
> status
🤖 LLM Backend: ollama-qwen:1.8b
✅ Status: Ready

> transcribe assets/c-asr/samples/jfk.wav
🎤 Transcribing: assets/c-asr/samples/jfk.wav
Transcript: And so my fellow Americans ask not what your country can do for you...

> reform "hello world this is a test" punctuate
🤖 Reforming text (mode: punctuate)...
✅ Done in 0.3s (backend: ollama-qwen:1.8b)
Hello, world! This is a test.

> quit
👋 Goodbye!
```

---

## 📁 Where Are Recordings Saved?

### Live Mode Recordings
Auto-saved to: `~/Documents/Qwen3-ASR-Recordings/`

File naming: `live_YYYYMMDD_HHMMSS.wav`

Example: `live_20260301_143022.wav`

### Uploaded Files
Not saved - processed directly from upload location.

---

## 🛠️ Managing Ollama

### Check Ollama Status
```bash
# Check if running
curl http://localhost:11434/api/tags

# View available models
ollama list
```

### Pull a New Model
```bash
# Pull Qwen 1.8B (recommended, fast)
ollama pull qwen:1.8b

# Pull larger model (better quality, slower)
ollama pull qwen:7b

# Pull Qwen 2.5 (latest)
ollama pull qwen2.5:1.5b-instruct
```

### Stop Ollama
```bash
./scripts/stop_ollama.command
```

### View Ollama Logs
```bash
./scripts/view_ollama_logs.command
# or
tail -f ~/.ollama/logs/server.log
```

---

## 🔧 Troubleshooting

### "No transcription backend available"
**Solution:** The C binary is being used. Make sure models are downloaded:
```bash
cd assets/c-asr
ls qwen3-asr-0.6b/  # Should show model files
```

### "Connection refused" (Ollama not running)
**Solution:**
```bash
# Start Ollama
ollama serve &

# Or use the launcher which auto-starts it
./scripts/launch_web.command
```

### Port 7860 already in use
**Solution:** The app auto-detects free ports now. Just run:
```bash
./scripts/launch_web.command
```
It will find the next available port (7861, 7862, etc.).

### Audio recording not working
**Solution:** Check microphone permissions:
1. System Preferences → Security & Privacy → Microphone
2. Ensure Terminal (or iTerm) is checked

### Slow transcription
**Factors:**
- Larger models (1.7B) are slower than small (0.6B)
- Longer audio takes more time
- First transcription loads the model (subsequent ones are faster)

**Typical speeds:**
- C binary: ~0.7x real-time (3s audio → ~2s processing)
- MLX (if installed): ~0.1x real-time (3s audio → ~0.3s processing)

---

## 📊 Performance Tips

1. **Use C binary backend** (default) - Works on all Macs, no dependencies
2. **Use 0.6B model** - Faster than 1.7B, good accuracy
3. **Keep audio under 60 seconds** - For fastest processing
4. **Use WAV format** - Avoids conversion overhead
5. **Close other apps** - Frees up CPU/RAM for transcription

---

## 🎯 Use Cases

| Use Case | Recommended Mode | Settings |
|----------|------------------|----------|
| **Live lecture notes** | Web UI → Record | Auto-detect language, Punctuate |
| **Meeting transcription** | Web UI → Upload | Auto-detect, Key points |
| **Quick voice memo** | CLI | `cli_app.py memo.wav --reform clean` |
| **Interview transcription** | CLI | `cli_app.py interview.wav --reform summarize` |
| **Podcast to text** | Web UI → Upload | Auto-detect, Summarize |

---

## 📖 Additional Resources

- **Detailed Documentation:** `README.md`
- **Agent Guide:** `AGENTS.md`
- **Test Reports:** `tests/TRANSCRIPTION_TEST_REPORT.md`
- **Design Document:** `PROJECT_DESIGN.md`

---

## 🆘 Need Help?

1. Check system status: `python cli_app.py -i` → type `status`
2. View logs: `tail -f ~/.ollama/logs/server.log`
3. Restart everything:
   ```bash
   ./scripts/stop_ollama.command
   ./scripts/kill_servers.command
   ./scripts/launch_web.command
   ```

---

# 🚀 Quick Reference Card

## One-Line Commands

```bash
# Start Web UI
./scripts/launch_web.command

# Start CLI Interactive Mode
python cli_app.py -i

# Quick Transcribe
python cli_app.py audio.wav

# Transcribe + Reform
python cli_app.py audio.wav --reform punctuate

# Stop All Servers
./scripts/stop_ollama.command && ./scripts/kill_servers.command
```

---

## 🎛️ CLI Interactive Commands

```
help           Show all commands
status         Check system status
transcribe     Transcribe audio file
reform         Reform text with AI
process        Transcribe + reform in one step
quit           Exit
```

---

## 📝 Reform Modes

| Mode | What It Does |
|------|--------------|
| `none` | Raw transcription |
| `punctuate` | Add punctuation & capitalization |
| `summarize` | Brief summary |
| `clean` | Remove filler words (um, uh) |
| `key_points` | Extract bullet points |

---

## 📁 Important Paths

| Purpose | Location |
|---------|----------|
| Recordings | `~/Documents/Qwen3-ASR-Recordings/` |
| Ollama Logs | `~/.ollama/logs/server.log` |
| Models | `assets/c-asr/qwen3-asr-*/` |
| Sample Audio | `assets/c-asr/samples/` |

---

## 🔧 Common Issues Quick Fix

| Problem | Fix |
|---------|-----|
| Port in use | Auto-detected, just relaunch |
| Ollama not running | `./scripts/launch_web.command` auto-starts it |
| No microphone | System Preferences → Privacy → Microphone → Enable Terminal |
| Slow transcription | Use 0.6B model, keep audio under 60s |

---

## ⚡ Performance

| Backend | Speed | Quality |
|---------|-------|---------|
| C Binary | ~0.7x RT | Good |
| MLX (if installed) | ~0.1x RT | Excellent |

RT = Real-time (3s audio → 2s processing at 0.7x RT)

---

## 🐞 Debug Commands

```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check logs
tail -f ~/.ollama/logs/server.log

# List models
ollama list

# Pull new model
ollama pull qwen:1.8b
```

---

**Version:** 3.3.0  
**Last Updated:** 2026-03-01
