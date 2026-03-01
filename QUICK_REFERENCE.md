# Qwen3-ASR Pro - Quick Reference Card

## 🚀 One-Line Commands

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

## 🔧 Common Issues

| Problem | Fix |
|---------|-----|
| Port in use | Auto-detected, just relaunch |
| Ollama not running | `./scripts/launch_web.command` auto-starts it |
| No microphone | System Preferences → Privacy → Microphone → Enable Terminal |
| Slow transcription | Use 0.6B model, keep audio under 60s |

---

## 🌐 Web UI Ports

Default: `http://localhost:7860`  
If busy: `7861`, `7862`, etc. (auto-detected)

---

## 📊 Supported Formats

- **Best:** WAV (16kHz, mono)
- **Supported:** MP3, M4A, OGG, FLAC
- **Auto-converted:** All formats → WAV internally

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

*Print this page and keep it handy!*
