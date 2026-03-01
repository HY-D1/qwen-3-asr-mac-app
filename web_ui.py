#!/usr/bin/env python3
"""
Qwen3-ASR Pro - Web UI
Gradio-based web interface (works in browser, no tkinter needed)
"""

import os
import sys
import subprocess
import warnings
from pathlib import Path

# Suppress HTTP request warnings from Gradio/Uvicorn
import logging

# Filter out "Invalid HTTP request received" warnings
class HTTPWarningFilter(logging.Filter):
    def filter(self, record):
        # Filter out invalid HTTP request messages
        if "Invalid HTTP request received" in str(record.getMessage()):
            return False
        return True

# Apply filters to uvicorn loggers
uvicorn_logger = logging.getLogger("uvicorn.error")
uvicorn_logger.setLevel(logging.WARNING)
uvicorn_logger.addFilter(HTTPWarningFilter())

# Also filter gradio warnings
logging.getLogger("gradio").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import gradio as gr
except ImportError:
    print("Installing gradio...")
    os.system(f"{sys.executable} -m pip install gradio -q")
    import gradio as gr

from simple_llm import SimpleLLM

# Global LLM instance - will try Ollama first
print("🤖 Initializing LLM backend...")
llm = SimpleLLM(ollama_model="qwen:1.8b")
print(f"   Using: {llm.backend_name}")


def transcribe_with_c_binary(audio_file, model="small"):
    """Transcribe using the C binary implementation"""
    base_dir = os.path.dirname(__file__)
    c_binary = os.path.join(base_dir, "assets", "c-asr", "qwen_asr")
    
    # Verify binary exists
    if not os.path.exists(c_binary):
        return None, f"C binary not found at {c_binary}"
    
    # Map model size to C binary model directory
    model_map = {
        "0.6b": "qwen3-asr-0.6b",
        "1.7b": "qwen3-asr-1.7b",
        "small": "qwen3-asr-0.6b", 
        "large": "qwen3-asr-1.7b"
    }
    model_dir_name = model_map.get(model, "qwen3-asr-0.6b")
    model_dir = os.path.join(base_dir, "assets", "c-asr", model_dir_name)
    
    # Verify model directory exists
    if not os.path.exists(model_dir):
        return None, f"Model directory not found: {model_dir}"
    
    # Build command: qwen_asr -d <model_dir> -i <audio_file> --silent
    cmd = [c_binary, "-d", model_dir, "-i", audio_file, "--silent"]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            return result.stdout.strip(), "C-Binary"
        else:
            return None, f"C binary error (code {result.returncode}): {result.stderr}"
    except subprocess.TimeoutExpired:
        return None, "Transcription timeout (300s exceeded)"
    except Exception as e:
        return None, f"C binary failed: {str(e)}"


def transcribe_audio(audio_file, language="auto", model="0.6b"):
    """Transcribe audio file using available backends (C binary preferred)"""
    if audio_file is None:
        return "No audio file provided", "Error"
    
    if not os.path.exists(audio_file):
        return f"File not found: {audio_file}", "Error"
    
    # Try C binary first (most reliable)
    transcript, backend_info = transcribe_with_c_binary(audio_file, model=model)
    if transcript:
        return transcript, backend_info
    
    # Fall back to Python backends
    last_error = backend_info  # Store C binary error message
    
    try:
        # Try MLX first
        try:
            import mlx_audio.stt as mlx_stt
            mlx_model = mlx_stt.load("Qwen/Qwen3-ASR-0.6B")  # Use smaller model for speed
            result = mlx_model.generate(audio_file, language=None if language == "auto" else language)
            transcript = result.text if hasattr(result, 'text') else str(result)
            return transcript, "MLX (Apple Silicon)"
        except ImportError:
            pass
        
        # Fall back to CLI
        cmd = [sys.executable, '-m', 'mlx_qwen3_asr', audio_file, '--stdout-only']
        if language != "auto":
            cmd.extend(['--language', language])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return result.stdout.strip(), "MLX-CLI"
        else:
            last_error = f"MLX-CLI error: {result.stderr}"
            
    except Exception as e:
        last_error = f"Python backends failed: {str(e)}"
    
    # All backends failed
    return f"No transcription backend available. Last error: {last_error}", "Error"


def reform_text(text, mode):
    """Reform text using LLM"""
    if not text or not text.strip():
        return "No text to reform"
    
    if not llm.is_available():
        return text + "\n\n[Note: LLM not available, showing original text]"
    
    try:
        result = llm.process(text, mode)
        return result
    except Exception as e:
        return f"Error reforming text: {str(e)}"


def process_audio(audio_file, language, reform_mode, model="0.6b"):
    """Full pipeline: transcribe + reform"""
    if audio_file is None:
        return "Please upload an audio file", "", ""
    
    # Step 1: Transcribe
    transcript, backend = transcribe_audio(audio_file, language, model)
    
    if transcript.startswith("Error:") or backend == "Error":
        return transcript, "", ""
    
    # Step 2: Reform if requested
    if reform_mode != "none":
        reformed = reform_text(transcript, reform_mode)
    else:
        reformed = transcript
    
    # Get LLM info with model used
    llm_info = f"Backend: {backend} | Model: {model} | LLM: {llm.backend_name}"
    
    return transcript, reformed, llm_info


def record_and_transcribe(audio, language, reform_mode, model="0.6b"):
    """Handle recorded audio"""
    return process_audio(audio, language, reform_mode, model)


# Create Gradio interface
with gr.Blocks(title="Qwen3-ASR Pro", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Qwen3-ASR Pro - Web UI
    ### Speech-to-Text with AI Text Refinement
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            
            model = gr.Dropdown(
                choices=[
                    ("⚡ Fast (0.6B) - Quick results", "0.6b"),
                    ("🎯 Accurate (1.7B) - Better quality", "1.7b")
                ],
                value="0.6b",
                label="Transcription Model"
            )
            
            language = gr.Dropdown(
                choices=["auto", "en", "zh", "ja", "ko", "es", "fr", "de"],
                value="auto",
                label="Language"
            )
            
            reform_mode = gr.Dropdown(
                choices=[
                    ("No reforming", "none"),
                    ("Punctuate", "punctuate"),
                    ("Summarize", "summarize"),
                    ("Clean up", "clean"),
                    ("Key points", "key_points")
                ],
                value="punctuate",
                label="AI Text Reforming"
            )
            
            gr.Markdown(f"### Status")
            llm_status = gr.Textbox(
                value=f"Backend: {llm.backend_name}\nStatus: {'Ready' if llm.is_available() else 'Not available'}",
                label="AI Engine",
                interactive=False,
                lines=2
            )
            
            gr.Markdown("""
            **Model Info:**
            - **0.6B**: Fast (~0.7x real-time), good accuracy
            - **1.7B**: Slower, better accuracy for difficult audio
            
            **AI Backends:**
            - **C-Binary**: Fast local transcription
            - **MLX**: Apple Silicon optimized
            """)
        
        with gr.Column(scale=2):
            with gr.Tab("📁 Upload File"):
                audio_input = gr.Audio(
                    type="filepath",
                    label="Upload Audio (WAV, MP3, M4A, etc.)"
                )
                upload_btn = gr.Button("🚀 Transcribe & Reform", variant="primary")
            
            with gr.Tab("🎤 Record Audio"):
                gr.Markdown("Click 'Record' to start recording from microphone")
                record_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record Audio"
                )
                record_btn = gr.Button("🚀 Transcribe Recording", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Raw Transcript")
            raw_output = gr.Textbox(
                label="",
                lines=10,
                show_copy_button=True
            )
            copy_raw_btn = gr.Button("📋 Copy Raw Text", size="sm")
        
        with gr.Column():
            gr.Markdown("### ✨ AI-Reformed Text")
            reformed_output = gr.Textbox(
                label="",
                lines=10,
                show_copy_button=True
            )
            copy_reformed_btn = gr.Button("📋 Copy Reformed Text", size="sm")
    
    info_output = gr.Textbox(label="Info", interactive=False)
    
    # Event handlers for transcription
    upload_btn.click(
        fn=process_audio,
        inputs=[audio_input, language, reform_mode, model],
        outputs=[raw_output, reformed_output, info_output]
    )
    
    record_btn.click(
        fn=record_and_transcribe,
        inputs=[record_input, language, reform_mode, model],
        outputs=[raw_output, reformed_output, info_output]
    )
    
    # JavaScript-based copy handlers (more reliable than built-in)
    copy_raw_btn.click(
        fn=None,
        inputs=[raw_output],
        outputs=[],
        js="""
        async (text) => {
            if (!text) {
                alert("No text to copy!");
                return;
            }
            try {
                await navigator.clipboard.writeText(text);
                // Visual feedback
                const btn = document.activeElement;
                const originalText = btn.innerText;
                btn.innerText = "✅ Copied!";
                setTimeout(() => btn.innerText = originalText, 2000);
            } catch (err) {
                // Fallback for older browsers
                const textArea = document.createElement("textarea");
                textArea.value = text;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand("copy");
                document.body.removeChild(textArea);
                const btn = document.activeElement;
                const originalText = btn.innerText;
                btn.innerText = "✅ Copied!";
                setTimeout(() => btn.innerText = originalText, 2000);
            }
        }
        """
    )
    
    copy_reformed_btn.click(
        fn=None,
        inputs=[reformed_output],
        outputs=[],
        js="""
        async (text) => {
            if (!text) {
                alert("No text to copy!");
                return;
            }
            try {
                await navigator.clipboard.writeText(text);
                // Visual feedback
                const btn = document.activeElement;
                const originalText = btn.innerText;
                btn.innerText = "✅ Copied!";
                setTimeout(() => btn.innerText = originalText, 2000);
            } catch (err) {
                // Fallback for older browsers
                const textArea = document.createElement("textarea");
                textArea.value = text;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand("copy");
                document.body.removeChild(textArea);
                const btn = document.activeElement;
                const originalText = btn.innerText;
                btn.innerText = "✅ Copied!";
                setTimeout(() => btn.innerText = originalText, 2000);
            }
        }
        """
    )
    
    gr.Markdown("""
    ---
    ### 💡 Tips
    - Use **Punctuate** mode to add proper punctuation and capitalization
    - Use **Summarize** mode to create a concise summary
    - Use **Clean up** mode to remove filler words (um, uh, like)
    - Click **📋 Copy** buttons below each text box to copy results
    - Audio is processed locally - no data leaves your computer
    """)

if __name__ == "__main__":
    import os
    import socket
    
    def find_free_port(start=7860, max_port=7870):
        """Find a free port starting from start port"""
        for port in range(start, max_port + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
        return 0  # Let Gradio pick a random free port
    
    # Get port from environment, auto-detect, or use default
    port_env = os.environ.get("GRADIO_SERVER_PORT", "")
    if port_env:
        port = int(port_env)
    else:
        port = find_free_port()
    
    print("🚀 Starting Qwen3-ASR Pro Web UI...")
    print(f"📱 Open your browser and go to: http://localhost:{port}")
    print("")
    
    # Configure upload limits and queue settings to prevent HTTP errors
    demo.queue(
        max_size=20,  # Limit concurrent requests
        default_concurrency_limit=1  # Process one at a time
    )
    
    demo.launch(
        server_name="0.0.0.0",  # Allow access from other devices
        server_port=port,
        share=False,  # Set to True to create a public link
        show_error=True,
        quiet=False
    )
