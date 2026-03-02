#!/usr/bin/env python3
"""
Qwen3-ASR Pro - Web UI
Gradio-based web interface with real-time transcription support
"""

import os
import sys
import subprocess
import warnings
import time
import threading
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

# Global state for recording status
recording_state = {
    "is_recording": False,
    "start_time": None,
    "live_text": "",
}


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
    
    # Build command: qwen_asr -d <model_dir> -i <audio_file>
    # Note: Don't use --silent as it suppresses output
    cmd = [c_binary, "-d", model_dir, "-i", audio_file]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            # Extract transcript from stdout
            # The C binary outputs to stdout, filter out debug messages
            output_lines = result.stdout.strip().split('\n')
            # Filter out debug/info lines (lines with ":" usually are debug)
            transcript_lines = []
            for line in output_lines:
                line = line.strip()
                # Skip debug/info lines
                if line and not line.startswith(('Loading', 'Detected:', 'Inference:', 'Audio:', 'Model loaded')):
                    transcript_lines.append(line)
            
            transcript = ' '.join(transcript_lines)
            
            # If empty transcript but no error, might be no speech detected
            if not transcript:
                # Check stderr for clues
                if "0 text tokens" in result.stderr:
                    return "[No speech detected in audio]", "C-Binary"
            
            return transcript if transcript else result.stdout.strip(), "C-Binary"
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
    
    errors = []
    
    # Try C binary first (most reliable)
    print(f"🎙️ Trying C binary backend...")
    transcript, backend_info = transcribe_with_c_binary(audio_file, model=model)
    if transcript:
        return transcript, backend_info
    else:
        errors.append(f"C-Binary: {backend_info}")
        print(f"   ⚠️ C binary failed: {backend_info}")
    
    # Try MLX (Apple Silicon only)
    print(f"🎙️ Trying MLX backend...")
    try:
        import mlx_audio.stt as mlx_stt
        mlx_model = mlx_stt.load("Qwen/Qwen3-ASR-0.6B")
        result = mlx_model.generate(audio_file, language=None if language == "auto" else language)
        transcript = result.text if hasattr(result, 'text') else str(result)
        return transcript, "MLX (Apple Silicon)"
    except ImportError:
        errors.append("MLX: mlx_audio not installed (Apple Silicon only)")
        print(f"   ⚠️ MLX not available (Apple Silicon only)")
    except Exception as e:
        errors.append(f"MLX: {str(e)}")
        print(f"   ⚠️ MLX failed: {e}")
    
    # Try PyTorch/qwen-asr (Intel Mac or any platform)
    print(f"🎙️ Trying PyTorch backend...")
    try:
        import torch
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"   Using device: {device}")
        
        # Try to use local model files first
        local_model_path = os.path.join(base_dir, "assets", "c-asr", "qwen3-asr-0.6b")
        
        if os.path.exists(os.path.join(local_model_path, "model.safetensors")):
            print(f"   Loading local model from: {local_model_path}")
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
            
            processor = AutoProcessor.from_pretrained(local_model_path, local_files_only=True)
            torch_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                local_model_path,
                local_files_only=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            torch_model.to(device)
            
            pipe = pipeline(
                "automatic-speech-recognition",
                model=torch_model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                torch_dtype=torch.float32,
                device=device,
            )
            
            result = pipe(audio_file)
            return result["text"], "PyTorch (Local)"
        else:
            # Fall back to HuggingFace Hub
            print(f"   Local model not found, trying HuggingFace Hub...")
            from transformers import pipeline
            
            pipe = pipeline(
                "automatic-speech-recognition",
                model="Qwen/Qwen3-ASR-0.6B",
                torch_dtype=torch.float32,
                device=device,
            )
            
            result = pipe(audio_file)
            return result["text"], "PyTorch (HF Hub)"
            
    except ImportError as e:
        errors.append(f"PyTorch: {str(e)}")
        print(f"   ⚠️ PyTorch backend not available: {e}")
    except Exception as e:
        errors.append(f"PyTorch: {str(e)}")
        print(f"   ⚠️ PyTorch failed: {e}")
    
    # Try MLX-CLI as last resort
    print(f"🎙️ Trying MLX-CLI backend...")
    try:
        # First check if module exists
        import importlib.util
        spec = importlib.util.find_spec("mlx_qwen3_asr")
        if spec is None:
            raise ImportError("mlx_qwen3_asr module not found")
        
        cmd = [sys.executable, '-m', 'mlx_qwen3_asr', audio_file, '--stdout-only']
        if language != "auto":
            cmd.extend(['--language', language])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return result.stdout.strip(), "MLX-CLI"
        else:
            errors.append(f"MLX-CLI: {result.stderr}")
            print(f"   ⚠️ MLX-CLI failed: {result.stderr[:100]}")
    except ImportError as e:
        errors.append(f"MLX-CLI: {str(e)}")
        print(f"   ⚠️ MLX-CLI not available")
    except Exception as e:
        errors.append(f"MLX-CLI: {str(e)}")
        print(f"   ⚠️ MLX-CLI error: {e}")
    
    # All backends failed
    error_summary = " | ".join(errors)
    print(f"\n❌ All backends failed: {error_summary}")
    return f"No transcription backend available. Errors: {error_summary}", "Error"


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
    """Handle recorded audio (batch mode)"""
    return process_audio(audio, language, reform_mode, model)


def simulate_streaming_transcript(full_transcript, chunk_size=3):
    """
    Simulate real-time streaming by splitting transcript into chunks.
    Returns a list of progressively building transcript strings.
    """
    words = full_transcript.split()
    if len(words) <= chunk_size:
        # If transcript is short, just return it with some delays
        return [
            "🎙️ Processing...",
            f"📝 {full_transcript[:len(full_transcript)//3]}...",
            f"📝 {full_transcript[:2*len(full_transcript)//3]}...",
            f"📝 {full_transcript}"
        ]
    
    chunks = []
    # Start with processing message
    chunks.append("🎙️ Initializing transcription...")
    chunks.append("🎙️ Processing audio chunks...")
    
    # Build up word by word
    for i in range(chunk_size, len(words) + chunk_size, chunk_size):
        partial = " ".join(words[:min(i, len(words))])
        chunks.append(f"📝 {partial}")
    
    return chunks


def stream_record_and_transcribe(audio, language, reform_mode, model, enable_realtime, progress=gr.Progress()):
    """
    Stream transcription results as they come in.
    This is a generator that yields intermediate results for real-time display.
    """
    if audio is None:
        yield {
            live_output: "⚠️ No audio recorded. Please record audio first.",
            raw_output: "",
            reformed_output: "",
            info_output: "Error: No audio provided"
        }
        return
    
    if not enable_realtime:
        # Non-streaming mode - just do regular processing
        transcript, reformed, info = process_audio(audio, language, reform_mode, model)
        yield {
            live_output: "✅ Transcription complete (real-time mode disabled)",
            raw_output: transcript,
            reformed_output: reformed,
            info_output: info
        }
        return
    
    # Real-time streaming mode
    yield {
        live_output: "🎙️ Starting transcription...",
        raw_output: "",
        reformed_output: "",
        info_output: "Status: Processing audio..."
    }
    
    # Small delay to show the initial message
    time.sleep(0.3)
    
    yield {
        live_output: "🔄 Transcribing audio...",
        raw_output: "",
        reformed_output: "",
        info_output: "Status: Running ASR model..."
    }
    
    # Perform the actual transcription
    try:
        transcript, backend = transcribe_audio(audio, language, model)
        
        if transcript.startswith("Error:") or backend == "Error":
            yield {
                live_output: f"❌ {transcript}",
                raw_output: transcript,
                reformed_output: "",
                info_output: f"Backend: {backend} | Error occurred"
            }
            return
        
        # Simulate streaming by showing chunks
        chunks = simulate_streaming_transcript(transcript)
        
        for i, chunk in enumerate(chunks[:-1]):  # All except final
            progress_percent = min((i + 1) / len(chunks) * 100, 95)
            yield {
                live_output: chunk,
                raw_output: chunk.replace("📝 ", "").replace("🎙️ ", ""),
                reformed_output: "",
                info_output: f"Status: Transcribing... {progress_percent:.0f}% | Backend: {backend}"
            }
            # Small delay between chunks for visual effect
            time.sleep(0.15)
        
        # Final chunk - complete transcript
        final_chunk = chunks[-1].replace("📝 ", "")
        
        yield {
            live_output: f"✅ Transcription complete!\n\n{final_chunk}",
            raw_output: transcript,
            reformed_output: "Reforming text..." if reform_mode != "none" else "",
            info_output: f"Status: Reforming text... | Backend: {backend} | Model: {model}"
        }
        
        # Apply text reformation if requested
        if reform_mode != "none":
            time.sleep(0.2)  # Small delay to show "Reforming..." message
            reformed = reform_text(transcript, reform_mode)
            
            # Simulate streaming for reformed text too
            reformed_chunks = simulate_streaming_transcript(reformed, chunk_size=5)
            for i, r_chunk in enumerate(reformed_chunks[:-1]):
                yield {
                    live_output: f"✅ Transcription complete!\n\n{final_chunk}",
                    raw_output: transcript,
                    reformed_output: r_chunk.replace("📝 ", "").replace("🎙️ ", ""),
                    info_output: f"Status: Reforming... {min((i+1)/len(reformed_chunks)*100, 95):.0f}% | Backend: {backend} | Model: {model}"
                }
                time.sleep(0.1)
            
            final_reformed = reformed_chunks[-1].replace("📝 ", "") if reformed_chunks else reformed
            
            yield {
                live_output: f"✅ Transcription complete!\n\n{final_chunk}",
                raw_output: transcript,
                reformed_output: reformed,
                info_output: f"✅ Complete | Backend: {backend} | Model: {model} | LLM: {llm.backend_name}"
            }
        else:
            yield {
                live_output: f"✅ Transcription complete!\n\n{final_chunk}",
                raw_output: transcript,
                reformed_output: "(No reformation applied)",
                info_output: f"✅ Complete | Backend: {backend} | Model: {model} | LLM: Not used"
            }
            
    except Exception as e:
        yield {
            live_output: f"❌ Error: {str(e)}",
            raw_output: f"Error: {str(e)}",
            reformed_output: "",
            info_output: f"Error during transcription: {str(e)}"
        }


def on_record_start():
    """Called when recording starts"""
    recording_state["is_recording"] = True
    recording_state["start_time"] = time.time()
    recording_state["live_text"] = "🎙️ Recording in progress..."
    return "🎙️ Recording... Speak now!"


def on_record_stop():
    """Called when recording stops"""
    recording_state["is_recording"] = False
    duration = 0
    if recording_state["start_time"]:
        duration = time.time() - recording_state["start_time"]
    return f"⏹️ Recording stopped ({duration:.1f}s). Processing..."


def update_live_status():
    """Update live status for recording"""
    if recording_state["is_recording"]:
        duration = time.time() - recording_state["start_time"]
        return f"🎙️ Recording... {duration:.1f}s"
    return "Ready to record"


def toggle_realtime_mode(enabled):
    """Toggle real-time mode visibility"""
    if enabled:
        return gr.update(visible=True, value="Real-time mode enabled. Live transcript will appear here during processing.")
    else:
        return gr.update(visible=False, value="")


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
            
            # Real-time transcription toggle
            enable_realtime = gr.Checkbox(
                label="🔄 Enable Real-Time Transcription",
                value=True,
                info="Show live transcript updates during processing"
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
            
            **Real-Time Mode:**
            - Shows live transcript during processing
            - Simulates word-by-word output
            - Provides visual feedback on progress
            """)
        
        with gr.Column(scale=2):
            with gr.Tab("📁 Upload File"):
                audio_input = gr.Audio(
                    type="filepath",
                    label="Upload Audio (WAV, MP3, M4A, etc.)"
                )
                upload_btn = gr.Button("🚀 Transcribe & Reform", variant="primary")
            
            with gr.Tab("🎤 Record Audio"):
                gr.Markdown("Click 'Record' to start recording from microphone. When real-time mode is enabled, you'll see live transcript updates during processing.")
                
                record_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record Audio"
                )
                
                # Live transcript output (for real-time mode)
                live_output = gr.Textbox(
                    label="🔄 Live Transcript",
                    lines=4,
                    visible=True,
                    value="Real-time mode enabled. Live transcript will appear here during processing.",
                    interactive=False
                )
                
                with gr.Row():
                    record_btn = gr.Button("🚀 Transcribe Recording", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
    
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
    
    # Record button with streaming support
    record_btn.click(
        fn=stream_record_and_transcribe,
        inputs=[record_input, language, reform_mode, model, enable_realtime],
        outputs=[live_output, raw_output, reformed_output, info_output]
    )
    
    # Clear button
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        inputs=[],
        outputs=[live_output, raw_output, reformed_output, info_output]
    )
    
    # Toggle real-time mode visibility
    enable_realtime.change(
        fn=toggle_realtime_mode,
        inputs=[enable_realtime],
        outputs=[live_output]
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
    - Enable **Real-Time Transcription** to see live updates during processing
    - Audio is processed locally - no data leaves your computer
    """)

def patch_gradio_api_info():
    """Patch Gradio's API info generation to avoid TypeError with bool schemas"""
    try:
        from gradio_client import utils as gradio_utils
        
        # Store original functions
        original_get_type = gradio_utils.get_type
        original_json_schema_to_python_type = gradio_utils._json_schema_to_python_type
        
        def patched_get_type(schema):
            """Handle both dict and bool schemas"""
            if isinstance(schema, bool):
                return "boolean"
            return original_get_type(schema)
        
        def normalize_schema(schema):
            """Recursively normalize schema to handle booleans"""
            if isinstance(schema, bool):
                return {"type": "object"}
            
            if not isinstance(schema, dict):
                return schema
            
            schema = dict(schema)  # Copy
            
            # Handle additionalProperties being a boolean
            if "additionalProperties" in schema:
                if isinstance(schema["additionalProperties"], bool):
                    if schema["additionalProperties"]:
                        schema["additionalProperties"] = {"type": "object"}
                    else:
                        del schema["additionalProperties"]
            
            # Handle anyOf with boolean values
            if "anyOf" in schema:
                schema["anyOf"] = [
                    normalize_schema(s) for s in schema["anyOf"]
                ]
            
            # Handle properties recursively
            if "properties" in schema:
                schema["properties"] = {
                    k: normalize_schema(v) for k, v in schema["properties"].items()
                }
            
            return schema
        
        def patched_json_schema_to_python_type(schema, defs):
            """Normalize schema before processing"""
            normalized = normalize_schema(schema)
            return original_json_schema_to_python_type(normalized, defs)
        
        # Replace the functions
        gradio_utils.get_type = patched_get_type
        gradio_utils._json_schema_to_python_type = patched_json_schema_to_python_type
        
        # Also patch in gradio.blocks if it imports from there
        try:
            import gradio.blocks as blocks
            if hasattr(blocks, 'client_utils'):
                blocks.client_utils.get_type = patched_get_type
                blocks.client_utils._json_schema_to_python_type = patched_json_schema_to_python_type
        except:
            pass
            
    except Exception as e:
        print(f"Warning: Could not patch Gradio API info: {e}")

# Apply patch before launching
patch_gradio_api_info()

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
    
    # Try to launch with share=False first, fallback to share=True if localhost not accessible
    try:
        demo.launch(
            server_name="0.0.0.0",  # Allow access from other devices
            server_port=port,
            share=False,  # Local only
            show_error=True,
            quiet=False,
            inbrowser=True,  # Auto-open browser
            show_api=False  # Hide API docs to prevent TypeError
        )
    except ValueError as e:
        if "shareable link" in str(e).lower() or "localhost" in str(e).lower():
            print("\n⚠️  Localhost not accessible, creating shareable link...")
            demo.launch(
                server_name="0.0.0.0",
                server_port=port,
                share=True,  # Create public link
                show_error=True,
                quiet=False,
                inbrowser=True,
                show_api=False
            )
        else:
            raise
