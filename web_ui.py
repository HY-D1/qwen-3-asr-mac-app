#!/usr/bin/env python3
"""
Qwen3-ASR Pro - Web UI
Gradio-based web interface (works in browser, no tkinter needed)
"""

import os
import sys
import tempfile
import time
from pathlib import Path

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

def transcribe_audio(audio_file, language="auto"):
    """Transcribe audio file using MLX"""
    if audio_file is None:
        return "No audio file provided", ""
    
    try:
        # Try MLX first
        try:
            import mlx_audio.stt as mlx_stt
            model = mlx_stt.load("Qwen/Qwen3-ASR-0.6B")  # Use smaller model for speed
            result = model.generate(audio_file, language=None if language == "auto" else language)
            transcript = result.text if hasattr(result, 'text') else str(result)
            return transcript, "MLX (Apple Silicon)"
        except ImportError:
            pass
        
        # Fall back to CLI
        import subprocess
        cmd = [sys.executable, '-m', 'mlx_qwen3_asr', audio_file, '--stdout-only']
        if language != "auto":
            cmd.extend(['--language', language])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.stdout.strip(), "MLX-CLI"
        
    except Exception as e:
        return f"Error: {str(e)}", "Error"

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

def process_audio(audio_file, language, reform_mode):
    """Full pipeline: transcribe + reform"""
    if audio_file is None:
        return "Please upload an audio file", "", ""
    
    # Step 1: Transcribe
    transcript, backend = transcribe_audio(audio_file, language)
    
    if transcript.startswith("Error:"):
        return transcript, "", ""
    
    # Step 2: Reform if requested
    if reform_mode != "none":
        reformed = reform_text(transcript, reform_mode)
    else:
        reformed = transcript
    
    # Get LLM info
    llm_info = f"Backend: {backend} | LLM: {llm.backend_name}"
    
    return transcript, reformed, llm_info

def record_and_transcribe(audio, language, reform_mode):
    """Handle recorded audio"""
    return process_audio(audio, language, reform_mode)

# Create Gradio interface
with gr.Blocks(title="Qwen3-ASR Pro", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎙️ Qwen3-ASR Pro - Web UI
    ### Speech-to-Text with AI Text Refinement
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
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
            **About AI Backends:**
            - **ollama-qwen**: Local free model (recommended)
            - **openai**: GPT-4 (requires API key)
            - **transformers**: Local neural network
            - **rule-based**: Basic text cleanup
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
        
        with gr.Column():
            gr.Markdown("### ✨ AI-Reformed Text")
            reformed_output = gr.Textbox(
                label="",
                lines=10,
                show_copy_button=True
            )
    
    info_output = gr.Textbox(label="Info", interactive=False)
    
    # Event handlers
    upload_btn.click(
        fn=process_audio,
        inputs=[audio_input, language, reform_mode],
        outputs=[raw_output, reformed_output, info_output]
    )
    
    record_btn.click(
        fn=record_and_transcribe,
        inputs=[record_input, language, reform_mode],
        outputs=[raw_output, reformed_output, info_output]
    )
    
    gr.Markdown("""
    ---
    ### 💡 Tips
    - Use **Punctuate** mode to add proper punctuation and capitalization
    - Use **Summarize** mode to create a concise summary
    - Use **Clean up** mode to remove filler words (um, uh, like)
    - Audio is processed locally - no data leaves your computer
    """)

if __name__ == "__main__":
    print("🚀 Starting Qwen3-ASR Pro Web UI...")
    print("📱 Open your browser and go to: http://localhost:7860")
    print("")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow access from other devices
        server_port=7860,
        share=False,  # Set to True to create a public link
        show_error=True
    )
