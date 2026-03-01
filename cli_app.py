#!/usr/bin/env python3
"""
Qwen3-ASR Pro - CLI Version (No GUI)
Works around tkinter issues on macOS
"""

import os
import sys
import argparse
import time
import subprocess
import shlex
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def transcribe_audio(audio_path: str, language: str = None) -> str:
    """Transcribe audio file using available backends (C-Binary preferred)"""
    # Check if path is a directory
    if os.path.isdir(audio_path):
        print(f"""
❌ Error: Path is a directory, not a file
   File: {audio_path}
   Reason: Directories cannot be transcribed directly
   Solution: Please provide a path to an audio file
""")
        return None
    
    # Check file permissions
    if not os.access(audio_path, os.R_OK):
        print(f"""
❌ Error: Cannot read file (permission denied)
   File: {audio_path}
   Reason: Insufficient read permissions
   Solution: Check file permissions or run with appropriate privileges
""")
        return None
    
    try:
        print(f"🎤 Transcribing: {audio_path}")
        
        # Try C-Binary first (most reliable)
        base_dir = os.path.dirname(__file__)
        c_binary = os.path.join(base_dir, "assets", "c-asr", "qwen_asr")
        model_dir = os.path.join(base_dir, "assets", "c-asr", "qwen3-asr-0.6b")
        
        if os.path.exists(c_binary) and os.path.exists(model_dir):
            cmd = [c_binary, "-d", model_dir, "-i", audio_path, "--silent"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("   Backend: C-Binary")
                return result.stdout.strip()
        
        # Try MLX second
        try:
            import mlx_audio.stt as mlx_stt
            model = mlx_stt.load("Qwen/Qwen3-ASR-1.7B")
            result = model.generate(audio_path, language=language)
            print("   Backend: MLX-Audio")
            return result.text if hasattr(result, 'text') else str(result)
        except ImportError:
            pass
        
        # Fall back to CLI
        cmd = [sys.executable, '-m', 'mlx_qwen3_asr', audio_path, '--stdout-only']
        if language:
            cmd.extend(['--language', language])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("   Backend: MLX-CLI")
            return result.stdout.strip()
        
        print("❌ All transcription backends failed")
        return None
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()
        return None

def reform_text(text: str, mode: str = "punctuate") -> str:
    """Reform text using LLM"""
    try:
        from simple_llm import SimpleLLM
        llm = SimpleLLM()
        
        if not llm.is_available():
            print("⚠️  LLM not available, returning original text")
            return text
        
        print(f"🤖 Reforming text (mode: {mode})...")
        start = time.time()
        result = llm.process(text, mode)
        elapsed = time.time() - start
        
        print(f"✅ Done in {elapsed:.2f}s (backend: {llm.backend_name})")
        return result
        
    except Exception as e:
        print(f"❌ Reform error: {e}")
        return text

def process_audio_file(audio_path: str, output_path: str = None, 
                       language: str = None, reform_mode: str = None):
    """Process audio file: transcribe + optional reform"""
    
    if not os.path.exists(audio_path):
        print(f"""
❌ Error: File not found
   File: {audio_path}
   Reason: The specified path does not exist
   Solution: Please check the file path and try again
""")
        return
    
    # Check if path is a directory
    if os.path.isdir(audio_path):
        print(f"""
❌ Error: Path is a directory, not a file
   File: {audio_path}
   Reason: Directories cannot be processed directly
   Solution: Please provide a path to an audio file
""")
        return
    
    # Check file permissions
    if not os.access(audio_path, os.R_OK):
        print(f"""
❌ Error: Cannot read file (permission denied)
   File: {audio_path}
   Reason: Insufficient read permissions
   Solution: Check file permissions or run with appropriate privileges
""")
        return
    
    print("="*60)
    print("Qwen3-ASR Pro - CLI Mode")
    print("="*60)
    print()
    
    # Transcribe
    transcript = transcribe_audio(audio_path, language)
    if not transcript:
        print("❌ Transcription failed")
        return
    
    print()
    print("📝 RAW TRANSCRIPT:")
    print("-"*60)
    print(transcript)
    print("-"*60)
    print()
    
    # Reform if requested
    if reform_mode:
        reformed = reform_text(transcript, reform_mode)
        
        print()
        print(f"✨ REFORMED ({reform_mode.upper()}):")
        print("-"*60)
        print(reformed)
        print("-"*60)
        print()
        
        final_text = reformed
    else:
        final_text = transcript
    
    # Save to file
    if not output_path:
        base = Path(audio_path).stem
        output_path = f"{base}_transcript.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)
    
    print(f"💾 Saved to: {output_path}")
    print()
    print("="*60)
    print("Done!")
    print("="*60)

def show_help():
    """Display help message"""
    print("""
╔═══════════════════════════════════════════════════════════╗
║              Qwen3-ASR Pro - CLI Commands                 ║
╠═══════════════════════════════════════════════════════════╣
║  transcribe <file>      - Transcribe audio to text        ║
║  reform <text> <mode>   - AI text reforming               ║
║  process <file> <mode>  - Transcribe + reform             ║
║  status                 - Show backend status             ║
║  help                   - Show this help                  ║
║  quit                   - Exit                            ║
╠═══════════════════════════════════════════════════════════╣
║  Reform modes: punctuate, summarize, clean, key_points    ║
╚═══════════════════════════════════════════════════════════╝

Examples:
  > transcribe recording.wav
  > reform "hello world" punctuate
  > process meeting.wav summarize
""")

def get_llm_status():
    """Get LLM backend status"""
    try:
        from simple_llm import SimpleLLM
        llm = SimpleLLM()
        return {
            'available': llm.is_available(),
            'backend': llm.backend_name,
            'models': llm.available_models if hasattr(llm, 'available_models') else []
        }
    except:
        return {'available': False, 'backend': 'none', 'models': []}

def get_transcription_status():
    """Get transcription backend status"""
    backends = []
    
    # Check MLX
    try:
        import mlx_audio.stt
        backends.append('MLX-Audio (Apple Silicon)')
    except ImportError:
        pass
    
    # Check PyTorch
    try:
        import torch
        backends.append(f'PyTorch (MPS: {torch.backends.mps.is_available()})')
    except ImportError:
        pass
    
    # Check CLI fallback
    try:
        result = subprocess.run([sys.executable, '-m', 'mlx_qwen3_asr', '--help'], 
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            backends.append('MLX-CLI (fallback)')
    except:
        pass
    
    return backends if backends else ['No backend available']

def show_status():
    """Display system status"""
    print()
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║                   System Status                           ║")
    print("╠═══════════════════════════════════════════════════════════╣")
    
    # LLM Status
    llm = get_llm_status()
    status_icon = "✅" if llm['available'] else "❌"
    print(f"║  LLM Backend:    {llm['backend']:<36} ║")
    print(f"║  LLM Status:     {status_icon} {'Ready' if llm['available'] else 'Not Available':<33} ║")
    
    if llm['available'] and llm['models']:
        models_str = ", ".join(llm['models'][:3])
        print(f"║  Ollama Models:  {models_str:<36} ║")
    
    print("╠═══════════════════════════════════════════════════════════╣")
    
    # Transcription Status
    trans_backends = get_transcription_status()
    print(f"║  Transcription Backends:                                  ║")
    for backend in trans_backends:
        print(f"║    • {backend:<50} ║")
    
    print("╠═══════════════════════════════════════════════════════════╣")
    
    # Python version
    print(f"║  Python:         {sys.version.split()[0]:<36} ║")
    print(f"║  Platform:       {sys.platform:<36} ║")
    
    print("╚═══════════════════════════════════════════════════════════╝")
    print()

def interactive_mode():
    """Interactive CLI mode"""
    print()
    print("="*60)
    print("Qwen3-ASR Pro - Interactive CLI")
    print("="*60)
    print()
    
    # Check LLM
    try:
        from simple_llm import SimpleLLM
        llm = SimpleLLM()
        print(f"🤖 LLM Status: {llm.backend_name} ({'ready' if llm.is_available() else 'not available'})")
    except:
        print("⚠️  LLM not available")
    
    print()
    print("Commands: transcribe, reform, process, status, help, quit")
    print("Type 'help' for detailed usage information.")
    print()
    
    while True:
        try:
            cmd = input("> ").strip().split()
            if not cmd:
                continue
            
            if cmd[0] == 'quit':
                break
            
            elif cmd[0] == 'help':
                show_help()
            
            elif cmd[0] == 'status':
                show_status()
            
            elif cmd[0] == 'transcribe' and len(cmd) >= 2:
                audio_file = cmd[1]
                lang = cmd[2] if len(cmd) > 2 else None
                result = transcribe_audio(audio_file, lang)
                if result:
                    print("\nTranscript:")
                    print(result)
                    print()
            
            elif cmd[0] == 'reform' and len(cmd) >= 2:
                # Check if last arg is a valid mode
                valid_modes = ['punctuate', 'summarize', 'clean', 'key_points']
                if cmd[-1] in valid_modes and len(cmd) > 2:
                    mode = cmd[-1]
                    text = ' '.join(cmd[1:-1])
                else:
                    mode = 'punctuate'
                    text = ' '.join(cmd[1:])
                result = reform_text(text, mode)
                print("\nReformed:")
                print(result)
                print()
            
            elif cmd[0] == 'process' and len(cmd) >= 2:
                audio_file = cmd[1]
                valid_modes = ['punctuate', 'summarize', 'clean', 'key_points']
                mode = cmd[2] if len(cmd) > 2 and cmd[2] in valid_modes else 'punctuate'
                process_audio_file(audio_file, reform_mode=mode)
            
            else:
                if cmd[0] in ['transcribe', 'reform', 'process']:
                    print(f"Usage: {cmd[0]} <argument>. Type 'help' for details.")
                else:
                    print(f"Unknown command: {cmd[0]}. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("Goodbye!")

def main():
    parser = argparse.ArgumentParser(description='Qwen3-ASR Pro CLI')
    parser.add_argument('audio_file', nargs='?', help='Audio file to transcribe')
    parser.add_argument('-o', '--output', help='Output text file')
    parser.add_argument('-l', '--language', help='Language code (e.g., en, zh)')
    parser.add_argument('-r', '--reform', choices=['punctuate', 'summarize', 'clean', 'key_points'],
                       help='Reform mode')
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.audio_file:
        process_audio_file(args.audio_file, args.output, args.language, args.reform)
    else:
        # Default: interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
