#!/usr/bin/env python3
"""
Qwen3-ASR Pro - CLI Version (No GUI)
Works around tkinter issues on macOS
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def transcribe_audio(audio_path: str, language: str = None) -> str:
    """Transcribe audio file using MLX"""
    try:
        print(f"🎤 Transcribing: {audio_path}")
        
        # Try MLX first
        try:
            import mlx_audio.stt as mlx_stt
            model = mlx_stt.load("Qwen/Qwen3-ASR-1.7B")
            result = model.generate(audio_path, language=language)
            return result.text if hasattr(result, 'text') else str(result)
        except ImportError:
            pass
        
        # Fall back to CLI
        import subprocess
        cmd = [sys.executable, '-m', 'mlx_qwen3_asr', audio_path, '--stdout-only']
        if language:
            cmd.extend(['--language', language])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.stdout.strip()
        
    except Exception as e:
        print(f"❌ Transcription error: {e}")
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
        print(f"❌ File not found: {audio_path}")
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
    print("Commands:")
    print("  transcribe <audio_file> [language]  - Transcribe audio")
    print("  reform <text> [mode]                - Reform text")
    print("  process <audio_file> [mode]         - Transcribe + reform")
    print("  quit                                - Exit")
    print()
    
    while True:
        try:
            cmd = input("> ").strip().split()
            if not cmd:
                continue
            
            if cmd[0] == 'quit':
                break
            
            elif cmd[0] == 'transcribe' and len(cmd) >= 2:
                audio_file = cmd[1]
                lang = cmd[2] if len(cmd) > 2 else None
                result = transcribe_audio(audio_file, lang)
                if result:
                    print("\nTranscript:")
                    print(result)
                    print()
            
            elif cmd[0] == 'reform' and len(cmd) >= 2:
                text = ' '.join(cmd[1:-1]) if len(cmd) > 2 and cmd[-1] in ['punctuate', 'summarize', 'clean'] else ' '.join(cmd[1:])
                mode = cmd[-1] if cmd[-1] in ['punctuate', 'summarize', 'clean'] else 'punctuate'
                result = reform_text(text, mode)
                print("\nReformed:")
                print(result)
                print()
            
            elif cmd[0] == 'process' and len(cmd) >= 2:
                audio_file = cmd[1]
                mode = cmd[2] if len(cmd) > 2 else 'punctuate'
                process_audio_file(audio_file, reform_mode=mode)
            
            else:
                print("Unknown command. Type 'quit' to exit.")
        
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
