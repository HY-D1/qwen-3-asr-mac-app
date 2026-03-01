#!/usr/bin/env python3
"""
Minimal test for LiveStreamer to diagnose issues.
"""

import os
import sys
import time
import wave
import numpy as np
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import LiveStreamer, SAMPLE_RATE

# Paths
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
MODEL_0_6B = os.path.join(PROJECT_ROOT, 'assets', 'c-asr', 'qwen3-asr-0.6b')
BINARY_PATH = os.path.join(PROJECT_ROOT, 'assets', 'c-asr', 'qwen_asr')
SAMPLE_WAV = os.path.join(PROJECT_ROOT, 'assets', 'c-asr', 'samples', 'test_speech.wav')


def load_wav(filepath: str) -> np.ndarray:
    """Load a WAV file and return float32 numpy array."""
    with wave.open(filepath, 'rb') as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        
        raw_data = wf.readframes(nframes)
        audio_int16 = np.frombuffer(raw_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        
        if nchannels == 2:
            audio_float32 = audio_float32.reshape(-1, 2).mean(axis=1)
        
        return audio_float32


def test_basic():
    """Test basic functionality."""
    print("="*70)
    print("Minimal LiveStreamer Test")
    print("="*70)
    
    # Verify paths
    print(f"\n1. Verifying paths:")
    print(f"   Binary: {BINARY_PATH} - exists: {os.path.exists(BINARY_PATH)}")
    print(f"   Model: {MODEL_0_6B} - exists: {os.path.exists(MODEL_0_6B)}")
    print(f"   Sample: {SAMPLE_WAV} - exists: {os.path.exists(SAMPLE_WAV)}")
    
    # Load audio
    print(f"\n2. Loading audio...")
    audio = load_wav(SAMPLE_WAV)
    print(f"   Audio: {len(audio)/16000:.1f}s ({len(audio)} samples)")
    
    # Create streamer
    print(f"\n3. Creating LiveStreamer...")
    streamer = LiveStreamer(
        model_dir=MODEL_0_6B,
        binary_path=BINARY_PATH,
        sample_rate=SAMPLE_RATE
    )
    print(f"   Streamer created")
    print(f"   - chunk_duration: {streamer.chunk_duration}s")
    print(f"   - chunk_samples: {streamer.chunk_samples}")
    
    # Start
    print(f"\n4. Starting streamer...")
    outputs = []
    statuses = []
    
    def output_cb(text, is_partial=True):
        outputs.append(text)
        print(f"   [OUTPUT] {text[:60]}..." if len(text) > 60 else f"   [OUTPUT] {text}")
    
    def status_cb(status):
        statuses.append(status)
        print(f"   [STATUS] {status}")
    
    audio_file = streamer.start(
        output_callback=output_cb,
        status_callback=status_cb
    )
    print(f"   Started - is_running: {streamer.is_running}")
    print(f"   Output file: {audio_file}")
    
    # Feed audio
    print(f"\n5. Feeding audio...")
    chunk_size = int(0.1 * 16000)  # 100ms chunks
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) > 0:
            streamer.feed_audio(chunk)
    print(f"   Fed {len(audio)/16000:.1f}s of audio")
    
    # Wait
    print(f"\n6. Waiting for processing...")
    start_wait = time.time()
    max_wait = 60  # seconds
    
    while time.time() - start_wait < max_wait:
        with streamer.buffer_lock:
            pending = streamer._pending_chunks
            buffer_samples = sum(len(a) for a in streamer.audio_buffer)
        
        if pending == 0 and buffer_samples < streamer.chunk_samples * 0.5:
            print(f"   Processing complete after {time.time() - start_wait:.1f}s")
            break
        
        if int(time.time() - start_wait) % 5 == 0:
            print(f"   Waiting... pending: {pending}, buffer: {buffer_samples/16000:.1f}s")
        
        time.sleep(0.5)
    else:
        print(f"   WARNING: Timed out waiting for processing")
    
    # Stop
    print(f"\n7. Stopping streamer...")
    start_stop = time.time()
    final_file, transcript = streamer.stop()
    print(f"   Stopped in {time.time() - start_stop:.1f}s")
    print(f"   Final file: {final_file}")
    print(f"   Transcript: '{transcript}'")
    
    # Verify
    print(f"\n8. Verification:")
    print(f"   is_running: {streamer.is_running}")
    print(f"   File exists: {os.path.exists(final_file) if final_file else False}")
    print(f"   Transcript length: {len(transcript)} chars")
    
    if final_file and os.path.exists(final_file):
        file_size = os.path.getsize(final_file)
        print(f"   File size: {file_size/1024:.1f} KB")
    
    print(f"\n{'='*70}")
    print("Test Complete")
    print("="*70)
    
    return final_file, transcript


if __name__ == '__main__':
    try:
        final_file, transcript = test_basic()
        print("\n✓ Test completed successfully")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
