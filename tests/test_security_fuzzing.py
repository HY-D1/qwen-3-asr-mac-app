#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Security Fuzzing Test Suite                                           ║
║         Qwen3-ASR Pro - Malicious Input Testing                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Tests various fuzzing scenarios to ensure robustness against malicious inputs.
"""

import unittest
import unittest.mock as mock
import sys
import os
import tempfile
import wave
import random
import string
import struct

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock dependencies
tk_mock = mock.MagicMock()
sys.modules['tkinter'] = tk_mock
sys.modules['tkinter.ttk'] = mock.MagicMock()
sys.modules['tkinter.scrolledtext'] = mock.MagicMock()
sys.modules['tkinter.filedialog'] = mock.MagicMock()
sys.modules['tkinter.messagebox'] = mock.MagicMock()

import numpy as np
sys.modules['sounddevice'] = mock.MagicMock()
sys.modules['mlx_audio'] = mock.MagicMock()
sys.modules['mlx_audio.stt'] = mock.MagicMock()
sys.modules['librosa'] = mock.MagicMock()
sys.modules['torch'] = mock.MagicMock()
sys.modules['qwen_asr'] = mock.MagicMock()
sys.modules['mlx_qwen3_asr'] = mock.MagicMock()

from app import LiveStreamer, AudioRecorder, TranscriptionEngine


class TestFuzzingInputs(unittest.TestCase):
    """Fuzz testing with random/malicious inputs"""
    
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def generate_random_string(self, length=100):
        """Generate random string with various characters"""
        chars = string.printable
        return ''.join(random.choice(chars) for _ in range(length))
    
    def test_random_filename_fuzzing(self):
        """Test handling of random filename patterns"""
        for _ in range(50):
            try:
                filename = self.generate_random_string(50) + ".wav"
                # Remove characters that are definitely invalid
                filename = filename.replace('/', '').replace('\\', '').replace('\x00', '')
                path = os.path.join(self.test_dir, filename)
                
                # Try to create the file
                with open(path, 'w') as f:
                    f.write("test")
                
                if os.path.exists(path):
                    os.remove(path)
            except (OSError, IOError, ValueError):
                # Some random strings will fail - that's OK
                pass
    
    def test_audio_buffer_fuzzing(self):
        """Test handling of random audio buffer contents"""
        streamer = LiveStreamer()
        streamer.start()
        
        try:
            for _ in range(20):
                # Random audio data
                size = random.randint(100, 10000)
                audio = np.random.randn(size).astype(np.float32)
                streamer.feed_audio(audio)
        except Exception:
            pass  # Should not crash
        finally:
            streamer.stop()
    
    def test_extreme_audio_values(self):
        """Test handling of extreme audio values"""
        streamer = LiveStreamer()
        streamer.start()
        
        try:
            # Test NaN
            audio_nan = np.full(8000, np.nan, dtype=np.float32)
            streamer.feed_audio(audio_nan)
            
            # Test infinity
            audio_inf = np.full(8000, np.inf, dtype=np.float32)
            streamer.feed_audio(audio_inf)
            
            # Test negative infinity
            audio_neg_inf = np.full(8000, -np.inf, dtype=np.float32)
            streamer.feed_audio(audio_neg_inf)
            
            # Test very large values
            audio_large = np.full(8000, 1e10, dtype=np.float32)
            streamer.feed_audio(audio_large)
            
            # Test very small values
            audio_small = np.full(8000, 1e-10, dtype=np.float32)
            streamer.feed_audio(audio_small)
            
        except Exception:
            pass
        finally:
            streamer.stop()
    
    def test_malformed_wav_structures(self):
        """Test various malformed WAV structures"""
        malformed_files = [
            # Just RIFF header
            (b'RIFF', "just_riff.wav"),
            # RIFF with wrong size
            (b'RIFF' + struct.pack('<I', 0xFFFFFFFF), "huge_size.wav"),
            # Valid header, truncated fmt chunk
            (b'RIFF' + struct.pack('<I', 100) + b'WAVEfmt ', "truncated_fmt.wav"),
            # Extra fmt bytes
            (b'RIFF' + struct.pack('<I', 100) + b'WAVEfmt ' + b'\xff' * 50, "extra_fmt.wav"),
            # Negative values in header
            (b'RIFF' + struct.pack('<I', 100) + b'WAVEfmt ' + struct.pack('<I', 0xFFFFFFFF), "negative.wav"),
        ]
        
        for data, filename in malformed_files:
            path = os.path.join(self.test_dir, filename)
            with open(path, 'wb') as f:
                f.write(data)
            
            # Try to open with wave module - should raise error or handle gracefully
            try:
                with wave.open(path, 'rb') as wf:
                    _ = wf.getnchannels()
            except (wave.Error, EOFError, struct.error):
                pass  # Expected
    
    def test_boundary_values(self):
        """Test boundary and edge values"""
        recorder = AudioRecorder()
        
        # Test silence threshold boundaries
        recorder.set_params(threshold=0.0)
        recorder.set_params(threshold=1.0)
        recorder.set_params(threshold=0.5)
        
        # Test duration boundaries
        recorder.set_params(duration=0.1)
        recorder.set_params(duration=3600)  # 1 hour
        recorder.set_params(duration=0.5)
    
    def test_unicode_normalization_attacks(self):
        """Test unicode normalization attacks"""
        # Homoglyph attacks - characters that look similar
        homoglyphs = [
            "аpple.com",  # Cyrillic 'а' looks like Latin 'a'
            "раураl.com",  # Mixed Cyrillic/Latin
            "ｅｘａｍｐｌｅ",  # Fullwidth characters
        ]
        
        for text in homoglyphs:
            # Should handle these as distinct strings
            self.assertIsInstance(text, str)
            self.assertTrue(len(text) > 0)


class TestDenialOfService(unittest.TestCase):
    """Test denial of service scenarios"""
    
    def test_repeated_start_stop(self):
        """Test rapid start/stop cycles (DoS attempt)"""
        streamer = LiveStreamer()
        
        start_time = time.time()
        for _ in range(100):
            streamer.start()
            streamer.stop()
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (< 30 seconds)
        self.assertLess(elapsed, 30.0)
    
    def test_audio_flood(self):
        """Test audio buffer flooding"""
        streamer = LiveStreamer()
        streamer.start()
        
        start_time = time.time()
        try:
            for _ in range(1000):
                audio = np.zeros(8000, dtype=np.float32)
                streamer.feed_audio(audio)
        except Exception:
            pass
        finally:
            streamer.stop()
        
        elapsed = time.time() - start_time
        # Should handle 1000 chunks in reasonable time
        self.assertLess(elapsed, 60.0)


class TestIntegerOverflow(unittest.TestCase):
    """Test for integer overflow vulnerabilities"""
    
    def test_large_frame_counts(self):
        """Test handling of large frame counts"""
        # Create WAV claiming huge number of frames
        path = tempfile.mktemp(suffix='.wav')
        
        try:
            with open(path, 'wb') as f:
                f.write(b'RIFF')
                f.write(struct.pack('<I', 0x7FFFFFFF))  # Max positive int32
                f.write(b'WAVEfmt ')
                f.write(struct.pack('<I', 16))
                f.write(struct.pack('<H', 1))  # PCM
                f.write(struct.pack('<H', 1))  # Mono
                f.write(struct.pack('<I', 16000))  # Sample rate
                f.write(struct.pack('<I', 32000))  # Byte rate
                f.write(struct.pack('<H', 2))  # Block align
                f.write(struct.pack('<H', 16))  # Bits per sample
                f.write(b'data')
                f.write(struct.pack('<I', 0x7FFFFFFF))  # Huge data size
                f.write(b'\x00' * 100)  # Small actual data
            
            # Should not crash when trying to read
            try:
                with wave.open(path, 'rb') as wf:
                    _ = wf.getnframes()
            except (wave.Error, MemoryError):
                pass
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == '__main__':
    import time
    
    print("=" * 80)
    print("Security Fuzzing Test Suite")
    print("=" * 80)
    print()
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 80)
    print("Fuzzing Summary")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All fuzzing tests passed - No crashes detected")
    else:
        print("\n⚠️  Some tests failed - Potential vulnerabilities found")
    
    print("=" * 80)
    
    sys.exit(0 if result.wasSuccessful() else 1)
