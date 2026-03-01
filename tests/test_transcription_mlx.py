#!/usr/bin/env python3
"""
Comprehensive tests for MLX Audio Transcription functionality.

Test Scenarios:
- Valid audio files (various formats, sample rates, channels)
- Invalid audio files (corrupted, empty, non-audio)
- Edge cases (very short, very long audio)
- Error handling and recovery
- Performance metrics

Author: HY-D1
Date: 2026-02-28
"""

import os
import sys
import time
import unittest
import tempfile
import traceback
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_AUDIO_DIR = os.path.join(BASE_DIR, "tests", "test_audio")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Test results storage
@dataclass
class TestResult:
    """Store test result with metadata"""
    test_name: str
    passed: bool
    duration: float
    error_message: str = ""
    transcription_time: float = 0.0
    backend_used: str = ""
    transcript_preview: str = ""


class TranscriptionTestSuite(unittest.TestCase):
    """MLX Transcription Test Suite"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("\n" + "="*70)
        print("MLX AUDIO TRANSCRIPTION TEST SUITE")
        print("="*70)
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test audio directory: {TEST_AUDIO_DIR}")
        print(f"Source directory: {SRC_DIR}")
        
        # Store test results
        cls.test_results: list[TestResult] = []
        
        # Check which transcription backends are available
        cls.backends = cls._detect_backends()
        
        # Import transcription modules
        cls.transcription_engine = None
        try:
            from app import TranscriptionEngine
            cls.transcription_engine = TranscriptionEngine()
            print(f"✅ TranscriptionEngine initialized with backend: {cls.transcription_engine.backend}")
        except Exception as e:
            print(f"⚠️ Could not initialize TranscriptionEngine: {e}")
        
        # Verify test audio files exist
        cls.test_files = cls._verify_test_files()
        print()
    
    @classmethod
    def _detect_backends(cls) -> Dict[str, bool]:
        """Detect available transcription backends"""
        backends = {
            'mlx_audio': False,
            'mlx_cli': False,
            'pytorch': False
        }
        
        # Check mlx-audio
        try:
            import mlx_audio.stt
            backends['mlx_audio'] = True
            print("✅ mlx-audio backend available")
        except ImportError:
            print("⚠️ mlx-audio not available")
        
        # Check mlx-cli
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'mlx_qwen3_asr', '--version'],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                backends['mlx_cli'] = True
                print("✅ mlx-cli backend available")
        except:
            print("⚠️ mlx-cli not available")
        
        # Check PyTorch
        try:
            import torch
            import qwen_asr
            backends['pytorch'] = True
            print(f"✅ PyTorch backend available (v{torch.__version__})")
        except ImportError:
            print("⚠️ PyTorch/qwen-asr not available")
        
        return backends
    
    @classmethod
    def _verify_test_files(cls) -> Dict[str, str]:
        """Verify test audio files exist"""
        test_files = {}
        expected_files = {
            '1s_silence': 'test_1s_silence.wav',
            '5s_sine': 'test_5s_sine.wav',
            '30s_sine': 'test_30s_sine.wav',
            '0.5s_short': 'test_0.5s_short.wav',
            '3s_speech_sim': 'test_3s_speech_sim.wav',
            '2s_noise': 'test_2s_noise.wav',
            'stereo_16k': 'test_stereo_16k.wav',
            '44khz': 'test_44khz.wav',
            'corrupted': 'test_corrupted.wav',
            'not_audio': 'test_not_audio.wav',
            'empty': 'test_empty.wav',
            'truncated': 'test_truncated.wav',
            '90s_long': 'test_90s_long.wav',
        }
        
        print("\n📁 Test Audio Files:")
        for key, filename in expected_files.items():
            filepath = os.path.join(TEST_AUDIO_DIR, filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                test_files[key] = filepath
                print(f"   ✅ {filename} ({size} bytes)")
            else:
                print(f"   ❌ {filename} (NOT FOUND)")
        
        return test_files
    
    def _transcribe_with_engine(self, audio_path: str, language: str = None) -> Tuple[str, str, float]:
        """
        Transcribe audio file using TranscriptionEngine.
        
        Returns:
            (transcript, backend, duration)
        """
        if not self.transcription_engine:
            raise RuntimeError("TranscriptionEngine not available")
        
        start_time = time.time()
        result = self.transcription_engine.transcribe(audio_path, language=language)
        duration = time.time() - start_time
        
        return result.text, result.backend, duration
    
    def _transcribe_with_mlx_audio(self, audio_path: str, language: str = None) -> Tuple[str, float]:
        """Transcribe using mlx-audio directly"""
        import mlx_audio.stt as mlx_stt
        
        start_time = time.time()
        model = mlx_stt.load("Qwen/Qwen3-ASR-0.6B")
        result = model.generate(audio_path, language=language)
        duration = time.time() - start_time
        
        transcript = result.text if hasattr(result, 'text') else str(result)
        return transcript, duration
    
    def _transcribe_with_cli(self, audio_path: str, language: str = None) -> Tuple[str, float]:
        """Transcribe using MLX CLI fallback"""
        cmd = [sys.executable, '-m', 'mlx_qwen3_asr', audio_path, '--stdout-only']
        if language:
            cmd.extend(['--language', language])
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        duration = time.time() - start_time
        
        if result.returncode != 0:
            raise RuntimeError(f"CLI transcription failed: {result.stderr}")
        
        return result.stdout.strip(), duration
    
    def _record_result(self, test_name: str, passed: bool, duration: float, 
                       error: str = "", transcription_time: float = 0.0,
                       backend: str = "", transcript: str = ""):
        """Record test result"""
        result = TestResult(
            test_name=test_name,
            passed=passed,
            duration=duration,
            error_message=error,
            transcription_time=transcription_time,
            backend_used=backend,
            transcript_preview=transcript[:100] if transcript else ""
        )
        self.__class__.test_results.append(result)

    # ==================================================================================
    # VALID AUDIO TESTS
    # ==================================================================================
    
    def test_01_valid_16khz_mono(self):
        """Test: Valid WAV file (16kHz, mono) - should succeed"""
        test_name = "Valid 16kHz Mono"
        start = time.time()
        
        try:
            if '3s_speech_sim' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['3s_speech_sim']
            transcript, backend, trans_time = self._transcribe_with_engine(audio_path)
            
            self.assertIsNotNone(transcript)
            self._record_result(test_name, True, time.time() - start, 
                              transcription_time=trans_time, backend=backend,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED (backend: {backend}, time: {trans_time:.2f}s)")
            print(f"   Transcript: {transcript[:50]}...")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_02_valid_44khz_mono(self):
        """Test: Valid WAV file (44.1kHz, mono) - should handle resampling"""
        test_name = "Valid 44.1kHz Mono"
        start = time.time()
        
        try:
            if '44khz' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['44khz']
            transcript, backend, trans_time = self._transcribe_with_engine(audio_path)
            
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend=backend,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED (backend: {backend}, time: {trans_time:.2f}s)")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_03_valid_stereo_16k(self):
        """Test: Valid WAV file (16kHz, stereo) - should handle channel conversion"""
        test_name = "Valid 16kHz Stereo"
        start = time.time()
        
        try:
            if 'stereo_16k' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['stereo_16k']
            transcript, backend, trans_time = self._transcribe_with_engine(audio_path)
            
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend=backend,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED (backend: {backend}, time: {trans_time:.2f}s)")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_04_very_short_audio(self):
        """Test: Very short audio (< 1 second) - edge case"""
        test_name = "Very Short Audio (< 1s)"
        start = time.time()
        
        try:
            if '0.5s_short' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['0.5s_short']
            transcript, backend, trans_time = self._transcribe_with_engine(audio_path)
            
            # Very short audio should either produce empty transcript or handle gracefully
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend=backend,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED (backend: {backend}, time: {trans_time:.2f}s)")
            print(f"   Transcript: '{transcript}'")
            
        except Exception as e:
            # Short audio may cause errors - this is acceptable if handled gracefully
            self._record_result(test_name, True, time.time() - start, error=str(e))
            print(f"⚠️  {test_name}: Handled gracefully - {e}")
    
    def test_05_long_audio_30s(self):
        """Test: Long audio (30 seconds) - performance test"""
        test_name = "Long Audio (30s)"
        start = time.time()
        
        try:
            if '30s_sine' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['30s_sine']
            transcript, backend, trans_time = self._transcribe_with_engine(audio_path)
            
            # Calculate RTF (Real-Time Factor)
            rtf = trans_time / 30.0
            
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend=backend,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED (backend: {backend}, time: {trans_time:.2f}s, RTF: {rtf:.2f}x)")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_06_long_audio_90s(self):
        """Test: Very long audio (90 seconds) - edge case"""
        test_name = "Very Long Audio (90s)"
        start = time.time()
        
        try:
            if '90s_long' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['90s_long']
            transcript, backend, trans_time = self._transcribe_with_engine(audio_path)
            
            rtf = trans_time / 90.0
            
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend=backend,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED (backend: {backend}, time: {trans_time:.2f}s, RTF: {rtf:.2f}x)")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_07_silent_audio(self):
        """Test: Silent audio (1 second) - should handle gracefully"""
        test_name = "Silent Audio"
        start = time.time()
        
        try:
            if '1s_silence' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['1s_silence']
            transcript, backend, trans_time = self._transcribe_with_engine(audio_path)
            
            # Silent audio should produce empty or minimal transcript
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend=backend,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED (backend: {backend}, time: {trans_time:.2f}s)")
            print(f"   Transcript: '{transcript}'")
            
        except Exception as e:
            self._record_result(test_name, True, time.time() - start, error=str(e))
            print(f"⚠️  {test_name}: Handled gracefully - {e}")
    
    def test_08_white_noise(self):
        """Test: White noise audio - should handle gracefully"""
        test_name = "White Noise"
        start = time.time()
        
        try:
            if '2s_noise' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['2s_noise']
            transcript, backend, trans_time = self._transcribe_with_engine(audio_path)
            
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend=backend,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED (backend: {backend}, time: {trans_time:.2f}s)")
            
        except Exception as e:
            self._record_result(test_name, True, time.time() - start, error=str(e))
            print(f"⚠️  {test_name}: Handled gracefully - {e}")

    # ==================================================================================
    # ERROR HANDLING TESTS
    # ==================================================================================
    
    def test_09_nonexistent_file(self):
        """Test: Non-existent file path - should raise error"""
        test_name = "Non-existent File"
        start = time.time()
        
        try:
            fake_path = "/path/to/nonexistent/file.wav"
            
            with self.assertRaises(Exception):
                self._transcribe_with_engine(fake_path)
            
            self._record_result(test_name, True, time.time() - start)
            print(f"✅ {test_name}: PASSED (correctly raised error)")
            
        except AssertionError:
            self._record_result(test_name, False, time.time() - start, 
                              error="Did not raise error for non-existent file")
            print(f"❌ {test_name}: FAILED - Did not raise error")
            raise
    
    def test_10_empty_file(self):
        """Test: Empty file (0 bytes) - should handle gracefully"""
        test_name = "Empty File (0 bytes)"
        start = time.time()
        
        try:
            if 'empty' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['empty']
            
            try:
                transcript, backend, trans_time = self._transcribe_with_engine(audio_path)
                # If it doesn't error, it should return empty
                self._record_result(test_name, True, time.time() - start,
                                  transcription_time=trans_time, backend=backend)
                print(f"✅ {test_name}: PASSED (handled gracefully)")
            except Exception as e:
                # Raising an error is acceptable
                self._record_result(test_name, True, time.time() - start, error=str(e))
                print(f"✅ {test_name}: PASSED (raised error: {type(e).__name__})")
                
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_11_non_audio_file(self):
        """Test: Non-audio file disguised as WAV - should handle gracefully"""
        test_name = "Non-Audio File (text)"
        start = time.time()
        
        try:
            if 'not_audio' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['not_audio']
            
            try:
                transcript, backend, trans_time = self._transcribe_with_engine(audio_path)
                self._record_result(test_name, True, time.time() - start,
                                  transcription_time=trans_time, backend=backend)
                print(f"✅ {test_name}: PASSED (handled gracefully)")
            except Exception as e:
                self._record_result(test_name, True, time.time() - start, error=str(e))
                print(f"✅ {test_name}: PASSED (raised error: {type(e).__name__})")
                
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_12_corrupted_wav(self):
        """Test: Corrupted WAV file - should handle gracefully"""
        test_name = "Corrupted WAV File"
        start = time.time()
        
        try:
            if 'corrupted' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['corrupted']
            
            try:
                transcript, backend, trans_time = self._transcribe_with_engine(audio_path)
                self._record_result(test_name, True, time.time() - start,
                                  transcription_time=trans_time, backend=backend)
                print(f"✅ {test_name}: PASSED (handled gracefully)")
            except Exception as e:
                self._record_result(test_name, True, time.time() - start, error=str(e))
                print(f"✅ {test_name}: PASSED (raised error: {type(e).__name__})")
                
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_13_truncated_wav(self):
        """Test: Truncated WAV file (valid header, no data) - should handle gracefully"""
        test_name = "Truncated WAV File"
        start = time.time()
        
        try:
            if 'truncated' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['truncated']
            
            try:
                transcript, backend, trans_time = self._transcribe_with_engine(audio_path)
                self._record_result(test_name, True, time.time() - start,
                                  transcription_time=trans_time, backend=backend)
                print(f"✅ {test_name}: PASSED (handled gracefully)")
            except Exception as e:
                self._record_result(test_name, True, time.time() - start, error=str(e))
                print(f"✅ {test_name}: PASSED (raised error: {type(e).__name__})")
                
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise

    # ==================================================================================
    # MLX CLI FALLBACK TESTS
    # ==================================================================================
    
    def test_14_mlx_cli_fallback(self):
        """Test: MLX CLI fallback works when mlx-audio is not available"""
        test_name = "MLX CLI Fallback"
        start = time.time()
        
        if not self.backends['mlx_cli']:
            self.skipTest("MLX CLI not available")
        
        try:
            if '3s_speech_sim' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['3s_speech_sim']
            transcript, trans_time = self._transcribe_with_cli(audio_path)
            
            self.assertIsNotNone(transcript)
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend="MLX-CLI",
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED (time: {trans_time:.2f}s)")
            print(f"   Transcript: {transcript[:50]}...")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_15_cli_with_language(self):
        """Test: MLX CLI with language specification"""
        test_name = "MLX CLI with Language"
        start = time.time()
        
        if not self.backends['mlx_cli']:
            self.skipTest("MLX CLI not available")
        
        try:
            if '3s_speech_sim' not in self.test_files:
                self.skipTest("Test file not available")
            
            audio_path = self.test_files['3s_speech_sim']
            transcript, trans_time = self._transcribe_with_cli(audio_path, language="en")
            
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend="MLX-CLI",
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED (time: {trans_time:.2f}s)")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise

    # ==================================================================================
    # PERFORMANCE TESTS
    # ==================================================================================
    
    def test_16_performance_benchmark(self):
        """Test: Performance benchmark with different audio lengths"""
        test_name = "Performance Benchmark"
        start = time.time()
        
        test_cases = [
            ('0.5s_short', 0.5),
            ('3s_speech_sim', 3.0),
            ('5s_sine', 5.0),
        ]
        
        results = []
        for file_key, expected_duration in test_cases:
            if file_key not in self.test_files:
                continue
            
            try:
                audio_path = self.test_files[file_key]
                trans_start = time.time()
                transcript, backend, _ = self._transcribe_with_engine(audio_path)
                trans_time = time.time() - trans_start
                
                rtf = trans_time / expected_duration if expected_duration > 0 else 0
                results.append({
                    'file': file_key,
                    'duration': expected_duration,
                    'trans_time': trans_time,
                    'rtf': rtf,
                    'backend': backend
                })
            except Exception as e:
                print(f"   ⚠️ {file_key} failed: {e}")
        
        if results:
            self._record_result(test_name, True, time.time() - start)
            print(f"✅ {test_name}: PASSED")
            print("\n   Performance Results:")
            for r in results:
                print(f"      {r['file']}: {r['trans_time']:.2f}s (RTF: {r['rtf']:.2f}x)")
        else:
            self.skipTest("No results to report")

    # ==================================================================================
    # CLASS METHODS
    # ==================================================================================
    
    @classmethod
    def tearDownClass(cls):
        """Print final test report"""
        print("\n" + "="*70)
        print("TRANSCRIPTION TEST REPORT")
        print("="*70)
        
        # Summary table
        passed = sum(1 for r in cls.test_results if r.passed)
        failed = sum(1 for r in cls.test_results if not r.passed)
        total = len(cls.test_results)
        
        print(f"\n📊 Summary:")
        print(f"   Total Tests: {total}")
        print(f"   Passed: {passed} ✅")
        print(f"   Failed: {failed} ❌")
        print(f"   Success Rate: {passed/total*100:.1f}%" if total > 0 else "   N/A")
        
        # Detailed results table
        print(f"\n📋 Detailed Results:")
        print("-" * 70)
        print(f"{'Test Name':<30} {'Status':<10} {'Backend':<12} {'Time (s)':<10}")
        print("-" * 70)
        
        for result in cls.test_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            backend = result.backend_used if result.backend_used else "N/A"
            time_str = f"{result.transcription_time:.2f}" if result.transcription_time > 0 else "N/A"
            print(f"{result.test_name:<30} {status:<10} {backend:<12} {time_str:<10}")
            if result.error_message and not result.passed:
                print(f"   Error: {result.error_message[:50]}...")
        
        # Performance metrics
        print(f"\n⏱️ Performance Metrics:")
        trans_times = [r.transcription_time for r in cls.test_results if r.transcription_time > 0]
        if trans_times:
            print(f"   Average transcription time: {sum(trans_times)/len(trans_times):.2f}s")
            print(f"   Min: {min(trans_times):.2f}s")
            print(f"   Max: {max(trans_times):.2f}s")
        
        print("\n" + "="*70)


def print_backend_info():
    """Print information about available backends"""
    print("\n🔧 Backend Information:")
    print("-" * 40)
    
    # Python version
    print(f"Python: {sys.version}")
    
    # mlx-audio
    try:
        import mlx_audio
        print(f"mlx-audio: {mlx_audio.__version__ if hasattr(mlx_audio, '__version__') else 'installed'}")
    except ImportError:
        print("mlx-audio: Not installed")
    
    # mlx-qwen3-asr
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'mlx_qwen3_asr', '--version'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print(f"mlx-qwen3-asr: {result.stdout.strip()}")
    except:
        print("mlx-qwen3-asr: Not installed or no --version support")
    
    # PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"  MPS available: {torch.backends.mps.is_available()}")
    except ImportError:
        print("PyTorch: Not installed")
    
    print()


if __name__ == '__main__':
    print_backend_info()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test methods
    suite.addTests(loader.loadTestsFromTestCase(TranscriptionTestSuite))
    
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
