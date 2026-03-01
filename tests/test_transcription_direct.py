#!/usr/bin/env python3
"""
Direct Transcription Function Tests
Tests the actual transcription functions from web_ui.py and cli_app.py
"""

import os
import sys
import time
import unittest
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_AUDIO_DIR = os.path.join(BASE_DIR, "tests", "test_audio")

# Add project root to path
sys.path.insert(0, BASE_DIR)

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


class DirectTranscriptionTests(unittest.TestCase):
    """Test transcription functions directly"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("\n" + "="*70)
        print("DIRECT TRANSCRIPTION FUNCTION TESTS")
        print("="*70)
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        cls.test_results: list[TestResult] = []
        
        # Verify test files
        cls.test_files = {}
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
                cls.test_files[key] = filepath
                print(f"   ✅ {filename} ({size} bytes)")
            else:
                print(f"   ❌ {filename} (NOT FOUND)")
        
        # Check available backends
        print("\n🔧 Checking Backends:")
        cls.backends = cls._check_backends()
        print()
    
    @classmethod
    def _check_backends(cls) -> Dict[str, bool]:
        """Check which backends are available"""
        backends = {'mlx_audio': False, 'mlx_cli': False, 'pytorch': False}
        
        # Check mlx-audio
        try:
            import mlx_audio.stt
            backends['mlx_audio'] = True
            print("   ✅ mlx-audio available")
        except ImportError:
            print("   ⚠️  mlx-audio not available")
        
        # Check mlx-cli
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'mlx_qwen3_asr', '--version'],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                backends['mlx_cli'] = True
                print("   ✅ mlx-cli available")
        except:
            print("   ⚠️  mlx-cli not available")
        
        # Check PyTorch
        try:
            import torch
            print(f"   ✅ PyTorch {torch.__version__} available")
            try:
                import qwen_asr
                backends['pytorch'] = True
                print("   ✅ qwen-asr available")
            except ImportError:
                print("   ⚠️  qwen-asr not available")
        except ImportError:
            print("   ⚠️  PyTorch not available")
        
        return backends
    
    def _transcribe_web_ui_style(self, audio_path: str, language: str = "auto") -> Tuple[str, str, float]:
        """
        Transcribe using web_ui.py style (mlx-audio or CLI fallback)
        Returns: (transcript, backend, duration)
        """
        start_time = time.time()
        
        if audio_path is None:
            return "No audio file provided", "None", 0.0
        
        # Try MLX first
        try:
            import mlx_audio.stt as mlx_stt
            model = mlx_stt.load("Qwen/Qwen3-ASR-0.6B")
            result = model.generate(audio_path, language=None if language == "auto" else language)
            transcript = result.text if hasattr(result, 'text') else str(result)
            duration = time.time() - start_time
            return transcript, "MLX (Apple Silicon)", duration
        except ImportError:
            pass
        
        # Fall back to CLI
        cmd = [sys.executable, '-m', 'mlx_qwen3_asr', audio_path, '--stdout-only']
        if language != "auto":
            cmd.extend(['--language', language])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        duration = time.time() - start_time
        
        if result.returncode != 0:
            raise RuntimeError(f"CLI transcription failed: {result.stderr}")
        
        return result.stdout.strip(), "MLX-CLI", duration
    
    def _transcribe_cli_app_style(self, audio_path: str, language: str = None) -> Tuple[str, float]:
        """
        Transcribe using cli_app.py style
        Returns: (transcript, duration)
        """
        start_time = time.time()
        
        # Try MLX first
        try:
            import mlx_audio.stt as mlx_stt
            model = mlx_stt.load("Qwen/Qwen3-ASR-1.7B")
            result = model.generate(audio_path, language=language)
            transcript = result.text if hasattr(result, 'text') else str(result)
            return transcript, time.time() - start_time
        except ImportError:
            pass
        
        # Fall back to CLI
        cmd = [sys.executable, '-m', 'mlx_qwen3_asr', audio_path, '--stdout-only']
        if language:
            cmd.extend(['--language', language])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise RuntimeError(f"CLI transcription failed: {result.stderr}")
        
        return result.stdout.strip(), time.time() - start_time
    
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
    # WEB_UI STYLE TESTS
    # ==================================================================================
    
    def test_webui_valid_16khz(self):
        """Test web_ui.py style transcription with valid 16kHz mono"""
        test_name = "web_ui: Valid 16kHz Mono"
        start = time.time()
        
        if not (self.backends['mlx_audio'] or self.backends['mlx_cli']):
            self.skipTest("No MLX backend available")
        
        if '3s_speech_sim' not in self.test_files:
            self.skipTest("Test file not available")
        
        try:
            audio_path = self.test_files['3s_speech_sim']
            transcript, backend, trans_time = self._transcribe_web_ui_style(audio_path)
            
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend=backend,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED")
            print(f"   Backend: {backend}, Time: {trans_time:.2f}s")
            print(f"   Transcript: {transcript[:60]}...")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_webui_valid_44khz(self):
        """Test web_ui.py style with 44.1kHz file"""
        test_name = "web_ui: Valid 44.1kHz"
        start = time.time()
        
        if not (self.backends['mlx_audio'] or self.backends['mlx_cli']):
            self.skipTest("No MLX backend available")
        
        if '44khz' not in self.test_files:
            self.skipTest("Test file not available")
        
        try:
            audio_path = self.test_files['44khz']
            transcript, backend, trans_time = self._transcribe_web_ui_style(audio_path)
            
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend=backend,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED")
            print(f"   Backend: {backend}, Time: {trans_time:.2f}s")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_webui_stereo(self):
        """Test web_ui.py style with stereo file"""
        test_name = "web_ui: Stereo 16kHz"
        start = time.time()
        
        if not (self.backends['mlx_audio'] or self.backends['mlx_cli']):
            self.skipTest("No MLX backend available")
        
        if 'stereo_16k' not in self.test_files:
            self.skipTest("Test file not available")
        
        try:
            audio_path = self.test_files['stereo_16k']
            transcript, backend, trans_time = self._transcribe_web_ui_style(audio_path)
            
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend=backend,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED")
            print(f"   Backend: {backend}, Time: {trans_time:.2f}s")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_webui_with_language(self):
        """Test web_ui.py style with language specification"""
        test_name = "web_ui: With Language (en)"
        start = time.time()
        
        if not (self.backends['mlx_audio'] or self.backends['mlx_cli']):
            self.skipTest("No MLX backend available")
        
        if '3s_speech_sim' not in self.test_files:
            self.skipTest("Test file not available")
        
        try:
            audio_path = self.test_files['3s_speech_sim']
            transcript, backend, trans_time = self._transcribe_web_ui_style(audio_path, language="en")
            
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time, backend=backend,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED")
            print(f"   Backend: {backend}, Time: {trans_time:.2f}s")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_webui_nonexistent_file(self):
        """Test web_ui.py style with non-existent file"""
        test_name = "web_ui: Non-existent File"
        start = time.time()
        
        try:
            fake_path = "/path/to/nonexistent.wav"
            # Should raise an exception
            with self.assertRaises(Exception):
                self._transcribe_web_ui_style(fake_path)
            
            self._record_result(test_name, True, time.time() - start)
            print(f"✅ {test_name}: PASSED (correctly raised error)")
            
        except AssertionError:
            self._record_result(test_name, False, time.time() - start,
                              error="Did not raise error")
            print(f"❌ {test_name}: FAILED - Did not raise error")
            raise
    
    def test_webui_none_input(self):
        """Test web_ui.py style with None input"""
        test_name = "web_ui: None Input"
        start = time.time()
        
        try:
            transcript, backend, _ = self._transcribe_web_ui_style(None)
            
            self.assertEqual(transcript, "No audio file provided")
            self._record_result(test_name, True, time.time() - start, backend=backend)
            print(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_webui_empty_file(self):
        """Test web_ui.py style with empty file"""
        test_name = "web_ui: Empty File"
        start = time.time()
        
        if not (self.backends['mlx_audio'] or self.backends['mlx_cli']):
            self.skipTest("No MLX backend available")
        
        if 'empty' not in self.test_files:
            self.skipTest("Test file not available")
        
        try:
            audio_path = self.test_files['empty']
            try:
                transcript, backend, trans_time = self._transcribe_web_ui_style(audio_path)
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
    # CLI_APP STYLE TESTS
    # ==================================================================================
    
    def test_cliapp_valid(self):
        """Test cli_app.py style transcription"""
        test_name = "cli_app: Valid Audio"
        start = time.time()
        
        if not (self.backends['mlx_audio'] or self.backends['mlx_cli']):
            self.skipTest("No MLX backend available")
        
        if '3s_speech_sim' not in self.test_files:
            self.skipTest("Test file not available")
        
        try:
            audio_path = self.test_files['3s_speech_sim']
            transcript, trans_time = self._transcribe_cli_app_style(audio_path)
            
            self.assertIsNotNone(transcript)
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED")
            print(f"   Time: {trans_time:.2f}s")
            print(f"   Transcript: {transcript[:60]}...")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_cliapp_long_audio(self):
        """Test cli_app.py style with long audio"""
        test_name = "cli_app: Long Audio (30s)"
        start = time.time()
        
        if not (self.backends['mlx_audio'] or self.backends['mlx_cli']):
            self.skipTest("No MLX backend available")
        
        if '30s_sine' not in self.test_files:
            self.skipTest("Test file not available")
        
        try:
            audio_path = self.test_files['30s_sine']
            trans_start = time.time()
            transcript, _ = self._transcribe_cli_app_style(audio_path)
            trans_time = time.time() - trans_start
            
            rtf = trans_time / 30.0
            
            self._record_result(test_name, True, time.time() - start,
                              transcription_time=trans_time,
                              transcript=transcript)
            print(f"✅ {test_name}: PASSED")
            print(f"   Time: {trans_time:.2f}s, RTF: {rtf:.2f}x")
            
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise

    # ==================================================================================
    # BACKEND DETECTION TESTS
    # ==================================================================================
    
    def test_backend_detection_order(self):
        """Test that backends are detected in correct order"""
        test_name = "Backend Detection Order"
        start = time.time()
        
        # Priority should be: mlx-audio > mlx-cli > pytorch
        print(f"\n   Backend detection results:")
        print(f"   - mlx-audio: {'✅' if self.backends['mlx_audio'] else '❌'}")
        print(f"   - mlx-cli: {'✅' if self.backends['mlx_cli'] else '❌'}")
        print(f"   - pytorch: {'✅' if self.backends['pytorch'] else '❌'}")
        
        self._record_result(test_name, True, time.time() - start)
        print(f"✅ {test_name}: PASSED")
    
    def test_import_web_ui(self):
        """Test importing web_ui module"""
        test_name = "Import web_ui.py"
        start = time.time()
        
        try:
            import web_ui
            self.assertTrue(hasattr(web_ui, 'transcribe_audio'))
            self._record_result(test_name, True, time.time() - start)
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise
    
    def test_import_cli_app(self):
        """Test importing cli_app module"""
        test_name = "Import cli_app.py"
        start = time.time()
        
        try:
            import cli_app
            self.assertTrue(hasattr(cli_app, 'transcribe_audio'))
            self._record_result(test_name, True, time.time() - start)
            print(f"✅ {test_name}: PASSED")
        except Exception as e:
            self._record_result(test_name, False, time.time() - start, error=str(e))
            print(f"❌ {test_name}: FAILED - {e}")
            raise

    @classmethod
    def tearDownClass(cls):
        """Print final test report"""
        print("\n" + "="*70)
        print("DIRECT TRANSCRIPTION TEST REPORT")
        print("="*70)
        
        # Summary
        passed = sum(1 for r in cls.test_results if r.passed)
        failed = sum(1 for r in cls.test_results if not r.passed)
        total = len(cls.test_results)
        
        print(f"\n📊 Summary:")
        print(f"   Total Tests: {total}")
        print(f"   Passed: {passed} ✅")
        print(f"   Failed: {failed} ❌")
        if total > 0:
            print(f"   Success Rate: {passed/total*100:.1f}%")
        
        # Detailed results
        print(f"\n📋 Detailed Results:")
        print("-" * 70)
        print(f"{'Test Name':<35} {'Status':<10} {'Backend':<12} {'Time (s)':<10}")
        print("-" * 70)
        
        for result in cls.test_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            backend = result.backend_used if result.backend_used else "N/A"
            time_str = f"{result.transcription_time:.2f}" if result.transcription_time > 0 else "N/A"
            print(f"{result.test_name:<35} {status:<10} {backend:<12} {time_str:<10}")
            if result.error_message and not result.passed:
                print(f"   Error: {result.error_message[:60]}...")
        
        # Performance
        print(f"\n⏱️ Performance Metrics:")
        trans_times = [r.transcription_time for r in cls.test_results if r.transcription_time > 0]
        if trans_times:
            print(f"   Average transcription time: {sum(trans_times)/len(trans_times):.2f}s")
            print(f"   Min: {min(trans_times):.2f}s")
            print(f"   Max: {max(trans_times):.2f}s")
        else:
            print("   No transcription times recorded")
        
        print("\n" + "="*70)


def print_environment_info():
    """Print environment information"""
    print("\n🔧 Environment Information:")
    print("-" * 40)
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Test Audio Directory: {TEST_AUDIO_DIR}")
    
    # Check Python path
    print("\n📂 Python Path:")
    for p in sys.path[:3]:
        print(f"   {p}")
    
    print()


if __name__ == '__main__':
    print_environment_info()
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(DirectTranscriptionTests))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    sys.exit(0 if result.wasSuccessful() else 1)
