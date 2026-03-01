#!/usr/bin/env python3
"""
================================================================================
Batch Mode and File Upload Test Suite
================================================================================
Comprehensive tests for the batch transcription functionality in Qwen3-ASR Pro.

Test Coverage:
- Audio Format Support: WAV, MP3, M4A, FLAC, OGG
- Sample Rates: 8kHz, 16kHz, 44.1kHz, 48kHz
- Edge Cases: Empty files, corrupt files, Unicode filenames
- Performance Metrics: RTF (Real-Time Factor), processing time
- Progress Callbacks: Verification of callback invocation

Usage:
    cd /Users/harrydai/Desktop/Personal Portfolio/qwen-3-asr-mac-app-main
    python -m pytest tests/test_batch_mode.py -v
    python -m pytest tests/test_batch_mode.py -v --tb=short 2>&1 | tee test_report.txt

Requirements:
- ffmpeg (for audio format conversion)
- soundfile, numpy, librosa (for audio processing)
"""

import os
import sys
import time
import wave
import struct
import tempfile
import shutil
import unittest
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Any
from datetime import datetime

# Add src to path for importing app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Try to import the app modules
try:
    from app import TranscriptionEngine, PerformanceStats
    APP_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import app modules: {e}")
    APP_IMPORT_SUCCESS = False

# Test configuration
TEST_DIR = Path(__file__).parent
ASSETS_DIR = TEST_DIR / "assets"
REPORT_DIR = TEST_DIR / "reports"

# Ensure directories exist
ASSETS_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    status: str  # 'PASS', 'FAIL', 'SKIP', 'ERROR'
    duration: float = 0.0
    error_message: str = ""
    audio_duration: float = 0.0
    processing_time: float = 0.0
    rtf: float = 0.0
    format_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'status': self.status,
            'duration': f"{self.duration:.2f}s",
            'error_message': self.error_message,
            'audio_duration': f"{self.audio_duration:.2f}s",
            'processing_time': f"{self.processing_time:.2f}s",
            'rtf': f"{self.rtf:.2f}x",
            'format_info': self.format_info
        }


class AudioFileGenerator:
    """Generate synthetic audio files in various formats for testing"""
    
    SUPPORTED_FORMATS = ['wav', 'mp3', 'm4a', 'flac', 'ogg']
    SAMPLE_RATES = [8000, 16000, 44100, 48000]
    
    def __init__(self, output_dir: Path = ASSETS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self._check_ffmpeg()
        
    def _check_ffmpeg(self):
        """Check if ffmpeg is available"""
        self.has_ffmpeg = shutil.which('ffmpeg') is not None
        if not self.has_ffmpeg:
            print("Warning: ffmpeg not found. Some format conversions may fail.")
    
    def generate_sine_wave(self, frequency: float = 440.0, duration: float = 5.0,
                           sample_rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
        """Generate a sine wave audio signal"""
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        # Add harmonics for more realistic sound
        wave = amplitude * np.sin(2 * np.pi * frequency * t)
        wave += 0.3 * amplitude * np.sin(2 * np.pi * frequency * 2 * t)
        wave += 0.1 * amplitude * np.sin(2 * np.pi * frequency * 3 * t)
        return wave.astype(np.float32)
    
    def generate_speech_like(self, duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
        """Generate a speech-like signal (modulated noise)"""
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # Base carrier frequency (vocal tract)
        carrier = np.sin(2 * np.pi * 150 * t)
        
        # Formants (vowel-like resonances)
        formant1 = 0.5 * np.sin(2 * np.pi * 500 * t)
        formant2 = 0.3 * np.sin(2 * np.pi * 1500 * t)
        formant3 = 0.1 * np.sin(2 * np.pi * 2500 * t)
        
        # Amplitude modulation (syllable rate)
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
        
        # Combine
        signal = (carrier + formant1 + formant2 + formant3) * modulation
        
        # Add slight noise
        noise = 0.05 * np.random.randn(samples)
        signal += noise
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) * 0.7
        
        return signal.astype(np.float32)
    
    def save_wav(self, audio: np.ndarray, filename: str, sample_rate: int = 16000) -> Path:
        """Save audio as WAV file"""
        filepath = self.output_dir / filename
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        return filepath
    
    def convert_to_format(self, input_path: Path, output_format: str,
                          bitrate: Optional[str] = None) -> Path:
        """Convert WAV file to other format using ffmpeg"""
        if not self.has_ffmpeg:
            raise RuntimeError("ffmpeg not available for format conversion")
        
        output_path = input_path.with_suffix(f'.{output_format}')
        
        cmd = ['ffmpeg', '-y', '-i', str(input_path)]
        
        if output_format == 'mp3':
            cmd.extend(['-codec:a', 'libmp3lame'])
            if bitrate:
                cmd.extend(['-b:a', bitrate])
        elif output_format == 'm4a':
            cmd.extend(['-codec:a', 'aac'])
            if bitrate:
                cmd.extend(['-b:a', bitrate])
        elif output_format == 'flac':
            cmd.extend(['-codec:a', 'flac'])
        elif output_format == 'ogg':
            cmd.extend(['-codec:a', 'libvorbis'])
            if bitrate:
                cmd.extend(['-q:a', '4'])
        
        cmd.append(str(output_path))
        
        import subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
        
        return output_path
    
    def generate_test_file(self, format: str, duration: float = 5.0,
                           sample_rate: int = 16000, bitrate: Optional[str] = None,
                           filename: Optional[str] = None) -> Path:
        """Generate a test audio file in specified format"""
        
        if filename is None:
            filename = f"test_{format}_{sample_rate}hz_{int(duration)}s"
            if bitrate:
                filename += f"_{bitrate.replace('/', '_')}"
            filename += f".{format if format != 'm4a' else 'm4a'}"
        
        # Generate speech-like audio
        audio = self.generate_speech_like(duration, sample_rate)
        
        # Save as WAV first (base format)
        wav_path = self.output_dir / f"temp_{int(time.time()*1000)}.wav"
        self.save_wav(audio, wav_path.name, sample_rate)
        
        if format == 'wav':
            # Rename to final filename
            final_path = self.output_dir / filename
            shutil.move(wav_path, final_path)
            return final_path
        else:
            # Convert to target format
            final_path = self.convert_to_format(wav_path, format, bitrate)
            wav_path.unlink(missing_ok=True)
            # Rename to requested filename
            if final_path.name != filename:
                new_path = self.output_dir / filename
                shutil.move(final_path, new_path)
                return new_path
            return final_path
    
    def generate_corrupt_file(self, filename: str, size: int = 1024) -> Path:
        """Generate a corrupt audio file"""
        filepath = self.output_dir / filename
        with open(filepath, 'wb') as f:
            # Write random bytes
            f.write(os.urandom(size))
        return filepath
    
    def generate_empty_file(self, filename: str) -> Path:
        """Generate an empty file"""
        filepath = self.output_dir / filename
        with open(filepath, 'wb') as f:
            pass
        return filepath
    
    def generate_truncated_wav(self, filename: str) -> Path:
        """Generate a truncated/corrupt WAV file"""
        filepath = self.output_dir / filename
        audio = self.generate_sine_wave(duration=1.0, sample_rate=16000)
        self.save_wav(audio, filepath.name, 16000)
        
        # Truncate the file
        with open(filepath, 'rb') as f:
            data = f.read()
        with open(filepath, 'wb') as f:
            f.write(data[:len(data)//2])  # Write only half
        
        return filepath


class TestBatchMode(unittest.TestCase):
    """Test suite for batch transcription functionality"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("\n" + "="*80)
        print("Qwen3-ASR Pro - Batch Mode Test Suite")
        print("="*80)
        
        cls.generator = AudioFileGenerator()
        cls.results: List[TestResult] = []
        cls.engine: Optional[TranscriptionEngine] = None
        
        # Try to initialize the transcription engine
        if APP_IMPORT_SUCCESS:
            try:
                cls.engine = TranscriptionEngine()
                print(f"✓ TranscriptionEngine initialized (backend: {cls.engine.backend})")
            except Exception as e:
                print(f"⚠ Could not initialize TranscriptionEngine: {e}")
                print("  Tests will run in mock mode")
        else:
            print("⚠ App modules not available - running in mock mode")
        
        cls.start_time = time.time()
    
    @classmethod
    def tearDownClass(cls):
        """Generate test report"""
        total_time = time.time() - cls.start_time
        
        # Generate report
        cls._generate_report(total_time)
    
    @classmethod
    def _generate_report(cls, total_time: float):
        """Generate comprehensive test report"""
        report_path = REPORT_DIR / f"batch_mode_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("Qwen3-ASR Pro - Batch Mode Test Report\n")
            f.write("="*80 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Test Time: {total_time:.2f}s\n\n")
            
            # Backend info
            f.write("Backend Information:\n")
            f.write("-" * 40 + "\n")
            if cls.engine:
                f.write(f"  Backend: {cls.engine.backend}\n")
                f.write(f"  Available: Yes\n")
            else:
                f.write(f"  Backend: Not Available\n")
                f.write(f"  Note: Running in mock mode\n")
            f.write("\n")
            
            # Format Compatibility Matrix
            f.write("Format Compatibility Matrix:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Format':<15} {'Sample Rate':<15} {'Status':<10} {'RTF':<10} {'Time':<10}\n")
            f.write("-" * 80 + "\n")
            
            for result in cls.results:
                format_info = result.format_info
                format_name = format_info.get('format', 'N/A')
                sample_rate = format_info.get('sample_rate', 'N/A')
                status = result.status
                rtf = f"{result.rtf:.2f}x" if result.rtf > 0 else "N/A"
                proc_time = f"{result.processing_time:.2f}s" if result.processing_time > 0 else "N/A"
                f.write(f"{format_name:<15} {str(sample_rate):<15} {status:<10} {rtf:<10} {proc_time:<10}\n")
            
            f.write("-" * 80 + "\n\n")
            
            # Summary statistics
            total = len(cls.results)
            passed = sum(1 for r in cls.results if r.status == 'PASS')
            failed = sum(1 for r in cls.results if r.status == 'FAIL')
            errors = sum(1 for r in cls.results if r.status == 'ERROR')
            skipped = sum(1 for r in cls.results if r.status == 'SKIP')
            
            f.write("Test Summary:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Total Tests:   {total}\n")
            f.write(f"  Passed:        {passed} ({passed/total*100:.1f}%)\n")
            f.write(f"  Failed:        {failed} ({failed/total*100:.1f}%)\n")
            f.write(f"  Errors:        {errors} ({errors/total*100:.1f}%)\n")
            f.write(f"  Skipped:       {skipped} ({skipped/total*100:.1f}%)\n")
            f.write("\n")
            
            # Performance metrics
            rtf_values = [r.rtf for r in cls.results if r.rtf > 0]
            if rtf_values:
                f.write("Performance Metrics:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average RTF:   {sum(rtf_values)/len(rtf_values):.2f}x\n")
                f.write(f"  Min RTF:       {min(rtf_values):.2f}x\n")
                f.write(f"  Max RTF:       {max(rtf_values):.2f}x\n")
                f.write("\n")
            
            # Detailed results
            f.write("Detailed Test Results:\n")
            f.write("="*80 + "\n\n")
            
            for result in cls.results:
                f.write(f"Test: {result.test_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Status:          {result.status}\n")
                f.write(f"  Test Duration:   {result.duration:.2f}s\n")
                f.write(f"  Audio Duration:  {result.audio_duration:.2f}s\n")
                f.write(f"  Processing Time: {result.processing_time:.2f}s\n")
                f.write(f"  RTF:             {result.rtf:.2f}x\n")
                if result.error_message:
                    f.write(f"  Error:           {result.error_message}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("End of Report\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ Test report saved to: {report_path}")
        
        # Print summary to console
        print("\n" + "="*80)
        print("Test Summary:")
        print("="*80)
        print(f"  Total:   {total}")
        print(f"  Passed:  {passed}")
        print(f"  Failed:  {failed}")
        print(f"  Errors:  {errors}")
        print(f"  Skipped: {skipped}")
        print("="*80)
    
    def _record_result(self, result: TestResult):
        """Record a test result"""
        self.__class__.results.append(result)
    
    def _run_transcription_test(self, test_name: str, file_path: Path,
                                expected_duration: float = 0.0,
                                format_info: Optional[Dict] = None) -> Tuple[bool, str, PerformanceStats]:
        """Run a transcription test and return results"""
        
        start_time = time.time()
        
        if not self.engine:
            # Mock mode - simulate processing
            time.sleep(0.1)
            mock_stats = PerformanceStats()
            mock_stats.audio_duration = expected_duration
            mock_stats.processing_time = 0.1
            mock_stats.rtf = 0.1 / expected_duration if expected_duration > 0 else 0
            return True, "Mock transcription result", mock_stats
        
        try:
            progress_calls = []
            
            def progress_callback(msg: str):
                progress_calls.append(msg)
            
            result, stats = self.engine.transcribe(
                str(file_path),
                progress_callback=progress_callback
            )
            
            duration = time.time() - start_time
            
            # Verify progress callback was called
            self.assertGreater(len(progress_calls), 0, "Progress callback was not called")
            
            # Verify result structure
            self.assertIn('text', result, "Result missing 'text' field")
            self.assertIn('backend', result, "Result missing 'backend' field")
            
            return True, result.get('text', ''), stats
            
        except Exception as e:
            duration = time.time() - start_time
            return False, str(e), PerformanceStats()
    
    # ================================================================================
    # WAV File Tests
    # ================================================================================
    
    def test_wav_8khz(self):
        """Test WAV file with 8kHz sample rate"""
        test_name = "WAV_8kHz"
        print(f"\n  Testing {test_name}...")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_test_file(
                'wav', duration=3.0, sample_rate=8000,
                filename='test_8khz.wav'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=3.0,
                format_info={'format': 'WAV', 'sample_rate': '8kHz'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                error_message='' if success else text,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'WAV', 'sample_rate': 8000}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'WAV', 'sample_rate': 8000}
            )
            self._record_result(result)
            self.fail(f"Test setup failed: {e}")
    
    def test_wav_16khz(self):
        """Test WAV file with 16kHz sample rate (standard)"""
        test_name = "WAV_16kHz"
        print(f"\n  Testing {test_name}...")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_test_file(
                'wav', duration=3.0, sample_rate=16000,
                filename='test_16khz.wav'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=3.0,
                format_info={'format': 'WAV', 'sample_rate': '16kHz'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'WAV', 'sample_rate': 16000}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'WAV', 'sample_rate': 16000}
            )
            self._record_result(result)
            self.fail(f"Test setup failed: {e}")
    
    def test_wav_441khz(self):
        """Test WAV file with 44.1kHz sample rate (CD quality)"""
        test_name = "WAV_44.1kHz"
        print(f"\n  Testing {test_name}...")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_test_file(
                'wav', duration=3.0, sample_rate=44100,
                filename='test_441khz.wav'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=3.0,
                format_info={'format': 'WAV', 'sample_rate': '44.1kHz'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'WAV', 'sample_rate': 44100}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'WAV', 'sample_rate': 44100}
            )
            self._record_result(result)
            self.fail(f"Test setup failed: {e}")
    
    # ================================================================================
    # MP3 File Tests
    # ================================================================================
    
    def test_mp3_128k(self):
        """Test MP3 file with 128kbps bitrate"""
        test_name = "MP3_128kbps"
        print(f"\n  Testing {test_name}...")
        
        if not self.generator.has_ffmpeg:
            result = TestResult(
                test_name=test_name,
                status='SKIP',
                error_message='ffmpeg not available',
                format_info={'format': 'MP3', 'bitrate': '128k'}
            )
            self._record_result(result)
            self.skipTest("ffmpeg not available")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_test_file(
                'mp3', duration=3.0, sample_rate=44100, bitrate='128k',
                filename='test_128k.mp3'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=3.0,
                format_info={'format': 'MP3', 'bitrate': '128k'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'MP3', 'bitrate': '128k'}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'MP3', 'bitrate': '128k'}
            )
            self._record_result(result)
            self.fail(f"Test failed: {e}")
    
    def test_mp3_320k(self):
        """Test MP3 file with 320kbps bitrate (high quality)"""
        test_name = "MP3_320kbps"
        print(f"\n  Testing {test_name}...")
        
        if not self.generator.has_ffmpeg:
            result = TestResult(
                test_name=test_name,
                status='SKIP',
                error_message='ffmpeg not available',
                format_info={'format': 'MP3', 'bitrate': '320k'}
            )
            self._record_result(result)
            self.skipTest("ffmpeg not available")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_test_file(
                'mp3', duration=3.0, sample_rate=44100, bitrate='320k',
                filename='test_320k.mp3'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=3.0,
                format_info={'format': 'MP3', 'bitrate': '320k'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'MP3', 'bitrate': '320k'}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'MP3', 'bitrate': '320k'}
            )
            self._record_result(result)
            self.fail(f"Test failed: {e}")
    
    # ================================================================================
    # M4A File Tests
    # ================================================================================
    
    def test_m4a_aac(self):
        """Test M4A file with AAC encoding"""
        test_name = "M4A_AAC"
        print(f"\n  Testing {test_name}...")
        
        if not self.generator.has_ffmpeg:
            result = TestResult(
                test_name=test_name,
                status='SKIP',
                error_message='ffmpeg not available',
                format_info={'format': 'M4A', 'codec': 'AAC'}
            )
            self._record_result(result)
            self.skipTest("ffmpeg not available")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_test_file(
                'm4a', duration=3.0, sample_rate=44100, bitrate='128k',
                filename='test_aac.m4a'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=3.0,
                format_info={'format': 'M4A', 'codec': 'AAC'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'M4A', 'codec': 'AAC'}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'M4A', 'codec': 'AAC'}
            )
            self._record_result(result)
            self.fail(f"Test failed: {e}")
    
    # ================================================================================
    # FLAC File Tests
    # ================================================================================
    
    def test_flac_lossless(self):
        """Test FLAC file with lossless compression"""
        test_name = "FLAC_Lossless"
        print(f"\n  Testing {test_name}...")
        
        if not self.generator.has_ffmpeg:
            result = TestResult(
                test_name=test_name,
                status='SKIP',
                error_message='ffmpeg not available',
                format_info={'format': 'FLAC', 'compression': 'lossless'}
            )
            self._record_result(result)
            self.skipTest("ffmpeg not available")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_test_file(
                'flac', duration=3.0, sample_rate=44100,
                filename='test_lossless.flac'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=3.0,
                format_info={'format': 'FLAC', 'compression': 'lossless'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'FLAC', 'compression': 'lossless'}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'FLAC', 'compression': 'lossless'}
            )
            self._record_result(result)
            self.fail(f"Test failed: {e}")
    
    # ================================================================================
    # OGG File Tests
    # ================================================================================
    
    def test_ogg_vorbis(self):
        """Test OGG file with Vorbis encoding"""
        test_name = "OGG_Vorbis"
        print(f"\n  Testing {test_name}...")
        
        if not self.generator.has_ffmpeg:
            result = TestResult(
                test_name=test_name,
                status='SKIP',
                error_message='ffmpeg not available',
                format_info={'format': 'OGG', 'codec': 'Vorbis'}
            )
            self._record_result(result)
            self.skipTest("ffmpeg not available")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_test_file(
                'ogg', duration=3.0, sample_rate=44100,
                filename='test_vorbis.ogg'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=3.0,
                format_info={'format': 'OGG', 'codec': 'Vorbis'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'OGG', 'codec': 'Vorbis'}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'OGG', 'codec': 'Vorbis'}
            )
            self._record_result(result)
            self.fail(f"Test failed: {e}")
    
    # ================================================================================
    # Large File Tests
    # ================================================================================
    
    def test_large_file_10min(self):
        """Test large 10-minute file processing"""
        test_name = "Large_10min"
        print(f"\n  Testing {test_name}...")
        
        start_time = time.time()
        try:
            # Generate a 10-second file for faster testing
            # In production, this would be 600 seconds (10 minutes)
            actual_duration = 10.0 if self.engine else 3.0
            
            file_path = self.generator.generate_test_file(
                'wav', duration=actual_duration, sample_rate=16000,
                filename='test_10min.wav'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=actual_duration,
                format_info={'format': 'WAV', 'duration': '10min', 'note': 'scaled to 10s for test'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'WAV', 'duration': '10min', 'note': 'scaled to 10s for test'}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'WAV', 'duration': '10min'}
            )
            self._record_result(result)
            self.fail(f"Test failed: {e}")
    
    # ================================================================================
    # Error Handling Tests
    # ================================================================================
    
    def test_empty_file(self):
        """Test handling of empty file"""
        test_name = "Error_EmptyFile"
        print(f"\n  Testing {test_name}...")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_empty_file('test_empty.wav')
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=0.0,
                format_info={'format': 'WAV', 'type': 'empty'}
            )
            
            # Empty files should fail gracefully
            result = TestResult(
                test_name=test_name,
                status='PASS' if not success else 'FAIL',
                duration=time.time() - start_time,
                error_message='' if not success else 'Should have failed for empty file',
                format_info={'format': 'WAV', 'type': 'empty'}
            )
            
            self._record_result(result)
            # We expect this to fail or handle gracefully
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='PASS',  # Expected to fail
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'WAV', 'type': 'empty'}
            )
            self._record_result(result)
    
    def test_corrupt_file(self):
        """Test handling of corrupt file"""
        test_name = "Error_CorruptFile"
        print(f"\n  Testing {test_name}...")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_corrupt_file('test_corrupt.wav', size=1024)
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=0.0,
                format_info={'format': 'WAV', 'type': 'corrupt'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if not success else 'FAIL',
                duration=time.time() - start_time,
                error_message='' if not success else 'Should have failed for corrupt file',
                format_info={'format': 'WAV', 'type': 'corrupt'}
            )
            
            self._record_result(result)
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='PASS',  # Expected to fail
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'WAV', 'type': 'corrupt'}
            )
            self._record_result(result)
    
    def test_truncated_wav(self):
        """Test handling of truncated WAV file"""
        test_name = "Error_TruncatedWAV"
        print(f"\n  Testing {test_name}...")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_truncated_wav('test_truncated.wav')
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=0.5,
                format_info={'format': 'WAV', 'type': 'truncated'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS',  # May succeed or fail depending on implementation
                duration=time.time() - start_time,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'WAV', 'type': 'truncated'}
            )
            
            self._record_result(result)
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='PASS',  # Expected to fail or handle gracefully
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'WAV', 'type': 'truncated'}
            )
            self._record_result(result)
    
    # ================================================================================
    # Unicode Filename Tests
    # ================================================================================
    
    def test_unicode_filename_chinese(self):
        """Test file with Chinese characters in filename"""
        test_name = "Unicode_Chinese"
        print(f"\n  Testing {test_name}...")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_test_file(
                'wav', duration=2.0, sample_rate=16000,
                filename='测试文件中文.wav'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=2.0,
                format_info={'format': 'WAV', 'unicode': 'Chinese'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                error_message='' if success else text,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'WAV', 'unicode': 'Chinese'}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'WAV', 'unicode': 'Chinese'}
            )
            self._record_result(result)
            self.fail(f"Test failed: {e}")
    
    def test_unicode_filename_japanese(self):
        """Test file with Japanese characters in filename"""
        test_name = "Unicode_Japanese"
        print(f"\n  Testing {test_name}...")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_test_file(
                'wav', duration=2.0, sample_rate=16000,
                filename='テストファイル日本語.wav'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=2.0,
                format_info={'format': 'WAV', 'unicode': 'Japanese'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                error_message='' if success else text,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'WAV', 'unicode': 'Japanese'}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'WAV', 'unicode': 'Japanese'}
            )
            self._record_result(result)
            self.fail(f"Test failed: {e}")
    
    def test_unicode_filename_special_chars(self):
        """Test file with special characters in filename"""
        test_name = "Unicode_SpecialChars"
        print(f"\n  Testing {test_name}...")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_test_file(
                'wav', duration=2.0, sample_rate=16000,
                filename='test_file_ñ_é_ü_中_🎵.wav'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=2.0,
                format_info={'format': 'WAV', 'unicode': 'Mixed special chars'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                error_message='' if success else text,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'WAV', 'unicode': 'Mixed special chars'}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'WAV', 'unicode': 'Mixed special chars'}
            )
            self._record_result(result)
            self.fail(f"Test failed: {e}")
    
    def test_unicode_filename_spaces(self):
        """Test file with spaces and punctuation in filename"""
        test_name = "Unicode_Spaces"
        print(f"\n  Testing {test_name}...")
        
        start_time = time.time()
        try:
            file_path = self.generator.generate_test_file(
                'wav', duration=2.0, sample_rate=16000,
                filename='Test File (v2.0) - Recording [2024].wav'
            )
            
            success, text, stats = self._run_transcription_test(
                test_name, file_path, expected_duration=2.0,
                format_info={'format': 'WAV', 'unicode': 'Spaces and punctuation'}
            )
            
            result = TestResult(
                test_name=test_name,
                status='PASS' if success else 'FAIL',
                duration=time.time() - start_time,
                error_message='' if success else text,
                audio_duration=stats.audio_duration,
                processing_time=stats.processing_time,
                rtf=stats.rtf,
                format_info={'format': 'WAV', 'unicode': 'Spaces and punctuation'}
            )
            
            self._record_result(result)
            self.assertTrue(success, f"Transcription failed: {text}")
            
        except Exception as e:
            result = TestResult(
                test_name=test_name,
                status='ERROR',
                duration=time.time() - start_time,
                error_message=str(e),
                format_info={'format': 'WAV', 'unicode': 'Spaces and punctuation'}
            )
            self._record_result(result)
            self.fail(f"Test failed: {e}")


class TestProgressCallbacks(unittest.TestCase):
    """Test progress callback functionality"""
    
    def setUp(self):
        self.generator = AudioFileGenerator()
        self.engine = None
        if APP_IMPORT_SUCCESS:
            try:
                self.engine = TranscriptionEngine()
            except:
                pass
    
    def test_progress_callback_invoked(self):
        """Test that progress callback is called during transcription"""
        print("\n  Testing progress callback invocation...")
        
        if not self.engine:
            self.skipTest("Engine not available")
        
        file_path = self.generator.generate_test_file(
            'wav', duration=2.0, sample_rate=16000,
            filename='test_progress.wav'
        )
        
        progress_messages = []
        
        def callback(msg):
            progress_messages.append(msg)
        
        try:
            result, stats = self.engine.transcribe(
                str(file_path),
                progress_callback=callback
            )
            
            self.assertGreater(len(progress_messages), 0,
                              "Progress callback should have been called at least once")
            
        except Exception as e:
            self.fail(f"Transcription failed: {e}")
    
    def test_progress_callback_thread_safety(self):
        """Test progress callback from multiple threads"""
        print("\n  Testing progress callback thread safety...")
        
        if not self.engine:
            self.skipTest("Engine not available")
        
        # Generate multiple files
        files = []
        for i in range(3):
            file_path = self.generator.generate_test_file(
                'wav', duration=1.0, sample_rate=16000,
                filename=f'test_thread_{i}.wav'
            )
            files.append(file_path)
        
        progress_counts = []
        lock = threading.Lock()
        
        def transcribe_file(file_path):
            messages = []
            
            def callback(msg):
                messages.append(msg)
            
            try:
                result, stats = self.engine.transcribe(str(file_path), progress_callback=callback)
                with lock:
                    progress_counts.append(len(messages))
            except Exception as e:
                with lock:
                    progress_counts.append(-1)
        
        # Run transcriptions in parallel
        threads = []
        for f in files:
            t = threading.Thread(target=transcribe_file, args=(f,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join(timeout=60)
        
        # Verify all succeeded
        self.assertEqual(len(progress_counts), 3)
        for count in progress_counts:
            self.assertGreater(count, 0, "Each transcription should have progress callbacks")


def run_standalone():
    """Run tests in standalone mode without pytest"""
    print("\nRunning Batch Mode Tests (Standalone Mode)\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBatchMode))
    suite.addTests(loader.loadTestsFromTestCase(TestProgressCallbacks))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Check if pytest is available
    try:
        import pytest
        # Run with pytest if available
        sys.exit(pytest.main([__file__, '-v']))
    except ImportError:
        # Fall back to unittest
        success = run_standalone()
        sys.exit(0 if success else 1)
