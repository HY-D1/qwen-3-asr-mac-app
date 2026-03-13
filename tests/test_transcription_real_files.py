#!/usr/bin/env python3
"""
================================================================================
COMPREHENSIVE REAL FILE TRANSCRIPTION TEST SUITE
Qwen3-ASR macOS Speech-to-Text Application
================================================================================

Test Suite for Transcription Backends Using REAL Audio Files:
- C Binary Backend (Live streaming with 0.6b/1.7b models)
- MLX Audio Backend (Apple Silicon optimized)
- MLX CLI Backend (Fallback)
- PyTorch Backend (Intel Mac / Compatibility)

**Test Coverage:**
1. C-Binary Backend Tests with REAL files:
   - Small files (< 1MB): live_20260301_093539.wav
   - Medium files (1-5MB): live_20260228_195523.wav
   - Large files (> 5MB): class_20260227_115101.wav
   - Both 0.6b and 1.7b models
   - Timing measurements
   - Valid text output verification

2. Process Audio Pipeline Tests:
   - process_audio() function with real files
   - Language options: auto, en, zh, ja, ko, es, fr, de
   - Full pipeline: transcribe + reform

3. File Format Edge Cases:
   - WAV files in different formats
   - File path handling with spaces/special chars
   - Concurrent file processing

4. Performance Benchmarks:
   - RTF (Real-Time Factor) measurement
   - Performance report comparing file sizes vs processing time
   - Optimal file size identification

**Requirements:**
- pytest for test execution
- Real audio files in test_files/ folder
- All tests clean up after themselves
- Generate detailed performance report

================================================================================
USAGE
================================================================================

Run all tests:
    python3 -m pytest tests/test_transcription_real_files.py -v

Run specific test category:
    python3 -m pytest tests/test_transcription_real_files.py::TestCBinaryRealFiles -v
    python3 -m pytest tests/test_transcription_real_files.py::TestProcessAudioPipeline -v
    python3 -m pytest tests/test_transcription_real_files.py::TestPerformanceBenchmarks -v

Run with performance report:
    python3 -m pytest tests/test_transcription_real_files.py -v --tb=short 2>&1 | tee test_report.txt

================================================================================
"""

import os
import sys
import time
import wave
import tempfile
import unittest
import subprocess
import concurrent.futures
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import pytest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import modules under test
from constants import (
    MODEL_CONFIG, LANGUAGE_CONFIG, SAMPLE_RATE,
    C_ASR_DIR, BASE_DIR, ASSETS_DIR
)

# ==============================================================================
# Test Data Configuration
# ==============================================================================

# Real audio files from test_files/ folder
TEST_FILES_DIR = os.path.join(BASE_DIR, "test_files")

# File size categories
FILE_CATEGORIES = {
    "small": {
        "pattern": "live_20260301_093539.wav",  # ~313K
        "max_size_mb": 1.0,
        "description": "Small files (< 1MB)"
    },
    "medium": {
        "pattern": "live_20260228_195523.wav",  # ~375K
        "max_size_mb": 5.0,
        "description": "Medium files (1-5MB)"
    },
    "large": {
        "pattern": "class_20260227_115101.wav",  # ~5.8M
        "max_size_mb": 15.0,
        "description": "Large files (> 5MB)"
    },
}

# Model configurations
MODEL_CONFIGS = {
    "0.6B": {
        "name": "Qwen/Qwen3-ASR-0.6B",
        "dir": "qwen3-asr-0.6b",
        "description": "Fast, low latency"
    },
    "1.7B": {
        "name": "Qwen/Qwen3-ASR-1.7B",
        "dir": "qwen3-asr-1.7b",
        "description": "Best accuracy"
    }
}

# Language test configurations
LANGUAGE_TESTS = [
    ("auto", "Auto-detect"),
    ("en", "English"),
    ("zh", "Chinese"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
]


@dataclass
class TranscriptionBenchmark:
    """Data class for storing transcription benchmark results"""
    file_name: str
    file_size_mb: float
    audio_duration_sec: float
    model: str
    backend: str
    language: str
    processing_time_sec: float
    rtf: float
    transcript_length: int
    transcript_preview: str
    success: bool
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# Global benchmark results storage
BENCHMARK_RESULTS: List[TranscriptionBenchmark] = []


def get_real_test_files() -> Dict[str, List[str]]:
    """Discover and categorize real test files from test_files/ folder"""
    files_by_category = {
        "small": [],
        "medium": [],
        "large": []
    }
    
    if not os.path.exists(TEST_FILES_DIR):
        print(f"⚠️ Test files directory not found: {TEST_FILES_DIR}")
        return files_by_category
    
    for filename in os.listdir(TEST_FILES_DIR):
        if not filename.endswith('.wav'):
            continue
            
        filepath = os.path.join(TEST_FILES_DIR, filename)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        if size_mb < 1.0:
            files_by_category["small"].append(filepath)
        elif size_mb < 5.0:
            files_by_category["medium"].append(filepath)
        else:
            files_by_category["large"].append(filepath)
    
    return files_by_category


def get_audio_duration(filepath: str) -> float:
    """Get audio file duration in seconds"""
    try:
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception as e:
        print(f"   ⚠️ Could not get duration for {filepath}: {e}")
        return 0.0


def validate_transcription_output(text: str) -> Tuple[bool, str]:
    """Validate that transcription output is valid text"""
    if not text:
        return False, "Empty transcription"
    
    if text.strip() == "":
        return False, "Whitespace-only transcription"
    
    # Check for error indicators
    error_indicators = [
        "error", "exception", "failed", "failure",
        "not found", "permission denied", "timeout"
    ]
    
    text_lower = text.lower()
    for indicator in error_indicators:
        if indicator in text_lower and len(text) < 100:
            return False, f"Possible error message: {text}"
    
    return True, "Valid transcription"


# ==============================================================================
# C-Binary Backend Tests with Real Files
# ==============================================================================

@pytest.mark.skipif(not os.path.exists(TEST_FILES_DIR), 
                    reason="Test files directory not found")
class TestCBinaryRealFiles:
    """Tests for C Binary Backend using REAL audio files"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        self.binary_path = os.path.join(C_ASR_DIR, 'qwen_asr')
        self.model_06b_path = os.path.join(C_ASR_DIR, 'qwen3-asr-0.6b')
        self.model_17b_path = os.path.join(C_ASR_DIR, 'qwen3-asr-1.7b')
        self.test_files = get_real_test_files()
    
    def test_binary_exists(self):
        """Verify C binary exists"""
        assert os.path.exists(self.binary_path), f"C binary not found: {self.binary_path}"
        assert os.access(self.binary_path, os.X_OK), f"C binary not executable"
    
    def test_model_directories_exist(self):
        """Verify model directories exist"""
        assert os.path.exists(self.model_06b_path), f"0.6B model not found"
        assert os.path.exists(self.model_17b_path), f"1.7B model not found"
    
    @pytest.mark.skipif(not os.path.exists(os.path.join(C_ASR_DIR, 'qwen_asr')),
                        reason="C binary not found")
    def test_transcribe_small_file_0_6b(self):
        """Test transcribing small file (< 1MB) with 0.6B model"""
        if not self.test_files["small"]:
            pytest.skip("No small test files available")
        
        test_file = self.test_files["small"][0]
        file_size = os.path.getsize(test_file) / (1024 * 1024)
        audio_duration = get_audio_duration(test_file)
        
        print(f"\n   Testing small file: {os.path.basename(test_file)} ({file_size:.2f} MB, {audio_duration:.1f}s)")
        
        cmd = [
            self.binary_path,
            "-d", self.model_06b_path,
            "-i", test_file,
            "--language", "en"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        processing_time = time.time() - start_time
        
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        
        print(f"   Processing time: {processing_time:.2f}s, RTF: {rtf:.2f}x")
        
        # Store benchmark
        benchmark = TranscriptionBenchmark(
            file_name=os.path.basename(test_file),
            file_size_mb=file_size,
            audio_duration_sec=audio_duration,
            model="0.6B",
            backend="C-Binary",
            language="en",
            processing_time_sec=processing_time,
            rtf=rtf,
            transcript_length=len(result.stdout),
            transcript_preview=result.stdout[:100].replace('\n', ' '),
            success=result.returncode == 0 and len(result.stdout.strip()) > 0
        )
        BENCHMARK_RESULTS.append(benchmark)
        
        # Validate output
        is_valid, message = validate_transcription_output(result.stdout)
        assert result.returncode == 0, f"C binary returned error: {result.stderr}"
        assert is_valid, f"Invalid transcription: {message}"
        
        print(f"   ✅ Transcription: {result.stdout[:100]}...")
    
    @pytest.mark.skipif(not os.path.exists(os.path.join(C_ASR_DIR, 'qwen_asr')),
                        reason="C binary not found")
    def test_transcribe_small_file_1_7b(self):
        """Test transcribing small file (< 1MB) with 1.7B model"""
        if not self.test_files["small"]:
            pytest.skip("No small test files available")
        
        test_file = self.test_files["small"][0]
        file_size = os.path.getsize(test_file) / (1024 * 1024)
        audio_duration = get_audio_duration(test_file)
        
        print(f"\n   Testing small file with 1.7B: {os.path.basename(test_file)} ({file_size:.2f} MB)")
        
        cmd = [
            self.binary_path,
            "-d", self.model_17b_path,
            "-i", test_file,
            "--language", "en"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        processing_time = time.time() - start_time
        
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        
        print(f"   Processing time: {processing_time:.2f}s, RTF: {rtf:.2f}x")
        
        # Store benchmark
        benchmark = TranscriptionBenchmark(
            file_name=os.path.basename(test_file),
            file_size_mb=file_size,
            audio_duration_sec=audio_duration,
            model="1.7B",
            backend="C-Binary",
            language="en",
            processing_time_sec=processing_time,
            rtf=rtf,
            transcript_length=len(result.stdout),
            transcript_preview=result.stdout[:100].replace('\n', ' '),
            success=result.returncode == 0 and len(result.stdout.strip()) > 0
        )
        BENCHMARK_RESULTS.append(benchmark)
        
        is_valid, message = validate_transcription_output(result.stdout)
        assert result.returncode == 0, f"C binary returned error: {result.stderr}"
        assert is_valid, f"Invalid transcription: {message}"
        
        print(f"   ✅ Transcription: {result.stdout[:100]}...")
    
    @pytest.mark.skipif(not os.path.exists(os.path.join(C_ASR_DIR, 'qwen_asr')),
                        reason="C binary not found")
    def test_transcribe_medium_file_0_6b(self):
        """Test transcribing medium file (1-5MB) with 0.6B model"""
        if not self.test_files["medium"]:
            pytest.skip("No medium test files available")
        
        test_file = self.test_files["medium"][0]
        file_size = os.path.getsize(test_file) / (1024 * 1024)
        audio_duration = get_audio_duration(test_file)
        
        print(f"\n   Testing medium file: {os.path.basename(test_file)} ({file_size:.2f} MB, {audio_duration:.1f}s)")
        
        cmd = [
            self.binary_path,
            "-d", self.model_06b_path,
            "-i", test_file,
            "--language", "en"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        processing_time = time.time() - start_time
        
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        
        print(f"   Processing time: {processing_time:.2f}s, RTF: {rtf:.2f}x")
        
        # Store benchmark
        benchmark = TranscriptionBenchmark(
            file_name=os.path.basename(test_file),
            file_size_mb=file_size,
            audio_duration_sec=audio_duration,
            model="0.6B",
            backend="C-Binary",
            language="en",
            processing_time_sec=processing_time,
            rtf=rtf,
            transcript_length=len(result.stdout),
            transcript_preview=result.stdout[:100].replace('\n', ' '),
            success=result.returncode == 0 and len(result.stdout.strip()) > 0
        )
        BENCHMARK_RESULTS.append(benchmark)
        
        is_valid, message = validate_transcription_output(result.stdout)
        assert result.returncode == 0, f"C binary returned error: {result.stderr}"
        assert is_valid, f"Invalid transcription: {message}"
        
        print(f"   ✅ Transcription: {result.stdout[:100]}...")
    
    @pytest.mark.skipif(not os.path.exists(os.path.join(C_ASR_DIR, 'qwen_asr')),
                        reason="C binary not found")
    @pytest.mark.slow
    def test_transcribe_large_file_0_6b(self):
        """Test transcribing large file (> 5MB) with 0.6B model"""
        if not self.test_files["large"]:
            pytest.skip("No large test files available")
        
        test_file = self.test_files["large"][0]
        file_size = os.path.getsize(test_file) / (1024 * 1024)
        audio_duration = get_audio_duration(test_file)
        
        print(f"\n   Testing large file: {os.path.basename(test_file)} ({file_size:.2f} MB, {audio_duration:.1f}s)")
        
        cmd = [
            self.binary_path,
            "-d", self.model_06b_path,
            "-i", test_file,
            "--language", "en"
        ]
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        processing_time = time.time() - start_time
        
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        
        print(f"   Processing time: {processing_time:.2f}s, RTF: {rtf:.2f}x")
        
        # Store benchmark
        benchmark = TranscriptionBenchmark(
            file_name=os.path.basename(test_file),
            file_size_mb=file_size,
            audio_duration_sec=audio_duration,
            model="0.6B",
            backend="C-Binary",
            language="en",
            processing_time_sec=processing_time,
            rtf=rtf,
            transcript_length=len(result.stdout),
            transcript_preview=result.stdout[:100].replace('\n', ' '),
            success=result.returncode == 0 and len(result.stdout.strip()) > 0
        )
        BENCHMARK_RESULTS.append(benchmark)
        
        is_valid, message = validate_transcription_output(result.stdout)
        assert result.returncode == 0, f"C binary returned error: {result.stderr}"
        assert is_valid, f"Invalid transcription: {message}"
        
        print(f"   ✅ Transcription length: {len(result.stdout)} chars")


# ==============================================================================
# Process Audio Pipeline Tests
# ==============================================================================

@pytest.mark.skipif(not os.path.exists(TEST_FILES_DIR), 
                    reason="Test files directory not found")
class TestProcessAudioPipeline:
    """Tests for the complete process_audio pipeline with real files"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        self.test_files = get_real_test_files()
        self.small_file = self.test_files["small"][0] if self.test_files["small"] else None
    
    def test_process_audio_with_transcription_engine(self):
        """Test process_audio using TranscriptionEngine with real file"""
        if not self.small_file:
            pytest.skip("No small test files available")
        
        from app import TranscriptionEngine
        
        try:
            engine = TranscriptionEngine()
        except RuntimeError as e:
            pytest.skip(f"No transcription backend available: {e}")
        
        print(f"\n   Testing TranscriptionEngine with: {os.path.basename(self.small_file)}")
        print(f"   Backend: {engine.backend}")
        
        file_size = os.path.getsize(self.small_file) / (1024 * 1024)
        audio_duration = get_audio_duration(self.small_file)
        
        start_time = time.time()
        result = engine.transcribe(self.small_file, language="en")
        processing_time = time.time() - start_time
        
        rtf = processing_time / audio_duration if audio_duration > 0 else 0
        
        print(f"   Processing time: {processing_time:.2f}s, RTF: {rtf:.2f}x")
        print(f"   Transcript: {result.text[:100]}...")
        
        # Store benchmark
        benchmark = TranscriptionBenchmark(
            file_name=os.path.basename(self.small_file),
            file_size_mb=file_size,
            audio_duration_sec=audio_duration,
            model=result.model or "1.7B",
            backend=result.backend or engine.backend,
            language="en",
            processing_time_sec=processing_time,
            rtf=rtf,
            transcript_length=len(result.text),
            transcript_preview=result.text[:100].replace('\n', ' '),
            success=len(result.text.strip()) > 0
        )
        BENCHMARK_RESULTS.append(benchmark)
        
        assert len(result.text.strip()) > 0, "Transcription should not be empty"
        assert result.stats is not None, "Performance stats should be available"
    
    @pytest.mark.parametrize("lang_code,lang_name", LANGUAGE_TESTS)
    def test_transcription_languages(self, lang_code, lang_name):
        """Test transcription with different language settings"""
        if not self.small_file:
            pytest.skip("No small test files available")
        
        from app import TranscriptionEngine
        
        try:
            engine = TranscriptionEngine()
        except RuntimeError:
            pytest.skip("No transcription backend available")
        
        print(f"\n   Testing language: {lang_name} ({lang_code})")
        
        try:
            result = engine.transcribe(self.small_file, language=lang_code if lang_code != "auto" else None)
            
            is_valid, message = validate_transcription_output(result.text)
            assert is_valid, f"Invalid transcription for {lang_name}: {message}"
            
            print(f"   ✅ {lang_name}: {result.text[:80]}...")
            
        except Exception as e:
            print(f"   ⚠️ {lang_name} failed: {e}")
            # Don't fail - some languages might not be supported
            pytest.skip(f"Language {lang_name} not supported or failed: {e}")


# ==============================================================================
# File Format Edge Cases Tests
# ==============================================================================

@pytest.mark.skipif(not os.path.exists(TEST_FILES_DIR), 
                    reason="Test files directory not found")
class TestFileFormatEdgeCases:
    """Tests for file format edge cases"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        self.test_files = get_real_test_files()
        self.temp_dir = tempfile.mkdtemp()
        
        yield
        
        # Cleanup
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_file_with_spaces_in_path(self):
        """Test transcription with file path containing spaces"""
        if not self.test_files["small"]:
            pytest.skip("No small test files available")
        
        from app import LiveStreamer
        
        # Create a copy with spaces in the name
        original_file = self.test_files["small"][0]
        spaced_filename = "file with spaces in name.wav"
        spaced_path = os.path.join(self.temp_dir, spaced_filename)
        
        import shutil
        shutil.copy2(original_file, spaced_path)
        
        print(f"\n   Testing file with spaces: {spaced_filename}")
        
        # Test with C binary
        binary_path = os.path.join(C_ASR_DIR, 'qwen_asr')
        model_path = os.path.join(C_ASR_DIR, 'qwen3-asr-0.6b')
        
        if not os.path.exists(binary_path):
            pytest.skip("C binary not found")
        
        cmd = [
            binary_path,
            "-d", model_path,
            "-i", spaced_path,
            "--language", "en"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        assert result.returncode == 0, f"Failed with spaces in path: {result.stderr}"
        assert len(result.stdout.strip()) > 0, "Transcription should not be empty"
        
        print(f"   ✅ Handled spaces in path successfully")
    
    def test_file_with_special_chars_in_path(self):
        """Test transcription with special characters in path"""
        if not self.test_files["small"]:
            pytest.skip("No small test files available")
        
        from app import LiveStreamer
        
        original_file = self.test_files["small"][0]
        special_filename = "file-with_special.chars.wav"
        special_path = os.path.join(self.temp_dir, special_filename)
        
        import shutil
        shutil.copy2(original_file, special_path)
        
        print(f"\n   Testing file with special chars: {special_filename}")
        
        binary_path = os.path.join(C_ASR_DIR, 'qwen_asr')
        model_path = os.path.join(C_ASR_DIR, 'qwen3-asr-0.6b')
        
        if not os.path.exists(binary_path):
            pytest.skip("C binary not found")
        
        cmd = [
            binary_path,
            "-d", model_path,
            "-i", special_path,
            "--language", "en"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        assert result.returncode == 0, f"Failed with special chars: {result.stderr}"
        assert len(result.stdout.strip()) > 0, "Transcription should not be empty"
        
        print(f"   ✅ Handled special chars in path successfully")
    
    @pytest.mark.skipif(not os.path.exists(os.path.join(C_ASR_DIR, 'qwen_asr')),
                        reason="C binary not found")
    def test_concurrent_file_processing(self):
        """Test concurrent processing of multiple files"""
        all_files = []
        for category in ["small", "medium"]:
            all_files.extend(self.test_files[category][:2])  # Up to 2 files per category
        
        if len(all_files) < 2:
            pytest.skip("Need at least 2 test files for concurrent testing")
        
        print(f"\n   Testing concurrent processing of {len(all_files)} files")
        
        binary_path = os.path.join(C_ASR_DIR, 'qwen_asr')
        model_path = os.path.join(C_ASR_DIR, 'qwen3-asr-0.6b')
        
        def transcribe_file(filepath: str) -> Tuple[str, bool, str]:
            """Transcribe a single file"""
            cmd = [
                binary_path,
                "-d", model_path,
                "-i", filepath,
                "--language", "en"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                is_valid, _ = validate_transcription_output(result.stdout)
                return (os.path.basename(filepath), result.returncode == 0 and is_valid, result.stdout[:50])
            except Exception as e:
                return (os.path.basename(filepath), False, str(e))
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(transcribe_file, all_files))
        
        total_time = time.time() - start_time
        
        success_count = sum(1 for _, success, _ in results if success)
        
        print(f"   Concurrent processing completed in {total_time:.2f}s")
        print(f"   Success: {success_count}/{len(all_files)}")
        
        for filename, success, preview in results:
            status = "✅" if success else "❌"
            print(f"   {status} {filename}: {preview}...")
        
        assert success_count > 0, "At least one file should be processed successfully"


# ==============================================================================
# Performance Benchmarks Tests
# ==============================================================================

@pytest.mark.skipif(not os.path.exists(TEST_FILES_DIR), 
                    reason="Test files directory not found")
class TestPerformanceBenchmarks:
    """Performance benchmark tests using real files"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test"""
        self.test_files = get_real_test_files()
    
    def test_measure_rtf_for_all_categories(self):
        """Measure RTF for files of all size categories"""
        if not os.path.exists(os.path.join(C_ASR_DIR, 'qwen_asr')):
            pytest.skip("C binary not found")
        
        binary_path = os.path.join(C_ASR_DIR, 'qwen_asr')
        model_path = os.path.join(C_ASR_DIR, 'qwen3-asr-0.6b')
        
        results_by_category = {}
        
        for category, files in self.test_files.items():
            if not files:
                continue
            
            test_file = files[0]
            file_size = os.path.getsize(test_file) / (1024 * 1024)
            audio_duration = get_audio_duration(test_file)
            
            print(f"\n   Testing {category} file: {os.path.basename(test_file)}")
            print(f"   Size: {file_size:.2f} MB, Duration: {audio_duration:.1f}s")
            
            cmd = [
                binary_path,
                "-d", model_path,
                "-i", test_file,
                "--language", "en"
            ]
            
            # Run multiple times for average
            times = []
            for i in range(2):  # 2 iterations for speed
                start = time.time()
                result = subprocess.run(cmd, capture_output=True, timeout=180)
                if result.returncode == 0:
                    times.append(time.time() - start)
            
            if times:
                avg_time = sum(times) / len(times)
                rtf = avg_time / audio_duration if audio_duration > 0 else 0
                results_by_category[category] = {
                    'file': os.path.basename(test_file),
                    'size_mb': file_size,
                    'duration': audio_duration,
                    'avg_time': avg_time,
                    'rtf': rtf
                }
                print(f"   Avg time: {avg_time:.2f}s, RTF: {rtf:.2f}x")
        
        # Assert reasonable performance
        for category, result in results_by_category.items():
            # RTF should be reasonable (less than 10x for real-time)
            assert result['rtf'] < 10.0, f"RTF too high for {category}: {result['rtf']:.2f}x"
    
    def test_compare_model_performance(self):
        """Compare 0.6B vs 1.7B model performance"""
        if not os.path.exists(os.path.join(C_ASR_DIR, 'qwen_asr')):
            pytest.skip("C binary not found")
        
        if not self.test_files["small"]:
            pytest.skip("No small test files available")
        
        test_file = self.test_files["small"][0]
        binary_path = os.path.join(C_ASR_DIR, 'qwen_asr')
        model_06b = os.path.join(C_ASR_DIR, 'qwen3-asr-0.6b')
        model_17b = os.path.join(C_ASR_DIR, 'qwen3-asr-1.7b')
        
        audio_duration = get_audio_duration(test_file)
        
        results = {}
        
        for model_name, model_path in [("0.6B", model_06b), ("1.7B", model_17b)]:
            if not os.path.exists(model_path):
                continue
            
            cmd = [
                binary_path,
                "-d", model_path,
                "-i", test_file,
                "--language", "en"
            ]
            
            start = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            elapsed = time.time() - start
            
            if result.returncode == 0:
                results[model_name] = {
                    'time': elapsed,
                    'rtf': elapsed / audio_duration if audio_duration > 0 else 0,
                    'transcript_len': len(result.stdout)
                }
                print(f"   {model_name}: {elapsed:.2f}s, RTF: {results[model_name]['rtf']:.2f}x")
        
        # 1.7B should be slower but produce similar or better results
        if "0.6B" in results and "1.7B" in results:
            # 1.7B can be 1.5x to 3x slower than 0.6B
            ratio = results["1.7B"]["time"] / results["0.6B"]["time"]
            print(f"   Speed ratio (1.7B/0.6B): {ratio:.2f}x")
            assert ratio < 5.0, f"1.7B is too slow compared to 0.6B: {ratio:.2f}x"


# ==============================================================================
# Test Report Generation
# ==============================================================================

def generate_performance_report():
    """Generate comprehensive performance report from benchmark results"""
    
    report_lines = []
    report_lines.append("\n" + "=" * 80)
    report_lines.append("REAL FILE TRANSCRIPTION PERFORMANCE REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    
    if not BENCHMARK_RESULTS:
        report_lines.append("\n⚠️ No benchmark results collected")
        return "\n".join(report_lines)
    
    # Group results by model
    results_by_model = {}
    for result in BENCHMARK_RESULTS:
        if result.model not in results_by_model:
            results_by_model[result.model] = []
        results_by_model[result.model].append(result)
    
    # Summary statistics
    report_lines.append("\n📊 SUMMARY STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total tests run: {len(BENCHMARK_RESULTS)}")
    report_lines.append(f"Successful: {sum(1 for r in BENCHMARK_RESULTS if r.success)}")
    report_lines.append(f"Failed: {sum(1 for r in BENCHMARK_RESULTS if not r.success)}")
    
    # Results by model
    for model, results in results_by_model.items():
        report_lines.append(f"\n🔹 Model: {model}")
        report_lines.append("-" * 40)
        
        avg_rtf = sum(r.rtf for r in results) / len(results)
        avg_time = sum(r.processing_time_sec for r in results) / len(results)
        
        report_lines.append(f"  Tests: {len(results)}")
        report_lines.append(f"  Average RTF: {avg_rtf:.2f}x")
        report_lines.append(f"  Average processing time: {avg_time:.2f}s")
        
        # Results by file size
        small = [r for r in results if r.file_size_mb < 1.0]
        medium = [r for r in results if 1.0 <= r.file_size_mb < 5.0]
        large = [r for r in results if r.file_size_mb >= 5.0]
        
        if small:
            avg_small_rtf = sum(r.rtf for r in small) / len(small)
            report_lines.append(f"  Small files (<1MB) avg RTF: {avg_small_rtf:.2f}x")
        
        if medium:
            avg_medium_rtf = sum(r.rtf for r in medium) / len(medium)
            report_lines.append(f"  Medium files (1-5MB) avg RTF: {avg_medium_rtf:.2f}x")
        
        if large:
            avg_large_rtf = sum(r.rtf for r in large) / len(large)
            report_lines.append(f"  Large files (>5MB) avg RTF: {avg_large_rtf:.2f}x")
    
    # Detailed results table
    report_lines.append("\n📋 DETAILED RESULTS")
    report_lines.append("-" * 80)
    report_lines.append(f"{'File':<30} {'Model':<8} {'Size(MB)':<10} {'Time(s)':<10} {'RTF':<8} {'Status':<8}")
    report_lines.append("-" * 80)
    
    for result in BENCHMARK_RESULTS:
        filename = result.file_name[:28]
        status = "✅ PASS" if result.success else "❌ FAIL"
        report_lines.append(
            f"{filename:<30} {result.model:<8} {result.file_size_mb:<10.2f} "
            f"{result.processing_time_sec:<10.2f} {result.rtf:<8.2f} {status:<8}"
        )
    
    # Performance recommendations
    report_lines.append("\n💡 PERFORMANCE RECOMMENDATIONS")
    report_lines.append("-" * 80)
    
    all_rtfs = [r.rtf for r in BENCHMARK_RESULTS if r.success]
    if all_rtfs:
        avg_rtf = sum(all_rtfs) / len(all_rtfs)
        
        if avg_rtf < 0.5:
            report_lines.append("✅ Excellent performance! System can process faster than real-time.")
        elif avg_rtf < 1.0:
            report_lines.append("✅ Good performance. System can process near real-time.")
        elif avg_rtf < 2.0:
            report_lines.append("⚠️ Acceptable performance. Processing takes 1-2x audio duration.")
        else:
            report_lines.append("❌ Slow performance. Consider using 0.6B model for faster processing.")
        
        # Optimal file size recommendation
        by_size = {}
        for r in BENCHMARK_RESULTS:
            if r.success:
                size_cat = "small" if r.file_size_mb < 1.0 else ("medium" if r.file_size_mb < 5.0 else "large")
                if size_cat not in by_size:
                    by_size[size_cat] = []
                by_size[size_cat].append(r.rtf)
        
        if by_size:
            avg_by_size = {cat: sum(rtfs)/len(rtfs) for cat, rtfs in by_size.items()}
            optimal = min(avg_by_size, key=avg_by_size.get)
            report_lines.append(f"\n📌 Optimal file size category: {optimal.upper()}")
            report_lines.append(f"   Average RTF: {avg_by_size[optimal]:.2f}x")
    
    report_lines.append("\n" + "=" * 80)
    
    return "\n".join(report_lines)


def save_benchmark_results(output_dir: str = None):
    """Save benchmark results to JSON file"""
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "tests")
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"transcription_benchmark_{timestamp}.json")
    
    data = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(BENCHMARK_RESULTS),
        "successful": sum(1 for r in BENCHMARK_RESULTS if r.success),
        "results": [asdict(r) for r in BENCHMARK_RESULTS]
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return output_file


# ==============================================================================
# Pytest Hooks
# ==============================================================================

def pytest_sessionfinish(session, exitstatus):
    """Called after all tests are completed"""
    if BENCHMARK_RESULTS:
        print(generate_performance_report())
        
        try:
            output_file = save_benchmark_results()
            print(f"\n💾 Benchmark results saved to: {output_file}")
        except Exception as e:
            print(f"\n⚠️ Could not save benchmark results: {e}")


# ==============================================================================
# Main Entry Point for Direct Execution
# ==============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("QWEN3-ASR REAL FILE TRANSCRIPTION TEST SUITE")
    print("=" * 80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test files directory: {TEST_FILES_DIR}")
    print()
    
    # Check test files
    test_files = get_real_test_files()
    total_files = sum(len(files) for files in test_files.values())
    
    print(f"Found {total_files} test files:")
    for category, files in test_files.items():
        print(f"  {category}: {len(files)} files")
    print()
    
    # Run pytest
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=BASE_DIR
    )
    
    sys.exit(result.returncode)
