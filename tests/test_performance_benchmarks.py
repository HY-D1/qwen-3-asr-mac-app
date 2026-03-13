#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         Comprehensive Performance Benchmarks & Load Testing Suite             ║
║         Qwen3-ASR Pro - Real Audio File Performance Analysis                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Test Categories:
1. Transcription Performance Tests
   - Benchmark all real files in test_files/
   - Measure RTF (Real-Time Factor) for each file
   - Compare 0.6b vs 1.7b model performance
   - Generate performance report with graphs

2. File Size vs Performance Analysis
   - Test correlation between file size and processing time
   - Find optimal file size range
   - Identify performance bottlenecks
   - Create size/time scatter plot data

3. Memory Usage Profiling
   - Monitor memory consumption during transcription
   - Test with increasing file sizes
   - Check for memory leaks
   - Document peak memory usage

4. LLM Reforming Performance
   - Benchmark text reforming speed
   - Test with different text lengths
   - Measure Ollama/MLX response times
   - Compare reform modes performance

5. Load Testing
   - Test sequential processing of multiple files
   - Test batch processing
   - Measure throughput (files per minute)
   - Identify saturation points

6. Baseline Establishment
   - Create performance baselines
   - Document expected ranges
   - Set performance regression thresholds
   - Create performance monitoring report

Usage:
    Run all benchmarks:
        python -m pytest tests/test_performance_benchmarks.py -v
    
    Run specific benchmark category:
        python -m pytest tests/test_performance_benchmarks.py::TestTranscriptionPerformance -v
        python -m pytest tests/test_performance_benchmarks.py::TestFileSizePerformance -v
        python -m pytest tests/test_performance_benchmarks.py::TestLLMReformingPerformance -v
    
    Generate performance report:
        python tests/test_performance_benchmarks.py --report

Requirements:
    - pytest
    - psutil (for resource monitoring)
    - numpy (for audio analysis)
    - matplotlib (optional, for graphs)

Author: Performance Testing Suite
Version: 1.0.0
"""

import os
import sys
import time
import gc
import json
import wave
import tempfile
import threading
import subprocess
import queue
import unittest
import warnings
import traceback
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import collections
import statistics

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ==============================================================================
# Test Configuration
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_FILES_DIR = os.path.join(BASE_DIR, "test_files")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
C_ASR_DIR = os.path.join(ASSETS_DIR, "c-asr")
SAMPLES_DIR = os.path.join(C_ASR_DIR, "samples")
RECORDINGS_DIR = os.path.expanduser("~/Documents/Qwen3-ASR-Recordings")
REPORTS_DIR = os.path.join(TEST_DIR, "reports")

MODEL_0_6B = os.path.join(C_ASR_DIR, "qwen3-asr-0.6b")
MODEL_1_7B = os.path.join(C_ASR_DIR, "qwen3-asr-1.7b")
BINARY_PATH = os.path.join(C_ASR_DIR, "qwen_asr")

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.05

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'rtf_max': 3.0,              # Maximum acceptable RTF
    'rtf_target': 1.0,           # Target RTF for good performance
    'rtf_excellent': 0.5,        # Excellent RTF
    'llm_chars_per_sec': 50,     # Minimum LLM processing speed
    'memory_growth_max': 30,     # Max 30% memory growth
    'batch_throughput_min': 5,   # Min 5 files per minute
}

# Regression threshold (alert if performance degrades by more than X%)
REGRESSION_THRESHOLD = 0.20  # 20%

# ==============================================================================
# Optional Dependencies
# ==============================================================================
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Resource monitoring will be limited.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("numpy not available. Audio analysis will be limited.")

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ==============================================================================
# Data Classes
# ==============================================================================
@dataclass
class AudioFileInfo:
    """Information about an audio file"""
    path: str
    name: str
    size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    
    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 * 1024)
    
    @property
    def category(self) -> str:
        """Categorize by duration"""
        if self.duration_seconds < 15:
            return "short"
        elif self.duration_seconds < 60:
            return "medium"
        elif self.duration_seconds < 300:
            return "long"
        else:
            return "very_long"


@dataclass
class TranscriptionBenchmark:
    """Benchmark result for a single transcription"""
    file_info: AudioFileInfo
    model: str
    processing_time: float
    memory_start_mb: float
    memory_peak_mb: float
    memory_end_mb: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def rtf(self) -> float:
        """Real-Time Factor: processing_time / audio_duration"""
        if self.file_info.duration_seconds > 0:
            return self.processing_time / self.file_info.duration_seconds
        return 0.0
    
    @property
    def memory_growth_mb(self) -> float:
        return self.memory_end_mb - self.memory_start_mb
    
    def to_dict(self) -> Dict:
        return {
            'file': self.file_info.name,
            'duration': self.file_info.duration_seconds,
            'size_mb': self.file_info.size_mb,
            'model': self.model,
            'processing_time': self.processing_time,
            'rtf': self.rtf,
            'memory_start_mb': self.memory_start_mb,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_end_mb': self.memory_end_mb,
            'memory_growth_mb': self.memory_growth_mb,
            'timestamp': self.timestamp,
        }


@dataclass
class LLMReformBenchmark:
    """Benchmark result for LLM text reforming"""
    text_length: int
    mode: str
    backend: str
    processing_time: float
    memory_start_mb: float
    memory_peak_mb: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def chars_per_second(self) -> float:
        return self.text_length / self.processing_time if self.processing_time > 0 else 0
    
    def to_dict(self) -> Dict:
        return {
            'text_length': self.text_length,
            'mode': self.mode,
            'backend': self.backend,
            'processing_time': self.processing_time,
            'chars_per_second': self.chars_per_second,
            'memory_start_mb': self.memory_start_mb,
            'memory_peak_mb': self.memory_peak_mb,
        }


@dataclass
class LoadTestResult:
    """Load test result"""
    test_name: str
    total_files: int
    successful_files: int
    failed_files: int
    total_duration: float
    sequential_time: float
    avg_time_per_file: float
    throughput_files_per_min: float
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection"""
    test_name: str
    mean_rtf: float
    std_dev_rtf: float
    mean_memory_mb: float
    timestamp: str
    version: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ==============================================================================
# Utility Classes
# ==============================================================================
class AudioFileAnalyzer:
    """Analyze audio files in test_files directory"""
    
    @staticmethod
    def get_audio_info(filepath: str) -> Optional[AudioFileInfo]:
        """Get information about an audio file"""
        if not os.path.exists(filepath):
            return None
        
        try:
            with wave.open(filepath, 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                channels = wf.getnchannels()
                duration = frames / rate if rate > 0 else 0
                size = os.path.getsize(filepath)
                
                return AudioFileInfo(
                    path=filepath,
                    name=os.path.basename(filepath),
                    size_bytes=size,
                    duration_seconds=duration,
                    sample_rate=rate,
                    channels=channels
                )
        except Exception as e:
            print(f"  Warning: Could not analyze {filepath}: {e}")
            return None
    
    @staticmethod
    def get_all_test_files() -> List[AudioFileInfo]:
        """Get all test audio files"""
        files = []
        if os.path.exists(TEST_FILES_DIR):
            for filename in sorted(os.listdir(TEST_FILES_DIR)):
                if filename.endswith('.wav'):
                    filepath = os.path.join(TEST_FILES_DIR, filename)
                    info = AudioFileAnalyzer.get_audio_info(filepath)
                    if info:
                        files.append(info)
        return sorted(files, key=lambda x: x.duration_seconds)
    
    @staticmethod
    def get_files_by_category(category: str) -> List[AudioFileInfo]:
        """Get files by duration category"""
        all_files = AudioFileAnalyzer.get_all_test_files()
        return [f for f in all_files if f.category == category]


class MemoryMonitor:
    """Monitor memory usage during operations"""
    
    def __init__(self):
        self.snapshots: List[Tuple[float, float]] = []  # (timestamp, rss_mb)
        self.start_time: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
    
    def start(self, interval: float = 0.5):
        """Start monitoring in background"""
        self.start_time = time.perf_counter()
        self.snapshots = []
        self._stop.clear()
        
        if PSUTIL_AVAILABLE:
            self._thread = threading.Thread(target=self._monitor_loop, args=(interval,))
            self._thread.daemon = True
            self._thread.start()
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop"""
        while not self._stop.is_set():
            self.take_snapshot()
            time.sleep(interval)
    
    def stop(self):
        """Stop monitoring"""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.take_snapshot()  # Final snapshot
    
    def take_snapshot(self):
        """Take a memory snapshot"""
        if not PSUTIL_AVAILABLE or not self.process:
            return
        
        try:
            mem_info = self.process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            timestamp = time.perf_counter() - (self.start_time or time.perf_counter())
            self.snapshots.append((timestamp, rss_mb))
        except:
            pass
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics"""
        if not self.snapshots:
            return {'start': 0, 'peak': 0, 'end': 0, 'growth': 0}
        
        rss_values = [s[1] for s in self.snapshots]
        return {
            'start': rss_values[0],
            'peak': max(rss_values),
            'end': rss_values[-1],
            'growth': rss_values[-1] - rss_values[0]
        }


class PerformanceReport:
    """Generate performance reports"""
    
    def __init__(self):
        self.transcription_results: List[TranscriptionBenchmark] = []
        self.llm_results: List[LLMReformBenchmark] = []
        self.load_results: List[LoadTestResult] = []
        self.timestamp = datetime.now().isoformat()
    
    def add_transcription_result(self, result: TranscriptionBenchmark):
        self.transcription_results.append(result)
    
    def add_llm_result(self, result: LLMReformBenchmark):
        self.llm_results.append(result)
    
    def add_load_result(self, result: LoadTestResult):
        self.load_results.append(result)
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        summary = {
            'timestamp': self.timestamp,
            'system': self._get_system_info(),
            'transcription': self._summarize_transcription(),
            'llm_reforming': self._summarize_llm(),
            'load_testing': self._summarize_load(),
        }
        return summary
    
    def _get_system_info(self) -> Dict[str, str]:
        """Get system information"""
        import platform
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
        }
        
        if PSUTIL_AVAILABLE:
            mem = psutil.virtual_memory()
            info['total_memory_gb'] = f"{mem.total / (1024**3):.1f}"
            info['available_memory_gb'] = f"{mem.available / (1024**3):.1f}"
        
        return info
    
    def _summarize_transcription(self) -> Dict[str, Any]:
        """Summarize transcription results"""
        if not self.transcription_results:
            return {}
        
        rtfs = [r.rtf for r in self.transcription_results]
        times = [r.processing_time for r in self.transcription_results]
        memories = [r.memory_peak_mb for r in self.transcription_results]
        
        # Group by model
        by_model = {}
        for r in self.transcription_results:
            if r.model not in by_model:
                by_model[r.model] = []
            by_model[r.model].append(r)
        
        model_summary = {}
        for model, results in by_model.items():
            model_rtfs = [r.rtf for r in results]
            model_summary[model] = {
                'count': len(results),
                'mean_rtf': statistics.mean(model_rtfs),
                'median_rtf': statistics.median(model_rtfs),
                'min_rtf': min(model_rtfs),
                'max_rtf': max(model_rtfs),
            }
        
        # Group by category
        by_category = {}
        for r in self.transcription_results:
            cat = r.file_info.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(r.rtf)
        
        category_summary = {}
        for cat, rtfs in by_category.items():
            category_summary[cat] = {
                'count': len(rtfs),
                'mean_rtf': statistics.mean(rtfs),
                'median_rtf': statistics.median(rtfs),
            }
        
        return {
            'total_files': len(self.transcription_results),
            'mean_rtf': statistics.mean(rtfs),
            'median_rtf': statistics.median(rtfs),
            'std_rtf': statistics.stdev(rtfs) if len(rtfs) > 1 else 0,
            'min_rtf': min(rtfs),
            'max_rtf': max(rtfs),
            'mean_processing_time': statistics.mean(times),
            'mean_memory_peak_mb': statistics.mean(memories),
            'by_model': model_summary,
            'by_category': category_summary,
        }
    
    def _summarize_llm(self) -> Dict[str, Any]:
        """Summarize LLM results"""
        if not self.llm_results:
            return {}
        
        speeds = [r.chars_per_second for r in self.llm_results]
        
        # Group by mode
        by_mode = {}
        for r in self.llm_results:
            if r.mode not in by_mode:
                by_mode[r.mode] = []
            by_mode[r.mode].append(r.chars_per_second)
        
        mode_summary = {}
        for mode, speeds in by_mode.items():
            mode_summary[mode] = {
                'count': len(speeds),
                'mean_chars_per_sec': statistics.mean(speeds),
                'median_chars_per_sec': statistics.median(speeds),
            }
        
        return {
            'total_tests': len(self.llm_results),
            'mean_chars_per_sec': statistics.mean(speeds),
            'median_chars_per_sec': statistics.median(speeds),
            'by_mode': mode_summary,
        }
    
    def _summarize_load(self) -> Dict[str, Any]:
        """Summarize load test results"""
        if not self.load_results:
            return {}
        
        throughputs = [r.throughput_files_per_min for r in self.load_results]
        
        return {
            'total_tests': len(self.load_results),
            'mean_throughput': statistics.mean(throughputs),
            'max_throughput': max(throughputs),
            'min_throughput': min(throughputs),
        }
    
    def save_json(self, filepath: str):
        """Save report as JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.generate_summary(), f, indent=2)
    
    def generate_markdown(self) -> str:
        """Generate markdown report"""
        summary = self.generate_summary()
        
        md = f"""# Performance Benchmark Report

**Generated:** {self.timestamp}

## System Information

| Property | Value |
|----------|-------|
"""
        for key, value in summary['system'].items():
            md += f"| {key} | {value} |\n"
        
        # Transcription results
        if 'transcription' in summary and summary['transcription']:
            t = summary['transcription']
            md += f"""
## Transcription Performance

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total Files | {t['total_files']} |
| Mean RTF | {t['mean_rtf']:.3f}x |
| Median RTF | {t['median_rtf']:.3f}x |
| Min RTF | {t['min_rtf']:.3f}x |
| Max RTF | {t['max_rtf']:.3f}x |
| Std Dev RTF | {t['std_rtf']:.3f}x |
| Mean Processing Time | {t['mean_processing_time']:.2f}s |
| Mean Peak Memory | {t['mean_memory_peak_mb']:.1f} MB |

### By Model

"""
            for model, stats in t.get('by_model', {}).items():
                md += f"""#### {model}

| Metric | Value |
|--------|-------|
| Files Tested | {stats['count']} |
| Mean RTF | {stats['mean_rtf']:.3f}x |
| Median RTF | {stats['median_rtf']:.3f}x |
| Min RTF | {stats['min_rtf']:.3f}x |
| Max RTF | {stats['max_rtf']:.3f}x |

"""
            
            md += "### By Duration Category\n\n"
            md += "| Category | Count | Mean RTF | Median RTF |\n"
            md += "|----------|-------|----------|------------|\n"
            for cat, stats in t.get('by_category', {}).items():
                md += f"| {cat} | {stats['count']} | {stats['mean_rtf']:.3f}x | {stats['median_rtf']:.3f}x |\n"
        
        # LLM results
        if 'llm_reforming' in summary and summary['llm_reforming']:
            l = summary['llm_reforming']
            md += f"""
## LLM Reforming Performance

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total Tests | {l['total_tests']} |
| Mean Speed | {l['mean_chars_per_sec']:.1f} chars/sec |
| Median Speed | {l['median_chars_per_sec']:.1f} chars/sec |

### By Mode

| Mode | Count | Mean (chars/sec) | Median (chars/sec) |
|------|-------|------------------|---------------------|
"""
            for mode, stats in l.get('by_mode', {}).items():
                md += f"| {mode} | {stats['count']} | {stats['mean_chars_per_sec']:.1f} | {stats['median_chars_per_sec']:.1f} |\n"
        
        # Load test results
        if 'load_testing' in summary and summary['load_testing']:
            load = summary['load_testing']
            md += f"""
## Load Testing Results

| Metric | Value |
|--------|-------|
| Total Tests | {load['total_tests']} |
| Mean Throughput | {load['mean_throughput']:.1f} files/min |
| Max Throughput | {load['max_throughput']:.1f} files/min |
| Min Throughput | {load['min_throughput']:.1f} files/min |
"""
        
        return md
    
    def save_markdown(self, filepath: str):
        """Save markdown report"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.generate_markdown())


# ==============================================================================
# Test Suite: Transcription Performance
# ==============================================================================
@pytest.mark.skipif(not os.path.exists(BINARY_PATH), reason="C binary not available")
class TestTranscriptionPerformance:
    """Transcription performance benchmarks using real audio files"""
    
    @pytest.fixture(scope="class")
    def test_files(self):
        """Get all test files"""
        return AudioFileAnalyzer.get_all_test_files()
    
    @pytest.fixture(scope="class")
    def report(self):
        """Get shared report instance"""
        return PerformanceReport()
    
    def test_01_list_all_test_files(self, test_files):
        """List and categorize all test files"""
        print("\n" + "="*80)
        print("TEST FILES ANALYSIS")
        print("="*80)
        
        categories = {'short': [], 'medium': [], 'long': [], 'very_long': []}
        for f in test_files:
            categories[f.category].append(f)
        
        for cat, files in categories.items():
            if files:
                print(f"\n{cat.upper()} ({len(files)} files):")
                for f in files:
                    print(f"  {f.name}: {f.duration_seconds:.1f}s ({f.size_mb:.1f}MB)")
        
        print(f"\nTotal: {len(test_files)} files")
        total_duration = sum(f.duration_seconds for f in test_files)
        print(f"Total duration: {total_duration:.1f}s ({total_duration/60:.1f}min)")
        print("="*80)
        
        assert len(test_files) > 0, "No test files found"
    
    def test_02_benchmark_all_files_0_6b(self, test_files, report):
        """Benchmark all files with 0.6B model"""
        if not os.path.exists(MODEL_0_6B):
            pytest.skip("0.6B model not available")
        
        print("\n" + "="*80)
        print("BENCHMARKING ALL FILES - 0.6B MODEL")
        print("="*80)
        print(f"{'File':<40} {'Duration':<10} {'Time':<10} {'RTF':<8} {'Peak MB':<10}")
        print("-"*80)
        
        results = []
        for file_info in test_files[:10]:  # Limit to first 10 for time
            monitor = MemoryMonitor()
            monitor.start(interval=0.5)
            
            try:
                start = time.perf_counter()
                cmd = [BINARY_PATH, "-d", MODEL_0_6B, "-i", file_info.path, "--silent"]
                subprocess.run(cmd, capture_output=True, timeout=300)
                elapsed = time.perf_counter() - start
            finally:
                monitor.stop()
            
            mem_stats = monitor.get_stats()
            
            benchmark = TranscriptionBenchmark(
                file_info=file_info,
                model="0.6B",
                processing_time=elapsed,
                memory_start_mb=mem_stats['start'],
                memory_peak_mb=mem_stats['peak'],
                memory_end_mb=mem_stats['end']
            )
            
            results.append(benchmark)
            report.add_transcription_result(benchmark)
            
            print(f"{file_info.name[:38]:<40} {file_info.duration_seconds:>6.1f}s  "
                  f"{elapsed:>6.2f}s  {benchmark.rtf:>6.2f}x  {mem_stats['peak']:>6.0f}MB")
        
        print("="*80)
        
        # Assertions
        for r in results:
            assert r.rtf < PERFORMANCE_THRESHOLDS['rtf_max'], \
                f"RTF {r.rtf:.2f} exceeds max {PERFORMANCE_THRESHOLDS['rtf_max']} for {r.file_info.name}"
    
    def test_03_benchmark_sample_1_7b(self, test_files, report):
        """Benchmark sample files with 1.7B model"""
        if not os.path.exists(MODEL_1_7B):
            pytest.skip("1.7B model not available")
        
        # Select representative samples from each category
        short_files = [f for f in test_files if f.category == 'short'][:2]
        medium_files = [f for f in test_files if f.category == 'medium'][:2]
        
        test_subset = short_files + medium_files
        
        if not test_subset:
            pytest.skip("No suitable test files")
        
        print("\n" + "="*80)
        print("BENCHMARKING SAMPLE FILES - 1.7B MODEL")
        print("="*80)
        print(f"{'File':<40} {'Duration':<10} {'Time':<10} {'RTF':<8} {'Peak MB':<10}")
        print("-"*80)
        
        results = []
        for file_info in test_subset:
            monitor = MemoryMonitor()
            monitor.start(interval=0.5)
            
            try:
                start = time.perf_counter()
                cmd = [BINARY_PATH, "-d", MODEL_1_7B, "-i", file_info.path, "--silent"]
                subprocess.run(cmd, capture_output=True, timeout=300)
                elapsed = time.perf_counter() - start
            finally:
                monitor.stop()
            
            mem_stats = monitor.get_stats()
            
            benchmark = TranscriptionBenchmark(
                file_info=file_info,
                model="1.7B",
                processing_time=elapsed,
                memory_start_mb=mem_stats['start'],
                memory_peak_mb=mem_stats['peak'],
                memory_end_mb=mem_stats['end']
            )
            
            results.append(benchmark)
            report.add_transcription_result(benchmark)
            
            print(f"{file_info.name[:38]:<40} {file_info.duration_seconds:>6.1f}s  "
                  f"{elapsed:>6.2f}s  {benchmark.rtf:>6.2f}x  {mem_stats['peak']:>6.0f}MB")
        
        print("="*80)
        
        # Assertions
        for r in results:
            assert r.rtf < PERFORMANCE_THRESHOLDS['rtf_max'], \
                f"RTF {r.rtf:.2f} exceeds max for {r.file_info.name}"
    
    def test_04_model_comparison(self, test_files, report):
        """Compare 0.6B vs 1.7B model performance"""
        if not os.path.exists(MODEL_0_6B) or not os.path.exists(MODEL_1_7B):
            pytest.skip("Both models required for comparison")
        
        # Select a representative medium file
        medium_files = [f for f in test_files if f.category == 'medium']
        if not medium_files:
            pytest.skip("No medium files for comparison")
        
        test_file = medium_files[0]
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(f"Test file: {test_file.name} ({test_file.duration_seconds:.1f}s)")
        print("-"*80)
        
        results = {}
        for model_name, model_path in [("0.6B", MODEL_0_6B), ("1.7B", MODEL_1_7B)]:
            # Run 3 times for averaging
            times = []
            for _ in range(3):
                start = time.perf_counter()
                cmd = [BINARY_PATH, "-d", model_path, "-i", test_file.path, "--silent"]
                subprocess.run(cmd, capture_output=True, timeout=300)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
            
            mean_time = statistics.mean(times)
            rtf = mean_time / test_file.duration_seconds
            results[model_name] = {'mean_time': mean_time, 'rtf': rtf, 'times': times}
            
            print(f"{model_name}: {mean_time:.2f}s (RTF: {rtf:.2f}x)")
        
        # Compare
        if '0.6B' in results and '1.7B' in results:
            speedup = results['1.7B']['mean_time'] / results['0.6B']['mean_time']
            print(f"\n0.6B is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than 1.7B")
            
            # 1.7B should be slower but still within threshold
            assert results['1.7B']['rtf'] < PERFORMANCE_THRESHOLDS['rtf_max'], \
                "1.7B RTF exceeds threshold"
        
        print("="*80)


# ==============================================================================
# Test Suite: File Size vs Performance
# ==============================================================================
@pytest.mark.skipif(not os.path.exists(BINARY_PATH), reason="C binary not available")
class TestFileSizePerformance:
    """Analyze correlation between file size and processing time"""
    
    @pytest.fixture(scope="class")
    def test_files(self):
        return AudioFileAnalyzer.get_all_test_files()
    
    def test_01_size_time_correlation(self, test_files):
        """Test correlation between file size and processing time"""
        if not os.path.exists(MODEL_0_6B):
            pytest.skip("Model not available")
        
        print("\n" + "="*80)
        print("FILE SIZE VS PROCESSING TIME")
        print("="*80)
        
        # Select diverse samples
        samples = []
        for cat in ['short', 'medium', 'long']:
            cat_files = [f for f in test_files if f.category == cat]
            if cat_files:
                samples.extend(cat_files[:3])
        
        results = []
        print(f"{'Size (MB)':<12} {'Duration (s)':<14} {'Time (s)':<12} {'RTF':<10}")
        print("-"*80)
        
        for file_info in samples:
            start = time.perf_counter()
            cmd = [BINARY_PATH, "-d", MODEL_0_6B, "-i", file_info.path, "--silent"]
            subprocess.run(cmd, capture_output=True, timeout=300)
            elapsed = time.perf_counter() - start
            
            rtf = elapsed / file_info.duration_seconds if file_info.duration_seconds > 0 else 0
            results.append({
                'size_mb': file_info.size_mb,
                'duration': file_info.duration_seconds,
                'time': elapsed,
                'rtf': rtf
            })
            
            print(f"{file_info.size_mb:>8.2f}     {file_info.duration_seconds:>10.1f}    "
                  f"{elapsed:>8.2f}    {rtf:>8.2f}x")
        
        # Calculate correlation
        if len(results) >= 3 and NUMPY_AVAILABLE:
            sizes = [r['size_mb'] for r in results]
            times = [r['time'] for r in results]
            
            # Simple correlation coefficient
            mean_size = statistics.mean(sizes)
            mean_time = statistics.mean(times)
            
            numerator = sum((s - mean_size) * (t - mean_time) for s, t in zip(sizes, times))
            denom_size = math.sqrt(sum((s - mean_size) ** 2 for s in sizes))
            denom_time = math.sqrt(sum((t - mean_time) ** 2 for t in times))
            
            if denom_size > 0 and denom_time > 0:
                correlation = numerator / (denom_size * denom_time)
                print(f"\nSize-Time Correlation: {correlation:.3f}")
                print(f"  (1.0 = perfect positive, -1.0 = perfect negative, 0 = no correlation)")
        
        print("="*80)
    
    def test_02_optimal_file_size_range(self, test_files):
        """Find optimal file size range for transcription"""
        print("\n" + "="*80)
        print("OPTIMAL FILE SIZE ANALYSIS")
        print("="*80)
        
        # Group by size ranges
        ranges = {
            'small': (0, 1),      # 0-1 MB
            'medium': (1, 5),     # 1-5 MB
            'large': (5, 15),     # 5-15 MB
            'very_large': (15, float('inf'))  # >15 MB
        }
        
        for range_name, (min_size, max_size) in ranges.items():
            files_in_range = [
                f for f in test_files 
                if min_size <= f.size_mb < max_size or (max_size == float('inf') and f.size_mb >= min_size)
            ]
            
            total_duration = sum(f.duration_seconds for f in files_in_range)
            print(f"{range_name:12s}: {len(files_in_range):3d} files, "
                  f"total {total_duration:.1f}s ({total_duration/60:.1f}min)")
        
        print("\nRecommendation: Medium files (1-5MB) offer good balance of "
              "processing time and context for transcription.")
        print("="*80)
    
    def test_03_performance_bottlenecks(self, test_files):
        """Identify performance bottlenecks"""
        if not os.path.exists(MODEL_0_6B):
            pytest.skip("Model not available")
        
        print("\n" + "="*80)
        print("PERFORMANCE BOTTLENECK ANALYSIS")
        print("="*80)
        
        # Test very long file if available
        very_long = [f for f in test_files if f.category == 'very_long']
        
        if very_long:
            test_file = very_long[0]
            print(f"\nTesting very long file: {test_file.name}")
            print(f"  Size: {test_file.size_mb:.1f} MB")
            print(f"  Duration: {test_file.duration_seconds:.1f}s ({test_file.duration_seconds/60:.1f}min)")
            
            monitor = MemoryMonitor()
            monitor.start(interval=1.0)
            
            try:
                start = time.perf_counter()
                cmd = [BINARY_PATH, "-d", MODEL_0_6B, "-i", test_file.path]
                result = subprocess.run(cmd, capture_output=True, timeout=600)
                elapsed = time.perf_counter() - start
            finally:
                monitor.stop()
            
            mem_stats = monitor.get_stats()
            rtf = elapsed / test_file.duration_seconds
            
            print(f"\nResults:")
            print(f"  Processing time: {elapsed:.1f}s")
            print(f"  RTF: {rtf:.2f}x")
            print(f"  Peak memory: {mem_stats['peak']:.0f} MB")
            print(f"  Memory growth: {mem_stats['growth']:.0f} MB")
            
            if rtf > PERFORMANCE_THRESHOLDS['rtf_max']:
                print(f"\n⚠️  WARNING: RTF {rtf:.2f}x exceeds threshold "
                      f"({PERFORMANCE_THRESHOLDS['rtf_max']}x)")
                print("    Consider splitting files >5min for better performance")
        
        print("="*80)


# ==============================================================================
# Test Suite: Memory Usage
# ==============================================================================
@pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
class TestMemoryUsage:
    """Memory usage profiling during transcription"""
    
    @pytest.fixture(scope="class")
    def test_files(self):
        return AudioFileAnalyzer.get_all_test_files()
    
    def test_01_memory_with_increasing_sizes(self, test_files):
        """Test memory consumption with increasing file sizes"""
        if not os.path.exists(MODEL_0_6B):
            pytest.skip("Model not available")
        
        print("\n" + "="*80)
        print("MEMORY USAGE WITH INCREASING FILE SIZES")
        print("="*80)
        
        # Select one file from each category
        test_files_selected = []
        for cat in ['short', 'medium', 'long']:
            cat_files = [f for f in test_files if f.category == cat]
            if cat_files:
                test_files_selected.append(cat_files[0])
        
        print(f"{'Category':<12} {'Duration':<12} {'Baseline':<12} {'Peak':<12} {'Growth':<12}")
        print("-"*80)
        
        for file_info in test_files_selected:
            # Force GC before test
            gc.collect()
            time.sleep(0.5)
            
            monitor = MemoryMonitor()
            monitor.start(interval=0.5)
            
            try:
                cmd = [BINARY_PATH, "-d", MODEL_0_6B, "-i", file_info.path, "--silent"]
                subprocess.run(cmd, capture_output=True, timeout=300)
            finally:
                monitor.stop()
            
            mem_stats = monitor.get_stats()
            growth_pct = (mem_stats['growth'] / mem_stats['start'] * 100) if mem_stats['start'] > 0 else 0
            
            print(f"{file_info.category:<12} {file_info.duration_seconds:>8.1f}s   "
                  f"{mem_stats['start']:>8.0f}MB   {mem_stats['peak']:>8.0f}MB   "
                  f"{mem_stats['growth']:>6.0f}MB ({growth_pct:>4.1f}%)")
            
            # Assert reasonable memory growth
            assert growth_pct < PERFORMANCE_THRESHOLDS['memory_growth_max'], \
                f"Memory growth {growth_pct:.1f}% exceeds threshold"
        
        print("="*80)
    
    def test_02_memory_leak_detection(self, test_files):
        """Detect memory leaks over repeated processing"""
        if not os.path.exists(MODEL_0_6B):
            pytest.skip("Model not available")
        
        short_files = [f for f in test_files if f.category == 'short']
        if not short_files:
            pytest.skip("No short files available")
        
        test_file = short_files[0]
        
        print("\n" + "="*80)
        print("MEMORY LEAK DETECTION")
        print("="*80)
        print(f"Processing {test_file.name} 10 times...")
        
        memory_readings = []
        
        for i in range(10):
            gc.collect()
            
            # Get baseline
            if PSUTIL_AVAILABLE:
                baseline = psutil.Process().memory_info().rss / (1024 * 1024)
            else:
                baseline = 0
            
            cmd = [BINARY_PATH, "-d", MODEL_0_6B, "-i", test_file.path, "--silent"]
            subprocess.run(cmd, capture_output=True, timeout=60)
            
            # Get after
            if PSUTIL_AVAILABLE:
                after = psutil.Process().memory_info().rss / (1024 * 1024)
            else:
                after = 0
            
            memory_readings.append(after)
            
            if i % 3 == 0:
                print(f"  Iteration {i+1}: {after:.0f} MB")
        
        # Check for upward trend
        if len(memory_readings) >= 4:
            first_half = statistics.mean(memory_readings[:5])
            second_half = statistics.mean(memory_readings[5:])
            increase = second_half - first_half
            increase_pct = (increase / first_half * 100) if first_half > 0 else 0
            
            print(f"\nFirst 5 avg: {first_half:.0f} MB")
            print(f"Last 5 avg:  {second_half:.0f} MB")
            print(f"Increase:    {increase:.0f} MB ({increase_pct:.1f}%)")
            
            # Allow some growth but not excessive
            assert increase_pct < 50, f"Potential memory leak: {increase_pct:.1f}% increase"
        
        print("="*80)


# ==============================================================================
# Test Suite: LLM Reforming Performance
# ==============================================================================
class TestLLMReformingPerformance:
    """LLM text reforming performance benchmarks"""
    
    @pytest.fixture(scope="class")
    def reformer(self):
        """Initialize text reformer if available"""
        try:
            from text_reformer import TextReformer, ReformMode
            reformer = TextReformer()
            if reformer.is_available():
                return reformer
        except ImportError:
            pass
        return None
    
    def test_01_reformer_availability(self, reformer):
        """Check if LLM reformer is available"""
        if reformer is None:
            pytest.skip("TextReformer not available")
        print("\n✓ TextReformer is available")
    
    def test_02_benchmark_by_text_length(self, reformer):
        """Benchmark with different text lengths"""
        if reformer is None:
            pytest.skip("TextReformer not available")
        
        print("\n" + "="*80)
        print("LLM REFORMING - TEXT LENGTH BENCHMARK")
        print("="*80)
        
        # Generate test texts of different lengths
        base_text = "This is a test sentence for measuring LLM performance. "
        lengths = [100, 500, 1000]
        
        print(f"{'Length':<10} {'Time (s)':<12} {'Chars/sec':<12} {'Status':<10}")
        print("-"*80)
        
        for length in lengths:
            # Generate text of approximate length
            repetitions = max(1, length // len(base_text))
            test_text = (base_text * repetitions)[:length]
            
            monitor = MemoryMonitor()
            monitor.start(interval=0.2)
            
            try:
                start = time.perf_counter()
                result = reformer.reform(test_text, "punctuate")
                elapsed = time.perf_counter() - start
                success = result is not None and len(result.reformed_text) > 0
            except Exception as e:
                elapsed = time.perf_counter() - start
                success = False
            finally:
                monitor.stop()
            
            mem_stats = monitor.get_stats()
            chars_per_sec = length / elapsed if elapsed > 0 else 0
            status = "✓" if success else "✗"
            
            print(f"{length:<10} {elapsed:>8.3f}     {chars_per_sec:>8.1f}     {status:<10}")
            
            # Performance assertion
            if success:
                assert chars_per_sec >= PERFORMANCE_THRESHOLDS['llm_chars_per_sec'] * 0.5, \
                    f"LLM speed {chars_per_sec:.1f} chars/sec too slow"
        
        print("="*80)
    
    def test_03_benchmark_by_mode(self, reformer):
        """Benchmark different reform modes"""
        if reformer is None:
            pytest.skip("TextReformer not available")
        
        print("\n" + "="*80)
        print("LLM REFORMING - MODE COMPARISON")
        print("="*80)
        
        test_text = "This is a test sentence. " * 20  # ~500 chars
        
        try:
            from text_reformer import ReformMode
            modes = ["punctuate", "summarize", "clean"]
        except:
            modes = ["punctuate", "summarize", "clean"]
        
        print(f"{'Mode':<15} {'Time (s)':<12} {'Chars/sec':<12} {'Status':<10}")
        print("-"*80)
        
        for mode in modes:
            try:
                start = time.perf_counter()
                result = reformer.reform(test_text, mode)
                elapsed = time.perf_counter() - start
                success = result is not None
            except Exception as e:
                elapsed = time.perf_counter() - start
                success = False
            
            chars_per_sec = len(test_text) / elapsed if elapsed > 0 else 0
            status = "✓" if success else "✗"
            
            print(f"{mode:<15} {elapsed:>8.3f}     {chars_per_sec:>8.1f}     {status:<10}")
        
        print("="*80)


# ==============================================================================
# Test Suite: Load Testing
# ==============================================================================
@pytest.mark.skipif(not os.path.exists(BINARY_PATH), reason="C binary not available")
class TestLoadTesting:
    """Load testing with real audio files"""
    
    @pytest.fixture(scope="class")
    def test_files(self):
        return AudioFileAnalyzer.get_all_test_files()
    
    def test_01_sequential_processing(self, test_files):
        """Test sequential processing of multiple files"""
        if not os.path.exists(MODEL_0_6B):
            pytest.skip("Model not available")
        
        # Use short files for faster testing
        short_files = [f for f in test_files if f.category == 'short'][:5]
        
        if len(short_files) < 3:
            pytest.skip("Not enough short files")
        
        print("\n" + "="*80)
        print("SEQUENTIAL PROCESSING LOAD TEST")
        print("="*80)
        print(f"Processing {len(short_files)} files sequentially...")
        
        start_time = time.perf_counter()
        
        for file_info in short_files:
            cmd = [BINARY_PATH, "-d", MODEL_0_6B, "-i", file_info.path, "--silent"]
            subprocess.run(cmd, capture_output=True, timeout=120)
        
        total_time = time.perf_counter() - start_time
        throughput = len(short_files) / (total_time / 60)  # files per minute
        
        print(f"\nResults:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Throughput: {throughput:.1f} files/min")
        print(f"  Avg per file: {total_time/len(short_files):.2f}s")
        
        print("="*80)
    
    def test_02_batch_processing_simulation(self, test_files):
        """Simulate batch processing with concurrency"""
        if not os.path.exists(MODEL_0_6B):
            pytest.skip("Model not available")
        
        short_files = [f for f in test_files if f.category == 'short'][:6]
        
        if len(short_files) < 4:
            pytest.skip("Not enough short files")
        
        print("\n" + "="*80)
        print("BATCH PROCESSING SIMULATION")
        print("="*80)
        print(f"Processing {len(short_files)} files with concurrency=2...")
        
        def process_file(file_info):
            start = time.perf_counter()
            try:
                cmd = [BINARY_PATH, "-d", MODEL_0_6B, "-i", file_info.path, "--silent"]
                result = subprocess.run(cmd, capture_output=True, timeout=120)
                success = result.returncode == 0
            except Exception as e:
                success = False
            elapsed = time.perf_counter() - start
            return elapsed, success
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_file, f) for f in short_files]
            results = [f.result() for f in as_completed(futures)]
        
        total_time = time.perf_counter() - start_time
        successful = sum(1 for _, success in results if success)
        avg_time = sum(t for t, _ in results) / len(results) if results else 0
        throughput = len(short_files) / (total_time / 60)
        
        print(f"\nResults:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Successful: {successful}/{len(short_files)}")
        print(f"  Avg time per file: {avg_time:.2f}s")
        print(f"  Throughput: {throughput:.1f} files/min")
        
        assert successful == len(short_files), "All batch files should process successfully"
        assert throughput >= PERFORMANCE_THRESHOLDS['batch_throughput_min'], \
            f"Throughput {throughput:.1f} below minimum {PERFORMANCE_THRESHOLDS['batch_throughput_min']}"
        
        print("="*80)
    
    def test_03_saturation_point(self, test_files):
        """Identify saturation point with increasing concurrency"""
        if not os.path.exists(MODEL_0_6B):
            pytest.skip("Model not available")
        
        short_files = [f for f in test_files if f.category == 'short'][:6]
        
        if len(short_files) < 4:
            pytest.skip("Not enough short files")
        
        print("\n" + "="*80)
        print("SATURATION POINT ANALYSIS")
        print("="*80)
        
        def process_file(file_info):
            start = time.perf_counter()
            try:
                cmd = [BINARY_PATH, "-d", MODEL_0_6B, "-i", file_info.path, "--silent"]
                subprocess.run(cmd, capture_output=True, timeout=120)
            except:
                pass
            return time.perf_counter() - start
        
        print(f"{'Concurrency':<15} {'Total Time':<15} {'Throughput':<15}")
        print("-"*80)
        
        for concurrency in [1, 2]:
            start = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(process_file, f) for f in short_files]
                list(as_completed(futures))
            
            total_time = time.perf_counter() - start
            throughput = len(short_files) / (total_time / 60)
            
            print(f"{concurrency:<15} {total_time:>10.1f}s     {throughput:>10.1f}/min")
        
        print("\nNote: Higher concurrency may not improve throughput due to model loading overhead.")
        print("="*80)


# ==============================================================================
# Test Suite: Baseline Establishment
# ==============================================================================
class TestBaselineEstablishment:
    """Establish and compare performance baselines"""
    
    BASELINE_FILE = os.path.join(TEST_DIR, "performance_baselines.json")
    
    @classmethod
    def load_baselines(cls) -> Dict[str, Any]:
        """Load existing baselines"""
        if os.path.exists(cls.BASELINE_FILE):
            with open(cls.BASELINE_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    @classmethod
    def save_baselines(cls, baselines: Dict[str, Any]):
        """Save baselines to file"""
        with open(cls.BASELINE_FILE, 'w') as f:
            json.dump(baselines, f, indent=2)
    
    def test_01_check_existing_baselines(self):
        """Check for existing performance baselines"""
        baselines = self.load_baselines()
        
        print("\n" + "="*80)
        print("PERFORMANCE BASELINES")
        print("="*80)
        
        if baselines:
            print(f"Existing baselines found ({len(baselines)} tests)")
            for test_name, data in baselines.items():
                print(f"  {test_name}: RTF={data.get('mean_rtf', 'N/A')}")
        else:
            print("No existing baselines. Run with --save-baseline to create.")
        
        print("="*80)
    
    def test_02_establish_baseline(self):
        """Establish new performance baseline"""
        if not os.path.exists(BINARY_PATH) or not os.path.exists(MODEL_0_6B):
            pytest.skip("Prerequisites not available")
        
        test_files = AudioFileAnalyzer.get_all_test_files()
        short_files = [f for f in test_files if f.category == 'short']
        
        if not short_files:
            pytest.skip("No short files available")
        
        print("\n" + "="*80)
        print("ESTABLISHING NEW BASELINE")
        print("="*80)
        
        # Use first short file for baseline
        test_file = short_files[0]
        
        # Run multiple times for stable baseline
        rtfs = []
        for i in range(3):
            start = time.perf_counter()
            cmd = [BINARY_PATH, "-d", MODEL_0_6B, "-i", test_file.path, "--silent"]
            subprocess.run(cmd, capture_output=True, timeout=60)
            elapsed = time.perf_counter() - start
            rtf = elapsed / test_file.duration_seconds
            rtfs.append(rtf)
        
        mean_rtf = statistics.mean(rtfs)
        std_rtf = statistics.stdev(rtfs) if len(rtfs) > 1 else 0
        
        baseline = {
            'baseline_short_file': {
                'mean_rtf': mean_rtf,
                'std_dev_rtf': std_rtf,
                'test_file': test_file.name,
                'duration': test_file.duration_seconds,
                'timestamp': datetime.now().isoformat(),
            }
        }
        
        print(f"New baseline established:")
        print(f"  Mean RTF: {mean_rtf:.3f}x")
        print(f"  Std Dev:  {std_rtf:.3f}x")
        print(f"  File:     {test_file.name}")
        
        # Save baseline
        existing = self.load_baselines()
        existing.update(baseline)
        self.save_baselines(existing)
        
        print(f"\nBaseline saved to: {self.BASELINE_FILE}")
        print("="*80)
    
    def test_03_regression_detection(self):
        """Detect performance regression against baseline"""
        baselines = self.load_baselines()
        
        if 'baseline_short_file' not in baselines:
            pytest.skip("No baseline available for comparison")
        
        if not os.path.exists(BINARY_PATH) or not os.path.exists(MODEL_0_6B):
            pytest.skip("Prerequisites not available")
        
        baseline = baselines['baseline_short_file']
        baseline_rtf = baseline['mean_rtf']
        
        print("\n" + "="*80)
        print("REGRESSION DETECTION")
        print("="*80)
        
        # Run current test
        test_file_path = os.path.join(TEST_FILES_DIR, baseline['test_file'])
        if not os.path.exists(test_file_path):
            pytest.skip("Original test file not found")
        
        rtfs = []
        for i in range(3):
            start = time.perf_counter()
            cmd = [BINARY_PATH, "-d", MODEL_0_6B, "-i", test_file_path, "--silent"]
            subprocess.run(cmd, capture_output=True, timeout=60)
            elapsed = time.perf_counter() - start
            
            file_info = AudioFileAnalyzer.get_audio_info(test_file_path)
            rtf = elapsed / file_info.duration_seconds if file_info else 0
            rtfs.append(rtf)
        
        current_rtf = statistics.mean(rtfs)
        
        # Calculate regression
        regression = (current_rtf - baseline_rtf) / baseline_rtf if baseline_rtf > 0 else 0
        
        print(f"Baseline RTF: {baseline_rtf:.3f}x")
        print(f"Current RTF:  {current_rtf:.3f}x")
        print(f"Regression:   {regression*100:+.1f}%")
        print(f"Threshold:    {REGRESSION_THRESHOLD*100:.0f}%")
        
        if regression > REGRESSION_THRESHOLD:
            print(f"\n⚠️  WARNING: Performance regression detected!")
            print(f"    RTF increased by {regression*100:.1f}% (threshold: {REGRESSION_THRESHOLD*100:.0f}%)")
        elif regression < -0.1:
            print(f"\n✓ Performance improved by {abs(regression)*100:.1f}%")
        else:
            print(f"\n✓ Performance within acceptable range")
        
        assert regression < REGRESSION_THRESHOLD, \
            f"Performance regression: {regression*100:.1f}%"
        
        print("="*80)


# ==============================================================================
# Main Entry Point for Report Generation
# ==============================================================================
def generate_performance_report():
    """Generate comprehensive performance report"""
    print("\n" + "="*80)
    print("GENERATING PERFORMANCE REPORT")
    print("="*80)
    
    # Create reports directory
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Initialize report
    report = PerformanceReport()
    
    # Get test files
    test_files = AudioFileAnalyzer.get_all_test_files()
    
    # Run quick benchmarks
    if os.path.exists(BINARY_PATH) and os.path.exists(MODEL_0_6B):
        print("\nRunning transcription benchmarks...")
        for file_info in test_files[:5]:  # Sample 5 files
            print(f"  {file_info.name}...", end=" ", flush=True)
            
            monitor = MemoryMonitor()
            monitor.start(interval=0.5)
            
            try:
                start = time.perf_counter()
                cmd = [BINARY_PATH, "-d", MODEL_0_6B, "-i", file_info.path, "--silent"]
                subprocess.run(cmd, capture_output=True, timeout=300)
                elapsed = time.perf_counter() - start
            finally:
                monitor.stop()
            
            mem_stats = monitor.get_stats()
            
            benchmark = TranscriptionBenchmark(
                file_info=file_info,
                model="0.6B",
                processing_time=elapsed,
                memory_start_mb=mem_stats['start'],
                memory_peak_mb=mem_stats['peak'],
                memory_end_mb=mem_stats['end']
            )
            report.add_transcription_result(benchmark)
            print(f"RTF={benchmark.rtf:.2f}x")
    
    # Save reports
    json_path = os.path.join(REPORTS_DIR, f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    md_path = os.path.join(REPORTS_DIR, f"PERFORMANCE_REPORT.md")
    
    report.save_json(json_path)
    report.save_markdown(md_path)
    
    print(f"\nReports saved:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    
    # Print summary
    summary = report.generate_summary()
    if 'transcription' in summary:
        t = summary['transcription']
        print(f"\nSummary:")
        print(f"  Files tested: {t.get('total_files', 0)}")
        print(f"  Mean RTF: {t.get('mean_rtf', 0):.3f}x")
        print(f"  Median RTF: {t.get('median_rtf', 0):.3f}x")
    
    print("="*80)
    return report


if __name__ == "__main__":
    if "--report" in sys.argv:
        generate_performance_report()
    else:
        # Run with pytest
        import pytest
        pytest.main([__file__, "-v"] + sys.argv[1:])
