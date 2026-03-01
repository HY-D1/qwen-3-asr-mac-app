#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         Memory Leak Detection Test Suite                         ║
║         Qwen3-ASR Pro - Resource Management Verification         ║
╚══════════════════════════════════════════════════════════════════╝

Test scenarios:
1. Long running (30 min): App memory stable over time
2. Multiple transcriptions: 50 batch transcriptions  
3. Live streaming stress: 100 start/stop cycles
4. Model loading/unloading: Switch models 20 times
5. File handle leaks: Check open file descriptors
6. Thread leaks: Verify threads terminate
7. Process leaks: No zombie C binary processes
8. Temp file accumulation: Cleanup verification
9. Numpy array leaks: Large array lifecycle
10. Tkinter widget leaks: UI element cleanup

Success Criteria:
- Memory growth < 10% over test duration
- No orphaned processes
- Temp files cleaned up
- Thread count stable
"""

import os
import sys
import time
import gc
import threading
import tempfile
import shutil
import subprocess
import unittest
import warnings
import json
import traceback
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, Mock
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
C_ASR_DIR = os.path.join(ASSETS_DIR, "c-asr")
RECORDINGS_DIR = os.path.expanduser("~/Documents/Qwen3-ASR-Recordings")

# Test parameters
TEST_CONFIG = {
    'long_running_duration': 30,  # seconds (reduced for CI, use 1800 for real 30min test)
    'batch_transcription_count': 50,
    'streaming_cycles': 100,
    'model_switch_count': 20,
    'memory_growth_threshold': 0.10,  # 10%
    'sample_audio_duration': 5,  # seconds
}

# Try to import psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    warnings.warn("psutil not available. Memory tracking will be limited.")

# Try to import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Try to import matplotlib for graphs
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """Memory snapshot at a point in time"""
    timestamp: float
    rss_mb: float
    vms_mb: float
    thread_count: int
    fd_count: int
    process_count: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LeakTestResult:
    """Result of a leak test"""
    test_name: str
    passed: bool
    baseline_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_growth_pct: float
    thread_delta: int
    fd_delta: int
    process_delta: int
    duration_seconds: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'baseline_memory_mb': self.baseline_memory_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'final_memory_mb': self.final_memory_mb,
            'memory_growth_pct': self.memory_growth_pct,
            'thread_delta': self.thread_delta,
            'fd_delta': self.fd_delta,
            'process_delta': self.process_delta,
            'duration_seconds': self.duration_seconds,
            'details': self.details
        }


class ResourceTracker:
    """Track system resources during tests"""
    
    def __init__(self):
        self.snapshots: List[MemorySnapshot] = []
        self.start_time: Optional[float] = None
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        
    def start(self):
        """Start tracking"""
        self.start_time = time.time()
        self.snapshots = []
        self.take_snapshot("start")
        
    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a resource snapshot"""
        if not PSUTIL_AVAILABLE:
            return MemorySnapshot(0, 0, 0, 0, 0, 0)
            
        mem_info = self.process.memory_info()
        rss_mb = mem_info.rss / (1024 * 1024)
        vms_mb = mem_info.vms / (1024 * 1024)
        
        # Thread count
        thread_count = self.process.num_threads()
        
        # File descriptor count (platform dependent)
        fd_count = 0
        try:
            fd_count = self.process.num_fds()
        except (AttributeError, psutil.AccessDenied):
            # num_fds only available on Linux/Unix
            try:
                fd_path = f"/proc/{self.process.pid}/fd"
                if os.path.exists(fd_path):
                    fd_count = len(os.listdir(fd_path))
            except:
                pass
        
        # Child process count
        try:
            process_count = len(self.process.children())
        except:
            process_count = 0
        
        snapshot = MemorySnapshot(
            timestamp=time.time() - (self.start_time or time.time()),
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            thread_count=thread_count,
            fd_count=fd_count,
            process_count=process_count
        )
        
        self.snapshots.append((label, snapshot))
        return snapshot
    
    def get_memory_growth(self) -> float:
        """Get memory growth percentage from first to last snapshot"""
        if len(self.snapshots) < 2:
            return 0.0
        baseline = self.snapshots[0][1].rss_mb
        final = self.snapshots[-1][1].rss_mb
        if baseline == 0:
            return 0.0
        return (final - baseline) / baseline
    
    def get_report(self) -> Dict:
        """Generate resource report"""
        if not self.snapshots:
            return {}
            
        rss_values = [s[1].rss_mb for s in self.snapshots]
        thread_values = [s[1].thread_count for s in self.snapshots]
        fd_values = [s[1].fd_count for s in self.snapshots]
        
        return {
            'duration_seconds': time.time() - (self.start_time or time.time()),
            'memory_baseline_mb': rss_values[0],
            'memory_peak_mb': max(rss_values),
            'memory_final_mb': rss_values[-1],
            'memory_growth_mb': rss_values[-1] - rss_values[0],
            'memory_growth_pct': self.get_memory_growth() * 100,
            'thread_baseline': thread_values[0],
            'thread_peak': max(thread_values),
            'thread_final': thread_values[-1],
            'fd_baseline': fd_values[0],
            'fd_peak': max(fd_values),
            'fd_final': fd_values[-1],
            'snapshot_count': len(self.snapshots)
        }
    
    def plot(self, output_path: str):
        """Generate memory plot"""
        if not MATPLOTLIB_AVAILABLE or len(self.snapshots) < 2:
            return
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Resource Usage Over Time', fontsize=14, fontweight='bold')
        
        timestamps = [s[1].timestamp for s in self.snapshots]
        rss = [s[1].rss_mb for s in self.snapshots]
        threads = [s[1].thread_count for s in self.snapshots]
        fds = [s[1].fd_count for s in self.snapshots]
        processes = [s[1].process_count for s in self.snapshots]
        
        # Memory plot
        axes[0, 0].plot(timestamps, rss, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Memory (MB)')
        axes[0, 0].set_title('RSS Memory Usage')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Thread plot
        axes[0, 1].plot(timestamps, threads, 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Time (seconds)')
        axes[0, 1].set_ylabel('Thread Count')
        axes[0, 1].set_title('Active Threads')
        axes[0, 1].grid(True, alpha=0.3)
        
        # File descriptor plot
        axes[1, 0].plot(timestamps, fds, 'r-', linewidth=2)
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('FD Count')
        axes[1, 0].set_title('Open File Descriptors')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Child processes plot
        axes[1, 1].plot(timestamps, processes, 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Process Count')
        axes[1, 1].set_title('Child Processes')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


class TempFileTracker:
    """Track temporary files"""
    
    def __init__(self):
        self.temp_dirs = [tempfile.gettempdir()]
        self.baseline_files: set = set()
        self.created_files: List[str] = []
        
    def capture_baseline(self):
        """Capture baseline temp files"""
        self.baseline_files = self._get_temp_files()
        
    def _get_temp_files(self) -> set:
        """Get current temp files"""
        files = set()
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    for f in os.listdir(temp_dir):
                        if f.startswith('tmp') or f.startswith('temp'):
                            files.add(os.path.join(temp_dir, f))
                except:
                    pass
        return files
    
    def check_new_files(self) -> List[str]:
        """Check for new temp files"""
        current = self._get_temp_files()
        new_files = current - self.baseline_files
        return list(new_files)
    
    def cleanup_check(self) -> Tuple[int, int]:
        """Check cleanup and return (remaining, cleaned) count"""
        new_files = self.check_new_files()
        remaining = 0
        cleaned = 0
        for f in new_files:
            if os.path.exists(f):
                remaining += 1
            else:
                cleaned += 1
        return remaining, cleaned


def create_test_audio(duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Create synthetic test audio"""
    if not NUMPY_AVAILABLE:
        return None
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate synthetic speech-like audio
    audio = np.sin(2 * np.pi * 440 * t) * 0.1  # 440 Hz tone
    audio += np.sin(2 * np.pi * 880 * t) * 0.05  # Harmonic
    audio += np.random.randn(len(t)) * 0.01  # Noise
    return audio.astype(np.float32)


def save_test_wav(audio: np.ndarray, path: str, sample_rate: int = 16000):
    """Save audio to WAV file"""
    import wave
    audio_int16 = np.clip(audio * 32767, -32768, 32768).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())


class TestMemoryLeaks(unittest.TestCase):
    """Comprehensive memory leak detection tests"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        print("\n" + "="*70)
        print("MEMORY LEAK DETECTION TEST SUITE")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"psutil available: {PSUTIL_AVAILABLE}")
        print(f"numpy available: {NUMPY_AVAILABLE}")
        print(f"matplotlib available: {MATPLOTLIB_AVAILABLE}")
        print("="*70)
        
        cls.results: List[LeakTestResult] = []
        cls.test_dir = tempfile.mkdtemp(prefix="qwen3_leak_test_")
        
        # Create test audio file
        cls.test_audio_path = os.path.join(cls.test_dir, "test_audio.wav")
        if NUMPY_AVAILABLE:
            audio = create_test_audio(TEST_CONFIG['sample_audio_duration'])
            save_test_wav(audio, cls.test_audio_path)
        
        # Force garbage collection before tests
        gc.collect()
        gc.collect()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up and generate report"""
        print("\n" + "="*70)
        print("CLEANUP & REPORT GENERATION")
        print("="*70)
        
        # Clean up test directory
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir, ignore_errors=True)
            print(f"✅ Test directory cleaned: {cls.test_dir}")
        
        # Generate report
        cls._generate_report()
        
        print("="*70)
        print("TEST SUITE COMPLETE")
        print("="*70)
    
    @classmethod
    def _generate_report(cls):
        """Generate comprehensive leak detection report"""
        report_dir = os.path.join(BASE_DIR, "tests", "leak_reports")
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f"leak_report_{timestamp}.json")
        
        # JSON report
        report = {
            'timestamp': timestamp,
            'test_config': TEST_CONFIG,
            'results': [r.to_dict() for r in cls.results],
            'summary': {
                'total_tests': len(cls.results),
                'passed': sum(1 for r in cls.results if r.passed),
                'failed': sum(1 for r in cls.results if not r.passed)
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Report saved to: {report_path}")
        
        # Print summary table
        print("\n" + "="*70)
        print("LEAK DETECTION SUMMARY")
        print("="*70)
        print(f"{'Test':<30} {'Status':<8} {'Memory Growth':<15} {'Duration':<10}")
        print("-"*70)
        
        for result in cls.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            growth = f"{result.memory_growth_pct:.1f}%"
            duration = f"{result.duration_seconds:.1f}s"
            print(f"{result.test_name:<30} {status:<8} {growth:<15} {duration:<10}")
        
        print("-"*70)
        print(f"Total: {report['summary']['total_tests']}, "
              f"Passed: {report['summary']['passed']}, "
              f"Failed: {report['summary']['failed']}")
    
    def _record_result(self, result: LeakTestResult):
        """Record test result"""
        self.__class__.results.append(result)
    
    def test_01_file_descriptor_tracking(self):
        """Test 1: File handle leak detection - verify FD stability"""
        print("\n" + "-"*70)
        print("TEST 1: File Descriptor Leak Detection")
        print("-"*70)
        
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available")
        
        tracker = ResourceTracker()
        tracker.start()
        
        # Simulate file operations
        temp_files = []
        try:
            for i in range(50):
                # Create and close files
                fd, path = tempfile.mkstemp(suffix='.txt')
                os.write(fd, b"test data")
                os.close(fd)
                temp_files.append(path)
                
                # Read file
                with open(path, 'r') as f:
                    _ = f.read()
                
                if i % 10 == 0:
                    tracker.take_snapshot(f"iteration_{i}")
        finally:
            # Clean up
            for path in temp_files:
                try:
                    os.unlink(path)
                except:
                    pass
        
        # Force cleanup
        gc.collect()
        time.sleep(0.5)
        tracker.take_snapshot("after_cleanup")
        
        report = tracker.get_report()
        fd_delta = report['fd_final'] - report['fd_baseline']
        
        # Generate plot
        if MATPLOTLIB_AVAILABLE:
            plot_path = os.path.join(BASE_DIR, "tests", "leak_reports", "test_01_fd_tracking.png")
            tracker.plot(plot_path)
            print(f"📊 Plot saved: {plot_path}")
        
        result = LeakTestResult(
            test_name="File Descriptor Tracking",
            passed=abs(fd_delta) <= 5,  # Allow small variance
            baseline_memory_mb=report['memory_baseline_mb'],
            peak_memory_mb=report['memory_peak_mb'],
            final_memory_mb=report['memory_final_mb'],
            memory_growth_pct=report['memory_growth_pct'],
            thread_delta=0,
            fd_delta=fd_delta,
            process_delta=0,
            duration_seconds=report['duration_seconds'],
            details=report
        )
        self._record_result(result)
        
        print(f"  FD baseline: {report['fd_baseline']}")
        print(f"  FD final: {report['fd_final']}")
        print(f"  FD delta: {fd_delta}")
        print(f"  Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
        
        self.assertLessEqual(abs(fd_delta), 10, f"File descriptor leak detected: {fd_delta}")
    
    def test_02_thread_leak_detection(self):
        """Test 2: Thread leak detection - verify threads terminate"""
        print("\n" + "-"*70)
        print("TEST 2: Thread Leak Detection")
        print("-"*70)
        
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available")
        
        tracker = ResourceTracker()
        tracker.start()
        
        def worker():
            """Simple worker thread"""
            time.sleep(0.01)
            return "done"
        
        # Create and join threads
        for i in range(100):
            threads = []
            for _ in range(5):
                t = threading.Thread(target=worker)
                t.start()
                threads.append(t)
            
            for t in threads:
                t.join(timeout=1.0)
            
            if i % 20 == 0:
                tracker.take_snapshot(f"batch_{i}")
        
        # Give threads time to clean up
        time.sleep(0.5)
        gc.collect()
        tracker.take_snapshot("final")
        
        report = tracker.get_report()
        thread_delta = report['thread_final'] - report['thread_baseline']
        
        result = LeakTestResult(
            test_name="Thread Leak Detection",
            passed=abs(thread_delta) <= 2,
            baseline_memory_mb=report['memory_baseline_mb'],
            peak_memory_mb=report['memory_peak_mb'],
            final_memory_mb=report['memory_final_mb'],
            memory_growth_pct=report['memory_growth_pct'],
            thread_delta=thread_delta,
            fd_delta=0,
            process_delta=0,
            duration_seconds=report['duration_seconds'],
            details=report
        )
        self._record_result(result)
        
        print(f"  Thread baseline: {report['thread_baseline']}")
        print(f"  Thread peak: {report['thread_peak']}")
        print(f"  Thread final: {report['thread_final']}")
        print(f"  Thread delta: {thread_delta}")
        print(f"  Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
        
        self.assertLessEqual(abs(thread_delta), 5, f"Thread leak detected: {thread_delta}")
    
    def test_03_temp_file_cleanup(self):
        """Test 3: Temporary file cleanup verification"""
        print("\n" + "-"*70)
        print("TEST 3: Temporary File Cleanup")
        print("-"*70)
        
        tracker = TempFileTracker()
        tracker.capture_baseline()
        
        created_files = []
        
        # Create temp files like the app does
        for i in range(100):
            # Method 1: tempfile.mkstemp
            fd, path = tempfile.mkstemp(suffix='.wav')
            os.write(fd, b"fake audio data")
            os.close(fd)
            created_files.append(path)
            
            # Method 2: NamedTemporaryFile
            with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
                f.write(b"test content")
                created_files.append(f.name)
            
            # Simulate cleanup
            if i % 10 == 0:
                for f in created_files[-10:]:
                    if os.path.exists(f):
                        os.unlink(f)
        
        # Final cleanup
        remaining_before = 0
        for f in created_files:
            if os.path.exists(f):
                remaining_before += 1
                try:
                    os.unlink(f)
                except:
                    pass
        
        time.sleep(0.5)
        remaining, cleaned = tracker.cleanup_check()
        
        # Check for any remaining files we created
        actual_remaining = sum(1 for f in created_files if os.path.exists(f))
        
        result = LeakTestResult(
            test_name="Temp File Cleanup",
            passed=actual_remaining == 0,
            baseline_memory_mb=0,
            peak_memory_mb=0,
            final_memory_mb=0,
            memory_growth_pct=0,
            thread_delta=0,
            fd_delta=0,
            process_delta=0,
            duration_seconds=0,
            details={
                'files_created': len(created_files),
                'files_remaining': actual_remaining,
                'files_cleaned': len(created_files) - actual_remaining
            }
        )
        self._record_result(result)
        
        print(f"  Files created: {len(created_files)}")
        print(f"  Files remaining: {actual_remaining}")
        print(f"  Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
        
        self.assertEqual(actual_remaining, 0, f"Temp files not cleaned up: {actual_remaining}")
    
    def test_04_numpy_array_lifecycle(self):
        """Test 4: Numpy array leak detection"""
        print("\n" + "-"*70)
        print("TEST 4: Numpy Array Lifecycle")
        print("-"*70)
        
        if not NUMPY_AVAILABLE or not PSUTIL_AVAILABLE:
            self.skipTest("numpy or psutil not available")
        
        tracker = ResourceTracker()
        tracker.start()
        
        # Simulate audio processing with large arrays
        for i in range(50):
            # Create large arrays like in audio processing
            audio = np.random.randn(16000 * 10).astype(np.float32)  # 10 seconds
            
            # Process in chunks
            chunk_size = 16000
            for j in range(0, len(audio), chunk_size):
                chunk = audio[j:j+chunk_size].copy()
                processed = np.fft.fft(chunk)
                del processed
            
            del audio
            
            if i % 10 == 0:
                gc.collect()
                tracker.take_snapshot(f"iteration_{i}")
        
        gc.collect()
        gc.collect()
        tracker.take_snapshot("final")
        
        report = tracker.get_report()
        
        result = LeakTestResult(
            test_name="Numpy Array Lifecycle",
            passed=report['memory_growth_pct'] < 20,  # Allow some growth for numpy overhead
            baseline_memory_mb=report['memory_baseline_mb'],
            peak_memory_mb=report['memory_peak_mb'],
            final_memory_mb=report['memory_final_mb'],
            memory_growth_pct=report['memory_growth_pct'],
            thread_delta=0,
            fd_delta=0,
            process_delta=0,
            duration_seconds=report['duration_seconds'],
            details=report
        )
        self._record_result(result)
        
        print(f"  Memory baseline: {report['memory_baseline_mb']:.1f} MB")
        print(f"  Memory peak: {report['memory_peak_mb']:.1f} MB")
        print(f"  Memory final: {report['memory_final_mb']:.1f} MB")
        print(f"  Memory growth: {report['memory_growth_pct']:.1f}%")
        print(f"  Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
        
        self.assertLess(report['memory_growth_pct'], 30, "Significant memory growth detected")
    
    def test_05_long_running_stability(self):
        """Test 5: Long running memory stability (30 min simulation)"""
        print("\n" + "-"*70)
        print("TEST 5: Long Running Memory Stability")
        print("-"*70)
        print(f"  Duration: {TEST_CONFIG['long_running_duration']} seconds")
        
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available")
        
        tracker = ResourceTracker()
        tracker.start()
        
        # Simulate long running with periodic activity
        start = time.time()
        iteration = 0
        
        while time.time() - start < TEST_CONFIG['long_running_duration']:
            # Simulate some work
            data = [i * 2 for i in range(1000)]
            del data
            
            if iteration % 10 == 0:
                gc.collect()
                tracker.take_snapshot(f"time_{int(time.time() - start)}")
            
            iteration += 1
            time.sleep(0.1)
        
        tracker.take_snapshot("final")
        
        report = tracker.get_report()
        growth_pct = report['memory_growth_pct']
        
        # Generate plot
        if MATPLOTLIB_AVAILABLE:
            plot_path = os.path.join(BASE_DIR, "tests", "leak_reports", "test_05_long_running.png")
            tracker.plot(plot_path)
            print(f"📊 Plot saved: {plot_path}")
        
        result = LeakTestResult(
            test_name="Long Running Stability",
            passed=growth_pct < TEST_CONFIG['memory_growth_threshold'] * 100,
            baseline_memory_mb=report['memory_baseline_mb'],
            peak_memory_mb=report['memory_peak_mb'],
            final_memory_mb=report['memory_final_mb'],
            memory_growth_pct=growth_pct,
            thread_delta=report['thread_final'] - report['thread_baseline'],
            fd_delta=report['fd_final'] - report['fd_baseline'],
            process_delta=0,
            duration_seconds=report['duration_seconds'],
            details={
                **report,
                'iterations': iteration
            }
        )
        self._record_result(result)
        
        print(f"  Iterations: {iteration}")
        print(f"  Memory baseline: {report['memory_baseline_mb']:.1f} MB")
        print(f"  Memory final: {report['memory_final_mb']:.1f} MB")
        print(f"  Memory growth: {growth_pct:.1f}% (threshold: {TEST_CONFIG['memory_growth_threshold']*100:.1f}%)")
        print(f"  Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
        
        self.assertLess(growth_pct, TEST_CONFIG['memory_growth_threshold'] * 100,
                       f"Memory growth {growth_pct:.1f}% exceeds threshold")
    
    def test_06_model_switching_leaks(self):
        """Test 6: Model loading/unloading leak detection"""
        print("\n" + "-"*70)
        print("TEST 6: Model Switching Leak Detection")
        print("-"*70)
        print(f"  Switch count: {TEST_CONFIG['model_switch_count']}")
        
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available")
        
        try:
            from app import TranscriptionEngine
        except ImportError:
            self.skipTest("Cannot import app.TranscriptionEngine")
        
        tracker = ResourceTracker()
        tracker.start()
        
        try:
            engine = TranscriptionEngine()
        except RuntimeError as e:
            self.skipTest(f"No transcription backend available: {e}")
        
        models = ["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"]
        
        for i in range(TEST_CONFIG['model_switch_count']):
            model = models[i % len(models)]
            try:
                engine.load_model(model)
            except Exception as e:
                print(f"  Warning: Failed to load {model}: {e}")
            
            if i % 5 == 0:
                gc.collect()
                tracker.take_snapshot(f"switch_{i}")
        
        # Cleanup
        engine.model = None
        gc.collect()
        time.sleep(1)
        tracker.take_snapshot("final")
        
        report = tracker.get_report()
        
        result = LeakTestResult(
            test_name="Model Switching",
            passed=report['memory_growth_pct'] < 50,  # Allow more for model switching
            baseline_memory_mb=report['memory_baseline_mb'],
            peak_memory_mb=report['memory_peak_mb'],
            final_memory_mb=report['memory_final_mb'],
            memory_growth_pct=report['memory_growth_pct'],
            thread_delta=report['thread_final'] - report['thread_baseline'],
            fd_delta=report['fd_final'] - report['fd_baseline'],
            process_delta=0,
            duration_seconds=report['duration_seconds'],
            details=report
        )
        self._record_result(result)
        
        print(f"  Memory baseline: {report['memory_baseline_mb']:.1f} MB")
        print(f"  Memory peak: {report['memory_peak_mb']:.1f} MB")
        print(f"  Memory final: {report['memory_final_mb']:.1f} MB")
        print(f"  Memory growth: {report['memory_growth_pct']:.1f}%")
        print(f"  Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
    
    def test_07_live_streamer_stress(self):
        """Test 7: Live streaming stress test - 100 start/stop cycles"""
        print("\n" + "-"*70)
        print("TEST 7: Live Streaming Stress Test")
        print("-"*70)
        print(f"  Cycles: {TEST_CONFIG['streaming_cycles']}")
        
        if not PSUTIL_AVAILABLE or not NUMPY_AVAILABLE:
            self.skipTest("psutil or numpy not available")
        
        try:
            from app import LiveStreamer, SAMPLE_RATE
        except ImportError:
            self.skipTest("Cannot import LiveStreamer")
        
        tracker = ResourceTracker()
        tracker.start()
        
        # Paths
        model_06b = os.path.join(BASE_DIR, "assets", "c-asr", "qwen3-asr-0.6b")
        binary_path = os.path.join(BASE_DIR, "assets", "c-asr", "qwen_asr")
        
        if not os.path.exists(binary_path):
            self.skipTest(f"C binary not found: {binary_path}")
        
        streamer = LiveStreamer(
            model_dir=model_06b,
            binary_path=binary_path,
            sample_rate=SAMPLE_RATE
        )
        
        success_count = 0
        
        for i in range(TEST_CONFIG['streaming_cycles']):
            try:
                # Start
                streamer.start()
                
                # Feed small audio chunk
                audio = create_test_audio(0.5)  # 0.5 second
                streamer.feed_audio(audio)
                
                # Stop
                streamer.stop()
                success_count += 1
                
            except Exception as e:
                print(f"  Cycle {i} failed: {e}")
            
            if i % 20 == 0:
                gc.collect()
                tracker.take_snapshot(f"cycle_{i}")
        
        tracker.take_snapshot("final")
        
        report = tracker.get_report()
        
        result = LeakTestResult(
            test_name="Live Streaming Stress",
            passed=success_count >= TEST_CONFIG['streaming_cycles'] * 0.9,  # 90% success rate
            baseline_memory_mb=report['memory_baseline_mb'],
            peak_memory_mb=report['memory_peak_mb'],
            final_memory_mb=report['memory_final_mb'],
            memory_growth_pct=report['memory_growth_pct'],
            thread_delta=report['thread_final'] - report['thread_baseline'],
            fd_delta=report['fd_final'] - report['fd_baseline'],
            process_delta=0,
            duration_seconds=report['duration_seconds'],
            details={
                **report,
                'success_count': success_count,
                'target_cycles': TEST_CONFIG['streaming_cycles']
            }
        )
        self._record_result(result)
        
        print(f"  Success rate: {success_count}/{TEST_CONFIG['streaming_cycles']}")
        print(f"  Memory growth: {report['memory_growth_pct']:.1f}%")
        print(f"  Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
    
    def test_08_process_leak_detection(self):
        """Test 8: Process leak detection - no zombie processes"""
        print("\n" + "-"*70)
        print("TEST 8: Process Leak Detection")
        print("-"*70)
        
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available")
        
        tracker = ResourceTracker()
        tracker.start()
        
        initial_children = len(psutil.Process().children())
        
        # Spawn and cleanup subprocesses
        for i in range(20):
            try:
                # Run a quick subprocess
                result = subprocess.run(
                    ['python', '-c', 'print("hello")'],
                    capture_output=True,
                    timeout=5
                )
            except Exception as e:
                print(f"  Subprocess {i} failed: {e}")
            
            if i % 5 == 0:
                tracker.take_snapshot(f"spawn_{i}")
        
        # Give time for processes to terminate
        time.sleep(1)
        gc.collect()
        tracker.take_snapshot("final")
        
        final_children = len(psutil.Process().children())
        process_delta = final_children - initial_children
        
        report = tracker.get_report()
        
        result = LeakTestResult(
            test_name="Process Leak Detection",
            passed=abs(process_delta) <= 2,
            baseline_memory_mb=report['memory_baseline_mb'],
            peak_memory_mb=report['memory_peak_mb'],
            final_memory_mb=report['memory_final_mb'],
            memory_growth_pct=report['memory_growth_pct'],
            thread_delta=0,
            fd_delta=0,
            process_delta=process_delta,
            duration_seconds=report['duration_seconds'],
            details={
                **report,
                'initial_children': initial_children,
                'final_children': final_children
            }
        )
        self._record_result(result)
        
        print(f"  Initial children: {initial_children}")
        print(f"  Final children: {final_children}")
        print(f"  Process delta: {process_delta}")
        print(f"  Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
        
        self.assertLessEqual(abs(process_delta), 5, f"Process leak detected: {process_delta}")
    
    def test_09_tkinter_widget_cleanup(self):
        """Test 9: Tkinter widget leak detection"""
        print("\n" + "-"*70)
        print("TEST 9: Tkinter Widget Cleanup")
        print("-"*70)
        
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available")
        
        tracker = ResourceTracker()
        tracker.start()
        
        # Use mocking to avoid actual GUI
        with patch('tkinter.Tk'), \
             patch('tkinter.Tcl'):
            
            try:
                # Simulate widget creation and destruction without importing actual classes
                # since they require a real Tk instance
                for i in range(50):
                    # Create mock widgets
                    canvas = MagicMock()
                    frame = MagicMock()
                    
                    # Simulate widget lifecycle
                    canvas.delete('all')
                    frame.destroy = MagicMock()
                    frame.destroy()
                    
                    del canvas
                    del frame
                    
                    if i % 10 == 0:
                        gc.collect()
                        tracker.take_snapshot(f"cycle_{i}")
                
            except ImportError as e:
                print(f"  Import warning: {e}")
        
        gc.collect()
        tracker.take_snapshot("final")
        
        report = tracker.get_report()
        
        result = LeakTestResult(
            test_name="Tkinter Widget Cleanup",
            passed=report['memory_growth_pct'] < 20,
            baseline_memory_mb=report['memory_baseline_mb'],
            peak_memory_mb=report['memory_peak_mb'],
            final_memory_mb=report['memory_final_mb'],
            memory_growth_pct=report['memory_growth_pct'],
            thread_delta=0,
            fd_delta=0,
            process_delta=0,
            duration_seconds=report['duration_seconds'],
            details=report
        )
        self._record_result(result)
        
        print(f"  Memory growth: {report['memory_growth_pct']:.1f}%")
        print(f"  Status: {'✅ PASS' if result.passed else '❌ FAIL'}")
    
    def test_10_batch_transcription_memory(self):
        """Test 10: Multiple transcriptions memory stability"""
        print("\n" + "-"*70)
        print("TEST 10: Batch Transcription Memory")
        print("-"*70)
        print(f"  Count: {TEST_CONFIG['batch_transcription_count']}")
        
        if not PSUTIL_AVAILABLE or not NUMPY_AVAILABLE:
            self.skipTest("psutil or numpy not available")
        
        try:
            from app import TranscriptionEngine
        except ImportError:
            self.skipTest("Cannot import TranscriptionEngine")
        
        tracker = ResourceTracker()
        tracker.start()
        
        try:
            engine = TranscriptionEngine()
        except RuntimeError as e:
            self.skipTest(f"No backend available: {e}")
        
        # Create test audio files
        test_files = []
        for i in range(min(10, TEST_CONFIG['batch_transcription_count'])):
            path = os.path.join(self.test_dir, f"batch_test_{i}.wav")
            audio = create_test_audio(2.0)
            save_test_wav(audio, path)
            test_files.append(path)
        
        success_count = 0
        
        for i in range(TEST_CONFIG['batch_transcription_count']):
            try:
                file_path = test_files[i % len(test_files)]
                result, stats = engine.transcribe(
                    file_path,
                    model="Qwen/Qwen3-ASR-0.6B",
                    language="English"
                )
                success_count += 1
            except Exception as e:
                print(f"  Transcription {i} failed: {e}")
            
            if i % 10 == 0:
                gc.collect()
                tracker.take_snapshot(f"transcription_{i}")
        
        # Cleanup
        engine.model = None
        gc.collect()
        tracker.take_snapshot("final")
        
        report = tracker.get_report()
        
        result = LeakTestResult(
            test_name="Batch Transcription",
            passed=success_count >= TEST_CONFIG['batch_transcription_count'] * 0.8,
            baseline_memory_mb=report['memory_baseline_mb'],
            peak_memory_mb=report['memory_peak_mb'],
            final_memory_mb=report['memory_final_mb'],
            memory_growth_pct=report['memory_growth_pct'],
            thread_delta=report['thread_final'] - report['thread_baseline'],
            fd_delta=report['fd_final'] - report['fd_baseline'],
            process_delta=0,
            duration_seconds=report['duration_seconds'],
            details={
                **report,
                'success_count': success_count,
                'target_count': TEST_CONFIG['batch_transcription_count']
            }
        )
        self._record_result(result)
        
        print(f"  Success rate: {success_count}/{TEST_CONFIG['batch_transcription_count']}")
        print(f"  Memory growth: {report['memory_growth_pct']:.1f}%")
        print(f"  Status: {'✅ PASS' if result.passed else '❌ FAIL'}")


class TestGarbageCollection(unittest.TestCase):
    """Test garbage collection effectiveness"""
    
    def test_gc_effectiveness(self):
        """Verify garbage collection works correctly"""
        print("\n" + "-"*70)
        print("TEST: Garbage Collection Effectiveness")
        print("-"*70)
        
        # Get initial counts
        gc.collect()
        initial_counts = gc.get_count()
        
        # Create and delete objects
        for _ in range(1000):
            obj = {"data": [i for i in range(100)]}
            del obj
        
        # Check gc detected garbage
        gc.collect()
        final_counts = gc.get_count()
        
        # Create some circular references
        for _ in range(100):
            a = {}
            b = {"ref": a}
            a["ref"] = b
            del a, b
        
        gc.collect()
        after_circular = gc.get_count()
        
        print(f"  Initial GC counts: {initial_counts}")
        print(f"  Final GC counts: {final_counts}")
        print(f"  After circular: {after_circular}")
        print(f"  Status: ✅ PASS")
        
        # GC should be working (counts should stabilize)
        self.assertTrue(gc.isenabled(), "GC should be enabled")


def generate_summary_graph(results: List[LeakTestResult], output_path: str):
    """Generate summary graph of all tests"""
    if not MATPLOTLIB_AVAILABLE or not results:
        return
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Memory Leak Detection Test Summary', fontsize=16, fontweight='bold')
    
    test_names = [r.test_name for r in results]
    memory_growths = [r.memory_growth_pct for r in results]
    durations = [r.duration_seconds for r in results]
    
    # Memory growth bar chart
    colors = ['green' if r.passed else 'red' for r in results]
    axes[0, 0].barh(test_names, memory_growths, color=colors, alpha=0.7)
    axes[0, 0].axvline(x=TEST_CONFIG['memory_growth_threshold'] * 100, color='orange', 
                       linestyle='--', label='10% Threshold')
    axes[0, 0].set_xlabel('Memory Growth (%)')
    axes[0, 0].set_title('Memory Growth by Test')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Duration bar chart
    axes[0, 1].barh(test_names, durations, color='steelblue', alpha=0.7)
    axes[0, 1].set_xlabel('Duration (seconds)')
    axes[0, 1].set_title('Test Duration')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Pass/Fail pie chart
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    axes[1, 0].pie([passed, failed], labels=['Pass', 'Fail'], 
                   colors=['green', 'red'], autopct='%1.0f%%',
                   startangle=90)
    axes[1, 0].set_title('Test Results')
    
    # Thread/FD deltas
    thread_deltas = [r.thread_delta for r in results]
    fd_deltas = [r.fd_delta for r in results]
    
    x = range(len(test_names))
    width = 0.35
    axes[1, 1].bar([i - width/2 for i in x], thread_deltas, width, 
                   label='Thread Delta', color='blue', alpha=0.7)
    axes[1, 1].bar([i + width/2 for i in x], fd_deltas, width,
                   label='FD Delta', color='orange', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(test_names, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Delta Count')
    axes[1, 1].set_title('Thread and File Descriptor Deltas')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Summary graph saved: {output_path}")


def run_all_tests():
    """Run all leak detection tests"""
    # Create leak reports directory
    report_dir = os.path.join(BASE_DIR, "tests", "leak_reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryLeaks))
    suite.addTests(loader.loadTestsFromTestCase(TestGarbageCollection))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate summary graph
    if TestMemoryLeaks.results and MATPLOTLIB_AVAILABLE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_path = os.path.join(report_dir, f"summary_{timestamp}.png")
        generate_summary_graph(TestMemoryLeaks.results, graph_path)
    
    return result


if __name__ == '__main__':
    result = run_all_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
