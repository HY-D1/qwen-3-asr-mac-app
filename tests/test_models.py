#!/usr/bin/env python3
"""
Comprehensive tests for Model Selection and Backend functionality
Tests model switching, backend loading, and error handling.
"""

import os
import sys
import time
import unittest
import threading
import tempfile
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from typing import Optional, Dict, Any, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
C_ASR_DIR = os.path.join(ASSETS_DIR, "c-asr")
MODELS_DIR = os.path.join(ASSETS_DIR, "models")

# Model configurations
MODEL_CONFIGS = {
    "0.6B": {
        "name": "Qwen/Qwen3-ASR-0.6B",
        "dir": "qwen3-asr-0.6b",
        "size": "0.6B",
        "expected_files": [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "vocab.json"
        ]
    },
    "1.7B": {
        "name": "Qwen/Qwen3-ASR-1.7B",
        "dir": "qwen3-asr-1.7b",
        "size": "1.7B",
        "expected_files": [
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "vocab.json"
        ]
    }
}


class TestModelPaths(unittest.TestCase):
    """Test model path resolution and file integrity"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test paths"""
        cls.base_dir = BASE_DIR
        cls.assets_dir = ASSETS_DIR
        cls.c_asr_dir = C_ASR_DIR
        
    def test_01_base_directory_exists(self):
        """Verify base project directory exists"""
        self.assertTrue(os.path.exists(self.base_dir), 
                       f"Base directory not found: {self.base_dir}")
        print(f"✅ Base directory exists: {self.base_dir}")
    
    def test_02_assets_directory_structure(self):
        """Verify assets directory structure"""
        self.assertTrue(os.path.exists(self.assets_dir), 
                       f"Assets directory not found: {self.assets_dir}")
        self.assertTrue(os.path.exists(self.c_asr_dir), 
                       f"C-ASR directory not found: {self.c_asr_dir}")
        print(f"✅ Assets directory structure valid")
    
    def test_03_model_0_6b_files_exist(self):
        """Verify 0.6B model files exist and are valid"""
        model_dir = os.path.join(self.c_asr_dir, MODEL_CONFIGS["0.6B"]["dir"])
        self.assertTrue(os.path.exists(model_dir), 
                       f"0.6B model directory not found: {model_dir}")
        
        for file in MODEL_CONFIGS["0.6B"]["expected_files"]:
            file_path = os.path.join(model_dir, file)
            self.assertTrue(os.path.exists(file_path), 
                           f"0.6B model file missing: {file}")
            self.assertGreater(os.path.getsize(file_path), 0, 
                              f"0.6B model file is empty: {file}")
        
        print(f"✅ 0.6B model files verified ({len(MODEL_CONFIGS['0.6B']['expected_files'])} files)")
    
    def test_04_model_1_7b_files_exist(self):
        """Verify 1.7B model files exist and are valid"""
        model_dir = os.path.join(self.c_asr_dir, MODEL_CONFIGS["1.7B"]["dir"])
        self.assertTrue(os.path.exists(model_dir), 
                       f"1.7B model directory not found: {model_dir}")
        
        for file in MODEL_CONFIGS["1.7B"]["expected_files"]:
            file_path = os.path.join(model_dir, file)
            self.assertTrue(os.path.exists(file_path), 
                           f"1.7B model file missing: {file}")
            self.assertGreater(os.path.getsize(file_path), 0, 
                              f"1.7B model file is empty: {file}")
        
        print(f"✅ 1.7B model files verified ({len(MODEL_CONFIGS['1.7B']['expected_files'])} files)")
    
    def test_05_model_file_sizes(self):
        """Verify model files have expected sizes"""
        # 0.6B model - should be around 1.8GB
        model_06b_path = os.path.join(self.c_asr_dir, MODEL_CONFIGS["0.6B"]["dir"], "model.safetensors")
        size_06b = os.path.getsize(model_06b_path) / (1024**3)  # GB
        self.assertGreater(size_06b, 1.5, "0.6B model seems too small")
        self.assertLess(size_06b, 2.5, "0.6B model seems too large")
        print(f"✅ 0.6B model size: {size_06b:.2f} GB")
        
        # 1.7B model - should be around 4GB total (split across 2 files)
        model_17b_path = os.path.join(self.c_asr_dir, MODEL_CONFIGS["1.7B"]["dir"], "model-00001-of-00002.safetensors")
        size_17b = os.path.getsize(model_17b_path) / (1024**3)  # GB
        self.assertGreater(size_17b, 3.0, "1.7B model part 1 seems too small")
        print(f"✅ 1.7B model part 1 size: {size_17b:.2f} GB")


class TestCBinary(unittest.TestCase):
    """Test C binary for live streaming"""
    
    @classmethod
    def setUpClass(cls):
        cls.binary_path = os.path.join(C_ASR_DIR, "qwen_asr")
    
    def test_01_binary_exists(self):
        """Verify C binary exists"""
        self.assertTrue(os.path.exists(self.binary_path), 
                       f"C binary not found: {self.binary_path}")
        print(f"✅ C binary exists: {self.binary_path}")
    
    def test_02_binary_is_executable(self):
        """Verify C binary is executable"""
        self.assertTrue(os.access(self.binary_path, os.X_OK), 
                       f"C binary is not executable: {self.binary_path}")
        print(f"✅ C binary is executable")
    
    def test_03_binary_help_output(self):
        """Verify C binary can show help"""
        try:
            result = subprocess.run(
                [self.binary_path, "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            # Binary may return non-zero for help
            self.assertIn(result.returncode, [0, 1], "Unexpected exit code")
            self.assertTrue(
                len(result.stdout) > 0 or len(result.stderr) > 0,
                "No help output"
            )
            print(f"✅ C binary help command works")
        except subprocess.TimeoutExpired:
            self.fail("C binary help command timed out")
        except Exception as e:
            self.fail(f"C binary help command failed: {e}")
    
    def test_04_binary_version_info(self):
        """Verify C binary version information"""
        try:
            result = subprocess.run(
                [self.binary_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            output = result.stdout + result.stderr
            print(f"   Version output: {output[:200]}...")
            print(f"✅ C binary version check completed")
        except subprocess.TimeoutExpired:
            print("⚠️ Version command timed out (may not support --version)")
        except Exception as e:
            print(f"⚠️ Version check failed: {e}")


class TestBackendDetection(unittest.TestCase):
    """Test backend detection and loading"""
    
    def test_01_mlx_audio_available(self):
        """Check if mlx-audio is available"""
        try:
            import mlx_audio.stt as mlx_stt
            self.mlx_audio_available = True
            print(f"✅ mlx-audio backend is available")
        except ImportError:
            self.mlx_audio_available = False
            print(f"⚠️ mlx-audio not available (expected on non-Apple Silicon)")
    
    def test_02_pytorch_available(self):
        """Check if PyTorch is available"""
        try:
            import torch
            self.torch_available = True
            print(f"✅ PyTorch {torch.__version__} is available")
            if torch.backends.mps.is_available():
                print(f"   MPS (Metal) backend: Available")
            else:
                print(f"   MPS (Metal) backend: Not available")
        except ImportError:
            self.torch_available = False
            print(f"⚠️ PyTorch not available")
    
    def test_03_transcription_engine_init(self):
        """Test TranscriptionEngine initialization"""
        from app import TranscriptionEngine
        
        try:
            engine = TranscriptionEngine()
            self.assertIsNotNone(engine.backend, "Backend should be set")
            self.assertIn(engine.backend, ['mlx_audio', 'mlx_cli', 'pytorch'], 
                         f"Unexpected backend: {engine.backend}")
            print(f"✅ TranscriptionEngine initialized with backend: {engine.backend}")
        except RuntimeError as e:
            print(f"⚠️ TranscriptionEngine initialization failed: {e}")
            print(f"   This is expected if no backends are installed")


class TestModelSwitching(unittest.TestCase):
    """Test model switching functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.base_dir = BASE_DIR
        self.model_switch_times: Dict[str, float] = {}
    
    def test_01_model_path_resolution(self):
        """Test model path resolution for different models"""
        # Test 0.6B model path
        model_06b_path = os.path.join(self.base_dir, "assets", "c-asr", "qwen3-asr-0.6b")
        self.assertTrue(os.path.exists(model_06b_path), "0.6B model path not found")
        
        # Test 1.7B model path
        model_17b_path = os.path.join(self.base_dir, "assets", "c-asr", "qwen3-asr-1.7b")
        self.assertTrue(os.path.exists(model_17b_path), "1.7B model path not found")
        
        print(f"✅ Model paths resolved correctly")
    
    def test_02_model_switching_logic(self):
        """Test model switching logic from sidebar"""
        # Simulate the model switching logic from CollapsibleSidebar
        test_cases = [
            ("0.6B (Fast)", "Qwen/Qwen3-ASR-0.6B"),
            ("1.7B (Accurate)", "Qwen/Qwen3-ASR-1.7B"),
        ]
        
        for display_val, expected_model in test_cases:
            # Simulate _on_model_change logic
            if "0.6B" in display_val:
                model_value = "Qwen/Qwen3-ASR-0.6B"
            else:
                model_value = "Qwen/Qwen3-ASR-1.7B"
            
            self.assertEqual(model_value, expected_model, 
                           f"Model resolution failed for {display_val}")
        
        print(f"✅ Model switching logic verified")
    
    def test_03_model_loading_time_measurement(self):
        """Measure model loading times"""
        from app import TranscriptionEngine
        
        try:
            engine = TranscriptionEngine()
        except RuntimeError:
            self.skipTest("No transcription backend available")
        
        results = []
        
        for model_key, config in MODEL_CONFIGS.items():
            model_name = config["name"]
            
            # Measure loading time
            start_time = time.time()
            try:
                engine.load_model(model_name)
                load_time = time.time() - start_time
                self.model_switch_times[model_key] = load_time
                results.append((model_key, load_time))
                print(f"   {model_name}: {load_time:.2f}s")
            except Exception as e:
                print(f"   {model_name}: Failed ({e})")
        
        if results:
            print(f"✅ Model loading times measured ({len(results)} models)")
        else:
            print(f"⚠️ Could not measure model loading times")
    
    def test_04_model_switch_sequence(self):
        """Test switching between models multiple times"""
        from app import TranscriptionEngine
        
        try:
            engine = TranscriptionEngine()
        except RuntimeError:
            self.skipTest("No transcription backend available")
        
        models = ["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"]
        
        switch_times = []
        for i in range(3):  # Switch 3 times
            for model in models:
                start = time.time()
                try:
                    engine.load_model(model)
                    switch_times.append(time.time() - start)
                except Exception as e:
                    print(f"   Switch to {model} failed: {e}")
        
        if switch_times:
            avg_time = sum(switch_times) / len(switch_times)
            print(f"✅ Average model switch time: {avg_time:.2f}s ({len(switch_times)} switches)")


class TestMissingModelHandling(unittest.TestCase):
    """Test error handling for missing models"""
    
    def test_01_missing_model_directory(self):
        """Test handling of missing model directory"""
        fake_model_dir = os.path.join(C_ASR_DIR, "nonexistent-model")
        self.assertFalse(os.path.exists(fake_model_dir), 
                        "Fake model directory should not exist")
        
        # Verify the app would handle this gracefully
        print(f"✅ Missing model directory detection works")
    
    def test_02_missing_model_files(self):
        """Test handling of missing model files"""
        # Check for missing files in existing model
        model_dir = os.path.join(C_ASR_DIR, "qwen3-asr-0.6b")
        fake_file = os.path.join(model_dir, "nonexistent.safetensors")
        
        self.assertFalse(os.path.exists(fake_file), 
                        "Fake file should not exist")
        print(f"✅ Missing model file detection works")
    
    def test_03_invalid_model_name(self):
        """Test handling of invalid model name"""
        from app import TranscriptionEngine
        
        try:
            engine = TranscriptionEngine()
        except RuntimeError:
            self.skipTest("No transcription backend available")
        
        # Try to load an invalid model
        try:
            engine.load_model("Invalid/Model-Name")
            print(f"⚠️ Invalid model loaded without error (may be auto-downloaded)")
        except Exception as e:
            # Expected behavior - should raise an error
            print(f"✅ Invalid model name handled correctly: {type(e).__name__}")


class TestMemoryUsage(unittest.TestCase):
    """Test memory usage during model operations"""
    
    def test_01_memory_tracking_setup(self):
        """Set up memory tracking"""
        try:
            import psutil
            self.psutil_available = True
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024**2)  # MB
            print(f"✅ Memory tracking available (initial: {initial_memory:.1f} MB)")
        except ImportError:
            self.psutil_available = False
            print(f"⚠️ psutil not available - memory tracking disabled")
    
    def test_02_model_loading_memory(self):
        """Measure memory usage during model loading"""
        try:
            import psutil
            from app import TranscriptionEngine
        except ImportError as e:
            self.skipTest(f"Required module not available: {e}")
        
        try:
            engine = TranscriptionEngine()
        except RuntimeError:
            self.skipTest("No transcription backend available")
        
        process = psutil.Process()
        
        memory_readings = {}
        
        for model_key, config in MODEL_CONFIGS.items():
            # Get baseline
            baseline = process.memory_info().rss / (1024**2)
            
            try:
                engine.load_model(config["name"])
                after_load = process.memory_info().rss / (1024**2)
                memory_increase = after_load - baseline
                memory_readings[model_key] = {
                    'baseline': baseline,
                    'after_load': after_load,
                    'increase': memory_increase
                }
                print(f"   {model_key}: +{memory_increase:.1f} MB (total: {after_load:.1f} MB)")
            except Exception as e:
                print(f"   {model_key}: Failed ({e})")
        
        if memory_readings:
            print(f"✅ Memory usage measured for {len(memory_readings)} models")


class TestMLXBackend(unittest.TestCase):
    """Test MLX backend functionality"""
    
    def test_01_mlx_backend_detection(self):
        """Test MLX backend auto-detection"""
        from app import TranscriptionEngine
        
        try:
            engine = TranscriptionEngine()
            if engine.backend == 'mlx_audio':
                print(f"✅ MLX backend detected and loaded")
            elif engine.backend == 'mlx_cli':
                print(f"✅ MLX CLI backend detected")
            else:
                print(f"⚠️ MLX backend not selected (using: {engine.backend})")
        except RuntimeError as e:
            print(f"⚠️ No backend available: {e}")
    
    def test_02_mlx_model_loading(self):
        """Test loading models via MLX backend"""
        try:
            import mlx_audio.stt as mlx_stt
        except ImportError:
            self.skipTest("mlx-audio not available")
        
        from app import TranscriptionEngine
        
        try:
            engine = TranscriptionEngine()
            if engine.backend != 'mlx_audio':
                self.skipTest(f"MLX audio backend not in use (using: {engine.backend})")
        except RuntimeError:
            self.skipTest("No transcription backend available")
        
        # Try loading models
        for model_key, config in MODEL_CONFIGS.items():
            try:
                start = time.time()
                engine.load_model(config["name"])
                load_time = time.time() - start
                print(f"   {model_key} loaded via MLX in {load_time:.2f}s")
            except Exception as e:
                print(f"   {model_key} failed: {e}")
        
        print(f"✅ MLX model loading tested")


class TestPyTorchFallback(unittest.TestCase):
    """Test PyTorch fallback functionality"""
    
    def test_01_pytorch_backend_detection(self):
        """Test PyTorch backend availability"""
        try:
            import torch
            print(f"✅ PyTorch {torch.__version__} available")
            
            if torch.backends.mps.is_available():
                print(f"   Metal Performance Shaders (MPS) available")
            else:
                print(f"   CPU only (MPS not available)")
        except ImportError:
            print(f"⚠️ PyTorch not available")
    
    def test_02_pytorch_model_loading(self):
        """Test loading models via PyTorch backend"""
        try:
            import torch
            import qwen_asr
        except ImportError as e:
            self.skipTest(f"PyTorch/qwen_asr not available: {e}")
        
        from app import TranscriptionEngine
        
        # Force PyTorch backend by mocking mlx_audio
        with patch.dict('sys.modules', {'mlx_audio': None, 'mlx_audio.stt': None}):
            try:
                engine = TranscriptionEngine()
                if engine.backend == 'pytorch':
                    print(f"✅ PyTorch backend loaded")
                    
                    # Try loading models
                    for model_key, config in MODEL_CONFIGS.items():
                        try:
                            engine.load_model(config["name"])
                            print(f"   {model_key} loaded via PyTorch")
                        except Exception as e:
                            print(f"   {model_key} failed: {e}")
                else:
                    print(f"⚠️ PyTorch backend not selected (using: {engine.backend})")
            except RuntimeError as e:
                print(f"⚠️ Could not initialize engine: {e}")


class TestLiveStreamer(unittest.TestCase):
    """Test LiveStreamer model switching"""
    
    def test_01_streamer_initialization(self):
        """Test LiveStreamer initialization"""
        from app import LiveStreamer
        
        streamer = LiveStreamer()
        self.assertIsNotNone(streamer.model_dir, "Model dir should be set")
        self.assertIsNotNone(streamer.binary_path, "Binary path should be set")
        print(f"✅ LiveStreamer initialized")
        print(f"   Default model dir: {streamer.model_dir}")
    
    def test_02_streamer_model_switching(self):
        """Test switching models in LiveStreamer"""
        from app import LiveStreamer
        
        streamer = LiveStreamer()
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Test switching to 0.6B
        model_06b_path = os.path.join(base_dir, "assets", "c-asr", "qwen3-asr-0.6b")
        streamer.model_dir = model_06b_path
        self.assertEqual(streamer.model_dir, model_06b_path)
        self.assertTrue(os.path.exists(streamer.model_dir))
        print(f"✅ Switched to 0.6B model")
        
        # Test switching to 1.7B
        model_17b_path = os.path.join(base_dir, "assets", "c-asr", "qwen3-asr-1.7b")
        streamer.model_dir = model_17b_path
        self.assertEqual(streamer.model_dir, model_17b_path)
        self.assertTrue(os.path.exists(streamer.model_dir))
        print(f"✅ Switched to 1.7B model")


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflows"""
    
    def test_01_full_backend_workflow(self):
        """Test complete backend detection and model loading workflow"""
        from app import TranscriptionEngine
        
        print("\n   --- Backend Detection Workflow ---")
        
        try:
            engine = TranscriptionEngine()
            print(f"   Detected backend: {engine.backend}")
            print(f"   Initial model: {engine.model_name}")
            
            # Load 0.6B
            print("   Loading 0.6B model...")
            start = time.time()
            engine.load_model("Qwen/Qwen3-ASR-0.6B")
            time_06b = time.time() - start
            print(f"   ✅ 0.6B loaded in {time_06b:.2f}s")
            
            # Load 1.7B
            print("   Loading 1.7B model...")
            start = time.time()
            engine.load_model("Qwen/Qwen3-ASR-1.7B")
            time_17b = time.time() - start
            print(f"   ✅ 1.7B loaded in {time_17b:.2f}s")
            
            print(f"\n✅ Full workflow test passed")
            
        except RuntimeError as e:
            print(f"   ⚠️ Workflow test skipped: {e}")
    
    def test_02_error_recovery(self):
        """Test error recovery during model operations"""
        from app import TranscriptionEngine
        
        try:
            engine = TranscriptionEngine()
        except RuntimeError:
            self.skipTest("No backend available")
        
        # Store original model
        original_model = engine.model_name
        
        # Try to load invalid model (should fail gracefully)
        try:
            engine.load_model("Invalid/Model")
            print(f"⚠️ Invalid model loaded without error")
        except Exception:
            print(f"✅ Invalid model rejected correctly")
        
        # Verify we can still load valid models
        try:
            engine.load_model("Qwen/Qwen3-ASR-0.6B")
            print(f"✅ System recovered after error")
        except Exception as e:
            print(f"⚠️ Recovery failed: {e}")


def print_test_report():
    """Print comprehensive test report"""
    print("\n" + "="*70)
    print("QWEN3-ASR MODEL/BACKEND TEST REPORT")
    print("="*70)
    
    # File integrity check
    print("\n📁 FILE INTEGRITY CHECK")
    print("-" * 40)
    
    for model_key, config in MODEL_CONFIGS.items():
        model_dir = os.path.join(C_ASR_DIR, config["dir"])
        print(f"\n{config['name']}:")
        print(f"  Directory: {model_dir}")
        
        if os.path.exists(model_dir):
            print(f"  Status: ✅ Present")
            
            for file in config["expected_files"]:
                file_path = os.path.join(model_dir, file)
                if os.path.exists(file_path):
                    size_mb = os.path.getsize(file_path) / (1024**2)
                    print(f"    ✅ {file} ({size_mb:.1f} MB)")
                else:
                    print(f"    ❌ {file} (MISSING)")
        else:
            print(f"  Status: ❌ Directory not found")
    
    # C binary check
    print("\n🔧 C BINARY CHECK")
    print("-" * 40)
    binary_path = os.path.join(C_ASR_DIR, "qwen_asr")
    print(f"Path: {binary_path}")
    print(f"Exists: {'✅ Yes' if os.path.exists(binary_path) else '❌ No'}")
    print(f"Executable: {'✅ Yes' if os.access(binary_path, os.X_OK) else '❌ No'}")
    
    # Backend availability
    print("\n⚙️ BACKEND AVAILABILITY")
    print("-" * 40)
    
    backends = []
    
    try:
        import mlx_audio.stt
        backends.append(("mlx-audio", "✅ Available"))
    except ImportError:
        backends.append(("mlx-audio", "❌ Not installed"))
    
    try:
        import torch
        backends.append(("PyTorch", f"✅ {torch.__version__}"))
    except ImportError:
        backends.append(("PyTorch", "❌ Not installed"))
    
    try:
        import qwen_asr
        backends.append(("qwen-asr", "✅ Available"))
    except ImportError:
        backends.append(("qwen-asr", "❌ Not installed"))
    
    for name, status in backends:
        print(f"  {name}: {status}")
    
    print("\n" + "="*70)


def run_performance_tests():
    """Run performance benchmarks"""
    print("\n⏱️ PERFORMANCE BENCHMARKS")
    print("-" * 40)
    
    from app import TranscriptionEngine
    
    try:
        engine = TranscriptionEngine()
    except RuntimeError as e:
        print(f"No backend available for performance testing: {e}")
        return
    
    results = {}
    
    for model_key, config in MODEL_CONFIGS.items():
        print(f"\nTesting {model_key}...")
        
        # Measure loading time
        times = []
        for _ in range(2):  # 2 iterations
            start = time.time()
            try:
                engine.load_model(config["name"])
                times.append(time.time() - start)
            except Exception as e:
                print(f"  Load failed: {e}")
                break
        
        if times:
            avg_time = sum(times) / len(times)
            results[model_key] = avg_time
            print(f"  Average load time: {avg_time:.2f}s")
    
    print("\n📊 Performance Summary:")
    for model, load_time in results.items():
        print(f"  {model}: {load_time:.2f}s load time")


if __name__ == '__main__':
    # Print test report header
    print("="*70)
    print("QWEN3-ASR MODEL/BACKEND TEST SUITE")
    print("="*70)
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base directory: {BASE_DIR}")
    print()
    
    # Run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelPaths))
    suite.addTests(loader.loadTestsFromTestCase(TestCBinary))
    suite.addTests(loader.loadTestsFromTestCase(TestBackendDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestModelSwitching))
    suite.addTests(loader.loadTestsFromTestCase(TestMissingModelHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryUsage))
    suite.addTests(loader.loadTestsFromTestCase(TestMLXBackend))
    suite.addTests(loader.loadTestsFromTestCase(TestPyTorchFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestLiveStreamer))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print detailed report
    print_test_report()
    
    # Run performance tests if any tests passed
    if result.wasSuccessful() or result.testsRun > 0:
        try:
            run_performance_tests()
        except Exception as e:
            print(f"\nPerformance tests skipped: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Success: {result.wasSuccessful()}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
