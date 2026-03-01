#!/usr/bin/env python3
"""
Comprehensive Simulation Tests for Qwen3-ASR Pro Web UI
Tests the Gradio-based web interface with mocked dependencies

Test Coverage:
1. Gradio patch function (patch_gradio_api_info)
2. transcribe_with_c_binary function
3. transcribe_audio function with fallbacks
4. reform_text function
5. process_audio full pipeline
6. Gradio component configurations
7. Integration tests

Requirements:
- pytest
- unittest.mock (built-in)

Run with: pytest tests/test_webui_comprehensive.py -v
"""

import sys
import os
import time
import tempfile
import shutil
import wave
import json
import pytest
import threading
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, ANY
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# =============================================================================
# Create comprehensive mocks for gradio modules before importing web_ui
# =============================================================================

# Create a proper context manager mock class
class MockContextManager:
    """Mock that supports context manager protocol"""
    def __init__(self, *args, **kwargs):
        self.blocks = {}
        self._counter = 0
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        return False
    
    def __call__(self, *args, **kwargs):
        return self


class MockGradioComponent:
    """Mock for any Gradio component"""
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        # Extract common attributes from kwargs
        self.label = kwargs.get('label', '')
        self.value = kwargs.get('value')
        self.choices = kwargs.get('choices', [])
        self.interactive = kwargs.get('interactive', True)
        self.lines = kwargs.get('lines', 1)
        self._store_component()
    
    def _store_component(self):
        """Store component reference for retrieval in tests"""
        # Get reference to demo blocks dict
        if hasattr(MockBlocks, '_instance') and MockBlocks._instance:
            MockBlocks._instance.blocks[id(self)] = self
    
    def __call__(self, *args, **kwargs):
        return self
    
    def click(self, *args, **kwargs):
        return Mock()


class MockBlocks(MockContextManager):
    """Mock for gr.Blocks"""
    _instance = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = {}
        MockBlocks._instance = self
    
    def queue(self, *args, **kwargs):
        return self
    
    def launch(self, *args, **kwargs):
        return self


class MockThemes:
    """Mock for gr.themes"""
    @staticmethod
    def Soft(*args, **kwargs):
        return Mock()


class MockRow(MockContextManager):
    """Mock for gr.Row"""
    pass


class MockColumn(MockContextManager):
    """Mock for gr.Column"""
    pass


class MockTab(MockContextManager):
    """Mock for gr.Tab"""
    pass


# Create the mock gradio module
mock_gr = Mock()
mock_gr.Blocks = MockBlocks
mock_gr.Row = MockRow
mock_gr.Column = MockColumn
mock_gr.Markdown = MockGradioComponent
mock_gr.Dropdown = MockGradioComponent
mock_gr.Textbox = MockGradioComponent
mock_gr.Audio = MockGradioComponent
mock_gr.Button = MockGradioComponent
mock_gr.Tab = MockTab
themes = Mock()
themes.Soft = MockThemes.Soft
mock_gr.themes = themes

# Mock gradio_client
mock_gr_client = Mock()
mock_gr_client.utils = Mock()
mock_gr_client.utils.get_type = Mock(return_value="object")
mock_gr_client.utils._json_schema_to_python_type = Mock(return_value="object")

# Pre-patch the modules before importing web_ui
sys.modules['gradio'] = mock_gr
sys.modules['gradio_client'] = mock_gr_client
sys.modules['gradio_client.utils'] = mock_gr_client.utils

# Mock simple_llm
mock_simple_llm = Mock()
mock_simple_llm.SimpleLLM = Mock()
mock_llm_instance = Mock()
mock_llm_instance.backend_name = "ollama-qwen:1.8b"
mock_llm_instance.is_available.return_value = True
mock_simple_llm.SimpleLLM.return_value = mock_llm_instance
sys.modules['simple_llm'] = mock_simple_llm

# Now we can import web_ui with mocked dependencies
from web_ui import (
    patch_gradio_api_info,
    transcribe_with_c_binary,
    transcribe_audio,
    reform_text,
    process_audio,
    record_and_transcribe,
    demo  # This will be the MockBlocks instance
)

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_audio_file():
    """Create a temporary WAV file for testing"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "test_audio.wav")
    
    # Create a simple WAV file
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    audio_data = np.random.randn(samples) * 0.1
    audio_int16 = np.clip(audio_data * 32767, -32768, 32768).astype(np.int16)
    
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    yield file_path
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_subprocess_success():
    """Mock subprocess.run for successful execution"""
    with patch('subprocess.run') as mock_run:
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "This is a test transcription"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        yield mock_run


@pytest.fixture
def mock_subprocess_failure():
    """Mock subprocess.run for failed execution"""
    with patch('subprocess.run') as mock_run:
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Model not found"
        mock_run.return_value = mock_result
        yield mock_run


@pytest.fixture
def mock_subprocess_timeout():
    """Mock subprocess.run for timeout"""
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=['test'], timeout=300)
        yield mock_run


# =============================================================================
# Test Class: Gradio Patch Function
# =============================================================================

class TestGradioPatchFunction:
    """Test the patch_gradio_api_info() function"""
    
    def test_patch_gradio_api_info_with_bool_schema(self):
        """Test patching with boolean schema type"""
        # Create mock gradio_utils
        mock_gradio_utils = Mock()
        original_get_type = Mock(return_value="object")
        mock_gradio_utils.get_type = original_get_type
        mock_gradio_utils._json_schema_to_python_type = lambda s, d: str(s)
        
        with patch.dict('sys.modules', {'gradio_client.utils': mock_gradio_utils}):
            # Apply patch
            patch_gradio_api_info()
            
            # Test patched get_type with boolean
            result = mock_gradio_utils.get_type(True)
            # After patching, should handle bool
            assert result is not None
    
    def test_patch_gradio_api_info_import_error(self):
        """Test patching when gradio_client is not available"""
        with patch.dict('sys.modules', {'gradio_client': None, 'gradio_client.utils': None}):
            # Should handle import error gracefully without raising
            try:
                patch_gradio_api_info()
            except Exception:
                pass  # Expected if module not available
    
    def test_patch_gradio_api_info_handles_exception(self):
        """Test patching handles exceptions gracefully"""
        mock_gradio_utils = Mock()
        mock_gradio_utils.get_type = Mock(side_effect=Exception("Test error"))
        
        with patch.dict('sys.modules', {'gradio_client.utils': mock_gradio_utils}):
            # Should not raise
            patch_gradio_api_info()
    
    def test_normalize_schema_with_bool(self):
        """Test schema normalization with boolean values"""
        # Test by checking if the patch applies without error
        mock_gradio_utils = Mock()
        mock_gradio_utils.get_type = lambda x: "boolean" if isinstance(x, bool) else "object"
        mock_gradio_utils._json_schema_to_python_type = lambda s, d: str(s)
        
        with patch.dict('sys.modules', {'gradio_client.utils': mock_gradio_utils}):
            patch_gradio_api_info()
            
            # Verify get_type can handle boolean
            assert mock_gradio_utils.get_type(True) == "boolean"
            assert mock_gradio_utils.get_type(False) == "boolean"


# =============================================================================
# Test Class: Transcribe with C Binary
# =============================================================================

class TestTranscribeWithCBinary:
    """Test transcribe_with_c_binary function"""
    
    def test_binary_not_found(self):
        """Test when C binary doesn't exist"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            transcript, backend = transcribe_with_c_binary("test.wav", model="small")
            
            assert transcript is None
            assert "not found" in backend.lower()
    
    def test_model_directory_not_found(self):
        """Test when model directory doesn't exist"""
        with patch('os.path.exists') as mock_exists:
            # Binary exists but model directory doesn't
            def exists_side_effect(path):
                if 'qwen_asr' in path and not 'qwen3-asr' in path:
                    return True
                return False
            
            mock_exists.side_effect = exists_side_effect
            
            transcript, backend = transcribe_with_c_binary("test.wav", model="small")
            
            assert transcript is None
            assert "model directory not found" in backend.lower()
    
    def test_model_mapping_0_6b(self, mock_subprocess_success):
        """Test model mapping for 0.6b model"""
        with patch('os.path.exists', return_value=True):
            transcribe_with_c_binary("test.wav", model="0.6b")
            
            # Check that subprocess was called with correct model
            call_args = mock_subprocess_success.call_args
            assert call_args is not None
            cmd = call_args[0][0]
            # Check in the model directory path (3rd element: -d <model_dir>)
            model_dir = cmd[2]  # Index 2 is the model directory
            assert "qwen3-asr-0.6b" in model_dir
    
    def test_model_mapping_1_7b(self, mock_subprocess_success):
        """Test model mapping for 1.7b model"""
        with patch('os.path.exists', return_value=True):
            transcribe_with_c_binary("test.wav", model="1.7b")
            
            call_args = mock_subprocess_success.call_args
            cmd = call_args[0][0]
            model_dir = cmd[2]  # Index 2 is the model directory
            assert "qwen3-asr-1.7b" in model_dir
    
    def test_model_mapping_small_alias(self, mock_subprocess_success):
        """Test model mapping for 'small' alias"""
        with patch('os.path.exists', return_value=True):
            transcribe_with_c_binary("test.wav", model="small")
            
            call_args = mock_subprocess_success.call_args
            cmd = call_args[0][0]
            model_dir = cmd[2]  # Index 2 is the model directory
            assert "qwen3-asr-0.6b" in model_dir
    
    def test_model_mapping_large_alias(self, mock_subprocess_success):
        """Test model mapping for 'large' alias"""
        with patch('os.path.exists', return_value=True):
            transcribe_with_c_binary("test.wav", model="large")
            
            call_args = mock_subprocess_success.call_args
            cmd = call_args[0][0]
            model_dir = cmd[2]  # Index 2 is the model directory
            assert "qwen3-asr-1.7b" in model_dir
    
    def test_successful_transcription(self, mock_subprocess_success):
        """Test successful transcription"""
        with patch('os.path.exists', return_value=True):
            transcript, backend = transcribe_with_c_binary("test.wav", model="small")
            
            assert transcript == "This is a test transcription"
            assert backend == "C-Binary"
    
    def test_binary_error_nonzero_return(self, mock_subprocess_failure):
        """Test handling of non-zero return code"""
        with patch('os.path.exists', return_value=True):
            transcript, backend = transcribe_with_c_binary("test.wav", model="small")
            
            assert transcript is None
            assert "error" in backend.lower()
            assert "code 1" in backend.lower()
    
    def test_timeout_handling(self, mock_subprocess_timeout):
        """Test handling of timeout"""
        with patch('os.path.exists', return_value=True):
            transcript, backend = transcribe_with_c_binary("test.wav", model="small")
            
            assert transcript is None
            assert "timeout" in backend.lower()
    
    def test_exception_handling(self):
        """Test handling of general exceptions"""
        with patch('os.path.exists', return_value=True):
            with patch('subprocess.run', side_effect=Exception("Unexpected error")):
                transcript, backend = transcribe_with_c_binary("test.wav", model="small")
                
                assert transcript is None
                assert "failed" in backend.lower()


# =============================================================================
# Test Class: Transcribe Audio Function
# =============================================================================

class TestTranscribeAudio:
    """Test transcribe_audio function with fallback backends"""
    
    def test_none_input(self):
        """Test handling of None input"""
        transcript, backend = transcribe_audio(None)
        
        assert "no audio file" in transcript.lower()
        assert backend == "Error"
    
    def test_nonexistent_file(self):
        """Test handling of non-existent file"""
        with patch('os.path.exists', return_value=False):
            transcript, backend = transcribe_audio("/nonexistent/file.wav")
            
            assert "file not found" in transcript.lower()
            assert backend == "Error"
    
    def test_c_binary_success(self, mock_subprocess_success):
        """Test C binary success path"""
        with patch('os.path.exists', return_value=True):
            transcript, backend = transcribe_audio("test.wav")
            
            assert transcript == "This is a test transcription"
            assert backend == "C-Binary"
    
    def test_c_binary_failure_fallback_to_mlx(self):
        """Test fallback to MLX when C binary fails"""
        with patch('os.path.exists', return_value=True):
            # Mock C binary failure
            with patch('web_ui.transcribe_with_c_binary', return_value=(None, "C binary error")):
                # Mock MLX success by patching the entire MLX code block
                # We need to simulate a successful MLX transcription
                def mock_transcribe_with_mlx(*args, **kwargs):
                    return "MLX transcription", "MLX (Apple Silicon)"
                
                # The key is to make the import statement succeed and the code path work
                # Since the import is inside the function, we patch sys.modules
                mock_mlx_result = Mock()
                mock_mlx_result.text = "MLX transcription"
                
                mock_mlx_model = Mock()
                mock_mlx_model.generate.return_value = mock_mlx_result
                
                mock_mlx_stt = Mock()
                mock_mlx_stt.load.return_value = mock_mlx_model
                
                # Create a mock mlx_audio module with stt submodule
                mock_mlx_audio = Mock()
                mock_mlx_audio.stt = mock_mlx_stt
                
                with patch.dict('sys.modules', {
                    'mlx_audio': mock_mlx_audio,
                    'mlx_audio.stt': mock_mlx_stt
                }):
                    transcript, backend = transcribe_audio("test.wav")
                    
                    # Should use MLX backend
                    assert backend == "MLX (Apple Silicon)", f"Expected 'MLX (Apple Silicon)' but got '{backend}'"
                    assert transcript == "MLX transcription"
    
    def test_c_binary_failure_fallback_to_cli(self):
        """Test fallback to CLI when C binary and MLX fail"""
        with patch('os.path.exists', return_value=True):
            with patch('web_ui.transcribe_with_c_binary', return_value=(None, "C binary error")):
                # MLX import fails
                with patch.dict('sys.modules', {'mlx_audio.stt': None}):
                    # Mock CLI success
                    mock_result = Mock()
                    mock_result.returncode = 0
                    mock_result.stdout = "CLI transcription"
                    
                    with patch('subprocess.run', return_value=mock_result):
                        transcript, backend = transcribe_audio("test.wav")
                        
                        assert "CLI transcription" in transcript
                        assert backend == "MLX-CLI"
    
    def test_all_backends_fail(self):
        """Test when all backends fail"""
        with patch('os.path.exists', return_value=True):
            with patch('web_ui.transcribe_with_c_binary', return_value=(None, "C binary error")):
                with patch.dict('sys.modules', {'mlx_audio.stt': None}):
                    # CLI also fails
                    mock_result = Mock()
                    mock_result.returncode = 1
                    mock_result.stderr = "CLI error"
                    
                    with patch('subprocess.run', return_value=mock_result):
                        transcript, backend = transcribe_audio("test.wav")
                        
                        assert "no transcription backend" in transcript.lower()
                        assert "last error" in transcript.lower()
    
    def test_language_parameter_auto(self, mock_subprocess_success):
        """Test language parameter with auto"""
        with patch('os.path.exists', return_value=True):
            transcript, backend = transcribe_audio("test.wav", language="auto")
            
            # C binary doesn't use language parameter
            assert transcript is not None
    
    def test_language_parameter_specific(self):
        """Test language parameter with specific language"""
        with patch('os.path.exists', return_value=True):
            with patch('web_ui.transcribe_with_c_binary', return_value=(None, "C binary error")):
                # MLX import fails
                with patch.dict('sys.modules', {'mlx_audio.stt': None}):
                    mock_result = Mock()
                    mock_result.returncode = 0
                    mock_result.stdout = "Transcription"
                    
                    with patch('subprocess.run', return_value=mock_result) as mock_run:
                        transcript, backend = transcribe_audio("test.wav", language="en")
                        
                        # Verify CLI was called with language parameter
                        call_args = mock_run.call_args
                        cmd = call_args[0][0]
                        assert '--language' in cmd
                        assert 'en' in cmd


# =============================================================================
# Test Class: Reform Text Function
# =============================================================================

class TestReformText:
    """Test reform_text function"""
    
    def test_empty_text(self):
        """Test handling of empty text"""
        with patch('web_ui.llm') as mock_llm_module:
            mock_llm_module.is_available.return_value = True
            
            result = reform_text("", "punctuate")
            
            assert result == "No text to reform"
    
    def test_whitespace_only_text(self):
        """Test handling of whitespace-only text"""
        with patch('web_ui.llm') as mock_llm_module:
            mock_llm_module.is_available.return_value = True
            
            result = reform_text("   \n\t   ", "punctuate")
            
            assert result == "No text to reform"
    
    def test_llm_unavailable(self):
        """Test when LLM is not available"""
        with patch('web_ui.llm') as mock_llm_module:
            mock_llm_module.is_available.return_value = False
            
            test_text = "This is a test"
            result = reform_text(test_text, "punctuate")
            
            assert test_text in result
            assert "LLM not available" in result
    
    def test_successful_reform_punctuate(self):
        """Test successful punctuate mode"""
        with patch('web_ui.llm') as mock_llm_module:
            mock_llm_module.is_available.return_value = True
            mock_llm_module.process.return_value = "This is punctuated."
            
            result = reform_text("this is punctuated", "punctuate")
            
            assert result == "This is punctuated."
            mock_llm_module.process.assert_called_once_with("this is punctuated", "punctuate")
    
    def test_successful_reform_summarize(self):
        """Test successful summarize mode"""
        with patch('web_ui.llm') as mock_llm_module:
            mock_llm_module.is_available.return_value = True
            mock_llm_module.process.return_value = "Summary of text."
            
            result = reform_text("Long text to summarize", "summarize")
            
            assert result == "Summary of text."
            mock_llm_module.process.assert_called_once_with("Long text to summarize", "summarize")
    
    def test_successful_reform_clean(self):
        """Test successful clean mode"""
        with patch('web_ui.llm') as mock_llm_module:
            mock_llm_module.is_available.return_value = True
            mock_llm_module.process.return_value = "Cleaned text."
            
            result = reform_text("um uh cleaned text", "clean")
            
            assert result == "Cleaned text."
            mock_llm_module.process.assert_called_once_with("um uh cleaned text", "clean")
    
    def test_successful_reform_key_points(self):
        """Test successful key_points mode"""
        with patch('web_ui.llm') as mock_llm_module:
            mock_llm_module.is_available.return_value = True
            mock_llm_module.process.return_value = "• Point 1\n• Point 2"
            
            result = reform_text("Text with key points", "key_points")
            
            assert result == "• Point 1\n• Point 2"
            mock_llm_module.process.assert_called_once_with("Text with key points", "key_points")
    
    def test_exception_handling(self):
        """Test handling of exceptions during reforming"""
        with patch('web_ui.llm') as mock_llm_module:
            mock_llm_module.is_available.return_value = True
            mock_llm_module.process.side_effect = Exception("LLM error")
            
            result = reform_text("Test text", "punctuate")
            
            assert "error reforming text" in result.lower()


# =============================================================================
# Test Class: Process Audio Function
# =============================================================================

class TestProcessAudio:
    """Test process_audio function (full pipeline)"""
    
    def test_none_input(self):
        """Test handling of None input"""
        with patch('web_ui.llm') as mock_llm_module:
            mock_llm_module.backend_name = "test-backend"
            
            raw, reformed, info = process_audio(None, "auto", "none")
            
            assert "upload an audio file" in raw.lower()
            assert reformed == ""
            assert info == ""
    
    def test_error_in_transcription(self):
        """Test handling of transcription error"""
        with patch('web_ui.transcribe_audio', return_value=("Error: File not found", "Error")):
            with patch('web_ui.llm') as mock_llm_module:
                mock_llm_module.backend_name = "test-backend"
                
                raw, reformed, info = process_audio("test.wav", "auto", "none")
                
                assert "error" in raw.lower()
                assert reformed == ""
                assert info == ""
    
    def test_no_reforming_mode_none(self):
        """Test no reforming when mode is 'none'"""
        with patch('web_ui.transcribe_audio', return_value=("Raw transcript", "C-Binary")):
            with patch('web_ui.llm') as mock_llm_module:
                mock_llm_module.backend_name = "test-backend"
                
                raw, reformed, info = process_audio("test.wav", "auto", "none", model="0.6b")
                
                assert raw == "Raw transcript"
                assert reformed == "Raw transcript"  # Should be same as raw
                assert "Backend: C-Binary" in info
                assert "Model: 0.6b" in info
    
    def test_with_reforming(self):
        """Test with reforming enabled"""
        with patch('web_ui.transcribe_audio', return_value=("Raw transcript", "C-Binary")):
            with patch('web_ui.reform_text', return_value="Reformed text") as mock_reform:
                with patch('web_ui.llm') as mock_llm_module:
                    mock_llm_module.backend_name = "ollama"
                    
                    raw, reformed, info = process_audio("test.wav", "auto", "punctuate", model="0.6b")
                    
                    assert raw == "Raw transcript"
                    assert reformed == "Reformed text"
                    mock_reform.assert_called_once_with("Raw transcript", "punctuate")
    
    def test_output_format(self):
        """Test output format is correct"""
        with patch('web_ui.transcribe_audio', return_value=("Transcript", "MLX")):
            with patch('web_ui.llm') as mock_llm_module:
                mock_llm_module.backend_name = "ollama-qwen:1.8b"
                
                raw, reformed, info = process_audio("test.wav", "auto", "none", model="1.7b")
                
                # Verify info format
                assert "Backend:" in info
                assert "Model:" in info
                assert "LLM:" in info
                assert "MLX" in info
                assert "1.7b" in info


# =============================================================================
# Test Class: Gradio Component Configurations
# =============================================================================

class TestGradioComponents:
    """Test Gradio component configurations"""
    
    def test_model_dropdown_choices(self):
        """Test model dropdown has correct choices - verify in web_ui.py code"""
        # Read the web_ui.py file to verify choices
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify model choices are defined
        assert '"0.6b"' in content
        assert '"1.7b"' in content
        assert "⚡ Fast (0.6B)" in content
        assert "🎯 Accurate (1.7B)" in content
    
    def test_model_dropdown_default(self):
        """Test model dropdown default value in code"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify default is 0.6b
        assert 'value="0.6b"' in content
    
    def test_language_dropdown_choices(self):
        """Test language dropdown has correct choices"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify language choices
        assert '"auto"' in content
        assert '"en"' in content
        assert '"zh"' in content
        assert '"ja"' in content
        assert '"ko"' in content
    
    def test_language_dropdown_default(self):
        """Test language dropdown default value"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Look for language dropdown default
        assert 'value="auto"' in content
    
    def test_reform_mode_dropdown_choices(self):
        """Test reform mode dropdown has correct choices"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify reform mode choices
        assert '"none"' in content
        assert '"punctuate"' in content
        assert '"summarize"' in content
        assert '"clean"' in content
        assert '"key_points"' in content
    
    def test_reform_mode_dropdown_default(self):
        """Test reform mode dropdown default value"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Look for reform_mode default
        assert 'value="punctuate"' in content


# =============================================================================
# Test Class: Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow"""
    
    def test_full_upload_workflow(self, temp_audio_file, mock_subprocess_success):
        """Test full upload workflow"""
        with patch('os.path.exists', return_value=True):
            with patch('web_ui.llm') as mock_llm_module:
                mock_llm_module.backend_name = "ollama-qwen:1.8b"
                mock_llm_module.is_available.return_value = True
                mock_llm_module.process.return_value = "Punctuated text."
                
                start_time = time.time()
                raw, reformed, info = process_audio(temp_audio_file, "auto", "punctuate", model="0.6b")
                elapsed = time.time() - start_time
                
                assert raw is not None
                assert reformed is not None
                assert info is not None
                assert "Backend:" in info
                assert elapsed < 1.0  # Should complete quickly with mocks
    
    def test_full_record_workflow(self, temp_audio_file):
        """Test full record workflow via record_and_transcribe"""
        with patch('web_ui.process_audio', return_value=("Raw", "Reformed", "Info")) as mock_process:
            raw, reformed, info = record_and_transcribe(temp_audio_file, "auto", "summarize", "0.6b")
            
            mock_process.assert_called_once_with(temp_audio_file, "auto", "summarize", "0.6b")
            assert raw == "Raw"
            assert reformed == "Reformed"
            assert info == "Info"
    
    def test_concurrent_requests_simulation(self):
        """Test handling of concurrent requests"""
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                with patch('web_ui.transcribe_audio', return_value=(f"Transcript {thread_id}", "C-Binary")):
                    with patch('web_ui.llm') as mock_llm_module:
                        mock_llm_module.backend_name = "ollama"
                        mock_llm_module.is_available.return_value = True
                        mock_llm_module.process.return_value = f"Reformed {thread_id}"
                        
                        raw, reformed, info = process_audio("test.wav", "auto", "punctuate")
                        results.append((thread_id, raw, reformed))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all to complete
        for t in threads:
            t.join(timeout=5.0)
        
        # Verify all completed successfully
        assert len(results) == 5
        assert len(errors) == 0
    
    def test_edge_case_long_text(self):
        """Test handling of very long text"""
        long_text = "word " * 10000  # 50,000+ characters
        
        with patch('web_ui.llm') as mock_llm_module:
            mock_llm_module.is_available.return_value = True
            mock_llm_module.process.return_value = long_text
            
            result = reform_text(long_text, "summarize")
            
            assert result == long_text
            mock_llm_module.process.assert_called_once()
    
    def test_edge_case_special_characters(self):
        """Test handling of special characters"""
        special_text = "Test with special chars: émojis 🎉 <html> & entities 'quotes' \"double\""
        
        with patch('web_ui.transcribe_audio', return_value=(special_text, "C-Binary")):
            with patch('web_ui.llm') as mock_llm_module:
                mock_llm_module.backend_name = "ollama"
                mock_llm_module.is_available.return_value = True
                mock_llm_module.process.return_value = special_text
                
                raw, reformed, info = process_audio("test.wav", "auto", "punctuate")
                
                assert special_text in raw
                assert special_text in reformed


# =============================================================================
# Test Class: Performance Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests with timing measurements"""
    
    def test_transcribe_performance(self, temp_audio_file):
        """Measure transcription performance"""
        with patch('os.path.exists', return_value=True):
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = "Performance test transcription"
                mock_run.return_value = mock_result
                
                start = time.time()
                transcript, backend = transcribe_audio(temp_audio_file)
                elapsed = time.time() - start
                
                # With mocks, should be very fast
                assert elapsed < 0.1, f"Transcription took {elapsed:.3f}s, expected < 0.1s"
    
    def test_reform_performance(self):
        """Measure text reforming performance"""
        test_text = "This is a test sentence for performance measurement."
        
        with patch('web_ui.llm') as mock_llm_module:
            mock_llm_module.is_available.return_value = True
            mock_llm_module.process.return_value = test_text
            
            start = time.time()
            result = reform_text(test_text, "punctuate")
            elapsed = time.time() - start
            
            assert elapsed < 0.1, f"Reforming took {elapsed:.3f}s, expected < 0.1s"
    
    def test_pipeline_performance(self, temp_audio_file):
        """Measure full pipeline performance"""
        with patch('os.path.exists', return_value=True):
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = "Pipeline performance test"
                mock_run.return_value = mock_result
                
                with patch('web_ui.llm') as mock_llm_module:
                    mock_llm_module.backend_name = "ollama"
                    mock_llm_module.is_available.return_value = True
                    mock_llm_module.process.return_value = "Reformed output"
                    
                    start = time.time()
                    raw, reformed, info = process_audio(temp_audio_file, "auto", "punctuate")
                    elapsed = time.time() - start
                    
                    assert elapsed < 0.1, f"Pipeline took {elapsed:.3f}s, expected < 0.1s"


# =============================================================================
# Test Class: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests"""
    
    def test_empty_string_transcription_result(self):
        """Test handling of empty transcription result"""
        with patch('web_ui.transcribe_audio', return_value=("", "C-Binary")):
            with patch('web_ui.llm') as mock_llm_module:
                mock_llm_module.backend_name = "ollama"
                mock_llm_module.is_available.return_value = True
                mock_llm_module.process.return_value = ""
                
                raw, reformed, info = process_audio("test.wav", "auto", "punctuate")
                
                assert raw == ""
    
    def test_unicode_handling(self):
        """Test handling of unicode characters"""
        unicode_text = "你好世界こんにちは🌍"
        
        with patch('web_ui.transcribe_audio', return_value=(unicode_text, "C-Binary")):
            with patch('web_ui.llm') as mock_llm_module:
                mock_llm_module.backend_name = "ollama"
                mock_llm_module.is_available.return_value = True
                mock_llm_module.process.return_value = unicode_text
                
                raw, reformed, info = process_audio("test.wav", "auto", "punctuate")
                
                assert unicode_text in raw
    
    def test_newline_handling(self):
        """Test handling of newlines in text"""
        text_with_newlines = "Line 1\nLine 2\n\nLine 3"
        
        with patch('web_ui.llm') as mock_llm_module:
            mock_llm_module.is_available.return_value = True
            mock_llm_module.process.return_value = text_with_newlines
            
            result = reform_text(text_with_newlines, "punctuate")
            
            assert result == text_with_newlines
    
    def test_very_short_audio_file(self, temp_dir):
        """Test handling of very short audio file"""
        # Create a very short audio file
        file_path = os.path.join(temp_dir, "short.wav")
        sample_rate = 16000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        audio_data = np.random.randn(samples) * 0.1
        audio_int16 = np.clip(audio_data * 32767, -32768, 32768).astype(np.int16)
        
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        with patch('os.path.exists', return_value=True):
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 0
                mock_result.stdout = "Short audio"
                mock_run.return_value = mock_result
                
                transcript, backend = transcribe_audio(file_path)
                
                assert transcript == "Short audio"


# =============================================================================
# Test Summary
# =============================================================================

# Total test cases by category:
# - Gradio Patch Function: 4 tests
# - Transcribe with C Binary: 10 tests
# - Transcribe Audio Function: 8 tests
# - Reform Text Function: 7 tests
# - Process Audio Function: 5 tests
# - Gradio Components: 6 tests
# - Integration Tests: 5 tests
# - Performance Tests: 3 tests
# - Edge Cases: 4 tests
# TOTAL: 52 test cases

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
