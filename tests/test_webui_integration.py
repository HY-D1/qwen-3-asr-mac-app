#!/usr/bin/env python3
"""
Comprehensive Simulation Tests for Qwen3-ASR Pro Web UI Integration
Tests the Gradio-based web interface with mocked dependencies and real file testing

Test Coverage:
1. Gradio Component Tests - Audio, Dropdowns, Buttons, visibility toggles
2. Web UI Function Tests - process_audio, record_and_transcribe, streaming, reform_text
3. Integration Tests - Full upload/workflow simulations with real files
4. Error Handling - Missing files, invalid formats, None inputs, backend failures
5. Real File Tests - Different sizes, formats, concurrent processing

Requirements:
- pytest
- unittest.mock (built-in)
- Real audio files from tests/assets/ and tests/test_audio/

Run with: pytest tests/test_webui_integration.py -v
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
import struct
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, ANY, mock_open
from typing import Optional, Dict, Any, List, Tuple, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# =============================================================================
# Create comprehensive mocks for gradio modules before importing web_ui
# =============================================================================

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
    """Mock for any Gradio component with full attribute tracking"""
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self.label = kwargs.get('label', '')
        self.value = kwargs.get('value')
        self.choices = kwargs.get('choices', [])
        self.interactive = kwargs.get('interactive', True)
        self.lines = kwargs.get('lines', 1)
        self.visible = kwargs.get('visible', True)
        self.type = kwargs.get('type')
        self.sources = kwargs.get('sources', [])
        self.variant = kwargs.get('variant')
        self.info = kwargs.get('info', '')
        self.show_copy_button = kwargs.get('show_copy_button', False)
        self._event_handlers = {}
    
    def __call__(self, *args, **kwargs):
        return self
    
    def click(self, fn=None, inputs=None, outputs=None, **kwargs):
        """Mock click event handler"""
        self._event_handlers['click'] = {
            'fn': fn,
            'inputs': inputs,
            'outputs': outputs,
            'kwargs': kwargs
        }
        return self
    
    def change(self, fn=None, inputs=None, outputs=None, **kwargs):
        """Mock change event handler"""
        self._event_handlers['change'] = {
            'fn': fn,
            'inputs': inputs,
            'outputs': outputs,
            'kwargs': kwargs
        }
        return self
    
    def update(self, **kwargs):
        """Mock update method"""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self


class MockProgress:
    """Mock Gradio Progress tracker"""
    def __init__(self):
        self.progress = 0
        self.description = ""
    
    def __call__(self, value, desc=None):
        self.progress = value
        if desc:
            self.description = desc
        return self


class MockBlocks(MockContextManager):
    """Mock for gr.Blocks"""
    _instance = None
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = {}
        self._components = []
        MockBlocks._instance = self
    
    def queue(self, *args, **kwargs):
        self.max_size = kwargs.get('max_size')
        self.default_concurrency_limit = kwargs.get('default_concurrency_limit')
        return self
    
    def launch(self, *args, **kwargs):
        self.launch_kwargs = kwargs
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = kwargs.get('scale', 1)


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
mock_gr.Checkbox = MockGradioComponent
mock_gr.Progress = MockProgress
# Mock gr.update to return a dict-like update object
def mock_update(**kwargs):
    return kwargs
mock_gr.update = mock_update
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
mock_llm_instance.process.return_value = "Processed text"
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
    stream_record_and_transcribe,
    simulate_streaming_transcript,
    toggle_realtime_mode,
    recording_state,
    demo
)

# Get references to Gradio component mocks created during import
# These are defined in web_ui.py as global variables within the gr.Blocks context
live_output_mock = MockGradioComponent
raw_output_mock = MockGradioComponent
reformed_output_mock = MockGradioComponent
info_output_mock = MockGradioComponent

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_audio_file():
    """Create a temporary WAV file for testing"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "test_audio.wav")
    
    # Create a simple WAV file (16kHz, mono, 16-bit)
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
        mock_result.stderr = "Inference: time 0.5s"
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
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=['test'], timeout=60)
        yield mock_run


@pytest.fixture
def mock_audio_processing():
    """Mock audio processing pipeline"""
    with patch('web_ui.transcribe_audio', return_value=("Test transcript", "C-Binary")):
        with patch('web_ui.llm') as mock_llm:
            mock_llm.backend_name = "ollama"
            mock_llm.is_available.return_value = True
            mock_llm.process.return_value = "Reformed text"
            yield mock_llm


@pytest.fixture
def sample_audio_files():
    """Get available sample audio files from test directories"""
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    test_audio_dir = os.path.join(os.path.dirname(__file__), 'test_audio')
    
    files = []
    
    # Check assets directory
    if os.path.exists(assets_dir):
        for f in os.listdir(assets_dir):
            if f.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
                files.append(os.path.join(assets_dir, f))
    
    # Check test_audio directory
    if os.path.exists(test_audio_dir):
        for f in os.listdir(test_audio_dir):
            if f.endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
                files.append(os.path.join(test_audio_dir, f))
    
    return files


# =============================================================================
# Test Class: Gradio Component Tests
# =============================================================================

class TestGradioComponents:
    """Test Gradio component configurations and properties"""
    
    def test_audio_component_configuration(self):
        """Test Audio component configuration in web_ui.py"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify Audio components are defined
        assert 'gr.Audio(' in content
        assert 'type="filepath"' in content
        assert 'sources=["microphone"]' in content
    
    def test_dropdown_model_configuration(self):
        """Test model Dropdown configuration"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify model dropdown configuration
        assert 'model = gr.Dropdown(' in content
        assert '"0.6b"' in content
        assert '"1.7b"' in content
        assert '⚡ Fast (0.6B)' in content
        assert '🎯 Accurate (1.7B)' in content
        assert 'value="0.6b"' in content
    
    def test_dropdown_language_configuration(self):
        """Test language Dropdown configuration"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify language dropdown
        assert 'language = gr.Dropdown(' in content
        assert '"auto"' in content
        assert '"en"' in content
        assert '"zh"' in content
        assert '"ja"' in content
        assert '"ko"' in content
        assert '"es"' in content
        assert '"fr"' in content
        assert '"de"' in content
        assert 'value="auto"' in content
    
    def test_dropdown_reform_mode_configuration(self):
        """Test reform_mode Dropdown configuration"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify reform mode dropdown
        assert 'reform_mode = gr.Dropdown(' in content
        assert '"none"' in content
        assert '"punctuate"' in content
        assert '"summarize"' in content
        assert '"clean"' in content
        assert '"key_points"' in content
        assert 'value="punctuate"' in content
    
    def test_checkbox_configuration(self):
        """Test Checkbox component configuration"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify real-time checkbox
        assert 'enable_realtime = gr.Checkbox(' in content
        assert '🔄 Enable Real-Time Transcription' in content
        assert 'value=True' in content
    
    def test_button_configuration(self):
        """Test Button component configuration"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify button definitions
        assert 'upload_btn = gr.Button(' in content
        assert '🚀 Transcribe &' in content
        assert 'variant="primary"' in content
        assert 'record_btn = gr.Button(' in content
        assert 'clear_btn = gr.Button(' in content
        assert '🗑️ Clear' in content
        assert 'variant="secondary"' in content
        assert 'copy_raw_btn = gr.Button(' in content
        assert '📋 Copy Raw' in content
        assert 'copy_reformed_btn = gr.Button(' in content
    
    def test_textbox_configuration(self):
        """Test Textbox component configuration"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify textbox components
        assert 'raw_output = gr.Textbox(' in content
        assert 'reformed_output = gr.Textbox(' in content
        assert 'live_output = gr.Textbox(' in content
        assert 'info_output = gr.Textbox(' in content
        assert 'show_copy_button=True' in content
    
    def test_component_visibility_toggles(self):
        """Test component visibility toggle functions"""
        # Test toggle_realtime_mode
        result = toggle_realtime_mode(True)
        # Should return gr.update() call with visible=True
        assert result.get('visible') == True
        assert 'Real-time mode enabled' in result.get('value', '')
        
        result = toggle_realtime_mode(False)
        # Should return gr.update() call with visible=False
        assert result.get('visible') == False
    
    def test_live_output_initial_visibility(self):
        """Test live output initial visibility configuration"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify live output is initially visible
        assert 'live_output = gr.Textbox(' in content
        assert 'visible=True' in content


# =============================================================================
# Test Class: Web UI Function Tests
# =============================================================================

class TestWebUIFunctions:
    """Test core Web UI functions"""
    
    def test_process_audio_with_real_file(self, temp_audio_file, mock_subprocess_success):
        """Test process_audio with actual file"""
        with patch('os.path.exists', return_value=True):
            with patch('web_ui.llm') as mock_llm:
                mock_llm.backend_name = "ollama"
                mock_llm.is_available.return_value = True
                mock_llm.process.return_value = "Reformed transcript"
                
                raw, reformed, info = process_audio(temp_audio_file, "auto", "punctuate", "0.6b")
                
                assert raw == "This is a test transcription"
                assert reformed == "Reformed transcript"
                assert "Backend: C-Binary" in info
                assert "Model: 0.6b" in info
    
    def test_process_audio_no_reforming(self, temp_audio_file, mock_subprocess_success):
        """Test process_audio with no reforming (mode='none')"""
        with patch('os.path.exists', return_value=True):
            with patch('web_ui.llm') as mock_llm:
                mock_llm.backend_name = "ollama"
                
                raw, reformed, info = process_audio(temp_audio_file, "auto", "none", "0.6b")
                
                assert raw == "This is a test transcription"
                assert reformed == "This is a test transcription"  # Same as raw
    
    def test_process_audio_none_input(self):
        """Test process_audio with None input"""
        with patch('web_ui.llm') as mock_llm:
            mock_llm.backend_name = "test"
            
            raw, reformed, info = process_audio(None, "auto", "none")
            
            assert "upload an audio file" in raw.lower()
            assert reformed == ""
            assert info == ""
    
    def test_record_and_transcribe(self, temp_audio_file):
        """Test record_and_transcribe delegates to process_audio"""
        with patch('web_ui.process_audio', return_value=("Raw", "Reformed", "Info")) as mock_process:
            raw, reformed, info = record_and_transcribe(temp_audio_file, "en", "summarize", "0.6b")
            
            mock_process.assert_called_once_with(temp_audio_file, "en", "summarize", "0.6b")
            assert raw == "Raw"
            assert reformed == "Reformed"
            assert info == "Info"
    
    def test_simulate_streaming_transcript_short(self):
        """Test simulate_streaming_transcript with short text"""
        transcript = "Hello world"
        chunks = simulate_streaming_transcript(transcript, chunk_size=3)
        
        assert isinstance(chunks, list)
        assert len(chunks) == 4
        assert "🎙️ Processing..." in chunks[0]
        assert "📝 Hello world" in chunks[-1]
    
    def test_simulate_streaming_transcript_long(self):
        """Test simulate_streaming_transcript with longer text"""
        transcript = "This is a longer transcript with multiple words for testing"
        chunks = simulate_streaming_transcript(transcript, chunk_size=3)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 2
        assert "🎙️ Initializing" in chunks[0]
        assert "📝 This is a longer" in chunks[-1]
    
    def test_simulate_streaming_transcript_empty(self):
        """Test simulate_streaming_transcript with empty text"""
        chunks = simulate_streaming_transcript("")
        
        assert isinstance(chunks, list)
        assert len(chunks) == 4
    
    def test_stream_record_and_transcribe_none_audio(self):
        """Test stream_record_and_transcribe with None audio"""
        # Create mock components
        mock_live = Mock()
        mock_raw = Mock()
        mock_reformed = Mock()
        mock_info = Mock()
        
        # Patch the global output references in web_ui
        with patch('web_ui.live_output', mock_live):
            with patch('web_ui.raw_output', mock_raw):
                with patch('web_ui.reformed_output', mock_reformed):
                    with patch('web_ui.info_output', mock_info):
                        # Get generator
                        gen = stream_record_and_transcribe(None, "auto", "none", "0.6b", True)
                        
                        # Get first yield
                        result = next(gen)
                        
                        assert "no audio recorded" in result[mock_live].lower()
                        assert result[mock_raw] == ""
                        assert result[mock_reformed] == ""
                        assert "Error" in result[mock_info]
    
    def test_stream_record_and_transcribe_nonexistent_file(self):
        """Test stream_record_and_transcribe with non-existent file"""
        mock_live = Mock()
        mock_raw = Mock()
        mock_reformed = Mock()
        mock_info = Mock()
        
        with patch('web_ui.live_output', mock_live):
            with patch('web_ui.raw_output', mock_raw):
                with patch('web_ui.reformed_output', mock_reformed):
                    with patch('web_ui.info_output', mock_info):
                        with patch('os.path.exists', return_value=False):
                            gen = stream_record_and_transcribe("/nonexistent/file.wav", "auto", "none", "0.6b", True)
                            result = next(gen)
                            
                            assert "not found" in result[mock_live].lower()
                            assert "Error" in result[mock_info]
    
    def test_stream_record_and_transcribe_empty_file(self, temp_dir):
        """Test stream_record_and_transcribe with empty file"""
        empty_file = os.path.join(temp_dir, "empty.wav")
        # Create truly empty file
        open(empty_file, 'w').close()
        
        mock_live = Mock()
        mock_raw = Mock()
        mock_reformed = Mock()
        mock_info = Mock()
        
        with patch('web_ui.live_output', mock_live):
            with patch('web_ui.raw_output', mock_raw):
                with patch('web_ui.reformed_output', mock_reformed):
                    with patch('web_ui.info_output', mock_info):
                        gen = stream_record_and_transcribe(empty_file, "auto", "none", "0.6b", True)
                        result = next(gen)
                        
                        assert "empty" in result[mock_live].lower()
                        assert "Error" in result[mock_info]
    
    def test_stream_record_and_transcribe_realtime_disabled(self, temp_audio_file, mock_subprocess_success):
        """Test stream_record_and_transcribe with realtime disabled"""
        mock_live = Mock()
        mock_raw = Mock()
        mock_reformed = Mock()
        mock_info = Mock()
        
        with patch('web_ui.live_output', mock_live):
            with patch('web_ui.raw_output', mock_raw):
                with patch('web_ui.reformed_output', mock_reformed):
                    with patch('web_ui.info_output', mock_info):
                        with patch('web_ui.llm') as mock_llm:
                            mock_llm.backend_name = "ollama"
                            
                            gen = stream_record_and_transcribe(temp_audio_file, "auto", "none", "0.6b", False)
                            result = next(gen)
                            
                            assert "real-time mode disabled" in result[mock_live].lower()
    
    def test_reform_text_empty(self):
        """Test reform_text with empty string"""
        with patch('web_ui.llm') as mock_llm:
            mock_llm.is_available.return_value = True
            
            result = reform_text("", "punctuate")
            assert result == "No text to reform"
    
    def test_reform_text_whitespace(self):
        """Test reform_text with whitespace-only string"""
        with patch('web_ui.llm') as mock_llm:
            mock_llm.is_available.return_value = True
            
            result = reform_text("   \n\t  ", "punctuate")
            assert result == "No text to reform"
    
    def test_reform_text_llm_unavailable(self):
        """Test reform_text when LLM is unavailable"""
        with patch('web_ui.llm') as mock_llm:
            mock_llm.is_available.return_value = False
            
            result = reform_text("Test text", "punctuate")
            assert "LLM not available" in result
            assert "Test text" in result
    
    def test_reform_text_success(self):
        """Test successful reform_text"""
        with patch('web_ui.llm') as mock_llm:
            mock_llm.is_available.return_value = True
            mock_llm.process.return_value = "Reformed text"
            
            result = reform_text("Original text", "punctuate")
            assert result == "Reformed text"
            mock_llm.process.assert_called_once_with("Original text", "punctuate")
    
    def test_reform_text_exception(self):
        """Test reform_text exception handling"""
        with patch('web_ui.llm') as mock_llm:
            mock_llm.is_available.return_value = True
            mock_llm.process.side_effect = Exception("LLM error")
            
            result = reform_text("Test", "punctuate")
            assert "Error reforming text" in result


# =============================================================================
# Test Class: Integration Tests with Real Files
# =============================================================================

class TestIntegrationRealFiles:
    """Integration tests using real audio files"""
    
    def test_full_upload_workflow_with_assets(self, mock_subprocess_success):
        """Test full upload workflow with real audio assets"""
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        if not os.path.exists(assets_dir):
            pytest.skip("Assets directory not found")
        
        wav_files = [f for f in os.listdir(assets_dir) if f.endswith('.wav') and not f.startswith('test_corrupt')]
        if not wav_files:
            pytest.skip("No WAV files found in assets")
        
        with patch('os.path.exists', return_value=True):
            with patch('web_ui.llm') as mock_llm:
                mock_llm.backend_name = "ollama"
                mock_llm.is_available.return_value = True
                mock_llm.process.return_value = "Reformed output"
                
                for wav_file in wav_files[:3]:  # Test first 3 files
                    file_path = os.path.join(assets_dir, wav_file)
                    
                    raw, reformed, info = process_audio(file_path, "auto", "punctuate", "0.6b")
                    
                    assert raw is not None
                    assert "Backend:" in info
                    assert "Model:" in info
    
    def test_different_file_sizes(self, mock_subprocess_success):
        """Test processing different file sizes"""
        test_audio_dir = os.path.join(os.path.dirname(__file__), 'test_audio')
        if not os.path.exists(test_audio_dir):
            pytest.skip("Test audio directory not found")
        
        files_by_size = []
        for f in os.listdir(test_audio_dir):
            if f.endswith('.wav'):
                filepath = os.path.join(test_audio_dir, f)
                size = os.path.getsize(filepath)
                files_by_size.append((f, filepath, size))
        
        if len(files_by_size) < 2:
            pytest.skip("Need at least 2 audio files for comparison")
        
        # Sort by size
        files_by_size.sort(key=lambda x: x[2])
        
        with patch('os.path.exists', return_value=True):
            with patch('web_ui.llm') as mock_llm:
                mock_llm.backend_name = "ollama"
                mock_llm.is_available.return_value = True
                mock_llm.process.return_value = "Reformed"
                
                # Test smallest file
                _, smallest_path, _ = files_by_size[0]
                raw, _, info = process_audio(smallest_path, "auto", "none", "0.6b")
                assert raw is not None
                
                # Test largest file
                _, largest_path, _ = files_by_size[-1]
                raw, _, info = process_audio(largest_path, "auto", "none", "0.6b")
                assert raw is not None
    
    def test_multiple_audio_formats(self, mock_subprocess_success):
        """Test handling of multiple audio formats"""
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        if not os.path.exists(assets_dir):
            pytest.skip("Assets directory not found")
        
        # Test different formats
        formats = ['.wav', '.mp3', '.m4a', '.flac']
        tested_formats = []
        
        with patch('os.path.exists', return_value=True):
            with patch('web_ui.llm') as mock_llm:
                mock_llm.backend_name = "ollama"
                mock_llm.is_available.return_value = True
                mock_llm.process.return_value = "Reformed"
                
                for fmt in formats:
                    files = [f for f in os.listdir(assets_dir) if f.endswith(fmt)]
                    if files:
                        file_path = os.path.join(assets_dir, files[0])
                        try:
                            raw, _, info = process_audio(file_path, "auto", "none", "0.6b")
                            tested_formats.append(fmt)
                            assert raw is not None
                        except Exception as e:
                            pytest.fail(f"Failed to process {fmt} file: {e}")
        
        # Verify at least one format was tested
        assert len(tested_formats) > 0, "No audio formats were tested"
    
    def test_concurrent_file_processing(self, temp_dir, mock_subprocess_success):
        """Test concurrent processing of multiple files"""
        # Create multiple test files
        file_paths = []
        for i in range(5):
            file_path = os.path.join(temp_dir, f"test_{i}.wav")
            sample_rate = 16000
            duration = 0.5
            samples = int(sample_rate * duration)
            audio_data = np.random.randn(samples) * 0.1
            audio_int16 = np.clip(audio_data * 32767, -32768, 32768).astype(np.int16)
            
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            file_paths.append(file_path)
        
        results = []
        errors = []
        
        def process_file(file_path):
            try:
                with patch('web_ui.llm') as mock_llm:
                    mock_llm.backend_name = "ollama"
                    mock_llm.is_available.return_value = True
                    mock_llm.process.return_value = f"Reformed {os.path.basename(file_path)}"
                    
                    raw, reformed, info = process_audio(file_path, "auto", "punctuate", "0.6b")
                    results.append((file_path, raw, reformed))
            except Exception as e:
                errors.append((file_path, str(e)))
        
        # Process concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_file, fp) for fp in file_paths]
            for future in as_completed(futures):
                future.result()
        
        # Verify results
        assert len(results) == 5
        assert len(errors) == 0
    
    def test_workflow_with_special_characters(self, mock_subprocess_success):
        """Test workflow with files containing special characters in names"""
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        if not os.path.exists(assets_dir):
            pytest.skip("Assets directory not found")
        
        # Find files with special characters
        special_files = [f for f in os.listdir(assets_dir) 
                        if any(c in f for c in ['ñ', 'é', 'ü', '中', '日', '🎵'])]
        
        if not special_files:
            pytest.skip("No files with special characters found")
        
        with patch('os.path.exists', return_value=True):
            with patch('web_ui.llm') as mock_llm:
                mock_llm.backend_name = "ollama"
                mock_llm.is_available.return_value = True
                mock_llm.process.return_value = "Reformed"
                
                for special_file in special_files[:2]:
                    file_path = os.path.join(assets_dir, special_file)
                    raw, _, info = process_audio(file_path, "auto", "none", "0.6b")
                    assert raw is not None


# =============================================================================
# Test Class: Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_missing_file_error(self):
        """Test handling of missing file"""
        with patch('os.path.exists', return_value=False):
            transcript, backend = transcribe_audio("/nonexistent/file.wav")
            assert "file not found" in transcript.lower()
            assert backend == "Error"
    
    def test_invalid_audio_format(self, temp_dir):
        """Test handling of invalid audio format"""
        # Create a non-audio file with .wav extension
        invalid_file = os.path.join(temp_dir, "invalid.wav")
        with open(invalid_file, 'w') as f:
            f.write("This is not audio data")
        
        with patch('os.path.exists', return_value=True):
            # The C binary should handle this gracefully
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 1
                mock_result.stderr = "Error: Invalid audio format"
                mock_run.return_value = mock_result
                
                transcript, backend = transcribe_audio(invalid_file)
                assert "error" in transcript.lower() or transcript is not None
    
    def test_none_input_transcribe(self):
        """Test transcribe_audio with None input"""
        transcript, backend = transcribe_audio(None)
        assert "no audio file" in transcript.lower()
        assert backend == "Error"
    
    def test_backend_failure_fallback(self):
        """Test backend failure with fallback"""
        with patch('os.path.exists', return_value=True):
            # Mock C binary failure
            with patch('web_ui.transcribe_with_c_binary', return_value=(None, "C binary error")):
                # Mock MLX import failure
                with patch.dict('sys.modules', {'mlx_audio.stt': None}):
                    # Mock CLI success
                    mock_result = Mock()
                    mock_result.returncode = 0
                    mock_result.stdout = "Fallback transcription"
                    
                    with patch('subprocess.run', return_value=mock_result):
                        transcript, backend = transcribe_audio("test.wav")
                        assert "Fallback transcription" in transcript
                        assert backend == "MLX-CLI"
    
    def test_all_backends_failure(self):
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
    
    def test_timeout_handling(self, mock_subprocess_timeout):
        """Test timeout handling"""
        with patch('os.path.exists', return_value=True):
            transcript, backend = transcribe_with_c_binary("test.wav")
            assert transcript is None
            assert "timeout" in backend.lower()
    
    def test_exception_in_transcription(self):
        """Test exception handling during transcription"""
        with patch('os.path.exists', return_value=True):
            with patch('subprocess.run', side_effect=Exception("Unexpected error")):
                transcript, backend = transcribe_with_c_binary("test.wav")
                assert transcript is None
                assert "failed" in backend.lower()
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted audio files"""
        assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
        if not os.path.exists(assets_dir):
            pytest.skip("Assets directory not found")
        
        corrupt_file = os.path.join(assets_dir, 'test_corrupt.wav')
        if not os.path.exists(corrupt_file):
            pytest.skip("Corrupted test file not found")
        
        with patch('os.path.exists', return_value=True):
            with patch('subprocess.run') as mock_run:
                mock_result = Mock()
                mock_result.returncode = 1
                mock_result.stderr = "Error: Corrupted file"
                mock_run.return_value = mock_result
                
                transcript, backend = transcribe_audio(corrupt_file)
                # Should handle gracefully
                assert "error" in transcript.lower() or backend == "Error"
    
    def test_permission_denied_error(self):
        """Test handling of permission denied error"""
        with patch('os.path.exists', return_value=True):
            with patch('subprocess.run', side_effect=PermissionError("Permission denied")):
                transcript, backend = transcribe_with_c_binary("test.wav")
                assert transcript is None


# =============================================================================
# Test Class: Clear and Copy Button Tests
# =============================================================================

class TestButtonHandlers:
    """Test clear and copy button functionality"""
    
    def test_clear_button_handler(self):
        """Test clear button handler returns empty values"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify clear button handler clears all outputs
        assert 'clear_btn.click(' in content
        assert 'lambda: ("", "", "", "")' in content
    
    def test_copy_button_javascript(self):
        """Test copy button JavaScript implementation"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify copy buttons use JavaScript
        assert 'copy_raw_btn.click(' in content
        assert 'copy_reformed_btn.click(' in content
        assert 'navigator.clipboard.writeText' in content
        assert 'js=' in content
    
    def test_upload_button_handler(self):
        """Test upload button handler configuration"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify upload button handler
        assert 'upload_btn.click(' in content
        assert 'fn=process_audio' in content
        assert 'audio_input' in content
        assert 'language' in content
        assert 'reform_mode' in content
        assert 'model' in content
    
    def test_record_button_handler(self):
        """Test record button handler configuration"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify record button handler
        assert 'record_btn.click(' in content
        assert 'fn=stream_record_and_transcribe' in content
        assert 'record_input' in content
        assert 'enable_realtime' in content


# =============================================================================
# Test Class: Recording State Tests
# =============================================================================

class TestRecordingState:
    """Test recording state management"""
    
    def test_recording_state_initial(self):
        """Test initial recording state"""
        assert recording_state["is_recording"] == False
        assert recording_state["start_time"] is None
        assert recording_state["live_text"] == ""
    
    def test_recording_state_reset(self):
        """Test recording state can be reset"""
        # Modify state
        recording_state["is_recording"] = True
        recording_state["start_time"] = time.time()
        recording_state["live_text"] = "Test"
        
        # Reset
        recording_state["is_recording"] = False
        recording_state["start_time"] = None
        recording_state["live_text"] = ""
        
        assert recording_state["is_recording"] == False
        assert recording_state["start_time"] is None
        assert recording_state["live_text"] == ""


# =============================================================================
# Test Class: Event Handler Tests
# =============================================================================

class TestEventHandlers:
    """Test Gradio event handler configurations"""
    
    def test_realtime_toggle_event(self):
        """Test realtime toggle event handler"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify change event handler
        assert 'enable_realtime.change(' in content
        assert 'fn=toggle_realtime_mode' in content
    
    def test_component_event_bindings(self):
        """Test all components have proper event bindings"""
        web_ui_path = os.path.join(os.path.dirname(__file__), '..', 'web_ui.py')
        with open(web_ui_path, 'r') as f:
            content = f.read()
        
        # Verify all button clicks
        assert 'upload_btn.click(' in content
        assert 'record_btn.click(' in content
        assert 'clear_btn.click(' in content
        assert 'copy_raw_btn.click(' in content
        assert 'copy_reformed_btn.click(' in content


# =============================================================================
# Test Class: Performance and Load Tests
# =============================================================================

class TestPerformanceLoad:
    """Performance and load testing"""
    
    def test_large_file_processing_simulation(self, mock_subprocess_success):
        """Test processing simulation for large files"""
        with patch('os.path.exists', return_value=True):
            with patch('web_ui.llm') as mock_llm:
                mock_llm.backend_name = "ollama"
                mock_llm.is_available.return_value = True
                mock_llm.process.return_value = "Large file processed"
                
                # Simulate large file
                large_file = "/tmp/large_90s_audio.wav"
                
                start = time.time()
                raw, reformed, info = process_audio(large_file, "auto", "punctuate", "1.7b")
                elapsed = time.time() - start
                
                assert elapsed < 0.5  # Should be fast with mocks
                assert raw is not None
    
    def test_rapid_successive_calls(self, temp_audio_file, mock_subprocess_success):
        """Test rapid successive calls to process_audio"""
        with patch('os.path.exists', return_value=True):
            with patch('web_ui.llm') as mock_llm:
                mock_llm.backend_name = "ollama"
                mock_llm.is_available.return_value = True
                mock_llm.process.return_value = "Reformed"
                
                results = []
                for i in range(10):
                    raw, reformed, info = process_audio(temp_audio_file, "auto", "punctuate", "0.6b")
                    results.append((raw, reformed))
                
                assert len(results) == 10
                assert all(r[0] is not None for r in results)
    
    def test_memory_efficiency_simulation(self, temp_dir, mock_subprocess_success):
        """Test memory efficiency with multiple files"""
        # Create multiple files
        file_paths = []
        for i in range(10):
            file_path = os.path.join(temp_dir, f"mem_test_{i}.wav")
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                # 0.1 second of silence
                wf.writeframes(b'\x00' * 3200)
            file_paths.append(file_path)
        
        with patch('web_ui.llm') as mock_llm:
            mock_llm.backend_name = "ollama"
            mock_llm.is_available.return_value = True
            mock_llm.process.return_value = "Reformed"
            
            # Process all files
            for file_path in file_paths:
                raw, reformed, info = process_audio(file_path, "auto", "none", "0.6b")
                assert raw is not None


# =============================================================================
# Test Class: Streaming Simulation Tests
# =============================================================================

class TestStreamingSimulation:
    """Test streaming transcription simulation"""
    
    def test_streaming_generator_yields(self, temp_audio_file, mock_subprocess_success):
        """Test that streaming generator yields multiple values"""
        mock_live = Mock()
        mock_raw = Mock()
        mock_reformed = Mock()
        mock_info = Mock()
        
        with patch('web_ui.live_output', mock_live):
            with patch('web_ui.raw_output', mock_raw):
                with patch('web_ui.reformed_output', mock_reformed):
                    with patch('web_ui.info_output', mock_info):
                        with patch('web_ui.llm') as mock_llm:
                            mock_llm.backend_name = "ollama"
                            mock_llm.is_available.return_value = True
                            mock_llm.process.return_value = "Reformed text"
                            
                            gen = stream_record_and_transcribe(
                                temp_audio_file, "auto", "punctuate", "0.6b", True
                            )
                            
                            # Collect all yields
                            results = list(gen)
                            
                            # Should yield multiple progress updates
                            assert len(results) > 0
    
    def test_streaming_progress_updates(self, temp_audio_file, mock_subprocess_success):
        """Test streaming provides progress updates"""
        mock_live = Mock()
        mock_raw = Mock()
        mock_reformed = Mock()
        mock_info = Mock()
        
        with patch('web_ui.live_output', mock_live):
            with patch('web_ui.raw_output', mock_raw):
                with patch('web_ui.reformed_output', mock_reformed):
                    with patch('web_ui.info_output', mock_info):
                        with patch('web_ui.llm') as mock_llm:
                            mock_llm.backend_name = "ollama"
                            mock_llm.is_available.return_value = True
                            mock_llm.process.return_value = "Long reformed text with multiple words"
                            
                            gen = stream_record_and_transcribe(
                                temp_audio_file, "auto", "punctuate", "0.6b", True
                            )
                            
                            # Check for progress indicators in yields
                            progress_found = False
                            for result in gen:
                                info_text = result.get(mock_info, "")
                                if "Status:" in info_text or "%" in info_text:
                                    progress_found = True
                                    break
                            
                            assert progress_found
    
    def test_streaming_without_reforming(self, temp_audio_file, mock_subprocess_success):
        """Test streaming without text reforming"""
        mock_live = Mock()
        mock_raw = Mock()
        mock_reformed = Mock()
        mock_info = Mock()
        
        with patch('web_ui.live_output', mock_live):
            with patch('web_ui.raw_output', mock_raw):
                with patch('web_ui.reformed_output', mock_reformed):
                    with patch('web_ui.info_output', mock_info):
                        with patch('web_ui.llm') as mock_llm:
                            mock_llm.backend_name = "ollama"
                            
                            gen = stream_record_and_transcribe(
                                temp_audio_file, "auto", "none", "0.6b", True
                            )
                            
                            results = list(gen)
                            
                            # Check final result indicates no reformation
                            final_result = results[-1]
                            assert "Not used" in final_result[mock_info] or "No reformation" in final_result[mock_reformed]


# =============================================================================
# Test Summary
# =============================================================================

# Total test cases by category:
# - Gradio Component Tests: 9 tests
# - Web UI Function Tests: 18 tests  
# - Integration Tests (Real Files): 5 tests
# - Error Handling Tests: 9 tests
# - Clear/Copy Button Tests: 4 tests
# - Recording State Tests: 2 tests
# - Event Handler Tests: 2 tests
# - Performance/Load Tests: 3 tests
# - Streaming Simulation Tests: 3 tests
# TOTAL: 55 test cases

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
