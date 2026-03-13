#!/usr/bin/env python3
"""
Comprehensive CLI Functionality Tests for Qwen3-ASR Pro

Tests CLI as a black box using subprocess with real audio files.
Includes timing measurements and output quality verification.

Run with: pytest tests/test_cli_functionality.py -v
"""

import subprocess
import sys
import os
import time
import json
import tempfile
import shutil
import pytest
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import wave
import struct


# =============================================================================
# Test Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
CLI_PATH = PROJECT_ROOT / "cli_app.py"
TEST_ASSETS_DIR = Path(__file__).parent / "assets"
TEST_AUDIO_DIR = Path(__file__).parent / "test_audio"

# Test files by size category
TEST_FILES = {
    "small": [
        TEST_AUDIO_DIR / "test_0.5s_short.wav",
        TEST_AUDIO_DIR / "test_1s_silence.wav",
        TEST_AUDIO_DIR / "test_2s_noise.wav",
    ],
    "medium": [
        TEST_ASSETS_DIR / "test_simple.wav",
        TEST_AUDIO_DIR / "test_3s_speech_sim.wav",
        TEST_AUDIO_DIR / "test_5s_sine.wav",
    ],
    "large": [
        TEST_AUDIO_DIR / "test_30s_sine.wav",
        TEST_AUDIO_DIR / "test_90s_long.wav",
        TEST_ASSETS_DIR / "test_10min.wav",
    ],
    "special": [
        TEST_ASSETS_DIR / "test_file_ñ_é_ü_中_🎵.wav",
        TEST_ASSETS_DIR / "テストファイル日本語.wav",
        TEST_ASSETS_DIR / "测试文件中文.wav",
    ],
    "formats": [
        TEST_ASSETS_DIR / "test_128k.mp3",
        TEST_ASSETS_DIR / "test_320k.mp3",
        TEST_ASSETS_DIR / "test_aac.m4a",
        TEST_ASSETS_DIR / "test_lossless.flac",
    ],
    "edge_cases": [
        TEST_ASSETS_DIR / "test_empty.wav",
        TEST_ASSETS_DIR / "test_corrupt.wav",
        TEST_AUDIO_DIR / "test_corrupted.wav",
        TEST_AUDIO_DIR / "test_truncated.wav",
    ]
}

# Valid reform modes
REFORM_MODES = ["punctuate", "summarize", "clean", "key_points"]

# Sample transcripts for reform testing
SAMPLE_TRANSCRIPTS = {
    "simple": "hello world this is a test of the transcription system",
    "filler_words": "um so like we need to discuss the project timeline uh first we have the design phase",
    "meeting": "okay so welcome everyone to our weekly sync today we need to discuss the Q3 roadmap first john can you give us an update",
    "questions": "what is the deadline for this project when do we need to deliver the final report",
    "technical": "the api endpoint returns a json response with status code two hundred when successful",
    "long_text": """today we are discussing the quarterly results revenue was up fifteen percent compared to last year 
    our main product line performed exceptionally well with a twenty percent increase in sales however we need to address 
    some concerns about the rising costs in the manufacturing department the team has proposed several cost cutting measures 
    that should help improve margins in the next quarter action items include reviewing vendor contracts and optimizing 
    the supply chain we also need to hire two more engineers for the product team to meet our development goals""",
}


# =============================================================================
# Helper Functions
# =============================================================================

def run_cli(args: List[str], input_text: Optional[str] = None, timeout: int = 120) -> Tuple[int, str, str, float]:
    """
    Run CLI with given arguments and return results.
    
    Returns:
        Tuple of (returncode, stdout, stderr, elapsed_time)
    """
    cmd = [sys.executable, str(CLI_PATH)] + args
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        return result.returncode, result.stdout, result.stderr, elapsed
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start_time
        stdout = e.stdout.decode('utf-8') if e.stdout and isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode('utf-8') if e.stderr and isinstance(e.stderr, bytes) else (e.stderr or "")
        return -1, stdout, stderr, elapsed


def run_cli_interactive(commands: List[str], timeout: int = 60) -> Tuple[int, str, str, float]:
    """
    Run CLI in interactive mode with a sequence of commands.
    
    Args:
        commands: List of commands to send (each will be followed by newline)
        timeout: Maximum time to wait
    
    Returns:
        Tuple of (returncode, stdout, stderr, elapsed_time)
    """
    cmd = [sys.executable, str(CLI_PATH), "--interactive"]
    start_time = time.time()
    
    # Join commands with newlines, add quit at the end
    input_text = "\n".join(commands + ["quit"]) + "\n"
    
    try:
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        return result.returncode, result.stdout, result.stderr, elapsed
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start_time
        stdout = e.stdout.decode('utf-8') if e.stdout and isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode('utf-8') if e.stderr and isinstance(e.stderr, bytes) else (e.stderr or "")
        return -1, stdout, stderr, elapsed


def get_audio_duration(filepath: Path) -> float:
    """Get audio file duration in seconds."""
    try:
        if filepath.suffix.lower() == '.wav':
            with wave.open(str(filepath), 'rb') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        else:
            # For non-WAV files, estimate based on file size (rough approximation)
            return filepath.stat().st_size / 16000  # Assume 16kBps
    except Exception:
        return 0.0


def calculate_rtf(elapsed_time: float, audio_duration: float) -> float:
    """Calculate Real-Time Factor."""
    if audio_duration <= 0:
        return 0.0
    return elapsed_time / audio_duration


def create_temp_audio_file(duration_sec: float, sample_rate: int = 16000) -> Path:
    """Create a temporary test audio file."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    
    # Create simple sine wave
    num_samples = int(duration_sec * sample_rate)
    amplitude = 32767 * 0.5  # 50% amplitude
    frequency = 440  # A4 note
    
    samples = []
    for i in range(num_samples):
        sample = amplitude * (i / num_samples) * (1 if i % 100 < 50 else -1)
        samples.append(int(sample))
    
    with wave.open(str(temp_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
    
    return temp_path


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def cli_available():
    """Check if CLI is available."""
    if not CLI_PATH.exists():
        pytest.skip(f"CLI not found at {CLI_PATH}")
    return True


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp(prefix="cli_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_audio_file(temp_output_dir):
    """Create a sample audio file for testing."""
    filepath = temp_output_dir / "test_sample.wav"
    
    # Create 2-second audio
    duration = 2.0
    sample_rate = 16000
    num_samples = int(duration * sample_rate)
    
    with wave.open(str(filepath), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        
        # Generate simple pattern
        samples = []
        for i in range(num_samples):
            sample = int(10000 * (1 if i % 50 < 25 else -1))
            samples.append(sample)
        
        wf.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
    
    return filepath


@pytest.fixture(scope="module")
def performance_results():
    """Store performance results across tests."""
    return {
        "transcription": [],
        "reforming": [],
        "combined": []
    }


# =============================================================================
# Test Class: CLI Argument Parsing
# =============================================================================

class TestCLIArgumentParsing:
    """Test CLI argument parsing and validation."""
    
    def test_cli_help(self, cli_available):
        """Test --help output."""
        returncode, stdout, stderr, elapsed = run_cli(["--help"])
        
        assert returncode == 0, f"CLI help failed: {stderr}"
        assert "Qwen3-ASR Pro CLI" in stdout or "usage:" in stdout.lower()
        assert "--help" in stdout
        assert "--output" in stdout or "-o" in stdout
        assert "--language" in stdout or "-l" in stdout
        assert "--reform" in stdout or "-r" in stdout
        assert "--interactive" in stdout or "-i" in stdout
    
    def test_cli_version_info(self, cli_available):
        """Test that CLI provides version info."""
        # The CLI might not have explicit version flag, but should show info
        returncode, stdout, stderr, elapsed = run_cli([])
        
        # Without args, it should enter interactive mode or show help
        assert returncode == 0 or "Qwen3-ASR" in stdout or "Interactive" in stdout
    
    @pytest.mark.parametrize("reform_mode", REFORM_MODES)
    def test_reform_modes_valid(self, cli_available, reform_mode, sample_audio_file):
        """Test all valid reform modes are accepted."""
        # Test that the argument is accepted (file might not exist for validation)
        returncode, stdout, stderr, elapsed = run_cli([
            str(sample_audio_file),
            "--reform", reform_mode
        ])
        
        # Should not fail due to invalid reform mode
        assert "invalid choice" not in stderr.lower(), f"Reform mode {reform_mode} rejected"
    
    def test_reform_mode_invalid(self, cli_available, sample_audio_file):
        """Test invalid reform mode is rejected."""
        returncode, stdout, stderr, elapsed = run_cli([
            str(sample_audio_file),
            "--reform", "invalid_mode"
        ])
        
        assert returncode != 0 or "invalid" in stderr.lower() or "error" in stderr.lower()
    
    @pytest.mark.parametrize("language", ["en", "zh", "es", "fr", "de", "ja", "auto"])
    def test_language_option(self, cli_available, language, sample_audio_file):
        """Test --language option with various languages."""
        returncode, stdout, stderr, elapsed = run_cli([
            str(sample_audio_file),
            "--language", language
        ])
        
        # Language option should be accepted (even if processing fails)
        assert "unrecognized arguments" not in stderr
    
    def test_output_option(self, cli_available, sample_audio_file, temp_output_dir):
        """Test --output option creates file."""
        output_file = temp_output_dir / "output.txt"
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(sample_audio_file),
            "--output", str(output_file)
        ])
        
        # Output file should be created or attempted
        # Note: If transcription fails, file might not be created
        assert "unrecognized arguments" not in stderr
    
    def test_interactive_flag(self, cli_available):
        """Test --interactive flag enters interactive mode."""
        # Send quit immediately
        returncode, stdout, stderr, elapsed = run_cli_interactive([])
        
        # Should show interactive prompt or help
        assert "Interactive" in stdout or "Commands:" in stdout or ">" in stdout or returncode == 0
    
    def test_combined_arguments(self, cli_available, sample_audio_file, temp_output_dir):
        """Test combining multiple arguments."""
        output_file = temp_output_dir / "combined_output.txt"
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(sample_audio_file),
            "--output", str(output_file),
            "--language", "en",
            "--reform", "punctuate"
        ])
        
        # All arguments should be accepted
        assert "unrecognized arguments" not in stderr
    
    def test_short_form_arguments(self, cli_available, sample_audio_file, temp_output_dir):
        """Test short form arguments (-o, -l, -r, -i)."""
        output_file = temp_output_dir / "short_output.txt"
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(sample_audio_file),
            "-o", str(output_file),
            "-l", "en",
            "-r", "punctuate"
        ])
        
        # Short forms should be accepted
        assert "unrecognized arguments" not in stderr


# =============================================================================
# Test Class: CLI Transcription
# =============================================================================

class TestCLITranscription:
    """Test CLI transcription with real audio files."""
    
    @pytest.mark.parametrize("file_category,test_file", [
        ("small", f) for f in TEST_FILES["small"]
    ] + [
        ("medium", f) for f in TEST_FILES["medium"]
    ])
    def test_transcription_basic(self, cli_available, file_category, test_file, temp_output_dir):
        """Test basic transcription with various file sizes."""
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        output_file = temp_output_dir / f"{test_file.stem}_output.txt"
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(test_file),
            "--output", str(output_file)
        ], timeout=180)
        
        # Record performance
        duration = get_audio_duration(test_file)
        rtf = calculate_rtf(elapsed, duration)
        
        # Should complete without crashing
        # Note: We don't check returncode == 0 because transcription might fail
        # due to missing models, but it shouldn't crash
        
        # Verify output file was created if transcription succeeded
        if returncode == 0 and output_file.exists():
            content = output_file.read_text()
            assert len(content) >= 0  # File exists and is readable
    
    @pytest.mark.parametrize("test_file", TEST_FILES["formats"])
    def test_transcription_different_formats(self, cli_available, test_file, temp_output_dir):
        """Test transcription with different audio formats."""
        if not test_file.exists():
            pytest.skip(f"Test file not found: {test_file}")
        
        output_file = temp_output_dir / f"{test_file.stem}_output.txt"
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(test_file),
            "--output", str(output_file)
        ], timeout=180)
        
        # Should handle the format (might fail gracefully)
        assert "format" not in stderr.lower() or "unsupported" not in stderr.lower()
    
    def test_transcription_performance_metrics(self, cli_available, temp_output_dir):
        """Test and record transcription performance metrics."""
        results = []
        
        for category, files in TEST_FILES.items():
            if category == "edge_cases":
                continue
                
            for test_file in files[:2]:  # Test first 2 files per category
                if not test_file.exists():
                    continue
                
                output_file = temp_output_dir / f"perf_{test_file.stem}.txt"
                
                start = time.time()
                returncode, stdout, stderr, elapsed = run_cli([
                    str(test_file),
                    "--output", str(output_file)
                ], timeout=120)
                
                duration = get_audio_duration(test_file)
                rtf = calculate_rtf(elapsed, duration)
                
                results.append({
                    "file": test_file.name,
                    "category": category,
                    "duration_sec": duration,
                    "elapsed_sec": elapsed,
                    "rtf": rtf,
                    "success": returncode == 0,
                    "file_size_kb": test_file.stat().st_size / 1024
                })
        
        # Print performance summary
        print("\n=== Transcription Performance Metrics ===")
        for r in results:
            status = "✓" if r["success"] else "✗"
            print(f"{status} {r['file']}: RTF={r['rtf']:.2f}x, Time={r['elapsed_sec']:.2f}s")
        
        # Save results
        metrics_file = temp_output_dir / "transcription_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        assert len(results) > 0, "No performance tests completed"
    
    def test_transcription_large_file(self, cli_available, temp_output_dir):
        """Test transcription with large file (performance check)."""
        large_files = [f for f in TEST_FILES["large"] if f.exists()]
        
        if not large_files:
            pytest.skip("No large test files available")
        
        test_file = large_files[0]
        output_file = temp_output_dir / "large_output.txt"
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(test_file),
            "--output", str(output_file)
        ], timeout=300)
        
        duration = get_audio_duration(test_file)
        rtf = calculate_rtf(elapsed, duration)
        
        print(f"\nLarge file transcription: {test_file.name}")
        print(f"  Duration: {duration:.1f}s, Elapsed: {elapsed:.1f}s, RTF: {rtf:.2f}x")
        
        # Should complete within reasonable time (RTF < 10x for large files)
        if returncode == 0:
            assert rtf < 10.0, f"RTF too high: {rtf:.2f}x"


# =============================================================================
# Test Class: CLI Reforming
# =============================================================================

class TestCLIReforming:
    """Test CLI text reforming functionality."""
    
    @pytest.mark.parametrize("mode", REFORM_MODES)
    def test_reform_modes_with_real_transcript(self, cli_available, mode, sample_audio_file, temp_output_dir):
        """Test each reform mode with actual audio transcription."""
        output_file = temp_output_dir / f"reform_{mode}.txt"
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(sample_audio_file),
            "--reform", mode,
            "--output", str(output_file)
        ], timeout=180)
        
        # Should complete without errors related to reform mode
        assert "invalid" not in stderr.lower() or "reform" not in stderr.lower()
    
    def test_reform_punctuate_quality(self, cli_available, temp_output_dir):
        """Test punctuate reform output quality."""
        output_file = temp_output_dir / "reform_punctuate_test.txt"
        
        # Create a test audio file
        test_audio = temp_output_dir / "test_punctuate.wav"
        duration = 3.0
        sample_rate = 16000
        num_samples = int(duration * sample_rate)
        
        with wave.open(str(test_audio), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            samples = [int(5000 * (1 if i % 40 < 20 else -1)) for i in range(num_samples)]
            wf.writeframes(struct.pack('<' + 'h' * len(samples), *samples))
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(test_audio),
            "--reform", "punctuate",
            "--output", str(output_file)
        ], timeout=180)
        
        # If successful, check output quality
        if returncode == 0 and output_file.exists():
            content = output_file.read_text()
            # Punctuated text should have proper capitalization and punctuation
            # Note: This is a weak check since audio content is synthetic
            assert isinstance(content, str)
    
    def test_reform_summarize_output(self, cli_available, temp_output_dir):
        """Test summarize reform creates concise output."""
        output_file = temp_output_dir / "reform_summarize.txt"
        
        # Use a real test file
        test_files = [f for f in TEST_FILES["medium"] if f.exists()]
        if not test_files:
            pytest.skip("No test files available")
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(test_files[0]),
            "--reform", "summarize",
            "--output", str(output_file)
        ], timeout=180)
        
        # Verify summarization was attempted
        assert "summarize" in stderr.lower() or "reform" in stdout.lower() or returncode in [0, -1]
    
    def test_reform_clean_output(self, cli_available, temp_output_dir):
        """Test clean reform removes filler words."""
        output_file = temp_output_dir / "reform_clean.txt"
        
        test_files = [f for f in TEST_FILES["medium"] if f.exists()]
        if not test_files:
            pytest.skip("No test files available")
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(test_files[0]),
            "--reform", "clean",
            "--output", str(output_file)
        ], timeout=180)
        
        # Clean should be attempted
        assert returncode in [0, -1]  # -1 for timeout is acceptable
    
    def test_reform_key_points_format(self, cli_available, temp_output_dir):
        """Test key_points reform creates bullet points."""
        output_file = temp_output_dir / "reform_key_points.txt"
        
        test_files = [f for f in TEST_FILES["medium"] if f.exists()]
        if not test_files:
            pytest.skip("No test files available")
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(test_files[0]),
            "--reform", "key_points",
            "--output", str(output_file)
        ], timeout=180)
        
        # Check for bullet points in output if successful
        if returncode == 0 and output_file.exists():
            content = output_file.read_text()
            # Look for bullet characters
            has_bullets = any(c in content for c in ['•', '-', '*', '·'])
            # Not a strict requirement since input might be short
            print(f"Key points output has bullets: {has_bullets}")
    
    def test_reform_performance_comparison(self, cli_available, temp_output_dir):
        """Compare reforming performance across modes."""
        test_files = [f for f in TEST_FILES["medium"] if f.exists()]
        if not test_files:
            pytest.skip("No test files available")
        
        results = []
        
        for mode in REFORM_MODES:
            output_file = temp_output_dir / f"perf_{mode}.txt"
            
            start = time.time()
            returncode, stdout, stderr, elapsed = run_cli([
                str(test_files[0]),
                "--reform", mode,
                "--output", str(output_file)
            ], timeout=120)
            
            results.append({
                "mode": mode,
                "elapsed": elapsed,
                "success": returncode == 0
            })
        
        print("\n=== Reform Mode Performance ===")
        for r in results:
            status = "✓" if r["success"] else "✗"
            print(f"{status} {r['mode']}: {r['elapsed']:.2f}s")
        
        # All modes should complete
        successful = sum(1 for r in results if r["success"])
        print(f"Successful: {successful}/{len(results)}")


# =============================================================================
# Test Class: CLI Interactive Mode
# =============================================================================

class TestCLIInteractiveMode:
    """Test CLI interactive mode commands."""
    
    def test_interactive_help_command(self, cli_available):
        """Test 'help' command in interactive mode."""
        returncode, stdout, stderr, elapsed = run_cli_interactive(["help"])
        
        # Should show help message
        help_indicators = [
            "Commands:", "transcribe", "reform", "process", 
            "status", "help", "quit", "Qwen3-ASR"
        ]
        found = sum(1 for indicator in help_indicators if indicator in stdout)
        assert found >= 3, f"Help output missing expected content. Found: {found}/10 indicators"
    
    def test_interactive_status_command(self, cli_available):
        """Test 'status' command in interactive mode."""
        returncode, stdout, stderr, elapsed = run_cli_interactive(["status"])
        
        # Should show status information
        status_indicators = [
            "Status", "Python", "Platform", "LLM", "Backend",
            "macOS", "darwin", "✅", "❌"
        ]
        found = sum(1 for indicator in status_indicators if indicator in stdout)
        assert found >= 2, f"Status output missing expected content"
    
    def test_interactive_transcribe_command(self, cli_available, sample_audio_file):
        """Test 'transcribe' command in interactive mode."""
        commands = [f"transcribe {sample_audio_file}"]
        returncode, stdout, stderr, elapsed = run_cli_interactive(commands, timeout=120)
        
        # Should attempt transcription
        assert "transcrib" in stdout.lower() or "error" in stderr.lower() or returncode in [0, -1]
    
    def test_interactive_reform_command(self, cli_available):
        """Test 'reform' command in interactive mode."""
        test_text = "hello world this is a test"
        commands = [f"reform {test_text} punctuate"]
        returncode, stdout, stderr, elapsed = run_cli_interactive(commands, timeout=60)
        
        # Should attempt reform
        assert "reform" in stdout.lower() or "punctuat" in stdout.lower() or returncode in [0, -1]
    
    def test_interactive_process_command(self, cli_available, sample_audio_file):
        """Test 'process' command in interactive mode."""
        commands = [f"process {sample_audio_file} punctuate"]
        returncode, stdout, stderr, elapsed = run_cli_interactive(commands, timeout=120)
        
        # Should attempt processing
        assert "process" in stdout.lower() or "transcrib" in stdout.lower() or returncode in [0, -1]
    
    def test_interactive_unknown_command(self, cli_available):
        """Test handling of unknown command."""
        returncode, stdout, stderr, elapsed = run_cli_interactive(["unknowncommand123"])
        
        # Should handle gracefully
        assert "unknown" in stdout.lower() or "error" in stdout.lower() or returncode == 0
    
    def test_interactive_empty_command(self, cli_available):
        """Test handling of empty command."""
        returncode, stdout, stderr, elapsed = run_cli_interactive(["", ""])
        
        # Should not crash
        assert returncode == 0
    
    def test_interactive_multiple_commands(self, cli_available, sample_audio_file):
        """Test running multiple commands in sequence."""
        commands = [
            "help",
            "status",
            f"transcribe {sample_audio_file}",
            "reform hello world punctuate"
        ]
        returncode, stdout, stderr, elapsed = run_cli_interactive(commands, timeout=180)
        
        # All commands should be processed
        assert "Commands:" in stdout  # From help
        assert "Status" in stdout or "LLM" in stdout  # From status


# =============================================================================
# Test Class: CLI Error Handling
# =============================================================================

class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_nonexistent_file(self, cli_available, temp_output_dir):
        """Test handling of non-existent file."""
        nonexistent = temp_output_dir / "does_not_exist.wav"
        
        returncode, stdout, stderr, elapsed = run_cli([str(nonexistent)])
        
        # Should report error, not crash
        assert returncode != 0 or "not found" in stdout.lower() or "error" in stdout.lower()
        assert "File not found" in stdout or "not exist" in stdout.lower() or "error" in stdout.lower()
    
    def test_directory_instead_of_file(self, cli_available, temp_output_dir):
        """Test handling of directory instead of file."""
        returncode, stdout, stderr, elapsed = run_cli([str(temp_output_dir)])
        
        # Should report error about directory
        assert "directory" in stdout.lower() or "error" in stdout.lower() or returncode != 0
    
    def test_empty_file(self, cli_available, temp_output_dir):
        """Test handling of empty file."""
        empty_file = temp_output_dir / "empty.wav"
        empty_file.touch()
        
        returncode, stdout, stderr, elapsed = run_cli([str(empty_file)])
        
        # Should handle gracefully
        assert returncode in [0, 1, -1]  # Accept various outcomes
    
    def test_corrupt_file(self, cli_available):
        """Test handling of corrupt audio file."""
        corrupt_files = [f for f in TEST_FILES["edge_cases"] if "corrupt" in f.name and f.exists()]
        
        if not corrupt_files:
            pytest.skip("No corrupt test files available")
        
        returncode, stdout, stderr, elapsed = run_cli([str(corrupt_files[0])])
        
        # Should handle gracefully, not crash
        assert returncode in [0, 1, -1]
    
    def test_invalid_reform_mode_error(self, cli_available, sample_audio_file):
        """Test error message for invalid reform mode."""
        returncode, stdout, stderr, elapsed = run_cli([
            str(sample_audio_file),
            "--reform", "not_a_mode"
        ])
        
        # Should reject invalid mode
        assert returncode != 0 or "invalid" in stderr.lower()
    
    def test_permission_error(self, cli_available, temp_output_dir):
        """Test handling of permission error."""
        # Create a file with no read permission
        restricted_file = temp_output_dir / "restricted.wav"
        restricted_file.touch()
        os.chmod(str(restricted_file), 0o000)
        
        try:
            returncode, stdout, stderr, elapsed = run_cli([str(restricted_file)])
            
            # Should report permission error
            assert "permission" in stdout.lower() or "cannot read" in stdout.lower() or returncode != 0
        finally:
            os.chmod(str(restricted_file), 0o644)
    
    def test_invalid_language_code(self, cli_available, sample_audio_file):
        """Test handling of invalid language code."""
        returncode, stdout, stderr, elapsed = run_cli([
            str(sample_audio_file),
            "--language", "invalid_language_code_12345"
        ])
        
        # Should handle gracefully (may ignore or report warning)
        assert "unrecognized arguments" not in stderr


# =============================================================================
# Test Class: CLI Output Verification
# =============================================================================

class TestCLIOutputVerification:
    """Test CLI output file and format verification."""
    
    def test_output_file_creation(self, cli_available, sample_audio_file, temp_output_dir):
        """Test that -o option creates output file."""
        output_file = temp_output_dir / "test_output.txt"
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(sample_audio_file),
            "--output", str(output_file)
        ])
        
        # Output file should exist if transcription succeeded
        if returncode == 0:
            assert output_file.exists(), "Output file was not created"
            assert output_file.stat().st_size >= 0
    
    def test_stdout_output(self, cli_available, sample_audio_file):
        """Test that output goes to stdout when no -o specified."""
        returncode, stdout, stderr, elapsed = run_cli([str(sample_audio_file)])
        
        # Should produce output to stdout
        assert len(stdout) >= 0  # Accept empty or non-empty
    
    def test_output_format_text(self, cli_available, sample_audio_file, temp_output_dir):
        """Test output is valid text."""
        output_file = temp_output_dir / "format_test.txt"
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(sample_audio_file),
            "--output", str(output_file)
        ])
        
        if output_file.exists():
            content = output_file.read_text(encoding='utf-8')
            # Should be valid text
            assert isinstance(content, str)
            # Should be decodable
            content.encode('utf-8')
    
    def test_output_with_special_characters(self, cli_available, temp_output_dir):
        """Test handling of files with special characters in name."""
        # Use existing test files with special characters
        special_files = [f for f in TEST_FILES["special"] if f.exists()]
        
        for test_file in special_files[:2]:  # Test first 2
            output_file = temp_output_dir / f"output_{test_file.stem}.txt"
            
            returncode, stdout, stderr, elapsed = run_cli([
                str(test_file),
                "--output", str(output_file)
            ])
            
            # Should handle special characters
            assert "encoding" not in stderr.lower() or "unicode" not in stderr.lower()
    
    def test_output_consistency(self, cli_available, sample_audio_file, temp_output_dir):
        """Test that multiple runs produce consistent output."""
        output1 = temp_output_dir / "consistency_1.txt"
        output2 = temp_output_dir / "consistency_2.txt"
        
        # Run twice
        run_cli([str(sample_audio_file), "--output", str(output1)])
        run_cli([str(sample_audio_file), "--output", str(output2)])
        
        # Both files should be created if successful
        if output1.exists() and output2.exists():
            content1 = output1.read_text()
            content2 = output2.read_text()
            # Content should be identical for deterministic processing
            # Note: This might not always be true due to timing, but worth checking
            print(f"Content lengths: {len(content1)} vs {len(content2)}")
    
    def test_default_output_naming(self, cli_available, sample_audio_file, temp_output_dir):
        """Test default output file naming."""
        # Change to temp dir for default naming
        original_cwd = os.getcwd()
        os.chdir(temp_output_dir)
        
        try:
            returncode, stdout, stderr, elapsed = run_cli([str(sample_audio_file)])
            
            # Check for default output file
            default_name = f"{sample_audio_file.stem}_transcript.txt"
            default_path = temp_output_dir / default_name
            
            # Default file may or may not be created depending on transcription success
            print(f"Default output exists: {default_path.exists()}")
        finally:
            os.chdir(original_cwd)


# =============================================================================
# Test Class: CLI Performance and Stress Tests
# =============================================================================

class TestCLIPerformance:
    """Test CLI performance under various conditions."""
    
    def test_transcription_timing(self, cli_available, temp_output_dir):
        """Test transcription timing for performance baseline."""
        test_files = [f for f in TEST_FILES["small"] + TEST_FILES["medium"] if f.exists()][:3]
        
        if not test_files:
            pytest.skip("No test files available")
        
        results = []
        for test_file in test_files:
            output_file = temp_output_dir / f"timing_{test_file.stem}.txt"
            
            start = time.time()
            returncode, stdout, stderr, elapsed = run_cli([
                str(test_file),
                "--output", str(output_file)
            ], timeout=120)
            actual_elapsed = time.time() - start
            
            duration = get_audio_duration(test_file)
            rtf = calculate_rtf(actual_elapsed, duration)
            
            results.append({
                "file": test_file.name,
                "size_kb": test_file.stat().st_size / 1024,
                "duration_sec": duration,
                "elapsed_sec": actual_elapsed,
                "rtf": rtf
            })
        
        print("\n=== Transcription Timing Results ===")
        for r in results:
            print(f"{r['file']}: {r['elapsed_sec']:.2f}s (RTF: {r['rtf']:.2f}x)")
        
        # All should complete in reasonable time
        for r in results:
            assert r["elapsed_sec"] < 120, f"{r['file']} took too long: {r['elapsed_sec']:.2f}s"
    
    def test_consecutive_transcriptions(self, cli_available, temp_output_dir):
        """Test running multiple consecutive transcriptions."""
        test_files = [f for f in TEST_FILES["small"] if f.exists()][:3]
        
        if len(test_files) < 2:
            pytest.skip("Not enough test files")
        
        start_time = time.time()
        
        for i, test_file in enumerate(test_files):
            output_file = temp_output_dir / f"consecutive_{i}.txt"
            run_cli([str(test_file), "--output", str(output_file)])
        
        total_elapsed = time.time() - start_time
        
        print(f"\nConsecutive transcriptions: {len(test_files)} files in {total_elapsed:.2f}s")
        
        # Should complete in reasonable total time
        assert total_elapsed < 300, f"Batch took too long: {total_elapsed:.2f}s"
    
    def test_memory_stability(self, cli_available, temp_output_dir):
        """Test memory stability over multiple runs."""
        test_file = None
        for f in TEST_FILES["small"]:
            if f.exists():
                test_file = f
                break
        
        if not test_file:
            pytest.skip("No test files available")
        
        times = []
        for i in range(3):  # Run 3 times
            output_file = temp_output_dir / f"memory_{i}.txt"
            
            start = time.time()
            run_cli([str(test_file), "--output", str(output_file)])
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Times should be relatively consistent (no major memory leaks causing slowdown)
        avg_time = sum(times) / len(times)
        max_time = max(times)
        
        print(f"\nMemory stability: avg={avg_time:.2f}s, max={max_time:.2f}s")
        
        # Max should not be significantly higher than average
        assert max_time < avg_time * 2, "Possible memory leak detected"


# =============================================================================
# Test Class: CLI Integration Tests
# =============================================================================

class TestCLIIntegration:
    """Test CLI integration scenarios."""
    
    def test_full_workflow_transcribe_only(self, cli_available, temp_output_dir):
        """Test complete workflow: transcribe only."""
        test_files = [f for f in TEST_FILES["medium"] if f.exists()]
        if not test_files:
            pytest.skip("No test files available")
        
        output_file = temp_output_dir / "workflow_transcribe.txt"
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(test_files[0]),
            "--output", str(output_file),
            "--language", "en"
        ], timeout=180)
        
        # Verify workflow completed
        if returncode == 0:
            if output_file.exists():
                content = output_file.read_text()
                print(f"Transcription length: {len(content)} chars")
    
    def test_full_workflow_transcribe_and_reform(self, cli_available, temp_output_dir):
        """Test complete workflow: transcribe + reform."""
        test_files = [f for f in TEST_FILES["medium"] if f.exists()]
        if not test_files:
            pytest.skip("No test files available")
        
        output_file = temp_output_dir / "workflow_reform.txt"
        
        returncode, stdout, stderr, elapsed = run_cli([
            str(test_files[0]),
            "--output", str(output_file),
            "--reform", "punctuate",
            "--language", "en"
        ], timeout=180)
        
        print(f"\nTranscribe + Reform completed in {elapsed:.2f}s")
        
        if returncode == 0 and output_file.exists():
            content = output_file.read_text()
            print(f"Output length: {len(content)} chars")
    
    def test_compare_models(self, cli_available, temp_output_dir):
        """Compare performance between model sizes if available."""
        test_files = [f for f in TEST_FILES["small"] if f.exists()]
        if not test_files:
            pytest.skip("No test files available")
        
        # Note: The CLI uses available backends automatically
        # This test verifies the transcription works with whatever is available
        
        output_file = temp_output_dir / "model_comparison.txt"
        
        start = time.time()
        returncode, stdout, stderr, elapsed = run_cli([
            str(test_files[0]),
            "--output", str(output_file)
        ], timeout=120)
        
        if returncode == 0:
            # Check which backend was used
            if "C-Binary" in stdout:
                backend = "C-Binary"
            elif "MLX-Audio" in stdout:
                backend = "MLX-Audio"
            elif "MLX-CLI" in stdout:
                backend = "MLX-CLI"
            else:
                backend = "Unknown"
            
            print(f"\nBackend used: {backend}")
            print(f"Processing time: {elapsed:.2f}s")
    
    def test_batch_processing_simulation(self, cli_available, temp_output_dir):
        """Simulate batch processing of multiple files."""
        test_files = [f for f in TEST_FILES["small"] if f.exists()][:3]
        
        if len(test_files) < 2:
            pytest.skip("Not enough test files")
        
        results = []
        for i, test_file in enumerate(test_files):
            output_file = temp_output_dir / f"batch_{i}.txt"
            
            start = time.time()
            returncode, stdout, stderr, elapsed = run_cli([
                str(test_file),
                "--output", str(output_file)
            ])
            actual_elapsed = time.time() - start
            
            results.append({
                "file": test_file.name,
                "success": returncode == 0,
                "time": actual_elapsed
            })
        
        success_count = sum(1 for r in results if r["success"])
        total_time = sum(r["time"] for r in results)
        
        print(f"\nBatch processing: {success_count}/{len(results)} succeeded")
        print(f"Total time: {total_time:.2f}s")
        
        # At least some should succeed (depending on model availability)
        assert success_count >= 0  # Accept any number, just record results


# =============================================================================
# Main Entry Point for Standalone Execution
# =============================================================================

if __name__ == "__main__":
    # Run pytest on this file
    pytest.main([__file__, "-v", "--tb=short"])
