#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         Qwen3-ASR Pro - macOS Speech-to-Text                     ║
║         MLX Optimized • Real-time • Reliable                     ║
╚══════════════════════════════════════════════════════════════════╝

Based on official Qwen3-ASR and mlx-qwen3-asr documentation.
Uses Python API for better error handling and control.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import os
import sys
import tempfile
import time
import queue
import traceback
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
import wave

# Constants
APP_NAME = "Qwen3-ASR Pro"
VERSION = "2.3.0"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.05

# Colors
COLORS = {
    'bg': "#0f0f23",
    'surface': "#1a1a2e", 
    'card': "#16162a",
    'card_border': "#2a2a45",
    'primary': "#6366f1",
    'secondary': "#22d3ee",
    'success': "#10b981",
    'warning': "#f59e0b",
    'error': "#ef4444",
    'text': "#f1f5f9",
    'text_secondary': "#94a3b8",
    'text_muted': "#64748b",
}


@dataclass
class PerformanceStats:
    audio_duration: float = 0.0
    processing_time: float = 0.0
    rtf: float = 0.0


class AudioRecorder:
    """Audio recorder with configurable VAD"""
    
    def __init__(self, silence_threshold: float = 0.015,
                 silence_duration: float = 2.0,
                 level_callback: Optional[callable] = None):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.level_callback = level_callback
        self.is_recording = False
        self.frames = []
        self.stream = None
        self.silence_counter = 0
        self.current_level = 0.0
        self.is_speaking = False
        
    def set_params(self, threshold: float = None, duration: float = None):
        if threshold is not None:
            self.silence_threshold = threshold
        if duration is not None:
            self.silence_duration = duration
    
    def _calculate_level(self, audio_chunk: np.ndarray) -> float:
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return min(1.0, rms * 10)
    
    def _vad(self, audio_chunk: np.ndarray) -> bool:
        level = self._calculate_level(audio_chunk)
        self.current_level = level
        self.is_speaking = level > self.silence_threshold
        if self.level_callback:
            self.level_callback(level, self.is_speaking)
        return self.is_speaking
    
    def start(self):
        import sounddevice as sd
        self.is_recording = True
        self.frames = []
        self.silence_counter = 0
        max_silence_chunks = int(self.silence_duration / CHUNK_DURATION)
        
        def audio_callback(indata, frame_count, time_info, status):
            if not self.is_recording:
                return
            audio = indata.copy().flatten()
            self.frames.append(audio)
            is_speech = self._vad(audio)
            if is_speech:
                self.silence_counter = 0
            else:
                self.silence_counter += 1
                if self.silence_counter >= max_silence_chunks:
                    self.is_recording = False
        
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype=np.float32,
            blocksize=int(SAMPLE_RATE * CHUNK_DURATION),
            callback=audio_callback
        )
        self.stream.start()
    
    def stop(self) -> Optional[str]:
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if not self.frames:
            return None
        
        audio = np.concatenate(self.frames)
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        temp_file = tempfile.mktemp(suffix='.wav')
        
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        
        return temp_file


class TranscriptionEngine:
    """Transcription engine using mlx-qwen3-asr Python API"""
    
    def __init__(self):
        self.backend = None
        self.model = None
        self.model_name = None
        self.detect_backend()
    
    def detect_backend(self):
        """Detect available backend"""
        try:
            # Try to import mlx_audio
            import mlx_audio.stt as mlx_stt
            self.backend = 'mlx_audio'
            print("✅ Using mlx-audio backend")
            return
        except ImportError:
            pass
        
        try:
            # Try CLI availability
            import subprocess
            result = subprocess.run(
                [sys.executable, '-m', 'mlx_qwen3_asr', '--version'],
                capture_output=True, timeout=5
            )
            if result.returncode == 0:
                self.backend = 'mlx_cli'
                print("✅ Using mlx-qwen3-asr CLI backend")
                return
        except:
            pass
        
        try:
            # Fallback to PyTorch
            import qwen_asr
            self.backend = 'pytorch'
            print("✅ Using PyTorch (qwen-asr) backend")
            return
        except ImportError:
            pass
        
        self.backend = None
        raise RuntimeError("No transcription backend available. Please run SETUP.command")
    
    def load_model(self, model_name: str = "Qwen/Qwen3-ASR-0.6B", dtype: str = "float16"):
        """Load model using appropriate backend"""
        if self.backend == 'mlx_audio':
            import mlx_audio.stt as mlx_stt
            self.model = mlx_stt.load(model_name)
            self.model_name = model_name
        elif self.backend == 'pytorch':
            import torch
            from qwen_asr import Qwen3ASRModel
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = Qwen3ASRModel.from_pretrained(
                model_name,
                dtype=torch.float16 if dtype == "float16" else torch.float32,
                device_map=device
            )
            self.model_name = model_name
        # For CLI backend, we don't preload
    
    def transcribe(self, audio_path: str, model: str = "Qwen/Qwen3-ASR-0.6B",
                   dtype: str = "float16", language: Optional[str] = None,
                   progress_callback: Optional[callable] = None) -> tuple:
        """Transcribe audio file"""
        stats = PerformanceStats()
        start_time = time.time()
        
        try:
            import librosa
            stats.audio_duration = librosa.get_duration(path=audio_path)
        except:
            stats.audio_duration = 0
        
        if progress_callback:
            progress_callback("Transcribing...")
        
        if self.backend == 'mlx_audio':
            result = self._transcribe_mlx_audio(audio_path, model, language)
        elif self.backend == 'mlx_cli':
            result = self._transcribe_mlx_cli(audio_path, model, dtype, language)
        elif self.backend == 'pytorch':
            result = self._transcribe_pytorch(audio_path, model, dtype, language)
        else:
            raise RuntimeError("No backend available")
        
        stats.processing_time = time.time() - start_time
        if stats.audio_duration > 0:
            stats.rtf = stats.processing_time / stats.audio_duration
        
        return result, stats
    
    def _transcribe_mlx_audio(self, audio_path: str, model: str, language: Optional[str]):
        """Use mlx-audio Python API"""
        # Load model if not loaded or different model
        if self.model is None or self.model_name != model:
            import mlx_audio.stt as mlx_stt
            self.model = mlx_stt.load(model)
            self.model_name = model
        
        # Generate - only pass language if specified, otherwise let model auto-detect
        if language:
            result = self.model.generate(audio_path, language=language)
        else:
            result = self.model.generate(audio_path)
        
        return {
            'text': result.text if hasattr(result, 'text') else str(result),
            'backend': 'MLX-Audio',
            'model': model
        }
    
    def _transcribe_mlx_cli(self, audio_path: str, model: str, dtype: str, language: Optional[str]):
        """Use mlx-qwen3-asr CLI"""
        import subprocess
        
        cmd = [
            sys.executable, '-m', 'mlx_qwen3_asr',
            audio_path,
            '--model', model,
            '--dtype', dtype,
            '--stdout-only'
        ]
        
        if language and language != "Auto":
            cmd.extend(['--language', language])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Transcription failed"
            raise RuntimeError(error_msg)
        
        return {
            'text': result.stdout.strip(),
            'backend': 'MLX-CLI',
            'model': model
        }
    
    def _transcribe_pytorch(self, audio_path: str, model: str, dtype: str, language: Optional[str]):
        """Use PyTorch qwen-asr"""
        import torch
        from qwen_asr import Qwen3ASRModel
        
        # Load model if needed
        if self.model is None or self.model_name != model:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = Qwen3ASRModel.from_pretrained(
                model,
                dtype=torch.float16 if dtype == "float16" else torch.float32,
                device_map=device
            )
            self.model_name = model
        
        lang = None if language == "Auto" else language
        results = self.model.transcribe(audio=audio_path, language=lang)
        
        return {
            'text': results[0].text,
            'backend': 'PyTorch',
            'model': model
        }


class ModernButton(tk.Canvas):
    """Modern button with hover effects"""
    
    def __init__(self, parent, text, command, width=200, height=48,
                 bg_color=None, fg_color=None, font_size=13, icon=None, **kwargs):
        super().__init__(parent, width=width, height=height, 
                        highlightthickness=0, bd=0, **kwargs)
        
        self.bg_color = bg_color or COLORS['primary']
        self.fg_color = fg_color or COLORS['text']
        self.hover_color = self._lighten(self.bg_color, 1.15)
        self.active_color = self._lighten(self.bg_color, 0.9)
        self.command = command
        self.text = text
        self.icon = icon
        self.font_size = font_size
        self.is_hovered = False
        self.is_pressed = False
        
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Button-1>', self._on_press)
        self.bind('<ButtonRelease-1>', self._on_release)
        
        self.draw()
    
    def _lighten(self, color, factor):
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        new_rgb = tuple(min(255, int(c * factor)) for c in rgb)
        return '#{:02x}{:02x}{:02x}'.format(*new_rgb)
    
    def draw(self):
        self.delete('all')
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        r = 12
        color = self.active_color if self.is_pressed else (self.hover_color if self.is_hovered else self.bg_color)
        self._draw_rounded_rect(0, 0, w, h, r, color)
        
        display_text = f"{self.icon} {self.text}" if self.icon else self.text
        self.create_text(w/2, h/2, text=display_text, fill=self.fg_color,
                        font=('SF Pro Display', self.font_size, 'bold'))
    
    def _draw_rounded_rect(self, x1, y1, x2, y2, radius, color):
        points = [
            x1 + radius, y1, x2 - radius, y1, x2, y1 + radius,
            x2, y2 - radius, x2 - radius, y2, x1 + radius, y2,
            x1, y2 - radius, x1, y1 + radius,
        ]
        self.create_polygon(points, fill=color, smooth=True)
    
    def _on_enter(self, e):
        self.is_hovered = True
        self.draw()
    
    def _on_leave(self, e):
        self.is_hovered = False
        self.is_pressed = False
        self.draw()
    
    def _on_press(self, e):
        self.is_pressed = True
        self.draw()
    
    def _on_release(self, e):
        self.is_pressed = False
        self.draw()
        if self.command:
            self.command()


class WaveformVisualizer(tk.Canvas):
    """Audio level visualizer"""
    
    def __init__(self, parent, width=300, height=80, **kwargs):
        super().__init__(parent, width=width, height=height,
                        highlightthickness=0, bd=0, bg=COLORS['card'], **kwargs)
        self.width = width
        self.height = height
        self.level = 0.0
        self.peak = 0.0
        self.is_speaking = False
        self.history = [0.0] * 40
        self.draw()
    
    def update_level(self, level: float, speaking: bool = False):
        self.level = level
        self.is_speaking = speaking
        if level > self.peak:
            self.peak = level
        else:
            self.peak *= 0.95
        self.history.pop(0)
        self.history.append(level)
        self.draw()
    
    def reset(self):
        self.level = 0.0
        self.peak = 0.0
        self.is_speaking = False
        self.history = [0.0] * 40
        self.draw()
    
    def draw(self):
        self.delete('all')
        self.create_rectangle(0, 0, self.width, self.height, fill=COLORS['card'], outline="")
        
        bar_width = self.width / len(self.history)
        center_y = self.height / 2
        
        for i, amp in enumerate(self.history):
            amp = max(0.02, amp)
            bar_height = amp * self.height * 0.8
            x1 = i * bar_width + 1
            x2 = (i + 1) * bar_width - 1
            y1 = center_y - bar_height / 2
            y2 = center_y + bar_height / 2
            
            if amp < 0.3:
                color = COLORS['success']
            elif amp < 0.7:
                color = COLORS['warning']
            else:
                color = COLORS['error']
            
            self.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        
        # Speaking indicator
        if self.is_speaking:
            self.create_oval(8, 8, 18, 18, fill=COLORS['success'], outline="")
        else:
            self.create_oval(8, 8, 18, 18, fill=COLORS['text_muted'], outline="")


class QwenASRApp:
    """Main application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{VERSION}")
        self.root.geometry("950x750")
        self.root.minsize(850, 650)
        self.root.configure(bg=COLORS['bg'])
        
        # Initialize components
        try:
            self.engine = TranscriptionEngine()
        except RuntimeError as e:
            messagebox.showerror("Error", str(e))
            root.destroy()
            return
        
        self.recorder = AudioRecorder(level_callback=self.on_audio_level)
        self.is_recording = False
        self.status_queue = queue.Queue()
        
        self.setup_ui()
        self.check_status()
    
    def setup_ui(self):
        main = tk.Frame(self.root, bg=COLORS['bg'])
        main.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        self._create_header(main)
        
        # Content
        content = tk.Frame(main, bg=COLORS['bg'])
        content.pack(fill='both', expand=True, pady=(20, 0))
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)
        
        left = self._create_sidebar(content)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 15))
        
        right = self._create_main_area(content)
        right.grid(row=0, column=1, sticky='nsew')
        
        # Status bar
        self.status_bar = tk.Label(main, text="Ready", bg=COLORS['bg'], 
                                  fg=COLORS['text_muted'], font=('SF Pro Text', 11),
                                  anchor='w')
        self.status_bar.pack(fill='x', pady=(15, 0))
    
    def _create_header(self, parent):
        header = tk.Frame(parent, bg=COLORS['bg'])
        header.pack(fill='x')
        
        tk.Label(header, text="🎙️ Qwen3-ASR Pro", font=('SF Pro Display', 20, 'bold'),
                bg=COLORS['bg'], fg=COLORS['text']).pack(side='left')
        
        self.backend_label = tk.Label(header, text="", font=('SF Pro Mono', 10),
                                     bg=COLORS['surface'], fg=COLORS['success'],
                                     padx=10, pady=4)
        self.backend_label.pack(side='right')
        self._update_backend_label()
    
    def _update_backend_label(self):
        if self.engine.backend == 'mlx_audio':
            self.backend_label.config(text="⚡ MLX", fg=COLORS['success'])
        elif self.engine.backend == 'mlx_cli':
            self.backend_label.config(text="⚡ MLX-CLI", fg=COLORS['success'])
        elif self.engine.backend == 'pytorch':
            self.backend_label.config(text="PyTorch", fg=COLORS['warning'])
    
    def _create_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=COLORS['surface'], padx=16, pady=16)
        sidebar.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        # Recording Section
        rec_frame = tk.LabelFrame(sidebar, text=" Recording ", bg=COLORS['card'],
                                  fg=COLORS['text'], font=('SF Pro Text', 11, 'bold'),
                                  padx=12, pady=12)
        rec_frame.pack(fill='x', pady=(0, 12))
        rec_frame.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        self.record_btn = ModernButton(rec_frame, text="Start Recording", 
                                      command=self.toggle_recording,
                                      width=200, height=48,
                                      bg_color=COLORS['success'], icon="🎤")
        self.record_btn.pack(pady=10)
        
        self.record_time = tk.Label(rec_frame, text="00:00", font=('SF Mono', 32, 'bold'),
                                   bg=COLORS['card'], fg=COLORS['text'])
        self.record_time.pack()
        
        tk.Label(rec_frame, text="Audio Level:", font=('SF Pro Text', 10),
                bg=COLORS['card'], fg=COLORS['text_secondary']).pack(anchor='w', pady=(10, 0))
        
        self.waveform = WaveformVisualizer(rec_frame, width=220, height=60)
        self.waveform.pack(fill='x', pady=(0, 5))
        
        self.record_status = tk.Label(rec_frame, text="Ready", font=('SF Pro Text', 11),
                                     bg=COLORS['card'], fg=COLORS['text_muted'])
        self.record_status.pack()
        
        # Upload Section
        upload_frame = tk.LabelFrame(sidebar, text=" Upload ", bg=COLORS['card'],
                                     fg=COLORS['text'], font=('SF Pro Text', 11, 'bold'),
                                     padx=12, pady=12)
        upload_frame.pack(fill='x', pady=(0, 12))
        upload_frame.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        self.drop_zone = tk.Frame(upload_frame, bg=COLORS['surface'], height=80)
        self.drop_zone.pack(fill='x', pady=5)
        self.drop_zone.pack_propagate(False)
        
        drop_text = tk.Label(self.drop_zone, text="📁\nDrop file or click", 
                            font=('SF Pro Text', 11),
                            bg=COLORS['surface'], fg=COLORS['secondary'])
        drop_text.place(relx=0.5, rely=0.5, anchor='center')
        
        self.drop_zone.bind('<Button-1>', lambda e: self.browse_file())
        drop_text.bind('<Button-1>', lambda e: self.browse_file())
        
        # Settings
        settings_frame = tk.LabelFrame(sidebar, text=" Settings ", bg=COLORS['card'],
                                       fg=COLORS['text'], font=('SF Pro Text', 11, 'bold'),
                                       padx=12, pady=12)
        settings_frame.pack(fill='x')
        settings_frame.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        # Model
        tk.Label(settings_frame, text="Model:", font=('SF Pro Text', 10),
                bg=COLORS['card'], fg=COLORS['text_secondary']).pack(anchor='w')
        self.model_combo = ttk.Combobox(settings_frame, 
                                       values=["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B"],
                                       state='readonly', width=25)
        self.model_combo.set("Qwen/Qwen3-ASR-0.6B")
        self.model_combo.pack(fill='x', pady=(0, 10))
        
        # Language
        tk.Label(settings_frame, text="Language:", font=('SF Pro Text', 10),
                bg=COLORS['card'], fg=COLORS['text_secondary']).pack(anchor='w')
        self.lang_combo = ttk.Combobox(settings_frame,
                                      values=["Auto", "English", "Chinese", "Japanese", 
                                             "Korean", "Spanish", "French", "German"],
                                      state='readonly', width=25)
        self.lang_combo.set("Auto")
        self.lang_combo.pack(fill='x', pady=(0, 10))
        
        # Silence Duration
        tk.Label(settings_frame, text="Auto-stop silence:", font=('SF Pro Text', 10),
                bg=COLORS['card'], fg=COLORS['text_secondary']).pack(anchor='w')
        
        self.silence_var = tk.DoubleVar(value=2.0)
        slider_row = tk.Frame(settings_frame, bg=COLORS['card'])
        slider_row.pack(fill='x', pady=(0, 5))
        
        self.silence_slider = ttk.Scale(slider_row, from_=0.5, to=5.0, 
                                       orient='horizontal', variable=self.silence_var,
                                       command=self.on_silence_changed)
        self.silence_slider.pack(side='left', fill='x', expand=True)
        
        self.silence_label = tk.Label(slider_row, text="2.0s", font=('SF Pro Mono', 9),
                                     bg=COLORS['card'], fg=COLORS['secondary'], width=5)
        self.silence_label.pack(side='right', padx=(5, 0))
        
        # Presets
        presets = tk.Frame(settings_frame, bg=COLORS['card'])
        presets.pack(fill='x', pady=(5, 0))
        
        for name, val in [("Fast", 0.8), ("Normal", 2.0), ("Patient", 3.5)]:
            btn = tk.Button(presets, text=name, font=('SF Pro Text', 9),
                           bg=COLORS['surface'], fg=COLORS['text_secondary'],
                           relief='flat', bd=0, cursor='hand2',
                           command=lambda v=val: self.set_silence_preset(v))
            btn.pack(side='left', padx=(0, 6))
        
        return sidebar
    
    def _create_main_area(self, parent):
        content = tk.Frame(parent, bg=COLORS['surface'], padx=16, pady=16)
        content.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        # Header
        header = tk.Frame(content, bg=COLORS['surface'])
        header.pack(fill='x', pady=(0, 10))
        
        tk.Label(header, text="Transcription", font=('SF Pro Display', 16, 'bold'),
                bg=COLORS['surface'], fg=COLORS['text']).pack(side='left')
        
        # Stats
        self.stats_label = tk.Label(header, text="", font=('SF Pro Mono', 9),
                                   bg=COLORS['surface'], fg=COLORS['text_secondary'])
        self.stats_label.pack(side='right')
        
        # Buttons
        btn_frame = tk.Frame(header, bg=COLORS['surface'])
        btn_frame.pack(side='right', padx=(0, 15))
        
        for icon, cmd in [("🗑️", self.clear), ("📋", self.copy), ("💾", self.save)]:
            tk.Button(btn_frame, text=icon, font=('SF Pro', 12),
                     bg=COLORS['surface'], fg=COLORS['text_secondary'],
                     relief='flat', bd=0, cursor='hand2', command=cmd).pack(side='left', padx=2)
        
        # Text area
        self.text_area = scrolledtext.ScrolledText(
            content, wrap=tk.WORD, font=('SF Mono', 13),
            bg=COLORS['card'], fg=COLORS['text'],
            insertbackground=COLORS['primary'],
            padx=15, pady=15, relief='flat', bd=0,
            highlightthickness=1, highlightcolor=COLORS['primary'],
            highlightbackground=COLORS['card_border']
        )
        self.text_area.pack(fill='both', expand=True)
        
        return content
    
    def on_silence_changed(self, value):
        duration = float(value)
        self.silence_label.config(text=f"{duration:.1f}s")
        self.recorder.set_params(duration=duration)
    
    def set_silence_preset(self, seconds):
        self.silence_var.set(seconds)
        self.silence_label.config(text=f"{seconds:.1f}s")
        self.recorder.set_params(duration=seconds)
        self.status_bar.config(text=f"Silence preset: {seconds}s")
    
    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        self.is_recording = True
        self.recorder.set_params(duration=self.silence_var.get())
        
        self.record_btn.text = "Stop"
        self.record_btn.bg_color = COLORS['error']
        self.record_btn.draw()
        
        self.record_status.config(text="Recording...", fg=COLORS['warning'])
        self.recorder.start()
        self.update_timer()
    
    def update_timer(self):
        if self.is_recording and self.recorder.is_recording:
            elapsed = len(self.recorder.frames) * CHUNK_DURATION
            mins, secs = int(elapsed // 60), int(elapsed % 60)
            self.record_time.config(text=f"{mins:02d}:{secs:02d}")
            self.root.after(100, self.update_timer)
        elif self.is_recording and not self.recorder.is_recording:
            # Auto-stopped
            self.stop_recording()
    
    def stop_recording(self):
        self.is_recording = False
        
        self.record_btn.text = "Start Recording"
        self.record_btn.bg_color = COLORS['success']
        self.record_btn.draw()
        
        self.status_bar.config(text="Processing...")
        audio_file = self.recorder.stop()
        self.waveform.reset()
        
        if audio_file:
            self.process_file(audio_file, is_temp=True)
        else:
            self.record_status.config(text="No audio", fg=COLORS['error'])
    
    def on_audio_level(self, level, speaking):
        self.waveform.update_level(level, speaking)
        if speaking:
            self.record_status.config(text="🟢 Speaking", fg=COLORS['success'])
        else:
            silence = self.recorder.silence_counter * CHUNK_DURATION
            self.record_status.config(text=f"⚪ Silence: {silence:.1f}s", fg=COLORS['text_muted'])
    
    def browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Audio",
            filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg"), ("All", "*.*")]
        )
        if path:
            self.process_file(path)
    
    def process_file(self, file_path: str, is_temp: bool = False):
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File not found")
            return
        
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', f"Loading {os.path.basename(file_path)}...")
        self.status_bar.config(text="Processing...")
        
        def progress(msg):
            self.status_queue.put(('status', msg))
        
        def worker():
            try:
                model = self.model_combo.get()
                language = self.lang_combo.get()
                if language == "Auto":
                    language = None
                
                result, stats = self.engine.transcribe(
                    file_path, model=model, language=language,
                    progress_callback=progress
                )
                
                self.status_queue.put(('success', result))
                self.status_queue.put(('stats', stats))
            except Exception as e:
                error_msg = str(e)
                traceback_str = traceback.format_exc()
                print(f"Error: {error_msg}\n{traceback_str}")
                self.status_queue.put(('error', error_msg))
            finally:
                if is_temp and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
        
        threading.Thread(target=worker, daemon=True).start()
    
    def check_status(self):
        try:
            while True:
                msg_type, data = self.status_queue.get_nowait()
                
                if msg_type == 'status':
                    self.status_bar.config(text=data)
                elif msg_type == 'success':
                    self.show_result(data)
                elif msg_type == 'stats':
                    self.update_stats(data)
                elif msg_type == 'error':
                    self.show_error(data)
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_status)
    
    def show_result(self, result):
        self.text_area.delete('1.0', tk.END)
        meta = f"Backend: {result['backend']} | Model: {result['model']}\n{'─'*50}\n\n"
        self.text_area.insert('1.0', meta)
        self.text_area.insert(tk.END, result['text'])
        self.text_area.tag_config("meta", foreground=COLORS['text_muted'])
        self.text_area.tag_add("meta", "1.0", "3.0")
        self.record_status.config(text="Ready", fg=COLORS['text_muted'])
    
    def update_stats(self, stats):
        rtf_str = f"{stats.rtf:.2f}x"
        self.stats_label.config(text=f"RTF: {rtf_str}")
        word_count = len(self.text_area.get('1.0', tk.END).split())
        self.status_bar.config(text=f"✅ Done! {word_count} words | RTF: {rtf_str}")
    
    def show_error(self, error):
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', f"Error: {error}")
        self.status_bar.config(text="❌ Error")
        messagebox.showerror("Transcription Error", error)
    
    def clear(self):
        self.text_area.delete('1.0', tk.END)
        self.status_bar.config(text="Ready")
    
    def copy(self):
        text = self.text_area.get('1.0', tk.END).strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_bar.config(text="📋 Copied!")
    
    def save(self):
        text = self.text_area.get('1.0', tk.END).strip()
        if not text:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=f"transcript_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if path:
            with open(path, 'w') as f:
                f.write(text)
            self.status_bar.config(text=f"💾 Saved!")


def main():
    root = tk.Tk()
    if sys.platform == 'darwin':
        root.tk.call('tk', 'scaling', 2.0)
    
    try:
        app = QwenASRApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
