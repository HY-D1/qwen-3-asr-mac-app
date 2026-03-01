#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         Qwen3-ASR Pro - macOS Speech-to-Text                     ║
║         MLX Optimized • Real-time Streaming • Adaptive UI        ║
╚══════════════════════════════════════════════════════════════════╝

Features:
- Auto language detection (50+ languages supported)
- Live streaming transcription with 0.6B model
- Upload transcription with 1.7B model (best accuracy)
- Raw audio auto-saved
- Responsive design for all window sizes

Version: 3.2.0
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import threading
import concurrent.futures
import os
import sys

# Suppress HuggingFace tokenizers warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import tempfile
import time
import queue
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import wave
import subprocess
import json

# Import constants
from constants import (
    APP_NAME, VERSION, SAMPLE_RATE, CHUNK_DURATION, RECORDINGS_DIR,
    BASE_DIR, C_ASR_DIR, MODEL_CONFIG, LANGUAGE_CONFIG,
    DEFAULT_LANGUAGE, DEFAULT_SILENCE_DURATION,
    LIVE_CHUNK_DURATION, LIVE_MAX_PENDING_CHUNKS, LIVE_MIN_REMAINING_SECONDS,
    MIN_WIDTH_COMPACT, MIN_WIDTH_MOBILE, COLORS
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PerformanceStats:
    """Transcription performance statistics"""
    audio_duration: float = 0.0
    processing_time: float = 0.0
    rtf: float = 0.0
    backend: str = ""
    model: str = ""


@dataclass
class TranscriptionResult:
    """Transcription result with metadata"""
    text: str = ""
    language: Optional[str] = None
    confidence: Optional[float] = None
    backend: str = ""
    model: str = ""
    stats: PerformanceStats = field(default_factory=PerformanceStats)


class ProcessingMode(Enum):
    """Audio processing mode"""
    LIVE = "live"      # Real-time streaming (0.6B model)
    UPLOAD = "upload"  # File upload (1.7B model)


# =============================================================================
# TTK Styles Configuration
# =============================================================================

def configure_ttk_styles():
    """Configure ttk styles for light theme"""
    style = ttk.Style()
    style.theme_use('clam')
    
    # Combobox styling
    style.configure(
        'TCombobox',
        fieldbackground=COLORS['input_bg'],
        background=COLORS['surface'],
        foreground=COLORS['input_fg'],
        arrowcolor=COLORS['text_secondary'],
        selectbackground=COLORS['select_bg'],
        selectforeground=COLORS['select_fg'],
        padding=5
    )
    style.map('TCombobox', 
        fieldbackground=[('readonly', COLORS['input_bg']), ('active', COLORS['surface_light'])],
        selectbackground=[('readonly', COLORS['select_bg'])],
        selectforeground=[('readonly', COLORS['select_fg'])]
    )
    
    # Scale/Slider styling
    style.configure(
        'TScale',
        background=COLORS['surface'],
        troughcolor=COLORS['surface_light'],
        bordercolor=COLORS['card_border']
    )
    
    # Scrollbar styling
    style.configure(
        'TScrollbar',
        background=COLORS['surface_light'],
        troughcolor=COLORS['bg'],
        bordercolor=COLORS['card_border'],
        arrowcolor=COLORS['text_secondary']
    )
    
    # Progress bar
    style.configure(
        'TProgressbar',
        background=COLORS['primary'],
        troughcolor=COLORS['surface_light']
    )
    
    return style


# =============================================================================
# UI Components
# =============================================================================

class CollapsibleSidebar(tk.Frame):
    """Modern collapsible sidebar with simplified controls"""
    
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=COLORS['surface'], **kwargs)
        self.app = app
        self.is_expanded = True
        self.expanded_width = 260
        self.compact_width = 60
        
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Toggle button
        self.toggle_btn = tk.Button(
            self, text="◀", font=('SF Pro', 12),
            bg=COLORS['surface'], fg=COLORS['text_secondary'],
            relief='flat', bd=0, cursor='hand2',
            command=self.toggle_sidebar
        )
        self.toggle_btn.grid(row=0, column=0, pady=8, sticky='n')
        
        # Canvas with scrollbar
        self.canvas = tk.Canvas(self, bg=COLORS['surface'], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.content = tk.Frame(self.canvas, bg=COLORS['surface'])
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content, anchor='nw', width=240)
        
        self.canvas.grid(row=1, column=0, sticky='nsew', padx=4)
        self.scrollbar.grid(row=1, column=1, sticky='ns')
        
        self.content.bind('<Configure>', self._on_frame_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Build sections
        self._build_recording_section()
        self._build_saved_file_section()
        self._build_upload_section()
        self._build_settings_section()
        
        self.config(width=self.expanded_width)
    
    def _on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def _build_recording_section(self):
        """Recording controls"""
        self.rec_frame = tk.LabelFrame(
            self.content, text=" 🎙️ Recording ", 
            bg=COLORS['card'], fg=COLORS['text'], 
            font=('SF Pro Text', 10, 'bold'),
            padx=10, pady=10
        )
        self.rec_frame.pack(fill='x', pady=(0, 10))
        self.rec_frame.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        # Record button
        self.record_btn_frame = tk.Frame(self.rec_frame, bg=COLORS['card'])
        self.record_btn_frame.pack(fill='x', pady=5)
        
        self.record_icon = tk.Label(
            self.record_btn_frame, text="🔴", font=('SF Pro', 28),
            bg=COLORS['card'], fg=COLORS['error'], cursor='hand2'
        )
        self.record_icon.pack()
        self.record_icon.bind('<Button-1>', lambda e: self.app.toggle_recording())
        
        self.record_text = tk.Label(
            self.record_btn_frame, text="Start Recording",
            font=('SF Pro Text', 10, 'bold'),
            bg=COLORS['card'], fg=COLORS['text']
        )
        self.record_text.pack()
        self.record_text.bind('<Button-1>', lambda e: self.app.toggle_recording())
        
        # Timer
        self.record_time = tk.Label(
            self.rec_frame, text="00:00", font=('SF Mono', 24, 'bold'),
            bg=COLORS['card'], fg=COLORS['primary']
        )
        self.record_time.pack(pady=5)
        
        # Waveform
        self.waveform = WaveformVisualizer(self.rec_frame, width=220, height=50)
        self.waveform.pack(fill='x', pady=5)
        
        # Status
        self.record_status = tk.Label(
            self.rec_frame, text="Ready", font=('SF Pro Text', 9),
            bg=COLORS['card'], fg=COLORS['text_muted']
        )
        self.record_status.pack()
        
        # Live mode indicator
        self.mode_indicator = tk.Label(
            self.rec_frame, text="⚡ Live Mode • 0.6B",
            font=('SF Pro Text', 8),
            bg=COLORS['card'], fg=COLORS['secondary']
        )
        self.mode_indicator.pack(pady=(5, 0))
    
    def _build_saved_file_section(self):
        """Saved recording info"""
        self.file_frame = tk.LabelFrame(
            self.content, text=" 💾 Saved ", 
            bg=COLORS['card'], fg=COLORS['text'],
            font=('SF Pro Text', 10, 'bold'),
            padx=10, pady=10
        )
        self.file_frame.pack(fill='x', pady=(0, 10))
        self.file_frame.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        self.file_label = tk.Label(
            self.file_frame, text="No recording", 
            font=('SF Pro Text', 9),
            bg=COLORS['card'], fg=COLORS['text_secondary'],
            wraplength=200, justify='left'
        )
        self.file_label.pack(anchor='w')
        
        self.folder_btn = tk.Button(
            self.file_frame, text="📁 Open Folder",
            font=('SF Pro Text', 9),
            bg=COLORS['surface'], fg=COLORS['text_secondary'],
            relief='flat', bd=0, cursor='hand2',
            command=self.app.open_recordings_folder
        )
        self.folder_btn.pack(fill='x', pady=(8, 0))
    
    def _build_upload_section(self):
        """File upload section"""
        self.upload_frame = tk.LabelFrame(
            self.content, text=" 📤 Upload ", 
            bg=COLORS['card'], fg=COLORS['text'],
            font=('SF Pro Text', 10, 'bold'),
            padx=10, pady=10
        )
        self.upload_frame.pack(fill='x', pady=(0, 10))
        self.upload_frame.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        # Info label
        tk.Label(
            self.upload_frame, text="Auto: 1.7B model\nBest accuracy",
            font=('SF Pro Text', 9),
            bg=COLORS['card'], fg=COLORS['text_secondary'],
            justify='center'
        ).pack(pady=(0, 5))
        
        self.drop_zone = tk.Frame(
            self.upload_frame, bg=COLORS['surface_light'], 
            height=50, highlightbackground=COLORS['primary'],
            highlightthickness=1
        )
        self.drop_zone.pack(fill='x')
        self.drop_zone.pack_propagate(False)
        
        drop_text = tk.Label(
            self.drop_zone, text="📁 Drop or Click", 
            font=('SF Pro Text', 10),
            bg=COLORS['surface_light'], fg=COLORS['secondary']
        )
        drop_text.place(relx=0.5, rely=0.5, anchor='center')
        
        self.drop_zone.bind('<Button-1>', lambda e: self.app.browse_file())
        drop_text.bind('<Button-1>', lambda e: self.app.browse_file())
    
    def _build_settings_section(self):
        """Simplified settings"""
        self.settings_frame = tk.LabelFrame(
            self.content, text=" ⚙️ Settings ", 
            bg=COLORS['card'], fg=COLORS['text'],
            font=('SF Pro Text', 10, 'bold'),
            padx=10, pady=10
        )
        self.settings_frame.pack(fill='x')
        self.settings_frame.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        # Language selection (with Auto-detect)
        tk.Label(
            self.settings_frame, text="Language:", 
            font=('SF Pro Text', 9),
            bg=COLORS['card'], fg=COLORS['text_secondary']
        ).pack(anchor='w')
        
        lang_frame = tk.Frame(self.settings_frame, bg=COLORS['card'])
        lang_frame.pack(fill='x', pady=(2, 8))
        
        self.lang_combo = ttk.Combobox(
            lang_frame,
            values=list(LANGUAGE_CONFIG.keys()),
            state='readonly', width=18, font=('SF Pro Text', 9)
        )
        self.lang_combo.set(DEFAULT_LANGUAGE)
        self.lang_combo.pack(fill='x')
        
        # Silence duration
        tk.Label(
            self.settings_frame, text="Auto-stop silence:", 
            font=('SF Pro Text', 9),
            bg=COLORS['card'], fg=COLORS['text_secondary']
        ).pack(anchor='w', pady=(5, 0))
        
        self.silence_var = tk.DoubleVar(value=DEFAULT_SILENCE_DURATION)
        slider_frame = tk.Frame(self.settings_frame, bg=COLORS['card'])
        slider_frame.pack(fill='x')
        
        self.silence_slider = ttk.Scale(
            slider_frame, from_=0.5, to=60.0,
            orient='horizontal', variable=self.silence_var,
            command=self.app.on_silence_changed
        )
        self.silence_slider.pack(side='left', fill='x', expand=True)
        
        self.silence_label = tk.Label(
            slider_frame, text=f"{DEFAULT_SILENCE_DURATION:.0f}s", 
            font=('SF Pro Mono', 8),
            bg=COLORS['card'], fg=COLORS['secondary'], width=4
        )
        self.silence_label.pack(side='right', padx=(3, 0))
        
        # Presets
        preset_frame = tk.Frame(self.settings_frame, bg=COLORS['card'])
        preset_frame.pack(fill='x', pady=(8, 0))
        
        for name, val in [("Fast", 0.8), ("Class", 30.0), ("Max", 60.0)]:
            btn = tk.Button(
                preset_frame, text=name, font=('SF Pro Text', 8),
                bg=COLORS['surface'], fg=COLORS['text_secondary'],
                relief='flat', bd=0, cursor='hand2',
                command=lambda v=val: self.app.set_silence_preset(v)
            )
            btn.pack(side='left', padx=(0, 5))
    
    def get_language(self) -> str:
        """Get selected language code"""
        lang_name = self.lang_combo.get()
        return LANGUAGE_CONFIG.get(lang_name, {}).get('code', 'auto')
    
    def get_language_name(self) -> str:
        """Get selected language display name"""
        return self.lang_combo.get()
    
    def toggle_sidebar(self):
        """Toggle between expanded and compact modes"""
        self.is_expanded = not self.is_expanded
        
        if self.is_expanded:
            self.config(width=self.expanded_width)
            self.toggle_btn.config(text="◀")
            self.canvas.grid()
            self.scrollbar.grid()
        else:
            self.config(width=self.compact_width)
            self.toggle_btn.config(text="▶")
            self.canvas.grid_remove()
            self.scrollbar.grid_remove()
    
    def adapt_to_width(self, width):
        """Automatically adapt sidebar based on window width"""
        if width < MIN_WIDTH_COMPACT and self.is_expanded:
            self.is_expanded = False
            self.config(width=self.compact_width)
            self.toggle_btn.config(text="▶")
            self.canvas.grid_remove()
            self.scrollbar.grid_remove()
        elif width >= MIN_WIDTH_COMPACT + 100 and not self.is_expanded:
            self.is_expanded = True
            self.config(width=self.expanded_width)
            self.toggle_btn.config(text="◀")
            self.canvas.grid()
            self.scrollbar.grid()


class SlideOutPanel(tk.Frame):
    """Slide-out panel for mobile/small screens"""
    
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=COLORS['surface'], **kwargs)
        self.app = app
        self.is_open = False
        self.panel_width = 300
        
        self.overlay = tk.Frame(parent, bg='black')
        self.overlay.bind('<Button-1>', self.close)
        
        self.content = CollapsibleSidebar(self, app)
        self.content.pack(fill='both', expand=True)
        
        self.place(relx=1.0, rely=0, relheight=1.0, width=0)
    
    def open(self):
        """Slide in from right"""
        self.is_open = True
        self.overlay.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)
        self.overlay.config(bg='black')
        
        for width in range(0, self.panel_width + 1, 20):
            self.place(relx=1.0, rely=0, relheight=1.0, width=width, anchor='ne')
            self.update()
            time.sleep(0.01)
    
    def close(self, event=None):
        """Slide out to right"""
        self.is_open = False
        self.overlay.place_forget()
        
        for width in range(self.panel_width, -1, -20):
            self.place(relx=1.0, rely=0, relheight=1.0, width=width, anchor='ne')
            self.update()
            time.sleep(0.01)
        
        self.place_forget()


class BottomBar(tk.Frame):
    """Bottom control bar for mobile mode"""
    
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=COLORS['surface'], height=60, **kwargs)
        self.app = app
        
        self.record_btn = tk.Button(
            self, text="🔴", font=('SF Pro', 28),
            bg=COLORS['error'], fg='white',
            relief='flat', bd=0, cursor='hand2',
            command=app.toggle_recording
        )
        self.record_btn.pack(side='left', padx=20)
        
        self.timer_label = tk.Label(
            self, text="00:00", font=('SF Mono', 18, 'bold'),
            bg=COLORS['surface'], fg=COLORS['primary']
        )
        self.timer_label.pack(side='left', padx=10)
        
        tk.Frame(self, bg=COLORS['surface']).pack(side='left', expand=True)
        
        self.settings_btn = tk.Button(
            self, text="⚙️", font=('SF Pro', 20),
            bg=COLORS['surface'], fg=COLORS['text_secondary'],
            relief='flat', bd=0, cursor='hand2',
            command=app.toggle_settings_panel
        )
        self.settings_btn.pack(side='right', padx=15)
        
        self.files_btn = tk.Button(
            self, text="📁", font=('SF Pro', 20),
            bg=COLORS['surface'], fg=COLORS['text_secondary'],
            relief='flat', bd=0, cursor='hand2',
            command=app.open_recordings_folder
        )
        self.files_btn.pack(side='right', padx=5)


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


# =============================================================================
# Live Streaming Module
# =============================================================================

class LiveStreamer:
    """
    Live streaming transcription using C implementation.
    Processes audio in chunks for real-time transcription.
    Always uses 0.6B model for stability and low latency.
    """
    
    def __init__(self, 
                 model_dir: str = None,
                 binary_path: str = None,
                 sample_rate: int = SAMPLE_RATE):
        # Set up paths
        base_dir = BASE_DIR
        if model_dir is None:
            model_dir = os.path.join(C_ASR_DIR, MODEL_CONFIG["live"]["model_dir"])
        if binary_path is None:
            binary_path = os.path.join(C_ASR_DIR, "qwen_asr")
            
        self.model_dir = model_dir if os.path.isabs(model_dir) else os.path.join(base_dir, model_dir)
        self.binary_path = binary_path if os.path.isabs(binary_path) else os.path.join(base_dir, binary_path)
        
        self.sample_rate = sample_rate
        self.chunk_duration = LIVE_CHUNK_DURATION
        self.chunk_samples = int(self.chunk_duration * sample_rate)
        
        # State
        self.raw_frames = []
        self.is_running = False
        self.transcript_buffer = ""
        self.current_audio_file = None
        self.output_callback = None
        self.status_callback = None
        self.language = "en"  # Default language code
        self.auto_detect_lang = False
        
        # Threading
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self._pending_chunks = 0
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=LIVE_MAX_PENDING_CHUNKS)
        
        # Stats
        self.stats = {
            'chunks_processed': 0,
            'total_audio_seconds': 0.0,
            'detected_language': None,
        }
    
    def start(self, 
              output_callback: Optional[Callable] = None, 
              status_callback: Optional[Callable] = None,
              language: str = "en",
              auto_detect: bool = False) -> str:
        """
        Start live streaming transcription.
        
        Args:
            output_callback: Called with (text, is_partial) when new text available
            status_callback: Called with status messages
            language: Language code (e.g., 'en', 'zh', 'auto')
            auto_detect: Whether to auto-detect language
            
        Returns:
            Path to raw audio file that will be created
        """
        self.is_running = True
        self.raw_frames = []
        self.audio_buffer = []
        self.transcript_buffer = ""
        self.output_callback = output_callback
        self.status_callback = status_callback
        self.auto_detect_lang = auto_detect
        
        # Set language
        if auto_detect or language == "auto":
            self.language = "auto"
            self.auto_detect_lang = True
        else:
            self.language = language
        
        # Create output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        self.current_audio_file = os.path.join(RECORDINGS_DIR, f"live_{timestamp}.wav")
        
        # Reset stats
        self.stats = {
            'chunks_processed': 0,
            'total_audio_seconds': 0.0,
            'detected_language': None,
        }
        
        return self.current_audio_file
    
    def feed_audio(self, audio_chunk: np.ndarray):
        """Feed audio chunk to streaming engine"""
        if not self.is_running:
            return
        
        # Save for final WAV file
        self.raw_frames.append(audio_chunk.copy())
        
        # Add to processing buffer
        with self.buffer_lock:
            self.audio_buffer.append(audio_chunk)
            total_samples = sum(len(a) for a in self.audio_buffer)
            
            # Process when we have enough audio and not too many pending
            if total_samples >= self.chunk_samples and self._pending_chunks < LIVE_MAX_PENDING_CHUNKS:
                combined = np.concatenate(self.audio_buffer)
                to_process = combined[:self.chunk_samples].copy()
                remaining = combined[self.chunk_samples:]
                
                self.audio_buffer = [remaining] if len(remaining) > 0 else []
                self._pending_chunks += 1
                self.stats['total_audio_seconds'] += self.chunk_duration
                
                # Submit to thread pool
                self._executor.submit(self._process_chunk, to_process)
    
    def _process_chunk(self, audio: np.ndarray):
        """Process a single audio chunk - runs in thread pool"""
        temp_file = None
        try:
            if not self.is_running:
                return
            
            self._notify_status("Processing chunk...")
            
            # Convert to int16
            audio_int16 = np.clip(audio * 32767, -32768, 32768).astype(np.int16)
            
            # Write to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_file = f.name
            
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            # Build command
            cmd = [
                self.binary_path,
                "-d", self.model_dir,
                "-i", temp_file,
            ]
            
            # Add language parameter
            if self.language and self.language != "auto":
                cmd.extend(["--language", self.language])
            
            # Run C binary
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            # Parse output
            stdout_text = result.stdout.decode('utf-8', errors='replace')
            stderr_text = result.stderr.decode('utf-8', errors='replace')
            
            # Extract transcription
            transcript = self._extract_transcription(stdout_text)
            
            if transcript:
                self.transcript_buffer += transcript + " "
                self.stats['chunks_processed'] += 1
                
                # Try to detect language from first chunk
                if self.stats['chunks_processed'] == 1 and self.auto_detect_lang:
                    detected = self._detect_language_from_text(transcript)
                    if detected:
                        self.stats['detected_language'] = detected
                
                if self.output_callback:
                    self.output_callback(transcript + " ", is_partial=True)
            
            # Show timing info
            for line in stderr_text.split('\n'):
                if "Inference:" in line or "Audio:" in line:
                    self._notify_status(line.strip())
                    
        except Exception as e:
            print(f"LiveStreamer: Error processing chunk: {e}")
        finally:
            with self.buffer_lock:
                self._pending_chunks = max(0, self._pending_chunks - 1)
            self._cleanup_temp_file(temp_file)
    
    def _process_chunk_sync(self, audio: np.ndarray):
        """Process chunk synchronously (for remaining audio at stop)"""
        temp_file = None
        try:
            audio_int16 = np.clip(audio * 32767, -32768, 32768).astype(np.int16)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_file = f.name
            
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            cmd = [
                self.binary_path,
                "-d", self.model_dir,
                "-i", temp_file,
            ]
            
            if self.language and self.language != "auto":
                cmd.extend(["--language", self.language])
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            stdout_text = result.stdout.decode('utf-8', errors='replace')
            transcript = self._extract_transcription(stdout_text)
            
            if transcript:
                self.transcript_buffer += transcript + " "
                if self.output_callback:
                    self.output_callback(transcript + " ", is_partial=True)
                    
        except Exception as e:
            print(f"LiveStreamer: Error in sync processing: {e}")
        finally:
            self._cleanup_temp_file(temp_file)
    
    def _extract_transcription(self, stdout_text: str) -> str:
        """Extract transcription from C binary output"""
        for line in stdout_text.split('\n'):
            line = line.strip()
            if line and not line.startswith('Inference:') and not line.startswith('Audio:'):
                return line
        return ""
    
    def _detect_language_from_text(self, text: str) -> Optional[str]:
        """
        Simple language detection based on character ranges.
        Returns language code or None if uncertain.
        """
        if not text:
            return None
        
        # Count characters by script
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
        has_japanese = any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text)
        has_korean = any('\uac00' <= c <= '\ud7af' for c in text)
        has_cyrillic = any('\u0400' <= c <= '\u04ff' for c in text)
        has_arabic = any('\u0600' <= c <= '\u06ff' for c in text)
        has_thai = any('\u0e00' <= c <= '\u0e7f' for c in text)
        
        if has_chinese:
            return 'zh'
        elif has_japanese:
            return 'ja'
        elif has_korean:
            return 'ko'
        elif has_cyrillic:
            return 'ru'
        elif has_arabic:
            return 'ar'
        elif has_thai:
            return 'th'
        
        # Default to English for Latin script
        return 'en'
    
    def _notify_status(self, message: str):
        """Send status notification"""
        if self.status_callback:
            self.status_callback(message)
    
    def _cleanup_temp_file(self, temp_file: Optional[str]):
        """Safely remove temp file"""
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def stop(self) -> tuple:
        """
        Stop streaming and return results.
        
        Returns:
            (audio_file_path, transcript_text)
        """
        self.is_running = False
        
        # Wait for pending chunks
        for _ in range(60):
            with self.buffer_lock:
                if self._pending_chunks == 0:
                    break
            time.sleep(0.1)
        
        # Shutdown executor
        self._executor.shutdown(wait=False)
        
        # Process remaining audio
        with self.buffer_lock:
            if self.audio_buffer:
                remaining = np.concatenate(self.audio_buffer)
                min_samples = int(self.sample_rate * LIVE_MIN_REMAINING_SECONDS)
                if len(remaining) > min_samples:
                    self._process_chunk_sync(remaining)
                self.audio_buffer = []
        
        # Save WAV file
        if self.raw_frames and self.current_audio_file:
            audio = np.concatenate(self.raw_frames)
            audio_int16 = np.clip(audio * 32767, -32768, 32768).astype(np.int16)
            
            with wave.open(self.current_audio_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
        
        return self.current_audio_file, self.transcript_buffer.strip()


# =============================================================================
# Audio Recording Module
# =============================================================================

class AudioRecorder:
    """Audio recorder with VAD (Voice Activity Detection)"""
    
    def __init__(self, 
                 silence_threshold: float = 0.015,
                 silence_duration: float = DEFAULT_SILENCE_DURATION,
                 level_callback: Optional[Callable] = None):
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
        """Update recording parameters"""
        if threshold is not None:
            self.silence_threshold = threshold
        if duration is not None:
            self.silence_duration = duration
    
    def _calculate_level(self, audio_chunk: np.ndarray) -> float:
        """Calculate audio level (RMS)"""
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        return min(1.0, rms * 10)
    
    def _vad(self, audio_chunk: np.ndarray) -> bool:
        """Voice activity detection"""
        level = self._calculate_level(audio_chunk)
        self.current_level = level
        self.is_speaking = level > self.silence_threshold
        if self.level_callback:
            self.level_callback(level, self.is_speaking)
        return self.is_speaking
    
    def start(self):
        """Start recording"""
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
        """Stop recording and return temp file path"""
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if not self.frames:
            return None
        
        audio = np.concatenate(self.frames)
        audio_int16 = np.clip(audio * 32767, -32768, 32768).astype(np.int16)
        temp_file = tempfile.mktemp(suffix='.wav')
        
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        
        return temp_file
    
    def get_raw_audio(self) -> Optional[np.ndarray]:
        """Get raw audio as numpy array"""
        if not self.frames:
            return None
        return np.concatenate(self.frames)


# =============================================================================
# Transcription Engine (Upload Mode)
# =============================================================================

class TranscriptionEngine:
    """
    Transcription engine for file uploads.
    Auto-detects backend (MLX/PyTorch) and always uses 1.7B model for best accuracy.
    Supports auto language detection.
    """
    
    def __init__(self):
        self.backend = None
        self.model = None
        self.model_name = None
        self.supported_languages = set(LANGUAGE_CONFIG.keys())
        self._detect_backend()
    
    def load_model(self, model_name: str, dtype: str = "float16"):
        """
        Load a model (backward compatibility).
        In the new design, models are loaded automatically during transcribe().
        This method just stores the model name for potential future use.
        """
        self.model_name = model_name
        # Note: Actual model loading happens in transcribe() based on MODEL_CONFIG
    
    def _detect_backend(self):
        """Detect available ML backend"""
        # Try MLX Audio first (fastest on Apple Silicon)
        try:
            import mlx_audio.stt as mlx_stt
            self.backend = 'mlx_audio'
            print("✅ Using mlx-audio backend")
            return
        except ImportError:
            pass
        
        # Try MLX CLI
        try:
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
        
        # Fall back to PyTorch
        try:
            import qwen_asr
            self.backend = 'pytorch'
            print("✅ Using PyTorch (qwen-asr) backend")
            return
        except ImportError:
            pass
        
        self.backend = None
        raise RuntimeError("No transcription backend available. Please run SETUP.command")
    
    def transcribe(self, 
                   audio_path: str,
                   language: Optional[str] = None,
                   progress_callback: Optional[Callable] = None) -> TranscriptionResult:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code or None for auto-detect
            progress_callback: Called with progress messages
            
        Returns:
            TranscriptionResult with text and metadata
        """
        result = TranscriptionResult()
        stats = PerformanceStats()
        start_time = time.time()
        
        # Get audio duration
        try:
            import librosa
            stats.audio_duration = librosa.get_duration(path=audio_path)
        except:
            stats.audio_duration = 0
        
        if progress_callback:
            progress_callback("Transcribing...")
        
        # Always use 1.7B model for uploads
        model = MODEL_CONFIG["upload"]["model_id"]
        
        # Handle language
        lang_code = language if language and language != "auto" else None
        
        try:
            if self.backend == 'mlx_audio':
                result_data = self._transcribe_mlx_audio(audio_path, model, lang_code)
            elif self.backend == 'mlx_cli':
                result_data = self._transcribe_mlx_cli(audio_path, model, lang_code)
            elif self.backend == 'pytorch':
                result_data = self._transcribe_pytorch(audio_path, model, lang_code)
            else:
                raise RuntimeError("No backend available")
            
            # Fill result
            result.text = result_data.get('text', '')
            result.backend = result_data.get('backend', self.backend)
            result.model = result_data.get('model', model)
            
            # Try to detect language if auto
            if not lang_code and result.text:
                result.language = self._detect_language(result.text)
            else:
                result.language = lang_code
            
        except Exception as e:
            result.text = f""
            raise e
        
        stats.processing_time = time.time() - start_time
        if stats.audio_duration > 0:
            stats.rtf = stats.processing_time / stats.audio_duration
        stats.backend = result.backend
        stats.model = result.model
        
        result.stats = stats
        return result
    
    def _transcribe_mlx_audio(self, audio_path: str, model: str, language: Optional[str]):
        """Transcribe using mlx-audio"""
        import mlx_audio.stt as mlx_stt
        
        if self.model is None or self.model_name != model:
            self.model = mlx_stt.load(model)
            self.model_name = model
        
        # Generate with or without language hint
        if language:
            result = self.model.generate(audio_path, language=language)
        else:
            result = self.model.generate(audio_path)
        
        return {
            'text': result.text if hasattr(result, 'text') else str(result),
            'backend': 'MLX-Audio',
            'model': model
        }
    
    def _transcribe_mlx_cli(self, audio_path: str, model: str, language: Optional[str]):
        """Transcribe using mlx-qwen3-asr CLI"""
        cmd = [
            sys.executable, '-m', 'mlx_qwen3_asr',
            audio_path,
            '--model', model,
            '--dtype', 'float16',
            '--stdout-only'
        ]
        
        if language:
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
    
    def _transcribe_pytorch(self, audio_path: str, model: str, language: Optional[str]):
        """Transcribe using PyTorch"""
        import torch
        from qwen_asr import Qwen3ASRModel
        
        if self.model is None or self.model_name != model:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self.model = Qwen3ASRModel.from_pretrained(
                model,
                dtype=torch.float16,
                device_map=device
            )
            self.model_name = model
        
        lang = None if language == "auto" else language
        results = self.model.transcribe(audio=audio_path, language=lang)
        
        return {
            'text': results[0].text if results else "",
            'backend': 'PyTorch',
            'model': model
        }
    
    def _detect_language(self, text: str) -> Optional[str]:
        """
        Detect language from transcribed text.
        Simple heuristic based on character ranges.
        """
        if not text:
            return None
        
        # Character range checks
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text[:1000])
        has_japanese = any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text[:1000])
        has_korean = any('\uac00' <= c <= '\ud7af' for c in text[:1000])
        has_cyrillic = any('\u0400' <= c <= '\u04ff' for c in text[:1000])
        has_arabic = any('\u0600' <= c <= '\u06ff' for c in text[:1000])
        has_thai = any('\u0e00' <= c <= '\u0e7f' for c in text[:1000])
        
        if has_chinese:
            return 'zh'
        elif has_japanese:
            return 'ja'
        elif has_korean:
            return 'ko'
        elif has_cyrillic:
            return 'ru'
        elif has_arabic:
            return 'ar'
        elif has_thai:
            return 'th'
        
        return 'en'


# =============================================================================
# Main Application
# =============================================================================

class QwenASRApp:
    """Main application controller with responsive UI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{VERSION}")
        self.root.geometry("1100x800")
        self.root.minsize(450, 550)
        self.root.configure(bg=COLORS['bg'])
        
        # Configure styles
        self.style = configure_ttk_styles()
        
        # Window state
        self.current_width = 1100
        self.layout_mode = "desktop"
        
        # Recording state
        self.is_recording = False
        self.recording_start_time = None
        self.elapsed_seconds = 0
        
        # Transcription state
        self.live_transcript = ""
        self.current_raw_file = None
        self.status_queue = queue.Queue()
        
        # Initialize components
        try:
            self.engine = TranscriptionEngine()
        except RuntimeError as e:
            messagebox.showerror("Error", str(e))
            root.destroy()
            return
        
        self.recorder = AudioRecorder(level_callback=self.on_audio_level)
        self.live_streamer = LiveStreamer()
        
        # Bind events
        self.root.bind('<Configure>', self.on_window_resize)
        
        # Build UI
        self.setup_ui()
        self.check_status()
    
    # =====================================================================
    # UI Setup & Layout
    # =====================================================================
    
    def setup_ui(self):
        """Setup main UI components"""
        main = tk.Frame(self.root, bg=COLORS['bg'])
        main.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Create components
        self.sidebar = CollapsibleSidebar(main, self)
        self.main_content = self._create_main_content(main)
        self.bottom_bar = BottomBar(self.root, self)
        self.slide_panel = SlideOutPanel(self.root, self)
        
        # Initial layout
        self.sidebar.pack(side='left', fill='y', padx=(0, 10))
        self.main_content.pack(side='left', fill='both', expand=True)
    
    def _create_main_content(self, parent):
        """Create main content area"""
        content = tk.Frame(parent, bg=COLORS['surface'])
        content.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        # Header
        header = tk.Frame(content, bg=COLORS['surface'], padx=15, pady=10)
        header.pack(fill='x')
        
        title_frame = tk.Frame(header, bg=COLORS['surface'])
        title_frame.pack(side='left')
        
        tk.Label(
            title_frame, text="🎓 Live Transcription", 
            font=('SF Pro Display', 16, 'bold'),
            bg=COLORS['surface'], fg=COLORS['text']
        ).pack(anchor='w')
        
        tk.Label(
            title_frame, text="Real-time recording with auto-save • Auto language detection",
            font=('SF Pro Text', 10),
            bg=COLORS['surface'], fg=COLORS['text_secondary']
        ).pack(anchor='w')
        
        # Live indicator
        self.live_indicator = tk.Label(
            header, text="● LIVE", font=('SF Pro Mono', 11, 'bold'),
            bg=COLORS['surface'], fg=COLORS['text_muted']
        )
        self.live_indicator.pack(side='right', padx=10)
        
        # Stats
        self.stats_label = tk.Label(
            header, text="", font=('SF Pro Mono', 9),
            bg=COLORS['surface'], fg=COLORS['text_secondary']
        )
        self.stats_label.pack(side='right', padx=5)
        
        # Action buttons
        btn_frame = tk.Frame(header, bg=COLORS['surface'])
        btn_frame.pack(side='right', padx=5)
        
        for icon, cmd in [("🗑️", self.clear), ("📋", self.copy), ("💾", self.save)]:
            tk.Button(
                btn_frame, text=icon, font=('SF Pro', 14),
                bg=COLORS['surface'], fg=COLORS['text_secondary'],
                relief='flat', bd=0, cursor='hand2', command=cmd
            ).pack(side='left', padx=3)
        
        # Text area
        text_frame = tk.Frame(content, bg=COLORS['card'], padx=2, pady=2)
        text_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        self.text_area = scrolledtext.ScrolledText(
            text_frame, wrap=tk.WORD, font=('SF Mono', 12),
            bg=COLORS['card'], fg=COLORS['text'],
            insertbackground=COLORS['primary'],
            padx=12, pady=12, relief='flat', bd=0,
            highlightthickness=0
        )
        self.text_area.pack(fill='both', expand=True)
        
        # Configure tags
        self.text_area.tag_config("live", foreground=COLORS['secondary'])
        self.text_area.tag_config("meta", foreground=COLORS['text_muted'])
        self.text_area.tag_config("title", foreground=COLORS['primary'], font=('SF Pro Display', 14, 'bold'))
        self.text_area.tag_config("detected", foreground=COLORS['success'], font=('SF Pro Text', 10))
        
        # Welcome message
        self._show_welcome_message()
        
        return content
    
    def _show_welcome_message(self):
        """Show initial welcome message"""
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', "🎓 Qwen3-ASR Pro Ready\n\n", "title")
        self.text_area.insert(tk.END, "Click ")
        self.text_area.insert(tk.END, "🔴 Start Recording", "live")
        self.text_area.insert(tk.END, " to begin live transcription.\n\n")
        self.text_area.insert(tk.END, "Features:\n", "meta")
        self.text_area.insert(tk.END, "• Live mode: 0.6B model (low latency)\n")
        self.text_area.insert(tk.END, "• Upload mode: 1.7B model (best accuracy)\n")
        self.text_area.insert(tk.END, "• Auto language detection (50+ languages)\n")
        self.text_area.insert(tk.END, "• Raw audio auto-saved\n\n")
        self.text_area.insert(tk.END, f"Recordings saved to:\n{RECORDINGS_DIR}", "meta")
    
    def on_window_resize(self, event):
        """Handle window resize"""
        if event.widget == self.root:
            new_width = event.width
            if abs(new_width - self.current_width) > 50:
                self.current_width = new_width
                self.adapt_layout(new_width)
    
    def adapt_layout(self, width):
        """Adapt UI based on window width"""
        if width < MIN_WIDTH_MOBILE and self.layout_mode != "mobile":
            self.layout_mode = "mobile"
            self.show_mobile_layout()
        elif width < MIN_WIDTH_COMPACT and self.layout_mode != "compact":
            self.layout_mode = "compact"
            self.show_compact_layout()
        elif width >= MIN_WIDTH_COMPACT and self.layout_mode != "desktop":
            self.layout_mode = "desktop"
            self.show_desktop_layout()
    
    def show_desktop_layout(self):
        """Show full desktop layout"""
        self.bottom_bar.pack_forget()
        self.slide_panel.place_forget()
        self.sidebar.pack(side='left', fill='y', padx=(0, 10))
        self.main_content.pack(side='left', fill='both', expand=True)
    
    def show_compact_layout(self):
        """Show compact layout with collapsed sidebar"""
        self.bottom_bar.pack_forget()
        self.slide_panel.place_forget()
        self.sidebar.pack(side='left', fill='y', padx=(0, 10))
        self.main_content.pack(side='left', fill='both', expand=True)
        if self.sidebar.is_expanded:
            self.sidebar.toggle_sidebar()
    
    def show_mobile_layout(self):
        """Show mobile layout"""
        self.sidebar.pack_forget()
        self.main_content.pack(fill='both', expand=True)
        self.bottom_bar.pack(side='bottom', fill='x')
    
    def toggle_settings_panel(self):
        """Toggle settings panel (mobile)"""
        if self.slide_panel.is_open:
            self.slide_panel.close()
        else:
            self.slide_panel.open()
    
    # =====================================================================
    # Recording Controls
    # =====================================================================
    
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start live recording"""
        self.is_recording = True
        self.live_transcript = ""
        self.recording_start_time = time.time()
        self.elapsed_seconds = 0
        
        # Clear and show start message
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', "🎓 Live Transcription\n\n", "title")
        self.text_area.insert(tk.END, "Listening...\n", "meta")
        
        # Update UI
        self.live_indicator.config(fg=COLORS['error'])
        self.sidebar.record_icon.config(fg=COLORS['error'])
        self.sidebar.record_text.config(text="Stop Recording")
        self.sidebar.record_status.config(text="🔴 Recording...", fg=COLORS['error'])
        
        # Get settings
        language = self.sidebar.get_language()
        auto_detect = (self.sidebar.get_language_name() == "Auto")
        
        # Start live streamer
        model_dir = os.path.join(C_ASR_DIR, MODEL_CONFIG["live"]["model_dir"])
        self.live_streamer.model_dir = model_dir
        
        raw_file = self.live_streamer.start(
            output_callback=self.on_live_transcript,
            status_callback=self.on_live_status,
            language=language,
            auto_detect=auto_detect
        )
        self.current_raw_file = raw_file
        
        # Start audio capture
        self.recorder.set_params(duration=self.sidebar.silence_var.get())
        self._start_audio_capture()
        self.update_timer()
    
    def _start_audio_capture(self):
        """Start capturing audio for live streaming"""
        import sounddevice as sd
        
        audio_buffer = []
        chunk_samples = int(SAMPLE_RATE * LIVE_CHUNK_DURATION)
        
        def audio_callback(indata, frame_count, time_info, status):
            if not self.is_recording:
                return
            
            audio = indata.copy().flatten()
            audio_buffer.append(audio)
            total_samples = sum(len(a) for a in audio_buffer)
            
            # Update waveform
            level = self.recorder._calculate_level(audio)
            is_speech = level > self.recorder.silence_threshold
            if self.recorder.level_callback:
                self.recorder.level_callback(level, is_speech)
            
            # Feed to streamer when we have enough audio
            if total_samples >= chunk_samples:
                combined = np.concatenate(audio_buffer)
                to_feed = combined[:chunk_samples]
                remaining = combined[chunk_samples:]
                self.live_streamer.feed_audio(to_feed)
                audio_buffer.clear()
                if len(remaining) > 0:
                    audio_buffer.append(remaining)
        
        self.audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=int(SAMPLE_RATE * 0.1),
            callback=audio_callback
        )
        self.audio_stream.start()
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        self.recording_start_time = None
        
        # Update UI
        self.live_indicator.config(fg=COLORS['text_muted'])
        self.sidebar.record_icon.config(fg=COLORS['success'])
        self.sidebar.record_text.config(text="Start Recording")
        self.sidebar.record_status.config(text="Processing...", fg=COLORS['warning'])
        
        # Stop audio stream
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        
        # Stop live streamer
        raw_file, transcript = self.live_streamer.stop()
        
        # Update file info
        if raw_file and os.path.exists(raw_file):
            file_size = os.path.getsize(raw_file) / (1024 * 1024)
            self.sidebar.file_label.config(
                text=f"📁 {os.path.basename(raw_file)}\n💾 {file_size:.1f} MB",
                fg=COLORS['success']
            )
        
        # Show final result
        self._show_final_transcript(raw_file, transcript)
        self.sidebar.record_status.config(text="✅ Saved", fg=COLORS['success'])
        self.sidebar.waveform.reset()
    
    def _show_final_transcript(self, raw_file: str, transcript: str):
        """Display final transcript with metadata"""
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', "✅ Recording Complete\n", "title")
        
        if raw_file:
            self.text_area.insert(tk.END, f"📁 Saved: {os.path.basename(raw_file)}\n", "meta")
        
        # Show detected language
        detected_lang = self.live_streamer.stats.get('detected_language')
        if detected_lang:
            lang_name = self._get_language_name(detected_lang)
            self.text_area.insert(tk.END, f"🌐 Detected: {lang_name}\n", "detected")
        
        self.text_area.insert(tk.END, "─" * 50 + "\n\n", "meta")
        self.text_area.insert(tk.END, transcript if transcript else "(No speech detected)")
    
    def update_timer(self):
        """Update recording timer"""
        if self.is_recording and self.recording_start_time:
            elapsed = int(time.time() - self.recording_start_time)
            self.elapsed_seconds = elapsed
            mins, secs = elapsed // 60, elapsed % 60
            time_str = f"{mins:02d}:{secs:02d}"
            
            self.sidebar.record_time.config(text=time_str)
            if hasattr(self, 'bottom_bar'):
                self.bottom_bar.timer_label.config(text=time_str)
            
            self.root.after(100, self.update_timer)
    
    def on_live_transcript(self, text: str, is_partial: bool = True):
        """Handle live transcript update from streamer"""
        self.live_transcript += text
        self.root.after(0, self._update_transcript_ui)
    
    def _update_transcript_ui(self):
        """Update transcript UI (called from main thread)"""
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', "🎓 Live Transcription\n", "title")
        
        # Show detected language if available
        detected_lang = self.live_streamer.stats.get('detected_language')
        if detected_lang:
            lang_name = self._get_language_name(detected_lang)
            self.text_area.insert(tk.END, f"🌐 Detected: {lang_name}\n", "detected")
        
        self.text_area.insert(tk.END, "─" * 50 + "\n\n", "meta")
        self.text_area.insert(tk.END, self.live_transcript)
        self.text_area.see(tk.END)
    
    def on_live_status(self, status: str):
        """Handle status updates from streamer"""
        if "Loading" in status:
            self.root.after(0, lambda: self.text_area.insert(tk.END, "\n⏳ Loading model...\n", "meta"))
    
    def on_audio_level(self, level: float, speaking: bool):
        """Handle audio level update"""
        self.sidebar.waveform.update_level(level, speaking)
        
        if speaking:
            self.sidebar.record_status.config(text="🟢 Speaking", fg=COLORS['success'])
        else:
            silence = self.recorder.silence_counter * CHUNK_DURATION
            self.sidebar.record_status.config(text=f"⚪ Silence: {silence:.0f}s", fg=COLORS['text_muted'])
    
    def on_silence_changed(self, value):
        """Handle silence duration change"""
        duration = float(value)
        self.sidebar.silence_label.config(text=f"{duration:.0f}s")
        self.recorder.set_params(duration=duration)
    
    def set_silence_preset(self, seconds: float):
        """Set silence preset"""
        self.sidebar.silence_var.set(seconds)
        self.sidebar.silence_label.config(text=f"{seconds:.0f}s")
        self.recorder.set_params(duration=seconds)
    
    # =====================================================================
    # File Processing
    # =====================================================================
    
    def browse_file(self):
        """Browse for audio file"""
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac"),
                ("All Files", "*.*")
            ]
        )
        if path:
            self.process_file(path)
    
    def process_file(self, file_path: str, is_temp: bool = False):
        """Process uploaded audio file"""
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File not found")
            return
        
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', f"Processing {os.path.basename(file_path)}...\n", "title")
        self.text_area.insert(tk.END, "\nUsing 1.7B model for best accuracy...", "meta")
        
        def worker():
            try:
                language = self.sidebar.get_language()
                lang_name = self.sidebar.get_language_name()
                
                result = self.engine.transcribe(
                    file_path, 
                    language=language if lang_name != "Auto" else None
                )
                
                self.status_queue.put(('success', result))
            except Exception as e:
                traceback.print_exc()
                self.status_queue.put(('error', str(e)))
            finally:
                if is_temp and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
        
        threading.Thread(target=worker, daemon=True).start()
    
    def check_status(self):
        """Check status queue for results"""
        try:
            while True:
                msg_type, data = self.status_queue.get_nowait()
                
                if msg_type == 'success':
                    self.show_result(data)
                elif msg_type == 'error':
                    self.show_error(data)
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_status)
    
    def show_result(self, result: TranscriptionResult):
        """Show transcription result"""
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', "📝 Transcription Result\n", "title")
        self.text_area.insert(tk.END, f"Backend: {result.backend} | Model: {result.model}\n", "meta")
        
        # Show detected language
        if result.language:
            lang_name = self._get_language_name(result.language)
            self.text_area.insert(tk.END, f"🌐 Language: {lang_name}\n", "detected")
        
        stats = result.stats
        if stats.rtf > 0:
            self.text_area.insert(tk.END, f"RTF: {stats.rtf:.2f}x | Duration: {stats.audio_duration:.1f}s\n", "meta")
        
        self.text_area.insert(tk.END, "─" * 50 + "\n\n", "meta")
        self.text_area.insert(tk.END, result.text)
        self.sidebar.record_status.config(text="Ready", fg=COLORS['text_muted'])
    
    def show_error(self, error_msg: str):
        """Show error message"""
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', "❌ Error\n\n", "title")
        self.text_area.insert(tk.END, str(error_msg))
        self.sidebar.record_status.config(text="Error", fg=COLORS['error'])
        messagebox.showerror("Transcription Error", str(error_msg))
    
    # =====================================================================
    # Utilities
    # =====================================================================
    
    def _get_language_name(self, code: str) -> str:
        """Get language name from code"""
        for name, config in LANGUAGE_CONFIG.items():
            if config.get('code') == code:
                return name
        return code
    
    def open_recordings_folder(self):
        """Open recordings folder in Finder"""
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        os.system(f'open "{RECORDINGS_DIR}"')
    
    def clear(self):
        """Clear text area"""
        self.text_area.delete('1.0', tk.END)
        self.stats_label.config(text="")
        self._show_welcome_message()
    
    def copy(self):
        """Copy transcript to clipboard"""
        text = self.text_area.get('1.0', tk.END).strip()
        # Clean up
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            if not any(line.startswith(p) for p in ['Backend:', 'RTF:', '─', '🎓', '✅', '⏳', '📁', '🌐', '📝']):
                clean_lines.append(line)
        clean_text = '\n'.join(clean_lines).strip()
        
        if clean_text:
            self.root.clipboard_clear()
            self.root.clipboard_append(clean_text)
            self.stats_label.config(text="Copied!", fg=COLORS['success'])
    
    def save(self):
        """Save transcript to file"""
        text = self.text_area.get('1.0', tk.END).strip()
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            if not any(line.startswith(p) for p in ['Backend:', 'RTF:', '─', '🎓', '✅', '⏳', '📁', '🌐', '📝']):
                clean_lines.append(line)
        clean_text = '\n'.join(clean_lines).strip()
        
        if clean_text:
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
            )
            if path:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(clean_text)
                self.stats_label.config(text="Saved!", fg=COLORS['success'])


# =============================================================================
# Entry Point
# =============================================================================

def main():
    root = tk.Tk()
    app = QwenASRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
