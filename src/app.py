#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         Qwen3-ASR Pro - macOS Speech-to-Text                     ║
║         MLX Optimized • Real-time Streaming • Adaptive UI        ║
╚══════════════════════════════════════════════════════════════════╝

Features:
- Collapsible sidebar with icon/compact modes
- Responsive design for small windows
- Live streaming transcription
- Raw audio saved automatically
- High-contrast accessible design
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
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import wave
import subprocess

# Constants
APP_NAME = "Qwen3-ASR Pro"
VERSION = "3.1.1"
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.05
RECORDINGS_DIR = os.path.expanduser("~/Documents/Qwen3-ASR-Recordings")

# Responsive breakpoints
MIN_WIDTH_COMPACT = 750  # Switch to compact mode
MIN_WIDTH_MOBILE = 550   # Switch to mobile/bottom bar mode

# Colors - Light theme (easy on eyes)
COLORS = {
    'bg': "#f8fafc",              # Light slate background
    'surface': "#ffffff",          # White surface
    'surface_light': "#f1f5f9",    # Light gray
    'card': "#ffffff",             # White card
    'card_border': "#e2e8f0",      # Light border
    'primary': "#4f46e5",          # Indigo
    'primary_hover': "#4338ca",     # Darker indigo
    'secondary': "#0891b2",        # Cyan
    'success': "#16a34a",          # Green
    'warning': "#d97706",          # Orange
    'error': "#dc2626",            # Red
    'text': "#1e293b",             # Dark slate text
    'text_secondary': "#475569",   # Medium slate
    'text_muted': "#94a3b8",       # Light slate
    'input_bg': "#ffffff",         # White input
    'input_fg': "#1e293b",         # Dark text
    'select_bg': "#4f46e5",        # Indigo select
    'select_fg': "#ffffff",        # White on select
}


def configure_ttk_styles():
    """Configure ttk styles for light theme"""
    style = ttk.Style()
    
    # Configure TCombobox for light theme
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
    
    # Combobox dropdown styling
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


@dataclass
class PerformanceStats:
    audio_duration: float = 0.0
    processing_time: float = 0.0
    rtf: float = 0.0


class CollapsibleSidebar(tk.Frame):
    """Modern collapsible sidebar with icon and expanded modes"""
    
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=COLORS['surface'], **kwargs)
        self.app = app
        self.is_expanded = True
        self.expanded_width = 260
        self.compact_width = 60
        
        # Configure grid
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Toggle button at top
        self.toggle_btn = tk.Button(
            self, text="◀", font=('SF Pro', 12),
            bg=COLORS['surface'], fg=COLORS['text_secondary'],
            relief='flat', bd=0, cursor='hand2',
            command=self.toggle_sidebar
        )
        self.toggle_btn.grid(row=0, column=0, pady=8, sticky='n')
        
        # Create canvas with scrollbar for scrollable content
        self.canvas = tk.Canvas(self, bg=COLORS['surface'], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Content frame inside canvas
        self.content = tk.Frame(self.canvas, bg=COLORS['surface'])
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content, anchor='nw', width=240)
        
        # Grid the canvas and scrollbar
        self.canvas.grid(row=1, column=0, sticky='nsew', padx=4)
        self.scrollbar.grid(row=1, column=1, sticky='ns')
        
        # Update scroll region when content changes
        self.content.bind('<Configure>', self._on_frame_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # Build sections
        self._build_recording_section()
        self._build_saved_file_section()
        self._build_upload_section()
        self._build_settings_section()
        
        # Set initial width
        self.config(width=self.expanded_width)
    
    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event):
        """When canvas is resized, resize the inner frame width"""
        canvas_width = event.width
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
    
    def _build_recording_section(self):
        """Recording controls section"""
        self.rec_frame = tk.LabelFrame(
            self.content, text=" Recording ", 
            bg=COLORS['card'], fg=COLORS['text'], 
            font=('SF Pro Text', 10, 'bold'),
            padx=10, pady=10
        )
        self.rec_frame.pack(fill='x', pady=(0, 10))
        self.rec_frame.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        # Record button with icon
        self.record_btn_frame = tk.Frame(self.rec_frame, bg=COLORS['card'])
        self.record_btn_frame.pack(fill='x', pady=5)
        
        self.record_icon = tk.Label(
            self.record_btn_frame, text="🎙️", font=('SF Pro', 24),
            bg=COLORS['card'], fg=COLORS['success'], cursor='hand2'
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
    
    def _build_saved_file_section(self):
        """Saved recording info"""
        self.file_frame = tk.LabelFrame(
            self.content, text=" Saved ", 
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
        
        # Open folder button
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
            self.content, text=" Upload ", 
            bg=COLORS['card'], fg=COLORS['text'],
            font=('SF Pro Text', 10, 'bold'),
            padx=10, pady=10
        )
        self.upload_frame.pack(fill='x', pady=(0, 10))
        self.upload_frame.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        self.drop_zone = tk.Frame(
            self.upload_frame, bg=COLORS['surface_light'], 
            height=70, highlightbackground=COLORS['primary'],
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
        """Collapsible settings accordion"""
        self.settings_frame = tk.LabelFrame(
            self.content, text=" Settings ", 
            bg=COLORS['card'], fg=COLORS['text'],
            font=('SF Pro Text', 10, 'bold'),
            padx=10, pady=10
        )
        self.settings_frame.pack(fill='x')
        self.settings_frame.configure(highlightbackground=COLORS['card_border'], highlightthickness=1)
        
        # Mode selector
        tk.Label(
            self.settings_frame, text="Mode:", 
            font=('SF Pro Text', 9),
            bg=COLORS['card'], fg=COLORS['text_secondary']
        ).pack(anchor='w')
        
        self.mode_var = tk.StringVar(value="live")
        mode_frame = tk.Frame(self.settings_frame, bg=COLORS['card'])
        mode_frame.pack(fill='x', pady=(2, 8))
        
        tk.Radiobutton(
            mode_frame, text="🎓 Live", variable=self.mode_var, 
            value="live", command=self.app.on_mode_changed,
            bg=COLORS['card'], fg=COLORS['text'], selectcolor=COLORS['primary'],
            font=('SF Pro Text', 9)
        ).pack(side='left')
        
        tk.Radiobutton(
            mode_frame, text="⚡ Fast", variable=self.mode_var,
            value="batch", command=self.app.on_mode_changed,
            bg=COLORS['card'], fg=COLORS['text'], selectcolor=COLORS['primary'],
            font=('SF Pro Text', 9)
        ).pack(side='left', padx=(10, 0))
        
        # Model
        tk.Label(
            self.settings_frame, text="Model:", 
            font=('SF Pro Text', 9),
            bg=COLORS['card'], fg=COLORS['text_secondary']
        ).pack(anchor='w', pady=(5, 0))
        
        # Model with custom styling for visibility
        model_frame = tk.Frame(self.settings_frame, bg=COLORS['card'])
        model_frame.pack(fill='x', pady=(2, 8))
        
        self.model_combo = ttk.Combobox(
            model_frame,
            values=["0.6B (Fast)", "1.7B (Accurate)"],
            state='readonly', width=18, font=('SF Pro Text', 9)
        )
        self.model_combo.set("1.7B (Accurate)")
        self.model_combo.pack(fill='x')
        # Initialize model value
        self._model_value = "Qwen/Qwen3-ASR-1.7B"
        # Bind to convert display value to actual model name
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_change)
        
        # Language
        tk.Label(
            self.settings_frame, text="Language:", 
            font=('SF Pro Text', 9),
            bg=COLORS['card'], fg=COLORS['text_secondary']
        ).pack(anchor='w')
        
        # Language with custom styling
        lang_frame = tk.Frame(self.settings_frame, bg=COLORS['card'])
        lang_frame.pack(fill='x', pady=(2, 8))
        
        self.lang_combo = ttk.Combobox(
            lang_frame,
            values=["Auto", "English", "Chinese", "Japanese", 
                   "Korean", "Spanish", "French", "German"],
            state='readonly', width=18, font=('SF Pro Text', 9)
        )
        self.lang_combo.set("English")
        self.lang_combo.pack(fill='x')
        
        # Silence duration
        tk.Label(
            self.settings_frame, text="Auto-stop silence:", 
            font=('SF Pro Text', 9),
            bg=COLORS['card'], fg=COLORS['text_secondary']
        ).pack(anchor='w')
        
        self.silence_var = tk.DoubleVar(value=30.0)
        slider_frame = tk.Frame(self.settings_frame, bg=COLORS['card'])
        slider_frame.pack(fill='x')
        
        self.silence_slider = ttk.Scale(
            slider_frame, from_=0.5, to=60.0,
            orient='horizontal', variable=self.silence_var,
            command=self.app.on_silence_changed
        )
        self.silence_slider.pack(side='left', fill='x', expand=True)
        
        self.silence_label = tk.Label(
            slider_frame, text="30s", font=('SF Pro Mono', 8),
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
    
    def _on_model_change(self, event=None):
        """Handle model selection change"""
        display_val = self.model_combo.get()
        # Store actual model value
        if "0.6B" in display_val:
            self._model_value = "Qwen/Qwen3-ASR-0.6B"
        else:
            self._model_value = "Qwen/Qwen3-ASR-1.7B"
    
    def get_model(self):
        """Get the actual model name"""
        return getattr(self, '_model_value', "Qwen/Qwen3-ASR-1.7B")
    
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
        
        # Create overlay frame for closing
        self.overlay = tk.Frame(parent, bg='black')
        self.overlay.bind('<Button-1>', self.close)
        
        # Panel content
        self.content = CollapsibleSidebar(self, app)
        self.content.pack(fill='both', expand=True)
        
        # Place off-screen initially
        self.place(relx=1.0, rely=0, relheight=1.0, width=0)
    
    def open(self):
        """Slide in from right"""
        self.is_open = True
        self.overlay.place(relx=0, rely=0, relwidth=1.0, relheight=1.0)
        self.overlay.config(bg='black')
        
        # Animate slide in
        for width in range(0, self.panel_width + 1, 20):
            self.place(relx=1.0, rely=0, relheight=1.0, width=width, anchor='ne')
            self.update()
            time.sleep(0.01)
    
    def close(self, event=None):
        """Slide out to right"""
        self.is_open = False
        self.overlay.place_forget()
        
        # Animate slide out
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
        
        # Record button (center, large)
        self.record_btn = tk.Button(
            self, text="🎙️", font=('SF Pro', 28),
            bg=COLORS['success'], fg=COLORS['text'],
            relief='flat', bd=0, cursor='hand2',
            command=app.toggle_recording
        )
        self.record_btn.pack(side='left', padx=20)
        
        # Timer
        self.timer_label = tk.Label(
            self, text="00:00", font=('SF Mono', 18, 'bold'),
            bg=COLORS['surface'], fg=COLORS['primary']
        )
        self.timer_label.pack(side='left', padx=10)
        
        # Spacer
        tk.Frame(self, bg=COLORS['surface']).pack(side='left', expand=True)
        
        # Settings button
        self.settings_btn = tk.Button(
            self, text="⚙️", font=('SF Pro', 20),
            bg=COLORS['surface'], fg=COLORS['text_secondary'],
            relief='flat', bd=0, cursor='hand2',
            command=app.toggle_settings_panel
        )
        self.settings_btn.pack(side='right', padx=15)
        
        # Files button
        self.files_btn = tk.Button(
            self, text="📁", font=('SF Pro', 20),
            bg=COLORS['surface'], fg=COLORS['text_secondary'],
            relief='flat', bd=0, cursor='hand2',
            command=app.open_recordings_folder
        )
        self.files_btn.pack(side='right', padx=5)


class LiveStreamer:
    """Live streaming transcription using C implementation - chunked processing"""
    
    def __init__(self, model_dir="assets/c-asr/qwen3-asr-0.6b", 
                 binary_path="assets/c-asr/qwen_asr",
                 sample_rate=16000):
        # Convert relative paths to absolute paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root
        self.model_dir = os.path.join(base_dir, model_dir)
        self.binary_path = os.path.join(base_dir, binary_path)
        self.sample_rate = sample_rate
        self.chunk_duration = 5.0  # Process 5-second chunks for better accuracy
        self.chunk_samples = int(self.chunk_duration * sample_rate)
        self.raw_frames = []
        self.is_running = False
        self.transcript_buffer = ""
        self.current_audio_file = None
        self.output_callback = None
        self.status_callback = None
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self._pending_chunks = 0
        # Use a thread pool with single worker to serialize chunk processing
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
    def start(self, output_callback=None, status_callback=None):
        """Start live streaming transcription"""
        self.is_running = True
        self.raw_frames = []
        self.audio_buffer = []
        self.transcript_buffer = ""
        self.output_callback = output_callback
        self.status_callback = status_callback
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        self.current_audio_file = os.path.join(RECORDINGS_DIR, f"class_{timestamp}.wav")
        
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
            if total_samples >= self.chunk_samples and self._pending_chunks < 1:
                # Extract chunk
                combined = np.concatenate(self.audio_buffer)
                to_process = combined[:self.chunk_samples].copy()
                remaining = combined[self.chunk_samples:]
                
                # Clear buffer and keep remaining
                self.audio_buffer = [remaining] if len(remaining) > 0 else []
                self._pending_chunks += 1
                
                # Submit to thread pool (serializes execution)
                self._executor.submit(self._process_chunk, to_process)
    
    def _process_chunk(self, audio: np.ndarray):
        """Process a single audio chunk with C binary - runs in thread pool"""
        temp_file = None
        try:
            if not self.is_running:
                return
                
            if self.status_callback:
                self.status_callback("Processing chunk...")
            
            # Convert to int16
            audio_int16 = np.clip(audio * 32767, -32768, 32768).astype(np.int16)
            
            # Write to temporary WAV file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_file = f.name
            
            # Write proper WAV file
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            # Run C binary with file input - use subprocess.run for simplicity
            cmd = [
                self.binary_path,
                "-d", self.model_dir,
                "-i", temp_file,
                "--language", "English"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            # Parse output
            stdout_text = result.stdout.decode('utf-8', errors='replace')
            stderr_text = result.stderr.decode('utf-8', errors='replace')
            
            # Extract transcription (first non-empty line of stdout)
            transcript = ""
            for line in stdout_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('Inference:') and not line.startswith('Audio:'):
                    transcript = line
                    break
            
            if transcript:
                self.transcript_buffer += transcript + " "
                if self.output_callback:
                    self.output_callback(transcript + " ", is_partial=True)
            
            # Show timing info
            for line in stderr_text.split('\n'):
                if "Inference:" in line or "Audio:" in line:
                    if self.status_callback:
                        self.status_callback(line.strip())
                        
        except Exception as e:
            print(f"LiveStreamer: Error processing chunk: {e}")
        finally:
            with self.buffer_lock:
                self._pending_chunks = max(0, self._pending_chunks - 1)
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    def stop(self) -> tuple:
        """Stop streaming and return results"""
        self.is_running = False
        
        # Wait for pending chunks to complete (with timeout)
        for _ in range(60):  # Wait up to 6 seconds
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
                if len(remaining) > self.sample_rate * 0.5:  # At least 0.5 second
                    # Process synchronously
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
        
        return self.current_audio_file, self.transcript_buffer
    
    def _process_chunk_sync(self, audio: np.ndarray):
        """Process chunk synchronously (for remaining audio at stop)"""
        temp_file = None
        try:
            audio_int16 = np.clip(audio * 32767, -32768, 32768).astype(np.int16)
            
            import tempfile
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
                "--language", "English"
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            stdout_text = result.stdout.decode('utf-8', errors='replace')
            for line in stdout_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('Inference:') and not line.startswith('Audio:'):
                    self.transcript_buffer += line + " "
                    if self.output_callback:
                        self.output_callback(line + " ", is_partial=True)
                    break
                    
        except Exception as e:
            print(f"LiveStreamer: Error in sync processing: {e}")
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass


class AudioRecorder:
    """Audio recorder with configurable VAD"""
    
    def __init__(self, silence_threshold: float = 0.015,
                 silence_duration: float = 5.0,
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
        audio_int16 = np.clip(audio * 32767, -32768, 32768).astype(np.int16)
        temp_file = tempfile.mktemp(suffix='.wav')
        
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        
        return temp_file
    
    def get_raw_audio(self) -> Optional[np.ndarray]:
        if not self.frames:
            return None
        return np.concatenate(self.frames)


class TranscriptionEngine:
    """Transcription engine using mlx-qwen3-asr Python API"""
    
    def __init__(self):
        self.backend = None
        self.model = None
        self.model_name = None
        self.detect_backend()
    
    def detect_backend(self):
        try:
            import mlx_audio.stt as mlx_stt
            self.backend = 'mlx_audio'
            print("✅ Using mlx-audio backend")
            return
        except ImportError:
            pass
        
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
        
        try:
            import qwen_asr
            self.backend = 'pytorch'
            print("✅ Using PyTorch (qwen-asr) backend")
            return
        except ImportError:
            pass
        
        self.backend = None
        raise RuntimeError("No transcription backend available. Please run SETUP.command")
    
    def load_model(self, model_name: str = "Qwen/Qwen3-ASR-1.7B", dtype: str = "float16"):
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
    
    def transcribe(self, audio_path: str, model: str = "Qwen/Qwen3-ASR-1.7B",
                   dtype: str = "float16", language: Optional[str] = None,
                   progress_callback: Optional[callable] = None) -> tuple:
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
        if self.model is None or self.model_name != model:
            import mlx_audio.stt as mlx_stt
            self.model = mlx_stt.load(model)
            self.model_name = model
        
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
        import torch
        from qwen_asr import Qwen3ASRModel
        
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
    """Main application with responsive adaptive UI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} v{VERSION}")
        self.root.geometry("1100x800")
        self.root.minsize(450, 550)
        self.root.configure(bg=COLORS['bg'])
        
        # Configure ttk styles for light theme
        self.style = configure_ttk_styles()
        
        # Track window size
        self.current_width = 1100
        self.layout_mode = "desktop"  # desktop, compact, mobile
        
        # Recording tracking
        self.recording_start_time = None
        self.elapsed_seconds = 0
        
        # Base directory for assets
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Initialize components
        try:
            self.engine = TranscriptionEngine()
        except RuntimeError as e:
            messagebox.showerror("Error", str(e))
            root.destroy()
            return
        
        self.recorder = AudioRecorder(level_callback=self.on_audio_level)
        self.live_streamer = LiveStreamer()
        self.is_recording = False
        self.is_live_mode = True
        self.status_queue = queue.Queue()
        self.current_raw_file = None
        self.live_transcript = ""
        
        # Bind resize event
        self.root.bind('<Configure>', self.on_window_resize)
        
        self.setup_ui()
        self.check_status()
    
    def on_window_resize(self, event):
        """Handle window resize for responsive layout"""
        if event.widget == self.root:
            new_width = event.width
            if abs(new_width - self.current_width) > 50:  # Threshold to avoid excessive updates
                self.current_width = new_width
                self.adapt_layout(new_width)
    
    def adapt_layout(self, width):
        """Adapt UI layout based on window width"""
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
        """Show full desktop layout with sidebar"""
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
        """Show mobile layout with bottom bar and slide-out panel"""
        self.sidebar.pack_forget()
        self.main_content.pack(fill='both', expand=True)
        self.bottom_bar.pack(side='bottom', fill='x')
    
    def setup_ui(self):
        """Setup main UI"""
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
            title_frame, text="Real-time class recording with auto-save",
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
        
        # Text area with custom scrollbar
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
        
        # Initial message
        self.text_area.insert('1.0', "🎓 Ready to record\n\n", "title")
        self.text_area.insert(tk.END, "Click 'Start Recording' to begin live transcription.\n\n")
        self.text_area.insert(tk.END, "Your recordings will be automatically saved to:\n")
        self.text_area.insert(tk.END, f"{RECORDINGS_DIR}", "meta")
        
        return content
    
    def toggle_settings_panel(self):
        """Toggle settings slide-out panel (mobile)"""
        if self.slide_panel.is_open:
            self.slide_panel.close()
        else:
            self.slide_panel.open()
    
    def on_mode_changed(self):
        """Handle recording mode change"""
        mode = self.sidebar.mode_var.get()
        self.is_live_mode = (mode == "live")
        
        if self.is_live_mode:
            self.sidebar.record_text.config(text="Start Recording")
        else:
            self.sidebar.record_text.config(text="Start (Fast Mode)")
    
    def on_silence_changed(self, value):
        """Handle silence duration change"""
        duration = float(value)
        self.sidebar.silence_label.config(text=f"{duration:.0f}s")
        self.recorder.set_params(duration=duration)
    
    def set_silence_preset(self, seconds):
        """Set silence preset"""
        self.sidebar.silence_var.set(seconds)
        self.sidebar.silence_label.config(text=f"{seconds:.0f}s")
        self.recorder.set_params(duration=seconds)
    
    def open_recordings_folder(self):
        """Open recordings folder in Finder"""
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        os.system(f'open "{RECORDINGS_DIR}"')
    
    def toggle_recording(self):
        """Toggle recording state"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording"""
        self.is_recording = True
        self.live_transcript = ""
        self.recording_start_time = time.time()
        self.elapsed_seconds = 0
        self.text_area.delete('1.0', tk.END)
        
        # Update UI
        self.live_indicator.config(fg=COLORS['error'])
        self.sidebar.record_icon.config(fg=COLORS['error'])
        self.sidebar.record_text.config(text="Stop Recording")
        self.sidebar.record_status.config(text="🔴 Recording...", fg=COLORS['error'])
        
        if self.is_live_mode:
            self.start_live_recording()
        else:
            self.start_batch_recording()
    
    def start_live_recording(self):
        """Start live streaming recording"""
        model = self.sidebar.get_model()
        model_name = "qwen3-asr-0.6b" if "0.6B" in model else "qwen3-asr-1.7b"
        self.live_streamer.model_dir = os.path.join(self.base_dir, "assets", "c-asr", model_name)
        
        raw_file = self.live_streamer.start(
            output_callback=self.on_live_transcript,
            status_callback=self.on_live_status
        )
        self.current_raw_file = raw_file
        
        # Start audio capture
        self.recorder.set_params(duration=self.sidebar.silence_var.get())
        self.start_audio_capture()
        self.update_timer()
    
    def start_batch_recording(self):
        """Start batch recording"""
        self.recorder.set_params(duration=self.sidebar.silence_var.get())
        self.recorder.start()
        self.update_timer()
    
    def start_audio_capture(self):
        """Capture audio for live streaming"""
        import sounddevice as sd
        
        # Buffer for accumulating audio into chunks
        audio_buffer = []
        
        def audio_callback(indata, frame_count, time_info, status):
            if not self.is_recording:
                return
            audio = indata.copy().flatten()
            audio_buffer.append(audio)
            
            # Calculate level and call level_callback for waveform
            level = self.recorder._calculate_level(audio)
            is_speech = level > self.recorder.silence_threshold
            if self.recorder.level_callback:
                self.recorder.level_callback(level, is_speech)
            
            # Feed to streamer
            combined = np.concatenate(audio_buffer)
            self.live_streamer.feed_audio(combined)
            audio_buffer.clear()
        
        self.audio_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.float32,
            blocksize=int(SAMPLE_RATE * 0.1),  # 100ms chunks
            callback=audio_callback
        )
        self.audio_stream.start()
    
    def on_live_transcript(self, text, is_partial=True):
        """Handle live transcript update"""
        self.live_transcript += text
        
        # Use after() to update UI from main thread
        self.root.after(0, self._update_transcript_ui)
    
    def _update_transcript_ui(self):
        """Update transcript UI (called from main thread)"""
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', "🎓 Live Class Transcription\n", "title")
        self.text_area.insert(tk.END, "─" * 50 + "\n\n", "meta")
        self.text_area.insert(tk.END, self.live_transcript)
        self.text_area.see(tk.END)
    
    def on_live_status(self, status):
        """Handle live streamer status"""
        if "Loading" in status:
            self.root.after(0, lambda: self.text_area.insert(tk.END, "\n⏳ Loading model...\n", "meta"))
    
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
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        self.recording_start_time = None
        
        # Update UI
        self.live_indicator.config(fg=COLORS['text_muted'])
        self.sidebar.record_icon.config(fg=COLORS['success'])
        self.sidebar.record_text.config(text="Start Recording")
        self.sidebar.record_status.config(text="Processing...", fg=COLORS['warning'])
        
        if self.is_live_mode:
            self.stop_live_recording()
        else:
            self.stop_batch_recording()
    
    def stop_live_recording(self):
        """Stop live recording"""
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        
        raw_file, transcript = self.live_streamer.stop()
        
        # Update file info
        if raw_file and os.path.exists(raw_file):
            file_size = os.path.getsize(raw_file) / (1024 * 1024)
            self.sidebar.file_label.config(
                text=f"📁 {os.path.basename(raw_file)}\n💾 {file_size:.1f} MB",
                fg=COLORS['success']
            )
        
        # Final transcript display
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', "✅ Recording Complete\n", "title")
        self.text_area.insert(tk.END, f"📁 Saved: {os.path.basename(raw_file)}\n", "meta")
        self.text_area.insert(tk.END, "─" * 50 + "\n\n", "meta")
        self.text_area.insert(tk.END, transcript)
        
        self.sidebar.record_status.config(text="✅ Saved", fg=COLORS['success'])
        self.sidebar.waveform.reset()
    
    def stop_batch_recording(self):
        """Stop batch recording"""
        audio_file = self.recorder.stop()
        self.sidebar.waveform.reset()
        
        if audio_file:
            # Save to recordings folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(RECORDINGS_DIR, exist_ok=True)
            saved_file = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")
            import shutil
            shutil.copy(audio_file, saved_file)
            self.current_raw_file = saved_file
            
            self.process_file(audio_file, is_temp=True)
    
    def on_audio_level(self, level, speaking):
        """Handle audio level update"""
        self.sidebar.waveform.update_level(level, speaking)
        
        if speaking:
            self.sidebar.record_status.config(text="🟢 Speaking", fg=COLORS['success'])
        else:
            silence = self.recorder.silence_counter * CHUNK_DURATION
            self.sidebar.record_status.config(text=f"⚪ Silence: {silence:.0f}s", fg=COLORS['text_muted'])
    
    def browse_file(self):
        """Browse for audio file"""
        path = filedialog.askopenfilename(
            title="Select Audio",
            filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg"), ("All", "*.*")]
        )
        if path:
            self.process_file(path)
    
    def process_file(self, file_path: str, is_temp: bool = False):
        """Process audio file"""
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "File not found")
            return
        
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', f"Processing {os.path.basename(file_path)}...")
        
        def worker():
            try:
                model = self.sidebar.get_model()
                language = self.sidebar.lang_combo.get()
                if language == "Auto":
                    language = None
                
                result, stats = self.engine.transcribe(
                    file_path, model=model, language=language
                )
                
                self.status_queue.put(('success', result))
                self.status_queue.put(('stats', stats))
            except Exception as e:
                self.status_queue.put(('error', str(e)))
            finally:
                if is_temp and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
        
        threading.Thread(target=worker, daemon=True).start()
    
    def check_status(self):
        """Check status queue"""
        try:
            while True:
                msg_type, data = self.status_queue.get_nowait()
                
                if msg_type == 'success':
                    self.show_result(data)
                elif msg_type == 'stats':
                    self.update_stats(data)
                elif msg_type == 'error':
                    self.show_error(data)
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_status)
    
    def show_result(self, result):
        """Show transcription result"""
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', "Transcription Result\n", "title")
        self.text_area.insert(tk.END, f"Backend: {result['backend']} | Model: {result['model']}\n", "meta")
        self.text_area.insert(tk.END, "─" * 50 + "\n\n", "meta")
        self.text_area.insert(tk.END, result['text'])
        self.sidebar.record_status.config(text="Ready", fg=COLORS['text_muted'])
    
    def update_stats(self, stats):
        """Update stats display"""
        rtf_str = f"{stats.rtf:.2f}x"
        self.stats_label.config(text=f"RTF: {rtf_str}")
    
    def show_error(self, error_msg):
        """Show error message"""
        self.text_area.delete('1.0', tk.END)
        self.text_area.insert('1.0', f"Error:\n{error_msg}")
        self.sidebar.record_status.config(text="Error", fg=COLORS['error'])
        messagebox.showerror("Transcription Error", error_msg)
    
    def clear(self):
        """Clear text area"""
        self.text_area.delete('1.0', tk.END)
        self.stats_label.config(text="")
    
    def copy(self):
        """Copy transcript to clipboard"""
        text = self.text_area.get('1.0', tk.END).strip()
        # Clean up
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            if not any(line.startswith(p) for p in ['Backend:', '─', '🎓', '✅', '⏳', '📁']):
                clean_lines.append(line)
        clean_text = '\n'.join(clean_lines).strip()
        
        self.root.clipboard_clear()
        self.root.clipboard_append(clean_text)
        self.stats_label.config(text="Copied!", fg=COLORS['success'])
    
    def save(self):
        """Save transcript to file"""
        text = self.text_area.get('1.0', tk.END).strip()
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            if not any(line.startswith(p) for p in ['Backend:', '─', '🎓', '✅', '⏳', '📁']):
                clean_lines.append(line)
        clean_text = '\n'.join(clean_lines).strip()
        
        if clean_text:
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text", "*.txt"), ("All", "*.*")]
            )
            if path:
                with open(path, 'w') as f:
                    f.write(clean_text)
                self.stats_label.config(text="Saved!", fg=COLORS['success'])


def main():
    root = tk.Tk()
    app = QwenASRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
# LiveStreamer class replaced with chunked processing
# See end of file for new implementation
