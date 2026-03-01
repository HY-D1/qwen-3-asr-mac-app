#!/usr/bin/env python3
"""
Constants and configuration for Qwen3-ASR Pro
"""

import os

# App Info
APP_NAME = "Qwen3-ASR Pro"
VERSION = "3.2.0"

# Audio Settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.05

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.expanduser("~/Documents/Qwen3-ASR-Recordings")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
C_ASR_DIR = os.path.join(ASSETS_DIR, "c-asr")
MODELS_DIR = os.path.join(ASSETS_DIR, "models")

# Responsive Breakpoints
MIN_WIDTH_COMPACT = 750
MIN_WIDTH_MOBILE = 550

# Model Configuration
# Live mode always uses 0.6B for stability and low latency
# Upload mode always uses 1.7B for best accuracy
MODEL_CONFIG = {
    "live": {
        "model_id": "Qwen/Qwen3-ASR-0.6B",
        "model_dir": "qwen3-asr-0.6b",  # C binary model directory
        "display_name": "0.6B (Live Optimized)",
        "description": "Fast, low latency for real-time streaming",
    },
    "upload": {
        "model_id": "Qwen/Qwen3-ASR-1.7B", 
        "display_name": "1.7B (Best Accuracy)",
        "description": "Maximum accuracy for file transcription",
    }
}

# Language Configuration with Auto-detection
# Qwen3-ASR supports 50+ languages, we provide common ones with auto-detect
LANGUAGE_CONFIG = {
    "Auto": {
        "code": "auto",
        "description": "Auto-detect language",
        " supports_auto": True,
    },
    "English": {
        "code": "en",
        "description": "English",
    },
    "Chinese (Simplified)": {
        "code": "zh",
        "description": "简体中文",
    },
    "Chinese (Traditional)": {
        "code": "zh-TW", 
        "description": "繁體中文",
    },
    "Japanese": {
        "code": "ja",
        "description": "日本語",
    },
    "Korean": {
        "code": "ko", 
        "description": "한국어",
    },
    "Spanish": {
        "code": "es",
        "description": "Español",
    },
    "French": {
        "code": "fr",
        "description": "Français",
    },
    "German": {
        "code": "de",
        "description": "Deutsch",
    },
    "Italian": {
        "code": "it",
        "description": "Italiano",
    },
    "Portuguese": {
        "code": "pt",
        "description": "Português",
    },
    "Russian": {
        "code": "ru",
        "description": "Русский",
    },
    "Arabic": {
        "code": "ar",
        "description": "العربية",
    },
    "Hindi": {
        "code": "hi",
        "description": "हिन्दी",
    },
    "Vietnamese": {
        "code": "vi",
        "description": "Tiếng Việt",
    },
    "Thai": {
        "code": "th",
        "description": "ไทย",
    },
    "Indonesian": {
        "code": "id",
        "description": "Bahasa Indonesia",
    },
}

# Default settings
DEFAULT_LANGUAGE = "Auto"
DEFAULT_SILENCE_DURATION = 30.0  # seconds

# Live Streaming Settings
LIVE_CHUNK_DURATION = 5.0  # Process 5-second chunks
LIVE_MAX_PENDING_CHUNKS = 1  # Max concurrent chunk processing
LIVE_MIN_REMAINING_SECONDS = 0.5  # Minimum audio to process on stop

# Light Theme Colors
COLORS = {
    'bg': "#f8fafc",
    'surface': "#ffffff",
    'surface_light': "#f1f5f9",
    'card': "#ffffff",
    'card_border': "#e2e8f0",
    'primary': "#4f46e5",
    'primary_hover': "#4338ca",
    'secondary': "#0891b2",
    'success': "#16a34a",
    'warning': "#d97706",
    'error': "#dc2626",
    'text': "#1e293b",
    'text_secondary': "#475569",
    'text_muted': "#94a3b8",
    'input_bg': "#ffffff",
    'input_fg': "#1e293b",
    'select_bg': "#4f46e5",
    'select_fg': "#ffffff",
}
