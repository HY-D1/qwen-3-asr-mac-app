#!/usr/bin/env python3
"""
Constants and configuration for Qwen3-ASR Pro
"""

import os

# App Info
APP_NAME = "Qwen3-ASR Pro"
VERSION = "3.1.1"

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
