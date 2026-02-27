#!/usr/bin/env python3
"""
Qwen3-ASR Pro - Entry Point
macOS Speech-to-Text Application
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import main

if __name__ == "__main__":
    main()
