#!/usr/bin/env python3
"""Debug which import causes macOS version error"""

import sys

def test_import(name, import_cmd):
    """Test a single import"""
    print(f"Testing {name}...", end=" ", flush=True)
    try:
        exec(import_cmd)
        print("✅ OK")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

# Test imports in order
imports = [
    ("numpy", "import numpy; print(f'  version: {numpy.__version__}')"),
    ("torch", "import torch; print(f'  version: {torch.__version__}')"),
    ("sounddevice", "import sounddevice; print('  sounddevice imported')"),
    ("transformers", "from transformers import AutoTokenizer; print('  transformers imported')"),
    ("tokenizers", "import tokenizers; print(f'  version: {tokenizers.__version__}')"),
    ("huggingface_hub", "import huggingface_hub; print(f'  version: {huggingface_hub.__version__}')"),
    ("urllib3", "import urllib3; print(f'  version: {urllib3.__version__}')"),
    ("requests", "import requests; print('  requests imported')"),
    ("wave", "import wave; print('  wave imported')"),
    ("tempfile", "import tempfile; print('  tempfile imported')"),
]

print("="*60)
print("DEBUGGING IMPORTS")
print("="*60)
print()

failed = []
for name, cmd in imports:
    if not test_import(name, cmd):
        failed.append(name)
    sys.stdout.flush()

print()
print("="*60)
if failed:
    print(f"❌ FAILED: {', '.join(failed)}")
else:
    print("✅ All basic imports work")
print("="*60)

# Now test tkinter
print()
print("Testing tkinter...", end=" ", flush=True)
try:
    import tkinter as tk
    print("✅ OK")
    print(f"  Tcl/Tk version: {tk.Tcl().eval('info patchlevel')}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    failed.append("tkinter")

# Test app imports
print()
print("="*60)
print("TESTING APP IMPORTS")
print("="*60)
print()

app_imports = [
    ("constants", "from constants import APP_NAME; print(f'  APP_NAME: {APP_NAME}')"),
    ("simple_llm", "from simple_llm import SimpleLLM; print('  SimpleLLM imported')"),
]

for name, cmd in app_imports:
    test_import(name, cmd)
    sys.stdout.flush()

print()
print("="*60)
print("DONE - If you see 'macOS 26 required' above,")
print("the LAST 'Testing...' line shows the culprit")
print("="*60)
