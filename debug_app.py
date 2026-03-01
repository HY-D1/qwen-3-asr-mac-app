#!/usr/bin/env python3
"""Debug app initialization step by step"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("="*60)
print("STEP 1: Testing imports from src/ directory")
print("="*60)

try:
    import constants
    print("✅ constants imported")
    print(f"   Version: {constants.VERSION}")
except Exception as e:
    print(f"❌ constants failed: {e}")
    sys.exit(1)

try:
    from simple_llm import SimpleLLM
    print("✅ simple_llm imported")
except Exception as e:
    print(f"❌ simple_llm failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*60)
print("STEP 2: Testing tkinter window creation")
print("="*60)

try:
    import tkinter as tk
    root = tk.Tk()
    print("✅ tk.Tk() created")
    root.destroy()
    print("✅ tk.Tk() destroyed")
except Exception as e:
    print(f"❌ tkinter failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*60)
print("STEP 3: Testing SimpleLLM initialization")
print("="*60)

try:
    llm = SimpleLLM()
    print(f"✅ SimpleLLM initialized")
    print(f"   Backend: {llm.backend_name}")
    print(f"   Available: {llm.is_available()}")
except Exception as e:
    print(f"❌ SimpleLLM failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*60)
print("✅ ALL TESTS PASSED - App should work!")
print("="*60)
print()
print("Run the actual app with:")
print("  python src/main.py")
