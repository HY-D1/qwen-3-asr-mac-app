#!/usr/bin/env python3
"""
Ollama Backend for Local LLM
Uses Ollama to run Qwen and other open source models
"""

import subprocess
import json
import os
import sys
from typing import Optional

class OllamaBackend:
    """Backend using Ollama local LLM server"""
    
    AVAILABLE_MODELS = {
        "qwen:7b": "Qwen 7B (Best quality, ~4GB)",
        "qwen:4b": "Qwen 4B (Good quality, ~2.5GB)",
        "qwen:1.8b": "Qwen 1.8B (Fast, ~1GB)",
        "llama3.2:3b": "Llama 3.2 3B (Meta, ~2GB)",
        "phi3:3.8b": "Phi-3 3.8B (Microsoft, ~2GB)",
        "gemma:2b": "Gemma 2B (Google, ~1.5GB)",
    }
    
    def __init__(self, model: str = "qwen:1.8b"):
        self.model = model
        self.available = False
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is installed and running"""
        try:
            # Check if ollama command exists
            result = subprocess.run(
                ["which", "ollama"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print("❌ Ollama not found. Install from https://ollama.com")
                return
            
            # Check if ollama server is running
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print(f"✅ Ollama server running")
                self.available = True
                
                # Check if model is available
                if self.model not in result.stdout:
                    print(f"📥 Model {self.model} not found. Pulling...")
                    self._pull_model()
            else:
                print("⚠️  Ollama server not running. Trying to start...")
                self._start_server()
                
        except Exception as e:
            print(f"❌ Ollama check failed: {e}")
    
    def _start_server(self):
        """Start Ollama server in background"""
        try:
            # Start ollama serve in background
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            
            # Wait a bit for server to start
            import time
            time.sleep(2)
            
            # Check again
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print("✅ Ollama server started")
                self.available = True
                
                # Pull model if needed
                if self.model not in result.stdout:
                    self._pull_model()
            else:
                print("❌ Failed to start Ollama server")
                
        except Exception as e:
            print(f"❌ Could not start Ollama: {e}")
    
    def _pull_model(self):
        """Pull the model from Ollama"""
        try:
            print(f"📥 Downloading {self.model} (this may take a few minutes)...")
            result = subprocess.run(
                ["ollama", "pull", self.model],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"✅ Model {self.model} ready")
            else:
                print(f"❌ Failed to pull model: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Model download timed out")
        except Exception as e:
            print(f"❌ Error pulling model: {e}")
    
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate text using Ollama"""
        if not self.available:
            return "Error: Ollama not available"
        
        try:
            # Build request
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if system:
                data["system"] = system
            
            # Call Ollama API
            result = subprocess.run(
                ["curl", "-s", "http://localhost:11434/api/generate"],
                input=json.dumps(data),
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                return f"Error: {result.stderr}"
            
            # Parse response
            response = json.loads(result.stdout)
            return response.get("response", "")
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def process_text(self, text: str, mode: str = "punctuate") -> str:
        """Process text with appropriate prompt"""
        
        prompts = {
            "punctuate": {
                "system": "You are a text editor. Add proper punctuation and capitalization. Fix obvious errors. Return ONLY the improved text.",
                "prompt": f"Add punctuation and capitalization to this text:\n\n{text}\n\nImproved text:"
            },
            "summarize": {
                "system": "You are a summarizer. Create a concise summary.",
                "prompt": f"Summarize this text:\n\n{text}\n\nSummary:"
            },
            "clean": {
                "system": "You are an editor. Remove filler words (um, uh, like, you know) and fix grammar.",
                "prompt": f"Clean up this text:\n\n{text}\n\nCleaned text:"
            },
            "key_points": {
                "system": "You extract key points from text.",
                "prompt": f"Extract key points from this text:\n\n{text}\n\nKey points:"
            }
        }
        
        config = prompts.get(mode, prompts["punctuate"])
        return self.generate(config["prompt"], config["system"])


def test_ollama():
    """Test Ollama backend"""
    print("Testing Ollama backend...")
    print("="*60)
    
    backend = OllamaBackend("qwen:1.8b")
    
    if not backend.available:
        print("\n❌ Ollama not available")
        print("\nTo install Ollama:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        return
    
    print("\n✅ Ollama is ready!")
    print(f"   Model: {backend.model}")
    
    # Test generation
    test_text = "um so like we need to discuss the project timeline uh first we have design"
    print(f"\nTest text: {test_text}")
    print("\nProcessing...")
    
    result = backend.process_text(test_text, "punctuate")
    print(f"\nResult: {result}")


if __name__ == "__main__":
    test_ollama()
