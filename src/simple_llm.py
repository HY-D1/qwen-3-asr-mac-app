#!/usr/bin/env python3
"""
Simple LLM Text Processor - Multiple Backend Options
Just works™ - picks the first available backend
"""

import os
import sys
import json
import traceback
from typing import Optional, Dict, Any, List

# =============================================================================
# Backend 0: Ollama (Local Open Source LLM - Best Free Option)
# =============================================================================

class OllamaBackend:
    """Use Ollama to run local open source models like Qwen, Llama, Phi"""
    
    def __init__(self, model: str = "qwen:1.8b"):
        self.model = model
        self.available = False
        self._init()
    
    def _init(self):
        try:
            import subprocess
            import json
            
            # Check if ollama is installed
            result = subprocess.run(
                ["which", "ollama"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return
            
            # Check if server is running
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                self.available = True
                print(f"✅ Using Ollama backend ({self.model})")
                
                # Pull model if not present
                if self.model not in result.stdout:
                    print(f"📥 Pulling {self.model}...")
                    subprocess.run(
                        ["ollama", "pull", self.model],
                        capture_output=True,
                        timeout=300
                    )
            else:
                # Try to start server
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
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
                    self.available = True
                    print(f"✅ Ollama backend started ({self.model})")
                    
        except Exception as e:
            print(f"Ollama not available: {e}")
    
    def process(self, text: str, mode: str = "punctuate") -> str:
        if not self.available:
            return text
        
        import subprocess
        import json
        
        system_prompts = {
            "punctuate": "Add proper punctuation and capitalization. Fix obvious errors. Return ONLY the improved text.",
            "summarize": "Create a concise summary. Return ONLY the summary.",
            "clean": "Remove filler words (um, uh, like, you know) and fix grammar. Return ONLY the cleaned text.",
            "key_points": "Extract key points as bullet points. Return ONLY the bullet points."
        }
        
        user_prompts = {
            "punctuate": f"Add punctuation to:\n\n{text}",
            "summarize": f"Summarize:\n\n{text}",
            "clean": f"Clean up:\n\n{text}",
            "key_points": f"Key points from:\n\n{text}"
        }
        
        data = {
            "model": self.model,
            "system": system_prompts.get(mode, system_prompts["punctuate"]),
            "prompt": user_prompts.get(mode, user_prompts["punctuate"]),
            "stream": False
        }
        
        try:
            result = subprocess.run(
                [
                    "curl", "-s", "-X", "POST",
                    "http://localhost:11434/api/generate",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(data)
                ],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                return response.get("response", text)
        except:
            pass
        
        return text


# =============================================================================
# Backend 1: OpenAI API (Easiest - just need API key)
# =============================================================================

class OpenAIBackend:
    """Use OpenAI API - most reliable, requires internet"""
    
    def __init__(self):
        self.client = None
        self.available = False
        self._init()
    
    def _init(self):
        try:
            from openai import OpenAI
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
                self.available = True
                print("✅ Using OpenAI API backend")
        except:
            pass
    
    def process(self, text: str, mode: str = "punctuate") -> str:
        if not self.available:
            return text
        
        prompts = {
            "punctuate": "Add proper punctuation and capitalization to this text. Fix obvious errors. Return ONLY the improved text:\n\n",
            "summarize": "Summarize this text concisely. Return ONLY the summary:\n\n",
            "key_points": "Extract key points as bullet points. Return ONLY the bullet points:\n\n",
            "format": "Format this as meeting notes with sections. Return ONLY the formatted notes:\n\n",
            "clean": "Remove filler words (um, uh, like) and fix grammar. Return ONLY the cleaned text:\n\n"
        }
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a text processing assistant."},
                    {"role": "user", "content": prompts.get(mode, prompts["punctuate"]) + text}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error: {e}")
            return text


# =============================================================================
# Backend 2: Transformers Pipeline (Simple local model)
# =============================================================================

class TransformersBackend:
    """Use HuggingFace transformers - downloads model once
    
    Best models for Mac M1 8GB:
    - Qwen/Qwen2.5-0.5B-Instruct: ~1GB, very fast, fits easily
    - Qwen/Qwen2.5-1.5B-Instruct: ~3GB, better quality, still fits
    - mlx-community/Qwen2.5-0.5B-Instruct-4bit: Optimized for Apple Silicon
    """
    
    def __init__(self, model_size="0.5B"):
        self.pipeline = None
        self.available = False
        # Use models compatible with transformers 4.35.0
        # Qwen2.5 needs transformers 4.36+, so we use compatible alternatives
        # Note: These models will download on first use
        self.model_options = {
            "0.5B": "sshleifer/tiny-gpt2",             # Tiny model for testing
            "1.5B": "gpt2",                            # GPT-2 (not ideal but works)
            "3B": "gpt2-medium",                       # Larger GPT-2
        }
        self.model_name = self.model_options.get(model_size, self.model_options["0.5B"])
        self._init()
    
    def _init(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"📥 Loading {self.model_name} (first time may take 1-2 min)...")
            
            # Determine best device
            if torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
                dtype = torch.float16
                print("   Using Apple Metal (MPS) acceleration")
            elif torch.cuda.is_available():
                device = "cuda"
                dtype = torch.float16
                print("   Using CUDA GPU")
            else:
                device = "cpu"
                dtype = torch.float32
                print("   Using CPU")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # For smaller models, don't use device_map to avoid issues
            if "0.5B" in self.model_name:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True
                )
                self.model = self.model.to(device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto",
                    trust_remote_code=True
                )
            
            self.device = device
            self.available = True
            print(f"✅ Loaded {self.model_name}")
            
        except Exception as e:
            print(f"Transformers backend not available: {e}")
            import traceback
            traceback.print_exc()
    
    def process(self, text: str, mode: str = "punctuate") -> str:
        if not self.available:
            return text
        
        # Better prompts for instruction models
        prompts = {
            "punctuate": "Add proper punctuation and capitalization to the following text. Fix obvious errors. Return ONLY the improved text:\n\nText: ",
            "summarize": "Summarize the following text concisely. Return ONLY the summary:\n\nText: ",
            "key_points": "Extract the key points from the following text as bullet points. Return ONLY the bullet points:\n\nText: ",
            "format": "Format the following text as structured meeting notes. Return ONLY the formatted notes:\n\nText: ",
            "clean": "Remove filler words (um, uh, like, you know) and fix grammar. Return ONLY the cleaned text:\n\nText: "
        }
        
        try:
            import torch
            
            # Build prompt with instruction format
            base_prompt = prompts.get(mode, prompts["punctuate"])
            full_prompt = f"<|im_start|>user\n{base_prompt}{text}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            
            # Move to appropriate device
            if hasattr(self, 'device') and self.device != "auto":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            elif torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate with appropriate parameters for small models
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(len(text.split()) * 2, 500),  # Dynamic max tokens
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract assistant response
            if "<|im_start|>assistant" in result:
                result = result.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in result:
                result = result.split("<|im_end|>")[0]
            
            result = result.strip()
            
            # If empty or too short, return original
            if len(result) < len(text) * 0.5:
                return text
            
            return result
            
        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return text


# =============================================================================
# Backend 3: Rule-based (Always works, no dependencies)
# =============================================================================

class RuleBasedBackend:
    """Simple rule-based text improvement - always available"""
    
    def __init__(self):
        self.available = True
        print("✅ Using rule-based backend (basic improvements)")
    
    def process(self, text: str, mode: str = "punctuate") -> str:
        if mode == "punctuate":
            return self._punctuate(text)
        elif mode == "clean":
            return self._clean(text)
        elif mode == "summarize":
            return self._simple_summarize(text)
        else:
            return self._punctuate(text)
    
    def _punctuate(self, text: str) -> str:
        """Basic punctuation fixes"""
        # Capitalize first letter of sentences
        import re
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Capitalize after periods
        sentences = re.split(r'([.!?]\s+)', text)
        result = []
        for i, sent in enumerate(sentences):
            if i == 0 or (i > 0 and sentences[i-1] in ['. ', '! ', '? ']):
                sent = sent.capitalize()
            result.append(sent)
        
        text = ''.join(result)
        
        # Ensure ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        # Add common punctuation patterns
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        
        return text.strip()
    
    def _clean(self, text: str) -> str:
        """Remove filler words"""
        fillers = ['um', 'uh', 'like', 'you know', 'sort of', 'kind of']
        import re
        
        text = re.sub(r'\s+', ' ', text)
        
        for filler in fillers:
            # Remove as whole word
            pattern = r'\b' + re.escape(filler) + r'\b\s*'
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Fix double spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _simple_summarize(self, text: str) -> str:
        """Extract first sentence of each paragraph as summary"""
        import re
        paragraphs = text.split('\n\n')
        summary = []
        for para in paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', para.strip())
            if sentences:
                summary.append(sentences[0])
        return ' '.join(summary[:3]) if summary else text[:200] + "..."


# =============================================================================
# Main Interface
# =============================================================================

class SimpleLLM:
    """
    Simple LLM interface that tries multiple backends
    Priority: Ollama > OpenAI > Transformers > Rule-based
    """
    
    def __init__(self, model_size="0.5B", prefer_openai=False, ollama_model="qwen:1.8b"):
        """
        Args:
            model_size: "0.5B", "1.5B", or "3B" (for local models)
            prefer_openai: If True, try OpenAI first even if it requires env var
            ollama_model: Ollama model to use (qwen:1.8b, llama3.2:3b, etc.)
        """
        self.backend = None
        self.backend_name = "none"
        self.model_size = model_size
        self.ollama_model = ollama_model
        
        # Try backends in order
        # 1. Try Ollama first (local, free, open source)
        self._try_ollama()
        
        # 2. Try OpenAI if key is set
        if not self.backend and (prefer_openai or os.getenv('OPENAI_API_KEY')):
            self._try_openai()
        
        # 3. Try local transformers
        if not self.backend:
            self._try_transformers(model_size)
        
        # 4. Fall back to rule-based
        if not self.backend:
            self._use_rule_based()
    
    def _try_ollama(self):
        """Try Ollama local LLM"""
        backend = OllamaBackend(self.ollama_model)
        if backend.available:
            self.backend = backend
            self.backend_name = f"ollama-{self.ollama_model}"
    
    def _try_openai(self):
        """Try OpenAI API"""
        backend = OpenAIBackend()
        if backend.available:
            self.backend = backend
            self.backend_name = "openai"
    
    def _try_transformers(self, model_size):
        """Try local transformers model"""
        backend = TransformersBackend(model_size)
        if backend.available:
            self.backend = backend
            self.backend_name = "transformers"
    
    def _use_rule_based(self):
        """Fall back to rule-based"""
        self.backend = RuleBasedBackend()
        self.backend_name = "rule-based"
    
    def is_available(self) -> bool:
        return self.backend is not None and self.backend_name != "none"
    
    def process(self, text: str, mode: str = "punctuate") -> str:
        """
        Process text
        
        Modes:
        - punctuate: Add punctuation and capitalization
        - summarize: Create summary
        - key_points: Extract key points
        - format: Format as meeting notes
        - clean: Remove filler words
        """
        if not text or not text.strip():
            return text
        
        if not self.backend:
            return text
        
        return self.backend.process(text, mode)
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Simple analysis"""
        import re
        
        words = text.split()
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "char_count": len(text),
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
            "backend_used": self.backend_name
        }


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    print("Testing SimpleLLM...\n")
    
    llm = SimpleLLM()
    
    test_text = "um so like we need to discuss the project timeline uh first we have the design phase then development and finally testing and deployment"
    
    print(f"Backend: {llm.backend_name}")
    print(f"\nOriginal:\n{test_text}\n")
    
    print(f"Punctuated:\n{llm.process(test_text, 'punctuate')}\n")
    print(f"Cleaned:\n{llm.process(test_text, 'clean')}\n")
    print(f"Summary:\n{llm.process(test_text, 'summarize')}\n")
    
    print("Analysis:", llm.analyze(test_text))
