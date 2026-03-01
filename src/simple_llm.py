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
    """Use HuggingFace transformers - downloads model once"""
    
    def __init__(self):
        self.pipeline = None
        self.available = False
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Small, fast model
        self._init()
    
    def _init(self):
        try:
            from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
            import torch
            
            print(f"📥 Loading {self.model_name} (first time may take 1-2 min)...")
            
            # Use tiny model for fast inference
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu"
            )
            
            self.available = True
            print(f"✅ Loaded {self.model_name}")
            
        except Exception as e:
            print(f"Transformers backend not available: {e}")
    
    def process(self, text: str, mode: str = "punctuate") -> str:
        if not self.available:
            return text
        
        prompts = {
            "punctuate": "Add proper punctuation:\n\n",
            "summarize": "Summarize:\n\n",
            "key_points": "Key points:\n\n",
            "format": "Meeting notes:\n\n",
            "clean": "Clean up:\n\n"
        }
        
        try:
            import torch
            
            full_prompt = prompts.get(mode, prompts["punctuate"]) + text + "\n\nImproved:"
            
            inputs = self.tokenizer(full_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to("cuda")
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the response part
            if "Improved:" in result:
                result = result.split("Improved:")[-1].strip()
            
            return result
            
        except Exception as e:
            print(f"Generation error: {e}")
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
    Priority: OpenAI > Transformers > Rule-based
    """
    
    def __init__(self):
        self.backend = None
        self.backend_name = "none"
        
        # Try backends in order
        self._try_openai()
        if not self.backend:
            self._try_transformers()
        if not self.backend:
            self._use_rule_based()
    
    def _try_openai(self):
        """Try OpenAI API"""
        backend = OpenAIBackend()
        if backend.available:
            self.backend = backend
            self.backend_name = "openai"
    
    def _try_transformers(self):
        """Try local transformers model"""
        backend = TransformersBackend()
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
