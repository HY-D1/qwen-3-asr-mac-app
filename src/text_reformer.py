#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         LLM Text Reformer - Intelligent Transcription Output     ║
║         MLX Optimized • 8GB RAM Compatible • Real-time           ║
╚══════════════════════════════════════════════════════════════════╝

Features:
- Smart text reformation (punctuation, paragraphs, formatting)
- Summarization and key point extraction
- Correlation analysis across multiple transcripts
- Topic modeling and entity extraction
- Sentiment analysis

Version: 1.0.0
"""

import os
import sys
import json
import time
import traceback
import threading
import queue
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

# Import constants


# =============================================================================
# Data Classes
# =============================================================================

class ReformMode(Enum):
    """Text reformation modes"""
    PUNCTUATE = "punctuate"           # Add proper punctuation
    PARAGRAPH = "paragraph"           # Structure into paragraphs
    SUMMARIZE = "summarize"           # Create summary
    KEY_POINTS = "key_points"         # Extract key points
    FORMAT = "format"                 # Format as meeting notes
    TRANSLATE = "translate"           # Translate to another language
    CLEAN = "clean"                   # Clean up filler words


@dataclass
class ReformResult:
    """Text reformation result"""
    original_text: str = ""
    reformed_text: str = ""
    mode: ReformMode = ReformMode.PUNCTUATE
    processing_time: float = 0.0
    tokens_used: int = 0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorrelationResult:
    """Correlation analysis result"""
    topics: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    keywords: List[str] = field(default_factory=list)
    summary: str = ""
    related_segments: List[Tuple[int, int, float]] = field(default_factory=list)  # (start, end, relevance)


# =============================================================================
# Prompts for LLM
# =============================================================================

REFORM_PROMPTS = {
    ReformMode.PUNCTUATE: """You are a text refinement assistant. Your task is to add proper punctuation and capitalization to the following transcribed text. Fix obvious transcription errors and make the text readable while preserving the original meaning. Do not add or remove content.

Transcription:
{text}

Return only the punctuated text without any explanations or markdown formatting.""",

    ReformMode.PARAGRAPH: """You are a text structuring assistant. Your task is to organize the following transcribed text into well-structured paragraphs. Group related sentences together, add proper paragraph breaks, and improve readability while preserving all original content.

Transcription:
{text}

Return only the structured text with proper paragraphs. No explanations or markdown.""",

    ReformMode.SUMMARIZE: """You are a summarization assistant. Create a concise summary of the following transcribed text that captures the main points and key information. The summary should be approximately 20-30% of the original length.

Transcription:
{text}

Return only the summary without any explanations or markdown formatting.""",

    ReformMode.KEY_POINTS: """You are an information extraction assistant. Extract the key points from the following transcribed text as a bulleted list. Focus on important facts, decisions, action items, and main ideas.

Transcription:
{text}

Return only the key points as a simple bulleted list (using • or -). No explanations or additional formatting.""",

    ReformMode.FORMAT: """You are a meeting notes formatter. Format the following transcribed text as structured meeting notes with these sections:
- Date/Time: (extract if mentioned, otherwise leave as [Not specified])
- Attendees: (extract names if mentioned)
- Key Discussion Points: (main topics discussed)
- Decisions Made: (any decisions or conclusions)
- Action Items: (tasks, deadlines, or follow-ups)

Transcription:
{text}

Return the formatted notes. Use clear section headers.""",

    ReformMode.TRANSLATE: """You are a translation assistant. Translate the following text to {target_language}. Maintain the original meaning, tone, and style as closely as possible.

Text:
{text}

Return only the translated text without any explanations or markdown.""",

    ReformMode.CLEAN: """You are a text cleanup assistant. Remove filler words (um, uh, like, you know, etc.), repeated phrases, and false starts from the following transcribed text. Make it flow naturally while preserving all meaningful content.

Transcription:
{text}

Return only the cleaned text without any explanations or markdown.""",
}

CORRELATION_PROMPT = """You are a text analysis assistant. Analyze the following transcript and provide a structured analysis in JSON format:

Transcription:
{text}

Provide analysis in this exact JSON format:
{
    "topics": ["list of main topics discussed"],
    "entities": [
        {"name": "entity name", "type": "person/organization/location/product", "mentions": count}
    ],
    "sentiment": "positive/negative/neutral",
    "sentiment_score": 0.0 to 1.0,
    "keywords": ["important keywords from the text"],
    "summary": "brief 2-3 sentence summary"
}

Return ONLY the JSON, no other text."""


# =============================================================================
# Text Reformer Engine
# =============================================================================

class TextReformer:
    """
    LLM-based text reformation engine.
    Uses MLX-LLM for efficient inference on Apple Silicon (8GB RAM compatible).
    Falls back to llama-cpp-python for Intel Macs.
    """
    
    # Model configuration for 8GB RAM systems
    MODEL_CONFIG = {
        "mlx": {
            "name": "mlx-community/Qwen2.5-3B-Instruct-4bit",
            "description": "3B parameter model optimized for MLX (4-bit quantized, ~1.8GB)",
            "max_tokens": 2048,
            "temperature": 0.3,
        },
        "llama_cpp": {
            "name": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            "description": "Quantized model for CPU inference",
            "max_tokens": 2048,
            "temperature": 0.3,
            "n_ctx": 4096,
        }
    }
    
    def __init__(self, progress_callback: Optional[Callable] = None):
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.progress_callback = progress_callback
        self._lock = threading.Lock()
        self._model_loaded = False
        
        # Model paths
        self.models_dir = os.path.expanduser("~/Library/Application Support/Qwen3-ASR/models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Detect and initialize backend
        self._detect_backend()
    
    def _detect_backend(self):
        """Detect and initialize the best available backend"""
        # Try MLX first (best for Apple Silicon)
        try:
            import mlx.core as mx
            from mlx_lm import load, generate
            self.backend = 'mlx'
            self._load_mlx = load
            self._generate_mlx = generate
            print("✅ TextReformer: Using MLX backend (Apple Silicon optimized)")
            return
        except ImportError:
            pass
        
        # Fall back to llama-cpp
        try:
            from llama_cpp import Llama
            self.backend = 'llama_cpp'
            self._Llama = Llama
            print("✅ TextReformer: Using llama.cpp backend (CPU)")
            return
        except ImportError:
            pass
        
        self.backend = None
        print("⚠️ TextReformer: No LLM backend available. Install with: pip install mlx-lm llama-cpp-python")
    
    def is_available(self) -> bool:
        """Check if LLM backend is available"""
        return self.backend is not None
    
    def load_model(self, force_reload: bool = False) -> bool:
        """
        Load the LLM model.
        Returns True if successful.
        """
        if self._model_loaded and not force_reload:
            return True
        
        if not self.backend:
            return False
        
        try:
            if self.progress_callback:
                self.progress_callback("Loading LLM model...")
            
            if self.backend == 'mlx':
                self._load_mlx_model()
            elif self.backend == 'llama_cpp':
                self._load_llama_cpp_model()
            
            self._model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ TextReformer: Failed to load model: {e}")
            traceback.print_exc()
            return False
    
    def _load_mlx_model(self):
        """Load MLX model"""
        model_name = self.MODEL_CONFIG["mlx"]["name"]
        
        if self.progress_callback:
            self.progress_callback(f"Downloading {model_name}...")
        
        # mlx_lm.load handles caching automatically
        self.model, self.tokenizer = self._load_mlx(model_name)
        print(f"✅ TextReformer: Loaded {model_name}")
    
    def _load_llama_cpp_model(self):
        """Load llama.cpp model"""
        model_file = self.MODEL_CONFIG["llama_cpp"]["name"]
        model_path = os.path.join(self.models_dir, model_file)
        
        # Download if not exists
        if not os.path.exists(model_path):
            self._download_gguf_model(model_file)
        
        if self.progress_callback:
            self.progress_callback(f"Loading {model_file}...")
        
        self.model = self._Llama(
            model_path=model_path,
            n_ctx=self.MODEL_CONFIG["llama_cpp"]["n_ctx"],
            verbose=False
        )
        print(f"✅ TextReformer: Loaded {model_file}")
    
    def _download_gguf_model(self, model_file: str):
        """Download GGUF model from HuggingFace"""
        from urllib.request import urlopen
        import shutil
        
        # Qwen2.5-3B-Instruct Q4_K_M from HuggingFace
        base_url = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main"
        url = f"{base_url}/{model_file}"
        
        print(f"📥 TextReformer: Downloading {model_file}...")
        
        model_path = os.path.join(self.models_dir, model_file)
        
        with urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            chunk_size = 8192
            
            with open(model_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0 and self.progress_callback:
                        progress = downloaded / total_size
                        self.progress_callback(f"Downloading: {progress*100:.1f}%")
        
        print(f"✅ TextReformer: Downloaded {model_file}")
    
    def reform(self, 
               text: str, 
               mode: ReformMode = ReformMode.PUNCTUATE,
               target_language: Optional[str] = None) -> ReformResult:
        """
        Reform text using LLM.
        
        Args:
            text: Input text to reform
            mode: Reformation mode
            target_language: Target language for translation mode
            
        Returns:
            ReformResult with reformed text and metadata
        """
        result = ReformResult(original_text=text, mode=mode)
        
        if not text.strip():
            result.reformed_text = ""
            return result
        
        if not self._model_loaded:
            if not self.load_model():
                # Fallback: return original text if model unavailable
                result.reformed_text = text
                return result
        
        start_time = time.time()
        
        try:
            # Get prompt
            prompt_template = REFORM_PROMPTS[mode]
            if mode == ReformMode.TRANSLATE and target_language:
                prompt = prompt_template.format(text=text, target_language=target_language)
            else:
                prompt = prompt_template.format(text=text)
            
            # Generate
            with self._lock:
                if self.backend == 'mlx':
                    reformed = self._generate_mlx_text(prompt)
                else:
                    reformed = self._generate_llama_cpp_text(prompt)
            
            result.reformed_text = reformed.strip()
            result.processing_time = time.time() - start_time
            result.confidence = 0.9  # Placeholder
            
        except Exception as e:
            print(f"❌ TextReformer: Reform failed: {e}")
            result.reformed_text = text  # Fallback to original
        
        return result
    
    def analyze_correlations(self, text: str) -> CorrelationResult:
        """
        Analyze text for topics, entities, sentiment, and correlations.
        
        Args:
            text: Input text to analyze
            
        Returns:
            CorrelationResult with analysis data
        """
        result = CorrelationResult()
        
        if not text.strip():
            return result
        
        if not self._model_loaded:
            if not self.load_model():
                return result
        
        try:
            prompt = CORRELATION_PROMPT.format(text=text[:4000])  # Limit context
            
            with self._lock:
                if self.backend == 'mlx':
                    response = self._generate_mlx_text(prompt)
                else:
                    response = self._generate_llama_cpp_text(prompt)
            
            # Parse JSON response
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            
            result.topics = data.get('topics', [])
            result.entities = data.get('entities', [])
            result.sentiment = data.get('sentiment', 'neutral')
            result.sentiment_score = data.get('sentiment_score', 0.0)
            result.keywords = data.get('keywords', [])
            result.summary = data.get('summary', '')
            
        except Exception as e:
            print(f"❌ TextReformer: Analysis failed: {e}")
        
        return result
    
    def _generate_mlx_text(self, prompt: str) -> str:
        """Generate text using MLX"""
        from mlx_lm import generate
        
        config = self.MODEL_CONFIG["mlx"]
        
        # Format for instruction model
        if self.tokenizer:
            messages = [{"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        response = generate(
            self.model,
            self.tokenizer,
            formatted,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            verbose=False
        )
        
        # Extract assistant response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
            response = response.replace("<|im_end|>", "").strip()
        
        return response
    
    def _generate_llama_cpp_text(self, prompt: str) -> str:
        """Generate text using llama.cpp"""
        config = self.MODEL_CONFIG["llama_cpp"]
        
        # Format for Qwen2.5 chat
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        output = self.model(
            formatted,
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            stop=["<|im_end|>"],
            echo=False
        )
        
        return output['choices'][0]['text'].strip()
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or other content"""
        # Try to find JSON block
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        # Find JSON object
        start = text.find('{')
        end = text.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        
        return text
    
    def unload_model(self):
        """Unload model to free memory"""
        with self._lock:
            self.model = None
            self.tokenizer = None
            self._model_loaded = False
            import gc
            gc.collect()
            
            # Force MLX memory cleanup if available
            if self.backend == 'mlx':
                try:
                    import mlx.core as mx
                    mx.eval(mx.array(0))  # Trigger memory cleanup
                except:
                    pass
            
            print("✅ TextReformer: Model unloaded")


# =============================================================================
# Batch Processing
# =============================================================================

# =============================================================================
# Utility Functions
# =============================================================================

# =============================================================================
# Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    # Test the reformer
    print("Testing TextReformer...")
    
    reformer = TextReformer(progress_callback=lambda x: print(f"Progress: {x}"))
    
    if not reformer.is_available():
        print("No LLM backend available. Install dependencies:")
        print("  pip install mlx-lm")
        print("  pip install llama-cpp-python")
        sys.exit(1)
    
    # Load model
    if not reformer.load_model():
        print("Failed to load model")
        sys.exit(1)
    
    # Test text
    test_text = """
    um so like we need to discuss the project timeline uh 
    first we have the design phase then development 
    and finally testing and deployment 
    the deadline is end of march we need to make sure 
    everyone is on board with this schedule
    """
    
    print("\n" + "="*50)
    print("Original text:")
    print(test_text)
    
    # Test different modes
    for mode in [ReformMode.CLEAN, ReformMode.PUNCTUATE, ReformMode.KEY_POINTS]:
        print("\n" + "="*50)
        print(f"Mode: {mode.value}")
        result = reformer.reform(test_text, mode)
        print(f"Result ({result.processing_time:.2f}s):")
        print(result.reformed_text)
    
    # Test correlation analysis
    print("\n" + "="*50)
    print("Correlation Analysis:")
    analysis = reformer.analyze_correlations(test_text)
    print(f"Topics: {analysis.topics}")
    print(f"Sentiment: {analysis.sentiment} ({analysis.sentiment_score:.2f})")
    print(f"Keywords: {analysis.keywords}")
    print(f"Summary: {analysis.summary}")
    
    # Cleanup
    reformer.unload_model()
    print("\n✅ Test complete")
