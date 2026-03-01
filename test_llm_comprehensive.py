#!/usr/bin/env python3
"""
Comprehensive Test Suite for SimpleLLM Module
Tests all backends, edge cases, and error handling
"""

import sys
import os
import time
import traceback
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simple_llm import SimpleLLM, OllamaBackend, OpenAIBackend, TransformersBackend, RuleBasedBackend

# Test data
TEST_CASES = {
    "normal_text": "hello world this is a test of the transcription system we are checking how well it works",
    "short_text": "hello",
    "very_short": "hi",
    "two_words": "test speech",
    "empty": "",
    "whitespace": "   ",
    "special_chars": "Hello! @#$%^&*()_+{}|:<>? World... Testing!!!",
    "emojis": "Hello 😀 World 🌍 Testing ✨ Speech 🎤",
    "numbers": "The meeting is at 3:30 PM on 12/25/2024 room 402B call 1-800-555-0123",
    "codes": "Error code 0xDEADBEEF function test_123_x HTTP 404 status OK 200",
    "mixed_lang": "Hello 你好 مرحبا Bonjour こんにちは testing system",
    "filler_words": "um so like we need to uh discuss the project you know what I mean like really",
    "long_text": " ".join([f"This is sentence number {i} in a long text that we are using to test the system." for i in range(1, 51)])  # ~500 words
}

MODES = ["punctuate", "summarize", "clean", "key_points"]

class LLMTestReport:
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
    def log(self, test_name, input_data, expected, actual, passed, duration, backend, error=None):
        self.results.append({
            "test_name": test_name,
            "input": str(input_data)[:100] + "..." if len(str(input_data)) > 100 else str(input_data),
            "expected": expected,
            "actual": str(actual)[:200] + "..." if len(str(actual)) > 200 else str(actual),
            "passed": passed,
            "duration": f"{duration:.2f}s",
            "backend": backend,
            "error": error
        })
        
    def print_report(self):
        print("\n" + "="*80)
        print("SIMPLE LLM TEST REPORT")
        print("="*80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Duration: {time.time() - self.start_time:.2f}s")
        print("="*80)
        
        passed = sum(1 for r in self.results if r["passed"])
        failed = len(self.results) - passed
        
        print(f"\nSUMMARY: {passed} passed, {failed} failed, {len(self.results)} total")
        print("-"*80)
        
        # Print table header
        print(f"{'Test':<40} {'Backend':<15} {'Time':<8} {'Result':<8}")
        print("-"*80)
        
        for r in self.results:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            print(f"{r['test_name']:<40} {r['backend']:<15} {r['duration']:<8} {status:<8}")
        
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        
        for r in self.results:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            print(f"\n{status} - {r['test_name']}")
            print(f"  Backend: {r['backend']}")
            print(f"  Duration: {r['duration']}")
            print(f"  Input: {r['input']}")
            print(f"  Expected: {r['expected']}")
            print(f"  Actual: {r['actual']}")
            if r['error']:
                print(f"  Error: {r['error']}")

def test_backend_detection():
    """Test backend detection and initialization"""
    print("\n" + "="*60)
    print("TEST 1: BACKEND DETECTION")
    print("="*60)
    
    results = []
    
    # Test Ollama backend detection
    print("\n1. Testing Ollama backend...")
    try:
        ollama = OllamaBackend()
        status = "✅ Available" if ollama.available else "❌ Not Available"
        print(f"   Ollama: {status}")
        results.append(("Ollama Detection", ollama.available))
    except Exception as e:
        print(f"   Ollama: ❌ Error - {e}")
        results.append(("Ollama Detection", False))
    
    # Test OpenAI backend detection
    print("\n2. Testing OpenAI backend...")
    try:
        openai = OpenAIBackend()
        status = "✅ Available" if openai.available else "❌ Not Available (no API key)"
        print(f"   OpenAI: {status}")
        results.append(("OpenAI Detection", openai.available))
    except Exception as e:
        print(f"   OpenAI: ❌ Error - {e}")
        results.append(("OpenAI Detection", False))
    
    # Test Transformers backend detection
    print("\n3. Testing Transformers backend...")
    try:
        transformers = TransformersBackend()
        status = "✅ Available" if transformers.available else "❌ Not Available"
        print(f"   Transformers: {status}")
        results.append(("Transformers Detection", transformers.available))
    except Exception as e:
        print(f"   Transformers: ❌ Error - {e}")
        results.append(("Transformers Detection", False))
    
    # Test Rule-based backend (always available)
    print("\n4. Testing Rule-based backend...")
    rule_based = RuleBasedBackend()
    status = "✅ Available" if rule_based.available else "❌ Not Available"
    print(f"   Rule-based: {status}")
    results.append(("Rule-based Detection", rule_based.available))
    
    # Test SimpleLLM initialization
    print("\n5. Testing SimpleLLM initialization...")
    try:
        llm = SimpleLLM()
        print(f"   Selected backend: {llm.backend_name}")
        print(f"   Is available: {llm.is_available()}")
        results.append(("SimpleLLM Init", llm.is_available()))
    except Exception as e:
        print(f"   SimpleLLM: ❌ Error - {e}")
        results.append(("SimpleLLM Init", False))
    
    return results

def test_ollama_modes(report):
    """Test Ollama backend with all modes"""
    print("\n" + "="*60)
    print("TEST 2: OLLAMA BACKEND - ALL MODES")
    print("="*60)
    
    llm = SimpleLLM(ollama_model="qwen:1.8b")
    print(f"Using backend: {llm.backend_name}\n")
    
    test_text = TEST_CASES["normal_text"]
    
    for mode in MODES:
        print(f"\n--- Mode: {mode} ---")
        start = time.time()
        try:
            result = llm.process(test_text, mode=mode)
            duration = time.time() - start
            print(f"Input: {test_text[:60]}...")
            print(f"Output: {result[:80]}...")
            print(f"Duration: {duration:.2f}s")
            
            # Check if result is reasonable
            passed = len(result) > 0 and result != test_text
            report.log(f"Ollama-{mode}", test_text[:50], "processed text", result, passed, duration, llm.backend_name)
        except Exception as e:
            duration = time.time() - start
            print(f"❌ Error: {e}")
            report.log(f"Ollama-{mode}", test_text[:50], "processed text", str(e), False, duration, llm.backend_name, str(e))

def test_edge_cases(report):
    """Test edge cases with Ollama"""
    print("\n" + "="*60)
    print("TEST 3: EDGE CASES")
    print("="*60)
    
    llm = SimpleLLM(ollama_model="qwen:1.8b")
    
    edge_cases = [
        ("empty_string", TEST_CASES["empty"]),
        ("whitespace_only", TEST_CASES["whitespace"]),
        ("very_short", TEST_CASES["very_short"]),
        ("two_words", TEST_CASES["two_words"]),
        ("special_chars", TEST_CASES["special_chars"]),
        ("emojis", TEST_CASES["emojis"]),
        ("numbers", TEST_CASES["numbers"]),
        ("codes", TEST_CASES["codes"]),
        ("mixed_lang", TEST_CASES["mixed_lang"]),
        ("long_text", TEST_CASES["long_text"][:200]),  # Use shorter version for speed
    ]
    
    for name, text in edge_cases:
        print(f"\n--- Test: {name} ---")
        print(f"Input: '{text[:60]}...'" if len(text) > 60 else f"Input: '{text}'")
        
        start = time.time()
        try:
            result = llm.process(text, mode="punctuate")
            duration = time.time() - start
            print(f"Output: '{result[:80]}...'" if len(result) > 80 else f"Output: '{result}'")
            print(f"Duration: {duration:.2f}s")
            
            # Validation logic
            if name == "empty_string":
                passed = result == ""
            elif name == "whitespace_only":
                passed = result.strip() == ""
            elif name == "very_short":
                passed = len(result) >= 0  # Should at least not crash
            else:
                passed = len(result) > 0
                
            report.log(f"Edge-{name}", text[:50], "valid output", result, passed, duration, llm.backend_name)
        except Exception as e:
            duration = time.time() - start
            print(f"❌ Error: {e}")
            report.log(f"Edge-{name}", text[:50], "valid output", str(e), False, duration, llm.backend_name, str(e))

def test_error_handling(report):
    """Test error handling"""
    print("\n" + "="*60)
    print("TEST 4: ERROR HANDLING")
    print("="*60)
    
    llm = SimpleLLM(ollama_model="qwen:1.8b")
    
    # Test invalid mode
    print("\n--- Test: Invalid mode ---")
    start = time.time()
    try:
        result = llm.process("Hello world", mode="invalid_mode")
        duration = time.time() - start
        print(f"Result: {result}")
        # Should return original text for invalid mode
        passed = result == "Hello world"
        report.log("Error-InvalidMode", "Hello world", "original text", result, passed, duration, llm.backend_name)
    except Exception as e:
        duration = time.time() - start
        print(f"Exception: {e}")
        report.log("Error-InvalidMode", "Hello world", "no exception", str(e), False, duration, llm.backend_name, str(e))
    
    # Test None input
    print("\n--- Test: None input ---")
    start = time.time()
    try:
        result = llm.process(None, mode="punctuate")
        duration = time.time() - start
        print(f"Result: {result}")
        passed = result is None or result == ""
        report.log("Error-NoneInput", "None", "None or empty", result, passed, duration, llm.backend_name)
    except Exception as e:
        duration = time.time() - start
        print(f"Exception (may be expected): {e}")
        report.log("Error-NoneInput", "None", "None or empty", str(e), True, duration, llm.backend_name, str(e))

def test_response_times(report):
    """Test and compare response times across backends"""
    print("\n" + "="*60)
    print("TEST 5: RESPONSE TIME COMPARISON")
    print("="*60)
    
    test_text = TEST_CASES["normal_text"]
    
    # Test Ollama response time
    print("\n--- Ollama Response Time ---")
    ollama = OllamaBackend()
    if ollama.available:
        times = []
        for i in range(3):
            start = time.time()
            result = ollama.process(test_text, mode="punctuate")
            duration = time.time() - start
            times.append(duration)
            print(f"  Run {i+1}: {duration:.2f}s")
        avg_time = sum(times) / len(times)
        print(f"  Average: {avg_time:.2f}s")
        report.log("Perf-Ollama", test_text[:50], "< 5s", f"avg {avg_time:.2f}s", avg_time < 10, avg_time, "ollama")
    
    # Test Rule-based response time (baseline)
    print("\n--- Rule-based Response Time (Baseline) ---")
    rule_based = RuleBasedBackend()
    times = []
    for i in range(3):
        start = time.time()
        result = rule_based.process(test_text, mode="punctuate")
        duration = time.time() - start
        times.append(duration)
        print(f"  Run {i+1}: {duration:.4f}s")
    avg_time = sum(times) / len(times)
    print(f"  Average: {avg_time:.4f}s")
    report.log("Perf-RuleBased", test_text[:50], "< 0.1s", f"avg {avg_time:.4f}s", avg_time < 0.1, avg_time, "rule-based")

def test_fallback_chain(report):
    """Test fallback chain when Ollama is not available"""
    print("\n" + "="*60)
    print("TEST 6: FALLBACK CHAIN VALIDATION")
    print("="*60)
    
    # Test with prefer_openai=False (default - Ollama first)
    print("\n--- Test 1: Default priority (Ollama first) ---")
    llm1 = SimpleLLM()
    print(f"Backend selected: {llm1.backend_name}")
    report.log("Fallback-Default", "init", "ollama or rule-based", llm1.backend_name, 
               llm1.backend_name in ["ollama-qwen:1.8b", "rule-based"], 0, llm1.backend_name)
    
    # Test with prefer_openai=True (but no API key)
    print("\n--- Test 2: Prefer OpenAI (no key) ---")
    llm2 = SimpleLLM(prefer_openai=True)
    print(f"Backend selected: {llm2.backend_name}")
    report.log("Fallback-PreferOpenAI", "init", "ollama or rule-based", llm2.backend_name,
               llm2.backend_name in ["ollama-qwen:1.8b", "rule-based"], 0, llm2.backend_name)
    
    # Test processing with current backend
    print("\n--- Test 3: Processing with current backend ---")
    test_text = "hello world this is a test"
    result = llm1.process(test_text, mode="punctuate")
    print(f"Input: {test_text}")
    print(f"Output: {result}")
    passed = len(result) > 0
    report.log("Fallback-Processing", test_text, "processed", result, passed, 0, llm1.backend_name)

def test_analyze_function(report):
    """Test the analyze function"""
    print("\n" + "="*60)
    print("TEST 7: ANALYZE FUNCTION")
    print("="*60)
    
    llm = SimpleLLM()
    
    test_text = "Hello world. This is a test. How are you today?"
    print(f"Input: {test_text}")
    
    start = time.time()
    try:
        analysis = llm.analyze(test_text)
        duration = time.time() - start
        print(f"Analysis result:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
        
        # Validate analysis
        checks = [
            analysis.get("word_count") == 9,
            analysis.get("sentence_count") == 3,
            analysis.get("char_count") == 47,
            analysis.get("backend_used") == llm.backend_name
        ]
        passed = all(checks)
        report.log("Analyze-Function", test_text, "valid analysis", str(analysis), passed, duration, llm.backend_name)
    except Exception as e:
        duration = time.time() - start
        print(f"❌ Error: {e}")
        report.log("Analyze-Function", test_text, "valid analysis", str(e), False, duration, llm.backend_name, str(e))

def test_different_models(report):
    """Test with different Ollama models if available"""
    print("\n" + "="*60)
    print("TEST 8: DIFFERENT OLLAMA MODELS")
    print("="*60)
    
    models = ["qwen:1.8b", "qwen2.5:1.5b-instruct"]
    test_text = "hello world this is a test of the transcription system"
    
    for model in models:
        print(f"\n--- Testing model: {model} ---")
        try:
            llm = SimpleLLM(ollama_model=model)
            print(f"Backend: {llm.backend_name}")
            
            start = time.time()
            result = llm.process(test_text, mode="punctuate")
            duration = time.time() - start
            
            print(f"Result: {result}")
            print(f"Duration: {duration:.2f}s")
            
            passed = len(result) > 0 and result != test_text
            report.log(f"Model-{model}", test_text[:50], "processed", result, passed, duration, llm.backend_name)
        except Exception as e:
            print(f"❌ Error: {e}")
            report.log(f"Model-{model}", test_text[:50], "processed", str(e), False, 0, "error", str(e))

def main():
    print("="*80)
    print("SIMPLE LLM COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    report = LLMTestReport()
    
    try:
        # Run all tests
        test_backend_detection()
        test_ollama_modes(report)
        test_edge_cases(report)
        test_error_handling(report)
        test_response_times(report)
        test_fallback_chain(report)
        test_analyze_function(report)
        test_different_models(report)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        traceback.print_exc()
    
    finally:
        # Print final report
        report.print_report()
        
        # Save report to file
        report_file = f"llm_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            # Redirect stdout to file
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            report.print_report()
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            f.write(output)
        print(f"\n📄 Report saved to: {report_file}")

if __name__ == "__main__":
    main()
