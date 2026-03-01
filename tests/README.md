# UI Tests for Qwen3-ASR Pro

This directory contains comprehensive UI tests for the tkinter-based macOS speech-to-text application.

## Running the Tests

### Run all tests with verbose output:
```bash
python tests/test_ui.py
```

### Run with pytest:
```bash
pytest tests/test_ui.py -v
```

### Run specific test class:
```bash
python -m unittest tests.test_ui.TestColorConstants -v
```

### Run specific test:
```bash
python -m unittest tests.test_ui.TestColorConstants.test_all_colors_defined -v
```

## Test Coverage

### 1. Color Constants (`TestColorConstants`)
- ✅ All 17 colors defined in constants.py
- ✅ Valid hex format (#RRGGBB)
- ✅ Light theme verification

### 2. Responsive Breakpoints (`TestResponsiveBreakpoints`)
- ✅ Mobile breakpoint: 550px
- ✅ Compact breakpoint: 750px
- ✅ Correct ordering

### 3. Sidebar Behavior (`TestSidebarBehavior`)
- ✅ Expanded width: 260px
- ✅ Compact width: 60px
- ✅ Auto-collapse below 750px
- ✅ Auto-expand above 850px

### 4. Responsive Layout (`TestResponsiveLayout`)
- ✅ Desktop mode (> 750px)
- ✅ Compact mode (550-750px)
- ✅ Mobile mode (< 550px)

### 5. Control States (`TestControlStates`)
- ✅ Idle state (green: #16a34a)
- ✅ Recording state (red: #dc2626)
- ✅ Processing state (orange: #d97706)
- ✅ State transition sequence

### 6. Theme Consistency (`TestThemeConsistency`)
- ✅ Light backgrounds (> 200 brightness)
- ✅ Text contrast validation
- ✅ Primary color consistency
- ✅ Semantic colors distinct

### 7. Model Selector (`TestModelSelector`)
- ✅ 2 model options (0.6B, 1.7B)
- ✅ Default: 1.7B (Accurate)
- ✅ Readonly state

### 8. Language Selector (`TestLanguageSelector`)
- ✅ 8 language options
- ✅ Default: English
- ✅ Readonly state

### 9. Waveform Visualizer (`TestWaveformVisualizer`)
- ✅ 40-bar history buffer
- ✅ Color levels (green/yellow/red)
- ✅ Level update logic

### 10. Text Area Updates (`TestTextAreaUpdates`)
- ✅ Tag configurations (live, meta, title)
- ✅ Thread-safe update pattern
- ✅ Font configuration

### 11. Progress Indicators (`TestProgressIndicators`)
- ✅ Progress bar colors
- ✅ Live indicator states

### 12. Error Dialogs (`TestErrorDialogs`)
- ✅ Error color (#dc2626)
- ✅ Dialog structure

### 13. Performance Stats (`TestPerformanceStats`)
- ✅ Stats dataclass structure
- ✅ RTF calculation
- ✅ Display formatting

### 14. TTK Styles (`TestTTKStyles`)
- ✅ Combobox styling
- ✅ Scale/Slider styling
- ✅ Progress bar styling

### 15. Mobile Bottom Bar (`TestMobileBottomBar`)
- ✅ Height: 60px
- ✅ 4 components (record, timer, settings, files)
- ✅ Visibility at different widths

### 16. Slide Out Panel (`TestSlideOutPanel`)
- ✅ Width: 300px
- ✅ Initial state: closed

### 17. UI Element Verification (`TestUIElementVerification`)
- ✅ 17 UI elements tracked
- ✅ Properties validation

### 18. Timer Formatting (`TestTimerFormatting`)
- ✅ MM:SS format
- ✅ Various durations (0s to 3600s)
- ✅ Monospace font

### 19. Silence Presets (`TestSilencePresets`)
- ✅ Fast: 0.8s
- ✅ Class: 30s
- ✅ Max: 60s
- ✅ Slider range: 0.5-60s

### 20. Responsive Behavior Matrix (`TestResponsiveBehaviorMatrix`)
- ✅ All width scenarios tested

### 21. Window Configuration (`TestWindowConfiguration`)
- ✅ Default size: 1100x800
- ✅ Min size: 450x550
- ✅ Resize threshold: 50px

### 22. Recording Modes (`TestRecordingModes`)
- ✅ Live mode (🎓 Live)
- ✅ Batch mode (⚡ Fast)
- ✅ Default: live

### 23. Action Buttons (`TestActionButtons`)
- ✅ Clear (🗑️)
- ✅ Copy (📋)
- ✅ Save (💾)

### 24. Status Messages (`TestStatusMessages`)
- ✅ All states with correct colors

## Test Report

The test suite generates a detailed report including:
1. UI Element Verification Checklist
2. Responsive Behavior at Breakpoints
3. Theme Color Verification
4. Control State Colors
5. Typography
6. UI Update Performance
7. Potential UI Glitches/Issues
8. Recommendations

## Design Decisions

### Why No GUI Testing?
These tests focus on:
- Configuration validation (colors, sizes, constants)
- Logic verification (responsive breakpoints, state transitions)
- Structure validation (element properties, relationships)

Actual GUI testing would require:
- X11 display server or macOS GUI
- Screenshot comparison (fragile)
- Slow execution
- Platform-specific issues

### Thread Safety Testing
The app uses `root.after()` for thread-safe UI updates from background threads. Tests verify:
- The pattern is documented
- Queue-based communication is used
- Callback mechanisms are in place

## Adding New Tests

To add a new test:

```python
class TestNewFeature(unittest.TestCase):
    """Test description"""
    
    def test_specific_behavior(self):
        """Test description"""
        # Test code here
        self.assertEqual(expected, actual)
```

## Continuous Integration

These tests are suitable for CI/CD because they:
- Run without a display
- Complete in < 1 second
- Have no external dependencies (except constants.py)
- Provide clear pass/fail results
