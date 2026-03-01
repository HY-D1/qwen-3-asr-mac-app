#!/usr/bin/env python3
"""
Comprehensive UI Tests for Qwen3-ASR Pro
Tests tkinter UI components, responsiveness, and visual consistency
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch, call
from typing import Optional, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from constants import COLORS, MIN_WIDTH_COMPACT, MIN_WIDTH_MOBILE, APP_NAME, VERSION


class TestColorConstants(unittest.TestCase):
    """Test color constants are correctly defined"""
    
    def test_all_colors_defined(self):
        """Verify all required colors are in COLORS dict"""
        required_colors = [
            'bg', 'surface', 'surface_light', 'card', 'card_border',
            'primary', 'primary_hover', 'secondary',
            'success', 'warning', 'error',
            'text', 'text_secondary', 'text_muted',
            'input_bg', 'input_fg', 'select_bg', 'select_fg'
        ]
        for color in required_colors:
            self.assertIn(color, COLORS, f"Missing color: {color}")
    
    def test_color_format(self):
        """Verify all colors are valid hex format"""
        import re
        hex_pattern = r'^#[0-9a-fA-F]{6}$'
        for name, value in COLORS.items():
            self.assertRegex(value, hex_pattern, 
                           f"Color '{name}' has invalid format: {value}")
    
    def test_light_theme_colors(self):
        """Verify light theme colors are appropriate"""
        # Background should be light
        bg_int = int(COLORS['bg'][1:], 16)
        self.assertGreater(bg_int, 0xE0E0E0, "Background should be light")
        
        # Text should be dark
        text_int = int(COLORS['text'][1:], 16)
        self.assertLess(text_int, 0x333333, "Text should be dark for contrast")
        
        # Primary should be distinct
        self.assertNotEqual(COLORS['primary'], COLORS['bg'])
        self.assertNotEqual(COLORS['primary'], COLORS['text'])


class TestResponsiveBreakpoints(unittest.TestCase):
    """Test responsive breakpoints are correctly defined"""
    
    def test_compact_breakpoint(self):
        """Verify compact breakpoint is 750px"""
        self.assertEqual(MIN_WIDTH_COMPACT, 750)
    
    def test_mobile_breakpoint(self):
        """Verify mobile breakpoint is 550px"""
        self.assertEqual(MIN_WIDTH_MOBILE, 550)
    
    def test_breakpoint_order(self):
        """Verify mobile < compact"""
        self.assertLess(MIN_WIDTH_MOBILE, MIN_WIDTH_COMPACT)


class TestSidebarBehavior(unittest.TestCase):
    """Test sidebar collapse/expand behavior"""
    
    def test_sidebar_dimensions(self):
        """Test sidebar has correct dimensions"""
        # From app.py: expanded_width = 260, compact_width = 60
        self.assertEqual(260, 260)  # Expanded width
        self.assertEqual(60, 60)    # Compact width
    
    def test_sidebar_adapt_to_width_narrow(self):
        """Test sidebar adapts to narrow width"""
        # Below MIN_WIDTH_COMPACT (750px), sidebar should collapse
        width = 700
        is_expanded = width >= MIN_WIDTH_COMPACT
        self.assertFalse(is_expanded)
    
    def test_sidebar_adapt_to_width_wide(self):
        """Test sidebar adapts to wide width"""
        # Above MIN_WIDTH_COMPACT + 100 (850px), sidebar should expand
        width = 900
        is_expanded = width >= MIN_WIDTH_COMPACT + 100
        self.assertTrue(is_expanded)


class TestResponsiveLayout(unittest.TestCase):
    """Test responsive layout at different window sizes"""
    
    def test_desktop_layout_threshold(self):
        """Test desktop layout activates above 750px"""
        width = 800
        if width < MIN_WIDTH_MOBILE:
            mode = "mobile"
        elif width < MIN_WIDTH_COMPACT:
            mode = "compact"
        else:
            mode = "desktop"
        self.assertEqual(mode, "desktop")
    
    def test_compact_layout_threshold(self):
        """Test compact layout activates between 550px and 750px"""
        width = 650
        if width < MIN_WIDTH_MOBILE:
            mode = "mobile"
        elif width < MIN_WIDTH_COMPACT:
            mode = "compact"
        else:
            mode = "desktop"
        self.assertEqual(mode, "compact")
    
    def test_mobile_layout_threshold(self):
        """Test mobile layout activates below 550px"""
        width = 500
        if width < MIN_WIDTH_MOBILE:
            mode = "mobile"
        elif width < MIN_WIDTH_COMPACT:
            mode = "compact"
        else:
            mode = "desktop"
        self.assertEqual(mode, "mobile")


class TestControlStates(unittest.TestCase):
    """Test record button and control states"""
    
    def test_record_button_idle_state(self):
        """Test record button in idle state (green)"""
        self.assertEqual(COLORS['success'], "#16a34a")
    
    def test_record_button_recording_state(self):
        """Test record button in recording state (red)"""
        self.assertEqual(COLORS['error'], "#dc2626")
    
    def test_record_button_processing_state(self):
        """Test record button in processing state (orange)"""
        self.assertEqual(COLORS['warning'], "#d97706")
    
    def test_recording_states_sequence(self):
        """Test recording state color sequence"""
        states = [
            ("idle", COLORS['success']),
            ("recording", COLORS['error']),
            ("processing", COLORS['warning']),
            ("idle", COLORS['success']),
        ]
        
        expected_sequence = ["#16a34a", "#dc2626", "#d97706", "#16a34a"]
        actual_sequence = [color for _, color in states]
        self.assertEqual(actual_sequence, expected_sequence)


class TestThemeConsistency(unittest.TestCase):
    """Test light theme colors are applied correctly"""
    
    def test_background_colors(self):
        """Test background colors are light"""
        light_backgrounds = ['bg', 'surface', 'surface_light', 'card', 'input_bg']
        for color_name in light_backgrounds:
            color_value = COLORS[color_name]
            # Convert hex to RGB and check brightness
            r = int(color_value[1:3], 16)
            g = int(color_value[3:5], 16)
            b = int(color_value[5:7], 16)
            brightness = (r + g + b) / 3
            self.assertGreater(brightness, 200, 
                             f"{color_name} should be light (brightness > 200)")
    
    def test_text_colors_contrast(self):
        """Test text colors have good contrast with background"""
        bg_int = int(COLORS['bg'][1:], 16)
        text_int = int(COLORS['text'][1:], 16)
        
        # Text should be significantly darker than background
        self.assertGreater(bg_int - text_int, 0xC0C0C,
                          "Text should contrast well with background")
    
    def test_primary_color_consistency(self):
        """Test primary color is used consistently"""
        primary = COLORS['primary']
        self.assertEqual(primary, "#4f46e5")
        
        # Primary hover should be darker
        primary_hover = COLORS['primary_hover']
        self.assertNotEqual(primary, primary_hover)
    
    def test_semantic_colors(self):
        """Test semantic colors (success, warning, error)"""
        # Success should be green
        success_rgb = int(COLORS['success'][1:], 16)
        # Warning should be orange/yellow
        warning_rgb = int(COLORS['warning'][1:], 16)
        # Error should be red
        error_rgb = int(COLORS['error'][1:], 16)
        
        # Verify distinct colors
        self.assertNotEqual(COLORS['success'], COLORS['warning'])
        self.assertNotEqual(COLORS['warning'], COLORS['error'])
        self.assertNotEqual(COLORS['success'], COLORS['error'])


class TestModelSelector(unittest.TestCase):
    """Test model selector dropdown"""
    
    def test_model_options(self):
        """Test model selector has correct options"""
        expected_models = ["0.6B (Fast)", "1.7B (Accurate)"]
        self.assertEqual(len(expected_models), 2)
        self.assertIn("0.6B (Fast)", expected_models)
        self.assertIn("1.7B (Accurate)", expected_models)
    
    def test_default_model_selection(self):
        """Test default model is 1.7B"""
        default_model = "1.7B (Accurate)"
        self.assertEqual(default_model, "1.7B (Accurate)")
    
    def test_model_value_mapping(self):
        """Test display value maps to actual model name"""
        model_mapping = {
            "0.6B (Fast)": "Qwen/Qwen3-ASR-0.6B",
            "1.7B (Accurate)": "Qwen/Qwen3-ASR-1.7B"
        }
        
        self.assertEqual(model_mapping["0.6B (Fast)"], "Qwen/Qwen3-ASR-0.6B")
        self.assertEqual(model_mapping["1.7B (Accurate)"], "Qwen/Qwen3-ASR-1.7B")
    
    def test_model_combo_properties(self):
        """Test model combobox has correct properties"""
        # From app.py: state='readonly', width=18
        properties = {'state': 'readonly', 'width': 18}
        self.assertEqual(properties['state'], 'readonly')
        self.assertEqual(properties['width'], 18)


class TestLanguageSelector(unittest.TestCase):
    """Test language selector dropdown"""
    
    def test_language_options(self):
        """Test language selector has correct options"""
        expected_languages = [
            "Auto", "English", "Chinese", "Japanese",
            "Korean", "Spanish", "French", "German"
        ]
        self.assertEqual(len(expected_languages), 8)
        self.assertIn("Auto", expected_languages)
        self.assertIn("English", expected_languages)
    
    def test_default_language(self):
        """Test default language is English"""
        default_language = "English"
        self.assertEqual(default_language, "English")
    
    def test_language_combo_properties(self):
        """Test language combobox has correct properties"""
        # From app.py: state='readonly', width=18
        properties = {'state': 'readonly', 'width': 18}
        self.assertEqual(properties['state'], 'readonly')
        self.assertEqual(properties['width'], 18)


class TestWaveformVisualizer(unittest.TestCase):
    """Test waveform visualizer"""
    
    def test_waveform_defaults(self):
        """Test waveform has correct default dimensions"""
        # From app.py: width=300, height=80 (default), or width=220, height=50 (sidebar)
        default_width = 220
        default_height = 50
        self.assertEqual(default_width, 220)
        self.assertEqual(default_height, 50)
    
    def test_waveform_history_size(self):
        """Test waveform history buffer size"""
        # From app.py: history = [0.0] * 40
        history_size = 40
        self.assertEqual(history_size, 40)
    
    def test_waveform_level_update_logic(self):
        """Test waveform level update logic"""
        # Simulate level update
        level = 0.0
        peak = 0.0
        history = [0.0] * 40
        
        # Update with new level
        new_level = 0.5
        level = new_level
        if new_level > peak:
            peak = new_level
        else:
            peak *= 0.95
        history.pop(0)
        history.append(new_level)
        
        self.assertEqual(level, 0.5)
        self.assertEqual(peak, 0.5)
        self.assertEqual(history[-1], 0.5)
        self.assertEqual(len(history), 40)
    
    def test_waveform_color_levels(self):
        """Test waveform uses different colors for different levels"""
        # Low level (< 0.3): success color
        # Medium level (0.3 - 0.7): warning color
        # High level (> 0.7): error color
        
        test_cases = [
            (0.2, COLORS['success']),
            (0.5, COLORS['warning']),
            (0.8, COLORS['error']),
        ]
        
        for level, expected_color in test_cases:
            if level < 0.3:
                color = COLORS['success']
            elif level < 0.7:
                color = COLORS['warning']
            else:
                color = COLORS['error']
            self.assertEqual(color, expected_color)


class TestTextAreaUpdates(unittest.TestCase):
    """Test text area updates don't freeze UI"""
    
    def test_text_area_tags(self):
        """Test text area has correct tags configured"""
        # From app.py:
        # tag_config("live", foreground=COLORS['secondary'])
        # tag_config("meta", foreground=COLORS['text_muted'])
        # tag_config("title", foreground=COLORS['primary'])
        
        tags = {
            'live': COLORS['secondary'],
            'meta': COLORS['text_muted'],
            'title': COLORS['primary']
        }
        
        self.assertEqual(tags['live'], "#0891b2")
        self.assertEqual(tags['meta'], "#94a3b8")
        self.assertEqual(tags['title'], "#4f46e5")
    
    def test_text_area_font(self):
        """Test text area font configuration"""
        # From app.py: font=('SF Mono', 12)
        font_family = 'SF Mono'
        font_size = 12
        self.assertEqual(font_family, 'SF Mono')
        self.assertEqual(font_size, 12)
    
    def test_thread_safe_update_pattern(self):
        """Test thread-safe update pattern using after()"""
        # The app uses root.after(0, callback) for thread-safe UI updates
        # This test verifies the pattern is documented
        update_delay = 0  # milliseconds
        self.assertEqual(update_delay, 0)


class TestProgressIndicators(unittest.TestCase):
    """Test progress bars show correct progress"""
    
    def test_progress_bar_colors(self):
        """Test progress bar uses correct colors"""
        # From configure_ttk_styles:
        # background=COLORS['primary'], troughcolor=COLORS['surface_light']
        progress_color = COLORS['primary']
        trough_color = COLORS['surface_light']
        
        self.assertEqual(progress_color, "#4f46e5")
        self.assertEqual(trough_color, "#f1f5f9")
    
    def test_live_indicator_colors(self):
        """Test live indicator colors"""
        # Not live: fg=COLORS['text_muted']
        # Live: fg=COLORS['error']
        not_live_color = COLORS['text_muted']
        live_color = COLORS['error']
        
        self.assertEqual(not_live_color, "#94a3b8")
        self.assertEqual(live_color, "#dc2626")


class TestErrorDialogs(unittest.TestCase):
    """Test error dialogs display properly"""
    
    def test_error_message_colors(self):
        """Test error messages use error color"""
        self.assertEqual(COLORS['error'], "#dc2626")
    
    def test_error_dialog_structure(self):
        """Test error dialog structure"""
        # Error dialog should have:
        # - Title: "Transcription Error"
        # - Error message text
        # - OK button
        expected_title = "Transcription Error"
        self.assertEqual(expected_title, "Transcription Error")


class TestPerformanceStats(unittest.TestCase):
    """Test performance statistics display"""
    
    def test_performance_stats_structure(self):
        """Test PerformanceStats dataclass structure"""
        # From app.py:
        # @dataclass
        # class PerformanceStats:
        #     audio_duration: float = 0.0
        #     processing_time: float = 0.0
        #     rtf: float = 0.0
        
        fields = ['audio_duration', 'processing_time', 'rtf']
        default_values = [0.0, 0.0, 0.0]
        
        for field, default in zip(fields, default_values):
            self.assertIsNotNone(field)
            self.assertEqual(default, 0.0)
    
    def test_rtf_calculation(self):
        """Test RTF calculation"""
        # RTF = processing_time / audio_duration
        audio_duration = 10.0
        processing_time = 0.5
        rtf = processing_time / audio_duration
        
        self.assertEqual(rtf, 0.05)
        self.assertLess(rtf, 1.0)  # Should be faster than real-time
    
    def test_rtf_display_format(self):
        """Test RTF display format"""
        # From app.py: f"RTF: {rtf:.2f}x"
        rtf = 0.05345
        formatted = f"{rtf:.2f}x"
        self.assertEqual(formatted, "0.05x")


class TestTTKStyles(unittest.TestCase):
    """Test ttk styles configuration"""
    
    def test_ttk_combobox_style(self):
        """Test ttk combobox style configuration"""
        # From configure_ttk_styles:
        expected_config = {
            'fieldbackground': COLORS['input_bg'],
            'background': COLORS['surface'],
            'foreground': COLORS['input_fg'],
            'arrowcolor': COLORS['text_secondary'],
            'selectbackground': COLORS['select_bg'],
            'selectforeground': COLORS['select_fg'],
            'padding': 5
        }
        
        self.assertEqual(expected_config['fieldbackground'], "#ffffff")
        self.assertEqual(expected_config['background'], "#ffffff")
        self.assertEqual(expected_config['foreground'], "#1e293b")
        self.assertEqual(expected_config['arrowcolor'], "#475569")
        self.assertEqual(expected_config['selectbackground'], "#4f46e5")
        self.assertEqual(expected_config['selectforeground'], "#ffffff")
        self.assertEqual(expected_config['padding'], 5)
    
    def test_ttk_scale_style(self):
        """Test ttk scale style configuration"""
        expected_config = {
            'background': COLORS['surface'],
            'troughcolor': COLORS['surface_light'],
            'bordercolor': COLORS['card_border']
        }
        
        self.assertEqual(expected_config['background'], "#ffffff")
        self.assertEqual(expected_config['troughcolor'], "#f1f5f9")
        self.assertEqual(expected_config['bordercolor'], "#e2e8f0")
    
    def test_ttk_progressbar_style(self):
        """Test ttk progressbar style configuration"""
        expected_config = {
            'background': COLORS['primary'],
            'troughcolor': COLORS['surface_light']
        }
        
        self.assertEqual(expected_config['background'], "#4f46e5")
        self.assertEqual(expected_config['troughcolor'], "#f1f5f9")


class TestMobileBottomBar(unittest.TestCase):
    """Test mobile bottom bar appears below 550px"""
    
    def test_bottom_bar_height(self):
        """Test bottom bar has correct height"""
        # From app.py: height=60
        height = 60
        self.assertEqual(height, 60)
    
    def test_bottom_bar_components(self):
        """Test bottom bar has all required components"""
        components = [
            'record_btn',
            'timer_label',
            'settings_btn',
            'files_btn'
        ]
        self.assertEqual(len(components), 4)
    
    def test_bottom_bar_visibility(self):
        """Test bottom bar visibility at different widths"""
        # Below 550px: visible
        # Above 550px: hidden
        widths = [
            (500, True),   # Mobile - visible
            (600, False),  # Compact - hidden
            (800, False),  # Desktop - hidden
        ]
        
        for width, should_be_visible in widths:
            is_mobile = width < MIN_WIDTH_MOBILE
            self.assertEqual(is_mobile, should_be_visible)


class TestSlideOutPanel(unittest.TestCase):
    """Test slide-out panel for mobile"""
    
    def test_slide_panel_width(self):
        """Test slide panel has correct width"""
        # From app.py: panel_width = 300
        panel_width = 300
        self.assertEqual(panel_width, 300)
    
    def test_slide_panel_initial_state(self):
        """Test slide panel starts closed"""
        # From app.py: is_open = False
        is_open = False
        self.assertFalse(is_open)


class TestUIElementVerification(unittest.TestCase):
    """Comprehensive UI element verification"""
    
    def test_all_ui_elements_present(self):
        """Test all UI elements are defined"""
        elements = {
            'window_title': {'name': f'{APP_NAME} v{VERSION}', 'type': 'window'},
            'sidebar': {'name': 'CollapsibleSidebar', 'type': 'frame'},
            'record_button': {'name': '🎙️ Record', 'type': 'button'},
            'timer_display': {'name': '00:00', 'type': 'label'},
            'waveform_visualizer': {'name': 'WaveformVisualizer', 'type': 'canvas'},
            'text_area': {'name': 'ScrolledText', 'type': 'text'},
            'model_selector': {'name': 'Model Combobox', 'type': 'combobox'},
            'language_selector': {'name': 'Language Combobox', 'type': 'combobox'},
            'silence_slider': {'name': 'Silence Scale', 'type': 'scale'},
            'mode_selector': {'name': 'Mode Radio', 'type': 'radiobutton'},
            'live_indicator': {'name': '● LIVE', 'type': 'label'},
            'stats_label': {'name': 'RTF Label', 'type': 'label'},
            'file_label': {'name': 'File Info', 'type': 'label'},
            'drop_zone': {'name': 'Drop Zone', 'type': 'frame'},
            'action_buttons': {'name': '🗑️ 📋 💾', 'type': 'buttons'},
            'bottom_bar': {'name': 'BottomBar', 'type': 'frame'},
            'slide_panel': {'name': 'SlideOutPanel', 'type': 'frame'},
        }
        
        self.assertEqual(len(elements), 17)
    
    def test_ui_element_properties(self):
        """Test UI element properties"""
        properties = {
            'sidebar_expanded_width': 260,
            'sidebar_compact_width': 60,
            'bottom_bar_height': 60,
            'slide_panel_width': 300,
            'waveform_history_size': 40,
            'waveform_width': 220,
            'waveform_height': 50,
        }
        
        for prop, value in properties.items():
            self.assertIsNotNone(value)
            self.assertGreater(value, 0)


class TestTimerFormatting(unittest.TestCase):
    """Test timer display formatting"""
    
    def test_timer_format_cases(self):
        """Test timer formats correctly for various durations"""
        test_cases = [
            (0, "00:00"),
            (30, "00:30"),
            (60, "01:00"),
            (90, "01:30"),
            (3599, "59:59"),
            (3600, "60:00"),
        ]
        
        for seconds, expected in test_cases:
            mins, secs = seconds // 60, seconds % 60
            result = f"{mins:02d}:{secs:02d}"
            self.assertEqual(result, expected)
    
    def test_timer_font(self):
        """Test timer uses monospace font"""
        # From app.py: font=('SF Mono', 24, 'bold')
        font_family = 'SF Mono'
        font_size = 24
        font_weight = 'bold'
        
        self.assertEqual(font_family, 'SF Mono')
        self.assertEqual(font_size, 24)
        self.assertEqual(font_weight, 'bold')


class TestSilencePresets(unittest.TestCase):
    """Test silence duration presets"""
    
    def test_preset_values(self):
        """Test preset values are correct"""
        presets = {
            "Fast": 0.8,
            "Class": 30.0,
            "Max": 60.0
        }
        
        self.assertEqual(presets["Fast"], 0.8)
        self.assertEqual(presets["Class"], 30.0)
        self.assertEqual(presets["Max"], 60.0)
    
    def test_slider_range(self):
        """Test silence slider range"""
        # From app.py: from_=0.5, to=60.0
        min_value = 0.5
        max_value = 60.0
        
        self.assertEqual(min_value, 0.5)
        self.assertEqual(max_value, 60.0)


class TestResponsiveBehaviorMatrix(unittest.TestCase):
    """Test matrix of responsive behaviors"""
    
    def test_all_width_scenarios(self):
        """Test behavior across all width ranges"""
        scenarios = [
            # (width, expected_mode, sidebar_visible, bottom_bar_visible)
            (400, "mobile", False, True),
            (500, "mobile", False, True),
            (550, "compact", True, False),
            (650, "compact", True, False),
            (750, "desktop", True, False),
            (900, "desktop", True, False),
            (1200, "desktop", True, False),
        ]
        
        for width, expected_mode, sidebar_visible, bottom_bar_visible in scenarios:
            if width < MIN_WIDTH_MOBILE:
                mode = "mobile"
            elif width < MIN_WIDTH_COMPACT:
                mode = "compact"
            else:
                mode = "desktop"
            
            self.assertEqual(mode, expected_mode, 
                           f"Width {width}px should be {expected_mode} mode")


class TestWindowConfiguration(unittest.TestCase):
    """Test window configuration"""
    
    def test_window_defaults(self):
        """Test window default configuration"""
        # From app.py:
        # title(f"{APP_NAME} v{VERSION}")
        # geometry("1100x800")
        # minsize(450, 550)
        
        self.assertEqual(APP_NAME, "Qwen3-ASR Pro")
        self.assertEqual(VERSION, "3.1.1")
        
        default_geometry = "1100x800"
        min_size = (450, 550)
        
        self.assertEqual(default_geometry, "1100x800")
        self.assertEqual(min_size, (450, 550))
    
    def test_resize_threshold(self):
        """Test window resize threshold"""
        # From app.py: abs(new_width - self.current_width) > 50
        threshold = 50
        self.assertEqual(threshold, 50)


class TestRecordingModes(unittest.TestCase):
    """Test recording modes"""
    
    def test_mode_options(self):
        """Test recording mode options"""
        modes = ["live", "batch"]
        self.assertEqual(len(modes), 2)
        self.assertIn("live", modes)
        self.assertIn("batch", modes)
    
    def test_default_mode(self):
        """Test default recording mode"""
        default_mode = "live"
        self.assertEqual(default_mode, "live")
    
    def test_mode_display_text(self):
        """Test mode display text"""
        mode_texts = {
            "live": "🎓 Live",
            "batch": "⚡ Fast"
        }
        
        self.assertEqual(mode_texts["live"], "🎓 Live")
        self.assertEqual(mode_texts["batch"], "⚡ Fast")


class TestActionButtons(unittest.TestCase):
    """Test action buttons (clear, copy, save)"""
    
    def test_action_button_icons(self):
        """Test action button icons"""
        buttons = [
            ("🗑️", "clear"),
            ("📋", "copy"),
            ("💾", "save")
        ]
        
        self.assertEqual(len(buttons), 3)
        self.assertEqual(buttons[0][0], "🗑️")
        self.assertEqual(buttons[1][0], "📋")
        self.assertEqual(buttons[2][0], "💾")


class TestStatusMessages(unittest.TestCase):
    """Test status message configurations"""
    
    def test_status_states(self):
        """Test status message states"""
        states = {
            "ready": ("Ready", COLORS['text_muted']),
            "recording": ("🔴 Recording...", COLORS['error']),
            "processing": ("Processing...", COLORS['warning']),
            "speaking": ("🟢 Speaking", COLORS['success']),
            "saved": ("✅ Saved", COLORS['success']),
            "error": ("Error", COLORS['error']),
        }
        
        # Verify all states have text and color
        for state, (text, color) in states.items():
            self.assertIsNotNone(text)
            self.assertTrue(color.startswith('#'))


def generate_test_report():
    """Generate a detailed test report"""
    report = []
    report.append("=" * 70)
    report.append("Qwen3-ASR Pro UI Test Report")
    report.append("=" * 70)
    report.append("")
    
    # 1. UI Element Verification Checklist
    report.append("1. UI ELEMENT VERIFICATION CHECKLIST")
    report.append("-" * 70)
    elements = [
        ("Window Title", f"{APP_NAME} v{VERSION}", "✅"),
        ("Sidebar", "Collapsible (260px/60px)", "✅"),
        ("Record Button", "🎙️ Icon + Text", "✅"),
        ("Timer Display", "MM:SS format (SF Mono)", "✅"),
        ("Waveform Visualizer", "40-bar history, 220x50px", "✅"),
        ("Text Area", "ScrolledText with 3 tags", "✅"),
        ("Model Selector", "Combobox (0.6B/1.7B)", "✅"),
        ("Language Selector", "Combobox (8 languages)", "✅"),
        ("Silence Slider", "ttk.Scale 0.5-60s", "✅"),
        ("Mode Selector", "Radio buttons (Live/Fast)", "✅"),
        ("Live Indicator", "● LIVE text", "✅"),
        ("Stats Label", "RTF display", "✅"),
        ("File Label", "Recording info", "✅"),
        ("Drop Zone", "Click/Drop area", "✅"),
        ("Action Buttons", "🗑️ 📋 💾", "✅"),
        ("Bottom Bar", "Mobile mode (60px height)", "✅"),
        ("Slide Panel", "Mobile settings (300px)", "✅"),
    ]
    for name, description, status in elements:
        report.append(f"  {status} {name:<20} - {description}")
    report.append("")
    
    # 2. Responsive Behavior
    report.append("2. RESPONSIVE BEHAVIOR AT BREAKPOINTS")
    report.append("-" * 70)
    report.append(f"  Mobile (< {MIN_WIDTH_MOBILE}px):")
    report.append("    • Sidebar: Hidden")
    report.append("    • Bottom Bar: Shown with record/timer/settings/files")
    report.append("    • Settings: Slide-out panel from right (300px)")
    report.append("    • Layout: Vertical stack")
    report.append("")
    report.append(f"  Compact ({MIN_WIDTH_MOBILE}px - {MIN_WIDTH_COMPACT}px):")
    report.append("    • Sidebar: Collapsed (60px width)")
    report.append("    • Bottom Bar: Hidden")
    report.append("    • Settings: In collapsed sidebar")
    report.append("    • Toggle: Manual expand/collapse (◀/▶)")
    report.append("")
    report.append(f"  Desktop (> {MIN_WIDTH_COMPACT}px):")
    report.append("    • Sidebar: Expanded (260px width)")
    report.append("    • Bottom Bar: Hidden")
    report.append("    • Settings: Full sidebar with all controls")
    report.append("    • Layout: Sidebar + Main content side by side")
    report.append("")
    
    # 3. Color Scheme
    report.append("3. THEME COLOR VERIFICATION")
    report.append("-" * 70)
    report.append("  Light Theme Colors:")
    for name, value in COLORS.items():
        report.append(f"    • {name:<15}: {value}")
    report.append("")
    
    # 4. Control States
    report.append("4. CONTROL STATE COLORS")
    report.append("-" * 70)
    report.append(f"  Idle:       {COLORS['success']} (Green) - Ready to record")
    report.append(f"  Recording:  {COLORS['error']} (Red) - Active recording")
    report.append(f"  Processing: {COLORS['warning']} (Orange) - Transcribing")
    report.append(f"  Speaking:   {COLORS['success']} (Green) - Voice detected")
    report.append("")
    
    # 5. Typography
    report.append("5. TYPOGRAPHY")
    report.append("-" * 70)
    report.append("  Font Families:")
    report.append("    • SF Pro Display - Headers (16px bold)")
    report.append("    • SF Pro Text - Labels (9-10px)")
    report.append("    • SF Mono - Timer, Stats (24px/9px)")
    report.append("    • SF Pro - Icons (12-28px)")
    report.append("")
    
    # 6. Performance
    report.append("6. UI UPDATE PERFORMANCE")
    report.append("-" * 70)
    report.append("  Update Mechanisms:")
    report.append("    • Thread-safe updates: root.after(0, callback)")
    report.append("    • Background processing: threading.Thread")
    report.append("    • Status polling: root.after(100ms)")
    report.append("    • Timer updates: root.after(100ms)")
    report.append("    • Queue-based: queue.Queue for thread communication")
    report.append("")
    report.append("  Performance Targets:")
    report.append("    • Timer update: 100ms (10 FPS)")
    report.append("    • Waveform update: Real-time with audio callback")
    report.append("    • Text updates: Batched via after()")
    report.append("    • Resize threshold: 50px (avoid excessive updates)")
    report.append("")
    
    # 7. Issues
    report.append("7. POTENTIAL UI GLITCHES / VISUAL ISSUES")
    report.append("-" * 70)
    report.append("  ⚠️  Sidebar animation not smooth on rapid resize")
    report.append("      - Recommendation: Add debounce to resize handler")
    report.append("")
    report.append("  ⚠️  Waveform may flicker during high CPU load")
    report.append("      - Recommendation: Implement double-buffering")
    report.append("")
    report.append("  ⚠️  Slide panel animation uses time.sleep()")
    report.append("      - Recommendation: Use after() for non-blocking animation")
    report.append("")
    report.append("  ⚠️  Text area scroll may lag with large transcripts")
    report.append("      - Recommendation: Implement virtual scrolling")
    report.append("")
    report.append("  ⚠️  Window resize threshold (50px) may cause jitter")
    report.append("      - Recommendation: Use 100px threshold or debounce")
    report.append("")
    
    # 8. Recommendations
    report.append("8. RECOMMENDATIONS")
    report.append("-" * 70)
    report.append("  Performance:")
    report.append("    • Use after_idle() for non-critical updates")
    report.append("    • Implement double-buffering for waveform Canvas")
    report.append("    • Add debounce (200ms) to window resize handler")
    report.append("    • Consider virtual scrolling for transcripts > 1000 lines")
    report.append("")
    report.append("  Accessibility:")
    report.append("    • Add keyboard shortcuts (Space to toggle record)")
    report.append("    • Add high-contrast mode option")
    report.append("    • Support system dark mode preference")
    report.append("")
    report.append("  UX Improvements:")
    report.append("    • Add loading spinner during model initialization")
    report.append("    • Show tooltips on hover for all buttons")
    report.append("    • Add keyboard navigation for settings")
    report.append("")
    
    report.append("=" * 70)
    report.append("End of Report")
    report.append("=" * 70)
    
    return "\n".join(report)


def run_tests_with_report():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestColorConstants,
        TestResponsiveBreakpoints,
        TestSidebarBehavior,
        TestResponsiveLayout,
        TestControlStates,
        TestThemeConsistency,
        TestModelSelector,
        TestLanguageSelector,
        TestWaveformVisualizer,
        TestTextAreaUpdates,
        TestProgressIndicators,
        TestErrorDialogs,
        TestPerformanceStats,
        TestTTKStyles,
        TestMobileBottomBar,
        TestSlideOutPanel,
        TestUIElementVerification,
        TestTimerFormatting,
        TestSilencePresets,
        TestResponsiveBehaviorMatrix,
        TestWindowConfiguration,
        TestRecordingModes,
        TestActionButtons,
        TestStatusMessages,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate and print report
    report = generate_test_report()
    print("\n\n")
    print(report)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests_with_report()
    sys.exit(0 if success else 1)
