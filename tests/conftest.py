"""
Pytest configuration for Qwen3-ASR tests
"""

import pytest

def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers based on test name patterns
    for item in items:
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)
