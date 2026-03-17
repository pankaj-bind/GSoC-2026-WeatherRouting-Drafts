import pytest
import os

# Marker registration
def pytest_configure(config):
    config.addinivalue_line("markers", "monitoring: produces figures (not in CI)")
    config.addinivalue_line("markers", "integration: end-to-end regression tests")
    config.addinivalue_line("markers", "genetic: genetic algorithm tests")
    config.addinivalue_line("markers", "maripower: requires maripower package")


@pytest.fixture(scope="session")
def test_data_dir():
    """Absolute path to the shared test data directory."""
    return os.path.join(os.path.dirname(__file__), "data")
