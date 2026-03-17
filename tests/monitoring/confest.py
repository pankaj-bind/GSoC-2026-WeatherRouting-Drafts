import pytest
import os


def pytest_addoption(parser):
    parser.addoption(
        "--monitoring-output",
        default="./monitoring_output",
        help="Directory for monitoring figure output",
    )


@pytest.fixture
def monitoring_output_dir(request):
    """Directory where monitoring tests save their figures."""
    output = request.config.getoption("--monitoring-output")
    os.makedirs(output, exist_ok=True)
    return output
