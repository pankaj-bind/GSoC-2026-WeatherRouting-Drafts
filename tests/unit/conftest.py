import os
import pytest
import numpy as np
from datetime import datetime

from WeatherRoutingTool.config import Config
from WeatherRoutingTool.algorithms.isobased import IsoBased
from WeatherRoutingTool.algorithms.isofuel import IsoFuel
from WeatherRoutingTool.constraints.constraints import ConstraintsList, ConstraintPars
from WeatherRoutingTool.ship.direct_power_boat import DirectPowerBoat
from WeatherRoutingTool.ship.ship_config import ShipConfig


@pytest.fixture(scope="module")
def test_config():
    """Load the standard test configuration."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.tests.json")
    return Config.assign_config(config_path)


@pytest.fixture(scope="module")
def dummy_isobased(test_config):
    """IsoBased routing algorithm initialized from test config.

    module scope is safe: IsoBased.__init__ only reads config values
    and initializes numpy arrays — no data files are loaded.
    """
    return IsoBased(test_config)


@pytest.fixture(scope="module")
def dummy_isofuel(test_config):
    """IsoFuel routing algorithm initialized from test config."""
    return IsoFuel(test_config)


@pytest.fixture(scope="module")
def dummy_constraint_list():
    """Lightweight constraint list with 1/10° resolution."""
    pars = ConstraintPars()
    pars.resolution = 1.0 / 10
    return ConstraintsList(pars)


@pytest.fixture(scope="module")
def dummy_dpm_boat():
    """DirectPowerBoat initialized from simpleship test config + test data."""
    return _create_dpm_boat("simpleship")


@pytest.fixture(scope="module")
def dummy_dpm_boat_manual():
    """DirectPowerBoat initialized from manualship test config + test data."""
    return _create_dpm_boat("manualship")


def _create_dpm_boat(config_variant):
    """Shared builder matching basic_test_func.create_dummy_Direct_Power_Ship."""
    dirname = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(dirname, f"config.tests_{config_variant}.json")
    ship_config = ShipConfig.assign_config(path=config_path)
    boat = DirectPowerBoat(ship_config)
    # Note: weather_path is set in DirectPowerBoat.__init__ from ShipConfig.
    # courses_path and depth_path are monkey-patched by the existing test
    # helpers (basic_test_func.py). We replicate that pattern here to
    # maintain behavioral parity during migration.
    boat.weather_path = os.path.join(dirname, "data", "tests_weather_data.nc")
    boat.courses_path = os.path.join(dirname, "data", "CoursesRouteStatus.nc")
    boat.depth_path = os.path.join(dirname, "data", "tests_depth_data.nc")
    boat.load_data()
    return boat


@pytest.fixture
def sample_route_array():
    """Synthetic 5-waypoint route: [lat, lon, speed_m/s]."""
    return np.array([
        [37.0, 1.0, 6.0],
        [37.5, 2.0, 6.0],
        [38.0, 3.0, 6.0],
        [38.5, 4.0, 6.0],
        [39.0, 5.0, 6.0],
    ])


@pytest.fixture
def sqlite_engine():
    """In-memory SQLite engine with automatic disposal."""
    from sqlalchemy import create_engine
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()
