import json
import os
import pytest
import numpy as np
from pathlib import Path

from WeatherRoutingTool.config import Config
from WeatherRoutingTool.ship.ship_factory import ShipFactory
from WeatherRoutingTool.ship.ship_config import ShipConfig
from WeatherRoutingTool.weather_factory import WeatherFactory
from WeatherRoutingTool.constraints.constraints import ConstraintsListFactory, WaterDepth
from WeatherRoutingTool.algorithms.routingalg_factory import RoutingAlgFactory
from WeatherRoutingTool.utils.maps import Map


REFERENCE_DIR = Path(__file__).parent / "reference_data"


@pytest.fixture(scope="module")
def genetic_config():
    """Configuration for genetic regression test — small region, low generations."""
    config_path = Path(__file__).parent.parent / "config_regression_genetic.json"
    return Config.assign_config(str(config_path))


@pytest.fixture(scope="module")
def genetic_route(genetic_config):
    """
    Execute the genetic algorithm once per module and return the RouteParams.

    Mirrors the wiring in execute_routing.py but returns the route object
    instead of writing it to disk.
    """
    default_map = Map(*genetic_config.DEFAULT_MAP)
    departure_time = genetic_config.DEPARTURE_TIME

    wt = WeatherFactory.get_weather(
        genetic_config._DATA_MODE_WEATHER,
        genetic_config.WEATHER_DATA,
        departure_time,
        genetic_config.TIME_FORECAST,
        genetic_config.DELTA_TIME_FORECAST,
        default_map,
    )
    ship_config = ShipConfig.assign_config(path=str(genetic_config.CONFIG_PATH))
    boat = ShipFactory.get_ship(genetic_config.BOAT_TYPE, ship_config)

    water_depth = WaterDepth(
        genetic_config._DATA_MODE_DEPTH,
        boat.get_required_water_depth(),
        default_map,
        genetic_config.DEPTH_DATA,
    )
    constraint_list = ConstraintsListFactory.get_constraints_list(
        constraints_string_list=genetic_config.CONSTRAINTS_LIST,
        data_mode=genetic_config._DATA_MODE_DEPTH,
        min_depth=boat.get_required_water_depth(),
        map_size=default_map,
        depthfile=genetic_config.DEPTH_DATA,
        waypoints=genetic_config.INTERMEDIATE_WAYPOINTS,
        courses_path=genetic_config.COURSES_FILE,
    )

    alg = RoutingAlgFactory.get_routing_alg(genetic_config)
    # init_fig is a no-op for Genetic (inherited pass from RoutingAlg),
    # but called here to match the execute_routing.py contract.
    alg.init_fig(water_depth=water_depth, map_size=default_map)

    # alg.execute_routing returns (RouteParams, error_code)
    min_fuel_route, error_code = alg.execute_routing(boat, wt, constraint_list)
    return min_fuel_route


@pytest.mark.integration
class TestGeneticRegression:
    """Verify genetic algorithm produces identical routes with fixed seed."""

    def test_route_object_valid(self, genetic_route):
        """Sanity: the returned RouteParams has expected structure."""
        assert genetic_route is not None
        assert genetic_route.count > 0
        assert genetic_route.get_full_fuel() > 0

    def test_waypoints_reproduce_exactly(self, genetic_route):
        """Waypoint arrays must match the committed reference GeoJSON exactly."""
        ref_geojson = json.loads(
            (REFERENCE_DIR / "genetic_route.geojson").read_text()
        )
        ref_coords = ref_geojson["features"][0]["geometry"]["coordinates"]
        ref_lons = [c[0] for c in ref_coords]
        ref_lats = [c[1] for c in ref_coords]

        np.testing.assert_array_equal(
            genetic_route.lats_per_step.flatten(), ref_lats
        )
        np.testing.assert_array_equal(
            genetic_route.lons_per_step.flatten(), ref_lons
        )

    def test_fuel_reproduces_exactly(self, genetic_route):
        """Total fuel must match the committed reference scalar."""
        ref_summary = json.loads(
            (REFERENCE_DIR / "genetic_summary.json").read_text()
        )
        assert genetic_route.get_full_fuel() == ref_summary["total_fuel_kg"]

    def test_route_count_matches(self, genetic_route):
        """Number of routing steps must match reference."""
        ref_summary = json.loads(
            (REFERENCE_DIR / "genetic_summary.json").read_text()
        )
        assert genetic_route.count == ref_summary["waypoint_count"]
