"""Unit tests for RoutingProblem.get_power."""
import pytest
import numpy as np
from datetime import datetime
from astropy import units as u

from WeatherRoutingTool.algorithms.genetic.problem import RoutingProblem
from WeatherRoutingTool.routeparams import RouteParams


class TestGetPower:
    """Verify RoutingProblem.get_power produces correct fuel computation."""

    @pytest.fixture
    def routing_problem(self, dummy_dpm_boat, dummy_constraint_list):
        """RoutingProblem with DirectPowerBoat and standard constraints."""
        departure = datetime(2025, 4, 1, 0, 0)
        arrival = datetime(2025, 4, 4, 0, 0)
        speed = 6.0 * u.meter / u.second
        return RoutingProblem(
            departure_time=departure,
            arrival_time=arrival,
            boat=dummy_dpm_boat,
            boat_speed=speed,
            constraint_list=dummy_constraint_list,
        )

    @pytest.fixture
    def simple_route(self):
        """3-waypoint route within test weather data bounds."""
        return np.array([
            [37.0, 1.0, 6.0],
            [37.5, 2.0, 6.0],
            [38.0, 3.0, 6.0],
        ])

    def test_returns_tuple_of_float_and_shipparams(self, routing_problem, simple_route):
        """get_power returns (total_fuel: float, shipparams: ShipParams)."""
        result = routing_problem.get_power(simple_route)
        assert isinstance(result, tuple)
        assert len(result) == 2
        total_fuel, shipparams = result
        assert isinstance(total_fuel, (float, np.floating))

    def test_fuel_is_positive(self, routing_problem, simple_route):
        """Total fuel consumption must be non-negative for any valid route."""
        total_fuel, _ = routing_problem.get_power(simple_route)
        assert total_fuel > 0

    def test_baseline_fuel_value(self, routing_problem, simple_route):
        """Verify fuel matches the known-good baseline (current output frozen).

        Tolerance rationale: fuel computation involves Haversine distances,
        interpolated wind/wave fields, and iterative power integration. The
        pipeline is deterministic on a given platform, so rtol=1e-6 guards
        against regressions while allowing for floating-point scheduling
        differences across compilers/BLAS implementations.
        """
        total_fuel, _ = routing_problem.get_power(simple_route)
        # Baseline recorded on Python 3.12.10, Windows, tests/data/tests_weather_data.nc
        EXPECTED_FUEL = 8417.693240060958  # kg
        np.testing.assert_allclose(total_fuel, EXPECTED_FUEL, rtol=1e-6)

    def test_single_segment_route(self, routing_problem):
        """Minimal route: 2 waypoints, 1 segment."""
        route = np.array([
            [37.0, 1.0, 6.0],
            [37.5, 2.0, 6.0],
        ])
        total_fuel, shipparams = routing_problem.get_power(route)
        assert total_fuel > 0

    def test_longer_route_more_fuel(self, routing_problem, simple_route):
        """A route with more waypoints (longer distance) should consume more fuel."""
        short_route = simple_route[:2]  # 2 waypoints
        fuel_short, _ = routing_problem.get_power(short_route)
        fuel_long, _ = routing_problem.get_power(simple_route)  # 3 waypoints
        assert fuel_long > fuel_short

    def test_speed_from_arrival_time(self, dummy_dpm_boat, dummy_constraint_list):
        """When speed sentinel is -99, speed is derived from arrival time."""
        departure = datetime(2025, 4, 1, 0, 0)
        arrival = datetime(2025, 4, 4, 0, 0)
        sentinel_speed = -99.0 * u.meter / u.second

        problem = RoutingProblem(
            departure_time=departure,
            arrival_time=arrival,
            boat=dummy_dpm_boat,
            boat_speed=sentinel_speed,
            constraint_list=dummy_constraint_list,
        )
        route = np.array([
            [37.0, 1.0, 6.0],
            [38.0, 3.0, 6.0],
        ])
        total_fuel, _ = problem.get_power(route)
        assert total_fuel > 0  # Should compute without error

    def test_identical_waypoints_zero_distance(self, routing_problem):
        """Two identical waypoints produce zero distance — fuel should be zero or near-zero.

        Note: This edge case has not been run yet. If get_power raises on
        zero-length segments, this test will be converted to a
        pytest.raises assertion during Community Bonding.
        """
        route = np.array([
            [37.0, 1.0, 6.0],
            [37.0, 1.0, 6.0],
        ])
        total_fuel, _ = routing_problem.get_power(route)
        assert total_fuel >= 0  # No negative fuel
        np.testing.assert_allclose(total_fuel, 0, atol=1e-3)

    def test_shipparams_has_fuel_rate(self, routing_problem, simple_route):
        """ShipParams returned by get_power must expose get_fuel_rate()."""
        _, shipparams = routing_problem.get_power(simple_route)
        fuel_rate = shipparams.get_fuel_rate()
        assert len(fuel_rate) == len(simple_route) - 1  # N-1 segments
        assert np.all(fuel_rate >= 0)
