import copy
import math
import random
from math import radians, cos, sin, asin, sqrt

import numpy as np

from grid_reducer.altdss.altdss_models import Circuit


class BasePrivacyConfig:
    geo_coordinate_noise = None
    non_geo_coordinate_noise = None


class LowPrivacyConfig(BasePrivacyConfig):
    geo_coordinate_noise = 5000
    non_geo_coordinate_noise = 0.01


class MediumPrivacyConfig(BasePrivacyConfig):
    geo_coordinate_noise = 3500
    non_geo_coordinate_noise = 0.05


class HighPrivacyConfig(BasePrivacyConfig):
    geo_coordinate_noise = 2000
    non_geo_coordinate_noise = 0.1


def apply_gaussian_dp_noise(value: float, std_dev: float) -> float:
    noise = np.random.normal(0, std_dev)
    return value + noise


def apply_planar_laplace_noise(x: float, y: float, epsilon: float) -> tuple[float, float]:
    theta = 2 * math.pi * random.random()
    u1, u2 = random.random(), random.random()
    r = -(1 / epsilon) * math.log(u1 * u2)

    # Apply noise in polar coordinates
    x_noisy = x + r * math.cos(theta)
    y_noisy = y + r * math.sin(theta)
    return x_noisy, y_noisy


def is_geo_coordinate(x: float, y: float) -> bool:
    """
    Determines if coordinates are in standard geo-coordinate ranges,
    excluding transformed layout coordinates (like from kamada_kawai_layout).

    Transformed layouts typically produce coordinates in [0,1] or [-1,1] range.
    """
    # Check if coordinates are within standard geo bounds
    geo_bounds = (-180.0 <= x <= 180.0) and (-90.0 <= y <= 90.0)

    # Exclude transformed layout coordinates (typical range: [-1.5, 1.5])
    is_transformed = (-1.5 <= x <= 1.5) and (-1.5 <= y <= 1.5)

    return geo_bounds and not is_transformed


def check_if_all_coords_are_none(circuit: Circuit) -> bool:
    for bus in circuit.Bus:
        if bus.X is not None or bus.Y is not None:
            return False
    return True


def check_if_circuit_is_geo(circuit: Circuit) -> bool:
    for bus in circuit.Bus:
        if bus.X is not None and bus.Y is not None:
            if not is_geo_coordinate(bus.X, bus.Y):
                return False
    return False if check_if_all_coords_are_none(circuit) else True


def get_dp_circuit(circuit: Circuit, noise_config: BasePrivacyConfig) -> Circuit:
    """
    Applies differential privacy to all bus coordinates:
    - Planar Laplace noise for all geo-coordinates (including switch-connected)
    - Gaussian noise for transformed layout coordinates

    Args:
        circuit (Circuit): Original circuit
        noise_level (str): "low", "medium", or "high" noise strength

    Returns:
        Circuit: New circuit with perturbed bus coordinates
    """

    new_buses = []
    is_geo = check_if_circuit_is_geo(circuit)
    for bus in circuit.Bus:
        new_bus = copy.deepcopy(bus)
        if new_bus.X is not None and new_bus.Y is not None:
            if is_geo:
                new_bus.X, new_bus.Y = apply_planar_laplace_noise(
                    new_bus.X, new_bus.Y, int(noise_config.geo_coordinate_noise)
                )
            else:
                noise_scale = float(noise_config.non_geo_coordinate_noise)
                new_bus.X = apply_gaussian_dp_noise(new_bus.X, noise_scale)
                new_bus.Y = apply_gaussian_dp_noise(new_bus.Y, noise_scale)
        new_buses.append(new_bus)

    new_circuit = copy.deepcopy(circuit)
    new_circuit.Bus = new_buses
    return new_circuit


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Returns distance in meters between two (lon, lat) points
    """
    # Earth radius in meters
    R = 6371000
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    distance = R * c
    return distance


def coord_distance(x1: float, y1: float, x2: float, y2: float, is_geo: bool) -> float:
    """Returns the geographic (haversine) or Euclidean distance between two coordinate pairs."""
    if is_geo:
        return haversine(x1, y1, x2, y2)
    else:
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def get_circuit_noisy_distances(
    original_circuit: Circuit, noisy_circuit: Circuit, is_geo: bool = None
):
    """
    Computes the distances between corresponding buses in the original and noisy circuits:
    - Geographic distance in meters if geo-coordinates are used
    - Euclidean distance in coordinate space otherwise

    Args:
        original_circuit (Circuit): Reference circuit with original bus coordinates
        noisy_circuit (Circuit): Perturbed circuit with noisy bus coordinates
        is_geo (bool, optional): If True, use geographic (meter) distances; otherwise, use coordinate space.
            If None, inferred from original_circuit.

    Returns:
        list[float]: List of distances between each matching pair of buses. Only buses with valid coordinates in both
        circuits are included.
    """
    if is_geo is None:
        is_geo = check_if_circuit_is_geo(original_circuit)
    dists = []
    for orig_bus, noisy_bus in zip(original_circuit.Bus, noisy_circuit.Bus, strict=False):
        if (
            orig_bus.X is not None
            and orig_bus.Y is not None
            and noisy_bus.X is not None
            and noisy_bus.Y is not None
        ):
            d = coord_distance(orig_bus.X, orig_bus.Y, noisy_bus.X, noisy_bus.Y, is_geo)
            dists.append(d)
    return dists
