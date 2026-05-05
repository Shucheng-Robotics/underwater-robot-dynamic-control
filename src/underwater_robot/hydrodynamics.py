"""Reusable hydrodynamic force helpers for underwater robotics."""

from __future__ import annotations

import numpy as np


def quadratic_body_drag(
    velocity_body: np.ndarray,
    projected_area: np.ndarray,
    drag_coefficients: np.ndarray,
    density: float = 1000.0,
) -> np.ndarray:
    """Return a dissipative diagonal quadratic drag force in the body frame.

    The formulation is component-wise but scaled by the total speed, which
    guarantees ``velocity_body.T @ force <= 0`` for non-negative areas and drag
    coefficients.
    """
    v = np.asarray(velocity_body, dtype=float).reshape(3)
    speed = np.linalg.norm(v)
    if speed < 1e-12:
        return np.zeros(3)
    area = np.asarray(projected_area, dtype=float).reshape(3)
    cd = np.asarray(drag_coefficients, dtype=float).reshape(3)
    return -0.5 * density * speed * area * cd * v


def normal_quadratic_drag(
    velocity: np.ndarray,
    normal: np.ndarray,
    coefficients: np.ndarray,
    density: float = 1000.0,
    area_scale: float = 1.0,
) -> np.ndarray:
    """Return Morison-style drag from velocity projected along a normal vector."""
    v = np.asarray(velocity, dtype=float).reshape(3)
    n = np.asarray(normal, dtype=float).reshape(3)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        return np.zeros(3)
    n = n / n_norm
    coeff = np.asarray(coefficients, dtype=float).reshape(3)
    vn = np.dot(v, n) * n
    return -0.5 * density * coeff * np.linalg.norm(vn) * vn * area_scale
