"""Rigid-body math utilities for underwater robotics.

The module intentionally keeps dependencies light and focuses on reusable
operations needed by multibody and marine-robot dynamics code: skew matrices,
quaternion normalization, rotation conversion, and quaternion kinematics.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def skew(v: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix such that ``skew(v) @ w = v × w``."""
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=float,
    )


def normalize_quaternion(q_wxyz: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a quaternion in ``[w, x, y, z]`` convention.

    Invalid or near-zero quaternions are mapped to the identity quaternion.  This
    makes long ODE integrations more robust when small numerical drift appears.
    """
    q = np.asarray(q_wxyz, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_wxyz_to_rot(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert a ``[w, x, y, z]`` quaternion to a rotation matrix."""
    q = normalize_quaternion(q_wxyz)
    w, x, y, z = q
    return np.array(
        [
            [2 * (w**2 + x**2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 2 * (w**2 + y**2) - 1, 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w**2 + z**2) - 1],
        ],
        dtype=float,
    )


def rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a rotation matrix to a ``[w, x, y, z]`` quaternion."""
    q_xyzw = Rotation.from_matrix(np.asarray(R, dtype=float)).as_quat()
    return normalize_quaternion(np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]))


def quat_derivative_wxyz(q_wxyz: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Return quaternion derivative for body-frame angular velocity.

    The state convention is ``x = [p_wb, q_wxyz, v_b, omega_b]`` where ``q_wxyz``
    maps body-frame coordinates to world-frame coordinates.
    """
    q = normalize_quaternion(q_wxyz)
    w, x, y, z = q
    G = np.array([[-x, w, z, -y], [-y, -z, w, x], [-z, y, -x, w]], dtype=float)
    return 0.5 * (G.T @ np.asarray(omega_body, dtype=float).reshape(3))


def euler_zyx_from_quat_wxyz(q_wxyz: np.ndarray, degrees: bool = False) -> np.ndarray:
    """Return yaw-pitch-roll Euler angles from a ``[w, x, y, z]`` quaternion."""
    R = quat_wxyz_to_rot(q_wxyz)
    return Rotation.from_matrix(R).as_euler("ZYX", degrees=degrees)


def rotation_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return world-from-body rotation for roll, pitch, yaw angles."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ],
        dtype=float,
    )
