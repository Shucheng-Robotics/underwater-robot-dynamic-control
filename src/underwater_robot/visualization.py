"""Plotting helpers for underwater robot simulation results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .math_utils import euler_zyx_from_quat_wxyz
from .simulation import SimulationResult


def euler_history_zyx(result: SimulationResult, degrees: bool = True) -> np.ndarray:
    """Return yaw-pitch-roll history from a simulation result."""
    euler = np.zeros((result.state.shape[1], 3))
    for k in range(result.state.shape[1]):
        euler[k, :] = euler_zyx_from_quat_wxyz(result.state[3:7, k], degrees=degrees)
    return euler


def plot_state_history(result: SimulationResult, show: bool = True):
    """Plot position, attitude, linear velocity and angular velocity."""
    t = result.time
    euler = euler_history_zyx(result, degrees=True)
    fig, axes = plt.subplots(4, 3, figsize=(12, 9), constrained_layout=True)

    labels = ["x [m]", "y [m]", "z [m]"]
    for i in range(3):
        axes[0, i].plot(t, result.position[i, :])
        axes[0, i].set_ylabel(labels[i])
        axes[0, i].grid(True)

    labels = ["yaw [deg]", "pitch [deg]", "roll [deg]"]
    for i in range(3):
        axes[1, i].plot(t, euler[:, i])
        axes[1, i].set_ylabel(labels[i])
        axes[1, i].grid(True)

    labels = ["vx [m/s]", "vy [m/s]", "vz [m/s]"]
    for i in range(3):
        axes[2, i].plot(t, result.velocity_body[i, :])
        axes[2, i].set_ylabel(labels[i])
        axes[2, i].grid(True)

    labels = ["wx [rad/s]", "wy [rad/s]", "wz [rad/s]"]
    for i in range(3):
        axes[3, i].plot(t, result.angular_velocity_body[i, :])
        axes[3, i].set_ylabel(labels[i])
        axes[3, i].set_xlabel("time [s]")
        axes[3, i].grid(True)

    if show:
        plt.show()
    return fig, axes


def plot_position(time: np.ndarray, position: np.ndarray, reference_position: np.ndarray | None = None, title: str = "Position", show: bool = True):
    """Plot 3D position components, optionally against a reference."""
    t = np.asarray(time, dtype=float)
    p = np.asarray(position, dtype=float)
    if p.shape[0] != 3:
        raise ValueError("position must have shape (3, N).")
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), constrained_layout=True)
    fig.suptitle(title)
    labels = ["x [m]", "y [m]", "z [m]"]
    ref = None if reference_position is None else np.asarray(reference_position, dtype=float)
    for i, ax in enumerate(axes):
        ax.plot(t, p[i, :], label="actual")
        if ref is not None:
            ax.plot(t, ref[i, :], linestyle="--", label="reference")
        ax.set_ylabel(labels[i])
        ax.grid(True)
    axes[-1].set_xlabel("time [s]")
    if ref is not None:
        axes[0].legend()
    if show:
        plt.show()
    return fig, axes
