"""Reference trajectory primitives for underwater robot control."""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Protocol

import numpy as np


@dataclass
class TrajectoryPoint:
    """Desired vehicle reference at a single time instant."""

    position: np.ndarray
    velocity: np.ndarray
    yaw: float | None = None
    yaw_rate: float = 0.0


class ReferenceTrajectory(Protocol):
    """Callable protocol for reference trajectories."""

    def __call__(self, t: float) -> TrajectoryPoint:
        """Return the trajectory point at time ``t``."""


@dataclass
class ConstantPositionTrajectory:
    """Station-keeping reference."""

    position: np.ndarray
    yaw: float | None = None

    def __call__(self, t: float) -> TrajectoryPoint:  # noqa: ARG002
        p = np.asarray(self.position, dtype=float).reshape(3)
        return TrajectoryPoint(position=p, velocity=np.zeros(3), yaw=self.yaw, yaw_rate=0.0)


@dataclass
class LineTrajectory:
    """Straight-line trajectory with constant velocity."""

    start: np.ndarray
    velocity: np.ndarray
    yaw: float | None = None

    def __call__(self, t: float) -> TrajectoryPoint:
        start = np.asarray(self.start, dtype=float).reshape(3)
        velocity = np.asarray(self.velocity, dtype=float).reshape(3)
        return TrajectoryPoint(position=start + velocity * t, velocity=velocity, yaw=self.yaw, yaw_rate=0.0)


@dataclass
class CircleTrajectory:
    """Horizontal circular reference trajectory."""

    center: np.ndarray
    radius: float
    angular_rate: float
    z: float = 0.0
    yaw_tangent: bool = True

    def __call__(self, t: float) -> TrajectoryPoint:
        center = np.asarray(self.center, dtype=float).reshape(3)
        a = self.angular_rate * t
        position = center + np.array([self.radius * cos(a), self.radius * sin(a), self.z], dtype=float)
        velocity = np.array(
            [-self.radius * self.angular_rate * sin(a), self.radius * self.angular_rate * cos(a), 0.0],
            dtype=float,
        )
        yaw = a + pi / 2.0 if self.yaw_tangent else None
        return TrajectoryPoint(position=position, velocity=velocity, yaw=yaw, yaw_rate=self.angular_rate)
