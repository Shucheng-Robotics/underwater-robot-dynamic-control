"""Closed-loop trajectory tracking controllers."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, pi
from typing import Any

import numpy as np

from ..math_utils import euler_zyx_from_quat_wxyz, quat_wxyz_to_rot
from ..trajectories import ReferenceTrajectory, TrajectoryPoint
from .gait_mapping import BodyVelocityCommand, OscillatingFinGaitMapper
from .pid import PIDController, PIDGains


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to the interval ``[-pi, pi)``."""
    return float((angle + pi) % (2.0 * pi) - pi)


@dataclass
class TrackingDiagnostics:
    """Diagnostics returned by a trajectory tracking controller."""

    reference: TrajectoryPoint
    position_error_world: np.ndarray
    velocity_command_world: np.ndarray
    velocity_command_body: np.ndarray
    yaw_error: float
    body_command: BodyVelocityCommand


class TrajectoryTrackingController:
    """PID-based trajectory tracker for oscillating-fin underwater robots.

    The controller computes a desired world-frame velocity from position error,
    rotates it into the body frame, extracts a forward-speed command, computes a
    yaw-rate command, and maps these high-level commands to fin gait parameters.
    The mapping is intentionally configurable because oscillating-fin vehicles
    often require robot-specific calibration.
    """

    def __init__(
        self,
        trajectory: ReferenceTrajectory,
        position_gains: PIDGains | None = None,
        yaw_gains: PIDGains | None = None,
        gait_mapper: OscillatingFinGaitMapper | None = None,
        forward_axis: int = 2,
        velocity_limits: tuple[float, float] = (-1.0, 1.0),
        yaw_rate_limits: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        if forward_axis not in (0, 1, 2):
            raise ValueError("forward_axis must be 0, 1 or 2.")
        self.trajectory = trajectory
        self.position_pid = PIDController(
            gains=position_gains or PIDGains(kp=(0.4, 0.4, 0.4), ki=0.0, kd=(0.05, 0.05, 0.05)),
            size=3,
            output_limits=(velocity_limits[0], velocity_limits[1]),
            integral_limits=(-2.0, 2.0),
        )
        self.yaw_pid = PIDController(
            gains=yaw_gains or PIDGains(kp=0.8, ki=0.0, kd=0.05),
            size=1,
            output_limits=(yaw_rate_limits[0], yaw_rate_limits[1]),
            integral_limits=(-1.0, 1.0),
        )
        self.gait_mapper = gait_mapper or OscillatingFinGaitMapper()
        self.forward_axis = forward_axis
        self.last_diagnostics: TrackingDiagnostics | None = None

    def reset(self) -> None:
        """Reset all internal PID memory."""
        self.position_pid.reset()
        self.yaw_pid.reset()
        self.last_diagnostics = None

    def compute(self, t: float, x: np.ndarray, dt: float) -> tuple[Any, TrackingDiagnostics]:
        """Compute a gait command and diagnostics from the current state."""
        state = np.asarray(x, dtype=float).reshape(13)
        reference = self.trajectory(float(t))
        position = state[0:3]
        R_wb = quat_wxyz_to_rot(state[3:7])

        position_error = np.asarray(reference.position, dtype=float).reshape(3) - position
        velocity_ff = np.asarray(reference.velocity, dtype=float).reshape(3)
        velocity_fb = self.position_pid.update(position_error, dt)
        velocity_cmd_world = velocity_ff + velocity_fb
        velocity_cmd_body = R_wb.T @ velocity_cmd_world

        current_yaw = float(euler_zyx_from_quat_wxyz(state[3:7])[0])
        if reference.yaw is None:
            desired_yaw = atan2(velocity_cmd_world[1], velocity_cmd_world[0]) if np.linalg.norm(velocity_cmd_world[:2]) > 1e-9 else current_yaw
        else:
            desired_yaw = float(reference.yaw)
        yaw_error = wrap_to_pi(desired_yaw - current_yaw)
        yaw_rate_fb = float(self.yaw_pid.update(np.array([yaw_error]), dt)[0])

        vertical_speed = float(velocity_cmd_world[2])
        body_command = BodyVelocityCommand(
            forward_speed=float(velocity_cmd_body[self.forward_axis]),
            yaw_rate=float(reference.yaw_rate + yaw_rate_fb),
            vertical_speed=vertical_speed,
        )
        gait = self.gait_mapper.map(body_command)
        diagnostics = TrackingDiagnostics(
            reference=reference,
            position_error_world=position_error,
            velocity_command_world=velocity_cmd_world,
            velocity_command_body=velocity_cmd_body,
            yaw_error=yaw_error,
            body_command=body_command,
        )
        self.last_diagnostics = diagnostics
        return gait, diagnostics
