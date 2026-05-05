"""Map high-level control commands to oscillating-fin gait parameters."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from ..config import OscillatingFinGait


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return float(np.clip(value, bounds[0], bounds[1]))


@dataclass
class BodyVelocityCommand:
    """High-level body-frame command used by the gait mapper.

    ``forward_speed`` is expressed along the configured robot forward axis;
    ``yaw_rate`` is positive for counter-clockwise turning in the world Z-axis
    convention; ``vertical_speed`` provides a simple depth/pitch modulation term.
    """

    forward_speed: float = 0.0
    yaw_rate: float = 0.0
    vertical_speed: float = 0.0


@dataclass
class OscillatingFinGaitMapper:
    """Heuristic gait mapper for two-fin underwater robots.

    This mapper is not intended to be a calibrated low-level controller.  It is
    a practical software bridge between trajectory-control outputs and the
    prescribed gait interface used by :class:`OscillatingFinRobot`.
    """

    base_omega: float = pi
    base_beta: float = pi / 3.0
    base_yaw_amplitude: float = 40.0 * pi / 180.0
    yaw_phase: float = 280.0 * pi / 360.0
    forward_to_omega: float = 1.5
    yaw_to_omega_delta: float = 0.8
    vertical_to_beta_delta: float = 0.4
    omega_limits: tuple[float, float] = (0.2, 8.0)
    beta_limits: tuple[float, float] = (-pi, pi)
    yaw_amplitude_limits: tuple[float, float] = (5.0 * pi / 180.0, 70.0 * pi / 180.0)

    def map(self, command: BodyVelocityCommand) -> OscillatingFinGait:
        """Convert a body velocity/yaw command to an oscillating-fin gait."""
        omega_center = self.base_omega + self.forward_to_omega * float(command.forward_speed)
        omega_delta = self.yaw_to_omega_delta * float(command.yaw_rate)
        omega_left = _clip(omega_center - omega_delta, self.omega_limits)
        omega_right = _clip(omega_center + omega_delta, self.omega_limits)

        beta_delta = self.vertical_to_beta_delta * float(command.vertical_speed)
        beta_left = _clip(self.base_beta + beta_delta, self.beta_limits)
        beta_right = _clip(self.base_beta + beta_delta, self.beta_limits)

        # Keep oscillation amplitude positive and bounded.  A small increase with
        # command magnitude makes the example visibly respond to tracking error.
        amplitude = _clip(
            self.base_yaw_amplitude + 0.05 * abs(command.forward_speed),
            self.yaw_amplitude_limits,
        )

        return OscillatingFinGait(
            omega_left=omega_left,
            omega_right=omega_right,
            beta_left=beta_left,
            beta_right=beta_right,
            yaw_amplitude=amplitude,
            yaw_phase=self.yaw_phase,
        )
