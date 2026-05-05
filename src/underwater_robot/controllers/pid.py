"""PID controllers for underwater robot closed-loop simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


def _as_vector(value: float | Iterable[float] | np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(size, float(arr))
    arr = arr.reshape(-1)
    if arr.size != size:
        raise ValueError(f"Expected a scalar or a vector of length {size}, got shape {arr.shape}.")
    return arr.astype(float)


@dataclass
class PIDGains:
    """Vector PID gains.

    Each gain can be a scalar or a vector. Scalars are broadcast to every
    controlled channel.
    """

    kp: float | Iterable[float] | np.ndarray = 0.0
    ki: float | Iterable[float] | np.ndarray = 0.0
    kd: float | Iterable[float] | np.ndarray = 0.0


@dataclass
class PIDController:
    """Discrete-time vector PID controller with anti-windup.

    The controller computes

    ``u = kp * e + ki * integral(e) + kd * de/dt``

    and clips both the integral state and the final output.  It is intentionally
    independent of any vehicle model, so it can be used for position, depth,
    yaw, station-keeping, or low-level actuator tests.
    """

    gains: PIDGains
    size: int
    output_limits: tuple[float | np.ndarray | None, float | np.ndarray | None] = (None, None)
    integral_limits: tuple[float | np.ndarray | None, float | np.ndarray | None] = (None, None)
    derivative_on_measurement: bool = False
    _integral: np.ndarray = field(init=False, repr=False)
    _previous_error: np.ndarray | None = field(default=None, init=False, repr=False)
    _previous_measurement: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.kp = _as_vector(self.gains.kp, self.size)
        self.ki = _as_vector(self.gains.ki, self.size)
        self.kd = _as_vector(self.gains.kd, self.size)
        self._integral = np.zeros(self.size, dtype=float)

    @property
    def integral(self) -> np.ndarray:
        """Return a copy of the current integral state."""
        return self._integral.copy()

    def reset(self) -> None:
        """Reset integral and derivative memory."""
        self._integral[:] = 0.0
        self._previous_error = None
        self._previous_measurement = None

    def _clip(self, value: np.ndarray, limits: tuple[float | np.ndarray | None, float | np.ndarray | None]) -> np.ndarray:
        lower, upper = limits
        if lower is not None:
            value = np.maximum(value, _as_vector(lower, self.size))
        if upper is not None:
            value = np.minimum(value, _as_vector(upper, self.size))
        return value

    def update(
        self,
        error: np.ndarray,
        dt: float,
        measurement: np.ndarray | None = None,
        reset_derivative: bool = False,
    ) -> np.ndarray:
        """Return the PID output for one discrete control update.

        Parameters
        ----------
        error:
            Current control error ``reference - measurement``.
        dt:
            Positive sample time in seconds.
        measurement:
            Optional measured value. Required only when
            ``derivative_on_measurement=True``.
        reset_derivative:
            If true, suppresses the derivative kick for this update.
        """
        if dt <= 0.0:
            raise ValueError("PIDController.update requires a positive dt.")
        error = np.asarray(error, dtype=float).reshape(self.size)

        self._integral += error * dt
        self._integral = self._clip(self._integral, self.integral_limits)

        if reset_derivative:
            derivative = np.zeros(self.size, dtype=float)
        elif self.derivative_on_measurement:
            if measurement is None:
                raise ValueError("measurement must be provided when derivative_on_measurement=True.")
            measurement = np.asarray(measurement, dtype=float).reshape(self.size)
            if self._previous_measurement is None:
                derivative = np.zeros(self.size, dtype=float)
            else:
                derivative = -(measurement - self._previous_measurement) / dt
            self._previous_measurement = measurement.copy()
        else:
            if self._previous_error is None:
                derivative = np.zeros(self.size, dtype=float)
            else:
                derivative = (error - self._previous_error) / dt
            self._previous_error = error.copy()

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return self._clip(output, self.output_limits)
