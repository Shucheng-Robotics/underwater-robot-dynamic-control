"""Simulation utilities for underwater robot ODE models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .config import OscillatingFinGait, SimulationConfig
from .math_utils import normalize_quaternion
from .models import OscillatingFinRobot


@dataclass
class SimulationResult:
    """Container returned by :func:`simulate_free_swimming`."""

    time: np.ndarray
    state: np.ndarray
    success: bool
    message: str

    @property
    def position(self) -> np.ndarray:
        return self.state[0:3, :]

    @property
    def quaternion_wxyz(self) -> np.ndarray:
        return self.state[3:7, :]

    @property
    def velocity_body(self) -> np.ndarray:
        return self.state[7:10, :]

    @property
    def angular_velocity_body(self) -> np.ndarray:
        return self.state[10:13, :]


def make_initial_state(config: SimulationConfig) -> np.ndarray:
    """Build a 13-state initial condition from a simulation config."""
    x0 = np.zeros(13, dtype=float)
    x0[0:3] = np.asarray(config.initial_position, dtype=float)
    x0[3:7] = normalize_quaternion(np.asarray(config.initial_quaternion_wxyz, dtype=float))
    x0[7:10] = np.asarray(config.initial_velocity_body, dtype=float)
    x0[10:13] = np.asarray(config.initial_angular_velocity_body, dtype=float)
    return x0


def simulate_free_swimming(
    model: OscillatingFinRobot | None = None,
    gait: OscillatingFinGait | None = None,
    config: SimulationConfig | None = None,
) -> SimulationResult:
    """Simulate free-swimming rigid-body dynamics.

    This wrapper is model-agnostic as long as ``model.state_derivative(t, x,
    gait)`` follows the 13-state convention used in this package.
    """
    model = model or OscillatingFinRobot()
    gait = gait or OscillatingFinGait()
    config = config or SimulationConfig()
    t_eval = config.time_vector
    max_step = config.max_step if config.max_step is not None else config.dt

    sol = solve_ivp(
        lambda t, x: model.state_derivative(t, x, gait),
        (config.t0, config.tf),
        make_initial_state(config),
        t_eval=t_eval,
        rtol=config.solver_rtol,
        atol=config.solver_atol,
        max_step=max_step,
    )

    state = sol.y.copy()
    if state.size:
        for k in range(state.shape[1]):
            state[3:7, k] = normalize_quaternion(state[3:7, k])
    return SimulationResult(time=sol.t, state=state, success=bool(sol.success), message=str(sol.message))


# Backward-compatible alias used by the previous package revision.
simulate_free_swim = simulate_free_swimming
