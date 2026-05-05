"""Closed-loop simulation utilities for underwater robot controllers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .config import SimulationConfig, UnderwaterRobotParameters
from .controllers import TrajectoryTrackingController
from .math_utils import normalize_quaternion
from .models import OscillatingFinRobot
from .simulation import SimulationResult, make_initial_state


@dataclass
class ClosedLoopSimulationResult(SimulationResult):
    """Simulation result with controller histories."""

    reference_position: np.ndarray
    position_error: np.ndarray
    velocity_command_world: np.ndarray
    velocity_command_body: np.ndarray
    body_command: np.ndarray
    gait_history: np.ndarray
    yaw_error: np.ndarray


def simulate_trajectory_tracking(
    model: OscillatingFinRobot | None = None,
    controller: TrajectoryTrackingController | None = None,
    config: SimulationConfig | None = None,
) -> ClosedLoopSimulationResult:
    """Run sample-and-hold closed-loop trajectory tracking.

    The controller is evaluated once per ``config.dt``.  The resulting gait is
    held constant while the continuous dynamics are integrated over the next
    sample interval.  This avoids corrupting PID integral state with the many
    internal function evaluations performed by adaptive ODE solvers.
    """
    if controller is None:
        raise ValueError("simulate_trajectory_tracking requires a TrajectoryTrackingController.")

    model = model or OscillatingFinRobot(UnderwaterRobotParameters(hydrodynamic_quadrature_order=20))
    config = config or SimulationConfig(tf=1.0, dt=0.05, max_step=0.01)
    time = config.time_vector
    if time.size < 2:
        raise ValueError("SimulationConfig must contain at least two time samples.")

    x = make_initial_state(config)
    state = np.zeros((13, time.size), dtype=float)
    state[:, 0] = x

    reference_position = np.zeros((3, time.size), dtype=float)
    position_error = np.zeros((3, time.size), dtype=float)
    velocity_command_world = np.zeros((3, time.size), dtype=float)
    velocity_command_body = np.zeros((3, time.size), dtype=float)
    body_command = np.zeros((3, time.size), dtype=float)  # [forward_speed, yaw_rate, vertical_speed]
    gait_history = np.zeros((5, time.size), dtype=float)  # [omega_l, omega_r, beta_l, beta_r, yaw_amplitude]
    yaw_error = np.zeros(time.size, dtype=float)

    success = True
    message = "The closed-loop integration completed successfully."
    max_step = config.max_step if config.max_step is not None else config.dt

    for k in range(time.size - 1):
        tk = float(time[k])
        dt = float(time[k + 1] - time[k])
        gait, diag = controller.compute(tk, x, dt)

        reference_position[:, k] = np.asarray(diag.reference.position, dtype=float).reshape(3)
        position_error[:, k] = diag.position_error_world
        velocity_command_world[:, k] = diag.velocity_command_world
        velocity_command_body[:, k] = diag.velocity_command_body
        body_command[:, k] = [diag.body_command.forward_speed, diag.body_command.yaw_rate, diag.body_command.vertical_speed]
        gait_history[:, k] = [gait.omega_left, gait.omega_right, gait.beta_left, gait.beta_right, gait.yaw_amplitude]
        yaw_error[k] = diag.yaw_error

        sol = solve_ivp(
            lambda t, x_local: model.state_derivative(t, x_local, gait),
            (tk, float(time[k + 1])),
            x,
            t_eval=[float(time[k + 1])],
            rtol=config.solver_rtol,
            atol=config.solver_atol,
            max_step=max_step,
        )
        if not sol.success:
            success = False
            message = str(sol.message)
            state = state[:, : k + 1]
            time = time[: k + 1]
            reference_position = reference_position[:, : k + 1]
            position_error = position_error[:, : k + 1]
            velocity_command_world = velocity_command_world[:, : k + 1]
            velocity_command_body = velocity_command_body[:, : k + 1]
            body_command = body_command[:, : k + 1]
            gait_history = gait_history[:, : k + 1]
            yaw_error = yaw_error[: k + 1]
            break

        x = sol.y[:, -1]
        x[3:7] = normalize_quaternion(x[3:7])
        state[:, k + 1] = x

    # Populate final history sample for easier plotting.
    if success:
        gait, diag = controller.compute(float(time[-1]), x, float(config.dt))
        reference_position[:, -1] = np.asarray(diag.reference.position, dtype=float).reshape(3)
        position_error[:, -1] = diag.position_error_world
        velocity_command_world[:, -1] = diag.velocity_command_world
        velocity_command_body[:, -1] = diag.velocity_command_body
        body_command[:, -1] = [diag.body_command.forward_speed, diag.body_command.yaw_rate, diag.body_command.vertical_speed]
        gait_history[:, -1] = [gait.omega_left, gait.omega_right, gait.beta_left, gait.beta_right, gait.yaw_amplitude]
        yaw_error[-1] = diag.yaw_error

    return ClosedLoopSimulationResult(
        time=time,
        state=state,
        success=success,
        message=message,
        reference_position=reference_position,
        position_error=position_error,
        velocity_command_world=velocity_command_world,
        velocity_command_body=velocity_command_body,
        body_command=body_command,
        gait_history=gait_history,
        yaw_error=yaw_error,
    )
