"""PID trajectory tracking demo for the underwater-robot package.

The controller is intentionally lightweight: a position PID produces a desired
world-frame velocity, a yaw PID aligns the body with the path, and a heuristic
mapper converts this high-level command to left/right oscillating-fin gait
parameters.
"""

from math import pi

from underwater_robot import (
    LineTrajectory,
    OscillatingFinGaitMapper,
    OscillatingFinRobot,
    PIDGains,
    SimulationConfig,
    TrajectoryTrackingController,
    UnderwaterRobotParameters,
    simulate_trajectory_tracking,
)
from underwater_robot.visualization import plot_position


def main(show: bool = True):
    params = UnderwaterRobotParameters(hydrodynamic_quadrature_order=8)
    model = OscillatingFinRobot(params)

    trajectory = LineTrajectory(start=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.08], yaw=0.0)
    mapper = OscillatingFinGaitMapper(
        base_omega=pi,
        base_beta=pi / 3.0,
        forward_to_omega=1.2,
        yaw_to_omega_delta=0.6,
    )
    controller = TrajectoryTrackingController(
        trajectory=trajectory,
        position_gains=PIDGains(kp=(0.2, 0.2, 0.4), ki=0.0, kd=(0.02, 0.02, 0.04)),
        yaw_gains=PIDGains(kp=0.5, ki=0.0, kd=0.02),
        gait_mapper=mapper,
        forward_axis=2,
        velocity_limits=(-0.4, 0.4),
        yaw_rate_limits=(-0.8, 0.8),
    )
    config = SimulationConfig(tf=0.10, dt=0.02, max_step=0.01, initial_velocity_body=(0.0, 0.0, 0.1))
    result = simulate_trajectory_tracking(model=model, controller=controller, config=config)

    print("success:", result.success)
    print("final position:", result.position[:, -1])
    print("final reference:", result.reference_position[:, -1])
    print("final tracking error:", result.position_error[:, -1])
    print("last gait [omega_l, omega_r, beta_l, beta_r, amplitude]:", result.gait_history[:, -1])

    if show:
        plot_position(result.time, result.position, result.reference_position, title="Closed-loop trajectory tracking", show=show)
    return result


if __name__ == "__main__":
    main(show=True)
