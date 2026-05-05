from math import pi

import numpy as np

from underwater_robot import (
    LineTrajectory,
    OscillatingFinGaitMapper,
    OscillatingFinRobot,
    PIDController,
    PIDGains,
    SimulationConfig,
    TrajectoryTrackingController,
    UnderwaterRobotParameters,
    simulate_trajectory_tracking,
)


def test_pid_controller_basic_response():
    pid = PIDController(PIDGains(kp=2.0, ki=0.5, kd=0.0), size=1, output_limits=(-10.0, 10.0))
    u1 = pid.update(np.array([1.0]), dt=0.1)
    u2 = pid.update(np.array([1.0]), dt=0.1)
    assert u1.shape == (1,)
    assert u2[0] > u1[0]
    assert np.isfinite(u2[0])


def test_gait_mapper_produces_bounded_gait():
    mapper = OscillatingFinGaitMapper(base_omega=pi, omega_limits=(0.5, 4.0))
    gait = mapper.map(command=__import__("underwater_robot").BodyVelocityCommand(forward_speed=10.0, yaw_rate=10.0))
    assert 0.5 <= gait.omega_left <= 4.0
    assert 0.5 <= gait.omega_right <= 4.0
    assert -pi <= gait.beta_left <= pi


def test_closed_loop_tracking_smoke():
    model = OscillatingFinRobot(UnderwaterRobotParameters(hydrodynamic_quadrature_order=3))
    trajectory = LineTrajectory(start=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.02], yaw=0.0)
    controller = TrajectoryTrackingController(trajectory=trajectory, velocity_limits=(-0.2, 0.2), yaw_rate_limits=(-0.5, 0.5))
    config = SimulationConfig(tf=0.02, dt=0.02, max_step=0.01, initial_velocity_body=(0.0, 0.0, 0.02))
    result = simulate_trajectory_tracking(model, controller, config)
    assert result.success
    assert result.state.shape[0] == 13
    assert result.reference_position.shape == result.position.shape
    assert result.gait_history.shape[0] == 5
    assert np.all(np.isfinite(result.state))
