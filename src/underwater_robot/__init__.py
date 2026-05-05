"""Underwater Robot: reusable hydrodynamic dynamics tools for marine robotics."""

from .config import (
    GaitParameters,
    MantaParameters,
    OscillatingFinGait,
    SimulationConfig,
    UnderwaterRobotParameters,
)
from .math_utils import (
    euler_zyx_from_quat_wxyz,
    normalize_quaternion,
    quat_derivative_wxyz,
    quat_wxyz_to_rot,
    rot_to_quat_wxyz,
    rotation_zyx,
    skew,
)
from .hydrodynamics import normal_quadratic_drag, quadratic_body_drag
from .models import FinSliceState, MantaRayModel, OscillatingFinRobot, UnderwaterRobotModel
from .simulation import SimulationResult, make_initial_state, simulate_free_swim, simulate_free_swimming
from .controllers import (
    BodyVelocityCommand,
    OscillatingFinGaitMapper,
    PIDController,
    PIDGains,
    TrajectoryTrackingController,
)
from .trajectories import CircleTrajectory, ConstantPositionTrajectory, LineTrajectory, ReferenceTrajectory, TrajectoryPoint
from .closed_loop import ClosedLoopSimulationResult, simulate_trajectory_tracking

__all__ = [
    "UnderwaterRobotParameters",
    "OscillatingFinGait",
    "SimulationConfig",
    "UnderwaterRobotModel",
    "OscillatingFinRobot",
    "FinSliceState",
    "SimulationResult",
    "make_initial_state",
    "simulate_free_swimming",
    "normalize_quaternion",
    "quat_wxyz_to_rot",
    "rot_to_quat_wxyz",
    "quat_derivative_wxyz",
    "euler_zyx_from_quat_wxyz",
    "rotation_zyx",
    "skew",
    "quadratic_body_drag",
    "normal_quadratic_drag",
    "PIDGains",
    "PIDController",
    "BodyVelocityCommand",
    "OscillatingFinGaitMapper",
    "TrajectoryTrackingController",
    "TrajectoryPoint",
    "ReferenceTrajectory",
    "ConstantPositionTrajectory",
    "LineTrajectory",
    "CircleTrajectory",
    "ClosedLoopSimulationResult",
    "simulate_trajectory_tracking",
    # Backward-compatible aliases.
    "MantaParameters",
    "GaitParameters",
    "MantaRayModel",
    "simulate_free_swim",
]
