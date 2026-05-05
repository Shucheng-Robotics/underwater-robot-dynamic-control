from math import pi

import numpy as np

from underwater_robot import (
    OscillatingFinGait,
    OscillatingFinRobot,
    SimulationConfig,
    UnderwaterRobotParameters,
    simulate_free_swimming,
)
from underwater_robot.math_utils import normalize_quaternion, quat_wxyz_to_rot


def test_quaternion_normalization_and_rotation():
    q = normalize_quaternion(np.array([2.0, 0.0, 0.0, 0.0]))
    assert np.allclose(q, [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(quat_wxyz_to_rot(q), np.eye(3))


def test_model_derivative_shape():
    model = OscillatingFinRobot(UnderwaterRobotParameters(hydrodynamic_quadrature_order=4))
    gait = OscillatingFinGait(omega_left=pi, omega_right=pi)
    x = np.zeros(13)
    x[3] = 1.0
    x[9] = 1.0
    dx = model.state_derivative(0.0, x, gait)
    assert dx.shape == (13,)
    assert np.all(np.isfinite(dx))


def test_short_simulation_runs():
    model = OscillatingFinRobot(UnderwaterRobotParameters(hydrodynamic_quadrature_order=4))
    gait = OscillatingFinGait(omega_left=pi, omega_right=pi)
    config = SimulationConfig(tf=0.01, dt=0.01)
    result = simulate_free_swimming(model, gait, config)
    assert result.state.shape[0] == 13
    assert result.time.size >= 2


def test_legacy_import_aliases():
    from mantaray_dynamics import MantaRayModel, MantaParameters, GaitParameters, simulate_free_swim

    model = MantaRayModel(MantaParameters(hydrodynamic_quadrature_order=4))
    gait = GaitParameters(omega_left=pi, omega_right=pi)
    result = simulate_free_swim(model, gait, SimulationConfig(tf=0.01, dt=0.01))
    assert result.success
