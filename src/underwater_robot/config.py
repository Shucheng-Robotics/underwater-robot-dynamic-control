"""Configuration dataclasses for underwater robot dynamics models."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Tuple

import numpy as np

Array3 = Tuple[float, float, float]


@dataclass
class UnderwaterRobotParameters:
    """Physical, geometric, hydrodynamic, and numerical parameters.

    The default values instantiate a two-fin bio-inspired swimmer derived from
    the original research script.  The names are intentionally generic so the
    same dataclass can seed related underwater robot models such as oscillating
    fin vehicles, biomimetic swimmers, or simplified AUV prototypes.
    """

    # Oscillating-fin geometry and mechanism biases.
    fin_width: float = 0.14
    theta_bias_s: float = 2.0 * pi / 9.0
    theta_bias_l: float = pi / 3.0
    link_a: float = 0.024
    link_b: float = 0.010
    link_c: float = 0.014
    link_e: float = 0.024
    link_l1: float = 0.056
    link_l2: float = 0.168
    link_l2_quadratic: float = 0.0045

    # Body-to-left/right actuator root locations in body frame.
    p_left_root: Array3 = (0.09879, 0.00093, -0.0104)
    p_right_root: Array3 = (-0.09879, 0.00093, -0.0104)
    p_distal_link: Array3 = (0.056, 0.0, 0.0)

    # Hydrodynamic coefficients for the two local fin segments.
    cn1: Array3 = (0.5, 1.8, 0.5)
    cn2: Array3 = (0.5, 1.8, 0.5)
    cp1: Array3 = (0.15, 1.0, 1.0)
    cp2: Array3 = (0.15, 1.0, 1.0)
    cr1: float = 0.2
    cr2: float = 0.3
    ce1: float = 0.8
    ce2: float = 0.7

    # Rigid-body drag, rotational damping and projected body areas.
    body_drag: Array3 = (0.672, 0.1344, 33.6)
    angular_damping: Array3 = (0.015, 0.010, 0.020)
    projected_area: Array3 = (0.0298, 0.0167, 0.0577)

    # Rigid-body inertial parameters.
    inertia: Array3 = (0.0218, 0.0220, 0.0396)
    mass: float = 3.68
    density: float = 1000.0
    gravity: float = 9.8
    buoyancy_offset: Array3 = (0.0, 0.0, 0.001)

    # Numerical parameters.
    hydrodynamic_quadrature_order: int = 50
    finite_difference_dt: float = 1e-5
    tangent_difference_ds: float = 1.4e-4
    safe_trig_clip: bool = True
    dissipative_body_drag: bool = True

    # Backward-compatible aliases used by older scripts.
    @property
    def p_bl(self) -> Array3:
        return self.p_left_root

    @property
    def p_br(self) -> Array3:
        return self.p_right_root

    @property
    def p_cd(self) -> Array3:
        return self.p_distal_link

    @property
    def cb(self) -> Array3:
        return self.body_drag

    @property
    def cw(self) -> Array3:
        return self.angular_damping

    @property
    def area(self) -> Array3:
        return self.projected_area

    def as_dict(self) -> dict:
        """Return a JSON-serializable dictionary of public dataclass fields."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class OscillatingFinGait:
    """Prescribed left/right fin gait parameters.

    The gait is generic: it describes two oscillating appendages with individual
    frequencies and spanwise phase offsets.  It can represent a manta-like
    pectoral-fin gait, a two-fin underwater robot gait, or a simplified flapping
    propulsion primitive.
    """

    omega_left: float = pi
    omega_right: float = pi
    beta_left: float = pi
    beta_right: float = 40.0 * pi / 180.0
    yaw_amplitude: float = 40.0 * pi / 180.0
    yaw_phase: float = 280.0 * pi / 360.0


@dataclass
class SimulationConfig:
    """ODE simulation settings and initial state."""

    t0: float = 0.0
    tf: float = 0.01
    dt: float = 0.01
    initial_position: Array3 = (0.0, 0.0, 0.0)
    initial_velocity_body: Array3 = (0.0, 0.0, 1.0)
    initial_angular_velocity_body: Array3 = (0.0, 0.0, 0.0)
    initial_quaternion_wxyz: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    solver_rtol: float = 1e-6
    solver_atol: float = 1e-9
    max_step: float | None = None

    @property
    def time_vector(self) -> np.ndarray:
        """Return an evenly spaced time vector including the final time."""
        n = int(round((self.tf - self.t0) / self.dt))
        return np.linspace(self.t0, self.t0 + n * self.dt, n + 1)


# Backward-compatible aliases.  New code should prefer the generic names above.
MantaParameters = UnderwaterRobotParameters
GaitParameters = OscillatingFinGait
