"""Generic oscillating-fin underwater robot dynamics model.

This module generalizes the original manta-inspired research script into a
reusable underwater robot model.  The default parameter set still reproduces a
manta-like two-fin swimmer, but the public naming and interfaces are framed for
broader use in underwater robotics: body-frame rigid-body dynamics, prescribed
oscillating appendage gaits, hydrodynamic fin loading, and free-swimming ODE
simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, pi, sin, sqrt
from typing import Tuple

import numpy as np
from scipy import integrate

from ..config import OscillatingFinGait, UnderwaterRobotParameters
from ..hydrodynamics import quadratic_body_drag
from ..math_utils import normalize_quaternion, quat_derivative_wxyz, quat_wxyz_to_rot, rotation_zyx
from .base import UnderwaterRobotModel


@dataclass
class FinSliceState:
    """Kinematic data for a single spanwise fin slice."""

    velocity_p1: np.ndarray
    velocity_p2: np.ndarray
    point_p1: np.ndarray
    point_p2: np.ndarray


class OscillatingFinRobot(UnderwaterRobotModel):
    """Rigid-body underwater robot with two prescribed oscillating fins.

    State vector convention:
        ``x = [p_wb(3), q_wxyz(4), v_b(3), omega_b(3)]``.

    The model is intentionally modular: other underwater robot variants can
    reuse the body-dynamics convention, the hydrodynamic force routines, and the
    gait interface while replacing the fin geometry or force law.
    """

    def __init__(self, params: UnderwaterRobotParameters | None = None) -> None:
        self.params = params or UnderwaterRobotParameters()

        # Body pose and velocity variables updated during integration.
        self.R_wb = np.eye(3)
        self.P_wb = np.zeros(3)
        self.Vb = np.zeros(3)
        self.Omega_b = np.zeros(3)

        # Current left/right actuator state.
        self.theta_left = 0.0
        self.theta_right = 0.0
        self.dtheta_left = 0.0
        self.dtheta_right = 0.0
        self.beta_left = 0.0
        self.beta_right = 0.0
        self.omega_left = 0.0
        self.omega_right = 0.0

        # Cached matrices and constants.
        self.Cn1 = np.diag(self.params.cn1)
        self.Cn2 = np.diag(self.params.cn2)
        self.Cp1 = np.diag(self.params.cp1)
        self.Cp2 = np.diag(self.params.cp2)
        self.CB = np.diag(self.params.body_drag)
        self.Cw = np.diag(self.params.angular_damping)
        self.A = np.diag(self.params.projected_area)
        self.J = np.diag(self.params.inertia)
        self.inv_J = np.linalg.inv(self.J)
        self.P_left_root = np.asarray(self.params.p_left_root, dtype=float)
        self.P_right_root = np.asarray(self.params.p_right_root, dtype=float)
        self.P_distal_link = np.asarray(self.params.p_distal_link, dtype=float)

        # Legacy field aliases used by older scripts and preserved for migration.
        self._sync_legacy_names()

    def _sync_legacy_names(self) -> None:
        """Synchronize old manta-specific attribute names with generic names."""
        self.theta1_l = self.theta_left
        self.theta1_r = self.theta_right
        self.d_theta1_l = self.dtheta_left
        self.d_theta1_r = self.dtheta_right
        self.beta_l = self.beta_left
        self.beta_r = self.beta_right
        self.omega_bl = self.omega_left
        self.omega_br = self.omega_right
        self.P_bl = self.P_left_root
        self.P_br = self.P_right_root
        self.P_cd = self.P_distal_link

    @property
    def W(self) -> float:
        """Fin span/width used as the hydrodynamic integration interval."""
        return self.params.fin_width

    @staticmethod
    def rotation_world_from_body_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Return world-from-body rotation from roll, pitch and yaw angles."""
        return rotation_zyx(roll, pitch, yaw)

    @staticmethod
    def appendage_bending_rotation(theta: float) -> np.ndarray:
        """Return the base-to-fin/appendage bending rotation matrix."""
        return np.array(
            [[cos(theta), -sin(theta), 0.0], [sin(theta), cos(theta), 0.0], [0.0, 0.0, 1.0]],
            dtype=float,
        )

    # Backward-compatible method name.
    bending_fin = appendage_bending_rotation

    def _safe_acos(self, x: float) -> float:
        if self.params.safe_trig_clip:
            x = float(np.clip(x, -1.0, 1.0))
        return acos(x)

    @staticmethod
    def _wrap_2pi(theta: float) -> float:
        return theta % (2.0 * pi)

    def appendage_link_angles(self, s: float, t: float, left: bool) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Return the two local link angles and segment midpoint vectors.

        Parameters
        ----------
        s:
            Spanwise coordinate in ``[0, W]``.
        t:
            Time in seconds.
        left:
            ``True`` for the left appendage and ``False`` for the right appendage.
        """
        omega = self.omega_left if left else self.omega_right
        beta = self.beta_left if left else self.beta_right
        theta_b = self._wrap_2pi(omega * t - s / self.W * beta)

        a, b, c, e = self.params.link_a, self.params.link_b, self.params.link_c, self.params.link_e
        u = sqrt(max(a**2 + b**2 - 2 * a * b * cos(theta_b), 1e-16))
        theta_r3 = self._safe_acos((e**2 + c**2 - u**2) / (2 * e * c))
        theta_r1 = self._safe_acos((e**2 + u**2 - c**2) / (2 * e * u))
        theta_r2 = self._safe_acos((a**2 + u**2 - b**2) / (2 * a * u))

        if theta_b <= pi:
            theta_s = self.params.theta_bias_s - (theta_r1 + theta_r2)
        else:
            theta_s = self.params.theta_bias_s - (theta_r1 - theta_r2)
        theta_l = self.params.theta_bias_l - theta_r3

        if not left:
            theta_s = -theta_s
            theta_l = -theta_l

        L1 = self.params.link_l1
        l2 = self.params.link_l2 - self.params.link_l2_quadratic * s**2

        if left:
            P1 = np.array([L1 * cos(theta_s) / 2, 0.0, -L1 * sin(theta_s) / 2], dtype=float)
            P2 = np.array([l2 * cos(theta_s + theta_l) / 2, 0.0, -l2 * sin(theta_s + theta_l) / 2], dtype=float)
        else:
            P1 = np.array([-L1 * cos(theta_s) / 2, 0.0, L1 * sin(theta_s) / 2], dtype=float)
            P2 = np.array([-l2 * cos(theta_s + theta_l) / 2, 0.0, l2 * sin(theta_s + theta_l) / 2], dtype=float)
        return theta_s, theta_l, P1, P2

    # Backward-compatible method name.
    slice_fin_angle = appendage_link_angles

    def appendage_curve_points(self, s: float, t: float, left: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return points on the first and second appendage link centerlines."""
        theta_s, theta_l, _, _ = self.appendage_link_angles(s, t, left)
        L1 = self.params.link_l1
        l2 = self.params.link_l2 - self.params.link_l2_quadratic * s**2

        if left:
            P_lc = np.array([0.01838507, self.W / 2, -0.0154269], dtype=float)
            I1 = np.array([P_lc[0] + L1 * cos(theta_s), P_lc[1] - s, P_lc[2] - L1 * sin(theta_s)])
            I2 = np.array([I1[0] + l2 * cos(theta_s + theta_l), I1[1], I1[2] - l2 * sin(theta_s + theta_l)])
            P1 = np.array([P_lc[0] + L1 * cos(theta_s) / 2, I1[1], P_lc[2] - L1 * sin(theta_s) / 2])
            P2 = np.array([I1[0] + l2 * cos(theta_s + theta_l) / 2, I1[1], I1[2] - l2 * sin(theta_s + theta_l) / 2])
        else:
            P_lc = np.array([-0.01838507, self.W / 2, -0.0154269], dtype=float)
            I1 = np.array([P_lc[0] - L1 * cos(theta_s), P_lc[1] - s, P_lc[2] + L1 * sin(theta_s)])
            I2 = np.array([I1[0] - l2 * cos(theta_s + theta_l), I1[1], I1[2] + l2 * sin(theta_s + theta_l)])
            P1 = np.array([P_lc[0] - L1 * cos(theta_s) / 2, P_lc[1] - s, P_lc[2] + L1 * sin(theta_s) / 2])
            P2 = np.array([I1[0] - l2 * cos(theta_s + theta_l) / 2, I1[1], I1[2] + l2 * sin(theta_s + theta_l) / 2])
        return I1, I2, P1, P2

    # Backward-compatible method name.
    fin_curve_point = appendage_curve_points

    def rigid_body_kinematics(self) -> None:
        """Propagate body velocities to the left and right appendage root frames."""
        R_left_body = self.appendage_bending_rotation(self.theta_left).T
        self.V_left = R_left_body @ (self.Vb + np.cross(self.Omega_b, self.P_left_root))
        self.Omega_left = R_left_body @ self.Omega_b + self.dtheta_left * np.array([0.0, 0.0, 1.0])

        R_right_body = self.appendage_bending_rotation(self.theta_right).T
        self.V_right = R_right_body @ (self.Vb + np.cross(self.Omega_b, self.P_right_root))
        self.Omega_right = R_right_body @ self.Omega_b + self.dtheta_right * np.array([0.0, 0.0, 1.0])

        # Legacy aliases used internally by compatibility scripts.
        self.Vl, self.Vr = self.V_left, self.V_right
        self.Omega_l, self.Omega_r = self.Omega_left, self.Omega_right

    def appendage_link_velocity(self, t: float, s: float, left: bool) -> Tuple[float, float]:
        """Return numerical time derivatives of the two local link angles."""
        h = self.params.finite_difference_dt
        theta_2_p, theta_3_p, _, _ = self.appendage_link_angles(s, t + h, left)
        theta_2_m, theta_3_m, _, _ = self.appendage_link_angles(s, t - h, left)
        return (theta_2_p - theta_2_m) / (2 * h), (theta_3_p - theta_3_m) / (2 * h)

    # Backward-compatible method name.
    fin_rotation_velocity = appendage_link_velocity

    def appendage_slice_kinematics(self, s: float, t: float, left: bool) -> FinSliceState:
        """Return midpoint velocities and coordinates for a spanwise appendage slice."""
        theta_2, theta_3, P1, P2 = self.appendage_link_angles(s, t, left)
        R_cl = np.array([[cos(theta_2), 0.0, -sin(theta_2)], [sin(theta_2), 0.0, cos(theta_2)], [0.0, 1.0, 0.0]])
        R_lc = R_cl.T
        P_lc = np.array([0.01838507, self.W / 2 - s, -0.0154269], dtype=float)
        if not left:
            P_lc[0] = -P_lc[0]

        if left:
            V_root = self.V_left
            Omega_root = self.Omega_left
        else:
            V_root = self.V_right
            Omega_root = self.Omega_right

        Vc = R_cl @ (V_root + np.cross(Omega_root, P_lc))
        dtheta_2, dtheta_3 = self.appendage_link_velocity(t, s, left)
        Omega_c = R_cl @ Omega_root + dtheta_2 * np.array([0.0, 1.0, 0.0])

        lVc = R_lc @ Vc + np.cross(P_lc, R_lc @ Omega_c)
        l_Omega_c = Omega_root + dtheta_2 * np.array([0.0, 1.0, 0.0])
        lV_P1 = lVc + np.cross(l_Omega_c, P1)

        R_dc = np.array([[cos(theta_3), 0.0, -sin(theta_3)], [sin(theta_3), 0.0, cos(theta_3)], [0.0, 1.0, 0.0]])
        R_ld = R_lc @ R_dc.T
        P_cd = self.P_distal_link if left else -self.P_distal_link
        Vd = R_dc @ (Vc + np.cross(Omega_c, P_cd))
        Omega_d = R_dc @ Omega_c + dtheta_3 * np.array([0.0, 1.0, 0.0])
        l_Omega_d = R_ld @ Omega_d
        lVd = R_ld @ Vd + (P_lc + R_lc @ P_cd)
        lV_P2 = lVd + np.cross(l_Omega_d, P2)
        return FinSliceState(lV_P1, lV_P2, P1, P2)

    # Backward-compatible method name.
    fin_slice_kinematics = appendage_slice_kinematics

    def slice_hydrodynamic_force(self, list_s: np.ndarray, t: float, left: bool) -> np.ndarray:
        """Return hydrodynamic wrench density samples along an appendage span."""
        s_values = np.atleast_1d(list_s)
        dW = np.zeros((6, len(s_values)))
        ds = self.params.tangent_difference_ds

        for i, s in enumerate(s_values):
            kinematics = self.appendage_slice_kinematics(float(s), t, left)
            lV_P1, lV_P2, P1, P2 = (
                kinematics.velocity_p1,
                kinematics.velocity_p2,
                kinematics.point_p1,
                kinematics.point_p2,
            )
            R1 = np.array([0.0, -1.0, 0.0])
            _, _, Q1, Q2 = self.appendage_curve_points(float(s), t, left)
            I1_p, I2_p, _, _ = self.appendage_curve_points(float(s) + ds, t, left)
            I1_m, I2_m, _, _ = self.appendage_curve_points(float(s) - ds, t, left)
            E1 = (I1_p - I1_m) / (2 * ds)
            E2 = (I2_p - I2_m) / (2 * ds)
            E1 = E1 / max(np.linalg.norm(E1), 1e-12)
            E2 = E2 / max(np.linalg.norm(E2), 1e-12)

            n1 = np.cross(P1, self.params.cr1 * R1 + self.params.ce1 * E1)
            n2 = np.cross(P2, self.params.cr2 * E1 + self.params.ce2 * E2)
            n1 = n1 / max(np.linalg.norm(n1), 1e-12)
            n2 = n2 / max(np.linalg.norm(n2), 1e-12)

            V_n_p1 = self.Cp1 @ (np.dot(lV_P1, n1) * n1)
            V_n_p2 = self.Cp2 @ (np.dot(lV_P2, n2) * n2)
            dF1 = -0.5 * self.params.density * (self.Cn1 @ (np.linalg.norm(V_n_p1) * V_n_p1)) * np.linalg.norm(P1) * 2
            dF2 = -0.5 * self.params.density * (self.Cn2 @ (np.linalg.norm(V_n_p2) * V_n_p2)) * np.linalg.norm(P2) * 2

            R_bf = self.appendage_bending_rotation(self.theta_left if left else self.theta_right)
            dW[:3, i] = R_bf @ (dF1 + dF2)
            dW[3:, i] = R_bf @ (np.cross(Q1, dF1) + np.cross(Q2, dF2))
        return dW

    def set_gait_at_time(self, t: float, gait: OscillatingFinGait) -> None:
        """Update appendage angles, angular velocities, frequencies and phase shifts."""
        self.omega_left = gait.omega_left
        self.omega_right = gait.omega_right
        self.beta_left = gait.beta_left
        self.beta_right = gait.beta_right
        T = 2 * pi / max(abs(gait.omega_left), 1e-12)
        self.theta_left = gait.yaw_amplitude * sin(2 * pi * t / T + gait.yaw_phase)
        self.theta_right = -gait.yaw_amplitude * sin(2 * pi * t / T + gait.yaw_phase)
        self.dtheta_left = 2 * pi / T * gait.yaw_amplitude * cos(2 * pi * t / T + gait.yaw_phase)
        self.dtheta_right = -2 * pi / T * gait.yaw_amplitude * cos(2 * pi * t / T + gait.yaw_phase)
        self._sync_legacy_names()

    def body_drag_force(self) -> np.ndarray:
        """Return dissipative body drag force in the body frame."""
        speed = np.linalg.norm(self.Vb)
        if speed < 1e-12:
            return np.zeros(3)
        vhat = self.Vb / speed
        if self.params.dissipative_body_drag:
            return quadratic_body_drag(
                velocity_body=self.Vb,
                projected_area=np.diag(self.A),
                drag_coefficients=np.diag(self.CB),
                density=self.params.density,
            )
        SB = float(vhat @ (self.A @ vhat))
        return -0.5 * self.params.density * speed**2 * SB * (self.CB @ vhat)

    def motion_equation(self, t: float) -> np.ndarray:
        """Return body acceleration ``[v_dot_b, omega_dot_b]``."""
        self.rigid_body_kinematics()
        WL, _ = integrate.fixed_quad(
            self.slice_hydrodynamic_force,
            0.0,
            self.W,
            n=self.params.hydrodynamic_quadrature_order,
            args=(t, True),
        )
        WR, _ = integrate.fixed_quad(
            self.slice_hydrodynamic_force,
            0.0,
            self.W,
            n=self.params.hydrodynamic_quadrature_order,
            args=(t, False),
        )

        FB = self.body_drag_force()
        MB = -self.Cw @ self.Omega_b
        gravity_direction_body = self.R_wb.T @ np.array([0.0, 0.0, -1.0])
        buoyancy_offset = np.asarray(self.params.buoyancy_offset, dtype=float)
        M_b = self.params.gravity * self.params.mass * np.cross(gravity_direction_body, buoyancy_offset)

        acc = np.zeros(6)
        acc[:3] = (WL[:3] + WR[:3] + FB - np.cross(self.Omega_b, self.params.mass * self.Vb)) / self.params.mass
        acc[3:] = self.inv_J @ (WL[3:] + WR[3:] + MB + M_b - np.cross(self.Omega_b, self.J @ self.Omega_b))
        return acc

    def state_derivative(self, t: float, x: np.ndarray, gait: OscillatingFinGait) -> np.ndarray:
        """Continuous-time state derivative for SciPy ODE solvers."""
        self.set_gait_at_time(t, gait)
        x = np.asarray(x, dtype=float).reshape(13)
        q = normalize_quaternion(x[3:7])
        self.R_wb = quat_wxyz_to_rot(q)
        self.P_wb = x[:3]
        self.Vb = x[7:10]
        self.Omega_b = x[10:13]

        xd = np.zeros(13)
        xd[:3] = self.R_wb @ self.Vb
        xd[3:7] = quat_derivative_wxyz(q, self.Omega_b)
        xd[7:13] = self.motion_equation(t)
        return xd

    def hydrodynamic_wrench_period(
        self,
        gait: OscillatingFinGait,
        velocity_body: np.ndarray | None = None,
        samples: int = 51,
    ):
        """Evaluate net left/right appendage wrench over one oscillation period."""
        if velocity_body is not None:
            self.Vb = np.asarray(velocity_body, dtype=float).reshape(3)
        T = 2 * pi / max(abs(gait.omega_left), 1e-12)
        t_values = np.linspace(0.0, T, samples)
        wrench = np.zeros((6, samples))
        wrench_left = np.zeros((6, samples))
        wrench_right = np.zeros((6, samples))
        for i, ti in enumerate(t_values):
            self.set_gait_at_time(float(ti), gait)
            self.rigid_body_kinematics()
            WL, _ = integrate.fixed_quad(
                self.slice_hydrodynamic_force,
                0.0,
                self.W,
                n=self.params.hydrodynamic_quadrature_order,
                args=(float(ti), True),
            )
            WR, _ = integrate.fixed_quad(
                self.slice_hydrodynamic_force,
                0.0,
                self.W,
                n=self.params.hydrodynamic_quadrature_order,
                args=(float(ti), False),
            )
            wrench[:, i] = WL + WR
            wrench_left[:, i] = WL
            wrench_right[:, i] = WR
        return t_values, wrench, np.mean(wrench, axis=1), wrench_left, wrench_right

    # Backward-compatible method name.
    hydrodynamic_force_period = hydrodynamic_wrench_period


# Compatibility alias for the original repository version.
MantaRayModel = OscillatingFinRobot
