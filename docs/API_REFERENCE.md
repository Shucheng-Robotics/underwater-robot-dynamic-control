# API Reference

## `underwater_robot.config`

### `UnderwaterRobotParameters`

Physical, hydrodynamic, geometric and numerical parameters.  Common fields:

- `fin_width`
- `mass`
- `inertia`
- `projected_area`
- `body_drag`, `angular_damping`
- `cn1`, `cn2`, `cp1`, `cp2`
- `hydrodynamic_quadrature_order`
- `finite_difference_dt`
- `dissipative_body_drag`

Backward-compatible aliases are provided for older code: `MantaParameters`, `p_bl`, `p_br`, `p_cd`, `cb`, `cw`, and `area`.

### `OscillatingFinGait`

Prescribed left/right appendage gait parameters:

```python
OscillatingFinGait(
    omega_left=pi,
    omega_right=pi,
    beta_left=pi / 3,
    beta_right=pi / 3,
    yaw_amplitude=40 * pi / 180,
)
```

Backward-compatible alias: `GaitParameters`.

### `SimulationConfig`

ODE simulation settings and initial state.  The state uses the convention:

```text
x = [p_wb, q_wxyz, v_b, omega_b]
```

## `underwater_robot.models`

### `OscillatingFinRobot(params=None)`

Core underwater robot dynamics class.

#### `state_derivative(t, x, gait)`

Returns `dx/dt` for SciPy ODE solvers.

#### `motion_equation(t)`

Returns `[v_dot_b, omega_dot_b]` using the current internal body and gait state.

#### `hydrodynamic_wrench_period(gait, velocity_body=None, samples=51)`

Returns time history and average appendage wrench over one oscillation period.

#### `appendage_link_angles(s, t, left)`

Returns linkage angles and midpoint vectors for one appendage slice.

#### `appendage_curve_points(s, t, left)`

Returns points along the two-link appendage centerline.

#### `appendage_slice_kinematics(s, t, left)`

Returns midpoint velocities and coordinates for one appendage slice.

### Legacy aliases

- `MantaRayModel = OscillatingFinRobot`
- `slice_fin_angle = appendage_link_angles`
- `fin_curve_point = appendage_curve_points`
- `fin_slice_kinematics = appendage_slice_kinematics`
- `hydrodynamic_force_period = hydrodynamic_wrench_period`

## `underwater_robot.simulation`

### `simulate_free_swimming(model=None, gait=None, config=None)`

Runs `scipy.integrate.solve_ivp` and returns a `SimulationResult`.

Backward-compatible alias: `simulate_free_swim`.

### `SimulationResult`

Fields:

- `time`
- `state`
- `success`
- `message`

Convenience properties:

- `position`
- `quaternion_wxyz`
- `velocity_body`
- `angular_velocity_body`

## `underwater_robot.math_utils`

Reusable rigid-body utilities:

- `skew`
- `normalize_quaternion`
- `quat_wxyz_to_rot`
- `rot_to_quat_wxyz`
- `quat_derivative_wxyz`
- `euler_zyx_from_quat_wxyz`
- `rotation_zyx`

## `underwater_robot.models.base`

### `UnderwaterRobotModel`

Abstract base interface for future vehicle models.  New models should implement:

```python
dx = model.state_derivative(t, x, command)
```

using the common 13-state convention.

## `underwater_robot.hydrodynamics`

Reusable force helpers:

- `quadratic_body_drag(velocity_body, projected_area, drag_coefficients, density=1000.0)`
- `normal_quadratic_drag(velocity, normal, coefficients, density=1000.0, area_scale=1.0)`

These helpers are intentionally generic and can be used by oscillating-fin models, thruster-based AUV models, or other marine robot dynamics prototypes.


## Control APIs

### `PIDGains` and `PIDController`

```python
from underwater_robot import PIDController, PIDGains
pid = PIDController(PIDGains(kp=1.0, ki=0.05, kd=0.01), size=3, output_limits=(-1, 1))
u = pid.update(error, dt=0.02)
```

`PIDController` supports vector channels, integral clipping, output limits, and optional derivative-on-measurement.

### Reference trajectories

```python
from underwater_robot import LineTrajectory, CircleTrajectory, ConstantPositionTrajectory
```

Each trajectory is callable and returns a `TrajectoryPoint` with position, velocity, yaw, and yaw-rate fields.

### `TrajectoryTrackingController`

```python
from underwater_robot import TrajectoryTrackingController, OscillatingFinGaitMapper
```

The tracking controller maps world-frame position/yaw errors into high-level body commands and then into `OscillatingFinGait` commands.

### `simulate_trajectory_tracking`

```python
result = simulate_trajectory_tracking(model, controller, config)
```

Returns `ClosedLoopSimulationResult`, which extends `SimulationResult` with `reference_position`, `position_error`, `velocity_command_world`, `velocity_command_body`, `body_command`, `gait_history`, and `yaw_error`.
