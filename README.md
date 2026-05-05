# Underwater Robot
Please cite: Zhang S, Xie X, Sui Y, et al. Dynamic modeling and stiffness optimization for a manta ray-inspired robot[J]. International Journal of Mechanical Sciences, 2026: 111417. 

A reusable Python dynamics library for underwater robot research and portfolio demonstration.  The current implementation provides a generic rigid-body underwater robot model with two prescribed oscillating fins/appendages, Morison-type hydrodynamic loading, quaternion attitude integration, SciPy-based simulation utilities, plotting tools, examples, tests, and documentation.

The project started from a single-file biomimetic swimmer research script and has been refactored into a GitHub-ready robotics software package.  The default parameter set still reproduces a manta-like oscillating-fin robot, but the public API is written in generic underwater-robot terms so the same package can be extended toward AUVs, biomimetic swimmers, multi-fin vehicles, and controller benchmarks.

## Highlights

- **Generic underwater-robot API**: `UnderwaterRobotParameters`, `OscillatingFinGait`, `OscillatingFinRobot`, and `simulate_free_swimming`.
- **13-state rigid-body dynamics**: position, quaternion attitude, body-frame linear velocity, and body-frame angular velocity.
- **Oscillating appendage kinematics**: left/right fin linkage geometry with spanwise phase offsets.
- **Hydrodynamic wrench integration**: local normal-velocity projection and Morison-type quadratic drag integrated along the appendage span.
- **Closed-loop control module**: PID control, reference trajectories, gait mapping, and sample-and-hold trajectory tracking simulation.
- **Numerical robustness**: quaternion normalization, clipped inverse-trigonometric calculations, optional dissipative body drag, lightweight smoke tests.
- **GitHub-ready engineering**: `src/` package layout, examples, tests, docs, CI workflow, license, citation file, and legacy script preservation.
- **Backward compatibility**: the previous `mantaray_dynamics` import path still works as a compatibility layer.

## Repository structure

```text
underwater-robot/
├── src/
│   ├── underwater_robot/
│   │   ├── config.py                       # generic physical, gait and simulation dataclasses
│   │   ├── math_utils.py                   # quaternion, rotation and rigid-body helpers
│   │   ├── models/
│   │   │   └── oscillating_fin_robot.py    # core underwater robot dynamics model
│   │   ├── simulation.py                   # solve_ivp simulation wrapper
│   │   ├── controllers/                    # PID and trajectory tracking controllers
│   ├── trajectories/                    # line, circle and station-keeping references
│   ├── closed_loop.py                   # sample-and-hold closed-loop simulation
│   └── visualization.py                # plotting helpers
│   └── mantaray_dynamics/                  # compatibility import layer
├── examples/
│   ├── free_swimming_demo.py
│   ├── hydrodynamic_wrench_sweep.py
│   └── trajectory_tracking_demo.py
├── tests/
│   └── test_import_and_smoke.py
├── docs/
│   ├── MODEL_OVERVIEW.md
│   ├── API_REFERENCE.md
│   ├── ENGINEERING_NOTES.md
│   └── CONTROL.md
├── legacy/
│   ├── original_manta.py                   # original single-file research script
│   └── updated_manta_model.py              # later flexible-fin research script snapshot
├── pyproject.toml
└── README.md
```

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/<your-user>/underwater-robot.git
cd underwater-robot
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

On Git Bash / Linux / macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Quick start

Run a short free-swimming simulation:

```python
from math import pi

from underwater_robot import (
    OscillatingFinGait,
    OscillatingFinRobot,
    SimulationConfig,
    UnderwaterRobotParameters,
    simulate_free_swimming,
)

params = UnderwaterRobotParameters(hydrodynamic_quadrature_order=20)
gait = OscillatingFinGait(
    omega_left=pi,
    omega_right=pi,
    beta_left=pi,
    beta_right=40 * pi / 180,
    yaw_amplitude=40 * pi / 180,
)
config = SimulationConfig(tf=0.05, dt=0.01, initial_velocity_body=(0.0, 0.0, 1.0))

model = OscillatingFinRobot(params)
result = simulate_free_swimming(model, gait, config)
print(result.success)
print(result.position[:, -1])
```

Run bundled examples:

```bash
python examples/free_swimming_demo.py
python examples/hydrodynamic_wrench_sweep.py
```

## State and interface conventions

The body state is stored as:

```text
x = [p_wb, q_wxyz, v_b, omega_b]
```

where:

- `p_wb ∈ R^3`: body origin position in the world frame;
- `q_wxyz ∈ S^3`: unit quaternion in `[w, x, y, z]` convention;
- `v_b ∈ R^3`: body-frame linear velocity;
- `omega_b ∈ R^3`: body-frame angular velocity.

The main continuous-time dynamics interface is:

```python
dx = model.state_derivative(t, x, gait)
```

The hydrodynamic wrench over one appendage oscillation period can be evaluated with:

```python
t, wrench, mean_wrench, wrench_left, wrench_right = model.hydrodynamic_wrench_period(gait)
```

## Main APIs

### `UnderwaterRobotParameters`

Physical, hydrodynamic and numerical parameters for the model.  The default values correspond to the original two-fin biomimetic swimmer, but the naming is generic enough for extension.

Important fields include:

- `mass`, `inertia`, `projected_area`;
- `body_drag`, `angular_damping`;
- `fin_width`, link lengths and mechanism offsets;
- `cn1`, `cn2`, `cp1`, `cp2` hydrodynamic coefficients;
- `hydrodynamic_quadrature_order` and finite-difference settings.

### `OscillatingFinGait`

Prescribed left/right appendage gait:

```python
gait = OscillatingFinGait(
    omega_left=pi,
    omega_right=pi,
    beta_left=pi / 3,
    beta_right=pi / 3,
    yaw_amplitude=40 * pi / 180,
)
```

### `OscillatingFinRobot`

Core model class. Important methods:

- `appendage_link_angles(s, t, left)`: two-link appendage geometry;
- `appendage_curve_points(s, t, left)`: appendage centerline and midpoint geometry;
- `appendage_slice_kinematics(s, t, left)`: velocity propagation to appendage slice midpoints;
- `slice_hydrodynamic_force(list_s, t, left)`: local Morison-type wrench samples;
- `motion_equation(t)`: body acceleration calculation;
- `state_derivative(t, x, gait)`: ODE right-hand side;
- `hydrodynamic_wrench_period(gait)`: one-period wrench sweep.

### Backward-compatible aliases

Older scripts can still use:

```python
from mantaray_dynamics import MantaRayModel, MantaParameters, GaitParameters, simulate_free_swim
```

New code should prefer:

```python
from underwater_robot import OscillatingFinRobot, UnderwaterRobotParameters, OscillatingFinGait, simulate_free_swimming
```


## Closed-loop trajectory tracking

The project now includes a usable control stack for simulation-level trajectory tracking:

```python
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

model = OscillatingFinRobot(UnderwaterRobotParameters(hydrodynamic_quadrature_order=8))
trajectory = LineTrajectory(start=[0, 0, 0], velocity=[0, 0, 0.08], yaw=0.0)
controller = TrajectoryTrackingController(
    trajectory=trajectory,
    position_gains=PIDGains(kp=(0.2, 0.2, 0.4), kd=(0.02, 0.02, 0.04)),
    yaw_gains=PIDGains(kp=0.5, kd=0.02),
    gait_mapper=OscillatingFinGaitMapper(base_omega=pi, base_beta=pi / 3),
    forward_axis=2,
)
config = SimulationConfig(tf=0.1, dt=0.02, max_step=0.01, initial_velocity_body=(0, 0, 0.1))
result = simulate_trajectory_tracking(model, controller, config)
print(result.position[:, -1])
print(result.reference_position[:, -1])
print(result.gait_history[:, -1])
```

Run the bundled demo:

```bash
python examples/trajectory_tracking_demo.py
```

The controller modules are intentionally modular:

- `PIDController`: generic vector PID with anti-windup and output limits;
- `LineTrajectory`, `CircleTrajectory`, `ConstantPositionTrajectory`: reusable references;
- `TrajectoryTrackingController`: converts position/yaw errors to body-frame commands;
- `OscillatingFinGaitMapper`: maps high-level commands to left/right fin gait parameters;
- `simulate_trajectory_tracking`: sample-and-hold closed-loop integration.

For real hardware or high-fidelity validation, calibrate or replace the gait mapper with an experimentally identified command-to-actuator model.

## Tests

```bash
pytest -q
```

The tests check package import, quaternion utilities, derivative shape, short integration, and legacy alias compatibility.


## Roadmap

- add a generic thruster-actuated AUV model using the same 13-state convention;
- add MPC/DeePC and disturbance-observer controllers on top of the PID tracking baseline;
- add animation export and mesh-based visualization;
- replace linkage-angle numerical differentiation with analytical derivatives;
- add benchmark datasets and parameter-identification examples.

## License

License.
