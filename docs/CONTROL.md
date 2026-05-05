# Control and Trajectory Tracking

This package includes a lightweight closed-loop control stack for underwater robot simulation.  It is designed to be simple enough for demonstrations while remaining modular enough to replace each layer with a calibrated controller.

## Control architecture

The control pipeline is:

```text
reference trajectory
      ↓
position/yaw PID controller
      ↓
high-level body command
      ↓
oscillating-fin gait mapper
      ↓
OscillatingFinRobot dynamics
```

The controller is intentionally separated from the dynamics model:

- `PIDController` is a generic vector PID with output limits and anti-windup.
- `TrajectoryTrackingController` converts position/yaw tracking errors into desired body-frame commands.
- `OscillatingFinGaitMapper` converts high-level commands into `OscillatingFinGait` parameters.
- `simulate_trajectory_tracking` performs sample-and-hold closed-loop simulation so PID integral state is updated once per control step instead of once per internal ODE evaluation.

## Minimal PID example

```python
import numpy as np
from underwater_robot import PIDController, PIDGains

pid = PIDController(PIDGains(kp=1.0, ki=0.1, kd=0.02), size=3, output_limits=(-1.0, 1.0))
error = np.array([0.2, -0.1, 0.05])
u = pid.update(error, dt=0.02)
```

## Trajectory tracking example

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
mapper = OscillatingFinGaitMapper(base_omega=pi, base_beta=pi / 3)
controller = TrajectoryTrackingController(
    trajectory=trajectory,
    position_gains=PIDGains(kp=(0.2, 0.2, 0.4), kd=(0.02, 0.02, 0.04)),
    yaw_gains=PIDGains(kp=0.5, kd=0.02),
    gait_mapper=mapper,
    forward_axis=2,
)
config = SimulationConfig(tf=0.1, dt=0.02, max_step=0.01, initial_velocity_body=(0, 0, 0.1))
result = simulate_trajectory_tracking(model, controller, config)
print(result.position[:, -1])
print(result.reference_position[:, -1])
print(result.gait_history[:, -1])
```

Run the bundled example:

```bash
python examples/trajectory_tracking_demo.py
```

## Notes on gait mapping

The default gait mapper is a heuristic interface rather than a calibrated controller.  It maps:

- forward command → mean oscillation frequency;
- yaw-rate command → left/right frequency difference;
- vertical command → spanwise phase bias.

For a physical robot or high-fidelity simulator, replace `OscillatingFinGaitMapper` with an identified mapping from command space to actuator parameters.
