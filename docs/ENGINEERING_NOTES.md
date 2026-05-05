# Engineering Notes

## Purpose of this repository

The original code was a single research script for a manta-like swimming robot.  This repository reorganizes it as a reusable Python package for underwater robot dynamics so that the model can be installed, imported, tested, documented, and extended.

## Main refactor decisions

1. **Use generic underwater-robot names**
   - New package name: `underwater_robot`.
   - New core class: `OscillatingFinRobot`.
   - New parameter class: `UnderwaterRobotParameters`.
   - New gait class: `OscillatingFinGait`.

2. **Separate model from experiments**
   - Core model code lives under `src/underwater_robot`.
   - Demonstrations live under `examples`.

3. **Preserve the original script**
   - `legacy/original_manta.py` keeps the historical reference implementation.

4. **Maintain compatibility**
   - `src/mantaray_dynamics` re-exports the new package for old imports.
   - Old class/method names are available as aliases where practical.

5. **Provide testable interfaces**
   - The public ODE right-hand side is `model.state_derivative(t, x, gait)`.
   - The simulation wrapper returns a structured `SimulationResult` object.

6. **Improve numerical hygiene**
   - Clip inverse trigonometric inputs.
   - Normalize quaternions after integration.
   - Avoid printing inside numerical kernels.
   - Use a dissipative body drag formulation by default.

## Portfolio positioning

For a robotics role, this project demonstrates:

- nonlinear rigid-body dynamics implementation;
- hydrodynamic modeling and numerical integration;
- quaternion attitude representation;
- modular Python package engineering;
- documentation, examples, tests, and CI setup;
- refactoring research code into reusable software.

## Future extensions

- generic thruster-based AUV model;
- controller interfaces for MPC, PID, and trajectory tracking;
- actuator command dataclasses distinct from prescribed gait dataclasses;
- mesh or URDF-inspired geometry adapters;
- benchmark validation datasets and parameter-identification examples.

## Extending to other underwater robots

To add a new vehicle model:

1. Create a new file under `src/underwater_robot/models/`, for example `thruster_auv.py`.
2. Subclass `UnderwaterRobotModel`.
3. Implement `state_derivative(t, x, command)` using the package state convention.
4. Reuse `underwater_robot.math_utils` for quaternion and rotation operations.
5. Reuse `underwater_robot.hydrodynamics` for common drag-force primitives.
6. Add one example under `examples/` and one smoke test under `tests/`.

This keeps the repository as a robotics dynamics library rather than a single-robot script collection.
