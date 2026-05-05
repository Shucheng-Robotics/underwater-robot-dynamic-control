"""Free-swimming oscillating-fin underwater robot demo.

Run from the project root:

    python examples/free_swimming_demo.py
"""

from math import pi

from underwater_robot import (
    OscillatingFinGait,
    OscillatingFinRobot,
    SimulationConfig,
    UnderwaterRobotParameters,
    simulate_free_swimming,
)
from underwater_robot.visualization import plot_state_history


def main() -> None:
    params = UnderwaterRobotParameters(hydrodynamic_quadrature_order=20)
    gait = OscillatingFinGait(
        omega_left=pi,
        omega_right=pi,
        beta_left=pi,
        beta_right=40.0 * pi / 180.0,
        yaw_amplitude=40.0 * pi / 180.0,
    )
    config = SimulationConfig(tf=0.05, dt=0.01, initial_velocity_body=(0.0, 0.0, 1.0))
    model = OscillatingFinRobot(params)
    result = simulate_free_swimming(model=model, gait=gait, config=config)
    print(f"success={result.success}, message={result.message}")
    print("final position [m] =", result.position[:, -1])
    print("final body velocity [m/s] =", result.velocity_body[:, -1])
    plot_state_history(result)


if __name__ == "__main__":
    main()
