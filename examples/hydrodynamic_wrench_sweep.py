"""Evaluate oscillating-appendage hydrodynamic wrench over one gait period."""

from math import pi

import matplotlib.pyplot as plt

from underwater_robot import OscillatingFinGait, OscillatingFinRobot, UnderwaterRobotParameters


def main() -> None:
    model = OscillatingFinRobot(UnderwaterRobotParameters(hydrodynamic_quadrature_order=20))
    gait = OscillatingFinGait(omega_left=pi, omega_right=pi, beta_left=pi / 3, beta_right=pi / 3)
    t, wrench, mean_wrench, _, _ = model.hydrodynamic_wrench_period(gait, velocity_body=[0, 0, 1], samples=25)
    print("mean wrench [Fx,Fy,Fz,Mx,My,Mz] =", mean_wrench)

    labels = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), constrained_layout=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(t, wrench[i, :])
        ax.set_title(labels[i])
        ax.set_xlabel("time [s]")
        ax.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
