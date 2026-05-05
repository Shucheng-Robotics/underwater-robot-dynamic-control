"""Underwater robot dynamics models."""

from .base import UnderwaterRobotModel
from .oscillating_fin_robot import FinSliceState, OscillatingFinRobot

# Compatibility alias for users coming from the original project name.
MantaRayModel = OscillatingFinRobot

__all__ = ["UnderwaterRobotModel", "FinSliceState", "OscillatingFinRobot", "MantaRayModel"]
