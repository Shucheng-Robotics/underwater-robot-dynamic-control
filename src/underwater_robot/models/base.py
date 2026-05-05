"""Base interfaces for underwater robot dynamics models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class UnderwaterRobotModel(ABC):
    """SciPy-compatible base interface for underwater robot models.

    Subclasses should use the common 13-state convention:

    ``x = [p_wb(3), q_wxyz(4), v_b(3), omega_b(3)]``.
    """

    @abstractmethod
    def state_derivative(self, t: float, x: np.ndarray, command: Any) -> np.ndarray:
        """Return the continuous-time state derivative ``dx/dt``."""
