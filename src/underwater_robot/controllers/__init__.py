"""Closed-loop control utilities for underwater robots."""

from .gait_mapping import BodyVelocityCommand, OscillatingFinGaitMapper
from .pid import PIDController, PIDGains
from .tracking import TrackingDiagnostics, TrajectoryTrackingController, wrap_to_pi

__all__ = [
    "PIDGains",
    "PIDController",
    "BodyVelocityCommand",
    "OscillatingFinGaitMapper",
    "TrajectoryTrackingController",
    "TrackingDiagnostics",
    "wrap_to_pi",
]
