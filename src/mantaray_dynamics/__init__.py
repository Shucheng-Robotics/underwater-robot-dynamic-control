"""Compatibility layer for the previous ``mantaray_dynamics`` package name.

New projects should import from :mod:`underwater_robot`.  This module is kept so
older scripts based on the original refactor continue to run.
"""

from underwater_robot import *  # noqa: F401,F403
