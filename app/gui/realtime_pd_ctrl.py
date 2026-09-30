from __future__ import annotations

"""Compatibility import for the GUI realtime ctrl indicator.

The implementation now lives in `app.estimation.ctrl_indicator` so the codebase
keeps estimation concerns separate from GUI wiring.
"""

from app.estimation.ctrl_indicator import RealtimePDDynamicCtrlConfig, RealtimePDDynamicCtrlEstimator

__all__ = ["RealtimePDDynamicCtrlConfig", "RealtimePDDynamicCtrlEstimator"]
