from __future__ import annotations

from enum import Enum


class GuiState(str, Enum):
    IDLE = "IDLE"
    RECORDING = "RECORDING"
    TRIAL_READY = "TRIAL_READY"
    ANALYZING = "ANALYZING"
    ANALYSIS_DONE = "ANALYSIS_DONE"
    KINEMATIC_REPLAY = "KINEMATIC_REPLAY"
    CONTROL_REPLAY = "CONTROL_REPLAY"
