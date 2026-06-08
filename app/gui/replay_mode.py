from __future__ import annotations

from enum import Enum


class ReplayMode(str, Enum):
    NONE = "NONE"
    KINEMATIC = "KINEMATIC"
    CONTROL = "CONTROL"
