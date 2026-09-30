from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowRect:
    x: int
    y: int
    width: int
    height: int


# Assumption: second monitor is placed to the right of the primary monitor.
MONITOR_1_X = 0
MONITOR_2_X = 1920

MAIN_GUI = WindowRect(
    x=20,
    y=40,
    width=1440,
    height=920,
)

MUSCLE_GRAPH = WindowRect(
    x=40,
    y=40,
    width=760,
    height=980,
)

LEAP_VIEWER = WindowRect(
    x=MONITOR_2_X + 980,
    y=40,
    width=700,
    height=500,
)

MUJOCO_HINT = WindowRect(
    x=MONITOR_2_X + 20,
    y=40,
    width=900,
    height=900,
)
