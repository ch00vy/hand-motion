from __future__ import annotations

"""Canonical live realtime entrypoint for the final architecture.

This wrapper keeps the live path focused on:
    Leap / joint rotations / qpos -> realtime ctrl mapping -> ctrl_realtime

It intentionally defaults to the pragmatic live path and does not surface the
archived predictor experiments as the main route.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.realtime.main_realtime import _cli_main  # noqa: E402


def _has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or str(arg).startswith(f"{option}=") for arg in argv)


def main() -> int:
    argv = list(sys.argv[1:])
    if not _has_option(argv, "--input-mode"):
        argv.extend(["--input-mode", "leap_live"])
    if not _has_option(argv, "--ctrl-runtime-mode"):
        argv.extend(["--ctrl-runtime-mode", "legacy_solver"])
    if not _has_option(argv, "--main-sim-mode"):
        argv.extend(["--main-sim-mode", "estimated_ctrl_step"])
    sys.argv = [sys.argv[0], *argv]
    return _cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
