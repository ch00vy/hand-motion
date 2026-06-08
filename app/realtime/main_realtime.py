"""
app/realtime/main_realtime.py
==============================
Entry point realtime runtime.

Saat ini file ini forward ke backend runtime transisional.
Setelah split backend selesai, modul-modul berikut akan menjadi sumber utama:
  - realtime_controller.py  : RealtimeQPController + konstanta
  - realtime_io.py          : LeapToMyoSuiteMapper + LeapDataListener + Canvas
  - realtime_loop.py        : main() loop + run_recorded_qpos_runtime_validation()
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _cli_main() -> int:
    """Forward ke backend runtime transisional selama split belum selesai."""
    from app.realtime.monolithic_runtime_backend import _cli_main as _orig_main  # type: ignore
    return _orig_main()


if __name__ == "__main__":
    raise SystemExit(_cli_main())
