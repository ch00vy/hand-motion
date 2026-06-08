"""
app/realtime/__init__.py
=========================
Package realtime runtime.

Saat ini masih berisi salinan file dari app/ (backward compat).
Setelah split main.py selesai, submodul ini akan menjadi sumber utama:
  - main_realtime.py          : argparse + _cli_main()
  - realtime_loop.py          : main() + run_recorded_qpos_runtime_validation()
  - realtime_controller.py    : RealtimeQPController + konstanta QP
  - realtime_io.py            : LeapToMyoSuiteMapper + LeapDataListener + Canvas
  - realtime_graph.py         : RealtimeGraphRuntime (sudah ada)
  - realtime_runtime_support.py (sudah ada)
  - runtime_recording.py      (sudah ada)
  - runtime_logger.py         (sudah ada)
"""
