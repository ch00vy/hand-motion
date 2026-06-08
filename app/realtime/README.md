# Realtime Package

This package contains the final realtime namespace.

Current state:

- `main_realtime.py` is the canonical Python realtime entrypoint.
- `monolithic_runtime_backend.py` is the transitional backend moved out of
  `app/main.py`.
- `realtime_controller.py`, `realtime_io.py`, and `realtime_loop.py` are clean
  module boundaries that re-export from the transitional backend until the
  full extraction is complete.

Realtime rules:

- Keep inverse dynamics and OSQP out of the live loop.
- Keep fast predictor experiments out of the main runtime path.
- Use pragmatic qpos to ctrl mapping for live control.

