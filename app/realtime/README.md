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

## Smart glove input

The GUI can receive a smart glove stream over UDP. Set the source and port
before starting the GUI:

```powershell
$env:HAND_QPOS_SOURCE = "smart_glove"
$env:HAND_QPOS_UDP_HOST = "127.0.0.1"
$env:HAND_QPOS_UDP_PORT = "15000"
python -m scripts.gui.run_trial_replay_and_classifier_gui
```

Send one JSON object per UDP datagram. The array must follow the model's 23
joint qpos order. Values are radians by default; set `units` to `degrees` for
degree-valued glove sensors:

```json
{"type":"smart_glove","seq":1,"units":"degrees","joint_angles":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}
```

`qpos`, `joint_angles`, and `angles` are accepted as the array field. The
existing LeapC publisher can continue using `type: "hand_qpos"` unchanged.

