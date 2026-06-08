# Realtime Workflow

Final realtime scope:

`Leap Motion / joint rotations / qpos -> t_des -> ctrl mapping -> MyoSuite/MuJoCo`

Rules:

- No inverse dynamics in the realtime loop.
- No OSQP in the realtime loop.
- No fast predictor as the main runtime path.
- Realtime graph is optional visualization only.

Main command:

```powershell
.\venv\Scripts\python.exe scripts\realtime\run_live_realtime_ctrl.py --main-sim-mode estimated_ctrl_step --ctrl-mode control
```

