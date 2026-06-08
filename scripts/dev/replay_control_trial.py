from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gui.inference_controller import BinaryGCNInferenceController
from app.gui.mujoco_replay_controller import MujocoReplayController
from app.gui.trial_loader import load_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay one trial using estimated ctrl_ref in the external MuJoCo viewer.")
    parser.add_argument("--trial", required=True, help="Path to qpos.csv, legacy ctrl_est_*.csv, or a trial directory.")
    parser.add_argument("--loops", type=int, default=3, help="Number of replay loops.")
    parser.add_argument("--hold-seconds", type=float, default=8.0, help="Seconds to keep the viewer open after replay.")
    parser.add_argument("--min-frame-sleep", type=float, default=0.015, help="Minimum sleep between frames.")
    parser.add_argument("--max-frame-sleep", type=float, default=0.05, help="Maximum sleep between frames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trial = load_trial(Path(args.trial))
    inference = BinaryGCNInferenceController()
    ctrl_ref = inference.load_control_reference(trial)
    quality = dict(ctrl_ref.get("quality") or {})

    print(f"[REPLAY] CONTROL_REPLAY trial={trial.trial_id} frames={trial.num_frames}")
    print(f"[REPLAY] ctrl_ref={ctrl_ref['source_path']}")
    print(f"[REPLAY] dt={float(ctrl_ref['dt']):.6f} ctrl_shape={tuple(ctrl_ref['ctrl_sequence'].shape)}")
    if quality:
        print(
            "[REPLAY] quality "
            f"rmse={float(quality.get('rmse_forward_error', 0.0)):.4f} "
            f"mean_joint={float(quality.get('mean_joint_error', 0.0)):.4f} "
            f"relative_residual={float(quality.get('relative_residual_mean', 0.0)):.4f}"
        )

    replay = MujocoReplayController()
    ok = replay.replay_control(
        trial=trial,
        ctrl_sequence=ctrl_ref["ctrl_sequence"],
        dt=float(ctrl_ref["dt"]),
        source_path=str(ctrl_ref["source_path"]),
    )
    if not ok:
        raise SystemExit(replay.status().message)

    frame_count = max(1, int(ctrl_ref["num_frames"]))
    sleep_s = max(float(args.min_frame_sleep), min(float(args.max_frame_sleep), float(ctrl_ref["dt"])))
    for loop_index in range(max(1, int(args.loops))):
        print(f"[REPLAY] loop {loop_index + 1}/{max(1, int(args.loops))}")
        replay.reset()
        for _ in range(frame_count - 1):
            replay.step(1)
            time.sleep(sleep_s)
        time.sleep(0.5)

    print(f"[REPLAY] done; holding viewer for {float(args.hold_seconds):.1f}s")
    time.sleep(max(0.0, float(args.hold_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
