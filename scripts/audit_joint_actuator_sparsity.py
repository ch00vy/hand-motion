from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import DEFAULT_MYOSUITE_ENV_ID, create_muscle_model_context  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit anatomical joint-actuator sparsity from the MyoHand moment-arm matrix."
    )
    parser.add_argument(
        "--env-id",
        default=DEFAULT_MYOSUITE_ENV_ID,
        help="Gym / MyoSuite environment id used to build the MyoHand model context.",
    )
    parser.add_argument(
        "--joint-name",
        action="append",
        dest="joint_names",
        default=[],
        help="Specific joint name to inspect. May be passed multiple times.",
    )
    parser.add_argument(
        "--abs-threshold",
        type=float,
        default=1e-6,
        help="Absolute moment-arm threshold used to classify a muscle as relevant.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Maximum number of actuator rows to print per joint.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write a JSON summary.",
    )
    return parser.parse_args()


def summarize_joint(moment_row: np.ndarray, actuator_names: list[str], threshold: float, top_k: int) -> dict:
    abs_row = np.abs(moment_row)
    relevant = [
        {
            "actuator_name": actuator_names[idx],
            "moment_arm": float(moment_row[idx]),
            "abs_moment_arm": float(abs_row[idx]),
        }
        for idx in np.where(abs_row > float(threshold))[0]
    ]
    relevant.sort(key=lambda item: item["abs_moment_arm"], reverse=True)
    return {
        "num_relevant_actuators": int(len(relevant)),
        "num_irrelevant_actuators": int(len(actuator_names) - len(relevant)),
        "relevant_fraction": float(len(relevant) / max(1, len(actuator_names))),
        "top_relevant_actuators": relevant[: int(max(1, top_k))],
    }


def main() -> int:
    args = parse_args()
    context = create_muscle_model_context(env_id=args.env_id)
    try:
        actuator_names = list(context.actuator_names)
        joint_names = list(context.joint_names)
        moment_matrix = np.asarray(context.moment_matrix, dtype=float)

        if moment_matrix.shape != (len(actuator_names), len(joint_names)):
            raise RuntimeError(
                f"Unexpected moment matrix shape {moment_matrix.shape}; "
                f"expected ({len(actuator_names)}, {len(joint_names)})"
            )

        requested_joint_names = list(args.joint_names) if args.joint_names else list(joint_names)
        report = {
            "env_id": str(args.env_id),
            "abs_threshold": float(args.abs_threshold),
            "num_actuators": int(len(actuator_names)),
            "num_joints": int(len(joint_names)),
            "joint_reports": {},
        }

        print(f"Environment: {args.env_id}")
        print(f"Actuators: {len(actuator_names)} | Joints: {len(joint_names)}")
        print(f"Absolute threshold: {args.abs_threshold:g}")
        print()

        for joint_name in requested_joint_names:
            if joint_name not in joint_names:
                print(f"[WARN] Joint not found: {joint_name}")
                continue
            joint_idx = joint_names.index(joint_name)
            moment_row = moment_matrix[:, joint_idx]
            joint_summary = summarize_joint(moment_row, actuator_names, args.abs_threshold, args.top_k)
            report["joint_reports"][joint_name] = joint_summary

            print(f"Joint: {joint_name}")
            print(
                f"  Relevant actuators: {joint_summary['num_relevant_actuators']}/{len(actuator_names)} "
                f"({joint_summary['relevant_fraction']:.3f})"
            )
            for item in joint_summary["top_relevant_actuators"]:
                print(
                    f"  - {item['actuator_name']}: "
                    f"{item['moment_arm']:+.6f} | abs={item['abs_moment_arm']:.6f}"
                )
            print()

        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Wrote JSON report to {output_path}")
        return 0
    finally:
        try:
            context.env.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
