"""Convert a Rokoko Smartgloves FBX recording to this project's 23-D qpos CSV.

Run this file with Blender, not the project's normal Python interpreter::

    blender --background --python scripts/convert_rokoko_fbx_to_qpos_blender.py -- \
      --input C:\\capture\\Clip.fbx --output-dir records\\normal\\subject_01\\rokoko_clip

The converter intentionally works from *relative bone rotations*, therefore it
does not depend on the global orientation selected in Rokoko Studio.  It is
designed for a single open-to-fist (or fist-to-open) take.  Inspect the
generated mapping report and the MuJoCo replay before using the result as a
research dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector


QPOS_SIZE = 23
FINGER_BASE = {2: 7, 3: 11, 4: 15, 5: 19}
FLEX_LIMIT = math.pi / 2.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input FBX exported by Rokoko Studio.")
    parser.add_argument("--output-dir", required=True, help="Directory to receive qpos.csv and metadata.")
    parser.add_argument("--hand", choices=("right", "left"), default="right")
    parser.add_argument("--motion-type", default="open_to_fist", choices=("open_to_fist", "fist_to_open"))
    parser.add_argument("--subject-id", default="subject_unknown")
    parser.add_argument("--trial-label", default="normal", choices=("normal", "abnormal", "testing"))
    parser.add_argument(
        "--calibration",
        choices=("auto", "first", "last"),
        default="auto",
        help="Frames representing the open hand. auto=first for open_to_fist, last for fist_to_open.",
    )
    parser.add_argument("--calibration-frames", type=int, default=15)
    return parser.parse_args(argv)


def normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("skele", ""))


def find_armature(hand: str):
    required = normalized(f"{hand}hand")
    candidates = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if not candidates:
        raise RuntimeError("No armature was found after importing the FBX.")
    for armature in candidates:
        names = {normalized(bone.name) for bone in armature.pose.bones}
        if required in names:
            return armature
    return max(candidates, key=lambda obj: len(obj.pose.bones))


def bone_lookup(armature, hand: str) -> dict[str, object]:
    available = {normalized(bone.name): bone for bone in armature.pose.bones}

    def get(label: str):
        key = normalized(label)
        if key not in available:
            raise RuntimeError(f"Required Rokoko bone '{label}' is missing. Available examples: {list(available)[:12]}")
        return available[key]

    prefix = hand.title()
    bones = {
        "hand": get(f"{prefix}Hand"),
        "forearm": get(f"{prefix}ForeArm"),
    }
    for digit in range(1, 6):
        for segment in ("Metacarpal", "Proximal", "Medial", "Distal"):
            label = f"{prefix}Finger{digit}{segment}"
            key = f"f{digit}_{segment.lower()}"
            try:
                bones[key] = get(label)
            except RuntimeError:
                if digit == 1 and segment == "Medial":
                    continue  # Rokoko thumb has no medial bone.
                raise
    return bones


def relative_rotation(pose_bone) -> Quaternion:
    if pose_bone.parent is None:
        return pose_bone.matrix.to_quaternion()
    return (pose_bone.parent.matrix.inverted() @ pose_bone.matrix).to_quaternion()


def rotation_vector(delta: Quaternion) -> Vector:
    quat = delta.normalized()
    angle = quat.angle
    if angle > math.pi:
        angle -= 2.0 * math.pi
    axis = quat.axis if abs(angle) > 1e-8 else Vector((0.0, 0.0, 0.0))
    return axis * angle


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))]


def dominant_axes(vectors: list[Vector]) -> tuple[int, int]:
    spreads = [quantile([abs(vector[idx]) for vector in vectors], 0.95) for idx in range(3)]
    order = sorted(range(3), key=lambda idx: spreads[idx], reverse=True)
    return order[0], order[1]


def calibration_indices(frame_count: int, args: argparse.Namespace) -> list[int]:
    count = min(max(1, int(args.calibration_frames)), frame_count)
    position = args.calibration
    if position == "auto":
        position = "first" if args.motion_type == "open_to_fist" else "last"
    return list(range(count)) if position == "first" else list(range(frame_count - count, frame_count))


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=str(input_path), use_anim=True)
    armature = find_armature(args.hand)
    bones = bone_lookup(armature, args.hand)
    scene = bpy.context.scene
    start, end = int(scene.frame_start), int(scene.frame_end)
    frames = list(range(start, end + 1))
    if not frames:
        raise RuntimeError("The FBX contains no animation frames.")
    calibration = calibration_indices(len(frames), args)

    tracked = ["hand", "forearm", "f1_metacarpal", "f1_proximal", "f1_distal"]
    for digit in range(2, 6):
        tracked.extend((f"f{digit}_proximal", f"f{digit}_medial", f"f{digit}_distal"))
    rotations = {name: [] for name in tracked}
    wrist_rotations = []
    for frame in frames:
        scene.frame_set(frame)
        for name in tracked:
            rotations[name].append(relative_rotation(bones[name]))
        wrist_rotations.append(relative_rotation(bones["forearm"]).inverted() @ relative_rotation(bones["hand"]))

    references = {}
    for name, samples in rotations.items():
        reference = samples[calibration[0]].copy()
        references[name] = reference

    vectors = {
        name: [rotation_vector(references[name].inverted() @ sample) for sample in samples]
        for name, samples in rotations.items()
    }
    axes = {name: dominant_axes(samples) for name, samples in vectors.items()}
    wrist_reference = wrist_rotations[calibration[0]].copy()
    wrist_vectors = [rotation_vector(wrist_reference.inverted() @ sample) for sample in wrist_rotations]
    wrist_axes = dominant_axes(wrist_vectors)

    # Sign each flexion axis so a closed hand has a positive qpos value.
    comparison = max(range(len(frames)), key=lambda index: sum(vectors[name][index].length for name in vectors if name != "hand"))
    flex_sign = {}
    for name, samples in vectors.items():
        axis, _ = axes[name]
        flex_sign[name] = 1.0 if samples[comparison][axis] >= 0.0 else -1.0

    qpos_rows = []
    for index, frame in enumerate(frames):
        qpos = [0.0] * QPOS_SIZE
        wrist = wrist_vectors[index]
        # FBX axes vary by skeleton preset. The mapping report records these axes for validation.
        wrist_main, wrist_second = wrist_axes
        qpos[0] = clip(wrist[wrist_main], -math.pi / 2.0, math.pi / 2.0)
        qpos[1] = clip(wrist[wrist_second], -0.2618, 0.2618)
        remaining = ({0, 1, 2} - {wrist_main, wrist_second}).pop()
        qpos[2] = clip(wrist[remaining], -0.7854, 0.7854)

        thumb_meta = vectors["f1_metacarpal"][index]
        thumb_axis, thumb_second = axes["f1_metacarpal"]
        qpos[3] = clip(thumb_meta[thumb_second], -0.5, 0.78)
        qpos[4] = clip(flex_sign["f1_metacarpal"] * thumb_meta[thumb_axis], -0.78, 0.70)
        for q_index, bone_name, lower, upper in (
            (5, "f1_proximal", -0.7854, 0.6981),
            (6, "f1_distal", -1.309, 0.4363),
        ):
            axis, _ = axes[bone_name]
            qpos[q_index] = clip(flex_sign[bone_name] * vectors[bone_name][index][axis], lower, upper)

        for digit, base in FINGER_BASE.items():
            proximal = f"f{digit}_proximal"
            mcp_axis, abd_axis = axes[proximal]
            qpos[base] = clip(flex_sign[proximal] * vectors[proximal][index][mcp_axis], 0.0, FLEX_LIMIT)
            qpos[base + 1] = clip(vectors[proximal][index][abd_axis], -0.2618, 0.2618)
            for offset, segment in ((2, "medial"), (3, "distal")):
                name = f"f{digit}_{segment}"
                axis, _ = axes[name]
                qpos[base + offset] = clip(flex_sign[name] * vectors[name][index][axis], 0.0, FLEX_LIMIT)
        qpos_rows.append(qpos)

    fps = float(scene.render.fps) / max(1, int(scene.render.fps_base))
    csv_path = output_dir / "qpos.csv"
    columns = ["frame_index", "t", "progress_percent", "subject_id", "trial_label", "label", "trial_id", "exercise_name", "motion_type"]
    columns.extend(f"qpos_{index}" for index in range(QPOS_SIZE))
    trial_id = output_dir.name
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, qpos in enumerate(qpos_rows):
            row = {
                "frame_index": index,
                "t": index / fps,
                "progress_percent": 100.0 * index / max(1, len(qpos_rows) - 1),
                "subject_id": args.subject_id,
                "trial_label": args.trial_label,
                "label": args.trial_label,
                "trial_id": trial_id,
                "exercise_name": args.motion_type,
                "motion_type": args.motion_type,
            }
            row.update({f"qpos_{joint}": value for joint, value in enumerate(qpos)})
            writer.writerow(row)

    report = {
        "source_fbx": str(input_path), "armature": armature.name, "hand": args.hand,
        "motion_type": args.motion_type, "frame_start": start, "frame_end": end,
        "num_frames": len(frames), "fps": fps, "qpos_dim": QPOS_SIZE,
        "calibration_frame_indices": [frames[index] for index in calibration],
        "dominant_rotation_axes": {**{name: list(value) for name, value in axes.items()}, "wrist": list(wrist_axes)},
        "note": "Automatic axis/sign inference. Validate qpos in MuJoCo before dataset use.",
    }
    (output_dir / "rokoko_fbx_mapping_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    meta = {
        "trial_id": trial_id, "subject_id": args.subject_id, "trial_label": args.trial_label,
        "label": args.trial_label, "motion_type": args.motion_type, "exercise_name": args.motion_type,
        "num_frames": len(frames), "qpos_dim": QPOS_SIZE, "dt": 1.0 / fps,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_mode": "rokoko_fbx", "source_fbx": str(input_path), "qpos_csv": str(csv_path),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Rokoko FBX] wrote {len(qpos_rows)} frames to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
