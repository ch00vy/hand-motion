from __future__ import annotations

"""Canonical offline biomechanical ctrl-reference builder.

This is the final-architecture entrypoint for:
    recorded qpos -> qvel/qacc -> inverse dynamics -> tau_target
    -> bounded actuator solve -> ctrl_ref

The current implementation reuses the existing trajectory-inverse builder.
"""

import sys
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.offline_biomech.biomech_reference_backend import main as backend_main  # noqa: E402


def _has_option(argv: list[str], option: str) -> bool:
    return any(arg == option or str(arg).startswith(f"{option}=") for arg in argv)


def _option_value(argv: list[str], option: str, default: str) -> str:
    for idx, arg in enumerate(argv):
        if arg == option and idx + 1 < len(argv):
            return str(argv[idx + 1])
        if str(arg).startswith(f"{option}="):
            return str(arg).split("=", 1)[1]
    return default


def _write_clean_output_aliases(output_dir: Path) -> None:
    aliases = {
        "teacher_trial_summary.csv": "biomech_reference_trial_summary.csv",
        "teacher_dataset_summary.json": "biomech_reference_dataset_summary.json",
        "teacher_dataset_combined.csv": "biomech_reference_dataset_combined.csv",
    }
    for legacy_name, clean_name in aliases.items():
        src = output_dir / legacy_name
        if src.exists():
            shutil.copy2(src, output_dir / clean_name)


def main() -> int:
    argv = list(sys.argv[1:])
    if not _has_option(argv, "--output-path"):
        argv.extend(["--output-path", str(PROJECT_ROOT / "artifacts" / "offline_biomech_ctrl_reference")])
    if not _has_option(argv, "--report-path"):
        argv.extend(["--report-path", str(PROJECT_ROOT / "reports" / "offline_biomech_ctrl_reference.md")])
    sys.argv = [sys.argv[0], *argv]
    exit_code = backend_main()
    if int(exit_code) == 0:
        _write_clean_output_aliases(Path(_option_value(argv, "--output-path", str(PROJECT_ROOT / "artifacts" / "offline_biomech_ctrl_reference"))).resolve())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
