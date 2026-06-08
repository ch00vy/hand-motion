from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.offline_biomech.train_temporal_gcn_prototype import main as temporal_main


def _option_value(argv: list[str], name: str, default: str) -> str:
    if name not in argv:
        return str(default)
    index = argv.index(name)
    if index + 1 >= len(argv):
        return str(default)
    return str(argv[index + 1])


def main() -> int:
    """Train the active dynamic Temporal GCN node classifier.

    This is the migration command. Defaults are intentionally stronger than the
    old prototype: dynamic edge weights, explicit progress feature, attention
    pooling, abnormal validation, and report figures.
    """
    argv = list(sys.argv[1:])
    skip_figures = "--skip-figures" in argv
    if skip_figures:
        argv.remove("--skip-figures")
    if "--model-variant" not in argv:
        argv.extend(["--model-variant", "dynamic"])
    if "--output-dir" not in argv:
        argv.extend(["--output-dir", "artifacts/dynamic_temporal_gcn_node_classifier"])
    if "--temporal-pooling" not in argv:
        argv.extend(["--temporal-pooling", "attention"])
    if "--add-progress-feature" not in argv:
        argv.append("--add-progress-feature")
    if "--class-weighting" not in argv:
        argv.append("--class-weighting")
    sys.argv = [sys.argv[0], *argv]
    result = temporal_main()
    if result == 0 and not skip_figures:
        from scripts.analysis.plot_temporal_gcn_training_results import main as plot_main

        output_dir = Path(_option_value(argv, "--output-dir", "artifacts/dynamic_temporal_gcn_node_classifier"))
        sys.argv = [
            sys.argv[0],
            "--run-dir",
            str(output_dir),
            "--output-dir",
            str(output_dir / "figures"),
            "--title-prefix",
            "Dynamic Temporal GCN",
        ]
        plot_main()
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
