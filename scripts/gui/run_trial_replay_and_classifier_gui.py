from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gui.main_window import main as gui_main, smoke_test  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the trial replay + final binary GCN classifier GUI.")
    parser.add_argument("--smoke-test", action="store_true", help="Check imports and key paths without opening the GUI.")
    parser.add_argument(
        "--show-external-leap-skeleton",
        action="store_true",
        help="Compatibility flag. The external Leap viewer now opens by default.",
    )
    parser.add_argument(
        "--no-auto-open-leap-viewer",
        action="store_true",
        help="Do not automatically open the external Leap viewer at GUI startup.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if bool(args.smoke_test):
        print(json.dumps(smoke_test(), indent=2))
        return 0
    gui_main(show_external_leap_skeleton=not bool(args.no_auto_open_leap_viewer))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
