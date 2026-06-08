from __future__ import annotations

"""Canonical offline biomechanical reference-sequence builder."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_reference_graph_sequence import main as legacy_main  # noqa: E402


def main() -> int:
    return legacy_main()


if __name__ == "__main__":
    raise SystemExit(main())
