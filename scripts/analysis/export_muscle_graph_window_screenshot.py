from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "paper_figures" / "fig_graph_muscle_relation_paper.png"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.gui.muscle_graph_controller import MuscleGraphController


def main() -> int:
    parser = argparse.ArgumentParser(description="Export clean Muscle Graph viewer screenshot for book/report figures.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k-edges", type=int, default=15)
    parser.add_argument("--width", type=int, default=1900)
    parser.add_argument("--height", type=int, default=1220)
    args = parser.parse_args()

    controller = MuscleGraphController(
        width=int(args.width),
        height=int(args.height),
        top_k_edges=int(args.top_k_edges),
    )
    output_path = controller.save_paper_image(Path(args.output), top_k_edges=int(args.top_k_edges))
    print(f"saved={output_path}")
    print(f"nodes={0 if controller.nodes is None else len(controller.nodes)}")
    print(f"edges={int(args.top_k_edges)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
