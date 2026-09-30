from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.gui.window_layout import MUSCLE_GRAPH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NODES_CSV = PROJECT_ROOT / "artifacts" / "final_book_figures" / "neo4j_export" / "neo4j_muscle_nodes.csv"
DEFAULT_EDGES_CSV = PROJECT_ROOT / "artifacts" / "final_book_figures" / "neo4j_export" / "neo4j_muscle_edges.csv"

RELATION_BGR = {
    "synergist": (70, 180, 65),
    "antagonist": (35, 145, 235),
    "functional_group": (120, 115, 105),
}
LIVE_CORE_NODES = {
    # Thumb
    "APL", "EPB", "EPL", "OP", "FPL",

    # Index
    "EIP", "EDC2", "FDS2", "FDP2",

    # Middle
    "EDC3", "FDS3", "FDP3",

    # Ring
    "EDC4", "FDS4", "FDP4",

    # Little
    "EDM", "EDC5", "FDS5", "FDP5",

    # Wrist / forearm
    "ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ",
}

CORE_EDGE_MUSCLES = LIVE_CORE_NODES


@dataclass
class MuscleGraphStatus:
    open: bool
    message: str
    nodes_displayed: int = 0
    edges_displayed: int = 0


class MuscleGraphController:
    """External OpenCV muscle relation graph viewer.

    Node merepresentasikan otot/aktuator dan bersifat statis. Aktivitas otot
    tidak digunakan untuk menghilangkan atau mengubah ukuran node pada
    visualisasi graph hubungan antarotot.
    """

    def __init__(
        self,
        *,
        nodes_csv: Path = DEFAULT_NODES_CSV,
        edges_csv: Path = DEFAULT_EDGES_CSV,
        top_k_edges: int = 50,
        show_all_edges: bool = False,
        window_name: str = "Muscle Relation Graph",
        width: int = MUSCLE_GRAPH.width,
        height: int = MUSCLE_GRAPH.height,
    ):
        self.nodes_csv = Path(nodes_csv)
        self.edges_csv = Path(edges_csv)
        self.top_k_edges = max(1, int(top_k_edges))
        self.show_all_edges = bool(show_all_edges)
        self.window_name = str(window_name)
        self.width = max(640, int(width))
        self.height = max(520, int(height))
        self.nodes: pd.DataFrame | None = None
        self.edges: pd.DataFrame | None = None
        self.static_edges: pd.DataFrame | None = None
        self.primary_nodes: set[str] = set()
        self.positions: dict[str, tuple[int, int]] = {}
        self.node_status: dict[str, str] = {}
        self.is_open = False
        self.message = "Muscle graph viewer closed."
        self._cv2 = None

    def open_or_focus(self) -> bool:
        if not self._ensure_loaded():
            return False
        try:
            import cv2
        except Exception as exc:
            self.message = f"OpenCV unavailable for muscle graph: {type(exc).__name__}: {exc}"
            return False
        self._cv2 = cv2
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(self.window_name, self.width, self.height)
            cv2.moveWindow(self.window_name, MUSCLE_GRAPH.x, MUSCLE_GRAPH.y)
            cv2.waitKey(1)
            self.is_open = True
            self.message = "Muscle graph viewer open."
            self.update(mode="IDLE", ctrl_values=None)
            return True
        except Exception as exc:
            self.is_open = False
            self.message = f"Failed to open muscle graph viewer: {type(exc).__name__}: {exc}"
            return False

    def close(self) -> None:
        if self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        self.is_open = False
        self.message = "Muscle graph viewer closed."

    def set_analysis_result(self, result: Optional[dict[str, Any]]) -> None:
        self.node_status = {}
        if not isinstance(result, dict):
            return
        for row in list(result.get("nodes") or []):
            name = str(row.get("node_name") or "")
            status = str(row.get("affected_final_status") or row.get("predicted_label") or "")
            if name:
                self.node_status[name] = status

    def update(
        self,
        *,
        mode: str,
        ctrl_values: Optional[np.ndarray],
        frame_index: int = 0,
        num_frames: int = 0,
    ) -> None:
        if not self.is_open:
            return
        if not self._ensure_loaded() or self._cv2 is None:
            return

        try:
            image = self._render(
                mode=str(mode),
                ctrl_values=ctrl_values,
                frame_index=int(frame_index),
                num_frames=int(num_frames),
            )

            if image is None or image.size == 0:
                raise RuntimeError("Rendered image is empty.")

            self._cv2.imshow(self.window_name, image)
            self._cv2.waitKey(1)
            self.message = f"Muscle graph active | mode={mode}"

        except Exception as exc:
            self.message = f"Muscle graph update failed: {type(exc).__name__}: {exc}"
            print(self.message)

            error_image = self._render_error_image(self.message)
            self._cv2.imshow(self.window_name, error_image)
            self._cv2.waitKey(1)

    def _render_error_image(self, message: str) -> np.ndarray:
        import cv2

        canvas = np.full((self.height, self.width, 3), (20, 20, 20), dtype=np.uint8)

        cv2.putText(
            canvas,
            "Muscle graph render error",
            (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (80, 180, 255),
            2,
            cv2.LINE_AA,
        )

        max_chars = 95
        words = str(message).split()
        lines: list[str] = []
        current = ""

        for word in words:
            if len(current) + len(word) + 1 > max_chars:
                lines.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()

        if current:
            lines.append(current)

        y = 135
        for line in lines[:8]:
            cv2.putText(
                canvas,
                line,
                (40, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            y += 30

        cv2.putText(
            canvas,
            "Check terminal for the full error message.",
            (40, self.height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def status(self) -> MuscleGraphStatus:
        return MuscleGraphStatus(
            open=bool(self.is_open),
            message=str(self.message),
            nodes_displayed=0 if self.nodes is None else int(len(self.nodes)),
            edges_displayed=0 if self.edges is None else int(len(self.edges)),
        )

    def render_paper_image(self, *, top_k_edges: int | None = None) -> np.ndarray:
        """Render a clean book/report image using the same OpenCV viewer pipeline.

        This is intended for screenshot-style paper figures: no debug overlay, no
        runtime status text, no diagnostic/clinical terminology.
        """
        if not self._ensure_loaded():
            raise RuntimeError(self.message)
        import cv2

        nodes = self.nodes if self.nodes is not None else pd.DataFrame()
        edges = self.edges if self.edges is not None else pd.DataFrame()
        top_k = max(1, int(top_k_edges if top_k_edges is not None else min(self.top_k_edges, 15)))
        positions = self._paper_positions_px(nodes)
        render_edges = self._paper_top_edges(edges, positions, top_k)
        canvas = np.full((self.height, self.width, 3), 255, dtype=np.uint8)

        title = "Graph Relasi Antarotot pada Gesture Open-to-Fist"
        subtitle = "Node = otot/aktuator otot | Edge = relasi antarotot"
        cv2.putText(canvas, title, (self.width // 2 - 430, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (31, 41, 55), 2, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (self.width // 2 - 260, 93), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (71, 85, 105), 1, cv2.LINE_AA)

        grid_color = (233, 238, 244)
        for x in (int(self.width * 0.20), int(self.width * 0.40), int(self.width * 0.60), int(self.width * 0.80)):
            cv2.line(canvas, (x, 165), (x, self.height - 145), grid_color, 1, cv2.LINE_AA)
        for y, label in (
            (int(self.height * 0.28), "Finger-specific"),
            (int(self.height * 0.58), "Shared finger muscles"),
            (int(self.height * 0.84), "Wrist / Forearm"),
        ):
            cv2.line(canvas, (80, y), (self.width - 80, y), grid_color, 1, cv2.LINE_AA)
            cv2.putText(canvas, label, (30, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

        for label, x in (
            ("Thumb", int(self.width * 0.10)),
            ("Index", int(self.width * 0.30)),
            ("Middle", int(self.width * 0.50)),
            ("Ring", int(self.width * 0.70)),
            ("Little", int(self.width * 0.90)),
        ):
            cv2.putText(canvas, label, (x - 42, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (71, 85, 105), 2, cv2.LINE_AA)

        max_weight = max(1e-12, float(render_edges["edge_weight"].max())) if not render_edges.empty else 1.0
        for row in render_edges.to_dict(orient="records"):
            src = str(row["source_muscle"])
            dst = str(row["target_muscle"])
            if src not in positions or dst not in positions:
                continue
            relation = str(row["relation_type"])
            color = RELATION_BGR.get(relation, RELATION_BGR["functional_group"])
            weight = float(row.get("edge_weight", 0.0))
            width = int(max(1, min(5, round(1.0 + 4.0 * weight / max_weight))))
            p0 = positions[src]
            p1 = positions[dst]
            curve = self._curve_amount(src, dst, relation)
            span = abs(float(p1[0] - p0[0])) / max(1.0, float(self.width))
            if span > 0.18:
                sign = 1.0 if curve >= 0.0 else -1.0
                curve += sign * (0.12 if span > 0.42 else 0.08 if span > 0.30 else 0.045)
                color = self._blend_bgr(color, 0.70 if span > 0.42 else 0.82)
            self._draw_curve(canvas, p0, p1, color=color, width=width, curve=curve)

        for row in nodes.to_dict(orient="records"):
            name = str(row["muscle_name"])
            group = str(row.get("muscle_group") or "")
            if name not in positions:
                continue
            x, y = positions[name]
            family = self._family(name, group)
            face = {
                "thumb": (246, 240, 232),
                "digit2": (255, 246, 239),
                "digit3": (245, 253, 236),
                "digit4": (235, 251, 255),
                "digit5": (250, 242, 253),
                "wrist": (248, 250, 252),
                "other": (248, 250, 252),
            }.get(family, (248, 250, 252))
            radius = 14
            cv2.circle(canvas, (x, y), radius, face, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), radius, (51, 65, 85), 2, cv2.LINE_AA)
            lx, ly, align_right = self._paper_label_px(name, x, y)
            cv2.line(canvas, (x, y), (lx, ly), (203, 213, 225), 1, cv2.LINE_AA)
            label_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0]
            tx = lx - label_size[0] if align_right else lx
            cv2.putText(canvas, name, (tx, ly + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (31, 41, 55), 1, cv2.LINE_AA)

        self._draw_paper_legend(canvas)
        return canvas

    def save_paper_image(self, output_path: Path, *, top_k_edges: int | None = None) -> Path:
        import cv2

        image = self.render_paper_image(top_k_edges=top_k_edges)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Failed to write paper graph image: {output_path}")
        return output_path

    def _ensure_loaded(self) -> bool:
        if self.nodes is not None and self.edges is not None:
            return True
        if not self.nodes_csv.exists() or not self.edges_csv.exists():
            self.message = f"Muscle graph CSV not found: {self.nodes_csv} / {self.edges_csv}"
            return False
        try:
            nodes = pd.read_csv(self.nodes_csv)
            edges = pd.read_csv(self.edges_csv)
            nodes["muscle_name"] = nodes["muscle_name"].astype(str)
            edges["source_muscle"] = edges["source_muscle"].astype(str)
            edges["target_muscle"] = edges["target_muscle"].astype(str)
            edges["relation_type"] = edges["relation_type"].astype(str)
            edges["edge_weight"] = pd.to_numeric(edges["edge_weight"], errors="coerce").fillna(0.0)
            edges = edges.loc[edges["edge_weight"] > 0.0].copy()
            edges = self._deduplicate_edges(edges)
            static_edges = self._balanced_top_edges(edges, self.top_k_edges)

            self.nodes = nodes
            self.edges = edges
            self.static_edges = static_edges
            self.primary_nodes = self._edge_node_set(static_edges)
            self.positions = self._live_positions_px(nodes)
            self.message = f"Muscle graph loaded: {len(nodes)} nodes, {len(edges)} candidate edges."
            return True
        except Exception as exc:
            self.message = f"Failed to load muscle graph CSV: {type(exc).__name__}: {exc}"
            return False

    def _balanced_top_edges(self, edges: pd.DataFrame, top_k: int) -> pd.DataFrame:
        edges = edges.sort_values("edge_weight", ascending=False)
        preferred = edges.loc[
            edges["source_muscle"].isin(CORE_EDGE_MUSCLES) & edges["target_muscle"].isin(CORE_EDGE_MUSCLES)
        ].copy()
        if len(preferred) >= max(8, min(top_k, 18)):
            edges = preferred
        selected = []
        used: set[int] = set()
        for relation in ("synergist", "antagonist", "functional_group"):
            part = edges.loc[edges["relation_type"] == relation].head(max(1, min(5, top_k // 3)))
            selected.append(part)
            used.update(int(idx) for idx in part.index)
        remaining_count = max(0, top_k - sum(len(item) for item in selected))
        remaining = edges.loc[~edges.index.isin(used)].head(remaining_count)
        return pd.concat([*selected, remaining], axis=0).drop_duplicates().sort_values("edge_weight", ascending=False).head(top_k)

    def _paper_top_edges(self, edges: pd.DataFrame, positions: dict[str, tuple[int, int]], top_k: int) -> pd.DataFrame:
        """Select visually readable paper edges without changing graph data.

        The base graph remains unchanged. This only filters the paper figure so
        very long cross-column edges do not dominate the book illustration.
        """
        if edges.empty:
            return edges
        candidates = self._balanced_top_edges(edges, max(int(top_k) * 4, int(top_k)))
        if candidates.empty:
            return candidates
        items = []
        for row in candidates.to_dict(orient="records"):
            src = str(row["source_muscle"])
            dst = str(row["target_muscle"])
            p0 = positions.get(src)
            p1 = positions.get(dst)
            span = 0.0 if p0 is None or p1 is None else abs(float(p1[0] - p0[0])) / max(1.0, float(self.width))
            weight = float(row.get("edge_weight", 0.0))
            score = weight / (1.0 + 3.0 * span * span)
            item = dict(row)
            item["_paper_span"] = span
            item["_paper_score"] = score
            items.append(item)
        ranked = pd.DataFrame(items).sort_values(["_paper_score", "edge_weight"], ascending=False)
        selected = ranked.head(max(1, int(top_k))).copy()
        return selected.drop(columns=[col for col in ("_paper_span", "_paper_score") if col in selected.columns])

    def _deduplicate_edges(self, edges: pd.DataFrame) -> pd.DataFrame:
        if edges.empty:
            return edges
        out = edges.copy()
        out["_pair"] = out.apply(
            lambda row: "::".join(sorted([str(row["source_muscle"]), str(row["target_muscle"])])),
            axis=1,
        )
        out = (
            out.sort_values("edge_weight", ascending=False)
            .drop_duplicates(subset=["_pair", "relation_type"], keep="first")
            .drop(columns=["_pair"])
        )
        return out

    def _paper_positions_px(self, nodes: pd.DataFrame) -> dict[str, tuple[int, int]]:
        w = int(self.width)
        h = int(self.height)
        x_by_digit = {
            "thumb": int(w * 0.10),
            "digit2": int(w * 0.30),
            "digit3": int(w * 0.50),
            "digit4": int(w * 0.70),
            "digit5": int(w * 0.90),
        }
        y_by_name = {
            "APL": int(h * 0.23), "EPB": int(h * 0.34), "EPL": int(h * 0.45), "OP": int(h * 0.56), "FPL": int(h * 0.67),
            "EIP": int(h * 0.21), "EDC2": int(h * 0.33), "RI2": int(h * 0.44), "LU_RB2": int(h * 0.55), "UI_UB2": int(h * 0.66), "FDS2": int(h * 0.76), "FDP2": int(h * 0.84),
            "EDC3": int(h * 0.34), "RI3": int(h * 0.45), "LU_RB3": int(h * 0.56), "UI_UB3": int(h * 0.67), "FDS3": int(h * 0.755), "FDP3": int(h * 0.855),
            "EDC4": int(h * 0.34), "RI4": int(h * 0.45), "LU_RB4": int(h * 0.56), "UI_UB4": int(h * 0.67), "FDS4": int(h * 0.77), "FDP4": int(h * 0.84),
            "EDM": int(h * 0.18), "EDC5": int(h * 0.31), "RI5": int(h * 0.45), "LU_RB5": int(h * 0.59), "UI_UB5": int(h * 0.715), "FDS5": int(h * 0.805), "FDP5": int(h * 0.885),
        }
        wrist_positions = {
            "ECRL": (int(w * 0.14), int(h * 0.905)),
            "ECRB": (int(w * 0.27), int(h * 0.905)),
            "ECU": (int(w * 0.41), int(h * 0.905)),
            "FCR": (int(w * 0.55), int(h * 0.905)),
            "FCU": (int(w * 0.66), int(h * 0.905)),
            "PL": (int(w * 0.76), int(h * 0.905)),
            "PT": (int(w * 0.86), int(h * 0.905)),
            "PQ": (int(w * 0.95), int(h * 0.905)),
        }
        positions: dict[str, tuple[int, int]] = {}
        for row in nodes.to_dict(orient="records"):
            name = str(row["muscle_name"])
            if name in wrist_positions:
                positions[name] = wrist_positions[name]
                continue
            family = self._family(name, str(row.get("muscle_group") or ""))
            positions[name] = (x_by_digit.get(family, int(w * 0.50)), y_by_name.get(name, int(h * 0.55)))
        return positions

    def _paper_label_px(self, name: str, x: int, y: int) -> tuple[int, int, bool]:
        if name in {"ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ"}:
            return x, y + 36, False
        if name in {"EDM", "EDC5", "RI5", "LU_RB5", "UI_UB5"}:
            return x - 30, y - 16, True
        if name in {"FDS5", "FDP5"}:
            return x + 24, y + 6, False
        if name in {"FDP2", "FDP3", "FDP4", "FDS2", "FDS3", "FDS4"}:
            return x + 24, y + 12, False
        return x + 24, y - 12, False

    def _curve_amount(self, src: str, dst: str, relation: str) -> float:
        code = sum(ord(ch) for ch in f"{src}:{dst}:{relation}")
        return float(((code % 7) - 3) / 100.0)

    def _blend_bgr(self, color: tuple[int, int, int], strength: float) -> tuple[int, int, int]:
        strength = float(np.clip(strength, 0.0, 1.0))
        return tuple(int(round((255.0 * (1.0 - strength)) + (float(channel) * strength))) for channel in color)

    def _draw_curve(
        self,
        canvas: np.ndarray,
        p0: tuple[int, int],
        p1: tuple[int, int],
        *,
        color: tuple[int, int, int],
        width: int,
        curve: float,
    ) -> None:
        import cv2

        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        length = max(1.0, float(np.hypot(dx, dy)))
        nx = -dy / length
        ny = dx / length
        offset = float(curve) * length
        cx = (x0 + x1) / 2.0 + nx * offset
        cy = (y0 + y1) / 2.0 + ny * offset
        pts = []
        for t in np.linspace(0.0, 1.0, 36):
            xa = (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * cx + t * t * x1
            ya = (1 - t) * (1 - t) * y0 + 2 * (1 - t) * t * cy + t * t * y1
            pts.append([int(round(xa)), int(round(ya))])
        cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, color, max(1, int(width)), cv2.LINE_AA)

    def _draw_paper_legend(self, canvas: np.ndarray) -> None:
        import cv2

        y = self.height - 24
        x = int(self.width * 0.27)
        cv2.circle(canvas, (x, y - 5), 7, (248, 250, 252), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y - 5), 7, (51, 65, 85), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Node = otot/aktuator otot", (x + 22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (31, 41, 55), 1, cv2.LINE_AA)
        x += 230
        for label, color in (
            ("Sinergis", RELATION_BGR["synergist"]),
            ("Antagonis", RELATION_BGR["antagonist"]),
            ("Kelompok fungsi", RELATION_BGR["functional_group"]),
        ):
            cv2.line(canvas, (x, y - 7), (x + 36, y - 7), color, 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (x + 45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (31, 41, 55), 1, cv2.LINE_AA)
            x += 170

    def _edge_node_set(self, edges: pd.DataFrame | None) -> set[str]:
        if edges is None or edges.empty:
            return set()
        out: set[str] = set()
        out.update(edges["source_muscle"].astype(str).tolist())
        out.update(edges["target_muscle"].astype(str).tolist())
        return out

    def _select_dynamic_edges(self, edges: pd.DataFrame, ctrl_by_name: dict[str, float]) -> pd.DataFrame:
        """Select relation edges for the current simulation frame.
        Jika self.show_all_edges=True, semua edge yang tersedia akan dikembalikan
        tanpa top-k dan tanpa primary_nodes filtering.
        """
        if edges.empty:
            return edges

        ctrl_available = any(float(value) > 1e-5 for value in ctrl_by_name.values())

        # DEBUG MODE: tampilkan semua edge.
        if self.show_all_edges:
            rows = []

            for row in edges.to_dict(orient="records"):
                src = str(row["source_muscle"])
                dst = str(row["target_muscle"])
                relation = str(row.get("relation_type", ""))

                src_act = float(ctrl_by_name.get(src, 0.0))
                dst_act = float(ctrl_by_name.get(dst, 0.0))

                if ctrl_available:
                    if relation == "synergist":
                        pair_activity = float(np.sqrt(max(0.0, src_act) * max(0.0, dst_act)))
                    elif relation == "antagonist":
                        pair_activity = max(abs(src_act - dst_act), 0.50 * max(src_act, dst_act))
                    else:
                        pair_activity = 0.5 * (src_act + dst_act)
                else:
                    # Kalau belum ada ctrl, semua edge tetap terlihat.
                    pair_activity = 0.65

                pair_activity = float(np.clip(pair_activity, 0.0, 1.0))

                item = dict(row)
                item["pair_activity"] = pair_activity
                item["dynamic_edge_weight"] = float(row.get("edge_weight", 0.0)) * pair_activity
                rows.append(item)

            return pd.DataFrame(rows).reset_index(drop=True)

        # MODE NORMAL: tetap pakai filter/top-k lama.
        ctrl_available = any(float(value) > 1e-5 for value in ctrl_by_name.values())
        if not ctrl_available:
            return (
                self.static_edges.copy()
                if self.static_edges is not None
                else self._balanced_top_edges(edges, self.top_k_edges)
            )

        preferred = edges.loc[
            edges["source_muscle"].isin(CORE_EDGE_MUSCLES)
            & edges["target_muscle"].isin(CORE_EDGE_MUSCLES)
        ].copy()

        source = preferred if len(preferred) >= max(8, min(self.top_k_edges, 18)) else edges.copy()

        if self.primary_nodes:
            source = source.loc[
                source["source_muscle"].isin(self.primary_nodes)
                & source["target_muscle"].isin(self.primary_nodes)
            ].copy()

        rows = []
        for row in source.to_dict(orient="records"):
            src = str(row["source_muscle"])
            dst = str(row["target_muscle"])
            relation = str(row.get("relation_type", ""))

            src_act = float(ctrl_by_name.get(src, 0.0))
            dst_act = float(ctrl_by_name.get(dst, 0.0))

            if relation == "synergist":
                pair_activity = float(np.sqrt(max(0.0, src_act) * max(0.0, dst_act)))
            elif relation == "antagonist":
                pair_activity = max(abs(src_act - dst_act), 0.50 * max(src_act, dst_act))
            else:
                pair_activity = 0.5 * (src_act + dst_act)

            pair_activity = float(np.clip(pair_activity, 0.0, 1.0))

            item = dict(row)
            item["pair_activity"] = pair_activity
            item["dynamic_edge_weight"] = float(row["edge_weight"]) * pair_activity
            rows.append(item)

        dynamic = pd.DataFrame(rows)

        if dynamic.empty:
            return dynamic

        max_activity = float(dynamic["pair_activity"].max())

        if max_activity > 0.02:
            dynamic = dynamic.loc[
                dynamic["pair_activity"] >= max(0.02, 0.15 * max_activity)
            ].copy()

        if dynamic.empty:
            return dynamic

        dynamic = dynamic.sort_values("dynamic_edge_weight", ascending=False)

        return dynamic.head(self.top_k_edges).reset_index(drop=True)

    def _live_positions_px(self, nodes: pd.DataFrame) -> dict[str, tuple[int, int]]:
        """Hand-like vertical layout for the live OpenCV muscle graph.

        Tujuan layout:
        - Tidak lagi berbentuk grid kotak.
        - Jari dibuat memanjang ke bawah.
        - Thumb dibuat miring di sisi kiri.
        - Wrist/forearm ditempatkan di bawah seperti pangkal tangan.
        """
        w = int(self.width)
        h = int(self.height)

        palm_x = int(w * 0.50)

        x_by_digit = {
            "digit2": int(w * 0.38),
            "digit3": int(w * 0.50),
            "digit4": int(w * 0.62),
            "digit5": int(w * 0.74),
        }

        thumb_positions = {
            "APL": (int(w * 0.22), int(h * 0.30)),
            "EPB": (int(w * 0.18), int(h * 0.39)),
            "EPL": (int(w * 0.20), int(h * 0.48)),
            "OP": (int(w * 0.27), int(h * 0.57)),
            "FPL": (int(w * 0.35), int(h * 0.66)),
        }

        finger_y = {
            "extensor_top": int(h * 0.18),
            "extensor_mid": int(h * 0.28),
            "intrinsic": int(h * 0.42),
            "flexor_mid": int(h * 0.58),
            "flexor_low": int(h * 0.68),
        }

        y_by_name = {
            # Index
            "EIP": finger_y["extensor_top"],
            "EDC2": finger_y["extensor_mid"],
            "RI2": finger_y["intrinsic"],
            "LU_RB2": int(h * 0.48),
            "UI_UB2": int(h * 0.53),
            "FDS2": finger_y["flexor_mid"],
            "FDP2": finger_y["flexor_low"],

            # Middle
            "EDC3": finger_y["extensor_mid"],
            "RI3": finger_y["intrinsic"],
            "LU_RB3": int(h * 0.48),
            "UI_UB3": int(h * 0.53),
            "FDS3": finger_y["flexor_mid"],
            "FDP3": finger_y["flexor_low"],

            # Ring
            "EDC4": finger_y["extensor_mid"],
            "RI4": finger_y["intrinsic"],
            "LU_RB4": int(h * 0.48),
            "UI_UB4": int(h * 0.53),
            "FDS4": finger_y["flexor_mid"],
            "FDP4": finger_y["flexor_low"],

            # Little
            "EDM": int(h * 0.22),
            "EDC5": finger_y["extensor_mid"],
            "RI5": finger_y["intrinsic"],
            "LU_RB5": int(h * 0.48),
            "UI_UB5": int(h * 0.53),
            "FDS5": finger_y["flexor_mid"],
            "FDP5": finger_y["flexor_low"],
        }

        wrist_positions = {
            "ECRL": (int(w * 0.31), int(h * 0.80)),
            "ECRB": (int(w * 0.41), int(h * 0.83)),
            "ECU": (int(w * 0.52), int(h * 0.85)),
            "FCR": (int(w * 0.62), int(h * 0.83)),
            "FCU": (int(w * 0.72), int(h * 0.80)),
            "PL": (int(w * 0.44), int(h * 0.91)),
            "PT": (int(w * 0.55), int(h * 0.94)),
            "PQ": (int(w * 0.66), int(h * 0.91)),
        }

        positions: dict[str, tuple[int, int]] = {}

        for row in nodes.to_dict(orient="records"):
            name = str(row["muscle_name"])
            group = str(row.get("muscle_group") or "")

            if name in thumb_positions:
                positions[name] = thumb_positions[name]
                continue

            if name in wrist_positions:
                positions[name] = wrist_positions[name]
                continue

            family = self._family(name, group)
            if family in x_by_digit:
                x = x_by_digit[family]
                y = y_by_name.get(name, int(h * 0.50))
            else:
                x = palm_x
                y = int(h * 0.55)

            positions[name] = (int(x), int(y))

        return positions

    def _put_live_label(self, canvas: np.ndarray, name: str, x: int, y: int) -> None:
        """Draw node label with simple collision-aware placement."""
        import cv2

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.34
        thickness = 1
        color = (188, 198, 212)

        label_size = cv2.getTextSize(name, font, scale, thickness)[0]
        label_w = int(label_size[0])

        wrist_names = {"ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ"}

        if name in wrist_names:
            tx = int(x - label_w // 2)
            ty = int(y + 34)

        elif name in {"APL", "EPB", "EPL", "OP", "FPL"}:
            tx = int(x - label_w - 16)
            ty = int(y + 4)

        elif x > int(self.width * 0.68):
            tx = int(x + 18)
            ty = int(y + 4)

        else:
            tx = int(x - label_w // 2)
            ty = int(y - 24)

        # Clamp supaya label tidak keluar canvas.
        tx = max(6, min(tx, int(self.width - label_w - 6)))
        ty = max(88, min(ty, int(self.height - 22)))

        cv2.putText(
            canvas,
            name,
            (tx, ty),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _positions(self, nodes: pd.DataFrame) -> dict[str, tuple[int, int]]:
        x_map = {
            "thumb": 150,
            "digit2": 365,
            "digit3": 540,
            "digit4": 705,
            "digit5": 875,
            "wrist": 520,
        }
        y_by_name = {
            "EIP": 170, "EDM": 210,
            "EDC2": 355, "EDC3": 350, "EDC4": 350, "EDC5": 390,
            "RI2": 470, "RI3": 470, "RI4": 470, "RI5": 495,
            "LU_RB2": 565, "LU_RB3": 565, "LU_RB4": 565, "LU_RB5": 575,
            "UI_UB2": 650, "UI_UB3": 650, "UI_UB4": 650, "UI_UB5": 650,
            "FDS2": 700, "FDS3": 700, "FDS4": 565, "FDS5": 620,
            "FDP2": 725, "FDP3": 725, "FDP4": 725, "FDP5": 765,
            "APL": 245, "EPB": 345, "EPL": 445, "OP": 545, "FPL": 645,
        }
        wrist_pos = {
            "ECRL": (930, 150), "ECRB": (520, 830), "ECU": (430, 860), "FCR": (790, 830),
            "FCU": (610, 875), "PL": (700, 870), "PT": (520, 895), "PQ": (600, 895),
        }
        out: dict[str, tuple[int, int]] = {}
        for row in nodes.to_dict(orient="records"):
            name = str(row["muscle_name"])
            group = str(row.get("muscle_group") or "")
            if name in wrist_pos:
                out[name] = wrist_pos[name]
                continue
            family = self._family(name, group)
            x = x_map.get(family, 700)
            y = y_by_name.get(name, 120)
            out[name] = (int(x), int(y))
        return out

    def _render(self, *, mode: str, ctrl_values: Optional[np.ndarray], frame_index: int, num_frames: int) -> np.ndarray:
        import cv2

        canvas = np.full((self.height, self.width, 3), (8, 10, 14), dtype=np.uint8)
        nodes = self.nodes if self.nodes is not None else pd.DataFrame()
        edges = self.edges if self.edges is not None else pd.DataFrame()
        ctrl = np.asarray(ctrl_values if ctrl_values is not None else [], dtype=float).reshape(-1)
        ctrl = np.where(np.isfinite(ctrl), ctrl, 0.0)
        ctrl = np.clip(ctrl, 0.0, 1.0)
        ctrl_by_name: dict[str, float] = {}
        for idx, name in enumerate(list(nodes["muscle_name"].astype(str))):
            ctrl_by_name[name] = float(ctrl[idx]) if idx < ctrl.size else 0.0

        # =========================
        # HAND-LIKE BACKGROUND GUIDE
        # =========================
        guide_color = (28, 31, 38)
        text_color = (130, 140, 150)

        status = f"mode={mode} | frame={frame_index + 1 if num_frames else 0}/{num_frames}"
        cv2.putText(
            canvas,
            status,
            (18, self.height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (105, 112, 124),
            1,
            cv2.LINE_AA,
        )

        finger_guides = [
            ("Index", int(self.width * 0.38)),
            ("Middle", int(self.width * 0.50)),
            ("Ring", int(self.width * 0.62)),
            ("Little", int(self.width * 0.74)),
        ]

        for label, x in finger_guides:
            cv2.line(
                canvas,
                (x, int(self.height * 0.15)),
                (x, int(self.height * 0.70)),
                guide_color,
                1,
                cv2.LINE_AA,
            )
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
            cv2.putText(
                canvas,
                label,
                (x - label_size[0] // 2, int(self.height * 0.12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                text_color,
                1,
                cv2.LINE_AA,
            )

        cv2.line(
            canvas,
            (int(self.width * 0.20), int(self.height * 0.31)),
            (int(self.width * 0.35), int(self.height * 0.66)),
            guide_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Thumb",
            (int(self.width * 0.12), int(self.height * 0.28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            text_color,
            1,
            cv2.LINE_AA,
        )

        cv2.ellipse(
            canvas,
            (int(self.width * 0.56), int(self.height * 0.61)),
            (int(self.width * 0.28), int(self.height * 0.18)),
            0,
            0,
            360,
            guide_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            "Palm / hand area",
            (int(self.width * 0.38), int(self.height * 0.76)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 100, 112),
            1,
            cv2.LINE_AA,
        )

        cv2.line(
            canvas,
            (int(self.width * 0.28), int(self.height * 0.78)),
            (int(self.width * 0.75), int(self.height * 0.78)),
            guide_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Wrist / forearm",
            (int(self.width * 0.39), int(self.height * 0.975)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (90, 100, 112),
            1,
            cv2.LINE_AA,
        )

        # =========================
        # EDGE RENDERING
        # =========================
        if self.show_all_edges:
            # Debug mode: tampilkan semua edge yang source dan target-nya punya posisi.
            render_edges = edges.copy()
            if not render_edges.empty:
                render_edges["pair_activity"] = 0.65
                render_edges["dynamic_edge_weight"] = render_edges["edge_weight"].astype(float)
        else:
            # Normal mode: pakai dynamic/top-k selection.
            render_edges = self._select_dynamic_edges(edges, ctrl_by_name)
            if self.primary_nodes and not render_edges.empty:
                render_edges = render_edges.loc[
                    render_edges["source_muscle"].isin(self.primary_nodes)
                    & render_edges["target_muscle"].isin(self.primary_nodes)
                ].copy()

        # Pastikan hanya edge yang kedua node-nya punya posisi yang digambar.
        if not render_edges.empty:
            render_edges = render_edges.loc[
                render_edges["source_muscle"].astype(str).isin(self.positions)
                & render_edges["target_muscle"].astype(str).isin(self.positions)
            ].copy()

        print(
            f"[MuscleGraph] show_all_edges={self.show_all_edges} | "
            f"edges_to_draw={len(render_edges)} | "
            f"nodes={len(nodes)} | positions={len(self.positions)}"
        )

        weight_column = "dynamic_edge_weight" if "dynamic_edge_weight" in render_edges.columns else "edge_weight"
        max_weight = max(1e-12, float(render_edges[weight_column].max())) if not render_edges.empty else 1.0

        for row in render_edges.to_dict(orient="records"):
            src = str(row["source_muscle"])
            dst = str(row["target_muscle"])

            if src not in self.positions or dst not in self.positions:
                continue

            relation = str(row["relation_type"])
            base_color = RELATION_BGR.get(relation, RELATION_BGR["functional_group"])

            pair_activity = float(row.get("pair_activity", 0.65))
            brightness = 0.25 + 0.75 * float(np.clip(pair_activity, 0.0, 1.0))

            color = tuple(int(c * brightness + 22 * (1.0 - brightness)) for c in base_color)
            dynamic_weight = float(row.get(weight_column, row.get("edge_weight", 0.0)))

            if self.show_all_edges:
                # Semua edge harus tipis agar tidak menutupi node.
                width = 1
            else:
                width = int(max(1, min(4, round(1.0 + 3.0 * dynamic_weight / max_weight))))

            p0 = self.positions[src]
            p1 = self.positions[dst]

            span = abs(float(p1[0] - p0[0])) / max(1.0, float(self.width))
            curve = self._curve_amount(src, dst, relation)

            if span > 0.35:
                curve += 0.10 if curve >= 0 else -0.10
            elif span > 0.20:
                curve += 0.06 if curve >= 0 else -0.06

            self._draw_curve(
                canvas,
                p0,
                p1,
                color=color,
                width=width,
                curve=curve,
            )

        for idx, row in enumerate(nodes.to_dict(orient="records")):
            name = str(row["muscle_name"])
            group = str(row.get("muscle_group") or "")
            if name not in self.positions:
                continue
            x, y = self.positions[name]
            activity = ctrl_by_name.get(name, 0.0)
            node_status = self.node_status.get(name, "")
            border = (125, 105, 85)
            if node_status == "terindikasi_menyimpang" or node_status == "deviating":
                border = (70, 70, 245)
            elif node_status == "berpotensi_terdampak":
                border = (0, 190, 245)
            # Node merepresentasikan otot/aktuator dan bersifat statis. Aktivitas otot
            # tidak digunakan untuk menghilangkan atau mengubah ukuran node pada
            # visualisasi graph hubungan antarotot.
            node_radius = 17
            outer_radius = 21
            cv2.circle(canvas, (x, y), node_radius, (68, 138, 202), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), outer_radius, (78, 63, 50), 1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), node_radius, border, 2, cv2.LINE_AA)
            if activity > 0.02:
                glow = int(max(3, min(9, round(3 + 6 * activity))))
                cv2.circle(canvas, (x, y), glow, (80, 180, 255), -1, cv2.LINE_AA)
            self._put_live_label(canvas, name, x, y)

        return canvas

    def _draw_legend(self, canvas: np.ndarray) -> None:
        import cv2

        x0, y0 = 36, self.height - 92
        items = [
            ("Node = muscle actuator (static)", (230, 235, 240)),
            ("green = synergist", RELATION_BGR["synergist"]),
            ("orange = antagonist", RELATION_BGR["antagonist"]),
            ("brown/gray = functional group", RELATION_BGR["functional_group"]),
            ("line width = relation strength; inner glow = current ctrl", (140, 150, 165)),
        ]
        for i, (text, color) in enumerate(items):
            y = y0 + i * 20
            if i == 0:
                cv2.circle(canvas, (x0 + 8, y - 4), 7, (68, 138, 202), -1, cv2.LINE_AA)
                cv2.circle(canvas, (x0 + 8, y - 4), 7, (120, 105, 88), 1, cv2.LINE_AA)
            else:
                cv2.line(canvas, (x0, y - 5), (x0 + 30, y - 5), color, 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (x0 + 45, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 218, 230), 1, cv2.LINE_AA)

    def _family(self, name: str, group: str) -> str:
        text = f"{name} {group}".lower()
        if "thumb" in text or name in {"EPL", "EPB", "FPL", "APL", "OP"}:
            return "thumb"
        if "digit2" in text or name.endswith("2"):
            return "digit2"
        if "digit3" in text or name.endswith("3"):
            return "digit3"
        if "digit4" in text or name.endswith("4"):
            return "digit4"
        if "digit5" in text or name.endswith("5") or name == "EDM":
            return "digit5"
        if "wrist" in text or name in {"ECRL", "ECRB", "ECU", "FCR", "FCU", "PL", "PT", "PQ"}:
            return "wrist"
        return "other"

    def _base_bgr(self, name: str, group: str) -> tuple[int, int, int]:
        family = self._family(name, group)
        palette = {
            "thumb": (235, 226, 210),
            "digit2": (235, 220, 205),
            "digit3": (220, 235, 222),
            "digit4": (198, 230, 245),
            "digit5": (232, 210, 232),
            "wrist": (232, 238, 244),
            "other": (230, 230, 230),
        }
        return palette.get(family, palette["other"])
