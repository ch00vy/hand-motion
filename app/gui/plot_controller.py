from __future__ import annotations

import tkinter as tk
from typing import Any

from app.gui.style import COLORS, FONTS


class NodeDeviationPlot:
    """Tk canvas plots for final binary GCN node activity-pattern scores."""

    def __init__(self, bar_canvas: tk.Canvas, heatmap_canvas: tk.Canvas | None = None):
        self.bar_canvas = bar_canvas
        self.heatmap_canvas = heatmap_canvas

    def draw_empty(self, message: str = "Run analysis to show muscle activity-pattern plot.") -> None:
        self._draw_empty_canvas(self.bar_canvas, message)
        if self.heatmap_canvas is not None:
            self._draw_empty_canvas(self.heatmap_canvas, "Run analysis to show muscle-vs-progress activity-pattern heatmap.")

    def draw_result(self, result: dict[str, Any]) -> None:
        self._draw_bar_plot(result)
        if self.heatmap_canvas is not None:
            self._draw_heatmap(result)

    def _draw_empty_canvas(self, canvas: tk.Canvas, message: str) -> None:
        canvas.delete("all")
        width = max(1, int(canvas.winfo_width()))
        height = max(1, int(canvas.winfo_height()))
        canvas.create_rectangle(0, 0, width, height, fill=COLORS["viewer"], outline=COLORS["border"])
        canvas.create_text(width // 2, height // 2, text=message, fill=COLORS["muted"], font=FONTS["body"])

    def _draw_bar_plot(self, result: dict[str, Any]) -> None:
        canvas = self.bar_canvas
        canvas.delete("all")
        width = max(1, int(canvas.winfo_width()))
        height = max(1, int(canvas.winfo_height()))
        canvas.create_rectangle(0, 0, width, height, fill=COLORS["viewer"], outline=COLORS["border"])
        rows = list(result.get("nodes") or [])
        if not rows:
            self._draw_empty_canvas(canvas, "No node results available.")
            return
        rows = sorted(
            rows,
            key=lambda item: (
                float(item.get("deviation_score", 0.0)),
                float(item.get("probability_deviating", 0.0)),
            ),
            reverse=True,
        )[:18]

        left = 138
        right = width - 24
        top = 30
        row_h = max(12, min(22, (height - top - 16) // max(1, len(rows))))
        max_score = max(1e-6, max(float(row.get("deviation_score", 0.0)) for row in rows))
        canvas.create_text(12, 12, anchor="nw", text="Skor Perbedaan Pola per Otot", fill=COLORS["text_soft"], font=FONTS["section"])

        for idx, row in enumerate(rows):
            y = top + idx * row_h
            node_name = str(row.get("node_name", ""))[:22]
            score = float(row.get("deviation_score", 0.0))
            label = str(row.get("predicted_label", ""))
            final_status = str(row.get("affected_final_status") or "")
            bar_w = int((right - left) * (score / max_score))
            if final_status == "berpotensi_terdampak":
                color = COLORS["warning"]
            elif final_status == "terindikasi_menyimpang" or label == "deviating":
                color = COLORS["danger"]
            else:
                color = COLORS["success"]
            canvas.create_text(12, y + row_h // 2, anchor="w", text=node_name, fill=COLORS["text"], font=FONTS["small"])
            canvas.create_rectangle(left, y + 3, right, y + row_h - 3, fill=COLORS["button_secondary"], outline="")
            canvas.create_rectangle(left, y + 3, left + bar_w, y + row_h - 3, fill=color, outline="")
            canvas.create_text(right, y + row_h // 2, anchor="e", text=f"{score:.2f}", fill=COLORS["text"], font=FONTS["mono"])

    def _draw_heatmap(self, result: dict[str, Any]) -> None:
        canvas = self.heatmap_canvas
        if canvas is None:
            return
        canvas.delete("all")
        width = max(1, int(canvas.winfo_width()))
        height = max(1, int(canvas.winfo_height()))
        canvas.create_rectangle(0, 0, width, height, fill=COLORS["viewer"], outline=COLORS["border"])
        heat_rows = list(result.get("node_progress_deviation") or [])
        if not heat_rows:
            self._draw_empty_canvas(canvas, "No per-progress muscle activity-pattern data available.")
            return

        node_scores: dict[str, float] = {}
        for row in heat_rows:
            node = str(row.get("node_name", ""))
            node_scores[node] = max(node_scores.get(node, 0.0), float(row.get("deviation_score", 0.0)))
        nodes = [node for node, _ in sorted(node_scores.items(), key=lambda item: item[1], reverse=True)[:20]]
        node_index = {node: idx for idx, node in enumerate(nodes)}
        steps = sorted({int(row.get("step_index", 0)) for row in heat_rows})
        if not nodes or not steps:
            self._draw_empty_canvas(canvas, "No heatmap bins available.")
            return
        step_index = {step: idx for idx, step in enumerate(steps)}
        matrix = [[0.0 for _ in steps] for _ in nodes]
        for row in heat_rows:
            node = str(row.get("node_name", ""))
            if node not in node_index:
                continue
            step = int(row.get("step_index", 0))
            if step not in step_index:
                continue
            matrix[node_index[node]][step_index[step]] = float(row.get("deviation_score", 0.0))

        left = 128
        right = width - 24
        top = 34
        bottom = height - 34
        plot_w = max(1, right - left)
        plot_h = max(1, bottom - top)
        cell_w = max(1.0, plot_w / max(1, len(steps)))
        cell_h = max(1.0, plot_h / max(1, len(nodes)))
        max_value = max(1e-6, max(max(row) for row in matrix))
        canvas.create_text(12, 12, anchor="nw", text="Heatmap Perbedaan Pola Aktivitas", fill=COLORS["text_soft"], font=FONTS["section"])
        for node, row_idx in node_index.items():
            y0 = top + row_idx * cell_h
            canvas.create_text(12, y0 + cell_h * 0.5, anchor="w", text=node[:20], fill=COLORS["axis_text"], font=FONTS["small"])
            for step, col_idx in step_index.items():
                value = matrix[row_idx][col_idx]
                ratio = min(1.0, max(0.0, value / max_value))
                color = self._heat_color(ratio)
                x0 = left + col_idx * cell_w
                canvas.create_rectangle(x0, y0, x0 + cell_w + 1, y0 + cell_h + 1, fill=color, outline="")
        for tick, label in ((0.0, "0%"), (0.5, "50%"), (1.0, "100%")):
            x = left + tick * plot_w
            canvas.create_line(x, bottom + 3, x, bottom + 8, fill=COLORS["threshold"])
            canvas.create_text(x, bottom + 18, text=label, fill=COLORS["axis_text"], font=FONTS["small"])

    def _heat_color(self, ratio: float) -> str:
        ratio = min(1.0, max(0.0, float(ratio)))
        low = (21, 27, 35)
        mid = (217, 154, 50)
        high = (201, 60, 67)
        if ratio < 0.5:
            t = ratio / 0.5
            rgb = tuple(int(low[i] * (1.0 - t) + mid[i] * t) for i in range(3))
        else:
            t = (ratio - 0.5) / 0.5
            rgb = tuple(int(mid[i] * (1.0 - t) + high[i] * t) for i in range(3))
        return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
