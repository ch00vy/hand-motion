from __future__ import annotations

import tkinter as tk
from tkinter import ttk


COLORS = {
    "bg": "#1E1E1E",
    "bg_alt": "#20252B",
    "panel": "#252B32",
    "panel_alt": "#2A2F36",
    "viewer": "#111827",
    "border": "#3A414B",
    "text": "#E5E7EB",
    "muted": "#AAB2BF",
    "muted_dark": "#7C8794",
    "accent": "#3B82F6",
    "accent_hover": "#2563EB",
    "cyan": "#22D3EE",
    "success": "#22C55E",
    "success_soft": "#123524",
    "danger": "#EF4444",
    "danger_soft": "#3D1E24",
    "warning": "#F59E0B",
    "warning_soft": "#3B2C12",
    "input": "#151A20",
    "table": "#1B2027",
    "table_alt": "#202731",
}

FONTS = {
    "title": ("Segoe UI", 16, "bold"),
    "section": ("Segoe UI", 10, "bold"),
    "body": ("Segoe UI", 9),
    "small": ("Segoe UI", 8),
    "metric": ("Segoe UI", 18, "bold"),
    "mono": ("Consolas", 9),
}


def configure_dark_theme(root: tk.Tk) -> None:
    root.configure(bg=COLORS["bg"])
    root.option_add("*Font", FONTS["body"])
    root.option_add("*selectBackground", COLORS["accent"])
    root.option_add("*selectForeground", COLORS["text"])

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    style.configure(".", background=COLORS["bg"], foreground=COLORS["text"], fieldbackground=COLORS["input"])
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("Panel.TFrame", background=COLORS["panel"])
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
    style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"])
    style.configure("Muted.TLabel", background=COLORS["panel"], foreground=COLORS["muted"])
    style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=FONTS["title"])
    style.configure("Section.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=FONTS["section"])
    style.configure("Metric.TLabel", background=COLORS["panel_alt"], foreground=COLORS["text"], font=FONTS["metric"])
    style.configure("Good.TLabel", background=COLORS["panel"], foreground=COLORS["success"])
    style.configure("Danger.TLabel", background=COLORS["panel"], foreground=COLORS["danger"])
    style.configure("Warning.TLabel", background=COLORS["panel"], foreground=COLORS["warning"])
    style.configure(
        "TEntry",
        fieldbackground=COLORS["input"],
        background=COLORS["input"],
        foreground=COLORS["text"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
        insertcolor=COLORS["text"],
        padding=6,
    )
    style.configure(
        "TCombobox",
        fieldbackground=COLORS["input"],
        background=COLORS["input"],
        foreground=COLORS["text"],
        arrowcolor=COLORS["muted"],
        bordercolor=COLORS["border"],
        padding=6,
    )
    style.map("TCombobox", fieldbackground=[("readonly", COLORS["input"])], foreground=[("readonly", COLORS["text"])])
    style.configure(
        "Horizontal.TScale",
        background=COLORS["panel"],
        troughcolor=COLORS["input"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["accent"],
        darkcolor=COLORS["accent"],
    )
    style.configure(
        "Treeview",
        background=COLORS["table"],
        fieldbackground=COLORS["table"],
        foreground=COLORS["text"],
        bordercolor=COLORS["border"],
        rowheight=28,
        font=FONTS["body"],
    )
    style.configure(
        "Treeview.Heading",
        background=COLORS["panel_alt"],
        foreground=COLORS["text"],
        relief="flat",
        font=FONTS["section"],
        padding=7,
    )
    style.map("Treeview", background=[("selected", COLORS["accent"])], foreground=[("selected", COLORS["text"])])
    style.configure("Vertical.TScrollbar", background=COLORS["panel_alt"], troughcolor=COLORS["input"], bordercolor=COLORS["border"])


def card(parent: tk.Widget, *, padx: int = 12, pady: int = 12) -> tk.Frame:
    frame = tk.Frame(
        parent,
        bg=COLORS["panel"],
        highlightbackground=COLORS["border"],
        highlightcolor=COLORS["border"],
        highlightthickness=1,
        bd=0,
        padx=padx,
        pady=pady,
    )
    return frame


def button(parent: tk.Widget, text: str, command, *, variant: str = "secondary") -> tk.Button:
    palette = {
        "primary": (COLORS["accent"], COLORS["accent_hover"], COLORS["text"]),
        "danger": (COLORS["danger"], "#DC2626", COLORS["text"]),
        "success": (COLORS["success"], "#16A34A", "#07140B"),
        "warning": (COLORS["warning"], "#D97706", "#111827"),
        "secondary": (COLORS["panel_alt"], COLORS["border"], COLORS["text"]),
    }
    bg, active_bg, fg = palette.get(str(variant), palette["secondary"])
    return tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        activebackground=active_bg,
        fg=fg,
        activeforeground=fg,
        relief=tk.FLAT,
        bd=0,
        padx=12,
        pady=8,
        cursor="hand2",
        font=FONTS["section"],
    )

