from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk


COLORS = {
    "bg": "#11161D",
    "bg_alt": "#151B23",
    "panel": "#1A2029",
    "panel_alt": "#202833",
    "viewer": "#151B23",
    "border": "#313B48",
    "border_light": "#28323E",
    "divider": "#313B48",
    "text": "#E6EAF0",
    "text_soft": "#D7DEE8",
    "muted": "#9CA6B5",
    "muted_dark": "#626C79",
    "disabled": "#626C79",
    "accent": "#2F6FEB",
    "accent_hover": "#3B7AF0",
    "accent_active": "#245CC5",
    "accent_soft": "#253957",
    "cyan": "#A8B1BD",
    "success": "#3FA66B",
    "success_soft": "#1D2A24",
    "success_tint": "#1D2A24",
    "danger": "#C93C43",
    "danger_hover": "#DD4B52",
    "danger_active": "#A63238",
    "danger_soft": "#2B2024",
    "danger_tint": "#2B2024",
    "warning": "#D99A32",
    "warning_hover": "#E2A84A",
    "warning_active": "#B87F24",
    "warning_soft": "#2B261D",
    "warning_tint": "#2B261D",
    "input": "#151B23",
    "table": "#1A2029",
    "table_alt": "#1D242E",
    "table_header": "#252E3A",
    "table_hover": "#283341",
    "table_selected": "#253957",
    "plot_grid": "#2B3541",
    "axis_text": "#A8B1BD",
    "threshold": "#8D97A5",
    "button_secondary": "#29323D",
    "button_secondary_hover": "#343F4D",
    "button_secondary_active": "#222A34",
    "button_border": "#3A4552",
    "button_disabled": "#20262E",
    "button_disabled_border": "#28313B",
}

FONTS = {
    "title": ("Segoe UI", 18, "bold"),
    "section": ("Segoe UI", 11, "bold"),
    "body": ("Segoe UI", 10),
    "small": ("Segoe UI", 9),
    "metric": ("Segoe UI", 17, "bold"),
    "mono": ("Consolas", 9),
}

BUTTON_VARIANTS = {
    "primary": {
        "bg": COLORS["accent"],
        "hover": COLORS["accent_hover"],
        "active": COLORS["accent_active"],
        "fg": COLORS["text"],
        "border": COLORS["accent"],
    },
    "danger": {
        "bg": COLORS["danger"],
        "hover": COLORS["danger_hover"],
        "active": COLORS["danger_active"],
        "fg": COLORS["text"],
        "border": COLORS["danger"],
    },
    "secondary": {
        "bg": COLORS["button_secondary"],
        "hover": COLORS["button_secondary_hover"],
        "active": COLORS["button_secondary_active"],
        "fg": COLORS["text"],
        "border": COLORS["button_border"],
    },
}
BUTTON_VARIANTS["success"] = BUTTON_VARIANTS["secondary"]
BUTTON_VARIANTS["warning"] = BUTTON_VARIANTS["secondary"]


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
    style.map(".", foreground=[("disabled", COLORS["disabled"])])
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("Panel.TFrame", background=COLORS["panel"])
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
    style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text_soft"])
    style.configure("Muted.TLabel", background=COLORS["panel"], foreground=COLORS["muted"])
    style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=FONTS["title"])
    style.configure("Section.TLabel", background=COLORS["panel"], foreground=COLORS["text_soft"], font=FONTS["section"])
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
        padding=(8, 6),
    )
    style.map(
        "TEntry",
        bordercolor=[("focus", COLORS["accent"]), ("active", "#465466"), ("disabled", COLORS["border_light"])],
        fieldbackground=[("disabled", COLORS["input"])],
        foreground=[("disabled", COLORS["disabled"])],
    )
    style.configure(
        "TCombobox",
        fieldbackground=COLORS["input"],
        background=COLORS["input"],
        foreground=COLORS["text"],
        arrowcolor=COLORS["muted"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
        padding=(8, 6),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", COLORS["input"]), ("disabled", COLORS["input"])],
        foreground=[("readonly", COLORS["text"]), ("disabled", COLORS["disabled"])],
        bordercolor=[("focus", COLORS["accent"]), ("active", "#465466"), ("disabled", COLORS["border_light"])],
        arrowcolor=[("disabled", COLORS["disabled"])],
    )
    style.configure(
        "Horizontal.TScale",
        background=COLORS["panel"],
        troughcolor=COLORS["button_secondary"],
        bordercolor=COLORS["border"],
        lightcolor=COLORS["accent"],
        darkcolor=COLORS["accent"],
        sliderlength=18,
    )
    style.configure(
        "Horizontal.TProgressbar",
        troughcolor=COLORS["button_secondary"],
        background=COLORS["accent"],
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
        background=COLORS["table_header"],
        foreground="#DCE2EA",
        relief="flat",
        font=FONTS["section"],
        padding=7,
    )
    style.map("Treeview", background=[("selected", COLORS["table_selected"])], foreground=[("selected", COLORS["text"])])
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


class RoundedButton(tk.Canvas):
    """Canvas-backed button with small radius and Tk Button-like state support."""

    def __init__(self, parent: tk.Widget, text: str, command, *, variant: str = "secondary") -> None:
        self._text = str(text)
        self._command = command
        self._variant = str(variant)
        self._state = tk.NORMAL
        self._mode = "normal"
        self._padx = 12
        self._pady = 6
        self._radius = 6
        self._font = tkfont.Font(font=FONTS["section"])
        self._parent_bg = _widget_bg(parent)
        width, height = self._requested_size()
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=self._parent_bg,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.bind("<Configure>", lambda _event: self._redraw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self._redraw()

    def configure(self, cnf=None, **kwargs):  # type: ignore[override]
        if isinstance(cnf, dict):
            kwargs.update(cnf)
            cnf = None
        if cnf is not None:
            return super().configure(cnf)

        redraw = False
        resize = False
        if "text" in kwargs:
            self._text = str(kwargs.pop("text"))
            redraw = True
            resize = True
        if "command" in kwargs:
            self._command = kwargs.pop("command")
        if "variant" in kwargs:
            self._variant = str(kwargs.pop("variant"))
            redraw = True
        if "state" in kwargs:
            self._state = str(kwargs.pop("state"))
            redraw = True
        if "padx" in kwargs:
            self._padx = int(kwargs.pop("padx"))
            redraw = True
            resize = True
        if "pady" in kwargs:
            self._pady = int(kwargs.pop("pady"))
            redraw = True
            resize = True
        if "radius" in kwargs:
            self._radius = max(0, int(kwargs.pop("radius")))
            redraw = True
        if kwargs:
            super().configure(**kwargs)
        if resize:
            width, height = self._requested_size()
            super().configure(width=width, height=height)
        if redraw:
            self._redraw()
        return None

    config = configure

    def cget(self, key: str):  # type: ignore[override]
        if key == "state":
            return self._state
        if key == "text":
            return self._text
        if key == "variant":
            return self._variant
        return super().cget(key)

    def _requested_size(self) -> tuple[int, int]:
        width = max(88, int(self._font.measure(self._text) + (2 * self._padx) + 10))
        height = max(30, int(self._font.metrics("linespace") + (2 * self._pady) + 6))
        return width, height

    def _palette(self) -> dict[str, str]:
        if self._state == tk.DISABLED:
            return {
                "bg": COLORS["button_disabled"],
                "border": COLORS["button_disabled_border"],
                "fg": COLORS["disabled"],
            }
        palette = BUTTON_VARIANTS.get(self._variant, BUTTON_VARIANTS["secondary"])
        bg = palette.get(self._mode, palette["bg"])
        return {"bg": bg, "border": palette["border"], "fg": palette["fg"]}

    def _redraw(self) -> None:
        self.delete("all")
        width = max(1, int(self.winfo_width() or self.cget("width")))
        height = max(1, int(self.winfo_height() or self.cget("height")))
        colors = self._palette()
        self.create_rectangle(0, 0, width, height, fill=self._parent_bg, outline="")
        self._rounded_rect(1, 1, width - 1, height - 1, self._radius, fill=colors["bg"], outline=colors["border"])
        self.create_text(width // 2, height // 2, text=self._text, fill=colors["fg"], font=FONTS["section"])

    def _rounded_rect(self, x1: int, y1: int, x2: int, y2: int, radius: int, *, fill: str, outline: str) -> None:
        if radius <= 0:
            self.create_rectangle(x1, y1, x2, y2, fill=fill, outline=outline)
            return
        points = [
            x1 + radius,
            y1,
            x2 - radius,
            y1,
            x2,
            y1,
            x2,
            y1 + radius,
            x2,
            y2 - radius,
            x2,
            y2,
            x2 - radius,
            y2,
            x1 + radius,
            y2,
            x1,
            y2,
            x1,
            y2 - radius,
            x1,
            y1 + radius,
            x1,
            y1,
        ]
        self.create_polygon(points, smooth=True, fill=fill, outline=outline)

    def _on_enter(self, _event: tk.Event) -> None:
        if self._state != tk.DISABLED:
            self._mode = "hover"
            self._redraw()

    def _on_leave(self, _event: tk.Event) -> None:
        if self._state != tk.DISABLED:
            self._mode = "normal"
            self._redraw()

    def _on_press(self, _event: tk.Event) -> None:
        if self._state != tk.DISABLED:
            self._mode = "active"
            self._redraw()

    def _on_release(self, event: tk.Event) -> None:
        if self._state == tk.DISABLED:
            return
        inside = 0 <= int(event.x) <= int(self.winfo_width()) and 0 <= int(event.y) <= int(self.winfo_height())
        self._mode = "hover" if inside else "normal"
        self._redraw()
        if inside and self._command is not None:
            self._command()


def button(parent: tk.Widget, text: str, command, *, variant: str = "secondary") -> RoundedButton:
    return RoundedButton(parent, text, command, variant=variant)


def configure_button_variant(widget: tk.Widget, variant: str) -> None:
    widget.configure(variant=str(variant))


def _widget_bg(widget: tk.Widget) -> str:
    try:
        return str(widget.cget("bg"))
    except Exception:
        return COLORS["panel"]
