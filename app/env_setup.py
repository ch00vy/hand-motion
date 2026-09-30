from __future__ import annotations

import os
import shutil
from pathlib import Path


def configure_git_executable() -> str:
    """Set GitPython's git executable path when Git is installed normally.

    MyoSuite can import GitPython indirectly. On fresh Windows devices,
    GitPython may fail with "Bad git executable" if `git.exe` is installed but
    not available on PATH. This keeps the app usable with the standard Git for
    Windows installation path.
    """

    existing = str(os.environ.get("GIT_PYTHON_GIT_EXECUTABLE") or "").strip()
    if existing and Path(existing).exists():
        return existing

    candidates = [
        shutil.which("git"),
        r"C:\Program Files\Git\cmd\git.exe",
        r"C:\Program Files\Git\bin\git.exe",
        r"C:\Program Files (x86)\Git\cmd\git.exe",
        r"C:\Program Files (x86)\Git\bin\git.exe",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = str(path)
            return str(path)
    return ""
