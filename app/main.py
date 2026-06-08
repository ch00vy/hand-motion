from __future__ import annotations

"""Compatibility entrypoint for the realtime runtime.

The large legacy implementation now lives in
`app.realtime.monolithic_runtime_backend` while the public `app.main` module
remains as a thin shim for existing imports and commands.
"""

from app.realtime import monolithic_runtime_backend as _backend

globals().update(
    {
        name: getattr(_backend, name)
        for name in dir(_backend)
        if not (name.startswith("__") and name.endswith("__"))
    }
)


if __name__ == "__main__":
    raise SystemExit(_backend._cli_main())
