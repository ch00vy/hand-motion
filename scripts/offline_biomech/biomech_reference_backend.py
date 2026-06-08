from __future__ import annotations

"""Clean-name facade over the legacy trajectory inverse-dynamics backend."""

from scripts.build_teacher_from_trajectory_inverse import (  # noqa: F401
    main,
    run_trajectory_teacher_trial as run_trajectory_biomech_reference_trial,
    solve_teacher_short_horizon as solve_biomech_reference_short_horizon,
)

