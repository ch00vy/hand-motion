# Main Script Entry Points

This repository now separates the final architecture into two visible script
families:

- `scripts/realtime/`: live pragmatic control.
- `scripts/offline_biomech/`: offline biomechanical reference, graph sequence,
  deviation analysis, and GCN node classification.

Do not use root-level `build_teacher_*` scripts as user-facing workflows. They
remain compatibility backends for the offline biomech path until the large
legacy implementation is fully renamed.

Archived experiment scripts live in `scripts/archive_experiments/`.
