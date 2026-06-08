# Proposed Multiclass Label Rule Revision

Do not continue direct single-label multiclass training with the current priority rule.

## Recommended Revision
1. Stage 1: binary `within_reference` vs `deviating` using aggregate deviation score.
2. Stage 2: subtype classification only for nodes already marked `deviating`.
3. Compute subtype flags independently: `onset_shift`, `too_weak`, `too_high`, `not_smooth`.
4. Allow multi-label subtype audit before forcing one primary class.
5. If a single class is required, assign by strongest normalized evidence, not fixed onset-first priority.

## Concrete Rule Changes To Test Next
- Require onset delay to exceed at least two progress steps or a reference-onset uncertainty band before `onset_shift` fires.
- Split onset into `early_onset`, `late_onset`, and `missing_onset` during audit, then merge only if distributions are healthy.
- Define `too_weak` and `too_high` using reference peak standard deviation when available, not only 10% peak tolerance.
- Define `not_smooth` using relative smoothness z-score or train-reference percentile, not only 15% smoothness tolerance.
- Report subtype flags separately from primary class to avoid hiding magnitude/smoothness issues behind onset.