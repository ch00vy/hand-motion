## Zone-Based Muscle Graph

This visualization stays muscle-centric:

- node = one MyoSuite actuator / muscle
- edge = a prior inter-muscle relation (`synergist`, `antagonist`, `functional_group`)

The graph uses a fixed analytical layout instead of realistic hand line art.
Muscles are arranged into stable anatomical zones:

- `thumb`
- `index`
- `middle`
- `ring`
- `little`
- `central_shared`
- `wrist_forearm`

This is only a visual layout. The semantic graph is still muscle-to-muscle, not joint-to-joint and not a skeletal graph.

## Dynamic Edge Visibility

The prior graph remains static in memory, but the renderer does not draw every prior edge at once.

An edge becomes visible only when:

- the edge exists in the prior biological graph
- both endpoint muscles exceed `node_activation_threshold`
- the dynamic score exceeds `edge_score_threshold`

Supported scoring modes:

- `coactivation`: `activation_i * activation_j * prior_weight`
- `rolling_correlation`: positive rolling correlation gates coactivation within a short recent window

## Visual Goal

The design goal is readability, stability, and scientific clarity:

- no outer hand silhouette
- no thumb/palm contour guides
- fixed zone columns for immediate interpretation
- activity encoded by node colour/size and edge strength

That keeps the figure suitable for analysis, proposals, and paper-style screenshots while preserving the original muscle interaction graph semantics.
