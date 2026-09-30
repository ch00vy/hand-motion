from __future__ import annotations

"""Estimation layer for the final demo pipeline.

Pipeline contract:
- qpos drives visual model replay.
- PD inverse produces realtime ctrl indicators for GUI sliders.
- inverse dynamics produces effort indicators for analysis features.
- GCN/node classification produces weakness indication.
"""

from app.estimation.ctrl_indicator import RealtimePDDynamicCtrlConfig, RealtimePDDynamicCtrlEstimator, map_qspace_signal_to_actuator_indicators
from app.estimation.feature_fusion import WEAKNESS_FEATURE_NAMES, WeaknessFeatureBundle, fuse_weakness_node_features, qpos_motion_scores
from app.estimation.inverse_dynamics import InverseDynamicsConfig, InverseDynamicsResult, compute_inverse_dynamics_effort
from app.estimation.pd_inverse import PDInverseConfig, PDInverseEstimator, PDInverseResult

__all__ = [
    "InverseDynamicsConfig",
    "InverseDynamicsResult",
    "PDInverseConfig",
    "PDInverseEstimator",
    "PDInverseResult",
    "RealtimePDDynamicCtrlConfig",
    "RealtimePDDynamicCtrlEstimator",
    "WEAKNESS_FEATURE_NAMES",
    "WeaknessFeatureBundle",
    "compute_inverse_dynamics_effort",
    "fuse_weakness_node_features",
    "map_qspace_signal_to_actuator_indicators",
    "qpos_motion_scores",
]
