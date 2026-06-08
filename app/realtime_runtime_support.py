from __future__ import annotations

from typing import Any, Callable

import numpy as np

MYOSUITE_RENDER_FPS = 60
ENABLE_MYOSUITE_RENDER = True
CTRL_ENV_EPS = 0.015
CTRL_ENV_CTRL_NORM_EPS = 0.08
CTRL_ENV_SPEEDUP = 3


def unwrap_env(env: Any) -> Any:
    try:
        return env.unwrapped
    except Exception:
        return env


def create_myosuite_env(gym_module: Any, myosuite_available: bool, env_id: str, info_label: str, warn_label: str) -> Any:
    if not myosuite_available:
        return None
    try:
        env = gym_module.make(env_id)
        env.reset()
        env = unwrap_env(env)
        print(f"[INFO] {info_label}")
        return env
    except Exception as exc:
        print(f"[WARN] Failed to create {warn_label}: {exc}")
        return None


def create_ctrl_env_context(env: Any) -> dict[str, Any]:
    context = {
        "env": env,
        "data_source": None,
        "steps_per_render": 1,
        "speedup": int(CTRL_ENV_SPEEDUP),
        "needs_reset": True,
    }
    if env is None:
        return context
    context["data_source"] = env.sim.data
    try:
        ctrl_dt = float(env.sim.model.opt.timestep)
        context["steps_per_render"] = max(1, int(round((1.0 / float(MYOSUITE_RENDER_FPS)) / ctrl_dt)))
    except Exception:
        context["steps_per_render"] = 5
    return context


def apply_qpos_to_view_env(env: Any, q_target: np.ndarray) -> None:
    qpos_view = env.sim.data.qpos
    n = min(len(qpos_view), len(q_target))
    qpos_view[:n] = q_target[:n]
    if n < len(qpos_view):
        qpos_view[n:] = 0.0


def update_main_env_ctrl_view(env: Any, ctrl_vec: np.ndarray | None) -> None:
    if ctrl_vec is None:
        return
    ctrl_arr = np.asarray(ctrl_vec, dtype=float).ravel()
    env.sim.data.ctrl[: min(len(ctrl_arr), int(env.sim.data.ctrl.shape[0]))] = ctrl_arr[: min(len(ctrl_arr), int(env.sim.data.ctrl.shape[0]))]


def _build_neutral_qpos(model: Any) -> np.ndarray:
    qpos = np.zeros(int(model.nq), dtype=float)
    for j in range(int(model.njnt)):
        jt = int(model.jnt_type[j])
        if jt not in (2, 3):
            continue
        adr = int(model.jnt_qposadr[j])
        lo = float(model.jnt_range[j, 0])
        hi = float(model.jnt_range[j, 1])
        if np.isfinite(lo) and np.isfinite(hi):
            qpos[adr] = 0.5 * (lo + hi)
    return qpos


def reset_ctrl_env_state(env_ctrl: Any) -> None:
    if env_ctrl is None:
        return
    try:
        env_ctrl.reset()
        env_ctrl = unwrap_env(env_ctrl)
    except Exception:
        pass
    try:
        env_ctrl.sim.data.qpos[:] = _build_neutral_qpos(env_ctrl.sim.model)
        env_ctrl.sim.data.qvel[:] = 0.0
        env_ctrl.sim.data.qacc[:] = 0.0
        env_ctrl.sim.data.ctrl[:] = 0.0
        if hasattr(env_ctrl.sim.data, "act") and len(env_ctrl.sim.data.act) > 0:
            env_ctrl.sim.data.act[:] = 0.0
        env_ctrl.sim.forward()
    except Exception:
        pass


def apply_ctrl_to_env(env_ctrl: Any, ctrl_vec: np.ndarray | None) -> np.ndarray:
    if env_ctrl is None:
        return np.zeros(0, dtype=float)
    if ctrl_vec is None:
        env_ctrl.sim.data.ctrl[:] = 0.0
        return np.zeros(int(env_ctrl.sim.data.ctrl.shape[0]), dtype=float)
    c_arr = np.asarray(ctrl_vec, dtype=float).ravel().copy()
    c_arr[np.abs(c_arr) < float(CTRL_ENV_EPS)] = 0.0
    env_ctrl.sim.data.ctrl[: min(len(c_arr), int(env_ctrl.sim.data.ctrl.shape[0]))] = c_arr[: min(len(c_arr), int(env_ctrl.sim.data.ctrl.shape[0]))]
    return c_arr


def print_ctrl_summary(ctrl_state: dict[str, Any] | np.ndarray) -> None:
    if isinstance(ctrl_state, dict):
        ctrl_raw_value = ctrl_state.get("ctrl_raw")
        ctrl_smoothed_value = ctrl_state.get("ctrl_smoothed")
        ctrl_analysis_value = ctrl_state.get("ctrl_analysis")
        ctrl_raw = np.asarray(ctrl_raw_value if ctrl_raw_value is not None else [], dtype=float)
        ctrl_smoothed = np.asarray(ctrl_smoothed_value if ctrl_smoothed_value is not None else ctrl_raw, dtype=float)
        ctrl_analysis = np.asarray(ctrl_analysis_value if ctrl_analysis_value is not None else ctrl_smoothed, dtype=float)
        ref = ctrl_analysis if ctrl_analysis.size else (ctrl_smoothed if ctrl_smoothed.size else ctrl_raw)
        if ref.size == 0:
            print("[CTRL] empty ctrl state")
            return
        active = int(np.sum(ref > 0.01))
        top5_idx = np.argsort(ref)[-5:][::-1]
        top5_str = " ".join([f"m{i}={ref[i]:.2f}" for i in top5_idx])
        flags = []
        if ctrl_state.get("deadband_applied"):
            flags.append("deadband")
        if ctrl_state.get("hold_applied"):
            flags.append("hold")
        if ctrl_state.get("freeze_applied"):
            flags.append("freeze")
        if ctrl_state.get("recomputed"):
            flags.append("recompute")
        flag_str = ",".join(flags) if flags else "live"
        print(
            "[CTRL] "
            f"mode={ctrl_state.get('mode', 'control')} flags={flag_str} "
            f"qdn={float(ctrl_state.get('qpos_delta_norm', 0.0)):.4f} "
            f"raw_dn={float(ctrl_state.get('ctrl_raw_delta_norm', 0.0)):.4f} "
            f"smooth_dn={float(ctrl_state.get('ctrl_smoothed_delta_norm', 0.0)):.4f} "
            f"analysis_dn={float(ctrl_state.get('ctrl_analysis_delta_norm', 0.0)):.4f} "
            f"act={active}/{len(ref)} mean={ref.mean():.3f} max={ref.max():.3f} | top5: {top5_str}"
        )
        if bool(ctrl_state.get("is_hold_mode")):
            hold_top = list(ctrl_state.get("hold_top_flexors") or [])
            hold_top_str = " ".join([f"{str(name)}={float(value):.2f}" for name, value in hold_top])
            hold_values = dict(ctrl_state.get("hold_flexor_values") or {})
            ordered_names = ("FDS2", "FDS3", "FDS4", "FDS5", "FDP2", "FDP3", "FDP4", "FDP5")
            hold_values_str = " ".join([f"{name}={float(hold_values.get(name, 0.0)):.2f}" for name in ordered_names])
            print(
                "[CTRL][HOLD] "
                f"prior={str(ctrl_state.get('hold_prior_label', 'none'))} "
                f"fist_hold_prior={int(bool(ctrl_state.get('fist_hold_prior_active')))} "
                f"top_flexors={hold_top_str or '-'} | {hold_values_str}"
            )
            index_values = dict(ctrl_state.get("index_hold_values") or {})
            print(
                "[CTRL][INDEX_HOLD] "
                f"active={int(bool(ctrl_state.get('index_hold_prior_active')))} "
                f"FDS2={float(index_values.get('FDS2', 0.0)):.2f} "
                f"FDP2={float(index_values.get('FDP2', 0.0)):.2f} "
                f"EIP={float(index_values.get('EIP', 0.0)):.2f} "
                f"EDC2={float(index_values.get('EDC2', 0.0)):.2f}"
            )
        return

    c = np.asarray(ctrl_state, dtype=float)
    active = int(np.sum(c > 0.01))
    top5_idx = np.argsort(c)[-5:][::-1]
    top5_str = " ".join([f"m{i}={c[i]:.2f}" for i in top5_idx])
    print(f"[CTRL] act={active}/{len(c)} mean={c.mean():.3f} max={c.max():.3f} | top5: {top5_str}")


def step_ctrl_env(env_ctrl: Any, ctrl_steps_per_render: int, ctrl_vec: np.ndarray | None, render: bool = True) -> None:
    if env_ctrl is None:
        return
    c_arr = apply_ctrl_to_env(env_ctrl, ctrl_vec)
    ctrl_norm = float(np.linalg.norm(c_arr)) if c_arr.size else 0.0
    if ctrl_norm <= float(CTRL_ENV_CTRL_NORM_EPS):
        env_ctrl.sim.data.ctrl[:] = 0.0
    for _ in range(int(ctrl_steps_per_render)):
        env_ctrl.sim.advance()
    env_ctrl.sim.forward()
    if render:
        try:
            env_ctrl.mj_render()
        except Exception:
            pass


def handle_runtime_key(
    key: int,
    recorder: Any,
    listener: Any,
    *,
    reset_callback: Callable[[Any, str], None],
) -> bool:
    if key == ord("x"):
        print("[INFO] Exit requested via key 'x'.")
        return False
    if key == ord("0"):
        reset_callback(listener, "manual reset", manual=True)
        return True
    if key == ord("r"):
        recorder.toggle()
        return True
    return True
