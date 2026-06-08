from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Optional

import numpy as np

from app.gui.realtime_pd_ctrl import RealtimePDDynamicCtrlEstimator
from app.gui.replay_mode import ReplayMode
from app.gui.trial_loader import TrialData


DEFAULT_MYOSUITE_ENV_ID = "myoHandPoseRandom-v0"
CTRL_VISUAL_SLIDER_DEADBAND = 1e-5


@dataclass
class ReplayStatus:
    loaded: bool
    playing: bool
    frame_index: int
    num_frames: int
    message: str
    viewer_open: bool = False
    ctrl_active: int = 0
    ctrl_max: float = 0.0
    ctrl_mean: float = 0.0
    internal_response_enabled: bool = False
    replay_mode: str = ReplayMode.NONE.value
    control_source_path: str = ""


class MujocoReplayController:
    """External MuJoCo/MyoSuite replay controller.

    The viewer is not embedded in the GUI; `mj_render()` opens/updates the
    normal MyoSuite render window while the GUI controls frame stepping.
    """

    def __init__(
        self,
        *,
        env_id: str = DEFAULT_MYOSUITE_ENV_ID,
        hide_visual_markers: bool = True,
        use_ctrl_estimate_preview: bool = True,
        ctrl_preview_mode: str = "legacy_solver",
        ctrl_visual_deadband: float = CTRL_VISUAL_SLIDER_DEADBAND,
    ):
        self.env_id = str(env_id)
        self.hide_visual_markers = bool(hide_visual_markers)
        self.use_ctrl_estimate_preview = bool(use_ctrl_estimate_preview)
        self.ctrl_preview_mode = str(ctrl_preview_mode or "legacy_solver").strip().lower()
        self.ctrl_visual_deadband = max(0.0, float(ctrl_visual_deadband))
        self.env: Optional[Any] = None
        self.controller_context: Optional[dict[str, Any]] = None
        self.pd_ctrl_estimator: Optional[RealtimePDDynamicCtrlEstimator] = None
        self.ctrl_log_state = {"controller_logged": False, "fallback_logged": False}
        self.last_ctrl_summary: Optional[dict[str, Any]] = None
        self.last_ctrl_vector: Optional[np.ndarray] = None
        self.trial: Optional[TrialData] = None
        self.control_sequence: Optional[np.ndarray] = None
        self.control_dt = 1.0 / 60.0
        self.control_source_path: Optional[str] = None
        self._control_sim_frame_index: Optional[int] = None
        self.replay_mode = ReplayMode.NONE
        self.frame_index = 0
        self.playing = False
        self.message = "No trial loaded."
        self.min_render_interval_seconds = 1.0 / 30.0
        self.render_error_cooldown_seconds = 1.0
        self._last_render_time = 0.0
        self._last_render_error_time = 0.0

    def _clean_ctrl_for_viewer(self, ctrl_vec: np.ndarray) -> np.ndarray:
        """Remove numerically tiny ctrl values so MuJoCo sliders do not show e- notation."""

        ctrl = np.asarray(ctrl_vec, dtype=float).reshape(-1).copy()
        ctrl = np.where(np.isfinite(ctrl), ctrl, 0.0)
        deadband = max(0.0, float(self.ctrl_visual_deadband))
        if deadband > 0.0:
            ctrl[np.abs(ctrl) < deadband] = 0.0
        return ctrl

    def set_trial(self, trial: TrialData) -> None:
        self.trial = trial
        self.control_sequence = None
        self.control_source_path = None
        self._control_sim_frame_index = None
        self.replay_mode = ReplayMode.NONE
        self.frame_index = 0
        self.playing = False
        self._reset_ctrl_estimator_state()
        self.message = f"Current trial ready: {trial.trial_id} ({trial.num_frames} frames). Trial replay not loaded yet."

    def load_trial(self, trial: TrialData) -> None:
        self.replay_kinematics(trial)

    def open_or_focus_viewer(self) -> bool:
        already_open = self.env is not None
        if not self.ensure_env():
            return False
        if already_open:
            if self.replay_mode == ReplayMode.NONE:
                self.show_neutral_preview()
            else:
                self.render_current()
            self.message = "MuJoCo viewer already open; render refreshed."
            return True
        self.show_neutral_preview()
        return True

    def replay_current_trial(self, trial: Optional[TrialData] = None) -> bool:
        return self.replay_kinematics(trial)

    def replay_kinematics(self, trial: Optional[TrialData] = None) -> bool:
        if trial is not None:
            self.set_trial(trial)
        if self.trial is None:
            self.message = "No current trial for kinematic replay."
            return False
        self.replay_mode = ReplayMode.KINEMATIC
        self.control_sequence = None
        self.control_source_path = None
        if not self.ensure_env():
            return False
        self.render_current()
        return True

    def replay_control(
        self,
        *,
        trial: TrialData,
        ctrl_sequence: np.ndarray,
        dt: float,
        source_path: str,
    ) -> bool:
        self.set_trial(trial)
        ctrl = np.asarray(ctrl_sequence, dtype=float)
        if ctrl.ndim != 2 or ctrl.shape[0] <= 0:
            self.message = "Control replay requires a non-empty ctrl_ref sequence."
            return False
        self.control_sequence = np.where(np.isfinite(ctrl), ctrl, 0.0)
        self.control_dt = max(1e-4, float(dt))
        self.control_source_path = str(source_path)
        self._control_sim_frame_index = None
        self.replay_mode = ReplayMode.CONTROL
        if not self.ensure_env():
            return False
        self.render_current()
        return True

    def show_neutral_preview(self) -> None:
        if not self.ensure_env():
            return
        try:
            qpos = np.zeros((int(self.env.sim.model.nq),), dtype=float)
            self._apply_qpos_and_render(qpos, force=True)
            self.message = "MuJoCo live preview opened. No trial replay loaded."
        except Exception as exc:
            self.message = f"Neutral preview failed: {type(exc).__name__}: {exc}"

    def ensure_env(self) -> bool:
        if self.env is not None:
            return True
        try:
            from myosuite.utils import gym
        except Exception as exc:
            self.message = f"MyoSuite unavailable: {type(exc).__name__}: {exc}"
            return False
        try:
            self.env = gym.make(self.env_id)
            if hasattr(self.env, "reset"):
                self.env.reset()
            try:
                self.env = self.env.unwrapped
            except Exception:
                pass
            self._configure_clean_viewer()
            self.message = f"MyoSuite env ready: {self.env_id}"
            return True
        except Exception as exc:
            self.message = f"Failed to create MyoSuite env: {type(exc).__name__}: {exc}"
            self.env = None
            return False

    def render_current(self) -> None:
        if self.replay_mode == ReplayMode.CONTROL:
            self._render_control_current()
            return
        self._render_kinematic_current()

    def _render_kinematic_current(self) -> None:
        if self.trial is None:
            return
        if not self.ensure_env():
            return
        try:
            qpos = np.asarray(self.trial.qpos[self.frame_index], dtype=float).reshape(-1)
            if self.use_ctrl_estimate_preview:
                qvel = self._trial_qvel_for_frame(self.frame_index)
                dt = self._trial_dt()
                rendered = self._apply_qpos_with_ctrl_display_and_render(
                    qpos,
                    force=True,
                    qvel_override=qvel,
                    dt_override=dt,
                )
                if rendered:
                    self.message = self._format_internal_response_message(
                        f"KINEMATIC_REPLAY qpos frame {self.frame_index + 1}/{self.trial.num_frames}"
                    )
                    return
            self._apply_qpos_and_render(qpos, force=True)
            self.message = f"KINEMATIC_REPLAY qpos frame {self.frame_index + 1}/{self.trial.num_frames} | ctrl estimate unavailable."
        except Exception as exc:
            self.message = f"Kinematic replay render failed: {type(exc).__name__}: {exc}"

    def render_qpos(self, qpos: np.ndarray, *, message: str = "Live recording preview") -> None:
        if not self.ensure_env():
            return
        now = time.perf_counter()
        if (now - self._last_render_error_time) < self.render_error_cooldown_seconds:
            return
        try:
            qpos_arr = np.asarray(qpos, dtype=float).reshape(-1)
            if self.use_ctrl_estimate_preview:
                rendered = self._apply_qpos_with_ctrl_display_and_render(qpos_arr, force=False)
                if rendered:
                    self.message = self._format_internal_response_message(message)
                    return
            rendered = self._apply_qpos_and_render(qpos_arr, force=False)
            if not rendered:
                return
            self.message = f"{message} | qpos-direct fallback"
        except Exception as exc:
            self._last_render_error_time = time.perf_counter()
            self.message = f"Live preview render failed: {type(exc).__name__}: {exc}"

    def _trial_dt(self) -> float:
        if self.trial is None:
            return 1.0 / 60.0
        try:
            dt = float(self.trial.meta.get("dt") or 0.0)
        except Exception:
            dt = 0.0
        return dt if dt > 0.0 else 1.0 / 60.0

    def _trial_qvel_for_frame(self, frame_index: int) -> np.ndarray:
        if self.trial is None or self.trial.num_frames <= 1:
            return np.zeros((0,), dtype=float)
        qpos = np.asarray(self.trial.qpos, dtype=float)
        idx = int(np.clip(int(frame_index), 0, max(0, self.trial.num_frames - 1)))
        dt = max(1e-4, self._trial_dt())
        if idx <= 0:
            qvel = (qpos[1] - qpos[0]) / dt
        else:
            qvel = (qpos[idx] - qpos[idx - 1]) / dt
        return np.where(np.isfinite(qvel), qvel, 0.0).reshape(-1)

    def _num_replay_frames(self) -> int:
        if self.replay_mode == ReplayMode.CONTROL and self.control_sequence is not None:
            return int(self.control_sequence.shape[0])
        return int(self.trial.num_frames if self.trial is not None else 0)

    def _control_substeps_per_frame(self) -> int:
        if self.env is None:
            return 1
        try:
            timestep = float(self.env.sim.model.opt.timestep)
        except Exception:
            timestep = 0.0
        if timestep <= 0.0:
            return 1
        return max(1, int(round(float(self.control_dt) / timestep)))

    def _reset_to_trial_initial_qpos(self) -> None:
        if self.trial is None or self.env is None:
            return
        data = self.env.sim.data
        model = self.env.sim.model
        full_qpos = np.zeros((int(model.nq),), dtype=float)
        qpos0 = np.asarray(self.trial.qpos[0], dtype=float).reshape(-1)
        full_qpos[: min(len(full_qpos), len(qpos0))] = qpos0[: min(len(full_qpos), len(qpos0))]
        data.qpos[:] = full_qpos
        data.qvel[:] = 0.0
        if hasattr(data, "qacc"):
            data.qacc[:] = 0.0
        data.ctrl[:] = 0.0
        if hasattr(data, "act") and len(data.act) > 0:
            data.act[:] = 0.0
        self.env.sim.forward()

    def _advance_with_ctrl(self, ctrl_vec: np.ndarray) -> None:
        if self.env is None:
            return
        ctrl_arr = self._clean_ctrl_for_viewer(ctrl_vec)
        data = self.env.sim.data
        data.ctrl[:] = 0.0
        data.ctrl[: min(len(ctrl_arr), int(data.ctrl.shape[0]))] = ctrl_arr[: min(len(ctrl_arr), int(data.ctrl.shape[0]))]
        self.last_ctrl_vector = data.ctrl.copy()
        for _ in range(self._control_substeps_per_frame()):
            try:
                self.env.sim.advance()
            except Exception:
                try:
                    import mujoco

                    raw_model = getattr(self.env.sim.model, "ptr", self.env.sim.model)
                    raw_data = getattr(self.env.sim.data, "ptr", self.env.sim.data)
                    mujoco.mj_step(raw_model, raw_data)
                except Exception:
                    self.env.sim.forward()
                    break

    def _render_control_current(self) -> None:
        if self.trial is None:
            return
        if self.control_sequence is None or self.control_sequence.size == 0:
            self.message = "CONTROL_REPLAY requires ctrl_ref. Run analysis first."
            return
        if not self.ensure_env():
            return
        try:
            ctrl_seq = np.asarray(self.control_sequence, dtype=float)
            frame_count = int(ctrl_seq.shape[0])
            target_index = int(np.clip(self.frame_index, 0, max(0, frame_count - 1)))
            current_sim_index = self._control_sim_frame_index
            if current_sim_index is None or target_index < current_sim_index:
                self._reset_to_trial_initial_qpos()
                current_sim_index = 0
            else:
                current_sim_index = int(current_sim_index)

            # Alignment convention:
            # qpos[0] is the recorded initial state. ctrl[i] advances the
            # simulation from qpos frame i toward qpos frame i + 1.
            for idx in range(current_sim_index, target_index):
                self._advance_with_ctrl(ctrl_seq[idx])
            self._control_sim_frame_index = target_index
            self.env.sim.forward()
            self.env.mj_render()
            self._configure_clean_viewer()
            self._last_render_time = time.perf_counter()
            err_text = ""
            if self.trial is not None and self.trial.num_frames:
                target_idx = min(target_index, self.trial.num_frames - 1)
                target_qpos = np.asarray(self.trial.qpos[target_idx], dtype=float).reshape(-1)
                sim_qpos = np.asarray(self.env.sim.data.qpos, dtype=float).reshape(-1)
                n = min(len(target_qpos), len(sim_qpos))
                if n:
                    rmse = float(np.sqrt(np.mean((sim_qpos[:n] - target_qpos[:n]) ** 2)))
                    err_text = f" | qpos_rmse_vs_record={rmse:.4f}"
            self.message = self._format_internal_response_message(
                f"CONTROL_REPLAY ctrl_ref frame {target_index + 1}/{frame_count}{err_text}"
            )
        except Exception as exc:
            self.message = f"Control replay failed: {type(exc).__name__}: {exc}"

    def _reset_ctrl_estimator_state(self) -> None:
        self.last_ctrl_summary = None
        self.last_ctrl_vector = None
        if self.pd_ctrl_estimator is not None:
            self.pd_ctrl_estimator.reset()
        self.ctrl_log_state = {"controller_logged": False, "fallback_logged": False}
        if self.controller_context is None:
            return
        try:
            from app.realtime.monolithic_runtime_backend import reset_controller_context_state

            data_source = self.env.sim.data if self.env is not None else None
            reset_controller_context_state(self.controller_context, data_source=data_source)
        except Exception:
            pass

    def _ctrl_stats(self) -> tuple[int, float, float]:
        ctrl = np.asarray(self.last_ctrl_vector if self.last_ctrl_vector is not None else [], dtype=float).reshape(-1)
        if ctrl.size == 0:
            return 0, 0.0, 0.0
        return int(np.sum(ctrl > 0.01)), float(np.max(ctrl)), float(np.mean(ctrl))

    def _format_internal_response_message(self, prefix: str) -> str:
        active, max_ctrl, mean_ctrl = self._ctrl_stats()
        method = ""
        if self.last_ctrl_summary is not None:
            method_name = str(self.last_ctrl_summary.get("method", "")).strip()
            if method_name:
                method = f" method={method_name}"
        return f"{prefix} | muscle ctrl{method} active={active} max={max_ctrl:.2f} mean={mean_ctrl:.3f}"

    def _ensure_ctrl_estimator(self) -> bool:
        if self.ctrl_preview_mode == "pd_dynamic":
            return self._ensure_pd_ctrl_estimator()
        if self.controller_context is not None:
            return True
        try:
            from app.realtime.monolithic_runtime_backend import (
                CtrlEstimationConfig,
                create_controller_context,
            )

            config = CtrlEstimationConfig(
                qpos_deadband_norm=0.01,
                ctrl_smoothing_alpha=0.18,
                hold_if_small_change=True,
                rolling_window=7,
                mode="control",
                freeze_mode=False,
                runtime_mode="legacy_solver",
                enable_hold_stabilizer=False,
                live_qpos_ema_alpha=1.0,
                live_qpos_deadband=0.0,
                live_qpos_max_step=0.0,
            )
            self.controller_context = create_controller_context(
                self.env_id,
                estimation_config=config,
                exercise_name="open_to_fist",
            )
            self.controller_context["input_mode"] = "leap_live"
            return self.controller_context.get("probe_model") is not None
        except Exception as exc:
            self.message = f"Ctrl estimator unavailable: {type(exc).__name__}: {exc}"
            self.controller_context = None
            return False

    def _ensure_pd_ctrl_estimator(self) -> bool:
        if self.pd_ctrl_estimator is not None:
            return True
        if self.env is None:
            return False
        try:
            names = self._actuator_names()
            self.pd_ctrl_estimator = RealtimePDDynamicCtrlEstimator(actuator_names=names)
            return True
        except Exception as exc:
            self.message = f"PD dynamic ctrl estimator unavailable: {type(exc).__name__}: {exc}"
            self.pd_ctrl_estimator = None
            return False

    def _actuator_names(self) -> list[str]:
        if self.env is None:
            return []
        model = self.env.sim.model
        raw_model = getattr(model, "ptr", model)
        nu = int(getattr(raw_model, "nu", getattr(model, "nu", 0)))
        names: list[str] = []
        try:
            import mujoco
        except Exception:
            mujoco = None
        for idx in range(max(0, nu)):
            name = ""
            if mujoco is not None:
                try:
                    name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(idx)) or ""
                except Exception:
                    name = ""
            names.append(name or f"actuator_{idx}")
        return names

    def _apply_qpos_with_ctrl_display_and_render(
        self,
        qpos: np.ndarray,
        *,
        force: bool = False,
        qvel_override: Optional[np.ndarray] = None,
        dt_override: Optional[float] = None,
    ) -> bool:
        now = time.perf_counter()
        if not force and (now - self._last_render_time) < self.min_render_interval_seconds:
            return False
        if not self._ensure_ctrl_estimator():
            return self._apply_qpos_and_render(qpos, force=force)
        try:
            if self.ctrl_preview_mode == "pd_dynamic":
                ctrl_vec, info = self._estimate_pd_dynamic_ctrl(
                    qpos,
                    qvel_override=qvel_override,
                    dt_override=dt_override,
                )
                self.last_ctrl_summary = dict(info)
                self.last_ctrl_vector = ctrl_vec.copy()
                return self._apply_qpos_and_render(qpos, force=force, ctrl_display=ctrl_vec)

            from app.realtime.monolithic_runtime_backend import estimate_ctrl_signals, prepare_target_qpos

            if dt_override is not None:
                self.controller_context["runtime_dt_override"] = float(dt_override)
            if qvel_override is not None:
                self.controller_context["runtime_qvel_override"] = np.asarray(qvel_override, dtype=float).reshape(-1)
            q_target = prepare_target_qpos(np.asarray(qpos, dtype=float), self.controller_context)
            ctrl_state = estimate_ctrl_signals(
                q_target=q_target,
                controller_context=self.controller_context,
                ctrl_data_source=self.env.sim.data,
                ctrl_log_state=self.ctrl_log_state,
            )
            if ctrl_state is None:
                return False
            self.last_ctrl_summary = dict(ctrl_state)
            ctrl_vec = self._clean_ctrl_for_viewer(np.asarray(ctrl_state.get("ctrl_raw", []), dtype=float))
            self.last_ctrl_vector = ctrl_vec.copy()
            return self._apply_qpos_and_render(qpos, force=force, ctrl_display=ctrl_vec)
        except Exception as exc:
            self.message = f"Ctrl display estimate failed: {type(exc).__name__}: {exc}"
            return self._apply_qpos_and_render(qpos, force=force)
        finally:
            if self.controller_context is not None:
                self.controller_context["runtime_dt_override"] = None
                self.controller_context["runtime_qvel_override"] = None

    def _estimate_pd_dynamic_ctrl(
        self,
        qpos: np.ndarray,
        *,
        qvel_override: Optional[np.ndarray] = None,
        dt_override: Optional[float] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.env is None or self.pd_ctrl_estimator is None:
            return np.zeros((0,), dtype=float), {"method": "pd_dynamic", "active": 0}
        data = self.env.sim.data
        model = self.env.sim.model
        target = np.asarray(qpos, dtype=float).reshape(-1)
        current_qpos = np.asarray(data.qpos, dtype=float).reshape(-1)
        current_qvel = np.asarray(data.qvel, dtype=float).reshape(-1)
        full_target = np.zeros((int(model.nq),), dtype=float)
        full_target[: min(len(full_target), len(target))] = target[: min(len(full_target), len(target))]
        ctrl, info = self.pd_ctrl_estimator.estimate(
            q_target=full_target,
            qpos_current=current_qpos,
            qvel_current=current_qvel,
            qvel_target=qvel_override,
            dt_override=dt_override,
        )
        return self._clean_ctrl_for_viewer(ctrl), dict(info)

    def _configure_clean_viewer(self) -> None:
        if not self.hide_visual_markers or self.env is None:
            return
        try:
            model = self.env.sim.model
            raw_model = getattr(model, "ptr", model)
            self._hide_target_sites(model=model, raw_model=raw_model)
            self._hide_target_error_tendons(model=model, raw_model=raw_model)
        except Exception:
            pass
        try:
            renderer = getattr(self.env.sim, "renderer", None)
            if renderer is not None and hasattr(renderer, "set_viewer_settings"):
                renderer.set_viewer_settings(render_tendon=True, render_actuator=True)
                self._apply_clean_viewer_options(renderer)
        except Exception:
            pass

    def _apply_clean_viewer_options(self, renderer: Any) -> None:
        try:
            import mujoco
        except Exception:
            mujoco = None

        options = []
        window = getattr(renderer, "_window", None)
        if window is not None and hasattr(window, "opt"):
            options.append(window.opt)
        scene_option = getattr(renderer, "_scene_option", None)
        if scene_option is not None:
            options.append(scene_option)

        for option in options:
            if hasattr(option, "sitegroup"):
                option.sitegroup[:] = 0
            if mujoco is not None and hasattr(option, "flags"):
                option.flags[int(mujoco.mjtVisFlag.mjVIS_TENDON)] = True
                option.flags[int(mujoco.mjtVisFlag.mjVIS_ACTUATOR)] = True
                option.flags[int(mujoco.mjtVisFlag.mjVIS_RANGEFINDER)] = False
                option.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTFORCE)] = False
                option.flags[int(mujoco.mjtVisFlag.mjVIS_CONTACTPOINT)] = False

    def _hide_target_sites(self, *, model: Any, raw_model: Any) -> None:
        """Hide only MyoSuite target markers, not tendon geometry.

        The thick colored guides in the hand viewer are MuJoCo `site` markers
        such as THtip_target/IFtip_target. Tendons are a separate visual layer
        and must remain visible for biomechanical inspection.
        """
        try:
            import mujoco
        except Exception:
            return

        target_indices: list[int] = []
        for idx in range(int(getattr(raw_model, "nsite", 0))):
            try:
                site_name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_SITE, int(idx)) or ""
            except Exception:
                site_name = ""
            if "target" in site_name.lower():
                target_indices.append(int(idx))

        if not target_indices:
            return

        models: list[Any] = [model]
        if raw_model is not model:
            models.append(raw_model)
        for target_model in models:
            for idx in target_indices:
                if idx >= int(getattr(target_model, "nsite", 0)):
                    continue
                if hasattr(target_model, "site_rgba"):
                    target_model.site_rgba[idx, 3] = 0.0
                if hasattr(target_model, "site_size"):
                    target_model.site_size[idx, :] = 0.0

    def _hide_target_error_tendons(self, *, model: Any, raw_model: Any) -> None:
        """Hide task error tendons while preserving anatomical muscle tendons."""
        try:
            import mujoco
        except Exception:
            return

        target_indices: list[int] = []
        for idx in range(int(getattr(raw_model, "ntendon", 0))):
            try:
                tendon_name = mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_TENDON, int(idx)) or ""
            except Exception:
                tendon_name = ""
            name_lower = tendon_name.lower()
            if name_lower.endswith("_err") or "tip_err" in name_lower:
                target_indices.append(int(idx))

        if not target_indices:
            return

        models: list[Any] = [model]
        if raw_model is not model:
            models.append(raw_model)
        for target_model in models:
            for idx in target_indices:
                if idx >= int(getattr(target_model, "ntendon", 0)):
                    continue
                if hasattr(target_model, "tendon_rgba"):
                    target_model.tendon_rgba[idx, 3] = 0.0

    def _apply_qpos_and_render(self, qpos: np.ndarray, *, force: bool = False, ctrl_display: Optional[np.ndarray] = None) -> bool:
        now = time.perf_counter()
        if not force and (now - self._last_render_time) < self.min_render_interval_seconds:
            return False
        self._configure_clean_viewer()
        data = self.env.sim.data
        model = self.env.sim.model
        full_qpos = np.zeros((int(model.nq),), dtype=float)
        qpos_arr = np.asarray(qpos, dtype=float).reshape(-1)
        full_qpos[: min(len(full_qpos), len(qpos_arr))] = qpos_arr[: min(len(full_qpos), len(qpos_arr))]
        data.qpos[:] = full_qpos
        data.qvel[:] = 0.0
        if hasattr(data, "qacc"):
            data.qacc[:] = 0.0
        if ctrl_display is not None:
            ctrl_arr = self._clean_ctrl_for_viewer(ctrl_display)
            data.ctrl[:] = 0.0
            data.ctrl[: min(len(ctrl_arr), int(data.ctrl.shape[0]))] = ctrl_arr[: min(len(ctrl_arr), int(data.ctrl.shape[0]))]
        self.env.sim.forward()
        self.env.mj_render()
        self._configure_clean_viewer()
        self._last_render_time = time.perf_counter()
        return True

    def play(self) -> None:
        if self.trial is None:
            return
        if self.replay_mode == ReplayMode.NONE:
            if not self.replay_kinematics():
                return
        elif self.env is None:
            self.render_current()
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def reset(self) -> None:
        self.frame_index = 0
        self.playing = False
        if self.replay_mode == ReplayMode.CONTROL:
            self._control_sim_frame_index = None
        if self.replay_mode != ReplayMode.NONE:
            self.render_current()

    def set_frame(self, frame_index: int) -> None:
        if self.trial is None:
            return
        self.frame_index = int(np.clip(int(frame_index), 0, max(0, self._num_replay_frames() - 1)))
        if self.replay_mode == ReplayMode.CONTROL and self._control_sim_frame_index is not None and self.frame_index < self._control_sim_frame_index:
            self._control_sim_frame_index = None
        if self.env is not None:
            self.render_current()

    def step(self, delta: int = 1) -> None:
        if self.trial is None:
            return
        frame_count = self._num_replay_frames()
        self.frame_index = int(np.clip(self.frame_index + int(delta), 0, max(0, frame_count - 1)))
        self.render_current()
        if self.frame_index >= frame_count - 1:
            self.playing = False

    def status(self) -> ReplayStatus:
        return ReplayStatus(
            loaded=self.trial is not None,
            playing=bool(self.playing),
            frame_index=int(self.frame_index),
            num_frames=int(self._num_replay_frames()),
            message=str(self.message),
            viewer_open=self.env is not None,
            ctrl_active=self._ctrl_stats()[0],
            ctrl_max=self._ctrl_stats()[1],
            ctrl_mean=self._ctrl_stats()[2],
            internal_response_enabled=bool(self.use_ctrl_estimate_preview),
            replay_mode=str(self.replay_mode.value),
            control_source_path=str(self.control_source_path or ""),
        )

    def close(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = None
        self.playing = False
        self.replay_mode = ReplayMode.NONE
        self._control_sim_frame_index = None
        self.message = "MuJoCo replay viewer closed."
