"""
offline_biomech_reference.py
Jalur biomekanik offline: qpos → qvel → qacc → inverse dynamics → OSQP → ctrl_ref.
"""
from __future__ import annotations
import csv
import warnings
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.ndimage import uniform_filter1d

try:
    import osqp
    OSQP_AVAILABLE = True
except ModuleNotFoundError:
    OSQP_AVAILABLE = False
    warnings.warn("[offline_biomech_reference] osqp not found; fallback to PGD.", stacklevel=2)

_EPS = 1e-10
_DEFAULT_SMOOTH_WINDOW = 5
_DEFAULT_REG = 1e-4
_DEFAULT_SMOOTH_REG = 1e-4
_DEFAULT_OSQP_MAX_ITER = 4000


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def _indexed_cols(field_names: list[str], prefix: str) -> list[str]:
    cols: list[tuple[int, str]] = []
    for n in field_names:
        if not n.startswith(prefix):
            continue
        try:
            cols.append((int(n[len(prefix):]), n))
        except ValueError:
            pass
    return [c for _, c in sorted(cols)]


def load_qpos_trial_csv(path: Path) -> dict[str, Any]:
    """Muat CSV rekaman → dict dengan kunci: qpos, timestamps, exercise_name, trial_id."""
    path = Path(path).resolve()
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        qcols = _indexed_cols(fields, "qpos_")
        if not qcols:
            raise ValueError(f"Tidak ada kolom qpos_ di {path}")
        rows = list(reader)
    qpos = np.array([[float(r.get(c) or 0.0) for c in qcols] for r in rows], dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[0] == 0:
        raise ValueError(f"CSV kosong: {path}")
    t_arr = (
        np.array([float(r.get("t") or 0.0) for r in rows], dtype=np.float64)
        if "t" in fields
        else np.arange(qpos.shape[0], dtype=np.float64) / 60.0
    )
    exercise_name = str(rows[0].get("exercise_name", "")).strip() if rows else ""
    trial_id = str(rows[0].get("trial_id", path.stem)).strip() if rows else path.stem
    return {
        "source_csv": str(path),
        "exercise_name": exercise_name,
        "trial_id": trial_id or path.stem,
        "session_id": str(rows[0].get("session_id", "")).strip() if rows else "",
        "qpos": qpos,
        "timestamps": t_arr,
    }


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------
def compute_qvel_from_qpos(
    qpos: np.ndarray,
    timestamps: np.ndarray,
    *,
    smooth: bool = True,
    smooth_window: int = _DEFAULT_SMOOTH_WINDOW,
) -> np.ndarray:
    """Central finite difference + optional smoothing."""
    qpos = np.asarray(qpos, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    T = qpos.shape[0]
    dt = np.diff(t)
    dt = np.where(dt > 1e-9, dt, 1e-9)
    qvel = np.zeros_like(qpos)
    for i in range(1, T - 1):
        dth = (t[i + 1] - t[i - 1]) / 2.0
        qvel[i] = (qpos[i + 1] - qpos[i - 1]) / max(dth, 1e-9)
    qvel[0] = (qpos[1] - qpos[0]) / dt[0]
    qvel[-1] = (qpos[-1] - qpos[-2]) / dt[-1]
    if smooth and smooth_window > 1 and T > smooth_window:
        qvel = uniform_filter1d(qvel, size=int(smooth_window), axis=0)
    return qvel


def compute_qacc_from_qvel(
    qvel: np.ndarray,
    timestamps: np.ndarray,
    *,
    smooth: bool = True,
    smooth_window: int = _DEFAULT_SMOOTH_WINDOW,
) -> np.ndarray:
    return compute_qvel_from_qpos(qvel, timestamps, smooth=smooth, smooth_window=smooth_window)


# ---------------------------------------------------------------------------
# MuJoCo helpers
# ---------------------------------------------------------------------------
def _raw(obj: Any) -> Any:
    for attr in ("mjModel", "mjData"):
        if hasattr(obj, attr):
            return getattr(obj, attr)
    return obj


def _build_moment_matrix(model: Any, data: Any) -> np.ndarray:
    m, d = _raw(model), _raw(data)
    nu, nv = int(m.nu), int(m.nv)
    mujoco.mj_forward(m, d)
    AM = np.zeros((nv, nu), dtype=np.float64)
    try:
        mujoco.mju_sparse2dense(
            AM,
            d.actuator_moment.reshape(-1),
            d.moment_rownnz,
            d.moment_rowadr,
            d.moment_colind.reshape(-1),
        )
    except Exception:
        try:
            am = np.asarray(d.actuator_moment, dtype=np.float64)
            AM = am if am.shape == (nv, nu) else am.T if am.shape == (nu, nv) else AM
        except Exception:
            pass
    return AM


def run_inverse_dynamics(model: Any, data: Any, qpos: np.ndarray, qvel: np.ndarray, qacc: np.ndarray) -> np.ndarray:
    m, d = _raw(model), _raw(data)
    nq, nv = int(m.nq), int(m.nv)
    d.qpos[:] = 0.0; d.qpos[:min(nq, len(qpos))] = np.asarray(qpos, dtype=np.float64)[:min(nq, len(qpos))]
    d.qvel[:] = 0.0; d.qvel[:min(nv, len(qvel))] = np.asarray(qvel, dtype=np.float64)[:min(nv, len(qvel))]
    d.qacc[:] = 0.0; d.qacc[:min(nv, len(qacc))] = np.asarray(qacc, dtype=np.float64)[:min(nv, len(qacc))]
    d.ctrl[:] = 0.0
    if int(getattr(m, "na", 0)) > 0:
        try: d.act[:] = 0.0
        except Exception: pass
    mujoco.mj_inverse(m, d)
    return np.asarray(d.qfrc_inverse, dtype=np.float64).copy()


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def _solve_pgd(P: np.ndarray, q: np.ndarray, nu: int) -> np.ndarray:
    x = np.zeros(nu, dtype=np.float64)
    L = float(max(np.linalg.eigvalsh(P).max(), _EPS))
    step = 0.5 / L
    for _ in range(800):
        x = np.clip(x - step * (P @ x + q), 0.0, 1.0)
    return x


def solve_muscle_ctrl_from_qpos(
    model: Any,
    data: Any,
    qpos: np.ndarray,
    qvel: np.ndarray,
    qacc: np.ndarray,
    *,
    reg: float = _DEFAULT_REG,
    smooth_reg: float = _DEFAULT_SMOOTH_REG,
    prev_ctrl: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """qpos+qvel+qacc → ctrl_ref ∈ [0,1]^nu via inverse dynamics + OSQP."""
    m, d = _raw(model), _raw(data)
    nu, nv = int(m.nu), int(m.nv)
    nq = int(m.nq)

    q_ = np.zeros(nq); q_[:min(nq, len(qpos))] = np.asarray(qpos)[:min(nq, len(qpos))]
    d.qpos[:] = q_
    d.qvel[:min(nv, len(qvel))] = np.asarray(qvel, dtype=np.float64)[:min(nv, len(qvel))]

    AM = _build_moment_matrix(model, data)
    tau = run_inverse_dynamics(model, data, q_, qvel, qacc)
    b = tau - AM @ np.zeros(nu)  # bias assumed 0

    P = 2.0 * (AM.T @ AM) + 2.0 * reg * np.eye(nu)
    q_vec = -2.0 * (AM.T @ b)
    if prev_ctrl is not None and smooth_reg > 0:
        prev = np.clip(np.asarray(prev_ctrl, dtype=np.float64)[:nu], 0.0, 1.0)
        P += 2.0 * smooth_reg * np.eye(nu)
        q_vec -= 2.0 * smooth_reg * prev

    if OSQP_AVAILABLE:
        try:
            import scipy.sparse as sp
            prob = osqp.OSQP()
            prob.setup(sp.csc_matrix(P), q_vec, sp.eye(nu, format="csc"),
                       np.zeros(nu), np.ones(nu),
                       warm_starting=True, verbose=False, max_iter=_DEFAULT_OSQP_MAX_ITER)
            res = prob.solve()
            ctrl_ref = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, 1.0)
            status = str(res.info.status)
        except Exception:
            ctrl_ref = _solve_pgd(P, q_vec, nu)
            status = "pgd_fallback"
    else:
        ctrl_ref = _solve_pgd(P, q_vec, nu)
        status = "pgd"

    residual = float(np.linalg.norm(AM @ ctrl_ref - b))
    return ctrl_ref, {
        "residual_norm": residual,
        "relative_torque_error": residual / max(float(np.linalg.norm(b)), _EPS),
        "solver_status": status,
        "tau_required": tau.copy(),
    }


# ---------------------------------------------------------------------------
# Pipeline sequence
# ---------------------------------------------------------------------------
def build_biomech_reference_ctrl(
    model: Any,
    data: Any,
    qpos_seq: np.ndarray,
    timestamps: np.ndarray,
    *,
    smooth_vel: bool = True,
    smooth_window: int = _DEFAULT_SMOOTH_WINDOW,
    reg: float = _DEFAULT_REG,
    smooth_reg: float = _DEFAULT_SMOOTH_REG,
    verbose: bool = True,
) -> dict[str, Any]:
    """qpos_seq → ctrl_ref_seq via inverse dynamics + OSQP."""
    qpos_seq = np.asarray(qpos_seq, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    T = qpos_seq.shape[0]
    m = _raw(model)
    nu, nv = int(m.nu), int(m.nv)

    qvel_seq = compute_qvel_from_qpos(qpos_seq, timestamps, smooth=smooth_vel, smooth_window=smooth_window)
    qacc_seq = compute_qacc_from_qvel(qvel_seq, timestamps, smooth=smooth_vel, smooth_window=smooth_window)

    ctrl_ref_seq = np.zeros((T, nu), dtype=np.float64)
    tau_seq = np.zeros((T, nv), dtype=np.float64)
    residuals = np.zeros(T, dtype=np.float64)
    rel_errors = np.zeros(T, dtype=np.float64)
    prev_ctrl: np.ndarray | None = None

    for i in range(T):
        try:
            ctrl_ref, info = solve_muscle_ctrl_from_qpos(
                model, data, qpos_seq[i], qvel_seq[i], qacc_seq[i],
                reg=reg, smooth_reg=smooth_reg, prev_ctrl=prev_ctrl,
            )
            ctrl_ref_seq[i] = ctrl_ref
            tau_seq[i] = info["tau_required"]
            residuals[i] = float(info["residual_norm"])
            rel_errors[i] = float(info["relative_torque_error"])
            prev_ctrl = ctrl_ref.copy()
        except Exception as exc:
            if verbose:
                print(f"[biomech_ref][WARN] frame {i}: {exc}")
            if prev_ctrl is not None:
                ctrl_ref_seq[i] = prev_ctrl
        if verbose and i % max(1, T // 10) == 0:
            print(f"  [{i:4d}/{T}] residual={residuals[i]:.5f}")

    return {
        "qpos": qpos_seq.astype(np.float32),
        "qvel": qvel_seq.astype(np.float32),
        "qacc": qacc_seq.astype(np.float32),
        "tau_required": tau_seq.astype(np.float32),
        "ctrl_ref": ctrl_ref_seq.astype(np.float32),
        "residual_norms": residuals.astype(np.float32),
        "relative_torque_errors": rel_errors.astype(np.float32),
        "timestamps": timestamps,
        "num_frames": T, "nu": nu, "nv": nv,
    }


def process_recording_dir(
    records_dir: Path,
    model: Any,
    data: Any,
    *,
    exercise_name_filter: set[str] | None = None,
    smooth_window: int = _DEFAULT_SMOOTH_WINDOW,
    reg: float = _DEFAULT_REG,
    smooth_reg: float = _DEFAULT_SMOOTH_REG,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """Proses semua CSV di records_dir → list biomech reference dicts."""
    results: list[dict[str, Any]] = []
    for csv_path in sorted(Path(records_dir).resolve().glob("*.csv")):
        try:
            meta = load_qpos_trial_csv(csv_path)
        except Exception as exc:
            if verbose:
                print(f"[WARN] Skip {csv_path.name}: {exc}")
            continue
        if exercise_name_filter and meta["exercise_name"] not in exercise_name_filter:
            continue
        if verbose:
            print(f"[biomech_ref] {csv_path.name} | exercise={meta['exercise_name']}")
        try:
            out = build_biomech_reference_ctrl(model, data, meta["qpos"], meta["timestamps"],
                                               smooth_window=smooth_window, reg=reg,
                                               smooth_reg=smooth_reg, verbose=verbose)
            results.append({**meta, **out})
        except Exception as exc:
            if verbose:
                print(f"[ERROR] {csv_path.name}: {exc}")
    if verbose:
        print(f"[biomech_ref] Done: {len(results)} trials processed from {records_dir}")
    return results


if __name__ == "__main__":
    print("Run via: python -m app.offline_biomech.offline_biomech_reference --help")
