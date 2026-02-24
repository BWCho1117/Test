from __future__ import annotations

import numpy as np
from lmfit import Parameters

# NOTE: use absolute imports to avoid "relative import with no known parent package"
from fitting.region_config import REGIONS, BACKGROUND
from fitting.models import build_model, seed_params

# --- NEW: adaptive window + weights helpers ---
def _adaptive_energy_mask(
    energy_eV: np.ndarray,
    params: Parameters,
    *,
    nsig: float = 6.0,
    pad: float = 0.0,
    min_points: int = 80,
) -> np.ndarray:
    """Mask selecting an energy window around predicted p0/p1/p2 peaks."""
    E = np.asarray(energy_eV, dtype=float)
    if E.size < int(min_points):
        return np.ones_like(E, dtype=bool)

    cens: list[float] = []
    sigs: list[float] = []
    for i in (0, 1, 2):
        ck = f"p{i}_cen"
        sk = f"p{i}_sig"
        if (ck in params) and (sk in params):
            cens.append(float(params[ck].value))
            sigs.append(float(abs(params[sk].value)))

    if not cens or not sigs:
        return np.ones_like(E, dtype=bool)
    if (not np.all(np.isfinite(cens))) or (not np.all(np.isfinite(sigs))):
        return np.ones_like(E, dtype=bool)

    width = float(nsig) * max(max(sigs), 1e-6) + float(pad)
    lo = float(min(cens) - width)
    hi = float(max(cens) + width)

    m = (E >= lo) & (E <= hi)
    if int(np.count_nonzero(m)) < int(min_points):
        return np.ones_like(E, dtype=bool)
    return m


def _peak_weights(
    energy_eV: np.ndarray,
    params: Parameters,
    *,
    nsig: float = 4.0,
    floor: float = 0.10,
) -> np.ndarray:
    """Weights emphasizing regions near predicted p0/p1/p2; normalized to max=1."""
    E = np.asarray(energy_eV, dtype=float)
    w = np.full_like(E, float(floor), dtype=float)

    for i in (0, 1, 2):
        ck = f"p{i}_cen"
        sk = f"p{i}_sig"
        if (ck not in params) or (sk not in params):
            continue
        cen = float(params[ck].value)
        sig = float(abs(params[sk].value))
        if (not np.isfinite(cen)) or (not np.isfinite(sig)) or sig <= 0:
            continue
        s = max(sig, 1e-6) * float(nsig)
        w += np.exp(-0.5 * ((E - cen) / s) ** 2)

    mx = float(np.max(w)) if w.size else 1.0
    return (w / mx) if mx > 0 else w


def _r2(y: np.ndarray, yfit: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yfit = np.asarray(yfit, dtype=float)
    ss_res = float(np.sum((y - yfit) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _grid_flat(vx: np.ndarray, vy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Row-major flatten convention:
      frame index = j*nx + i
      i -> vx index, j -> vy index
    """
    vx = np.asarray(vx, dtype=float)
    vy = np.asarray(vy, dtype=float)
    nx = len(vx)
    ny = len(vy)
    Vx_flat = np.tile(vx, ny)
    Vy_flat = np.repeat(vy, nx)
    return Vx_flat, Vy_flat


def _apply_rel_bound(params: Parameters, name: str, val: float, rel: float, *, vary: bool = True):
    if name not in params:
        return
    v = float(val)
    r = float(rel)
    lo = v * (1.0 - r)
    hi = v * (1.0 + r)
    if name.endswith("_sig"):
        lo = max(lo, 1e-6)
    params[name].set(value=v, min=min(lo, hi), max=max(lo, hi), vary=bool(vary))


def _apply_anchor(params: Parameters, region_cfg: dict, anchor_v0: dict, rel: float):
    """
    anchor_v0 expected:
      { "peaks": { "IX1": {"cen":..., "sig":...}, "IX2":..., "IX3":... } }
    region_cfg["peaks"] order -> p0, p1, p2 ...
    """
    peaks = region_cfg.get("peaks", [])
    a_peaks = (anchor_v0 or {}).get("peaks", {})
    for i, pname in enumerate(peaks):
        if pname not in a_peaks:
            continue
        pk = a_peaks[pname]
        if "cen" in pk:
            _apply_rel_bound(params, f"p{i}_cen", float(pk["cen"]), rel, vary=True)
        if "sig" in pk:
            _apply_rel_bound(params, f"p{i}_sig", float(pk["sig"]), rel, vary=True)


def _constrain_background(
    params: Parameters,
    y: np.ndarray,
    *,
    bg_sig_max: float = 0.03,
    bg_amp_factor: float = 0.35,
    bg_cen_vary: bool = True,
):
    y = np.asarray(y, dtype=float)
    ymax = float(np.max(y)) if y.size else 0.0

    if "bg_sig" in params:
        existing_min = float(getattr(params["bg_sig"], "min", 0.0) or 0.0)
        params["bg_sig"].min = max(existing_min, 1e-4)
        params["bg_sig"].max = float(bg_sig_max)

    if "bg_amp" in params and ymax > 0:
        params["bg_amp"].min = 0.0
        params["bg_amp"].max = float(bg_amp_factor) * ymax

    if "bg_cen" in params:
        params["bg_cen"].vary = bool(bg_cen_vary)


def _order_from_center_out(n: int, center_idx: int) -> list[int]:
    order = [center_idx]
    step = 1
    while len(order) < n:
        r = center_idx + step
        l = center_idx - step
        if r < n:
            order.append(r)
        if l >= 0 and len(order) < n:
            order.append(l)
        step += 1
    return order


def _tie_centers_to_shared_shift(
    params: Parameters,
    region_cfg: dict,
    anchor_v0: dict,
    *,
    tie_background: bool = True,
    dE_name: str = "dE",
    dE_init: float = 0.0,
    dE_min: float = -0.10,
    dE_max: float = 0.10,
):
    """
    Shared shift:
      p{i}_cen = anchor_cen_i + dE
      bg_cen   = bg0 + dE  (optional)

    anchor_v0 may optionally have:
      "background": {"cen": ...}
    """
    if dE_name not in params:
        params.add(dE_name, value=float(dE_init), min=float(dE_min), max=float(dE_max), vary=True)
    else:
        params[dE_name].set(value=float(dE_init), min=float(dE_min), max=float(dE_max), vary=True)

    peaks = region_cfg.get("peaks", [])
    a_peaks = (anchor_v0 or {}).get("peaks", {})

    for i, pname in enumerate(peaks):
        pk = a_peaks.get(pname)
        if not pk or "cen" not in pk:
            continue
        base = float(pk["cen"])
        p = f"p{i}_cen"
        if p in params:
            params[p].expr = f"{base} + {dE_name}"

    if tie_background and ("bg_cen" in params):
        bg0 = None
        if isinstance(anchor_v0, dict):
            bg0 = (anchor_v0.get("background") or {}).get("cen", None)

        if bg0 is None:
            vals = [float(a_peaks[p]["cen"]) for p in peaks if p in a_peaks and "cen" in a_peaks[p]]
            bg0 = float(np.mean(vals)) if vals else float(params["bg_cen"].value)

        params["bg_cen"].expr = f"{float(bg0)} + {dE_name}"


def _enforce_shared_shift_from_anchor(
    params: Parameters,
    region_cfg: dict,
    anchor_v0: dict,
    *,
    dE_name: str = "dE",
    dE_init: float = 0.0,
    dE_min: float = -0.10,
    dE_max: float = 0.10,
    tie_background: bool = True,
):
    # dE
    if dE_name not in params:
        params.add(dE_name, value=float(dE_init), min=float(dE_min), max=float(dE_max), vary=True)
    else:
        params[dE_name].set(value=float(dE_init), min=float(dE_min), max=float(dE_max), vary=True)

    # peaks
    peaks = region_cfg.get("peaks", [])
    a_peaks = (anchor_v0.get("peaks") or {})
    for i, pname in enumerate(peaks):
        if pname not in a_peaks or "cen" not in a_peaks[pname]:
            raise ValueError(f"anchor_v0 missing center for peak {pname!r}")
        base = float(a_peaks[pname]["cen"])
        key = f"p{i}_cen"
        if key in params:
            params[key].expr = f"{base} + {dE_name}"
            params[key].vary = False  # expr 쓰면 vary 의미 없지만, 명시적으로 잠금

    # background center
    if tie_background and ("bg_cen" in params):
        bg = anchor_v0.get("background") or {}
        if "cen" in bg:
            bg0 = float(bg["cen"])
            params["bg_cen"].expr = f"{bg0} + {dE_name}"
            params["bg_cen"].vary = False


def _constrain_bg_near_anchor(params: Parameters, anchor_v0: dict, *, amp_factor: float = 2.0, sig_factor: float = 1.6):
    """Keep bg_amp/bg_sig from blowing up broad+huge."""
    bg = anchor_v0.get("background") or {}
    if "bg_amp" in params and "amp" in bg and np.isfinite(bg["amp"]):
        a0 = float(bg["amp"])
        params["bg_amp"].min = 0.0
        params["bg_amp"].max = max(10.0, amp_factor * a0)

    if "bg_sig" in params and "sig" in bg and np.isfinite(bg["sig"]):
        s0 = float(bg["sig"])
        # broad+huge 방지: 상한을 anchor 근처로 제한
        params["bg_sig"].min = max(0.004, 0.6 * s0)
        params["bg_sig"].max = min(0.20, sig_factor * s0)


def _bg_center_out_of_window(energy_eV: np.ndarray, bg_cen: float, *, pad_eV: float = 0.0) -> bool:
    """Out-of-window if bg_cen is outside [Emin-pad, Emax+pad]."""
    e = np.asarray(energy_eV, dtype=float)
    if e.size < 2:
        return False
    emin = float(np.min(e)) - float(pad_eV)
    emax = float(np.max(e)) + float(pad_eV)
    if not np.isfinite(bg_cen):
        return False
    return (bg_cen < emin) or (bg_cen > emax)


def _disable_bg_if_out_of_window(
    params: Parameters,
    energy_eV: np.ndarray,
    *,
    m: float = 3.0,
    freeze_cen_sig: bool = True,
    mode: str = "center",       # NEW: "center" or "support"
    center_pad_eV: float = 0.0, # NEW: optional padding for center check
) -> bool:
    """
    Disable bg if it's not observable.
    mode="center": disable when bg_cen outside energy window (recommended for your case).
    mode="support": disable when [bg_cen - m*bg_sig, bg_cen + m*bg_sig] doesn't overlap window.
    """
    if ("bg_cen" not in params) or ("bg_sig" not in params) or ("bg_amp" not in params):
        return False

    bg_cen = float(params["bg_cen"].value)
    bg_sig = float(abs(params["bg_sig"].value))

    if mode == "center":
        out = _bg_center_out_of_window(energy_eV, bg_cen, pad_eV=float(center_pad_eV))
    elif mode == "support":
        out = _bg_out_of_window(energy_eV, bg_cen, bg_sig, m=float(m))
    else:
        raise ValueError(f"Unknown mode={mode!r}. Use 'center' or 'support'.")

    if out:
        params["bg_amp"].set(value=0.0, min=0.0, max=0.0, vary=False)
        if freeze_cen_sig:
            params["bg_cen"].vary = False
            params["bg_sig"].vary = False
        return True

    return False


def fit_efield_band(
    counts: np.ndarray,
    energy_eV: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    s0: float = -0.4,
    s_tol: float = 0.05,
    dmin: float | None = None,
    dmax: float | None = None,
    region_name: str = "ground",
    include_background: bool = True,
    baseline_constant: float = 600.0,
    baseline_clip_nonneg: bool = True,
    warm_start: bool = True,
    method: str = "least_squares",
    max_nfev: int = 2000,
    anchor_v0: dict | None = None,
    anchor_rel: float = 0.005,
    d0_tol: float = 0.08,
    # background constraints
    bg_sig_max: float = 0.03,
    bg_amp_factor: float = 0.35,
    bg_cen_vary: bool = True,
    # shared shift
    tie_shared_shift: bool = True,
    tie_background_center_to_shift: bool = True,
    dE_range: float = 0.10,
    # additional options
    bg_window_m: float = 3.0,   # NEW: how many sigmas to consider for "in-window"
    refit_bg_off: bool = True,  # keep arg even if not used for "refit"; we'll use it to enable disabling
    fit_d_min: float | None = None,   # NEW: only fit frames with D in [fit_d_min, fit_d_max]
    fit_d_max: float | None = None,   # NEW
    # --- NEW: robustness for edge frames ---
    two_stage_tie: bool = True,          # stage1: loose tol, stage2: tight tol
    stage1_tol: float = 0.10,            # ±10% (only for convergence)
    stage2_tol: float = 0.03,            # ±3% (your final constraint)
    r2_accept_for_warmstart: float = 0.60,  # only propagate good fits
    rescue_if_r2_below: float = 0.35,       # refit with wider window if below
    rescue_pad_eV: float = 0.030,           # widen adaptive window padding (eV)
) -> dict:
    energy_eV = np.asarray(energy_eV, dtype=float)
    counts = np.asarray(counts, dtype=float)

    # --- NEW: ensure x-axis is increasing (fix reversed x-axis plots & fit x) ---
    if energy_eV.size >= 2 and energy_eV[0] > energy_eV[-1]:
        energy_eV = energy_eV[::-1].copy()
        # counts: (nframes, nE) assumed. flip last axis
        if counts.ndim >= 2:
            counts = counts[..., ::-1].copy()

    Vx_flat, Vy_flat = _grid_flat(vx, vy)
    S = Vx_flat + Vy_flat
    D = Vx_flat - Vy_flat

    mask = np.abs(S - float(s0)) <= float(s_tol)

    # NEW: apply fit_d_min/fit_d_max too (this was previously ignored)
    if fit_d_min is not None:
        mask &= (D >= float(fit_d_min))
    if fit_d_max is not None:
        mask &= (D <= float(fit_d_max))

    idx = np.flatnonzero(mask).astype(int)
    if idx.size == 0:
        raise RuntimeError(f"No frames selected: |(Vx+Vy)-({s0})| <= {s_tol} produced 0 points.")

    order = np.argsort(D[idx])
    idx = idx[order]
    D_sel = D[idx]
    S_sel = S[idx]

    cfg = REGIONS[region_name]

    spectra_raw = counts[idx].astype(float)
    spectra = spectra_raw - float(baseline_constant)
    if baseline_clip_nonneg:
        spectra = np.clip(spectra, 0.0, None)

    best_fit = np.zeros_like(spectra, dtype=float)
    r2_list = np.full(idx.size, np.nan, dtype=float)
    success = np.zeros(idx.size, dtype=bool)
    per_idx_params = [{} for _ in range(idx.size)]
    per_idx_dump = [Parameters().dumps() for _ in range(idx.size)]

    k0 = int(np.argmin(np.abs(D_sel)))
    fit_order = _order_from_center_out(idx.size, k0)
    prev_best: dict | None = None
    prev_best_good: dict | None = None  # NEW: only warm-start from good frames

    bg_mode = []  # track per-frame background mode

    for kk in fit_order:
        y = spectra[kk]

        model, params = build_model(cfg, BACKGROUND, include_background=include_background)

        if include_background:
            _constrain_background(
                params,
                y,
                bg_sig_max=bg_sig_max,
                bg_amp_factor=bg_amp_factor,
                bg_cen_vary=bg_cen_vary,
            )

        # warm-start ONLY from good previous fit
        if warm_start and (prev_best_good is not None):
            params = seed_params(params, prev_best_good)

        # shared shift tie (kept)
        if tie_shared_shift and (anchor_v0 is not None):
            dE_init = float((prev_best_good or {}).get("dE", 0.0))
            _enforce_shared_shift_from_anchor(
                params,
                cfg,
                anchor_v0,
                dE_init=dE_init,
                dE_min=-float(dE_range),
                dE_max=+float(dE_range),
                tie_background=bool(tie_background_center_to_shift),
            )

        # near D=0: tighten anchor + keep dE ~ 0
        if (anchor_v0 is not None) and (abs(float(D_sel[kk])) <= float(d0_tol)):
            _apply_anchor(params, cfg, anchor_v0, float(anchor_rel))
            if "dE" in params:
                params["dE"].min = -0.01
                params["dE"].max = +0.01

        if include_background:
            _constrain_background(
                params,
                y,
                bg_sig_max=bg_sig_max,
                bg_amp_factor=bg_amp_factor,
                bg_cen_vary=bg_cen_vary,
            )

        if hasattr(params, "update_constraints"):
            params.update_constraints()

        # --- NEW: if bg is out-of-window, force bg off for this frame ---
        bg_disabled = False
        if include_background and refit_bg_off:
            bg_disabled = _disable_bg_if_out_of_window(
                params,
                energy_eV,
                m=float(bg_window_m),
                freeze_cen_sig=True,
                mode="center",        # NEW
                center_pad_eV=0.0,    # NEW (원하면 0.005~0.01 정도)
            )
            if hasattr(params, "update_constraints"):
                params.update_constraints()

        bg_mode.append("bg_off" if bg_disabled else "bg_on")

        # --- adaptive window + weights (your existing code) ---
        mE = _adaptive_energy_mask(energy_eV, params, nsig=6.0, pad=0.0, min_points=80)
        x_fit = energy_eV[mE]
        y_fit = y[mE]
        w_fit = _peak_weights(x_fit, params, nsig=4.0, floor=0.10)

        res = model.fit(y_fit, params, x=x_fit, weights=w_fit, max_nfev=max_nfev)

        # store FULL-grid eval (avoid shape mismatch)
        yfit_full = model.eval(params=res.params, x=energy_eV)
        best_fit[kk] = yfit_full
        r2_list[kk] = _r2(y, yfit_full)
        success[kk] = bool(getattr(res, "success", True))

        bp = {name: float(p.value) for name, p in res.params.items()}
        per_idx_params[kk] = bp
        per_idx_dump[kk] = res.params.dumps()
        prev_best = bp

    return {
        "energy_eV": energy_eV,
        "idx": idx,
        "D": D_sel,
        "S": S_sel,
        "region": np.array([region_name] * idx.size, dtype=object),
        "include_background": bool(include_background),
        "baseline_constant": float(baseline_constant),
        "s0": float(s0),
        "s_tol": float(s_tol),
        "spectra_raw": spectra_raw,
        "spectra": spectra,
        "best_fit": best_fit,
        "best_params": per_idx_params,
        "best_params_dump": per_idx_dump,
        "r2": r2_list,
        "success": success,
        "k0": k0,
        "bg_mode": bg_mode,  # track per-frame background mode
    }

    # --- DEBUG: check shared shift ties ---
    if tie_shared_shift and kk == 0:  # 또는 문제 프레임 kk에서만
        for name in ["p0_cen", "p1_cen", "p2_cen", "bg_cen", "dE"]:
            if name in params:
                print(f"[TIE CHECK] {name}: value={params[name].value}, expr={params[name].expr}")


def _require_triplet(cfg: dict) -> list[str]:
    peaks = list(cfg.get("peaks", []))
    if len(peaks) < 3:
        raise ValueError(f"Need at least 3 peaks in cfg['peaks'], got {peaks}")
    return peaks[:3]  # p0,p1,p2 order


def _apply_triplet_center_spacing_tie(
    params: Parameters,
    cfg: dict,
    anchor_v0: dict,
    *,
    dE_name: str = "dE",
    dE_init: float = 0.0,
    dE_min: float = -0.10,
    dE_max: float = +0.10,
    spacing_tol: float = 0.03,  # ±3%
):
    pnames = _require_triplet(cfg)

    a = anchor_v0.get("peaks", {})
    E0 = float(a[pnames[0]]["cen"])
    E1 = float(a[pnames[1]]["cen"])
    E2 = float(a[pnames[2]]["cen"])
    d01 = E1 - E0
    d12 = E2 - E1

    if dE_name not in params:
        params.add(dE_name, value=float(dE_init), min=float(dE_min), max=float(dE_max), vary=True)
    else:
        params[dE_name].set(value=float(dE_init), min=float(dE_min), max=float(dE_max), vary=True)

    if "g01" not in params:
        params.add("g01", value=1.0, min=1.0 - spacing_tol, max=1.0 + spacing_tol, vary=True)
    if "g12" not in params:
        params.add("g12", value=1.0, min=1.0 - spacing_tol, max=1.0 + spacing_tol, vary=True)

    for key in ("p0_cen", "p1_cen", "p2_cen"):
        if key not in params:
            raise ValueError(f"Expected parameter {key} in model params but not found.")

    params["p0_cen"].expr = f"{E0} + {dE_name}"
    params["p1_cen"].expr = f"({E0} + {dE_name}) + g01*({d01})"
    params["p2_cen"].expr = f"({E0} + {dE_name}) + g01*({d01}) + g12*({d12})"

    params["p0_cen"].vary = False
    params["p1_cen"].vary = False
    params["p2_cen"].vary = False


def _apply_triplet_sigma_tie_pm3pct(
    params: Parameters,
    cfg: dict,
    anchor_v0: dict,
    *,
    lw_tol: float = 0.03,  # ±3%
):
    """Keep p0/p1/p2 linewidths near anchor with ±lw_tol scaling."""
    pnames = _require_triplet(cfg)

    a = anchor_v0.get("peaks", {})
    s0 = float(a[pnames[0]]["sig"])
    s1 = float(a[pnames[1]]["sig"])
    s2 = float(a[pnames[2]]["sig"])

    for i, s in enumerate((s0, s1, s2)):
        sig_key = f"p{i}_sig"
        if sig_key not in params:
            raise ValueError(f"Expected parameter {sig_key} in model params but not found.")

        w = f"w{i}"
        if w not in params:
            params.add(w, value=1.0, min=1.0 - float(lw_tol), max=1.0 + float(lw_tol), vary=True)

        params[sig_key].expr = f"({s})*{w}"
        params[sig_key].vary = False