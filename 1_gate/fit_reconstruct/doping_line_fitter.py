import numpy as np

from .region_config import v_doping, select_region, REGIONS, BACKGROUND
from .models import build_model, seed_params


def extract_doping_line_indices(vx_vals, vy_vals, tol=0.03):
    """
    Diagonal line: pick points where vx ~= vy.
    Flattening is assumed row-major: frame_index = j*nx + i
      i -> vx index, j -> vy index
    """
    vx_vals = np.asarray(vx_vals, dtype=float)
    vy_vals = np.asarray(vy_vals, dtype=float)
    nx = len(vx_vals)
    ny = len(vy_vals)

    idxs, vxs, vys, vsum = [], [], [], []
    for j in range(ny):
        target = vy_vals[j]
        i = int(np.argmin(np.abs(vx_vals - target)))
        if abs(vx_vals[i] - target) <= tol:
            idxs.append(j * nx + i)
            vxs.append(float(vx_vals[i]))
            vys.append(float(vy_vals[j]))
            vsum.append(float(vx_vals[i] + vy_vals[j]))

    return (
        np.asarray(idxs, dtype=int),
        np.asarray(vsum, dtype=float),
        np.asarray(vxs, dtype=float),
        np.asarray(vys, dtype=float),
    )


def _remove_baseline(y: np.ndarray, mode: str = "percentile", p: float = 5.0, constant: float = 0.0):
    y = np.asarray(y, dtype=float)
    if mode is None or mode == "none":
        return y, 0.0

    if mode == "percentile":
        b = float(np.percentile(y, p))
    elif mode == "min":
        b = float(np.min(y))
    elif mode == "constant":
        b = float(constant)
    else:
        raise ValueError(f"Unknown baseline mode: {mode}")

    y2 = np.clip(y - b, 0.0, None)
    return y2, b


def _refit_ix1e_center_at_v1(
    energy_eV: np.ndarray,
    y: np.ndarray,
    *,
    include_background: bool,
    expected_ix1e: float = 1.420,
):
    """
    Refit a single spectrum with low_electron model to get a stable IX1+e center.
    Uses multi-start on p1_amp and returns best p1_cen.
    Assumes low_electron peaks order: ["IX1", "IX1+e", "IX2"] => p1 == IX1+e.
    """
    cfg = REGIONS["low_electron"]
    model, params = build_model(cfg, BACKGROUND, include_background=include_background)

    # Nudge p1 (IX1+e) near expected; bounds should come from region_config if set.
    if "p1_cen" in params:
        params["p1_cen"].set(value=float(expected_ix1e), vary=True)

    # crude local amplitude guess near expected center
    m = np.abs(energy_eV - expected_ix1e) <= 0.010
    local_max = float(np.max(y[m])) if np.any(m) else float(np.max(y))
    local_max = max(local_max, 0.0)

    trials = []
    for init_amp in (0.0, min(max(0.2 * local_max, 20.0), 500.0)):
        p = params.copy()
        if "p1_amp" in p:
            p["p1_amp"].value = float(init_amp)
        if "p1_cen" in p:
            p["p1_cen"].value = float(expected_ix1e)
        trials.append(model.fit(y, p, x=energy_eV))

    best = min(trials, key=lambda r: r.aic)
    cen = float(best.params["p1_cen"].value) if "p1_cen" in best.params else float(expected_ix1e)
    return cen


def _apply_rel_constraint(params, name: str, value: float, rel_tol: float = 0.01, *, vary: bool = True):
    """
    Constrain a parameter around a reference value within ±rel_tol (fraction).
    Example: rel_tol=0.01 means ±1%.
    """
    if name not in params:
        return
    v = float(value)
    r = float(rel_tol)
    params[name].set(value=v, min=v * (1.0 - r), max=v * (1.0 + r), vary=bool(vary))


def _tighten_to_prev(params, name: str, prev_val: float, *, back_tol: float, fwd_max: float):
    """Prefer blue-shift: allow small backshift(back_tol), allow limited forward shift(fwd_max)."""
    if name not in params:
        return
    lo = prev_val - float(back_tol)
    hi = prev_val + float(fwd_max)
    # intersect with existing bounds
    if params[name].min is None or lo > params[name].min:
        params[name].min = lo
    if params[name].max is None or hi < params[name].max:
        params[name].max = hi


def fit_doping_line(
    counts_flat,
    energy_eV,
    vx_vals,
    vy_vals,
    tol=0.03,
    *,
    warm_start=True,
    baseline_mode: str = "percentile",
    baseline_percentile: float = 5.0,
    baseline_constant: float = 0.0,
    include_background: bool = True,

    lock_ix1e_center: bool = False,
    ix1e_ref_v: float = 1.0,
    ix1e_lock_vmin: float = 0.0,
    ix1e_lock_vmax: float = 1.0,
    ix1e_expected_eV: float = 1.420,

    # --- NEW: cap IX1 linewidth in 0<V<1 using V≈0 reference ---
    cap_ix1_sigma: bool = False,
    ix1_sigma_ref_v: float = 0.0,
    ix1_sigma_cap_factor: float = 1.20,

    # --- NEW (optional): prevent IX1+e from collapsing to 0 amplitude in 0<V<1 ---
    ix1e_amp_floor: float | None = None,  # e.g. 2.0 (counts). None => keep min=0
    vmin: float | None = None,
    vmax: float | None = None,

    max_nfev: int = 2000,
    method: str = "leastsq",
    # --- NEW: manual anchors from your own fits at V=0 and V=1 ---
    manual_anchor: dict | None = None,
    manual_anchor_vmin: float = 0.0,
    manual_anchor_vmax: float = 1.0,
):
    energy_eV = np.asarray(energy_eV, dtype=float)
    counts_flat = np.asarray(counts_flat, dtype=float)

    if energy_eV[0] > energy_eV[-1]:
        energy_eV = energy_eV[::-1]
        counts_flat = counts_flat[:, ::-1]

    idxs, Vsum, vxs, vys = extract_doping_line_indices(vx_vals, vy_vals, tol=tol)
    spectra_raw = counts_flat[idxs]
    Vphys = np.array([v_doping(x, y) for x, y in zip(vxs, vys)], dtype=float)

    # --- NEW: filter by V range to reduce work ---
    if vmin is not None or vmax is not None:
        lo = -np.inf if vmin is None else float(vmin)
        hi =  np.inf if vmax is None else float(vmax)
        keep = (Vphys >= lo) & (Vphys <= hi)

        idxs = idxs[keep]
        Vsum = Vsum[keep]
        vxs = np.asarray(vxs)[keep]
        vys = np.asarray(vys)[keep]
        spectra_raw = spectra_raw[keep]
        Vphys = Vphys[keep]

    def run_pass(lock_p1_cen: float | None, ix1_sig_ref: float | None):
        out = {
            "idxs": idxs,
            "Vsum": Vsum,
            "V": Vphys,
            "vx": vxs,
            "vy": vys,
            "energy_eV": energy_eV,
            "spectra_raw": spectra_raw,
            "spectra": [],
            "baseline": [],
            "region": [],
            "success": [],
            "r2": [],
            "best_fit": [],
            "best_params": [],
            "include_background": bool(include_background),
            "ix1e_center_ref": None if lock_p1_cen is None else float(lock_p1_cen),
        }

        prev_best = None
        for k in range(len(Vphys)):
            V_k = float(Vphys[k])
            region_name = select_region(V_k)
            cfg = REGIONS[region_name]

            y_raw = spectra_raw[k]
            y, b = _remove_baseline(
                y_raw,
                mode=baseline_mode,
                p=baseline_percentile,
                constant=baseline_constant,
            )

            model, params = build_model(cfg, BACKGROUND, include_background=include_background)
            if warm_start and prev_best is not None:
                seed_params(params, prev_best)

            # windows
            in_manual_window = (float(manual_anchor_vmin) <= V_k <= float(manual_anchor_vmax))

            # --- manual anchors (center/linewidth) ---
            if manual_anchor and region_name == "low_electron" and in_manual_window:
                rel = float(manual_anchor.get("rel", 0.01))
                ix1 = manual_anchor.get("ix1", None)
                ix1e = manual_anchor.get("ix1e", None)

                if ix1:
                    if "cen" in ix1: _apply_rel_constraint(params, "p0_cen", ix1["cen"], ix1.get("rel", rel), vary=True)
                    if "sig" in ix1: _apply_rel_constraint(params, "p0_sig", ix1["sig"], ix1.get("rel", rel), vary=True)
                if ix1e:
                    if "cen" in ix1e: _apply_rel_constraint(params, "p1_cen", ix1e["cen"], ix1e.get("rel", rel), vary=True)
                    if "sig" in ix1e: _apply_rel_constraint(params, "p1_sig", ix1e["sig"], ix1e.get("rel", rel), vary=True)

                if k == 0:
                    for n in ("p0_cen", "p0_sig", "p1_cen", "p1_sig"):
                        if n in params:
                            print("[anchor-applied]", n, "value/min/max=",
                                  float(params[n].value), float(params[n].min), float(params[n].max))

            # --- cap IX1 sigma (ONLY where intended) ---
            if cap_ix1_sigma and region_name == "low_electron" and in_lock_window:
                if (ix1_sig_ref is not None) and ("p0_sig" in params):
                    params["p0_sig"].max = float(ix1_sig_ref) * float(ix1_sigma_cap_factor)

            # --- lock IX1+e center from V≈1 reference (ONLY where intended) ---
            if (lock_p1_cen is not None) and region_name == "low_electron" and in_lock_window:
                if "p1_cen" in params:
                    params["p1_cen"].set(value=float(lock_p1_cen), vary=False)

            # --- optional: prevent IX1+e amplitude collapsing to zero (ONLY where intended) ---
            if (ix1e_amp_floor is not None) and region_name == "low_electron" and in_lock_window:
                if "p1_amp" in params:
                    params["p1_amp"].min = float(ix1e_amp_floor)

            res = model.fit(y, params, x=energy_eV, method=method, max_nfev=max_nfev)

            ss_res = float(np.sum((y - res.best_fit) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            out["spectra"].append(y.astype(np.float32))
            out["baseline"].append(float(b))
            out["region"].append(region_name)
            out["success"].append(bool(res.success))
            out["r2"].append(float(r2))
            out["best_fit"].append(res.best_fit.astype(np.float32))
            out["best_params"].append({name: float(p.value) for name, p in res.params.items()})

            # NEW: store full lmfit Parameters (includes expr/constraints)
            out.setdefault("best_params_dump", []).append(res.params.dumps())

            prev_best = {name: float(p.value) for name, p in res.params.items()}

        out["spectra"] = np.stack(out["spectra"], axis=0)
        out["best_fit"] = np.stack(out["best_fit"], axis=0)
        out["baseline"] = np.asarray(out["baseline"], dtype=float)
        out["r2"] = np.asarray(out["r2"], dtype=float)
        out["success"] = np.asarray(out["success"], dtype=bool)
        out["manual_anchor"] = manual_anchor
        out["manual_anchor_window"] = (float(manual_anchor_vmin), float(manual_anchor_vmax))
        return out

    # -------- Pass 1 (free) --------
    pass1 = run_pass(lock_p1_cen=None, ix1_sig_ref=None)

    # --- compute IX1 sigma reference at V≈0 ---
    ix1_sig_ref = None
    if cap_ix1_sigma:
        k0 = int(np.argmin(np.abs(pass1["V"] - float(ix1_sigma_ref_v))))
        bp0 = pass1["best_params"][k0]
        if "p0_sig" in bp0:
            ix1_sig_ref = float(bp0["p0_sig"])

    # --- compute IX1+e center reference at V≈1 ---
    ix1e_cen_ref = None
    if lock_ix1e_center:
        k1 = int(np.argmin(np.abs(pass1["V"] - float(ix1e_ref_v))))
        bp1 = pass1["best_params"][k1]
        ix1e_cen_ref = float(bp1.get("p1_cen", ix1e_expected_eV))

    # -------- Pass 2 (apply constraints in 0<V<1) --------
    if (lock_ix1e_center or cap_ix1_sigma or (ix1e_amp_floor is not None)):
        pass2 = run_pass(lock_p1_cen=ix1e_cen_ref, ix1_sig_ref=ix1_sig_ref)
        pass2["ix1e_center_ref"] = ix1e_cen_ref
        pass2["ix1_sigma_ref"] = ix1_sig_ref
        pass2["ix1_sigma_cap_factor"] = float(ix1_sigma_cap_factor)
        pass2["ix1e_amp_floor"] = ix1e_amp_floor
        return pass2

    return pass1


def _param_for_peak(cfg: dict, peak_name: str, field: str) -> str | None:
    """
    peak_name이 cfg["peaks"] 안에 있으면 해당 peak의 파라미터 이름(p{i}_{field})을 반환.
    field: "cen" | "sig" | "amp"
    """
    peaks = cfg.get("peaks", [])
    if peak_name not in peaks:
        return None
    i = int(peaks.index(peak_name))
    return f"p{i}_{field}"