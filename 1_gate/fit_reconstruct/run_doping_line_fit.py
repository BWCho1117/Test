import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from lmfit import Parameters

BASE_DIR = Path(__file__).resolve().parent  # <-- 추가

from tools.instruments.princeton3.SPE3read import SPE3map
from tools.instruments.princeton3.SPE2read import SPE2map

from fitting.doping_line_fitter import fit_doping_line
from fitting.models import build_model, seed_params
from fitting.region_config import REGIONS, BACKGROUND

HC_EV_NM = 1239.841984


def load_spe(path: str):
    try:
        data = SPE3map(path)
        kind = "SPE3"
    except Exception:
        data = SPE2map(path)
        kind = "SPE2"

    wavelength_nm = np.asarray(data.wavelength, dtype=float)
    energy_eV = HC_EV_NM / wavelength_nm
    counts = np.asarray([row[0] for row in data.data], dtype=float)

    nb_frames = int(getattr(data, "nbOfFrames", counts.shape[0]))
    if counts.shape[0] != nb_frames:
        nb_frames = counts.shape[0]

    print(f"Loaded {kind}:")
    print("  energy_eV.shape =", energy_eV.shape)
    print("  counts.shape    =", counts.shape)
    print("  nb_frames       =", nb_frames)
    return energy_eV, counts, nb_frames


def infer_grid_size(nb_frames: int):
    side = int(np.sqrt(nb_frames))
    if side * side == nb_frames:
        return side, side
    raise RuntimeError(
        f"nbOfFrames={nb_frames} is not a perfect square. nx, ny를 수동으로 넣어야 합니다."
    )


def attach_click_inspector(fig_map, ax_map, result):
    """
    Left-click on the V–Energy map to inspect ONE spectrum + fit/components.
    """
    energy = result["energy_eV"]
    V = result["V"]

    fig_ins, ax_ins = plt.subplots(figsize=(7.5, 4))

    def show_index(i: int):
        i = int(np.clip(i, 0, len(V) - 1))
        V_i = float(V[i])
        region = result["region"][i]

        y_raw = result.get("spectra_raw", result["spectra"])[i]
        y = result["spectra"][i]
        yfit = result["best_fit"][i]
        baseline = float(result.get("baseline", np.zeros(len(V)))[i])

        include_bg = bool(result.get("include_background", True))

        cfg = REGIONS[region]
        model, params = build_model(cfg, BACKGROUND, include_background=include_bg)

        ax_ins.clear()

        if "best_params_dump" in result:
            params = Parameters().loads(result["best_params_dump"][i])
            if hasattr(params, "update_constraints"):
                params.update_constraints()
        else:
            # fallback: value dict
            saved = result["best_params"][i]
            for name, val in saved.items():
                if name in params:
                    params[name].set(value=float(val))
            if hasattr(params, "update_constraints"):
                params.update_constraints()

        comps = model.eval_components(params=params, x=energy)

        comp_sum = None
        for arr in comps.values():
            comp_sum = arr if comp_sum is None else (comp_sum + arr)

        y_eval = model.eval(params=params, x=energy)

        def _diff(a, b, tag):
            max_abs = float(np.max(np.abs(a - b)))
            max_rel = float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))
            print(f"[components-check] {tag}: max_abs={max_abs:.6g}, max_rel={max_rel:.6g}")

        if comp_sum is not None:
            _diff(comp_sum, yfit, "sum(components) vs stored best_fit")
        _diff(y_eval, yfit, "model.eval vs stored best_fit")
        if comp_sum is not None:
            _diff(comp_sum, y_eval, "sum(components) vs model.eval")

        ax_ins.plot(energy, y_raw, lw=1, alpha=0.30, label=f"raw (offset≈{baseline:.1f})")
        ax_ins.plot(energy, y, lw=1.2, label="offset-subtracted")
        ax_ins.plot(energy, yfit, lw=2.0, label="best fit")
        for name, arr in comps.items():
            ax_ins.plot(energy, arr, lw=1, alpha=0.7, label=name)

        ax_ins.set_title(f"i={i}, V={V_i:.3f}, region={region}, R2={result['r2'][i]:.4f}")
        ax_ins.set_xlabel("Energy (eV)")
        ax_ins.set_ylabel("Counts")
        ax_ins.grid(True, alpha=0.3)
        ax_ins.legend(fontsize=8, ncols=2)
        fig_ins.tight_layout()
        fig_ins.canvas.draw_idle()

        print(f"[inspect] i={i}, V={V_i:.3f}, region={region}, offset={baseline:.1f}, R2={result['r2'][i]:.4f}")

    def onclick(event):
        if event.inaxes != ax_map:
            return
        if event.button != 1:
            return
        y_click = float(event.ydata)
        i = int(np.argmin(np.abs(V - y_click)))
        show_index(i)

    fig_map.canvas.mpl_connect("button_press_event", onclick)
    print("Click inspector enabled: left-click on the V–Energy map.")
    return fig_ins, ax_ins


def _load_anchor_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Anchor JSON not found: {path} (cwd={Path.cwd()})")
    return json.loads(path.read_text(encoding="utf-8"))


def build_manual_anchor(anchor_v0_path: str, anchor_v1_path: str, rel: float = 0.01) -> dict:
    a0 = _load_anchor_json(anchor_v0_path)
    a1 = _load_anchor_json(anchor_v1_path)

    ix1 = a0["peaks"]["IX1"]

    # V=1 기준: IX1+e, IX2+e 둘 다 필요
    ix1e = a1["peaks"]["IX1+e"]
    ix2e = a1["peaks"]["IX2+e"]

    return {
        "rel": float(rel),
        "ix1":  {"cen": float(ix1["cen"]),  "sig": float(ix1["sig"]),  "rel": float(rel)},
        "ix1e": {"cen": float(ix1e["cen"]), "sig": float(ix1e["sig"]), "rel": float(rel)},
        "ix2e": {"cen": float(ix2e["cen"]), "sig": float(ix2e["sig"]), "rel": float(rel)},
    }


def verify_manual_anchor_applied(result: dict, manual_anchor: dict):
    """Check that p0/p1 stayed within ±rel in 0<V<1 & low_electron."""
    rel = float(manual_anchor.get("rel", 0.01))
    ix1_c0 = float(manual_anchor["ix1"]["cen"])
    ix1_s0 = float(manual_anchor["ix1"]["sig"])
    ix1e_c1 = float(manual_anchor["ix1e"]["cen"])
    ix1e_s1 = float(manual_anchor["ix1e"]["sig"])

    V = np.asarray(result["V"], dtype=float)
    regions = np.asarray(result["region"])
    bp = result["best_params"]

    mask = (V > 0.0) & (V < 1.0) & (regions == "low_electron")

    def _collect(name):
        out = np.full(len(bp), np.nan, dtype=float)
        for i in range(len(bp)):
            if name in bp[i]:
                out[i] = float(bp[i][name])
        return out

    p0_cen = _collect("p0_cen")
    p0_sig = _collect("p0_sig")
    p1_cen = _collect("p1_cen")
    p1_sig = _collect("p1_sig")

    dc0 = np.abs(p0_cen - ix1_c0) / ix1_c0
    ds0 = np.abs(p0_sig - ix1_s0) / ix1_s0
    dc1 = np.abs(p1_cen - ix1e_c1) / ix1e_c1
    ds1 = np.abs(p1_sig - ix1e_s1) / ix1e_s1

    def _report(label, arr):
        arr = arr[mask]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            print(f"[anchor-check] {label}: no points to check (mask empty)")
            return
        print(
            f"[anchor-check] {label}: max={arr.max()*100:.3f}%  "
            f"median={np.median(arr)*100:.3f}%  (tol={rel*100:.2f}%)"
        )

    _report("IX1 center  (p0_cen)", dc0)
    _report("IX1 sigma   (p0_sig)", ds0)
    _report("IX1+e center(p1_cen)", dc1)
    _report("IX1+e sigma (p1_sig)", ds1)

    bad = mask & ((dc0 > rel) | (ds0 > rel) | (dc1 > rel) | (ds1 > rel))
    print(f"[anchor-check] outside tol points: {int(np.sum(bad))} / {int(np.sum(mask))}")

    print("[anchor-check] unique regions in this run =", sorted(set(regions.tolist())))
    print("[anchor-check] points in (0<V<1 & low_electron) =", int(np.sum(mask)), "/", len(V))
    if np.sum(mask) > 0:
        ii = int(np.flatnonzero(mask)[0])
        print("[anchor-check] first masked point: i=", ii, "V=", float(V[ii]), "region=", regions[ii], "best_params keys=", list(bp[ii].keys())[:8])


def main():
    import fitting.doping_line_fitter as dlf
    print("[debug] doping_line_fitter loaded from:", dlf.__file__)

    SPE_PATH = r"C:\Users\bwcho\Heriot-Watt University Team Dropbox\RES_EPS_Quantum_Photonics_Lab\Experiments\Current Experiments\Bay4_Tatyana_Cho_WS2WSe2\2026\2026-01\2026-01-28\2H-PL_elfield_dopping_map-HS5039_-6-6-0.05step-0.5s_85uW01-28_18-16.SPE"

    VX_MIN, VX_MAX = -6.0, 6.0
    VY_MIN, VY_MAX = -6.0, 6.0

    energy_eV, counts, nb_frames = load_spe(SPE_PATH)
    nx, ny = infer_grid_size(nb_frames)

    vx = np.linspace(VX_MIN, VX_MAX, nx)
    vy = np.linspace(VY_MIN, VY_MAX, ny)

    step = float(abs(vx[1] - vx[0])) if nx > 1 else 0.05
    tol = max(0.6 * step, 0.02)
    print(f"Grid: nx={nx}, ny={ny}, step≈{step:.4f}, tol={tol:.4f}")

    INCLUDE_BACKGROUND = True   # <- 여기서 ON/OFF

    # --- anchor JSON: 네가 준 절대경로 그대로 사용 ---
    ANCHOR_V0 = r"C:\Users\bwcho\OneDrive\1_Project\1_HWU\Projects\Dipolar ladder excitons_wTatyana\Code\anchor_v0.json"
    ANCHOR_V1 = r"C:\Users\bwcho\OneDrive\1_Project\1_HWU\Projects\Dipolar ladder excitons_wTatyana\Code\anchor_v1.json"

    MANUAL_ANCHOR = build_manual_anchor(
        anchor_v0_path=ANCHOR_V0,
        anchor_v1_path=ANCHOR_V1,
        rel=0.004,  # ±0.2% (더 엄격)
    )

    result = fit_doping_line(
        counts,
        energy_eV,
        vx,
        vy,
        tol=tol,
        warm_start=True,
        baseline_mode="constant",
        baseline_constant=600.0,
        include_background=INCLUDE_BACKGROUND,

        manual_anchor=MANUAL_ANCHOR,
        manual_anchor_vmin=0.0,
        manual_anchor_vmax=1.0,

        # --- quick test: narrow V range ---
        vmin=-2,
        vmax=2,

        method="least_squares",
        max_nfev=1200,   # 일단 줄여서 빠르게 확인
    )

    verify_manual_anchor_applied(result, MANUAL_ANCHOR)

    print("IX1+e center ref =", result.get("ix1e_center_ref"), "at V=", result.get("ix1e_ref_V"))

    print("Doping line N =", len(result["V"]))
    print("V (phys) range =", float(np.min(result["V"])), "to", float(np.max(result["V"])))
    print("R2 median =", float(np.median(result["r2"])))
    print("Success rate =", float(np.mean(result["success"])))
    print("Baseline median (counts) =", float(np.median(result["baseline"])))

    # Plot 1: V–Energy map (click here)
    fig1, ax1 = plt.subplots(figsize=(7.5, 5))
    im = ax1.imshow(
        np.log10(np.clip(result["spectra"], 1, None)),
        aspect="auto",
        origin="lower",
        extent=[
            float(result["energy_eV"].min()),
            float(result["energy_eV"].max()),
            float(result["V"].min()),
            float(result["V"].max()),
        ],
        cmap="RdBu_r",
    )
    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel("V (phys)")
    ax1.set_title("Doping line spectra (log10)")
    fig1.colorbar(im, ax=ax1, label="log10 counts")
    fig1.tight_layout()

    attach_click_inspector(fig1, ax1, result)
    plt.show()

    # Plot 2: R^2 vs V
    plt.figure(figsize=(7.5, 3))
    plt.plot(result["V"], result["r2"], ".-", ms=3)
    plt.xlabel("V (phys)")
    plt.ylabel("R²")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()