import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from tools.instruments.princeton3.SPE3read import SPE3map
from tools.instruments.princeton3.SPE2read import SPE2map

from fitting.region_config import REGIONS, BACKGROUND, v_doping
from fitting.models import build_model, seed_params

HC_EV_NM = 1239.841984

# =========================
# USER DEFAULTS (edit here)
# =========================
DEFAULT_TARGET_VX = 1.1
DEFAULT_TARGET_VY = 1.1
DEFAULT_REGION = "ground"
DEFAULT_BASELINE = 600.0
DEFAULT_BASELINE_MODE = "constant"  # "constant" or "min"
DEFAULT_INCLUDE_BACKGROUND = True

# Optional: open file dialog starting folder
DEFAULT_SPE_START_DIR = r"C:\Users\bwcho\Heriot-Watt University Team Dropbox\RES_EPS_Quantum_Photonics_Lab\Experiments"


def _pick_spe_file_dialog(initial_dir: str | None = None) -> str:
    """Open a Windows file picker to choose a .SPE file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        raise SystemExit(f"tkinter is required for file dialog but failed to import: {e}")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select a .SPE file",
        initialdir=initial_dir or str(Path.cwd()),
        filetypes=[("Princeton SPE", "*.SPE;*.spe"), ("All files", "*.*")],
    )
    root.destroy()

    if not path:
        raise SystemExit("No file selected.")
    return path


def load_spe(path: str):
    try:
        data = SPE3map(path)
        kind = "SPE3"
    except Exception:
        data = SPE2map(path)
        kind = "SPE2"

    wavelength_nm = np.asarray(data.wavelength, dtype=float)
    energy_eV = HC_EV_NM / wavelength_nm
    counts = np.asarray([row[0] for row in data.data], dtype=float)  # (nFrames, nE)

    # enforce energy ascending
    if energy_eV[0] > energy_eV[-1]:
        energy_eV = energy_eV[::-1]
        counts = counts[:, ::-1]

    nb_frames = int(getattr(data, "nbOfFrames", counts.shape[0]))
    if counts.shape[0] != nb_frames:
        nb_frames = counts.shape[0]

    print(f"Loaded {kind}:")
    print(f"  spe        = {path}")
    print(f"  energy     = {energy_eV.shape}")
    print(f"  counts     = {counts.shape}")
    print(f"  nb_frames  = {nb_frames}")
    return energy_eV, counts, nb_frames


def infer_grid_size(nb_frames: int):
    side = int(np.sqrt(nb_frames))
    if side * side == nb_frames:
        return side, side
    raise RuntimeError(f"nbOfFrames={nb_frames} not a perfect square; set nx/ny manually.")


def diagonal_indices(vx_vals, vy_vals, tol):
    vx_vals = np.asarray(vx_vals, dtype=float)
    vy_vals = np.asarray(vy_vals, dtype=float)
    nx = len(vx_vals)
    ny = len(vy_vals)

    idxs, vxs, vys = [], [], []
    for j in range(ny):
        target = vy_vals[j]
        i = int(np.argmin(np.abs(vx_vals - target)))
        if abs(vx_vals[i] - target) <= tol:
            idxs.append(j * nx + i)
            vxs.append(float(vx_vals[i]))
            vys.append(float(vy_vals[j]))
    return np.asarray(idxs, dtype=int), np.asarray(vxs, dtype=float), np.asarray(vys, dtype=float)


def baseline_subtract(y_raw: np.ndarray, baseline_mode: str, baseline_constant: float):
    if baseline_mode == "constant":
        b = float(baseline_constant)
    elif baseline_mode == "min":
        b = float(np.min(y_raw))
    else:
        raise ValueError("baseline_mode must be 'constant' or 'min' for this GUI script.")
    y = np.clip(y_raw - b, 0.0, None)
    return y, b


@dataclass
class SliderSpec:
    name: str
    vmin: float
    vmax: float
    init: float


def make_slider_specs(params, y, energy, max_peaks=6, include_background=True):
    specs: list[SliderSpec] = []
    ymax = float(np.max(y)) if np.size(y) else 1.0

    if include_background and "bg_amp" in params:
        specs.append(SliderSpec("bg_amp", 0.0, max(10.0, 1.5 * ymax), float(params["bg_amp"].value)))
    if include_background and "bg_sig" in params:
        specs.append(SliderSpec("bg_sig", 0.004, 0.080, float(params["bg_sig"].value)))
    if include_background and "bg_cen" in params:
        c = float(params["bg_cen"].value)
        specs.append(SliderSpec("bg_cen", c - 0.05, c + 0.05, c))

    for i in range(max_peaks):
        a = f"p{i}_amp"
        c = f"p{i}_cen"
        s = f"p{i}_sig"
        if a not in params or c not in params or s not in params:
            continue
        cen = float(params[c].value)
        sig = float(params[s].value)
        specs.append(SliderSpec(a, 0.0, max(10.0, 2.0 * ymax), float(params[a].value)))
        specs.append(SliderSpec(c, cen - 0.03, cen + 0.03, cen))
        specs.append(SliderSpec(s, 0.004, 0.030, sig))

    return specs


def save_anchor_json(path: Path, region: str, cfg_peaks: list[str], params_dict: dict, baseline: float, target_v: float):
    peaks_out = {}
    for i, name in enumerate(cfg_peaks):
        peaks_out[name] = {
            "amp": float(params_dict.get(f"p{i}_amp", np.nan)),
            "cen": float(params_dict.get(f"p{i}_cen", np.nan)),
            "sig": float(params_dict.get(f"p{i}_sig", np.nan)),
        }

    out = {
        "region": region,
        "target_V": float(target_v),
        "baseline": float(baseline),
        "peaks": peaks_out,
        "background": {
            "amp": float(params_dict.get("bg_amp", np.nan)),
            "cen": float(params_dict.get("bg_cen", np.nan)),
            "sig": float(params_dict.get("bg_sig", np.nan)),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved anchor: {path}")


def main():
    ap = argparse.ArgumentParser()
    # CHANGED: --spe is optional; if missing, a file dialog opens
    ap.add_argument("--spe", default=None)

    ap.add_argument("--vxmin", type=float, default=-6.0)
    ap.add_argument("--vxmax", type=float, default=6.0)
    ap.add_argument("--vymin", type=float, default=-6.0)
    ap.add_argument("--vymax", type=float, default=6.0)

    # physical-V mode (optional)
    ap.add_argument("--targetV", type=float, default=None, help="physical V (after v_doping) to anchor near")

    # coordinate mode (optional; if not provided, defaults from top of file are used)
    ap.add_argument("--targetVx", type=float, default=None)
    ap.add_argument("--targetVy", type=float, default=None)

    ap.add_argument("--region", type=str, default=DEFAULT_REGION)
    ap.add_argument("--baseline", type=float, default=DEFAULT_BASELINE)
    ap.add_argument("--baselineMode", type=str, default=DEFAULT_BASELINE_MODE, choices=["constant", "min"])
    ap.add_argument("--includeBackground", action="store_true")
    ap.add_argument("--out", type=str, default="anchor.json")
    args = ap.parse_args()

    spe_path = args.spe or _pick_spe_file_dialog(DEFAULT_SPE_START_DIR)

    include_bg = bool(args.includeBackground) or bool(DEFAULT_INCLUDE_BACKGROUND)

    energy_eV, counts, nb_frames = load_spe(spe_path)
    nx, ny = infer_grid_size(nb_frames)

    vx = np.linspace(args.vxmin, args.vxmax, nx)
    vy = np.linspace(args.vymin, args.vymax, ny)

    step = float(abs(vx[1] - vx[0])) if nx > 1 else 0.05
    tol = max(0.6 * step, 0.02)
    print(f"Grid: nx={nx}, ny={ny}, step≈{step:.4f}, tol={tol:.4f}")

    # Decide pick mode:
    # - If targetVx/targetVy given -> coordinate mode
    # - Else if targetV given -> physical-V diagonal mode
    # - Else -> coordinate mode using DEFAULT_TARGET_VX/VY from top
    if (args.targetVx is not None) or (args.targetVy is not None) or (args.targetV is None):
        tx = float(DEFAULT_TARGET_VX if args.targetVx is None else args.targetVx)
        ty = float(DEFAULT_TARGET_VY if args.targetVy is None else args.targetVy)

        i = int(np.argmin(np.abs(vx - tx)))
        j = int(np.argmin(np.abs(vy - ty)))
        frame = int(j * nx + i)

        vx_pick = float(vx[i])
        vy_pick = float(vy[j])
        V_k = float(v_doping(vx_pick, vy_pick))

        print(f"Picked by coordinate: target=({tx:.4f},{ty:.4f})")
        print(f"Picked grid point   : (vx,vy)=({vx_pick:.4f},{vy_pick:.4f}), frame={frame}")
        print(f"Errors              : dvx={vx_pick-tx:+.6f}, dvy={vy_pick-ty:+.6f}")
        print(f"Vphys at picked point: v_doping(vx,vy)={V_k:.4f}")
        target_v_for_json = V_k
    else:
        idxs, vxs, vys = diagonal_indices(vx, vy, tol=tol)
        Vphys = np.array([v_doping(x, y) for x, y in zip(vxs, vys)], dtype=float)

        k = int(np.argmin(np.abs(Vphys - float(args.targetV))))
        frame = int(idxs[k])
        V_k = float(Vphys[k])

        print(f"Picked diagonal index k={k}, frame={frame}")
        print(f"Picked (vx,vy)=({vxs[k]:.4f},{vys[k]:.4f}), Vphys=v_doping(vx,vy)={V_k:.4f}")
        print(f"Requested targetV={float(args.targetV):.4f}, |Vphys-targetV|={abs(V_k-float(args.targetV)):.4g}")
        target_v_for_json = float(args.targetV)

    y_raw = counts[frame]
    y, b = baseline_subtract(y_raw, args.baselineMode, args.baseline)

    cfg = REGIONS[args.region]
    model, params = build_model(cfg, BACKGROUND, include_background=include_bg)

    res = model.fit(y, params, x=energy_eV)
    best = {name: float(p.value) for name, p in res.params.items()}
    seed_params(params, best)

    fig = plt.figure(figsize=(10.5, 8.5))
    ax = fig.add_axes([0.08, 0.52, 0.88, 0.44])
    (ln_raw,) = ax.plot(energy_eV, y_raw, lw=1, alpha=0.25, label=f"raw (offset≈{b:.1f})")
    (ln_y,) = ax.plot(energy_eV, y, lw=1.2, label="baseline-subtracted")
    (ln_fit,) = ax.plot(energy_eV, model.eval(params=params, x=energy_eV), lw=2.0, label="model (manual)")
    ax.set_title(f"Anchor GUI: region={args.region}")
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Counts")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    slider_specs = make_slider_specs(params, y, energy_eV, max_peaks=8, include_background=include_bg)
    slider_specs = slider_specs[:18]
    sliders = {}

    y_top = 0.46
    y_bottom = 0.06
    n = max(1, len(slider_specs))
    dy = (y_top - y_bottom) / n
    h = dy * 0.65

    for j, sp in enumerate(slider_specs):
        yj = y_top - (j + 1) * dy
        ax_s = fig.add_axes([0.12, yj, 0.72, h])
        sld = Slider(ax_s, sp.name, sp.vmin, sp.vmax, valinit=sp.init)
        sliders[sp.name] = sld

    ax_fit = fig.add_axes([0.86, 0.16, 0.10, 0.06])
    btn_fit = Button(ax_fit, "Auto Fit")

    ax_save = fig.add_axes([0.86, 0.08, 0.10, 0.06])
    btn_save = Button(ax_save, "Save JSON")

    def sync_from_sliders():
        for name, sld in sliders.items():
            if name in params and params[name].vary and (params[name].expr is None):
                params[name].value = float(sld.val)

    def redraw():
        ln_fit.set_ydata(model.eval(params=params, x=energy_eV))
        fig.canvas.draw_idle()

    def on_slider(_):
        sync_from_sliders()
        redraw()

    for sld in sliders.values():
        sld.on_changed(on_slider)

    def do_fit(_event=None):
        sync_from_sliders()
        r = model.fit(y, params, x=energy_eV, max_nfev=2000)
        best2 = {name: float(p.value) for name, p in r.params.items()}
        seed_params(params, best2)

        for name, sld in sliders.items():
            if name in params and params[name].expr is None:
                try:
                    sld.set_val(float(params[name].value))
                except Exception:
                    pass

        redraw()
        print(f"[Auto Fit] success={r.success}, chi2={r.chisqr:.4g}")

    def do_save(_event=None):
        p = {name: float(par.value) for name, par in params.items()}
        save_anchor_json(Path(args.out), args.region, cfg["peaks"], p, baseline=b, target_v=float(target_v_for_json))

    btn_fit.on_clicked(do_fit)
    btn_save.on_clicked(do_save)

    print("GUI usage: choose file -> move sliders -> Auto Fit -> Save JSON")
    plt.show()


if __name__ == "__main__":
    main()