import os
import sys
from pathlib import Path
import argparse

# ensure Code/ is on sys.path so "import fitting..." works no matter where terminal cwd is
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("MPLCONFIGDIR", r"C:\Users\bwcho\AppData\Local\matplotlib")

import numpy as np
import matplotlib.pyplot as plt
from lmfit import Parameters
import json

from tools.instruments.princeton3.SPE3read import SPE3map
from tools.instruments.princeton3.SPE2read import SPE2map

from fitting.efield_sweep_fitter import fit_efield_band
from fitting.models import build_model
from fitting.region_config import REGIONS, BACKGROUND

HC_EV_NM = 1239.841984
BASE_DIR = Path(__file__).resolve().parent


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
    raise RuntimeError(f"nbOfFrames={nb_frames} is not a perfect square. nx, ny를 수동으로 넣어야 합니다.")


def attach_click_inspector(fig_map, ax_map, result):
    energy = result["energy_eV"]
    D = result["D"]
    region = result["region"][0]

    fig_ins, ax_ins = plt.subplots(figsize=(7.5, 4))

    def show_index(i: int):
        i = int(np.clip(i, 0, len(D) - 1))
        y_raw = result["spectra_raw"][i]
        y = result["spectra"][i]
        yfit = result["best_fit"][i]
        r2 = result["r2"][i]

        cfg = REGIONS[region]
        model, params = build_model(cfg, BACKGROUND, include_background=bool(result["include_background"]))

        ax_ins.clear()

        params = Parameters().loads(result["best_params_dump"][i])
        if hasattr(params, "update_constraints"):
            params.update_constraints()

        comps = model.eval_components(params=params, x=energy)

        ax_ins.plot(energy, y_raw, lw=1, alpha=0.30, label=f"raw (offset≈{result['baseline_constant']:.1f})")
        ax_ins.plot(energy, y, lw=1.2, label="offset-subtracted")
        ax_ins.plot(energy, yfit, lw=2.0, label="best fit")
        for name, arr in comps.items():
            ax_ins.plot(energy, arr, lw=1, alpha=0.7, label=name)

        ax_ins.set_title(f"i={i}, D=Vx-Vy={float(D[i]):.3f}, region={region}, R2={float(r2):.4f}")
        ax_ins.set_xlabel("Energy (eV)")
        ax_ins.set_ylabel("Counts")
        ax_ins.grid(True, alpha=0.3)
        ax_ins.legend(fontsize=8, ncols=2)
        fig_ins.tight_layout()
        fig_ins.canvas.draw_idle()

    def onclick(event):
        if event.inaxes != ax_map or event.button != 1:
            return
        d_click = float(event.ydata)
        i = int(np.argmin(np.abs(D - d_click)))
        show_index(i)

    fig_map.canvas.mpl_connect("button_press_event", onclick)
    print("Click inspector enabled: left-click on the D–Energy map.")
    return fig_ins, ax_ins


def load_anchor_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# USER EDIT HERE
# =========================
FIT_D_MIN = 0.0    # 예: 0 < D < 5만 피팅하려면 0.0
FIT_D_MAX = 5.0    # 예: 5.0
# FIT_D_MIN = None # 전체 피팅하려면 None
# FIT_D_MAX = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmin", type=float, default=FIT_D_MIN, help="Only fit frames with D >= dmin")
    ap.add_argument("--dmax", type=float, default=FIT_D_MAX, help="Only fit frames with D <= dmax")
    args = ap.parse_args()

    SPE_PATH = r"C:\Users\bwcho\Heriot-Watt University Team Dropbox\RES_EPS_Quantum_Photonics_Lab\Experiments\Current Experiments\Bay4_Tatyana_Cho_WS2WSe2\2026\2026-01\2026-01-28\2H-PL_elfield_dopping_map-HS5039_-6-6-0.05step-0.5s_85uW01-28_18-16.SPE"

    VX_MIN, VX_MAX = -6.0, 6.0
    VY_MIN, VY_MAX = -6.0, 6.0

    energy_eV, counts, nb_frames = load_spe(SPE_PATH)
    nx, ny = infer_grid_size(nb_frames)
    vx = np.linspace(VX_MIN, VX_MAX, nx)
    vy = np.linspace(VY_MIN, VY_MAX, ny)

    step = float(abs(vx[1] - vx[0])) if nx > 1 else 0.05
    s_tol = max(0.6 * step, 0.05)  # V=0 band 폭 (필요시 0.05~0.15로 조절)
    s0 = -0.4  # (-0.2,-0.2) 기준

    INCLUDE_BACKGROUND = True

    anchor_v0 = load_anchor_json(r"C:\Users\bwcho\OneDrive\1_Project\1_HWU\Projects\Dipolar ladder excitons_wTatyana\Code\anchor_v0.json")

    result = fit_efield_band(
        counts,
        energy_eV,
        vx,
        vy,
        s0=s0,
        s_tol=s_tol,
        region_name="ground",
        include_background=INCLUDE_BACKGROUND,
        baseline_constant=600.0,
        warm_start=True,
        method="least_squares",
        max_nfev=2000,

        anchor_v0=anchor_v0,
        anchor_rel=0.005,   # ±0.5% (너무 빡빡하면 0.01로)
        d0_tol=0.08,        # |D|<=0.08에서만 anchor (그리드 step에 맞춰 조절)

        # bg: center는 shift로 같이 움직이게(tie) 할 거라서, 여기서는 vary=True로 두어도 OK
        bg_sig_max=0.03,     # D 큰 쪽에서 bg가 넓어져야 하면 0.03~0.06로 올려보기
        bg_amp_factor=0.30,
        bg_cen_vary=True,

        # NEW: shared shift on centers (IX1/2/3 + bg)
        tie_shared_shift=True,
        tie_background_center_to_shift=True,
        dE_range=0.10,       # E-field에서 이동량 크면 0.10~0.15로
        fit_d_min=args.dmin,
        fit_d_max=args.dmax,
    )

    print("D≈0 start index k0 =", int(result.get("k0", -1)))
    print("Selected N =", len(result["D"]))
    print("D range =", float(np.min(result["D"])), "to", float(np.max(result["D"])))
    print("Median R2 =", float(np.nanmedian(result["r2"])))
    print("Success rate =", float(np.mean(result["success"])))

    # D–Energy map
    fig1, ax1 = plt.subplots(figsize=(7.5, 5))
    im = ax1.imshow(
        np.log10(np.clip(result["spectra"], 1, None)),
        aspect="auto",
        origin="lower",
        extent=[
            float(result["energy_eV"].min()),
            float(result["energy_eV"].max()),
            float(result["D"].min()),
            float(result["D"].max()),
        ],
        cmap="RdBu_r",
    )
    ax1.set_xlabel("Energy (eV)")
    ax1.set_ylabel("D = Vx - Vy (E-field axis)")
    ax1.set_title(f"V≈0 band (Vx+Vy≈{s0}) spectra vs E-field (log10)")
    fig1.colorbar(im, ax=ax1, label="log10 counts")
    fig1.tight_layout()
    attach_click_inspector(fig1, ax1, result)

    # --- Figure 2: R2 vs D (always show)
    fig2, ax2 = plt.subplots(figsize=(7.5, 3))
    ax2.plot(result["D"], result["r2"], ".-", ms=3)
    ax2.set_xlabel("D = Vx - Vy")
    ax2.set_ylabel("R²")
    ax2.grid(True, alpha=0.3)

    thr = 0.90
    r2 = np.asarray(result["r2"], float)
    D = np.asarray(result["D"], float)
    bad = np.where(np.isfinite(r2) & (r2 < thr))[0]
    if bad.size:
        i0 = int(bad[0])
        ax2.axvline(float(D[i0]), color="r", lw=1.5, alpha=0.8, label=f"first R²<{thr}: D={D[i0]:.3f}")
        ax2.legend()
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()