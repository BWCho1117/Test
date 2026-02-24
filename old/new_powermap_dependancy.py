import os
import re
import math
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---------------------- USER SETTINGS ----------------------
# Put the full paths of the files you want to plot here:
h5_paths = [
    r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\2026-01-26\Power_dependancy at 3670 v=-3.12_voltage_scan_0.05V-0.45V_step0.010V_01-26_15-20.h5",
]


N_REMOVE = 1           # remove this many last frames (per file). 0 = keep all
SHARE_SCALE = True     # True = use same vmin/vmax for all maps; False = per-file autoscale
SAVE_FIG = False       # True = save the combined figure to disk
OUTFILE = "power_maps_comparison.png"  # used if SAVE_FIG is True
FIG_DPI = 200

# Calibration: Voltage -> Power (µW)
V_cal = np.array([0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.17,
                  0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
P_cal_uW = np.array([0.02, 0.10, 0.60, 2.17, 5.0, 12.0, 18.8,
                     30.0, 50.0, 70.0, 80.0, 90.0, 95.0, 100.0])

# ---------------------- helper funcs ----------------------
def extract_v_from_name(path):
    fn = os.path.basename(path)
    m = re.search(r"v=([0-9]*\.?[0-9]+)", fn)
    return m.group(1) if m else "unknown"

def load_and_prepare(path, n_remove=0):
    """Load file and return: energy_sorted (eV), power_uW (per frame), data_clipped (frames x energy)"""
    with h5py.File(path, "r") as f:
        raw = f["spectro_data"][...]            # might be (nV,1,nPix) or (nV,nPix)
        wvl = f["spectro_wavelength"][...]     # (nPix,)
        xLims = f["xLims"][...]
        xStep = float(f["xStep"][()])

    # collapse singleton detector dimension if present
    if raw.ndim == 3 and raw.shape[1] == 1:
        data = raw[:, 0, :]    # (nFrames, nPix)
    elif raw.ndim == 2:
        data = raw
    else:
        raise RuntimeError(f"Unexpected spectro_data shape {raw.shape} in {path}")

    # voltage axis and optional trimming
    voltages = np.arange(xLims[0], xLims[1] + 0.5 * xStep, xStep)
    if n_remove > 0:
        n_remove = min(n_remove, data.shape[0] - 1)
        data = data[:-n_remove, :]
        voltages = voltages[:-n_remove]

    # map voltages -> power (µW)
    power_uW = np.interp(voltages, V_cal, P_cal_uW)

    # convert wavelength -> energy (eV) and sort ascending energy
    energy_eV = 1240.0 / wvl
    idx = np.argsort(energy_eV)
    energy_sorted = energy_eV[idx]
    data_sorted = data[:, idx]

    # clip for log plotting (avoid zeros)
    data_clipped = np.clip(data_sorted, 1e-2, None)

    return energy_sorted, power_uW, data_clipped

# ---------------------- load all files ----------------------
datasets = []
for path in h5_paths:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    energy, power, data_clipped = load_and_prepare(path, n_remove=N_REMOVE)
    datasets.append({
        "path": path,
        "v": extract_v_from_name(path),
        "energy": energy,
        "power": power,
        "data": data_clipped
    })
    print(f"Loaded {os.path.basename(path)}: frames={data_clipped.shape[0]}, pix={data_clipped.shape[1]}")

# ---------------------- compute vmin/vmax (global or per-file) ----------------------
if SHARE_SCALE:
    # stack values to compute global percentiles
    all_vals = np.hstack([d["data"].ravel() for d in datasets])
    global_vmin = np.percentile(all_vals, 20)
    global_vmax = np.percentile(all_vals, 99.8)
    print(f"Using shared vmin/vmax: {global_vmin:.2e}, {global_vmax:.2e}")
else:
    for d in datasets:
        vals = d["data"].ravel()
        d["vmin"] = np.percentile(vals, 20)
        d["vmax"] = np.percentile(vals, 99.8)
        print(f"{os.path.basename(d['path'])} vmin/vmax: {d['vmin']:.2e}, {d['vmax']:.2e}")

# ---------------------- plotting layout ----------------------
n = len(datasets)
cols = min(3, n)
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)

# Flatten axes iterator
axes_flat = axes.flatten()

for ax_idx, (ax, d) in enumerate(zip(axes_flat, datasets)):
    energy = d["energy"]
    power = d["power"]
    data = d["data"]

    # decide vmin/vmax
    if SHARE_SCALE:
        vmin, vmax = global_vmin, global_vmax
    else:
        vmin, vmax = d["vmin"], d["vmax"]

    extent = [energy.min(), energy.max(), power.min(), power.max()]
    im = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        aspect="auto",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="RdBu_r"
    )

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Optical power (µW)")
    title = f"v={d['v']}"
    ax.set_title(title)

    # single colorbar per subplot if not using shared colorbar
    if not SHARE_SCALE:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Intensity (arb., log)")

# remove unused axes
for i in range(len(datasets), len(axes_flat)):
    fig.delaxes(axes_flat[i])

# Shared colorbar if requested
if SHARE_SCALE:
    # place a big colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax).set_label("Intensity (arb., log)")

plt.tight_layout(rect=[0, 0, 0.9, 1.0])  # leave space for shared colorbar
if SAVE_FIG:
    fig.savefig(OUTFILE, dpi=FIG_DPI)
    print(f"Saved figure to {OUTFILE}")

plt.show()
