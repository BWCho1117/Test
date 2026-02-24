# plot_power_map_fix.py
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Try to import scipy gaussian filter (recommended)
try:
    from scipy.ndimage import gaussian_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------------------- USER EDITABLE ----------------------
h5_paths = [
    r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\2026-01-30_power\2026-01-30_power\10_7_7.8nWto80nW_Power_dependancy at 5039 v=0(-0.2V)_0.01V-0.75V_step0.010V_01-30_13-07.h5",
    r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\2026-01-30_power\2026-01-30_power\10_6_90nWto800nW_Power_dependancy at 5039 v=0(-0.2V)_0.01V-0.75V_step0.010V_01-30_13-17.h5",
    r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\2026-01-30_power\2026-01-30_power\10_5_800nWto8.3uW_Power_dependancy at 5039 v=0(-0.2V)_0.01V-0.75V_step0.010V_01-30_13-55.h5",
    r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\2026-01-30_power\2026-01-30_power\10_4_9.6uWto84.5uW_Power_dependancy at 5039 v=0(-0.2V)_0.01V-0.90V_step0.010V_01-30_14-31.h5",
]

# choose smoothing: "soft", "medium", "strong"
smooth_choice = "medium"

# output options
OUTFILE_PDF = r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\2026-01-30_power\combined_power_map_smooth.pdf"
OUTFILE_PNG = OUTFILE_PDF.replace(".pdf", ".png")
FIG_DPI = 300

# calibration: voltage -> power (µW)
V_cal = np.array([0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.17,
                  0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
P_cal_uW = np.array([0.02, 0.10, 0.60, 2.17, 5.0, 12.0, 18.8,
                     30.0, 50.0, 70.0, 80.0, 90.0, 95.0, 100.0])

# ---------------------- helpers ----------------------
def load_file(path):
    with h5py.File(path, "r") as f:
        raw = f["spectro_data"][...]
        wvl = f["spectro_wavelength"][...]
        xLims = f["xLims"][...]
        xStep = float(f["xStep"][()])

    # collapse singleton detector dimension if present
    if raw.ndim == 3 and raw.shape[1] == 1:
        data = raw[:, 0, :]
    elif raw.ndim == 2:
        data = raw
    else:
        raise RuntimeError(f"Unexpected spectro_data shape {raw.shape} in {path}")

    voltages = np.arange(xLims[0], xLims[1] + 0.5 * xStep, xStep)
    n = min(len(voltages), data.shape[0])
    voltages = voltages[:n]
    data = data[:n, :]

    power_uW = np.interp(voltages, V_cal, P_cal_uW)

    energy_eV = 1240.0 / wvl
    idx = np.argsort(energy_eV)
    energy_sorted = energy_eV[idx]
    data_sorted = data[:, idx]
    return energy_sorted, power_uW, data_sorted

def common_energy_grid(energies, n_points=None):
    e_min = max(e.min() for e in energies)
    e_max = min(e.max() for e in energies)
    if n_points is None:
        n_points = int(np.median([len(e) for e in energies]))
    return np.linspace(e_min, e_max, n_points)

def interp_spectra_to_grid(energy_src, spectra_src, energy_dst):
    out = np.empty((spectra_src.shape[0], len(energy_dst)), dtype=float)
    for i in range(spectra_src.shape[0]):
        out[i, :] = np.interp(energy_dst, energy_src, spectra_src[i, :])
    return out

def edges_from_centers(x):
    x = np.asarray(x)
    if len(x) == 1:
        return np.array([x[0]-0.5, x[0]+0.5])
    dx = np.diff(x)
    dx = np.where(dx == 0, np.min(dx[dx > 0]) if np.any(dx > 0) else 1.0, dx)
    edges = np.empty(len(x) + 1)
    edges[1:-1] = (x[:-1] + x[1:]) / 2.0
    edges[0] = x[0] - dx[0] / 2.0
    edges[-1] = x[-1] + dx[-1] / 2.0
    return edges

def separable_gaussian_smooth(arr, sigma_y, sigma_x):
    # fallback if scipy not present: separable 1D convolutions
    def kernel1d(sigma):
        r = int(max(1, round(3 * sigma)))
        x = np.arange(-r, r+1, 1)
        k = np.exp(-(x**2) / (2 * sigma**2))
        k /= k.sum()
        return k
    ky = kernel1d(sigma_y)
    kx = kernel1d(sigma_x)
    # convolve along axis 0 (y)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, ky, mode='same'), 1, arr)
    out = np.apply_along_axis(lambda m: np.convolve(m, kx, mode='same'), 0, tmp)
    return out

# ---------------------- main processing ----------------------
# load files
loaded = []
energies = []
for p in h5_paths:
    if not os.path.isfile(p):
        raise FileNotFoundError(f"File not found: {p}")
    e, P, Z = load_file(p)
    loaded.append((p, e, P, Z))
    energies.append(e)
    print(f"Loaded: {os.path.basename(p)} -> frames={Z.shape[0]}, pix={Z.shape[1]}, P range {P.min():.3g}-{P.max():.3g} µW")

# common energy axis (overlap)
E = common_energy_grid(energies, n_points=None)

# bin per file in LINEAR power, average to remove step artefacts (smoothness from binning)
n_bins_per_file = 140   # change to taste (higher => smoother continuity)
gap_uW = 2.0            # small visual gap between stacked files

Y_parts = []
Z_parts = []
y_offset = 0.0

for p, e, P, Z in loaded:
    order = np.argsort(P)
    P_sorted = P[order]
    Z_sorted = Z[order]
    Zi = interp_spectra_to_grid(e, Z_sorted, E)
    Zi = np.clip(Zi, 1e-2, None)

    p_min, p_max = float(P_sorted.min()), float(P_sorted.max())
    if p_max <= p_min:
        continue
    bins = np.linspace(p_min, p_max, n_bins_per_file + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    Zb = np.full((n_bins_per_file, Zi.shape[1]), np.nan)
    for i in range(n_bins_per_file):
        mask = (P_sorted >= bins[i]) & (P_sorted < bins[i+1])
        if np.any(mask):
            Zb[i, :] = Zi[mask].mean(axis=0)
    valid = ~np.isnan(Zb).any(axis=1)
    centers = centers[valid]
    Zb = Zb[valid, :]
    y = centers + y_offset
    Y_parts.append(y)
    Z_parts.append(Zb)
    y_offset = y.max() + gap_uW

Y = np.concatenate(Y_parts)
Z = np.vstack(Z_parts)

# choose smoothing sigmas based on user choice
if smooth_choice == "soft":
    sigma_y, sigma_x = 0.8, 0.6
elif smooth_choice == "medium":
    sigma_y, sigma_x = 1.2, 1.0
elif smooth_choice == "strong":
    sigma_y, sigma_x = 2.0, 1.6
else:
    sigma_y, sigma_x = 1.2, 1.0

# smooth in LOG(intensity) space to preserve multiplicative contrasts
logZ = np.log10(Z)
if _HAS_SCIPY:
    logZ_smooth = gaussian_filter(logZ, sigma=(sigma_y, sigma_x), mode="nearest")
else:
    logZ_smooth = separable_gaussian_smooth(logZ, sigma_y, sigma_x)
Z_smooth = 10.0 ** logZ_smooth

# ---------------------- plotting (robust across Matplotlib versions) ----------------------
# sanity check shapes
assert Z_smooth.shape == (len(Y), len(E)), f"Z shape {Z_smooth.shape} vs (len(Y),len(E))=({len(Y)},{len(E)})"

E_edges = edges_from_centers(E)
Y_edges = edges_from_centers(Y)

vmin = np.percentile(Z_smooth, 20)
vmax = np.percentile(Z_smooth, 99.8)

plt.figure(figsize=(8.0, 6.8))

# IMPORTANT: use shading='auto' for broad matplotlib compatibility (avoids gouraud shape issues)
mesh = plt.pcolormesh(
    E_edges,
    Y_edges,
    Z_smooth,
    shading="auto",
    norm=LogNorm(vmin=vmin, vmax=vmax),
    cmap="viridis"
)

plt.xlabel("Energy (eV)", fontsize=12)
plt.ylabel("Stacked optical power (µW)", fontsize=12)
plt.title("Power-dependence map (combined, smooth)", fontsize=13)
plt.ylim(Y.min() * 0.98, Y.max() * 1.02)

cbar = plt.colorbar(mesh, pad=0.02)
cbar.set_label("Intensity (arb., log)", fontsize=11)

plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()

# save outputs
plt.savefig(OUTFILE_PDF, dpi=FIG_DPI, bbox_inches="tight")
plt.savefig(OUTFILE_PNG, dpi=FIG_DPI, bbox_inches="tight")
print("Saved:", OUTFILE_PDF, OUTFILE_PNG)

plt.show()

print("Smoothing backend:", "scipy.ndimage" if _HAS_SCIPY else "numpy separable fallback")
print("Smoothing sigmas (y, x):", sigma_y, sigma_x)
