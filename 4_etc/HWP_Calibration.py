# -*- coding: utf-8 -*-
"""
MATLAB -> Python conversion of:
- Plot raw/response-corrected spectro data vs wavelength
- Convert to energy axis and plot vs polarization
- Integrate intensity in a lambda range
- Robust sinusoidal fit with 180° periodicity and extremum detection
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.optimize import least_squares

# ============================================================
# User switches / parameters (same meaning as MATLAB)
# ============================================================

use_response_correction = False  # True: use response file correction; False: raw

folder_path = r"C:\Users\bwcho\Heriot-Watt University Team Dropbox\RES_EPS_Quantum_Photonics_Lab\Experiments\Current Experiments\Bay4_zhe_twse2\test\2026\2026-02\2026-02-02"
nn = "HWP-angletest-woQWP-860nm-0-179-1-1s2-2026-02-02_14-04-06.h5"
response_file = "HWP-response-0-180-2-whitelight-5s-2025-05-21_10-30-32.h5"

bg = 600

cmin, cmax = 0, 40
lambda_range = (840.1, 865.5)  # nm

smooth_window_energy = 1   # moving mean window along energy axis (MATLAB: smooth_window=1)

# ============================================================
# Helpers
# ============================================================

def read_h5_spectro(filepath):
    """Read spectro_data, spectro_wavelength, xPositions from a .h5 file."""
    with h5py.File(filepath, "r") as f:
        data = f["/spectro_data"][()]
        wvl = f["/spectro_wavelength"][()]
        xpos = f["/xPositions"][()]

    # MATLAB: data = double(reshape(data, le_size(le_size > 1)).');
    data = np.array(data, dtype=float)
    # Squeeze singleton dims and ensure 2D (n_angles, n_pixels)
    data = np.squeeze(data)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        # flatten everything except last axis (common: (1, n_pix, n_ang) etc.)
        # We'll reshape to (n_angles, n_pixels) assuming one axis is angles.
        # If your H5 layout differs, adjust here.
        data = data.reshape(data.shape[0], -1)

    wvl = np.array(wvl, dtype=float).squeeze()
    xpos = np.array(xpos, dtype=float).squeeze()

    return data, wvl, xpos


def movmean_1d(x, window):
    """Moving mean like MATLAB smoothdata(...,'movmean',window)."""
    if window <= 1:
        return x
    return uniform_filter1d(x, size=window, mode="nearest")


def sind_deg(x_deg):
    """sin() with degree input, like MATLAB sind."""
    return np.sin(np.deg2rad(x_deg))


def model_intensity(theta_deg, A, phi_deg, B):
    """A*sind(2*(theta + phi)) + B (all angles in degrees)."""
    return A * sind_deg(2.0 * (theta_deg + phi_deg)) + B


def residuals(params, x, y):
    A, phi, B = params
    return model_intensity(x, A, phi, B) - y


# ============================================================
# Main
# ============================================================

os.chdir(folder_path)

# --- Read experimental data
data, wvl, hwp_angles = read_h5_spectro(nn)
data = data - bg

collection_polarization = hwp_angles * 2.0  # same as MATLAB

processed_data = data.copy()

# --- Response correction (optional)
if use_response_correction:
    resp_data, resp_wvl, resp_angles = read_h5_spectro(response_file)
    resp_data = resp_data - bg

    # Angle matching check (MATLAB sorts and requires equality)
    exp_sorted = np.sort(hwp_angles)
    resp_sorted = np.sort(resp_angles)
    if exp_sorted.shape != resp_sorted.shape or not np.allclose(exp_sorted, resp_sorted, atol=1e-9):
        raise ValueError("实验角度与响应文件角度不匹配 (Angles mismatch between exp and response file)")

    # Smooth response along wavelength/pixel axis (movmean window=3)
    resp_smoothed = np.zeros_like(resp_data, dtype=float)
    for i in range(resp_data.shape[0]):
        resp_smoothed[i, :] = movmean_1d(resp_data[i, :], window=3)

    # Minimum response threshold (1% of global max)
    min_response = np.max(resp_smoothed) * 0.01
    resp_smoothed = np.maximum(resp_smoothed, min_response)

    # Normalize angle by angle
    normalized = np.zeros_like(data, dtype=float)
    # Note: Here we assume rows correspond to angles in the same order.
    # If your H5 stores angles unsorted, you'd need to reorder rows by sorting indices.
    for i in range(data.shape[0]):
        normalized[i, :] = data[i, :] / resp_smoothed[i, :]

    processed_data = normalized

# ============================================================
# Figure 1: Data vs wavelength (imagesc)
# ============================================================
plt.figure()
plt.imshow(
    processed_data,
    aspect="auto",
    origin="lower",
    extent=[wvl.min(), wvl.max(), hwp_angles.min(), hwp_angles.max()],
    cmap="RdYlBu_r",
)
plt.colorbar()
plt.xlabel("Wavelength (nm)", fontsize=13)
plt.ylabel("Voltage (V)", fontsize=13)  # keep same label as your MATLAB code

title_text = "Response Corrected Data vs Wavelength" if use_response_correction else "Original Data vs Wavelength"
plt.title(f"{title_text}\n{nn}", fontsize=14)

ax = plt.gca()
ax.tick_params(direction="in")
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# ============================================================
# Convert to energy axis and sort
# ============================================================
energy = 1240.0 / wvl  # eV
sort_idx = np.argsort(energy)
energy_sorted = energy[sort_idx]
data_sorted = processed_data[:, sort_idx]

# Smooth each spectrum (movmean along energy axis)
data_smoothed = np.zeros_like(data_sorted)
for i in range(data_sorted.shape[0]):
    data_smoothed[i, :] = movmean_1d(data_sorted[i, :], window=smooth_window_energy)

data_to_plot = data_smoothed

# ============================================================
# Figure 2: Energy vs polarization (pcolor -> pcolormesh)
# ============================================================
plt.figure(figsize=(5.6, 4.25), dpi=100)
X, Y = np.meshgrid(energy_sorted, collection_polarization)

# pcolormesh needs bin edges; easiest is shading='auto'
pcm = plt.pcolormesh(X, Y, data_to_plot, shading="auto", cmap="RdYlBu_r")
plt.xlabel("Energy (eV)", fontsize=13)
plt.ylabel("Collection Polarization (degree)", fontsize=13)
plt.title(nn, fontsize=14)

cb = plt.colorbar(pcm)
pcm.set_clim(cmin, cmax)

ax = plt.gca()
ax.tick_params(direction="in")
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.ylim(collection_polarization.min(), collection_polarization.max())

# ============================================================
# Integrated intensity in lambda_range
# ============================================================
# MATLAB: energy_range = 1240 ./ lambda_range([2,1])
# (note it flips because energy decreases with wavelength)
energy_min = 1240.0 / lambda_range[1]
energy_max = 1240.0 / lambda_range[0]

energy_mask = (energy_sorted >= energy_min) & (energy_sorted <= energy_max)
integrated_intensity = np.sum(data_to_plot[:, energy_mask], axis=1)

plt.figure()
plt.plot(collection_polarization, integrated_intensity, linewidth=2)
plt.xlabel("Collection Polarization (degree)", fontsize=13)
plt.ylabel("Integrated Intensity (a.u.)", fontsize=13)
plt.title(f"Integrated Intensity from {lambda_range[0]:.1f} nm to {lambda_range[1]:.1f} nm", fontsize=14)
plt.grid(True)

ax = plt.gca()
ax.tick_params(direction="in")
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
plt.ylim(0, 1.1 * np.max(integrated_intensity))

# ============================================================
# Robust sinusoidal fitting with 180° period + extremum detection
# Model: A*sind(2*(x + phi)) + B
# ============================================================
theta_sorted_idx = np.argsort(collection_polarization)
theta_sorted = collection_polarization[theta_sorted_idx]
intensity_sorted = integrated_intensity[theta_sorted_idx]

# Initial guess like MATLAB
max_idx = np.argmax(intensity_sorted)
min_idx = np.argmin(intensity_sorted)
max_int = intensity_sorted[max_idx]
min_int = intensity_sorted[min_idx]

A0 = (max_int - min_int) / 2.0
B0 = (max_int + min_int) / 2.0
phi0 = (45.0 - theta_sorted[max_idx]) / 2.0

p0 = np.array([A0, phi0, B0], dtype=float)

# Robust least squares (MATLAB Robust='Bisquare' -> use soft_l1/huber style)
res = least_squares(
    residuals,
    x0=p0,
    args=(theta_sorted, intensity_sorted),
    loss="soft_l1",
    f_scale=np.std(intensity_sorted) if np.std(intensity_sorted) > 0 else 1.0,
    max_nfev=5000,
)

A_fit, phi_fit, B_fit = res.x

# Compute R^2
y_pred = model_intensity(theta_sorted, A_fit, phi_fit, B_fit)
ss_res = np.sum((intensity_sorted - y_pred) ** 2)
ss_tot = np.sum((intensity_sorted - np.mean(intensity_sorted)) ** 2)
r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

# Extremum angles (0-180° range), same formulas as your MATLAB
x_max = np.mod((90.0 - 2.0 * phi_fit) / 2.0, 180.0)
x_min = np.mod(x_max + 90.0, 180.0)

# Generate fitted curve
theta_fit = np.linspace(0, 360, 500)
intensity_fit = model_intensity(theta_fit, A_fit, phi_fit, B_fit)

# Plot fit
plt.figure(figsize=(7.2, 5.0), dpi=100)
plt.scatter(theta_sorted, intensity_sorted, s=50, edgecolor="k", label="Experimental Data")
plt.plot(theta_fit, intensity_fit, linewidth=2.5, label=f"Fit (R² = {r2:.3f})")

# Mark extrema points
plt.plot(x_max, A_fit + B_fit, "^", markersize=12, markeredgecolor="k", label=f"Max at {x_max:.1f}°")
plt.plot(x_min, -A_fit + B_fit, "v", markersize=12, markeredgecolor="k", label=f"Min at {x_min:.1f}°")

plt.xlabel("Collection Polarization (degree)", fontsize=14, fontweight="bold")
plt.ylabel("Integrated Intensity (a.u.)", fontsize=14, fontweight="bold")
plt.title("Polarization Dependence with Extremum Detection", fontsize=15)

ax = plt.gca()
ax.set_xlim(0, 360)
ax.set_xticks(np.arange(0, 361, 30))
ax.set_ylim(0, np.max(intensity_sorted) * 1.1)
ax.grid(True, alpha=0.3)
ax.tick_params(direction="in")
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

plt.legend(loc="upper right")
plt.box(True)

plt.show()
