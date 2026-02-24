# -*- coding: utf-8 -*-
"""
MATLAB -> Python conversion:
- Load H5 spectro data
- Background subtraction
- Optional cosmic ray removal (statistical / median-based)
- Convert wavelength -> energy and sort
- Smooth each spectrum
- Split into two HWP groups via odd/even indexing
- Sort each group by B
- Plot 2D maps for each HWP
- Compute and plot "degree of valley polarization"
"""

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, median_filter
import tkinter as tk
from tkinter import filedialog
import io
from PIL import Image
import win32clipboard


# ============================================================
# User parameters (same meaning as MATLAB)
# ============================================================

# 파일 선택: 실행 시 폴더/파일 다이얼로그 열기
use_file_dialog = True

# Polarization 부호 반전 옵션
invert_polarization = True

folder_path = r"C:\Users\bwcho\Heriot-Watt University Team Dropbox\RES_EPS_Quantum_Photonics_Lab\Experiments\Current Experiments\Bay4_Tatyana_Cho_WS2WSe2\2026\2026-02\2026-02-02"
nn = "H2-PL_HS_-6_6_ratio_1_doping_step0.025_offset0_1s_90uW-02-02_17-20.h5"

# Plot parameters
cmin, cmax = 2, 500
xmin, xmax = 1.35, 1.55

# Background
bg = 600

# Cosmic ray removal
cosmic_threshold = 3          # std multiplier
median_filter_size = 3        # 3x3
enable_cosmic_removal = False # switch

# Smoothing window (movmean=3)
smooth_window = 3

# ============================================================
# Helper functions
# ============================================================

def read_h5(filepath):
    with h5py.File(filepath, "r") as f:
        data = f["/spectro_data"][()]
        wvl = f["/spectro_wavelength"][()]
        HWP_all = f["/xPositions"][()]
        B_all = f["/yPositions"][()]

    data = np.array(data, dtype=float).squeeze()
    wvl = np.array(wvl, dtype=float).squeeze()
    HWP_all = np.array(HWP_all, dtype=float).squeeze()
    B_all = np.array(B_all, dtype=float).squeeze()

    # Ensure 2D: (num_spectra, num_pixels)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim > 2:
        # If data stored with extra singleton dims, squeeze already did.
        # If still >2 dims, flatten everything except last axis
        data = data.reshape(data.shape[0], -1)

    return data, wvl, HWP_all, B_all


def movmean_1d(x, window):
    """Moving average like MATLAB smoothdata(...,'movmean',window)."""
    if window <= 1:
        return x
    return uniform_filter1d(x, size=window, mode="nearest")


def remove_cosmic_rays_statistical(data, threshold=3.0):
    """
    MATLAB-style: global mean/std -> mark outliers -> replace with local (3x3) median of normal neighbors.
    """
    data_cleaned = data.copy()
    mu = np.mean(data)
    sigma = np.std(data)
    if sigma == 0:
        return data_cleaned

    outliers = np.abs(data - mu) > threshold * sigma
    n_out = int(np.sum(outliers))
    if n_out == 0:
        return data_cleaned

    print(f"方法1检测到 {n_out} 个统计异常点")

    # replace each outlier with median of its 3x3 neighborhood excluding outliers
    n_rows, n_cols = data.shape
    global_med = np.median(data)

    for i in range(n_rows):
        i0 = max(0, i - 1)
        i1 = min(n_rows, i + 2)
        for j in range(n_cols):
            if not outliers[i, j]:
                continue
            j0 = max(0, j - 1)
            j1 = min(n_cols, j + 2)

            neighborhood = data[i0:i1, j0:j1]
            neigh_out = outliers[i0:i1, j0:j1]
            normal_neighbors = neighborhood[~neigh_out]

            if normal_neighbors.size > 0:
                data_cleaned[i, j] = np.median(normal_neighbors)
            else:
                data_cleaned[i, j] = global_med

    return data_cleaned


def remove_cosmic_rays_median(data, filter_size=3, threshold=3.0):
    """
    MATLAB-style:
    - median filter (symmetric padding)
    - diff = data - median
    - detect big positive spikes diff > threshold*std(diff)
    - replace with median-filtered value
    """
    data_cleaned = data.copy()

    # scipy median_filter uses reflect padding for mode='reflect' (close to symmetric)
    data_med = median_filter(data, size=(filter_size, filter_size), mode="reflect")
    diff = data - data_med
    diff_std = np.std(diff)
    if diff_std == 0:
        return data_cleaned

    cosmic_mask = diff > threshold * diff_std
    n_cosmic = int(np.sum(cosmic_mask))
    if n_cosmic > 0:
        print(f"方法2检测到 {n_cosmic} 个中值滤波异常点")
        data_cleaned[cosmic_mask] = data_med[cosmic_mask]

    return data_cleaned


def copy_fig_to_clipboard(fig):
    """Copy matplotlib figure to Windows clipboard as PNG."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        output = io.BytesIO()
        img.convert("RGB").save(output, "BMP")
        data = output.getvalue()[14:]
        output.close()
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()
        print("클립보드에 PNG 복사 완료")
    except Exception as e:
        print(f"클립보드 복사 실패: {e}")


def plot_2d_map(x, y, z, title, cmap="RdYlBu_r", clim=None, xlim=None, ylim=None, xlabel="", ylabel=""):
    fig = plt.figure(figsize=(5.6, 4.25), dpi=100)
    X, Y = np.meshgrid(x, y)

    # pcolor + shading interp -> pcolormesh with shading='auto' (closest)
    pcm = plt.pcolormesh(X, Y, z, shading="auto", cmap=cmap)
    cb = plt.colorbar(pcm)

    if clim is not None:
        pcm.set_clim(clim[0], clim[1])

    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.title(title, fontsize=14)

    ax = plt.gca()
    ax.tick_params(direction="out")
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.grid(True, which="both", alpha=0.4)

    return fig


# ============================================================
# Main
# ============================================================

if use_file_dialog:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    nn = filedialog.askopenfilename(
        title="Select .h5 file",
        initialdir=folder_path,
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    if not nn:
        raise SystemExit("No file selected.")

    # 선택된 파일 경로에서 폴더 자동 설정
    folder_path = os.path.dirname(nn)

os.chdir(folder_path)

# Read data
data, wvl, HWP_all, B_all = read_h5(nn)

# Background subtraction
data = data - bg

# Optional cosmic removal
if enable_cosmic_removal:
    data_filtered = remove_cosmic_rays_statistical(data, threshold=cosmic_threshold)
    # If you want method 2 as well:
    # data_filtered = remove_cosmic_rays_median(data_filtered, filter_size=median_filter_size, threshold=cosmic_threshold)
    data = data_filtered

# Convert to energy and sort
energy = 1240.0 / wvl
sort_idx = np.argsort(energy)
energy_sorted = energy[sort_idx]
data_sorted = data[:, sort_idx]

num_spectra = data_sorted.shape[0]
print(f"总光谱数量: {num_spectra}")

# Smooth each spectrum along energy axis
for i in range(num_spectra):
    data_sorted[i, :] = movmean_1d(data_sorted[i, :], window=smooth_window)

# Unique HWP and B values
HWP_unique = np.unique(HWP_all)
B_unique = np.unique(B_all)
if HWP_unique.size >= 2:
    print(f"HWP角度: {HWP_unique[0]:.1f}°, {HWP_unique[1]:.1f}°")
print(f"磁场范围: {np.min(B_unique):.2f} T 到 {np.max(B_unique):.2f} T，共{len(B_unique)}个点")

# Split by odd/even spectra indices (MATLAB: 1:2:end and 2:2:end)
odd_idx = np.arange(0, num_spectra, 2)   # Python 0-based: 0,2,4,...
even_idx = np.arange(1, num_spectra, 2)  # 1,3,5,...

print(f"B_all长度: {len(B_all)}")
print(f"HWP_all长度: {len(HWP_all)}")
print(f"奇数索引范围: {odd_idx.min()+1}到{odd_idx.max()+1}")  # +1 to mimic MATLAB print
print(f"偶数索引范围: {even_idx.min()+1}到{even_idx.max()+1}")

# Build "HWP_data" like MATLAB struct
HWP_data = [
    {
        "data": data_sorted[odd_idx, :],
        "B": B_all.copy(),
        "HWP_angle": float(HWP_all[odd_idx[0]]) if len(odd_idx) > 0 else np.nan
    },
    {
        "data": data_sorted[even_idx, :],
        "B": B_all.copy(),
        "HWP_angle": float(HWP_all[even_idx[0]]) if len(even_idx) > 0 else np.nan
    }
]

# Sort each group by B
for group in HWP_data:
    order = np.argsort(group["B"])
    group["B"] = group["B"][order]
    group["data"] = group["data"][order, :]

print(f"第一组HWP角度: {HWP_data[0]['HWP_angle']:.1f}°")
print(f"第二组HWP角度: {HWP_data[1]['HWP_angle']:.1f}°")

# ============================================================
# Plot 2D maps for each HWP
# ============================================================
for i, group in enumerate(HWP_data, start=1):
    fig = plot_2d_map(
        x=energy_sorted,
        y=group["B"],
        z=group["data"],
        title=f"HWP = {group['HWP_angle']:.1f}°",
        cmap="RdYlBu_r",
        clim=(cmin, cmax),
        xlim=(xmin, xmax),
        ylim=(float(np.min(group["B"])), float(np.max(group["B"]))),
        xlabel="Energy (eV)",
        ylabel="Voltage (V)",  # MATLAB said Gate Voltage, but data is B
    )
    copy_fig_to_clipboard(fig)

# ============================================================
# Valley polarization map
# MATLAB:
# data2 = -(HWP1 - HWP2)
# data3 = abs(HWP2 + HWP1) + 50
# data2 = data2 ./ data3
# ============================================================
data2 = -(HWP_data[0]["data"] - HWP_data[1]["data"])
if invert_polarization:
    data2 = -data2
data3 = np.abs(HWP_data[1]["data"] + HWP_data[0]["data"]) + 50.0
data2 = data2 / data3

fig = plot_2d_map(
    x=energy_sorted,
    y=HWP_data[0]["B"],
    z=data2,
    title="Degree of valley polarization",
    cmap="RdBu_r",
    clim=(-1, 1),
    xlim=(xmin, xmax),
    ylim=(float(np.min(HWP_data[0]["B"])), float(np.max(HWP_data[0]["B"]))),
    xlabel="Energy (eV)",
    ylabel="Voltage (V)",
)
copy_fig_to_clipboard(fig)

plt.show()
