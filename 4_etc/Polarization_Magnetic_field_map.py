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
# from matplotlib.widgets import Button  # <-- REMOVE (버튼 안 쓸 거면)

# ---------- Copy-to-clipboard controls ----------
ENABLE_COPY_PNG_HOTKEY = True
COPY_HOTKEYS = {"c", "ctrl+c"}   # 원하는 키로 바꾸세요

# 버튼 관련 옵션은 꺼두기(있다면)
ADD_COPY_PNG_BUTTON = False

# ---------- Normalization controls (NEW) ----------
NORMALIZATION_ENABLED = False          # True=오버레이/비교 플롯에서 정규화 사용, False=정규화 안 함
NORMALIZE_METHOD = "max"              # "max" | "area"
NORMALIZE_X_WINDOW = (1.39, 1.43)         # e.g. (1.38, 1.43) or None(전체)

# ---------- Polarization calculation controls ----------
POLARIZATION_USE_NORMALIZED = False   # 반드시 False: polarization은 raw로 계산

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
cmin, cmax = None, None

# Energy window (eV): used for normalization (and optionally x-limits)
xmin, xmax = 1.38 , 1.42

# Background
bg = 600

# Cosmic ray removal
cosmic_threshold = 3          # std multiplier
median_filter_size = 3        # 3x3
enable_cosmic_removal = False # switch

# Smoothing window (movmean=3)
smooth_window = 3

# ---------- NEW: Figure / axis view controls ----------
FIGSIZE_MAP = (5.6, 4.25)        # for 2D maps
FIGSIZE_COMPARE_BASE = (5, 1.5) # (width, height per row) for compare plots

COMPARE_YMIN = -0.05              # e.g. 0.0
COMPARE_YMAX = 15000             # e.g. 1.05

# ============================================================
# User parameters (same meaning as MATLAB)
# ============================================================

# Compare spectra at selected B-fields (same field, different polarization)
compare_B_list = [4,3, 2, 1, 0.5]   # Tesla
compare_B_tolerance = 0.15                  # Tesla (nearest match must be within this)
normalize_in_window = False                  # normalize using only xmin~xmax region
normalize_method = "max"                    # "max" (simple) or "area"

# ---------- Per-field polarization number display (ADD near other user controls) ----------
SHOW_POLARIZATION_VALUE_ON_COMPARE = True
PRINT_POLARIZATION_TABLE = True

# Use a window for the *polarization number* (RAW spectra). If None, reuse x_window passed to the plot fn.
POL_VALUE_X_WINDOW = (1.43, 1.46)   # eV (change as needed) or None
POL_VALUE_METRIC = "peak"           # "peak" | "area"
POL_EPS = 1e-12

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
        import io
        from PIL import Image
        import win32clipboard

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
        print("[copy] Figure copied to clipboard (PNG).")
    except Exception as e:
        print(f"[copy] Failed: {e}")


def enable_copy_png_hotkey(fig, hotkeys=COPY_HOTKEYS):
    """
    Add a keypress handler: press 'c' or 'ctrl+c' to copy figure to clipboard.
    No visible button is drawn on the figure.
    """
    if fig is None:
        return

    def _on_key(event):
        if event is None:
            return
        key = (event.key or "").lower()
        if key in hotkeys:
            copy_fig_to_clipboard(fig)

    cid = fig.canvas.mpl_connect("key_press_event", _on_key)

    # keep refs
    if not hasattr(fig, "_ui_refs"):
        fig._ui_refs = []
    fig._ui_refs.append(("copy_hotkey", cid))


def plot_2d_map(x, y, z, title, cmap="RdYlBu_r", clim=None, xlim=None, ylim=None, xlabel="", ylabel=""):
    fig = plt.figure(figsize=FIGSIZE_MAP, dpi=100)  # <-- MODIFIED (was hard-coded)
    X, Y = np.meshgrid(x, y)

    # pcolor + shading interp -> pcolormesh with shading='auto' (closest)
    pcm = plt.pcolormesh(X, Y, z, shading="auto", cmap=cmap)
    plt.colorbar(pcm)

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

    if ENABLE_COPY_PNG_HOTKEY:
        enable_copy_png_hotkey(fig)

    return fig


def plot_compare_polarizations_at_fields(
    HWP_data,
    energy,
    B_targets,
    tol=0.15,
    x_window=None,
    norm_method="max",
    # NEW: options (avoid NameError by not relying on globals)
    show_pol_value=True,
    print_pol_table=True,
    pol_value_x_window=None,
    pol_value_metric="peak",
    pol_eps=1e-12,
):
    """
    For each target B, pick nearest spectrum in each polarization group,
    optionally normalize for display, and optionally show polarization numbers
    computed from RAW (un-normalized) spectra.
    """
    n = len(B_targets)
    width, h_per_row = FIGSIZE_COMPARE_BASE
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(width, max(2.2, h_per_row * n)), sharex=True)  # <-- MODIFIED
    if n == 1:
        axes = [axes]

    g0, g1 = HWP_data[0], HWP_data[1]
    B0 = np.asarray(g0["B"], dtype=float)
    B1 = np.asarray(g1["B"], dtype=float)

    results = []

    # default window: if not provided, reuse plot x_window
    if pol_value_x_window is None:
        pol_value_x_window = x_window

    for ax, Bt in zip(axes, B_targets):
        i0, b0 = _nearest_index(B0, Bt)
        i1, b1 = _nearest_index(B1, Bt)

        if (abs(b0 - Bt) > tol) or (abs(b1 - Bt) > tol):
            ax.text(
                0.02, 0.5,
                f"B={Bt}T: no match within ±{tol}T\n(nearest {b0:.2f}T, {b1:.2f}T)",
                transform=ax.transAxes, va="center"
            )
            ax.grid(True, alpha=0.25)
            continue

        # --- raw (UN-normalized) spectra for polarization number ---
        y0_raw = np.asarray(g0["data"][i0, :], dtype=float)
        y1_raw = np.asarray(g1["data"][i1, :], dtype=float)

        # --- compare plot y-data: normalized or not (your existing toggle) ---
        if NORMALIZATION_ENABLED:
            use_window = NORMALIZE_X_WINDOW if NORMALIZE_X_WINDOW is not None else x_window
            use_method = NORMALIZE_METHOD if NORMALIZE_METHOD is not None else norm_method
            y0 = _normalize_spectrum(y0_raw, energy, x_window=use_window, method=use_method)
            y1 = _normalize_spectrum(y1_raw, energy, x_window=use_window, method=use_method)
            y_label = "Norm. Intensity"
        else:
            y0, y1 = y0_raw, y1_raw
            y_label = "Intensity (a.u.)"

        ax.plot(energy, y0, lw=1.2, label=f"HWP {g0['HWP_angle']:.1f}° (B={b0:.2f}T)")
        ax.plot(energy, y1, lw=1.2, label=f"HWP {g1['HWP_angle']:.1f}° (B={b1:.2f}T)")
        ax.set_title(f"Target B = {Bt} T", fontsize=10)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")

        # --- NEW: show polarization number on the subplot (computed from RAW) ---
        if show_pol_value:
            P = polarization_value_from_two_spectra(
                y0_raw, y1_raw, energy,
                x_window=pol_value_x_window,
                metric=pol_value_metric,
                eps=pol_eps
            )
            ax.text(
                0.02, 0.95,
                f"P={P:+.3f}  ({pol_value_metric}, {pol_value_x_window} eV)",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
            )
            results.append((Bt, b0, b1, P))

        # ---- existing: Y min/max control ----
        if (COMPARE_YMIN is not None) or (COMPARE_YMAX is not None):
            lo, hi = ax.get_ylim()
            lo = lo if COMPARE_YMIN is None else COMPARE_YMIN
            hi = hi if COMPARE_YMAX is None else COMPARE_YMAX
            ax.set_ylim(min(lo, hi), max(lo, hi))

    axes[-1].set_xlabel("Energy (eV)")
    if x_window is not None:
        axes[-1].set_xlim(x_window)

    fig.tight_layout()

    if ENABLE_COPY_PNG_HOTKEY:
        enable_copy_png_hotkey(fig)

    # --- NEW: print table to console ---
    if print_pol_table and results:
        print("\n[Polarization values (RAW spectra)]")
        print("TargetB(T)\tUsedB0(T)\tUsedB1(T)\tP")
        for Bt, b0, b1, P in results:
            print(f"{Bt:+.2f}\t\t{b0:+.2f}\t\t{b1:+.2f}\t\t{P:+.4f}")

    return fig


# ============================================================
# Helper functions
# ============================================================

def _nearest_index(arr: np.ndarray, value: float) -> tuple[int, float]:
    """
    Return (index, arr[index]) for the element closest to 'value'.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        raise ValueError("Empty array passed to _nearest_index().")
    i = int(np.nanargmin(np.abs(arr - float(value))))
    return i, float(arr[i])


def _normalize_spectrum(y: np.ndarray, x: np.ndarray, x_window=None, method: str = "max", eps: float = 1e-12) -> np.ndarray:
    """
    Normalize 1D spectrum y using only points within x_window.
    method:
      - "max": max(y in window) -> 1
      - "area": integral(y dx in window) -> 1
    If x_window is None, uses all points.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if x_window is None:
        sel = np.isfinite(x)
    else:
        x0, x1 = x_window
        lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
        sel = np.isfinite(x) & (x >= lo) & (x <= hi)

    if not np.any(sel):
        raise ValueError(f"_normalize_spectrum: no points in x_window={x_window}")

    if method == "max":
        scale = float(np.nanmax(y[sel]))
    elif method == "area":
        scale = float(np.trapz(y[sel], x[sel]))
    else:
        raise ValueError("method must be 'max' or 'area'")

    if (not np.isfinite(scale)) or scale <= eps:
        raise ValueError(f"_normalize_spectrum: bad scale={scale} for x_window={x_window}")

    return y / scale

def _mask_in_window(x: np.ndarray, x_window):
    x = np.asarray(x, dtype=float)
    if x_window is None:
        m = np.isfinite(x)
    else:
        a, b = x_window
        lo, hi = (a, b) if a <= b else (b, a)
        m = np.isfinite(x) & (x >= lo) & (x <= hi)
    if not np.any(m):
        raise ValueError(f"No points in x_window={x_window}")
    return m

def polarization_value_from_two_spectra(
    y_a: np.ndarray,
    y_b: np.ndarray,
    x: np.ndarray,
    x_window=None,
    metric: str = "peak",
    eps: float = 1e-12,
) -> float:
    """
    Compute a single polarization number from TWO RAW spectra (no normalization):
      P = (Ia - Ib) / (Ia + Ib)

    metric:
      - "peak": Ia = max(y_a in window), Ib = max(y_b in window)
      - "area": Ia = integral(y_a dx in window), Ib = integral(y_b dx in window)
    """
    y_a = np.asarray(y_a, dtype=float)
    y_b = np.asarray(y_b, dtype=float)
    x = np.asarray(x, dtype=float)

    m = _mask_in_window(x, x_window)

    if metric == "peak":
        Ia = float(np.nanmax(y_a[m]))
        Ib = float(np.nanmax(y_b[m]))
    elif metric == "area":
        Ia = float(np.trapz(y_a[m], x[m]))
        Ib = float(np.trapz(y_b[m], x[m]))
    else:
        raise ValueError("metric must be 'peak' or 'area'")

    return (Ia - Ib) / (Ia + Ib + eps)

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

# Build "HWP_data" like MATLAB struct (ensure B matches the selected spectra)
if len(B_all) == num_spectra:
    B_odd = B_all[odd_idx].copy()
    B_even = B_all[even_idx].copy()
elif len(B_all) == len(odd_idx) and len(B_all) == len(even_idx):
    # B provided once per field point (shared for both polarization streams)
    B_odd = B_all.copy()
    B_even = B_all.copy()
else:
    raise ValueError(
        f"B_all length mismatch: len(B_all)={len(B_all)}, "
        f"num_spectra={num_spectra}, odd={len(odd_idx)}, even={len(even_idx)}"
    )

HWP_data = [
    {
        "data": data_sorted[odd_idx, :],
        "B": B_odd,
        "HWP_angle": float(HWP_all[odd_idx[0]]) if len(odd_idx) > 0 else np.nan
    },
    {
        "data": data_sorted[even_idx, :],
        "B": B_even,
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

# --- NEW: compare (same B) spectra between two polarizations ---
xwin = (xmin, xmax) if normalize_in_window else None
fig_cmp = plot_compare_polarizations_at_fields(
    HWP_data=HWP_data,
    energy=energy_sorted,
    B_targets=compare_B_list,
    tol=compare_B_tolerance,
    x_window=xwin,
    norm_method=normalize_method,
    show_pol_value=True,
    print_pol_table=True,
    pol_value_x_window=POL_VALUE_X_WINDOW,
    pol_value_metric=POL_VALUE_METRIC,
    pol_eps=POL_EPS,
)
copy_fig_to_clipboard(fig_cmp)

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
    ylabel="B-field (T)",
)
copy_fig_to_clipboard(fig)

plt.show()

def compute_polarization_map(HWP_data, eps=1e-12):
    """
    Returns polarization map using UN-normalized intensities:
      P = (I0 - I1) / (I0 + I1)
    Assumes:
      HWP_data[0]["data"], HWP_data[1]["data"] are aligned as (nB, nE)
    """
    g0, g1 = HWP_data[0], HWP_data[1]

    # (중요) normalize된 걸 쓰지 않도록 강제
    I0_raw = np.asarray(g0["data"], dtype=float)
    I1_raw = np.asarray(g1["data"], dtype=float)

    # 만약 누가 실수로 normalized 데이터를 만들어뒀다면 여기서 차단
    if POLARIZATION_USE_NORMALIZED:
        raise ValueError("POLARIZATION_USE_NORMALIZED must be False for correct polarization.")

    P = (I0_raw - I1_raw) / (I0_raw + I1_raw + eps)
    return P

polarization_map = compute_polarization_map(HWP_data)
