from __future__ import annotations

import glob
import os
import re
import struct
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl  # NEW

# =========================
# User settings
# =========================
USER_SELECT_FILES = True
DATA_DIR = "."  # used only when USER_SELECT_FILES=False or dialog cancelled

# CHANGED: set to None to show ALL v values found in filenames
TARGET_V: float | None = None
V_TOL = 1e-9

# Polarization filter:
# - None: include all pol angles found, group by pol
# - float: include only that pol (within POL_TOL)
TARGET_POL: float | None = None
POL_TOL = 1e-3

# Plot options
STACKED_OFFSET_STEP = 0.2

# Stacked overlay (one panel, different pols overlaid)
STACK_OVERLAY_ONE_PANEL = True
STACK_COMMON_X_INTERP_TOL = 1e-12
STACK_SHOW_B_LABEL_EVERY = 0  # 0 disables; e.g. 5 labels every 5th B-offset

# Map options
MAP_CMAP = "inferno"

# Map overlay (one panel, different pols overlaid)
MAP_OVERLAY_ONE_PANEL = True
MAP_OVERLAY_ALPHA = 0.75  # CHANGED (조금 올려도 마스킹으로 배경 덜 덮임)
MAP_OVERLAY_CMAPS = ("Blues", "Reds", "Greens", "Purples")  # CHANGED (대비 큰 조합)
MAP_SCALE_PERCENTILES = (2, 98)

# NEW: below this percentile -> transparent (per pol)
MAP_OVERLAY_MASK_BELOW_PERCENTILE = 65  # 50~80 사이에서 조절 추천


# =========================
# File picking
# =========================
def pick_spe_files_dialog(initial_dir: str | None = None) -> list[str]:
    """Select one or many .spe files via dialog. Returns [] on cancel/interruption."""
    root = tk.Tk()
    root.withdraw()
    try:
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        try:
            root.update_idletasks()
            root.update()
        except Exception:
            pass

        paths = filedialog.askopenfilenames(
            parent=root,
            title="Select .spe files (multi-select)",
            initialdir=initial_dir or os.getcwd(),
            filetypes=[("SPE files", "*.spe"), ("All files", "*.*")],
        )
        return list(paths)
    except KeyboardInterrupt:
        return []
    finally:
        try:
            root.destroy()
        except Exception:
            pass


# =========================
# Filename parsing / selection
# Rule:
#   v{num}_{den}_{B}T_{pol}[_{seconds}s].spe
# Examples:
#   v0_1_0.5T_72.5.spe        -> v=0/1=0.0, B=0.5, pol=72.5, exposure=10s(default)
#   v1_3_-2T_117.5_30s.spe    -> v=1/3, B=-2, pol=117.5, exposure=30s
# =========================
def _v_match(v: float, target_v: float | None, tol: float) -> bool:
    if target_v is None:
        return True
    return abs(float(v) - float(target_v)) <= float(tol)


def _pol_match(pol: float, target_pol: float | None, tol: float) -> bool:
    """
    If target_pol is None -> accept all.
    Else accept if |pol-target_pol| <= tol.
    """
    if target_pol is None:
        return True
    return abs(float(pol) - float(target_pol)) <= float(tol)


def parse_filename(path: str) -> tuple[float, float, float, int]:
    """
    Parse:
      v{num}_{den}_{B}T_{pol}[_{seconds}s].spe
    Returns:
      (v, B[T], pol[deg], exposure_s)
    """
    fname = os.path.basename(path)

    pat = (
        r"^v(?P<vnum>-?\d+)_(?P<vden>\d+)_"
        r"(?P<B>-?\d+(?:\.\d+)?)T_"
        r"(?P<pol>-?\d+(?:\.\d+)?)"
        r"(?:_(?P<texp>\d+)s)?"
        r"\.spe$"
    )

    m = re.match(pat, fname)
    if not m:
        raise ValueError(
            f"Filename format not recognized: {fname}\n"
            "Expected like:\n"
            "  v0_1_0.5T_72.5.spe\n"
            "  v0_1_2.5T_72.5_30s.spe"
        )

    vnum = int(m.group("vnum"))
    vden = int(m.group("vden"))
    if vden == 0:
        raise ValueError(f"Invalid v denominator (0) in filename: {fname}")

    v = vnum / vden
    B = float(m.group("B"))
    pol = float(m.group("pol"))
    exposure_s = int(m.group("texp")) if m.group("texp") is not None else 10
    return float(v), B, pol, exposure_s


def collect_files_in_dir(
    data_dir: str,
    target_v: float | None,
    target_pol: float | None,
    pol_tol: float
) -> list[tuple[float, str]]:
    paths = glob.glob(os.path.join(data_dir, "*.spe"))
    return collect_from_paths(paths, target_v, target_pol, pol_tol)


def collect_from_paths(
    paths: list[str],
    target_v: float | None,
    target_pol: float | None,
    pol_tol: float
) -> list[tuple[float, str]]:
    """
    Returns sorted list of (B, filepath), filtered by v (optional) and pol (optional).
    """
    selected: list[tuple[float, str]] = []
    for p in paths:
        if not p.lower().endswith(".spe"):
            continue
        v, B, pol, _texp = parse_filename(p)
        if _v_match(v, target_v, V_TOL) and _pol_match(pol, target_pol, pol_tol):
            selected.append((float(B), p))

    selected.sort(key=lambda t: t[0])
    if not selected:
        raise RuntimeError(f"No matching .spe files for v={target_v}, pol={target_pol}±{pol_tol}")
    return selected


def group_by_v_then_pol(
    selected: list[tuple[float, str]],
    *,
    v_round: int = 8,
    pol_round: int = 3
) -> dict[float, dict[float, list[tuple[float, str]]]]:
    """
    selected: [(B, path), ...]
    returns: {v: {pol: [(B, path), ...]}}
    """
    out: dict[float, dict[float, list[tuple[float, str]]]] = {}
    for B, p in selected:
        v, _B, pol, _texp = parse_filename(p)
        vkey = round(float(v), v_round)
        pkey = round(float(pol), pol_round)
        out.setdefault(vkey, {}).setdefault(pkey, []).append((float(B), p))

    # sort each (v, pol) by B
    for vkey in out:
        for pkey in out[vkey]:
            out[vkey][pkey].sort(key=lambda t: t[0])

    return dict(sorted(out.items(), key=lambda kv: kv[0]))


# =========================
# SPE loading
# Supports:
#   - Princeton/WinSpec binary .SPE (classic 4100-byte header)
#   - (optional) 2-col text fallback
# =========================
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _looks_binary_spe(path: str, sniff_bytes: int = 2048) -> bool:
    with open(path, "rb") as fb:
        chunk = fb.read(sniff_bytes)
    return (b"\x00" in chunk) and (chunk.count(b"\x00") > 10)


def _load_binary_spe_princeton(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal WinSpec/Princeton binary .SPE reader (classic 4100-byte header).
    Returns (x, y) where x is pixel index.
    """
    HEADER_BYTES = 4100

    with open(path, "rb") as f:
        header = f.read(HEADER_BYTES)
        if len(header) < HEADER_BYTES:
            raise ValueError(f"SPE header too small: {os.path.basename(path)}")

        def u16(off: int) -> int:
            return int(struct.unpack_from("<H", header, off)[0])

        def i32(off: int) -> int:
            return int(struct.unpack_from("<i", header, off)[0])

        # Common offsets used by many WinSpec SPE variants
        xdim = u16(42)          # pixels in x
        ydim = u16(656)         # pixels in y
        dtype_code = u16(108)   # data type code
        nframes = i32(1446)     # number of frames

        if xdim <= 0 or ydim <= 0:
            raise ValueError(f"Invalid SPE dims parsed: xdim={xdim}, ydim={ydim} ({os.path.basename(path)})")
        if nframes <= 0:
            nframes = 1

        code_to_dtype = {
            0: np.float32,
            1: np.int32,
            2: np.int16,
            3: np.uint16,
            5: np.float64,
            6: np.uint8,
            8: np.uint32,
        }
        if dtype_code not in code_to_dtype:
            raise ValueError(f"Unsupported SPE dtype code={dtype_code} ({os.path.basename(path)})")

        dtype = code_to_dtype[dtype_code]
        bytes_per = np.dtype(dtype).itemsize
        expected = int(xdim) * int(ydim) * int(nframes) * bytes_per

        # Read payload
        f.seek(0, os.SEEK_END)
        fsize = f.tell()
        f.seek(HEADER_BYTES, os.SEEK_SET)

        if fsize >= HEADER_BYTES + expected:
            raw = np.fromfile(f, dtype=dtype, count=int(xdim) * int(ydim) * int(nframes))
        else:
            # fallback: read everything after header (some files have inconsistent metadata)
            raw = np.fromfile(f, dtype=dtype)

    if raw.size < xdim * ydim:
        raise ValueError(f"Failed to read SPE payload: {os.path.basename(path)} (read {raw.size} values)")

    if raw.size == xdim * ydim * nframes:
        cube = raw.reshape((nframes, ydim, xdim))
        img = cube.mean(axis=0)  # average frames
    else:
        img = raw[: xdim * ydim].reshape((ydim, xdim))

    # Convert 2D detector image -> 1D spectrum by summing rows
    y = img.sum(axis=0).astype(float).ravel()
    x = np.arange(y.size, dtype=float)  # pixel index
    return x, y


def _load_text_two_col(path: str) -> tuple[np.ndarray, np.ndarray]:
    # Try tolerant read first
    for enc in ("utf-8", "cp949", "latin1"):
        try:
            arr = np.genfromtxt(path, dtype=float, comments="#", delimiter=None, invalid_raise=False, encoding=enc)
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.size >= 2:
                arr = arr.reshape(1, -1)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                x = arr[:, 0].astype(float)
                y = arr[:, 1].astype(float)
                m = np.isfinite(x) & np.isfinite(y)
                x, y = x[m], y[m]
                if x.size >= 5:
                    return x, y
        except Exception:
            pass

    # Regex fallback
    with open(path, "rb") as fb:
        txt = fb.read().decode("latin1", errors="ignore")
    xs: list[float] = []
    ys: list[float] = []
    for line in txt.splitlines():
        nums = _FLOAT_RE.findall(line)
        if len(nums) >= 2:
            try:
                xs.append(float(nums[0]))
                ys.append(float(nums[1]))
            except Exception:
                continue
    if len(xs) < 5:
        raise ValueError(f"Could not parse 2-column text spectrum: {os.path.basename(path)}")
    return np.asarray(xs, float), np.asarray(ys, float)


def load_spectrum(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load spectrum from either binary SPE (default) or 2-column text."""
    if _looks_binary_spe(path):
        return _load_binary_spe_princeton(path)
    return _load_text_two_col(path)


# =========================
# Plot helpers
# =========================
def _common_x_from_first(groups: dict[float, list[tuple[float, str]]]) -> np.ndarray:
    first_pol = next(iter(groups.keys()))
    x0, _ = load_spectrum(groups[first_pol][0][1])
    return np.asarray(x0, float).copy()


def _build_pol_B_map(groups: dict[float, list[tuple[float, str]]]) -> dict[float, dict[float, str]]:
    out: dict[float, dict[float, str]] = {}
    for pol, items in groups.items():
        d: dict[float, str] = {}
        for B, f in items:
            d[float(B)] = f
        out[pol] = d
    return out


def _build_map_matrix(items: list[tuple[float, str]], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build map matrix aligned to X.
    Returns:
      B_arr: (nB,)
      I: (nB, nX)
    """
    B_list: list[float] = []
    rows: list[np.ndarray] = []
    for B, f in items:
        x, y = load_spectrum(f)
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        if x.shape != X.shape or np.nanmax(np.abs(x - X)) > STACK_COMMON_X_INTERP_TOL:
            y = np.interp(X, x, y)

        B_list.append(float(B))
        rows.append(y.astype(float))
    return np.asarray(B_list, float), np.vstack(rows)


# =========================
# Plots
# =========================
def plot_stacked_overlay_by_pol(
    groups: dict[float, list[tuple[float, str]]],
    offset_step: float = 0.2,
    *,
    v_label: str = ""
):
    """
    One-panel overlay stacked lines:
      - offset determined by global B index (union of all B values)
      - different pol -> different linestyle/color
    """
    pols = list(groups.keys())
    if not pols:
        raise ValueError("No groups to plot")

    X = _common_x_from_first(groups)
    all_B = sorted({float(B) for items in groups.values() for (B, _f) in items})
    pol_B_map = _build_pol_B_map(groups)

    fig, ax = plt.subplots(figsize=(8.6, 5.8))

    linestyles = ["-", "--", ":", "-."]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3"])

    for p_i, pol in enumerate(pols):
        ls = linestyles[p_i % len(linestyles)]
        col = colors[p_i % len(colors)]
        ax.plot([], [], linestyle=ls, color=col, label=f"pol {pol:g}°")

        for k, B in enumerate(all_B):
            f = pol_B_map[pol].get(B)
            if f is None:
                continue

            x, y = load_spectrum(f)
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            if x.shape != X.shape or np.nanmax(np.abs(x - X)) > STACK_COMMON_X_INTERP_TOL:
                y = np.interp(X, x, y)

            y_plot = y + k * offset_step
            ax.plot(X, y_plot, lw=1.0, linestyle=ls, color=col, alpha=0.9)

            if STACK_SHOW_B_LABEL_EVERY and (k % int(STACK_SHOW_B_LABEL_EVERY) == 0) and p_i == 0:
                ax.text(float(X.max()), float(y_plot[-1]), f" {B:g}T", va="center", fontsize=8, color="#444")

    ax.set_title(f"Overlay stacked spectra by polarization ({v_label})")
    ax.set_xlabel("Energy (eV) / Wavelength (nm) / Pixel index")
    ax.set_ylabel("Intensity (offset by B index)")
    ax.legend(title="Polarization", fontsize=9)
    fig.tight_layout()
    return fig


def plot_map_by_pol(groups: dict[float, list[tuple[float, str]]], cmap: str = "inferno", *, v_label: str = ""):
    """
    Separate panel per pol: (x, B) map using pcolormesh (handles non-uniform B spacing).
    """
    n = len(groups)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(7.2, max(3.2, 2.8 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    X = _common_x_from_first(groups)

    last_mappable = None
    for ax, (pol, items) in zip(axes, groups.items()):
        B_arr, I = _build_map_matrix(items, X)

        # pcolormesh expects coordinate vectors; use X and B_arr directly
        last_mappable = ax.pcolormesh(X, B_arr, I, shading="auto", cmap=cmap)
        ax.set_title(f"pol = {pol:g}°  ({v_label})")
        ax.set_ylabel("B (T)")

    axes[-1].set_xlabel("Energy (eV) / Wavelength (nm) / Pixel index")
    if last_mappable is not None:
        fig.colorbar(last_mappable, ax=axes, label="Intensity", shrink=0.92)
    fig.tight_layout()
    return fig


def plot_map_overlay_by_pol(groups: dict[float, list[tuple[float, str]]], *, v_label: str = ""):
    """
    One-panel overlay map:
      - Draw each pol as a semi-transparent pcolormesh with its own colormap.
      - Mask low-intensity background so pol colors remain distinguishable.
      - vmin/vmax computed PER polarization (robust percentiles).
    """
    pols = list(groups.keys())
    if not pols:
        raise ValueError("No groups to plot")

    X = _common_x_from_first(groups)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    cmaps = list(MAP_OVERLAY_CMAPS) if MAP_OVERLAY_CMAPS else ["viridis"]
    p_lo, p_hi = MAP_SCALE_PERCENTILES

    # overlay each pol
    for i, pol in enumerate(pols):
        B_arr, I = _build_map_matrix(groups[pol], X)

        # robust scaling per pol
        vmin = float(np.nanpercentile(I, p_lo))
        vmax = float(np.nanpercentile(I, p_hi))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = float(np.nanmin(I))
            vmax = float(np.nanmax(I))

        # mask low intensity -> transparent background (key for color separation)
        thr = float(np.nanpercentile(I, MAP_OVERLAY_MASK_BELOW_PERCENTILE))
        Im = np.ma.masked_less(I, thr)

        cmap_name = cmaps[i % len(cmaps)]
        # make masked values transparent
        cmap = mpl.colormaps.get_cmap(cmap_name).copy()
        cmap.set_bad((0, 0, 0, 0))

        ax.pcolormesh(
            X,
            B_arr,
            Im,
            shading="auto",
            cmap=cmap,
            alpha=MAP_OVERLAY_ALPHA,
            vmin=vmin,
            vmax=vmax,
        )

    ax.set_title(f"Overlay B-field map by polarization ({v_label})")
    ax.set_xlabel("Energy (eV) / Wavelength (nm) / Pixel index")
    ax.set_ylabel("B (T)")

    # legend: show which cmap is used for each pol
    handles: list[plt.Line2D] = []
    labels: list[str] = []
    for i, pol in enumerate(pols):
        cmap_name = cmaps[i % len(cmaps)]
        # take a representative strong color from the cmap for the legend line
        c = mpl.colormaps.get_cmap(cmap_name)(0.85)
        handles.append(plt.Line2D([0], [0], color=c, lw=6))
        labels.append(f"pol {pol:g}° ({cmap_name})")
    ax.legend(handles, labels, title="Overlay", fontsize=9, loc="upper right")

    fig.tight_layout()
    return fig


# =========================
# Compare spectra at fixed B (pol overlay)
# =========================
COMPARE_BY_B = True
COMPARE_B_VALUES = [-4,-1, 0, 1, 3, 5]  # 원하는 B들로 바꾸세요 (예: [0, 1] 또는 [0, 0.5, 1, 1.5, ...])
B_MATCH_TOL = 1e-6  # B 매칭 허용 오차 (파일에 0.999999 같이 저장된 경우 대비)
COMPARE_NORMALIZE = False  # True면 각 곡선을 자기 max로 나눠 shape 비교하기 좋게
COMPARE_SUBPLOT_COLS = 3
COMPARE_YLABEL = "Intensity"  # 또는 "Normalized intensity"


def _find_file_at_B(items: list[tuple[float, str]], B_target: float, tol: float) -> str | None:
    """
    items: [(B, filepath), ...] sorted by B
    returns filepath whose B is closest to B_target within tol, else None
    """
    Bs = np.asarray([b for b, _ in items], dtype=float)
    if Bs.size == 0:
        return None
    idx = int(np.argmin(np.abs(Bs - float(B_target))))
    if abs(float(Bs[idx]) - float(B_target)) <= float(tol):
        return items[idx][1]
    return None


def plot_compare_fixed_B_by_pol(
    groups: dict[float, list[tuple[float, str]]],
    B_values: list[float],
    *,
    tol: float = 1e-6,
    normalize: bool = False,
    ncols: int = 3,
    v_label: str = ""
):
    """
    For each B in B_values, make one subplot and overlay spectra of different pols.
    """
    pols = list(groups.keys())
    if len(pols) < 2:
        raise ValueError("Need at least 2 polarizations to compare at fixed B.")

    X = _common_x_from_first(groups)

    B_values = [float(b) for b in B_values]
    n = len(B_values)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8 * ncols, 3.3 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2", "C3", "C4"])
    linestyles = ["-", "--", ":", "-."]

    for i, B0 in enumerate(B_values):
        ax = axes[i]
        any_plotted = False

        for p_i, pol in enumerate(pols):
            f = _find_file_at_B(groups[pol], B0, tol)
            if f is None:
                continue

            _v, _B, _pol, texp = parse_filename(f)  # NEW: read exposure from filename

            x, y = load_spectrum(f)
            x = np.asarray(x, float)
            y = np.asarray(y, float)

            if x.shape != X.shape or np.nanmax(np.abs(x - X)) > STACK_COMMON_X_INTERP_TOL:
                y = np.interp(X, x, y)

            if normalize:
                denom = float(np.nanmax(y)) if np.isfinite(np.nanmax(y)) else 1.0
                if denom != 0:
                    y = y / denom

            ax.plot(
                X,
                y,
                color=colors[p_i % len(colors)],
                linestyle=linestyles[p_i % len(linestyles)],
                lw=1.4,
                label=f"pol {pol:g}° ({texp}s)",  # CHANGED: show 10s/30s etc
            )
            any_plotted = True

        ax.set_title(f"B = {B0:g} T")
        ax.grid(True, alpha=0.2)

        if not any_plotted:
            ax.text(0.5, 0.5, "No matching files", transform=ax.transAxes, ha="center", va="center", color="#777")

        if i % ncols == 0:
            ax.set_ylabel(COMPARE_YLABEL if not normalize else "Normalized intensity")

    # hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    axes[min(n - 1, len(axes) - 1)].set_xlabel("Energy (eV) / Wavelength (nm) / Pixel index")

    # one shared legend (use first non-empty axis)
    for ax in axes[:n]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", title="Polarization")
            break

    fig.suptitle(f"Fixed-B comparison across polarization ({v_label})", y=1.02)
    fig.tight_layout()
    return fig


# =========================
# Main (loop over v when TARGET_V=None)
# =========================
# Map display toggles (NEW)
SHOW_MAP_SEPARATE_BY_POL = True   # pol별로 각각(map을 subplot으로) 보기
SHOW_MAP_OVERLAY_BY_POL = True    # pol들을 한 패널에 overlay로 보기

def main():
    print("RUNNING FILE:", __file__)
    if USER_SELECT_FILES:
        picked = pick_spe_files_dialog(initial_dir=DATA_DIR)
        if not picked:
            print(
                "No .spe files selected (cancelled/interrupted). "
                f"Falling back to scanning DATA_DIR={os.path.abspath(DATA_DIR)}"
            )
            selected = collect_files_in_dir(DATA_DIR, TARGET_V, TARGET_POL, POL_TOL)
        else:
            selected = collect_from_paths(picked, TARGET_V, TARGET_POL, POL_TOL)
    else:
        selected = collect_files_in_dir(DATA_DIR, TARGET_V, TARGET_POL, POL_TOL)

    # NEW: if TARGET_V is None -> show all v values
    if TARGET_V is None:
        v_groups = group_by_v_then_pol(selected)
        for vval, pol_groups in v_groups.items():
            v_label = f"v={vval:g}"

            if COMPARE_BY_B and len(pol_groups) >= 2:
                plot_compare_fixed_B_by_pol(
                    pol_groups,
                    COMPARE_B_VALUES,
                    tol=B_MATCH_TOL,
                    normalize=COMPARE_NORMALIZE,
                    ncols=COMPARE_SUBPLOT_COLS,
                    v_label=v_label,
                )

            if STACK_OVERLAY_ONE_PANEL and len(pol_groups) >= 1:
                plot_stacked_overlay_by_pol(pol_groups, offset_step=STACKED_OFFSET_STEP, v_label=v_label)

            # CHANGED: show BOTH map styles (if enabled)
            if SHOW_MAP_SEPARATE_BY_POL:
                plot_map_by_pol(pol_groups, cmap=MAP_CMAP, v_label=v_label)

            if SHOW_MAP_OVERLAY_BY_POL and len(pol_groups) >= 2:
                plot_map_overlay_by_pol(pol_groups, v_label=v_label)

        plt.show()
        return

    # 기존: 특정 v만
    groups = group_by_polarization(selected)

    if COMPARE_BY_B and len(groups) >= 2:
        plot_compare_fixed_B_by_pol(
            groups,
            COMPARE_B_VALUES,
            tol=B_MATCH_TOL,
            normalize=COMPARE_NORMALIZE,
            ncols=COMPARE_SUBPLOT_COLS,
            v_label=f"v={TARGET_V:g}",
        )

    plot_stacked_overlay_by_pol(groups, offset_step=STACKED_OFFSET_STEP, v_label=f"v={TARGET_V:g}")

    # CHANGED: show BOTH map styles (if enabled)
    if SHOW_MAP_SEPARATE_BY_POL:
        plot_map_by_pol(groups, cmap=MAP_CMAP, v_label=f"v={TARGET_V:g}")

    if SHOW_MAP_OVERLAY_BY_POL and len(groups) >= 2:
        plot_map_overlay_by_pol(groups, v_label=f"v={TARGET_V:g}")

    plt.show()


if __name__ == "__main__":
    main()