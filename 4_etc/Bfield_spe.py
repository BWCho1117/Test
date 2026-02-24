import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import os
import sys
import tkinter as tk
from tkinter import filedialog
import struct  # ADD

# =========================
# User settings
# =========================
USER_SELECT_FILES = True

DATA_DIR = "."             # (자동 수집 모드일 때만 사용)
TARGET_V = 0               # filling factor v0

# 폴라 선택:
# - None: 선택된 파일들에서 "각 폴라 angle별"로 패널을 나눠서 모두 그림 (요청하신 동작)
# - float: 해당 폴라만 필터링해서 그림
TARGET_POL = None          # e.g. 117.5 or None(=all)
POL_TOL = 1e-3             # pol 비교 tolerance (e.g. 1e-3 ~ 1e-1)

# Visualization options
STACKED_OFFSET_STEP = 0.2
MAP_CMAP = "inferno"


# =========================
# 0) Pick .spe files (dialog)
# =========================
def pick_spe_files_dialog(initial_dir: str | None = None) -> list[str]:
    """
    Open file dialog to select one or many .spe files.
    Returns list of file paths.
    If cancelled/interrupted (Ctrl+C), returns [].
    """
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
# 1) Parse filename: v0_-0.5T_117.5.spe
# =========================
def parse_filename(path: str) -> tuple[int, float, float]:
    """
    Parse filename like: v0_-0.5T_117.5.spe
    Returns: (v:int, B:float in Tesla, pol:float in degrees)
    """
    fname = os.path.basename(path)
    pat = r"^v(?P<v>-?\d+)_?(?P<B>-?\d+(?:\.\d+)?)T_(?P<pol>\d+(?:\.\d+)?)\.spe$"
    m = re.match(pat, fname)
    if not m:
        raise ValueError(f"Filename format not recognized: {fname}")
    v = int(m.group("v"))
    B = float(m.group("B"))
    pol = float(m.group("pol"))
    return v, B, pol


# =========================
# 2) Load spectrum (robust for encoding/header)
# =========================
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _looks_binary_spe(path: str, sniff_bytes: int = 2048) -> bool:
    with open(path, "rb") as fb:
        chunk = fb.read(sniff_bytes)
    # Princeton Instruments SPE (binary) often contains lots of NULs early
    return (b"\x00" in chunk) and (chunk.count(b"\x00") > 10)

def _load_binary_spe_princeton(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimal Princeton Instruments SPE (binary) reader.
    Assumes classic 4100-byte header (WinSpec SPE v2/v3 family).
    Returns (x, y) where x is pixel index (wavelength axis usually not stored).
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

        # Common WinSpec offsets (widely used)
        xdim = u16(42)          # pixels in x
        ydim = u16(656)         # pixels in y
        dtype_code = u16(108)   # data type code
        nframes = i32(1446)     # number of frames

        if xdim <= 0 or ydim <= 0:
            raise ValueError(f"Invalid SPE dimensions parsed: xdim={xdim}, ydim={ydim} for {os.path.basename(path)}")
        if nframes <= 0:
            # some files store 0/1 inconsistently; treat as single frame
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
            raise ValueError(
                f"Unsupported SPE dtype code={dtype_code} for {os.path.basename(path)}. "
                "Export to ASCII/CSV or provide the format spec."
            )

        dtype = code_to_dtype[dtype_code]
        bytes_per = np.dtype(dtype).itemsize
        expected = int(xdim) * int(ydim) * int(nframes) * bytes_per

        # Read the data payload
        f.seek(0, os.SEEK_END)
        fsize = f.tell()
        if fsize < HEADER_BYTES + expected:
            # still try reading what exists (some SPE variants have different frame count metadata)
            f.seek(HEADER_BYTES, os.SEEK_SET)
            raw = np.fromfile(f, dtype=dtype)
        else:
            f.seek(HEADER_BYTES, os.SEEK_SET)
            raw = np.fromfile(f, dtype=dtype, count=int(xdim) * int(ydim) * int(nframes))

    if raw.size < xdim * ydim:
        raise ValueError(f"Failed to read SPE payload: {os.path.basename(path)} (read {raw.size} points)")

    # reshape as (frames, y, x) when possible
    if raw.size == xdim * ydim * nframes:
        cube = raw.reshape((nframes, ydim, xdim))
        img = cube.mean(axis=0)  # average frames
    else:
        # fallback: assume single frame
        img = raw[: xdim * ydim].reshape((ydim, xdim))

    # Convert 2D to 1D spectrum: sum along y (rows)
    y = img.sum(axis=0).astype(float)
    x = np.arange(y.size, dtype=float)
    return x, y


def load_spectrum(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum from:
      - text export (2 columns)
      - Princeton Instruments binary .SPE (minimal reader)
    """
    if _looks_binary_spe(path):
        return _load_binary_spe_princeton(path)

    # 1) try genfromtxt with tolerant settings
    for enc in ("utf-8", "cp949", "latin1"):
        try:
            arr = np.genfromtxt(
                path,
                dtype=float,
                comments="#",
                delimiter=None,
                invalid_raise=False,
                encoding=enc,
            )
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 1 and arr.size >= 2:
                # single row
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

    # 2) regex-based fallback: pick first two floats per line
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
        raise ValueError(f"Could not parse numeric 2-column data from: {os.path.basename(path)}")

    return np.asarray(xs, float), np.asarray(ys, float)


# =========================
# 3) Collect files
# =========================
def _pol_match(pol: float, target_pol: float | None, tol: float) -> bool:
    if target_pol is None:
        return True
    return abs(pol - float(target_pol)) <= float(tol)

def collect_files(data_dir: str, target_v: int, target_pol: float | None, pol_tol: float) -> list[tuple[float, str]]:
    files = glob.glob(os.path.join(data_dir, "*.spe"))
    selected: list[tuple[float, str]] = []
    for f in files:
        v, B, pol = parse_filename(f)
        if v == target_v and _pol_match(pol, target_pol, pol_tol):
            selected.append((B, f))
    selected.sort(key=lambda t: t[0])
    if len(selected) == 0:
        raise RuntimeError(
            f"No files found for v={target_v}, pol={target_pol}±{pol_tol} in {data_dir}"
        )
    return selected

def collect_from_picked_files(
    file_paths: list[str],
    target_v: int,
    target_pol: float | None,
    pol_tol: float
) -> list[tuple[float, str]]:
    selected: list[tuple[float, str]] = []
    for f in file_paths:
        v, B, pol = parse_filename(f)
        if v == target_v and _pol_match(pol, target_pol, pol_tol):
            selected.append((B, f))

    selected.sort(key=lambda t: t[0])
    if len(selected) == 0:
        raise RuntimeError(
            "No selected files matched the filter:\n"
            f"  v={target_v}, pol={target_pol}±{pol_tol}\n"
            "Tip: set TARGET_POL=None to include all polarizations."
        )
    return selected

def group_by_polarization(selected: list[tuple[float, str]], *, pol_round: int = 3) -> dict[float, list[tuple[float, str]]]:
    """
    Group (B, path) by polarization angle parsed from filename.
    Key is rounded pol for stable grouping.
    """
    groups: dict[float, list[tuple[float, str]]] = {}
    for B, f in selected:
        _v, _B, pol = parse_filename(f)
        key = round(float(pol), pol_round)
        groups.setdefault(key, []).append((float(B), f))

    # sort inside each group by B
    for k in list(groups.keys()):
        groups[k].sort(key=lambda t: t[0])
    return dict(sorted(groups.items(), key=lambda kv: kv[0]))


# =========================
# 4A) Stacked plot (multi-panel by pol)
# =========================
def plot_stacked_by_pol(groups: dict[float, list[tuple[float, str]]], offset_step: float = 0.2):
    n = len(groups)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(7.2, max(3.2, 2.8 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (pol, items) in zip(axes, groups.items()):
        offset = 0.0
        for B, f in items:
            x, y = load_spectrum(f)
            ax.plot(x, y + offset, lw=1.0, label=f"{B:g} T")
            offset += offset_step

        ax.set_title(f"pol = {pol:g}°  (v{TARGET_V})")
        ax.set_ylabel("Intensity (offset)")
        ax.legend(fontsize=8, ncol=2)

    axes[-1].set_xlabel("Energy (eV) or Wavelength (nm)")
    fig.tight_layout()
    plt.show()


# =========================
# 4B) Map plot (multi-panel by pol)
# =========================
def plot_map_by_pol(groups: dict[float, list[tuple[float, str]]], cmap: str = "inferno"):
    n = len(groups)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(7.2, max(3.2, 2.8 * n)), sharex=True)
    if n == 1:
        axes = [axes]

    # common X grid: use first spectrum of first group
    first_pol = next(iter(groups.keys()))
    x0, _ = load_spectrum(groups[first_pol][0][1])
    X = x0.copy()

    last_im = None
    for ax, (pol, items) in zip(axes, groups.items()):
        B_list: list[float] = []
        Y_list: list[np.ndarray] = []

        for B, f in items:
            x, y = load_spectrum(f)
            if len(x) != len(X) or np.max(np.abs(x - X)) > 1e-12:
                y = np.interp(X, x, y)
            B_list.append(float(B))
            Y_list.append(y.astype(float))

        I = np.vstack(Y_list)  # (nB, nX)
        last_im = ax.imshow(
            I,
            aspect="auto",
            origin="lower",
            extent=[float(X.min()), float(X.max()), float(min(B_list)), float(max(B_list))],
            cmap=cmap,
        )
        ax.set_title(f"pol = {pol:g}°  (v{TARGET_V})")
        ax.set_ylabel("B (T)")

    axes[-1].set_xlabel("Energy (eV) or Wavelength (nm)")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes, label="Intensity", shrink=0.92)
    fig.tight_layout()
    plt.show()


# =========================
# Main
# =========================
if __name__ == "__main__":
    if USER_SELECT_FILES:
        picked = pick_spe_files_dialog(initial_dir=DATA_DIR)
        if not picked:
            print(
                "No .spe files selected (cancelled/interrupted). "
                f"Falling back to scanning DATA_DIR={os.path.abspath(DATA_DIR)}"
            )
            selected = collect_files(DATA_DIR, TARGET_V, TARGET_POL, POL_TOL)
        else:
            selected = collect_from_picked_files(picked, TARGET_V, TARGET_POL, POL_TOL)
    else:
        selected = collect_files(DATA_DIR, TARGET_V, TARGET_POL, POL_TOL)

    groups = group_by_polarization(selected)

    # 폴라가 2개면 패널 2개 (요청사항)
    plot_stacked_by_pol(groups, offset_step=STACKED_OFFSET_STEP)
    plot_map_by_pol(groups, cmap=MAP_CMAP)