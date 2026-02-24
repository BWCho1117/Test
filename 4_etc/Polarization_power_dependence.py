import os
import re
import glob
import struct
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# User settings
# ============================================================
# X axis from .spe (XML footer, if present)
X_MODE = "spe_energy"          # "pixel" | "spe_wavelength" | "spe_energy"
SORT_X_ASC = True              # energy axis often comes out descending; sort for nicer plots

# Baseline subtraction (pixel indices)
BASELINE_RANGE = None          # e.g. (0, 100) or None

# Normalization master switch (NEW)
NORMALIZATION_ENABLED = False     # True=정규화 적용, False=정규화 안 함(원데이터/배경제거만)

# Normalization
NORMALIZE_MODE = "each"        # "each" (each spectrum peak=1) | "common" (shared scale) | "none"
NORM_METHOD = "max"            # "max" | "area"
NORM_X_RANGE = (1.38, 1.49)    # eV range used for normalization (works when X_MODE="spe_energy")
ROI_PIXEL = None               # fallback ROI in pixel if NORM_X_RANGE=None (e.g. (200,1200))


# ============================================================
# File picker (folder -> pick files) with console fallback
# ============================================================
def _console_select(files):
    if not files:
        return []
    print("\n[.spe files]")
    for i, fp in enumerate(files, 1):
        print(f"{i:3d}: {os.path.basename(fp)}")

    s = input("\nSelect indexes (e.g. 1,3,5-9) or 'all': ").strip().lower()
    if s in ("all", "*", ""):
        return files

    chosen = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            for k in range(min(a, b), max(a, b) + 1):
                if 1 <= k <= len(files):
                    chosen.add(files[k - 1])
        else:
            k = int(part)
            if 1 <= k <= len(files):
                chosen.add(files[k - 1])
    return sorted(chosen)


def select_spe_files(initial_dir=None):
    """
    Pick a folder, then pick .spe files (multi-select).
    If file selection is canceled, defaults to ALL .spe in the folder.
    Falls back to console selection if GUI isn't available.
    """
    initial_dir = initial_dir or os.getcwd()
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        folder = filedialog.askdirectory(
            title="Select folder containing .spe files",
            initialdir=initial_dir,
        )
        if not folder:
            return []

        files = sorted(glob.glob(os.path.join(folder, "*.spe")))
        if not files:
            print(f"[info] No .spe files found in: {folder}")
            return []

        picked = filedialog.askopenfilenames(
            title="Select .spe files (multi-select). Cancel = use ALL in folder",
            initialdir=folder,
            filetypes=[("SPE files", "*.spe"), ("All files", "*.*")],
        )
        return list(picked) if picked else files

    except Exception as e:
        print(f"[warn] GUI picker unavailable ({e}). Using console selection.")
        files = sorted(glob.glob(os.path.join(initial_dir, "*.spe")))
        return _console_select(files)


# ============================================================
# SPE reader (classic 4100-byte header; works for many SPE files)
# ============================================================
def read_spe(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.read(4100)

        # xdim (uint16) at offset 42, ydim (uint16) at offset 656
        xdim = struct.unpack_from("<H", header, 42)[0]
        ydim = struct.unpack_from("<H", header, 656)[0]

        # datatype (int16) at offset 108
        datatype_code = struct.unpack_from("<h", header, 108)[0]
        dt_map = {
            0: np.float32,
            1: np.int32,
            2: np.int16,
            3: np.uint16,
            5: np.float64,
            6: np.uint8,
            8: np.uint32,
        }
        dtype = dt_map.get(datatype_code, np.uint16)

        data = np.fromfile(f, dtype=dtype, count=xdim * ydim).reshape((ydim, xdim))

    return data.squeeze()


# ============================================================
# X-axis extraction from SPE 3.x XML footer (best-effort)
# ============================================================
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _extract_wavelength_from_spe_xml_footer(path: str, expected_len: int | None = None) -> np.ndarray | None:
    """
    Many SPE 3.x files store wavelength mapping in an XML footer.
    This attempts to parse the footer and find a float list matching expected_len.
    Returns wavelength array in nm, or None.
    """
    try:
        import xml.etree.ElementTree as ET

        size = os.path.getsize(path)
        # Read tail (footer typically lives near end). Increase if needed.
        tail_bytes = 4_000_000
        with open(path, "rb") as f:
            f.seek(max(0, size - tail_bytes))
            tail = f.read()

        # Find XML start
        start = tail.rfind(b"<?xml")
        if start < 0:
            start = tail.rfind(b"<SpeFormat")
        if start < 0:
            start = tail.rfind(b"<speformat")
        if start < 0:
            return None

        xml_text = tail[start:].decode("utf-8", errors="ignore").replace("\x00", "").strip()
        if not xml_text:
            return None

        root = ET.fromstring(xml_text)

        candidates: list[np.ndarray] = []

        def maybe_add(text: str):
            nums = _FLOAT_RE.findall(text)
            if len(nums) >= 10:
                arr = np.array([float(v) for v in nums], dtype=float)
                candidates.append(arr)

        for el in root.iter():
            tag = (el.tag or "").lower()

            # prioritize tags that look relevant
            if el.text and any(k in tag for k in ("wave", "wavelength", "calib", "mapping", "xaxis", "x-axis")):
                maybe_add(el.text)

            # also scan attributes
            for _, v in el.attrib.items():
                if isinstance(v, str) and len(v) > 20 and any(k in tag for k in ("wave", "wavelength", "calib", "mapping", "xaxis", "x-axis")):
                    maybe_add(v)

        if not candidates:
            # last resort: scan all text nodes (can be heavier but safer)
            for el in root.iter():
                if el.text and len(el.text) > 50:
                    maybe_add(el.text)

        if not candidates:
            return None

        # Prefer exact length match
        if expected_len is not None:
            for c in candidates:
                if len(c) == expected_len:
                    return c

        # Otherwise pick the longest (often wavelength array)
        return max(candidates, key=len)

    except Exception:
        return None


def spe_x_axis(path: str, x_mode: str, n_pixels: int) -> tuple[np.ndarray, str]:
    x_mode = (x_mode or "pixel").lower()

    if x_mode == "pixel":
        return np.arange(n_pixels), "Pixel"

    wl = _extract_wavelength_from_spe_xml_footer(path, expected_len=n_pixels)
    if wl is None:
        return np.arange(n_pixels), "Pixel (no .spe axis found)"

    if x_mode == "spe_wavelength":
        return wl, "Wavelength (nm)"

    if x_mode == "spe_energy":
        # E(eV) = hc / lambda(nm)
        hc_ev_nm = 1239.841984
        wl_safe = np.where(wl == 0, np.nan, wl)
        E = hc_ev_nm / wl_safe
        return E, "Energy (eV)"

    raise ValueError(f"Unknown X_MODE: {x_mode}")


def _maybe_sort_by_x(x: np.ndarray, y: np.ndarray, sort_asc: bool) -> tuple[np.ndarray, np.ndarray]:
    if not sort_asc:
        return x, y
    idx = np.argsort(x)
    return x[idx], y[idx]


# ============================================================
# Filename parser
# Example: "B=0, gain4, 0.1_72.5_5s.spe"
# -> B, gain, power, pol, exp_s
# ============================================================
def parse_filename(fname: str):
    base = os.path.basename(fname)

    mB = re.search(r"B\s*=\s*([-\d.]+)", base)
    mG = re.search(r"gain\s*([-\d.]+)", base, flags=re.IGNORECASE)
    mRest = re.search(r",\s*([-\d.]+)_([-\d.]+)_([-\d.]+)s\.spe$", base, flags=re.IGNORECASE)

    if not (mB and mG and mRest):
        return None

    B = float(mB.group(1))
    gain = float(mG.group(1))
    power = float(mRest.group(1))
    pol = float(mRest.group(2))
    exp_s = float(mRest.group(3))
    return B, gain, power, pol, exp_s


# ============================================================
# Baseline + normalization
# ============================================================
def subtract_baseline(y: np.ndarray, baseline_range=None) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if baseline_range is None:
        return y
    a, b = baseline_range
    a = max(0, int(a))
    b = min(len(y), int(b))
    if b <= a:
        return y
    return y - float(np.median(y[a:b]))


def _mask_from_xrange(x: np.ndarray, x_range):
    x = np.asarray(x)
    finite = np.isfinite(x)
    if x_range is None:
        return finite
    x0, x1 = x_range
    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
    mask = finite & (x >= lo) & (x <= hi)
    if not np.any(mask):
        raise ValueError(f"x_range={x_range} has no finite points in current x.")
    return mask


def normalize_each_spectrum_in_range(
    spectra: dict[float, np.ndarray],
    x: np.ndarray,
    x_range,
    method: str = "max",
    eps: float = 1e-12,
) -> dict[float, np.ndarray]:
    """
    Each spectrum normalized independently using only the points in x_range.
    method="max": max(y in range) -> 1
    method="area": integral in range -> 1
    """
    sel = _mask_from_xrange(x, x_range)
    out: dict[float, np.ndarray] = {}
    for pol, y in spectra.items():
        y = np.asarray(y, dtype=float)
        if method == "max":
            scale = float(np.nanmax(y[sel]))
        elif method == "area":
            scale = float(np.trapz(y[sel], x[sel]))
        else:
            raise ValueError("NORM_METHOD must be 'max' or 'area'")
        if not np.isfinite(scale) or scale <= eps:
            raise ValueError(f"Bad normalization scale for pol={pol}: {scale}")
        out[pol] = y / scale
    return out


def normalize_common_in_range(
    spectra: dict[float, np.ndarray],
    x: np.ndarray,
    x_range,
    method: str = "max",
    eps: float = 1e-12,
) -> dict[float, np.ndarray]:
    """
    Shared scale across all spectra, computed from values within x_range.
    """
    sel = _mask_from_xrange(x, x_range)
    if method == "max":
        all_vals = np.concatenate([np.asarray(y, float)[sel] for y in spectra.values()])
        scale = float(np.nanmax(all_vals))
    elif method == "area":
        scale = 0.0
        for y in spectra.values():
            y = np.asarray(y, float)
            scale += float(np.trapz(y[sel], x[sel]))
    else:
        raise ValueError("NORM_METHOD must be 'max' or 'area'")

    scale = max(scale, eps)
    return {pol: (np.asarray(y, float) / scale) for pol, y in spectra.items()}


def normalize_by_pixel_roi_common(spectra: dict[float, np.ndarray], roi=None, method="max", eps=1e-12):
    """
    Fallback (pixel ROI): shared scale computed in pixel ROI.
    """
    chunks = []
    for y in spectra.values():
        y = np.asarray(y, float)
        if roi is None:
            chunks.append(y)
        else:
            i0, i1 = roi
            chunks.append(y[int(i0):int(i1)])
    all_vals = np.concatenate(chunks)

    if method == "max":
        scale = float(np.nanmax(all_vals))
    elif method == "area":
        scale = float(np.trapz(all_vals))
    else:
        raise ValueError("NORM_METHOD must be 'max' or 'area'")

    scale = max(scale, eps)
    return {pol: (np.asarray(y, float) / scale) for pol, y in spectra.items()}


# ============================================================
# User settings (ADD)
# ============================================================
# Plot ranges (units follow X_MODE: spe_energy=eV, spe_wavelength=nm, pixel=pixel index)
PLOT_X_RANGE = (1.33, 1.49)     # e.g. (1.35, 1.55) or None
PLOT_Y_RANGE = None     # e.g. (0.0, 1.05) or None


# ============================================================
# Main
# ============================================================
BACKGROUND_LEVEL = 600.0   # 항상 깔린 background counts. 끄려면 None
CLIP_NEGATIVE = True       # background 뺀 뒤 음수는 0으로

def subtract_constant_background(y: np.ndarray, level: float | None, clip_negative: bool = True) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if level is None:
        return y
    y2 = y - float(level)
    return np.clip(y2, 0, None) if clip_negative else y2

def _apply_plot_ranges(ax, x_range=None, y_range=None):
    """Apply user-defined view ranges. None means autoscale."""
    if x_range is not None:
        x0, x1 = x_range
        ax.set_xlim(min(x0, x1), max(x0, x1))
    if y_range is not None:
        y0, y1 = y_range
        ax.set_ylim(min(y0, y1), max(y0, y1))

def main():
    files = select_spe_files(initial_dir=os.getcwd())
    if not files:
        raise SystemExit("No files selected.")

    groups: dict[tuple[float, float, float, float], dict[float, str]] = {}
    for fp in sorted(files):
        parsed = parse_filename(fp)
        if parsed is None:
            print(f"[skip] cannot parse filename: {os.path.basename(fp)}")
            continue
        B, gain, power, pol, exp_s = parsed
        key = (B, gain, power, exp_s)
        groups.setdefault(key, {})
        groups[key][pol] = fp

    if not groups:
        raise SystemExit("No valid files after parsing filenames.")

    for key, pol_map in sorted(groups.items()):
        B, gain, power, exp_s = key

        spectra: dict[float, np.ndarray] = {}
        x_ref = None
        x_label = "Pixel"

        for pol, fp in sorted(pol_map.items()):
            y = read_spe(fp)

            # (1) optional pixel-baseline (쓰고 있으면)
            y = subtract_baseline(y, BASELINE_RANGE)

            # (2) IMPORTANT: subtract constant background (600)
            y = subtract_constant_background(y, BACKGROUND_LEVEL, CLIP_NEGATIVE)

            spectra[pol] = y

            if x_ref is None:
                x_ref, x_label = spe_x_axis(fp, X_MODE, n_pixels=len(y))
                x_ref = np.asarray(x_ref, dtype=float)

        # Optional sort x for nicer energy axis plots (and keep y aligned)
        if SORT_X_ASC and x_ref is not None:
            idx = np.argsort(x_ref)
            x_ref_sorted = x_ref[idx]
            spectra = {pol: np.asarray(y, float)[idx] for pol, y in spectra.items()}
            x_ref = x_ref_sorted

        # ---- Normalize (MODIFIED: apply master on/off) ----
        if (not NORMALIZATION_ENABLED) or (NORMALIZE_MODE.lower() == "none"):
            spectra_plot = spectra
            norm_tag = "no-norm"
        else:
            if NORM_X_RANGE is not None and X_MODE.lower() in ("spe_energy", "spe_wavelength"):
                if NORMALIZE_MODE.lower() == "each":
                    spectra_plot = normalize_each_spectrum_in_range(
                        spectra, x=x_ref, x_range=NORM_X_RANGE, method=NORM_METHOD
                    )
                    norm_tag = f"each-{NORM_METHOD}"
                elif NORMALIZE_MODE.lower() == "common":
                    spectra_plot = normalize_common_in_range(
                        spectra, x=x_ref, x_range=NORM_X_RANGE, method=NORM_METHOD
                    )
                    norm_tag = f"common-{NORM_METHOD}"
                else:
                    raise ValueError("NORMALIZE_MODE must be 'each', 'common', or 'none'")
            else:
                spectra_plot = normalize_by_pixel_roi_common(spectra, roi=ROI_PIXEL, method=NORM_METHOD)
                norm_tag = f"common-{NORM_METHOD}-pixel"

        # Plot
        plt.figure()
        for pol, y in sorted(spectra_plot.items()):
            plt.plot(x_ref if x_ref is not None else np.arange(len(y)), y, label=f"pol={pol:g}°")

        extra = f"x-range={NORM_X_RANGE}" if (NORM_X_RANGE is not None and NORMALIZE_MODE.lower() != "none") else f"roi_pixel={ROI_PIXEL}"
        plt.title(f"B={B:g}T, gain={gain:g}, power={power:g}, exp={exp_s:g}s ({norm_tag}, {extra})")
        plt.xlabel(x_label)
        plt.ylabel("Normalized intensity" if (NORMALIZATION_ENABLED and NORMALIZE_MODE.lower() != "none") else "Intensity (a.u.)")
        plt.legend()

        # NEW: apply view ranges
        ax = plt.gca()
        _apply_plot_ranges(ax, PLOT_X_RANGE, PLOT_Y_RANGE)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
