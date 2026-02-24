import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import os
import tkinter as tk
from tkinter import filedialog
from scipy.ndimage import median_filter

HC_EV_NM = 1239.841984  # eV*nm


# ----------------------------
# 1) Parse filename metadata
# ----------------------------
# Example: v0_-0.5T_117.5_20.spe
FNAME_RE = re.compile(
    r'^(v(?P<fill>[-\d\.]+))_(?P<B>[-\d\.]+)T_(?P<ang>[-\d\.]+)_(?P<exp>[-\d\.]+)\.spe$'
)

def parse_meta(path: Path) -> dict:
    m = FNAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Filename does not match pattern: {path.name}")
    d = m.groupdict()
    return {
        "filling": d["fill"],                 # string like "0" or "-1.5"
        "B_T": float(d["B"]),                 # Tesla
        "angle_deg": float(d["ang"]),         # polarization angle
        "exp_s": float(d["exp"]),             # exposure seconds
    }


# ----------------------------
# 2) Read SPE 3.0 (Princeton Instruments)
#    - Header is 4100 bytes
#    - Data is uint16 in your files
#    - Wavelength axis is stored as comma-separated list in XML footer
# ----------------------------
def _u16(f, off):
    f.seek(off)
    return int.from_bytes(f.read(2), byteorder="little", signed=False)

def _u32(f, off):
    f.seek(off)
    return int.from_bytes(f.read(4), byteorder="little", signed=False)

def read_spe3(path: Path):
    with open(path, "rb") as f:
        xdim = _u16(f, 42)
        ydim = _u16(f, 656)
        nframes = _u32(f, 1446)
        dtype_code = _u16(f, 108)

        # Your uploaded files are dtype_code=3 and read correctly as uint16.
        # If your future files differ, extend this mapping.
        if dtype_code != 3:
            raise ValueError(f"Unexpected dtype_code={dtype_code} in {path.name} (expected 3 for uint16).")

        bpp = 2  # uint16
        header_len = 4100
        data_len = xdim * ydim * nframes * bpp

        # Read data
        f.seek(header_len)
        raw = f.read(xdim * ydim * nframes * bpp)
        y = np.frombuffer(raw, dtype=np.uint16).astype(float)

        # Read XML footer and extract wavelength mapping
        f.seek(header_len + data_len)
        footer = f.read()

    root = ET.fromstring(footer)
    ns = {"spe": "http://www.princetoninstruments.com/spe/2009"}
    wm = root.find(".//spe:WavelengthMapping", ns)
    if wm is None:
        raise RuntimeError(f"No WavelengthMapping found in {path.name} (XML footer).")

    wl_el = wm.find("spe:Wavelength", ns)
    if wl_el is None or wl_el.text is None:
        raise RuntimeError(f"No wavelength list found in {path.name} (XML footer).")

    # comma/space separated wavelength list
    wl_nm = np.array([float(v) for v in re.split(r"[\s,]+", wl_el.text.strip()) if v], dtype=float)
    if wl_nm.size != xdim:
        raise RuntimeError(f"Wavelength axis length ({wl_nm.size}) != xdim ({xdim}) in {path.name}.")

    # Convert wavelength -> energy (eV)
    E_eV = HC_EV_NM / wl_nm

    return E_eV, y
# ----------------------------
# 2.5) Cosmic ray removal
# ----------------------------
def remove_cosmic_rays_robust_1d(intensity, threshold=2.0, window=11, return_mask=False):
    """
    Robust 1D cosmic-ray removal: 연속된 이상치 구간도 mask 후 정상 구간으로 선형 보간.
    """
    x = intensity.astype(float).copy()
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1

    local_med = median_filter(x, size=window, mode='reflect')
    local_mad = median_filter(np.abs(x - local_med), size=window, mode='reflect')
    sigma = 1.4826 * local_mad
    if np.all(sigma == 0):
        sigma[:] = np.std(x) if np.std(x) > 0 else 1.0
    else:
        sigma[sigma == 0] = np.median(sigma[sigma > 0])

    dev = x - local_med
    mask = np.abs(dev) > (threshold * sigma)
    out = x.copy()
    if np.any(mask):
        idx = np.arange(out.size)
        good = ~mask
        # 연속된 mask 구간도 정상 구간 기준으로 선형 보간
        if good.sum() < 2:
            out[mask] = local_med[mask]
        else:
            out[mask] = np.interp(idx[mask], idx[good], out[good])
    if return_mask:
        return out, mask
    return out


# ----------------------------
# 3) Helper: pick files by 원하는 조건
# ----------------------------
def collect_files(folder: Path, filling: str, B_list, angle_list, exp_s=None, tol_B=1e-6, tol_ang=1e-6):
    """
    folder: directory containing .spe files
    filling: e.g. "0" for v0
    B_list: list of Tesla values to plot (exact match within tol_B)
    angle_list: list of angles to overlay (exact match within tol_ang)
    exp_s: if not None, only select this exposure time (exact match)
    """
    folder = Path(folder)
    all_files = sorted(folder.glob("v*_*.spe"))

    wanted = {}
    for fp in all_files:
        try:
            meta = parse_meta(fp)
        except ValueError:
            continue

        if meta["filling"] != filling:
            continue

        if exp_s is not None and abs(meta["exp_s"] - exp_s) > 1e-9:
            continue

        # match desired B and angle
        for B in B_list:
            if abs(meta["B_T"] - B) < tol_B:
                for ang in angle_list:
                    if abs(meta["angle_deg"] - ang) < tol_ang:
                        wanted[(B, ang)] = fp

    return wanted


# ----------------------------
# 4) Plot: stacked by B, overlay by angle
# ----------------------------
def plot_stacked_by_B(file_map, B_list, angle_list, xlim=None, ylim=None, normalize=True, title_prefix="", background=0, cosmic_threshold=5, cosmic_window=5):
    """
    file_map: dict {(B, ang): Path}
    ylim: (ymin, ymax) tuple or None
    cosmic_threshold: cosmic ray detection threshold (sigma units)
    cosmic_window: median filter window (odd integer)
    """
    n = len(B_list)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.1*n), sharex=True)

    if n == 1:
        axes = [axes]

    for i, B in enumerate(B_list):
        ax = axes[i]
        for ang in angle_list:
            key = (B, ang)
            if key not in file_map:
                continue
            E, I = read_spe3(file_map[key])
            # cosmic ray 제거 및 검출 개수 출력
            I_cleaned, mask = remove_cosmic_rays_robust_1d(I, threshold=cosmic_threshold, window=cosmic_window, return_mask=True)
            n_spikes = int(np.sum(mask))
            print(f"B={B}, angle={ang}: cosmic spikes detected={n_spikes} (threshold={cosmic_threshold}, window={cosmic_window})")
            I = I_cleaned

            # --- background subtraction ---
            if background != 0:
                I = I - background
                I[I < 0] = 0  # 음수 방지

            # --- normalization ---
            if normalize:
                if xlim is not None:
                    m = (E >= xlim[0]) & (E <= xlim[1])
                    denom = np.max(I[m]) if np.any(m) else np.max(I)
                else:
                    denom = np.max(I)
                if denom > 0:
                    I = I / denom

            # legend 이름 지정
            if ang == 95:
                legend_name = "σ+"
            elif ang == 140:
                legend_name = "σ-"
            else:
                legend_name = f"{ang}°"
            ax.plot(E, I, linewidth=1.5, label=legend_name)

        ax.set_ylabel("Norm. Intensity" if normalize else "Intensity (a.u.)")
        ax.set_title(f"Target B = {B:g} T")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.25)
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[-1].set_xlabel("Energy (eV)")
    if xlim is not None:
        axes[-1].set_xlim(xlim)

    fig.suptitle(title_prefix, y=0.995)
    try:
        fig.tight_layout()
    except Exception:
        fig.subplots_adjust(top=0.92, bottom=0.07, left=0.08, right=0.98, hspace=0.35)
    plt.show()


# ----------------------------
# 5) Example usage
# ----------------------------
def pick_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="폴더를 선택하세요 (.spe 파일 포함)")
    root.destroy()
    if not folder:
        raise SystemExit("폴더를 선택하지 않았습니다.")
    return Path(folder)

if __name__ == "__main__":
    # 폴더 선택
    data_dir = pick_folder()

    filling = "0"                       # v0
    B_list = [-6, -3, 0, 3, 6]          # 보고 싶은 field만
    angle_list = [95, 140]              # 보고 싶은 angle만
    exp_s = 40                          # (선택) exposure 필터. 필요 없으면 None

    # cosmic ray 제거 threshold/window 직접 설정
    cosmic_threshold = 10                # 원하는 값으로 조정 (낮추면 더 민감)
    cosmic_window = 5                 # 홀수(3,5,7 등)

    files = collect_files(data_dir, filling=filling, B_list=B_list, angle_list=angle_list, exp_s=exp_s)

    plot_stacked_by_B(
        files,
        B_list=B_list,
        angle_list=angle_list,
        xlim=(1.43, 1.47),
        ylim=(0, 1.05),
        normalize=True,
        title_prefix=f"v{filling}, exp={exp_s}s (LP/CP angle overlay)",
        background=600,
        cosmic_threshold=cosmic_threshold,
        cosmic_window=cosmic_window
    )

# 각도 추출용 정규식 (파일명 예시: v0_gain4_0.48_pi_pi_60.spe)
ANGLE_RE = re.compile(r'_(\d+\.?\d*)\.spe$')

def get_angle_from_filename(filename):
    m = ANGLE_RE.search(filename)
    if m:
        return float(m.group(1))
    return None

def main():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select .spe files",
        filetypes=[("SPE files", "*.spe")]
    )
    plt.figure()
    background = 600
    for idx, file_path in enumerate(file_paths):
        E, I = read_spe3(Path(file_path))
        I = I - background
        I[I < 0] = 0
        if idx == 0:
            I = I * 2
        angle = get_angle_from_filename(Path(file_path).name)
        if angle == 95:
            legend_name = "σ+"
        elif angle == 140:
            legend_name = "σ-"
        else:
            legend_name = Path(file_path).name
        plt.plot(E, I, label=legend_name)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.title("Energy vs. Intensity (Overlayed, background=600)")
    plt.show()

if __name__ == "__main__":
    main()
