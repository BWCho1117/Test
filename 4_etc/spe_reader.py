import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from pathlib import Path
import re

HC_EV_NM = 1239.841984  # eV*nm

def read_spe3(path: Path):
    with open(path, "rb") as f:
        xdim = int.from_bytes(f.read()[42:44], "little")
        f.seek(656)
        ydim = int.from_bytes(f.read(2), "little")
        f.seek(1446)
        nframes = int.from_bytes(f.read(4), "little")
        f.seek(108)
        dtype_code = int.from_bytes(f.read(2), "little")
        if dtype_code != 3:
            raise ValueError(f"Unexpected dtype_code={dtype_code} in {path.name} (expected 3 for uint16).")
        bpp = 2
        header_len = 4100
        data_len = xdim * ydim * nframes * bpp
        f.seek(header_len)
        raw = f.read(xdim * ydim * nframes * bpp)
        y = np.frombuffer(raw, dtype=np.uint16).astype(float)
        f.seek(header_len + data_len)
        footer = f.read()
    root = ET.fromstring(footer)
    ns = {"spe": "http://www.princetoninstruments.com/spe/2009"}
    wm = root.find(".//spe:WavelengthMapping", ns)
    wl_el = wm.find("spe:Wavelength", ns)
    wl_nm = np.array([float(v) for v in re.split(r"[\s,]+", wl_el.text.strip()) if v], dtype=float)
    E_eV = HC_EV_NM / wl_nm
    return E_eV, y

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
        if idx == 0:  # 첫 번째 파일만 intensity 2배
            I = I * 1
        plt.plot(E, I, label=Path(file_path).name + (" (x2)" if idx == 0 else ""))
    plt.xlabel("Energy (eV)")
    plt.ylabel("Intensity (a.u.)")
    plt.legend()
    plt.title("Energy vs. Intensity (Overlayed, background=600, first x2)")
    plt.show()

if __name__ == "__main__":
    main()