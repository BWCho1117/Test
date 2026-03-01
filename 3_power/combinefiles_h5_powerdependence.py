import re
import h5py
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


# -----------------------------
# USER SETTINGS: power ranges by tag
# -----------------------------
POWER_RANGES_UW = {
    "10_8": (0.0008, 0.008),  # 0.8 nW to 8 nW  (in uW)
    "10_7": (0.008,  0.08),   # 8 nW to 80 nW
    "10_6": (0.08,   0.8),    # 0.08 uW to 0.8 uW
    "10_5": (0.8,    8.0),    # 0.8 uW to 8 uW
    "10_4": (8.0,  110.0),    # 8 uW to 110 uW
}
TAG_ORDER = ["10_8", "10_7", "10_6", "10_5", "10_4"]  # low -> high

# ✅ Output folder fixed here
DEFAULT_OUT_DIR = Path(r"C:\Respo")


# -----------------------------
# Helpers
# -----------------------------
def extract_tag_from_filename(path: str) -> str | None:
    name = Path(path).name
    m = re.search(r"(10_[0-9]+)", name)
    if not m:
        return None
    tag = m.group(1)
    return tag if tag in POWER_RANGES_UW else None


def sort_files_by_tag(paths: list[str]) -> list[tuple[str, str]]:
    tagged = []
    for p in paths:
        tag = extract_tag_from_filename(p)
        if tag is None:
            raise ValueError(f"Cannot find a valid tag like 10_8..10_4 in filename:\n{p}")
        tagged.append((tag, p))

    tags = [t for t, _ in tagged]
    dup = {t for t in tags if tags.count(t) > 1}
    if dup:
        raise ValueError(
            f"Duplicate tag files selected: {sorted(dup)}\n"
            f"Select at most one file per tag (10_8..10_4)."
        )

    order_index = {t: i for i, t in enumerate(TAG_ORDER)}
    tagged.sort(key=lambda x: order_index[x[0]])
    return tagged


def read_h5_data(path: str):
    with h5py.File(path, "r") as h5:
        data = np.array(h5["/spectro_data"]).squeeze()
        wvl = np.array(h5["/spectro_wavelength"]).squeeze()

    if data.ndim != 2:
        raise ValueError(f"spectro_data must be 2D (n_spectra, n_pixels). Got shape {data.shape} in {path}")
    if wvl.ndim != 1:
        raise ValueError(f"spectro_wavelength must be 1D. Got shape {wvl.shape} in {path}")
    if data.shape[1] != wvl.shape[0]:
        raise ValueError(f"Mismatch: data.shape[1]={data.shape[1]} vs wvl.shape[0]={wvl.shape[0]} in {path}")

    return data, wvl


def wavelength_match(ref_wvl: np.ndarray, wvl: np.ndarray, tol=0.0) -> bool:
    if ref_wvl.shape != wvl.shape:
        return False
    if tol == 0.0:
        return np.array_equal(ref_wvl, wvl)
    return np.allclose(ref_wvl, wvl, atol=tol, rtol=0)


# -----------------------------
# Main combine function
# -----------------------------
def combine_power_h5(selected_paths: list[str], out_path: Path) -> Path:
    tagged_paths = sort_files_by_tag(selected_paths)

    all_spectra = []
    all_power = []
    wavelength_ref = None

    for tag, path in tagged_paths:
        data, wvl = read_h5_data(path)

        if wavelength_ref is None:
            wavelength_ref = wvl
        else:
            if not wavelength_match(wavelength_ref, wvl, tol=0.0):
                raise ValueError(
                    "spectro_wavelength differs between files.\n"
                    f"File: {path}\n"
                    "Fix: ensure all files were acquired with the same spectrometer axis."
                )

        n_spec = data.shape[0]
        pmin, pmax = POWER_RANGES_UW[tag]
        power_axis = np.linspace(pmin, pmax, n_spec)

        all_spectra.append(data)
        all_power.append(power_axis)

        print(f"[OK] {tag}: {Path(path).name} -> n_spec={n_spec}, power=({pmin}..{pmax}) uW")

    combined_spectra = np.vstack(all_spectra)
    combined_power = np.concatenate(all_power)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(out_path), "w") as h5:
        h5.create_dataset("spectro_data", data=combined_spectra)
        h5.create_dataset("spectro_wavelength", data=wavelength_ref)
        h5.create_dataset("power_axis", data=combined_power)

        # metadata
        h5.attrs["power_unit"] = "uW"
        h5.attrs["power_mapping"] = str(POWER_RANGES_UW)
        h5.attrs["file_order_low_to_high"] = ",".join([t for t, _ in tagged_paths])

    print(f"\n✅ Combined file saved to:\n{out_path}")
    return out_path


# -----------------------------
# UI / runner
# -----------------------------
def main():
    root = tk.Tk()
    root.withdraw()

    # 1) select h5 files
    messagebox.showinfo(
        "Select H5 files",
        "Select multiple .h5 files (e.g., 10_8 ... 10_4).\n"
        "Power ranges will be assigned automatically from filename tags."
    )

    paths = filedialog.askopenfilenames(
        title="Select .h5 files (10_8..10_4)",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
    )
    paths = list(paths)
    if not paths:
        print("No files selected.")
        return

    # 2) ask output filename only (folder is fixed to C:\Respo)
    out_name = filedialog.asksaveasfilename(
        title="Set output filename (will be saved under C:\\Respo)",
        initialdir=str(DEFAULT_OUT_DIR),
        initialfile="Combined_Powerdependence_v0.h5",
        defaultextension=".h5",
        filetypes=[("HDF5 files", "*.h5")],
    )
    if not out_name:
        print("No output name selected.")
        return

    # Force output into C:\Respo regardless of what user clicks
    out_path = DEFAULT_OUT_DIR / Path(out_name).name

    try:
        saved = combine_power_h5(paths, out_path)
        messagebox.showinfo("Done", f"Combined file saved:\n{saved}")
    except Exception as e:
        messagebox.showerror("Error", str(e))
        raise


if __name__ == "__main__":
    main()