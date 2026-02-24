import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk


HC_EV_NM = 1239.841984  # (h*c) in eV·nm


def wavelength_nm_to_energy(wavelength_nm: np.ndarray, *, unit: str = "eV") -> np.ndarray:
    """Convert wavelength in nm to photon energy.

    E(eV) = (h*c)/lambda = 1239.841984 / lambda(nm)
    """
    wl = np.asarray(wavelength_nm, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_eV = HC_EV_NM / wl

    unit = unit.strip().lower()
    if unit == "ev":
        return e_eV
    if unit == "mev":
        return 1e3 * e_eV
    raise ValueError(f"Unsupported energy unit: {unit!r} (use 'eV' or 'meV')")


def pick_h5_file(root: tk.Tk) -> Path:
    """Select an .h5/.hdf5 file (no folder dialog)."""
    try:
        root.attributes("-topmost", True)  # bring dialog to front (Windows)
    except Exception:
        pass

    file_path = filedialog.askopenfilename(
        parent=root,
        title="Select an HDF5 file",
        filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
    )
    if not file_path:
        raise SystemExit("No file selected. Exiting.")
    return Path(file_path)


def choose_view_mode(root: tk.Tk) -> str:
    """
    Returns "map" or "stack".
    Modal dialog without starting a second mainloop.
    """
    win = tk.Toplevel(root)
    win.title("Select view mode")
    win.resizable(False, False)
    try:
        win.attributes("-topmost", True)
    except Exception:
        pass

    mode_var = tk.StringVar(value="map")

    frm = ttk.Frame(win, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="How do you want to visualize the data?").pack(anchor="w", pady=(0, 8))
    ttk.Radiobutton(frm, text="Map (imshow)", value="map", variable=mode_var).pack(anchor="w")
    ttk.Radiobutton(frm, text="Stack (many spectra)", value="stack", variable=mode_var).pack(anchor="w")

    btns = ttk.Frame(frm)
    btns.pack(fill="x", pady=(10, 0))

    def _ok():
        win.destroy()

    ttk.Button(btns, text="OK", command=_ok).pack(side="right")

    win.protocol("WM_DELETE_WINDOW", _ok)
    win.grab_set()
    win.wait_window()  # modal wait

    mode = mode_var.get().strip().lower()
    return "stack" if mode == "stack" else "map"


def list_dataset_paths(h5: h5py.File) -> list[str]:
    paths: list[str] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append("/" + name if not name.startswith("/") else name)

    h5.visititems(visitor)
    return sorted(paths)


def read_first_matching_dataset(h5: h5py.File, candidates: list[str]) -> tuple[str, np.ndarray]:
    """
    Try to load the first dataset matching any candidate name/path.
    Returns (matched_key, dataset_array).
    """
    all_paths = list_dataset_paths(h5)

    # 1) exact match (top-level or absolute)
    for key in candidates:
        if key in h5:
            return (key, h5[key][:])
        if not key.startswith("/") and ("/" + key) in h5:
            k = "/" + key
            return (k, h5[k][:])

    # 2) match by basename anywhere in file
    for key in candidates:
        key_base = key.split("/")[-1]
        for p in all_paths:
            if p.endswith("/" + key_base) or p == key_base:
                return (p, h5[p][:])

    # 3) case-insensitive basename match
    cand_bases = {c.split("/")[-1].lower() for c in candidates}
    for p in all_paths:
        if p.split("/")[-1].lower() in cand_bases:
            return (p, h5[p][:])

    raise KeyError(
        "Could not find any of these datasets:\n"
        f"  candidates={candidates}\n\n"
        "Available datasets in this file:\n  - " + "\n  - ".join(all_paths[:200]) +
        ("\n  ... (truncated)" if len(all_paths) > 200 else "")
    )


def plot_map(
    I: np.ndarray,
    x: np.ndarray,
    B: np.ndarray,
    *,
    x_label: str,
    x_key: str,
    B_key: str,
    data_key: str,
    title: str,
):
    plt.figure()
    plt.imshow(
        I,
        aspect="auto",
        origin="lower",
        extent=[float(x.min()), float(x.max()), float(B.min()), float(B.max())],
    )
    plt.xlabel(f"{x_label} [{x_key}]")
    plt.ylabel(f"Magnetic field B (T) [{B_key}]")
    plt.title(title + f"\n(data={data_key})")
    plt.colorbar(label="Intensity (a.u.)")
    plt.tight_layout()
    plt.show()


def plot_stack(
    I: np.ndarray,
    x: np.ndarray,
    B: np.ndarray,
    *,
    x_label: str,
    x_key: str,
    B_key: str,
    data_key: str,
    title: str,
):
    """Plot many spectra as stacked lines (offset vertically)."""
    nB = int(I.shape[0])
    stride = max(1, nB // 35)
    idx = np.arange(0, nB, stride)

    amp = np.nanmedian(np.nanpercentile(I, 95, axis=1) - np.nanpercentile(I, 5, axis=1))
    if not np.isfinite(amp) or amp <= 0:
        amp = float(np.nanmax(I) - np.nanmin(I) + 1e-12)
    offset = 1.15 * amp

    plt.figure()
    for k, i in enumerate(idx):
        plt.plot(x, I[i, :] + k * offset, lw=1.0)

    plt.xlabel(f"{x_label} [{x_key}]")
    plt.ylabel("Intensity + offset (a.u.)")
    plt.title(title + f"\nStacked spectra (stride={stride})\n(data={data_key}, B={B_key})")
    plt.tight_layout()
    plt.show()


def main():
    # ---- axis choice ----
    # Set to True to display x-axis in energy (recommended for exciton plots).
    USE_ENERGY_AXIS = True
    ENERGY_UNIT = "eV"  # "eV" or "meV"

    # ---- pick file + mode via ONE Tk root ----
    _root = tk.Tk()
    _root.withdraw()

    h5_path = pick_h5_file(_root)
    view_mode = choose_view_mode(_root)

    try:
        _root.destroy()
    except Exception:
        pass

    with h5py.File(h5_path, "r") as f:
        wl_key, wl = read_first_matching_dataset(f, ["spectro_wavelength", "wavelength", "wl"])
        data_key, data_raw = read_first_matching_dataset(f, ["spectro_data", "data", "spectra"])

        wl = np.asarray(wl).squeeze()
        data_raw = np.asarray(data_raw)

        if wl.ndim != 1:
            raise ValueError(f"wavelength must be 1D, got shape={wl.shape} (from {wl_key})")

        # spectro_data shape normalize -> I: (Nfield, Nwl)
        if data_raw.ndim == 3:
            I0 = data_raw[:, 0, :]
        elif data_raw.ndim == 2:
            I0 = data_raw
        else:
            raise ValueError(f"Unexpected spectro data shape: {data_raw.shape} (from {data_key})")

        # If file stores (Nwl, Nfield), transpose
        if I0.shape[1] == wl.size:
            I = I0
        elif I0.shape[0] == wl.size:
            I = I0.T
        else:
            raise ValueError(
                f"Cannot align spectra with wavelength: I0.shape={I0.shape}, wl.size={wl.size} "
                f"(wl_key={wl_key}, data_key={data_key})"
            )

        # B axis from yLims/yStep if present
        if "yLims" in f and "yStep" in f:
            yLims = np.asarray(f["yLims"][:]).squeeze()
            yStep = float(np.asarray(f["yStep"][()]).squeeze())
            N = I.shape[0]
            B = float(yLims[0]) + np.arange(N, dtype=float) * yStep
            B_key = "yLims/yStep"
        else:
            B_key, B = read_first_matching_dataset(
                f, ["yPositions", "B", "field", "magnetic_field", "MagneticField"]
            )
            B = np.asarray(B).squeeze()

    # ---- NEW: validate + sort axes to guarantee field<->spectrum mapping ----
    if B.ndim != 1:
        raise ValueError(f"B must be 1D, got shape={B.shape} (from {B_key})")
    if B.size != I.shape[0]:
        raise ValueError(f"len(B) must match I rows: len(B)={B.size}, I.shape={I.shape} (B_key={B_key})")

    # x-axis (wavelength or energy) + sort ascending
    if USE_ENERGY_AXIS:
        x = wavelength_nm_to_energy(wl, unit=ENERGY_UNIT)
        x_label = f"Energy ({ENERGY_UNIT})"
        x_key = f"{wl_key}→E"
    else:
        x = wl.astype(float, copy=False)
        x_label = "Wavelength (nm)"
        x_key = wl_key

    x_idx = np.argsort(x)
    x = x[x_idx]
    I = I[:, x_idx]

    # sort B ascending
    b_idx = np.argsort(B)
    B = B[b_idx]
    I = I[b_idx, :]

    # (optional) baseline removal
    I = I - np.median(I, axis=1, keepdims=True)

    title = f"Spectra vs magnetic field\n{h5_path.name}"

    if view_mode == "stack":
        plot_stack(I, x, B, x_label=x_label, x_key=x_key, B_key=B_key, data_key=data_key, title=title)
    else:
        plot_map(I, x, B, x_label=x_label, x_key=x_key, B_key=B_key, data_key=data_key, title=title)


if __name__ == "__main__":
    main()
