import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Use a raw string (r"...") so backslashes aren't treated as escape sequences.
h5_path = Path(
    r"C:\Users\bwcho\Heriot-Watt University Team Dropbox\RES_EPS_Quantum_Photonics_Lab\Experiments\Current Experiments\Bay4_Tatyana_Cho_WS2WSe2\HWP_test_MF_5T_H2-PL_HS_-0.2_ratio_1__step1_0_to180__offset_0_1s_gain4_0.5-02-03_16-55-02-05_15-35.h5"
)

with h5py.File(str(h5_path), "r") as f:
    wl = f["spectro_wavelength"][:]          # (1340,) wavelength [nm]
    data = f["spectro_data"][:, 0, :]        # (N_angles, 1340)
    hwp_deg = f["xPositions"][:]             # (N_angles,) 0..179

def extract_intensity(wl, data, mode="integrate", wl_range=None, center_wl=None, window_nm=1.0):
    """
    mode:
      - "integrate": integrate over wl_range=(wlmin, wlmax)
      - "max": max within wl_range
      - "at": intensity at center_wl within +/- window_nm/2 (averaged)
      - "peak_track": find peak from mean spectrum and take "at" around it
    """
    wl = np.asarray(wl)
    data = np.asarray(data)

    if mode == "peak_track":
        mean_spec = data.mean(axis=0)
        peak_idx = np.argmax(mean_spec)
        center = wl[peak_idx]
        I = extract_intensity(wl, data, mode="at", center_wl=center, window_nm=window_nm)
        return I, center

    if wl_range is not None:
        wl_min, wl_max = wl_range
        mask = (wl >= wl_min) & (wl <= wl_max)
    else:
        mask = slice(None)

    if mode == "integrate":
        return np.trapz(data[:, mask], wl[mask], axis=1)

    if mode == "max":
        return data[:, mask].max(axis=1)

    if mode == "at":
        if center_wl is None:
            raise ValueError("center_wl must be provided for mode='at'")
        mask2 = (wl >= center_wl - window_nm/2) & (wl <= center_wl + window_nm/2)
        return data[:, mask2].mean(axis=1)

    raise ValueError(f"Unknown mode: {mode}")

# (1) Heatmap: spectrum vs HWP angle
plt.figure()
plt.imshow(
    data,
    aspect="auto",
    origin="lower",
    extent=[wl.min(), wl.max(), hwp_deg.min(), hwp_deg.max()],
)
plt.xlabel("Wavelength (nm)")
plt.ylabel("HWP angle (deg)")
plt.title("Spectra vs HWP angle (heatmap)")
plt.colorbar(label="Intensity (a.u.)")
plt.show()

# (2A) Peak-tracked intensity
I_peak, peak_wl = extract_intensity(wl, data, mode="peak_track", window_nm=1.0)

plt.figure()
plt.plot(hwp_deg, I_peak, marker="o", markersize=3, linewidth=1)
plt.xlabel("HWP angle (deg)")
plt.ylabel(f"Intensity near {peak_wl:.2f} nm (a.u.)")
plt.title("Angle-dependent intensity (peak-tracked)")
plt.show()

# (2B) ROI integrated intensity (edit the range you want)
I_roi = extract_intensity(wl, data, mode="integrate", wl_range=(850, 880))

plt.figure()
plt.plot(hwp_deg, I_roi, marker="o", markersize=3, linewidth=1)
plt.xlabel("HWP angle (deg)")
plt.ylabel("Integrated intensity (a.u.)")
plt.title("Angle-dependent intensity (ROI integrate)")
plt.show()
