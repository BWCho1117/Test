import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter


# ---------------- Cosmic Ray Removal ---------------- #

def remove_cosmic_rays_1d(spectrum, kernel_size=7, sigma_threshold=6):
    spectrum = spectrum.astype(float)
    baseline = median_filter(spectrum, size=kernel_size)

    diff = spectrum - baseline
    mad = np.median(np.abs(diff))

    if mad == 0:
        return spectrum

    threshold = sigma_threshold * 1.4826 * mad
    mask = np.abs(diff) > threshold

    cleaned = spectrum.copy()
    cleaned[mask] = baseline[mask]

    return cleaned


def remove_cosmic_rays(data, kernel_size=7, threshold=6):
    """
    data shape: (n_frames, n_wavelengths)
    """
    cleaned = np.empty_like(data)

    for i in range(data.shape[0]):
        cleaned[i] = remove_cosmic_rays_1d(
            data[i],
            kernel_size=kernel_size,
            sigma_threshold=threshold
        )

    return cleaned


# ---------------- Gaussian Model ---------------- #

def single_gauss(x, amp, cen, wid, bg):
    return amp * np.exp(-(x - cen) ** 2 / (2 * wid ** 2)) + bg


# ---------------- 2D Power Map ---------------- #

def plot_all_spectra_map(h5_path,
                         intensity_log=False,
                         y_log=True,
                         cosmic_kernel=7,
                         cosmic_threshold=6):

    with h5py.File(h5_path, 'r') as f:
        data  = f['spectro_data'][...]
        wvl   = f['spectro_wavelength'][...]
        power = f['power_axis'][...]

    # ----- Energy axis 정렬 -----
    energy = 1240.0 / wvl
    idx_energy = np.argsort(energy)
    energy_sorted = energy[idx_energy]
    data_sorted = data[:, idx_energy]

    # ----- Power 기준 정렬 -----
    idx_power = np.argsort(power)
    power_sorted = power[idx_power]
    data_sorted = data_sorted[idx_power, :]

    # ----- Cosmic ray 제거 -----
    data_sorted = remove_cosmic_rays(
        data_sorted,
        kernel_size=cosmic_kernel,
        threshold=cosmic_threshold
    )

    # ----- Intensity log 선택 -----
    if intensity_log:
        data_plot = np.log10(np.clip(data_sorted, 1e-6, None))
        label = "log10(Intensity)"
    else:
        data_plot = data_sorted
        label = "Intensity"

    # ----- Plot -----
    plt.figure(figsize=(8, 6))

    mesh = plt.pcolormesh(
        energy_sorted,
        power_sorted,
        data_plot,
        shading='auto',
        cmap='RdBu_r'
    )

    plt.colorbar(mesh, label=label)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Power (µW)")

    if y_log:
        plt.yscale("log")

    plt.tight_layout()
    plt.show()


# ---------------- Example Single Spectrum Fit ---------------- #

def plot_single_spectrum_fit(h5_path, spectrum_index=0):

    with h5py.File(h5_path, 'r') as f:
        data  = f['spectro_data'][...]
        wvl   = f['spectro_wavelength'][...]

    energy = 1240.0 / wvl
    idx = np.argsort(energy)
    energy = energy[idx]
    spectrum = data[spectrum_index][idx]

    spectrum = remove_cosmic_rays_1d(spectrum)

    bg = np.median(spectrum)

    mask = (energy >= 1.384) & (energy <= 1.42)
    x_fit = energy[mask]
    y_fit = spectrum[mask]

    peak_idx = np.argmax(y_fit)
    peak_cen = x_fit[peak_idx]
    peak_amp = y_fit[peak_idx] - bg

    p0 = [peak_amp, peak_cen, 0.004, bg]

    popt, _ = curve_fit(single_gauss, x_fit, y_fit, p0=p0)

    plt.figure(figsize=(7, 5))
    plt.plot(energy, spectrum, label="Cleaned Spectrum")
    plt.plot(x_fit, single_gauss(x_fit, *popt), '--', label="Fit")
    plt.axvline(popt[1], linestyle=':')
    plt.xlabel("Energy (eV)")
    plt.ylabel("Power (μW)")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------- Run ---------------- #

if __name__ == "__main__":

    h5_path = r"C:\Respo\Combined_Powerdependence_v1.h5"

    # 🔥 intensity linear / log 선택 가능
    plot_all_spectra_map(
        h5_path,
        intensity_log=True,   # True → log intensity
        y_log=True,            # True → log power axis
        cosmic_kernel=7,       # 조절 가능
        cosmic_threshold=6     # 조절 가능
    )

    plot_single_spectrum_fit(h5_path, spectrum_index=10)