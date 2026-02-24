# Plot fits (single, double, triple Gaussian) for every second spectrum
def plot_fits_every_second_spectrum(h5_path):
    with h5py.File(h5_path, 'r') as f:
        data = f['spectro_data'][...]
        wvl = f['spectro_wavelength'][...]
        energy = 1240.0 / wvl
        idx = np.argsort(energy)
        energy_sorted = energy[idx]
        data_sorted = data[:, idx]
        n_spectra = data_sorted.shape[0]
        for i in range(0, n_spectra, 2):
            spectrum = data_sorted[i]
            bg = np.median(spectrum)
            plt.figure(figsize=(8, 5))
            plt.plot(energy_sorted, spectrum, label=f'Spectrum {i}')
            plt.xlim(1.38, 1.48)
            popt = None
            if i < 4:
                # Single Gaussian
                mask = (energy_sorted >= 1.384) & (energy_sorted <= 1.42)
                x_fit = energy_sorted[mask]
                y_fit = spectrum[mask]
                # Use actual peak position and amplitude as initial guess
                peak_idx = np.argmax(y_fit)
                peak_cen = x_fit[peak_idx]
                peak_amp = y_fit[peak_idx] - bg
                p0 = [peak_amp, peak_cen, 0.004, bg]
                bounds = ([0, 1.384, 0.001, 0], [np.inf, 1.42, 0.03, np.max(spectrum)])
                try:
                    popt, _ = curve_fit(single_gauss, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
                    plt.plot(x_fit, single_gauss(x_fit, *popt), '--', label='Single Gauss Fit')
                    plt.axvline(popt[1], color='k', linestyle=':', alpha=0.5)
                except Exception:
                    plt.text(peak_cen, y_fit[peak_idx], 'Fit failed', color='red')
            elif i < 125:
                # Double Gaussian
                mask1 = (energy_sorted >= 1.384) & (energy_sorted <= 1.42)
                mask2 = (energy_sorted >= 1.43) & (energy_sorted <= 1.4575)
                mask = mask1 | mask2
                x_fit = energy_sorted[mask]
                y_fit = spectrum[mask]
                # Find two peaks: one in each region
                y1 = spectrum[mask1]
                x1 = energy_sorted[mask1]
                y2 = spectrum[mask2]
                x2 = energy_sorted[mask2]
                idx1 = np.argmax(y1)
                idx2 = np.argmax(y2)
                cen1 = x1[idx1] if len(x1) > 0 else 1.40
                amp1 = y1[idx1] - bg if len(y1) > 0 else spectrum.max()/2 - bg
                cen2 = x2[idx2] if len(x2) > 0 else 1.444
                amp2 = y2[idx2] - bg if len(y2) > 0 else spectrum.max()/2 - bg
                p0 = [amp1, cen1, 0.004, amp2, cen2, 0.004, bg]
                bounds = ([0, 1.384, 0.001, 0, 1.438, 0.001, 0],
                          [np.inf, 1.415, 0.03, np.inf, 1.4535, 0.03, np.max(spectrum)])
                try:
                    popt, _ = curve_fit(double_gauss, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
                    plt.plot(x_fit, double_gauss(x_fit, *popt), '--', label='Double Gauss Fit')
                    plt.axvline(popt[1], color='k', linestyle=':', alpha=0.5)
                    plt.axvline(popt[4], color='k', linestyle=':', alpha=0.5)
                except Exception:
                    plt.text(cen1, amp1+bg, 'Fit failed', color='red')
            else:
                # Triple Gaussian
                mask1 = (energy_sorted >= 1.384) & (energy_sorted <= 1.4151)
                # For spectra >= 200, widen the second peak region and bounds
                if i >= 180:
                    mask2 = (energy_sorted >= 1.444) & (energy_sorted <= 1.4530)
                    cen2_max = 1.4530
                else:
                    mask2 = (energy_sorted >= 1.444) & (energy_sorted <= 1.4476)
                    cen2_max = 1.4476
                mask3 = (energy_sorted >= 1.4631) & (energy_sorted <= 1.4767)
                mask = mask1 | mask2 | mask3
                x_fit = energy_sorted[mask]
                y_fit = spectrum[mask]
                # Find three peaks: one in each region
                y1 = spectrum[mask1]
                x1 = energy_sorted[mask1]
                y2 = spectrum[mask2]
                x2 = energy_sorted[mask2]
                y3 = spectrum[mask3]
                x3 = energy_sorted[mask3]
                idx1 = np.argmax(y1)
                idx2 = np.argmax(y2)
                idx3 = np.argmax(y3)
                cen1 = x1[idx1] if len(x1) > 0 else 1.40
                amp1 = y1[idx1] - bg if len(y1) > 0 else spectrum.max()/3 - bg
                cen2 = x2[idx2] if len(x2) > 0 else 1.446
                amp2 = y2[idx2] - bg if len(y2) > 0 else spectrum.max()/3 - bg
                cen3 = x3[idx3] if len(x3) > 0 else 1.4708
                amp3 = y3[idx3] - bg if len(y3) > 0 else spectrum.max()/3 - bg
                p0 = [amp1, cen1, 0.004, amp2, cen2, 0.004, amp3, cen3, 0.004, bg]
                bounds = ([0, 1.384, 0.001, 0, 1.444, 0.001, 0, 1.4631, 0.001, 0],
                          [np.inf, 1.4151, 0.03, np.inf, cen2_max, 0.03, np.inf, 1.4767, 0.03, np.max(spectrum)])
                try:
                    popt, _ = curve_fit(triple_gauss, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
                    plt.plot(x_fit, triple_gauss(x_fit, *popt), '--', label='Triple Gauss Fit')
                    plt.axvline(popt[1], color='k', linestyle=':', alpha=0.5)
                    plt.axvline(popt[4], color='k', linestyle=':', alpha=0.5)
                    plt.axvline(popt[7], color='k', linestyle=':', alpha=0.5)
                except Exception:
                    plt.text(cen1, amp1+bg, 'Fit failed', color='red')
            plt.xlabel('Energy (eV)')
            plt.ylabel('Intensity')
            plt.title(f'Spectrum {i} Gaussian Fit')
            plt.legend()
            plt.tight_layout()
            plt.show()
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter  # For cosmic ray removal
from scipy.optimize import curve_fit # For Gaussian fitting
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d

# ---------------------- USER EDITABLE ----------------------
h5_paths = [
    r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\v=1\Combined_Powerdependence_v1.h5"
]


def plot_all_spectra_map(h5_path):
    with h5py.File(h5_path, 'r') as f:
        data = f['spectro_data'][...]
        wvl = f['spectro_wavelength'][...]
        energy = 1240.0 / wvl
        idx = np.argsort(energy)
        energy_sorted = energy[idx]
        data_sorted = data[:, idx]
        plt.figure(figsize=(10, 6))
        plt.imshow(data_sorted, aspect='auto', extent=[energy_sorted.min(), energy_sorted.max(), 0, data.shape[0]],
                   origin='lower', cmap='jet')
        plt.colorbar(label='Intensity')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Frame (Power Step)')
        plt.title('All Spectra Map from Combined_Powerdependence_v1.h5')
        plt.tight_layout()
        plt.show()

def single_gauss(x, amp, cen, wid, bg):
    return amp * np.exp(-(x-cen)**2 / (2*wid**2)) + bg

def double_gauss(x, amp1, cen1, wid1, amp2, cen2, wid2, bg):
    return (amp1 * np.exp(-(x-cen1)**2 / (2*wid1**2)) +
            amp2 * np.exp(-(x-cen2)**2 / (2*wid2**2)) + bg)

def triple_gauss(x, amp1, cen1, wid1, amp2, cen2, wid2, amp3, cen3, wid3, bg):
    return (amp1 * np.exp(-(x-cen1)**2 / (2*wid1**2)) +
            amp2 * np.exp(-(x-cen2)**2 / (2*wid2**2)) +
            amp3 * np.exp(-(x-cen3)**2 / (2*wid3**2)) + bg)

# --- New function to plot peak center vs power ---
def rebin_1d(arr, factor):
    n = arr.shape[-1] // factor
    return arr[..., :n*factor].reshape(-1, n, factor).mean(axis=-1)

def plot_peak_center_vs_power(h5_path):
    with h5py.File(h5_path, 'r') as f:
        data = f['spectro_data'][...]
        wvl = f['spectro_wavelength'][...]
        energy = 1240.0 / wvl
        idx = np.argsort(energy)
        energy_sorted = energy[idx]
        data_sorted = data[:, idx]
        n_spectra = data_sorted.shape[0]
        power = np.linspace(0, 140, n_spectra)  # in μW
        peak_centers = []
        for i, spectrum in enumerate(data_sorted):
            bg = np.median(spectrum)
            popt = None
            centers = []
            if i < 4:
                # Single Gaussian
                mask = (energy_sorted >= 1.384) & (energy_sorted <= 1.41)
                x_fit = energy_sorted[mask]
                y_fit = spectrum[mask]
                peak_idx = np.argmax(y_fit)
                peak_cen = x_fit[peak_idx]
                peak_amp = y_fit[peak_idx] - bg
                p0 = [peak_amp, peak_cen, 0.004, bg]
                bounds = ([0, 1.384, 0.001, 0], [np.inf, 1.415, 0.03, np.max(spectrum)])
                try:
                    popt, _ = curve_fit(single_gauss, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
                    centers = [popt[1]]
                except Exception:
                    centers = [np.nan]
            elif i < 125:
                # Double Gaussian
                mask1 = (energy_sorted >= 1.384) & (energy_sorted <= 1.41)
                mask2 = (energy_sorted >= 1.438) & (energy_sorted <= 1.4535)
                mask = mask1 | mask2
                x_fit = energy_sorted[mask]
                y_fit = spectrum[mask]
                y1 = spectrum[mask1]
                x1 = energy_sorted[mask1]
                y2 = spectrum[mask2]
                x2 = energy_sorted[mask2]
                idx1 = np.argmax(y1)
                idx2 = np.argmax(y2)
                cen1 = x1[idx1] if len(x1) > 0 else 1.40
                amp1 = y1[idx1] - bg if len(y1) > 0 else spectrum.max()/2 - bg
                cen2 = x2[idx2] if len(x2) > 0 else 1.444
                amp2 = y2[idx2] - bg if len(y2) > 0 else spectrum.max()/2 - bg
                p0 = [amp1, cen1, 0.004, amp2, cen2, 0.004, bg]
                bounds = ([0, 1.384, 0.001, 0, 1.438, 0.001, 0],
                          [np.inf, 1.415, 0.03, np.inf, 1.4535, 0.03, np.max(spectrum)])
                try:
                    popt, _ = curve_fit(double_gauss, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
                    centers = [popt[1], popt[4]]
                except Exception:
                    centers = [np.nan, np.nan]
            else:
                # Triple Gaussian
                mask1 = (energy_sorted >= 1.384) & (energy_sorted <= 1.4151)
                if i >= 200:
                    mask2 = (energy_sorted >= 1.444) & (energy_sorted <= 1.4530)
                    cen2_max = 1.4530
                else:
                    mask2 = (energy_sorted >= 1.444) & (energy_sorted <= 1.4476)
                    cen2_max = 1.4476
                mask3 = (energy_sorted >= 1.4631) & (energy_sorted <= 1.4767)
                mask = mask1 | mask2 | mask3
                x_fit = energy_sorted[mask]
                y_fit = spectrum[mask]
                y1 = spectrum[mask1]
                x1 = energy_sorted[mask1]
                y2 = spectrum[mask2]
                x2 = energy_sorted[mask2]
                y3 = spectrum[mask3]
                x3 = energy_sorted[mask3]
                idx1 = np.argmax(y1)
                idx2 = np.argmax(y2)
                idx3 = np.argmax(y3)
                cen1 = x1[idx1] if len(x1) > 0 else 1.40
                amp1 = y1[idx1] - bg if len(y1) > 0 else spectrum.max()/3 - bg
                cen2 = x2[idx2] if len(x2) > 0 else 1.446
                amp2 = y2[idx2] - bg if len(y2) > 0 else spectrum.max()/3 - bg
                cen3 = x3[idx3] if len(x3) > 0 else 1.4708
                amp3 = y3[idx3] - bg if len(y3) > 0 else spectrum.max()/3 - bg
                p0 = [amp1, cen1, 0.004, amp2, cen2, 0.004, amp3, cen3, 0.004, bg]
                bounds = ([0, 1.384, 0.001, 0, 1.444, 0.001, 0, 1.4631, 0.001, 0],
                          [np.inf, 1.4151, 0.03, np.inf, cen2_max, 0.03, np.inf, 1.4767, 0.03, np.max(spectrum)])
                try:
                    popt, _ = curve_fit(triple_gauss, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
                    centers = [popt[1], popt[4], popt[7]]
                except Exception:
                    centers = [np.nan, np.nan, np.nan]
            while len(centers) < 3:
                centers.append(np.nan)
            peak_centers.append(centers)
        peak_centers = np.array(peak_centers)
        plt.figure(figsize=(8,5))
        n_peaks = peak_centers.shape[1] if peak_centers.ndim > 1 else 1
        for j in range(n_peaks):
            plt.plot(power, peak_centers[:,j], marker='o', label=f'Peak {j+1}')
        plt.xlabel('Power (μW)')
        plt.ylabel('Peak Center (eV)')
        plt.title('Peak Center vs Power (from robust fit)')
        plt.legend()
        plt.tight_layout()
        plt.show()

def adaptive_peak_fitting_gauss(h5_path):
    with h5py.File(h5_path, 'r') as f:
        data = f['spectro_data'][...]
        wvl = f['spectro_wavelength'][...]
        energy = 1240.0 / wvl
        idx = np.argsort(energy)
        energy_sorted = energy[idx]
        data_sorted = data[:, idx]
        n_spectra = data_sorted.shape[0]
        for i, spectrum in enumerate(data_sorted):
            bg = np.median(spectrum)
            plt.figure(figsize=(8, 5))
            plt.plot(energy_sorted, spectrum, label=f'Spectrum {i}')
            plt.xlim(1.38, 1.48)
            popt = None
            if i < 4:
                # Single Gaussian
                mask = (energy_sorted >= 1.384) & (energy_sorted <= 1.41)
                x_fit = energy_sorted[mask]
                y_fit = spectrum[mask]
                p0 = [spectrum.max() - bg, 1.40, 0.003, bg]
                bounds = ([0, 1.384, 0.001, 0], [np.inf, 1.41, 0.02, np.max(spectrum)])
                try:
                    popt, _ = curve_fit(single_gauss, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000)
                except Exception:
                    pass
                if popt is not None:
                    plt.plot(x_fit, single_gauss(x_fit, *popt), '--', label='Single Gauss Fit')
                    plt.axvline(popt[1], color='k', linestyle=':', alpha=0.5)
            elif i < 125:
                # Double Gaussian
                mask = ((energy_sorted >= 1.384) & (energy_sorted <= 1.41)) | ((energy_sorted >= 1.438) & (energy_sorted <= 1.4535))
                x_fit = energy_sorted[mask]
                y_fit = spectrum[mask]
                p0 = [spectrum.max()/2 - bg, 1.40, 0.003,
                      spectrum.max()/2 - bg, 1.444, 0.003, bg]
                bounds = ([0, 1.384, 0.001, 0, 1.438, 0.001, 0],
                          [np.inf, 1.41, 0.02, np.inf, 1.4535, 0.02, np.max(spectrum)])
                try:
                    popt, _ = curve_fit(double_gauss, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000)
                except Exception:
                    pass
                if popt is not None:
                    plt.plot(x_fit, double_gauss(x_fit, *popt), '--', label='Double Gauss Fit')
                    plt.axvline(popt[1], color='k', linestyle=':', alpha=0.5)
                    plt.axvline(popt[4], color='k', linestyle=':', alpha=0.5)
            else:
                # Triple Gaussian
                mask = ((energy_sorted >= 1.384) & (energy_sorted <= 1.41)) | \
                       ((energy_sorted >= 1.438) & (energy_sorted <= 1.4535)) | \
                       ((energy_sorted >= 1.4631) & (energy_sorted <= 1.4767))
                x_fit = energy_sorted[mask]
                y_fit = spectrum[mask]
                p0 = [spectrum.max()/3 - bg, 1.40, 0.003,
                      spectrum.max()/3 - bg, 1.444, 0.003,
                      spectrum.max()/3 - bg, 1.4708, 0.003, bg]
                bounds = ([0, 1.384, 0.001, 0, 1.438, 0.001, 0, 1.4631, 0.001, 0],
                          [np.inf, 1.41, 0.02, np.inf, 1.4535, 0.02, np.inf, 1.4767, 0.02, np.max(spectrum)])
                try:
                    popt, _ = curve_fit(triple_gauss, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000)
                except Exception:
                    pass
                if popt is not None:
                    plt.plot(x_fit, triple_gauss(x_fit, *popt), '--', label='Triple Gauss Fit')
                    plt.axvline(popt[1], color='k', linestyle=':', alpha=0.5)
                    plt.axvline(popt[4], color='k', linestyle=':', alpha=0.5)
                    plt.axvline(popt[7], color='k', linestyle=':', alpha=0.5)
            plt.xlabel('Energy (eV)')
            plt.ylabel('Intensity')
            plt.title(f'Spectrum {i} Gaussian Fit')
            plt.legend()
            plt.tight_layout()
            plt.show()


# Function to plot integrated intensity vs power for each peak (log-log scale)
def plot_integrated_intensity_vs_power(h5_path):
    # Use numpy.trapz for integration instead of simps
    def rebin_1d(arr, factor):
        n = arr.shape[-1] // factor
        return arr[..., :n*factor].reshape(-1, n, factor).mean(axis=-1)

    # Custom trapezoidal integration (since np.trapz is missing)
    def trapz(y, x):
        y = np.asarray(y)
        x = np.asarray(x)
        return ((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2).sum()
    with h5py.File(h5_path, 'r') as f:
        data = f['spectro_data'][...]
        wvl = f['spectro_wavelength'][...]
        energy = 1240.0 / wvl
        idx = np.argsort(energy)
        energy_sorted = energy[idx]
        data_sorted = data[:, idx]
        # Rebin energy and spectra by factor of 4
        rebin_factor = 2
        energy_rebinned = rebin_1d(energy_sorted, rebin_factor)
        data_rebinned = np.array([rebin_1d(spectrum, rebin_factor) for spectrum in data_sorted])
        n_spectra = data_rebinned.shape[0]
        power = np.linspace(0, 140, n_spectra)  # in μW
        # Remove 600 background
        data_rebinned = data_rebinned - 600
        # Define peak regions (same as fitting)
        peak_masks = [
            (energy_rebinned >= 1.384) & (energy_rebinned <= 1.41),
            (energy_rebinned >= 1.438) & (energy_rebinned <= 1.4535),
            (energy_rebinned >= 1.4631) & (energy_rebinned <= 1.4767)
        ]
        peak_labels = [f"Peak {i+1} (x={energy_rebinned[mask].mean():.2f})" for i, mask in enumerate(peak_masks)]
        intensities = [
            [trapz(spectrum[mask], energy_rebinned[mask]) for spectrum in data_rebinned]
            for mask in peak_masks
        ]
        plt.figure(figsize=(6,6))
        for j, intensity in enumerate(intensities):
            plt.plot(power, intensity, 'o', label=peak_labels[j])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Power (μW)')
        plt.ylabel('Integrated Intensity (a.u.)')
        plt.title('Power vs. Integrated Intensity')
        plt.legend()
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
        plot_all_spectra_map(h5_paths[0])
        #  adaptive_peak_fitting_gauss(h5_paths[0]

        # Plot fits for every second spectrum
        plot_fits_every_second_spectrum(h5_paths[0])
        
        # Plot peak center vs power
        plot_peak_center_vs_power(h5_paths[0])

        # Plot integrated intensity vs power (log-log)
        plot_integrated_intensity_vs_power(h5_paths[0])

