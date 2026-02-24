import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from scipy.optimize import curve_fit
from scipy import constants
from tkinter import Tk, filedialog
 
# --- DEVICE & PHYSICAL CONSTANTS ---
# Used for calculating carrier density from gate voltage.
epsilon_r = 3.0  # Relative permittivity of hBN
V_cnp = -0.4     # Charge neutrality point (V)
dt = 9e-9        # Top gate thickness (m)
db = 12e-9       # Bottom gate thickness (m)
e = constants.elementary_charge
epsilon_0 = constants.epsilon_0
 
def voltage_to_density(vg):
    """Converts gate voltage to carrier density in units of 10^12 cm^-2."""
    # Formula: n = (ε*ε₀*ΔVg/e) * (1/dt + 1/db)
    delta_vg = vg - V_cnp
    n_m2 = (epsilon_r * epsilon_0 * delta_vg / e) * (1/dt + 1/db)
    n_cm2 = n_m2 / 1e4  # Convert from m^-2 to cm^-2
    return n_cm2 / 1e12 # Return in units of 10^12 cm^-2
 
# Define normalization class
class MidpointNormalize(mcolors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=0, clip=False):
        self.midpoint = midpoint
        super().__init__(vmin, vmax, clip)
 
    def __call__(self, value, clip=None):
        # This normalization maps the data range to the colormap.
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))
 
# --- Gaussian for fitting ---
def gaussian(x, a, x0, sigma, offset):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) + offset
 
def double_gaussian(x, a1, x0_1, sigma1, a2, x0_2, sigma2, offset):
    """Models two Gaussian peaks."""
    return (a1 * np.exp(-((x - x0_1) ** 2) / (2 * sigma1 ** 2))) + \
           (a2 * np.exp(-((x - x0_2) ** 2) / (2 * sigma2 ** 2))) + \
           offset
 
def triple_gaussian(x, a1, x0_1, sigma1, a2, x0_2, sigma2, a3, x0_3, sigma3, offset):
    """Models three Gaussian peaks."""
    return (a1 * np.exp(-((x - x0_1) ** 2) / (2 * sigma1 ** 2))) + \
           (a2 * np.exp(-((x - x0_2) ** 2) / (2 * sigma2 ** 2))) + \
           (a3 * np.exp(-((x - x0_3) ** 2) / (2 * sigma3 ** 2))) + \
           offset
 
# --- 파일 선택 대화상자 열기 ---
root = Tk()
root.withdraw()  # tkinter의 빈 창 숨기기
file_list = filedialog.askopenfilenames(
    title="분석할 HDF5 파일을 선택하세요",
    filetypes=(("HDF5 files", "*.h5"), ("All files", "*.*"))
)
root.destroy() # 대화상자 닫은 후 tkinter 프로세스 종료

if not file_list:
    print("파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
    exit()
 
plt.ion()  # Turn on interactive mode
 
# Plot each file in a separate window
for file_path in file_list:
    try:
        with h5py.File(file_path, 'r') as h5_file:
            spectra = h5_file['spectro_data'][:, 0, :]
            wavelengths = h5_file['spectro_wavelength'][:]
            voltages = h5_file['xPositions'][:]
       
        photon_energy = 1240.0 / wavelengths
 
        # Flip data for correct orientation in the plot
        plot_data = spectra[::-1, ::-1]
 
        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(6, 7))
       
        # --- Define plot range and color limits ---
        # Adjust v_min and v_max for your raw data's intensity range.
        v_min, v_max = 0, 5000
        energy_min, energy_max = 1.37, 1.5
        voltage_min, voltage_max = -2.5, 2.5
 
        im = ax.imshow(
            plot_data,
            aspect='auto',
            extent=[photon_energy[-1], photon_energy[0], voltages[0], voltages[-1]],
            cmap='inferno', # ADVICE: Use a sequential colormap like 'inferno' for intensity data.
            vmin=v_min,
            vmax=v_max
        )
 
        # --- Axes and Labels ---
        ax.set_xlim(energy_min, energy_max)
        ax.set_ylim(voltage_min, voltage_max)
       
        ax.set_xlabel('Energy (eV)', fontsize=14)
        ax.set_ylabel('Voltage (V)', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
 
        # --- Add Carrier Density Axis ---
        ax2 = ax.twinx()
        density_min, density_max = voltage_to_density(voltage_min), voltage_to_density(voltage_max)
        ax2.set_ylim(density_min, density_max)
        ax2.set_ylabel(r'Carrier Density (10$^{12}$ cm$^{-2}$)', fontsize=14)
 
        # --- Create a small inset colorbar in the top right ---
        # Define position: [left, bottom, width, height] in figure coordinates
        cax = fig.add_axes([0.8, 0.8, 0.03, 0.1])
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.set_label('R (a.u.)', fontsize=8, rotation=270, labelpad=12, color = 'white')
        cbar.ax.tick_params(labelsize=6)
       
        plt.tight_layout(pad=1.5)
        plt.show(block=False)
 
        # --- Plotting in Log Scale ---
        fig_log, ax_log = plt.subplots(figsize=(6, 7))
       
        # Adjust the color range for the log plot to enhance peak visibility.
        positive_data = plot_data[plot_data > 0]
        if positive_data.size > 0:
            # ADVICE: Using the median (50th percentile) as vmin is great for background suppression.
            log_vmin = np.percentile(positive_data, 50)
            log_vmax = np.percentile(positive_data, 99.5)
        else:
            log_vmin, log_vmax = 1, 10000
 
        im_log = ax_log.imshow(
            plot_data,
            aspect='auto',
            extent=[photon_energy[-1], photon_energy[0], voltages[0], voltages[-1]],
            cmap='inferno', # ADVICE: Use a sequential colormap here as well.
            norm=mcolors.LogNorm(vmin=log_vmin, vmax=log_vmax)
        )
 
        # --- Axes and Labels for Log Plot ---
        ax_log.set_xlim(energy_min, energy_max)
        ax_log.set_ylim(voltage_min, voltage_max)
       
        ax_log.set_xlabel('Energy (eV)', fontsize=14)
        ax_log.set_ylabel('Voltage (V)', fontsize=14)
        ax_log.set_title('Intensity Map (Log Scale)', fontsize=14)
        ax_log.tick_params(axis='both', which='major', labelsize=12)
 
        # --- Add Carrier Density Axis (Log Plot) ---
        ax_log2 = ax_log.twinx()
        ax_log2.set_ylim(density_min, density_max)
        ax_log2.set_ylabel(r'Carrier Density (10$^{12}$ cm$^{-2}$)', fontsize=14)
 
        # --- Create a small inset colorbar in the top right (Log Plot) ---
        cax_log = fig_log.add_axes([0.8, 0.8, 0.03, 0.1])
        cbar_log = fig_log.colorbar(im_log, cax=cax_log, orientation='vertical')
        cbar_log.set_label('R (a.u.) - Log', fontsize=8, rotation=270, labelpad=12)
        cbar_log.ax.tick_params(labelsize=6)
       
        plt.tight_layout(pad=1.5)
        plt.show(block=False)
 
        # --- ADVICE: Plot First Derivative to Clearly See Peak Positions ---
        # Calculate the derivative dR/dE
        dE = np.diff(photon_energy)[0] # Energy step
        dR_dE = np.gradient(spectra, dE, axis=1)
        dR_dE_flipped = dR_dE[::-1, ::-1] # Flip for plotting
 
        fig_deriv, ax_deriv = plt.subplots(figsize=(6, 7))
       
        # For a derivative, a diverging colormap is correct.
        # We find the max absolute value to create a symmetric color scale.
        deriv_vmax = np.percentile(np.abs(dR_dE_flipped), 99)
 
        im_deriv = ax_deriv.imshow(
            dR_dE_flipped,
            aspect='auto',
            extent=[photon_energy[-1], photon_energy[0], voltages[0], voltages[-1]],
            cmap='seismic', # 'seismic' or 'RdBu_r' are good choices here.
            norm=MidpointNormalize(vmin=-deriv_vmax, vmax=deriv_vmax, midpoint=0)
        )
        ax_deriv.set_title('First Derivative (dR/dE)', fontsize=14)
        ax_deriv.set_xlabel('Energy (eV)', fontsize=14)
        ax_deriv.set_ylabel('Voltage (V)', fontsize=14)
        ax_deriv.set_xlim(energy_min, energy_max)
        ax_deriv.set_ylim(voltage_min, voltage_max)
 
        # --- Add Carrier Density Axis (Derivative Plot) ---
        ax_deriv2 = ax_deriv.twinx()
        ax_deriv2.set_ylim(density_min, density_max)
        ax_deriv2.set_ylabel(r'Carrier Density (10$^{12}$ cm$^{-2}$)', fontsize=14)
 
        # --- Create a small inset colorbar in the top right (Derivative Plot) ---
        cax_deriv = fig_deriv.add_axes([0.8, 0.8, 0.03, 0.1])
        cbar_deriv = fig_deriv.colorbar(im_deriv, cax=cax_deriv, orientation='vertical')
        cbar_deriv.set_label('dR/dE', fontsize=8, rotation=270, labelpad=12)
        cbar_deriv.ax.tick_params(labelsize=6)
        plt.tight_layout(pad=1.5)
        plt.show(block=False)
 
 
        # --- Fit and Plot Line Cut at -0.4V ---
        target_voltage = -0.4
       
        # Find the index of the voltage closest to the target
        idx = np.argmin(np.abs(voltages - target_voltage))
        actual_v = voltages[idx]
       
        # Extract the spectrum and energy axis for fitting
        line_cut_spectrum = spectra[idx, :]
        energy = photon_energy
       
        # --- Perform Triple Gaussian Fit ---
        fit_successful = False
        try:
            # Define energy ranges to find initial guesses for each peak
            range1_mask = (energy >= 1.39) & (energy <= 1.42)  # Peak at ~1.40 eV
            range2_mask = (energy >= 1.43) & (energy <= 1.45)  # Peak at ~1.44 eV
            range3_mask = (energy >= 1.45) & (energy <= 1.48)  # Peak at ~1.465 eV
 
            # Initial guesses for Peak 1
            idx1 = np.argmax(line_cut_spectrum[range1_mask])
            x0_1_guess = energy[range1_mask][idx1]
            a1_guess = line_cut_spectrum[range1_mask][idx1]
           
            # Initial guesses for Peak 2
            idx2 = np.argmax(line_cut_spectrum[range2_mask])
            x0_2_guess = energy[range2_mask][idx2]
            a2_guess = line_cut_spectrum[range2_mask][idx2]
 
            # Initial guesses for Peak 3
            idx3 = np.argmax(line_cut_spectrum[range3_mask])
            x0_3_guess = energy[range3_mask][idx3]
            a3_guess = line_cut_spectrum[range3_mask][idx3]
 
            offset_guess = np.min(line_cut_spectrum)
            sigma_guess = 0.005
 
            p0 = [a1_guess, x0_1_guess, sigma_guess, a2_guess, x0_2_guess, sigma_guess, a3_guess, x0_3_guess, sigma_guess, offset_guess]
            bounds = ([0, 1.39, 0.001, 0, 1.43, 0.001, 0, 1.45, 0.001, -np.inf],
                      [np.inf, 1.42, 0.05, np.inf, 1.45, 0.05, np.inf, 1.48, 0.05, np.inf])
 
            popt, _ = curve_fit(triple_gaussian, energy, line_cut_spectrum, p0=p0, bounds=bounds, maxfev=8000)
            fit_successful = True
        except (RuntimeError, ValueError) as e:
            print(f"\nFit failed for line cut at {actual_v:.3f}V: {e}")
 
        # --- Plot the results in a new window ---
        plt.figure(figsize=(10, 6))
        plt.plot(energy, line_cut_spectrum, 'o', label=f'Data at {actual_v:.3f} V', markersize=4)
 
        if fit_successful:
            # Plot the total fit
            plt.plot(energy, triple_gaussian(energy, *popt), '-', color='red', label='Total Fit')
           
            # Plot individual components
            offset = popt[9]
            peak1 = gaussian(energy, popt[0], popt[1], popt[2], offset) - offset
            peak2 = gaussian(energy, popt[3], popt[4], popt[5], offset) - offset
            peak3 = gaussian(energy, popt[6], popt[7], popt[8], offset) - offset
            plt.plot(energy, peak1 + offset, '--', color='green', label='Peak 1')
            plt.plot(energy, peak2 + offset, '--', color='purple', label='Peak 2')
            plt.plot(energy, peak3 + offset, '--', color='orange', label='Peak 3')
 
            # --- Print Fit Analysis ---
            fwhm_factor = 2 * np.sqrt(2 * np.log(2))
            fwhm1 = fwhm_factor * popt[2]
            fwhm2 = fwhm_factor * popt[5]
            fwhm3 = fwhm_factor * popt[8]
            peak1_center = popt[1]
            peak2_center = popt[4]
            peak3_center = popt[7]
 
            print(f"\n--- Fitting Analysis for Line Cut at {actual_v:.3f} V ---")
            print(f"  Peak 1 Center: {peak1_center:.4f} eV | FWHM: {fwhm1:.4f} eV")
            print(f"  Peak 2 Center: {peak2_center:.4f} eV | FWHM: {fwhm2:.4f} eV")
            print(f"  Peak 3 Center: {peak3_center:.4f} eV | FWHM: {fwhm3:.4f} eV")
 
        plt.xlabel('Photon Energy (eV)')
        plt.ylabel('Intensity (a.u.)')
        plt.title(f'Fitted Line Cut at {actual_v:.3f} V for {os.path.basename(file_path)}')
        plt.legend()
        plt.grid(True)
        plt.xlim(energy_min, energy_max)
        plt.tight_layout()
        plt.show(block=False)
 
    except Exception as e:
        print(f'An error occurred while plotting {file_path}: {e}')
 
plt.ioff()
 
# Keep all plot windows open until user presses Enter
input("\nPress Enter to close all plots...")