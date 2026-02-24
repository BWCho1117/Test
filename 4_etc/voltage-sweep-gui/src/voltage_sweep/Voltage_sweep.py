import h5py
import numpy as np
import matplotlib
import os
import matplotlib.pyplot as plt
from voltage_sweep.gui import choose_files_and_options

def main():
    # Select files using the GUI
    file_list, opts = choose_files_and_options()
    if not file_list or opts is None:
        print("No files selected or operation cancelled.")
        return

    plt.ion()

    for file_path in file_list:
        try:
            with h5py.File(file_path, 'r') as h5_file:
                spectra = h5_file['spectro_data'][:, 0, :]
                wavelengths = h5_file['spectro_wavelength'][:]
                voltages = h5_file['xPositions'][:]

            photon_energy = 1240.0 / wavelengths  # eV
            plot_data = spectra[::-1, ::-1]

            # Additional processing and plotting logic goes here...

        except Exception as e:
            print(f'An error occurred while processing {file_path}: {e}')

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()