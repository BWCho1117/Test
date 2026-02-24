import h5py
import numpy as np

# List of files in order: 10^8 → 10^4 (v=1 set)
h5_paths = [
    r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\v=1\10_8_Power_dependancy at 0B_angle117.5_4923_5s_v=1(2.2_2.2)_0.01V-0.95V_step0.025V_02-23_16-54.h5",
    r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\v=1\10_7_Power_dependancy at 0B_angle117.5_4923_5s_v=1(2.2_2.2)_0.01V-0.95V_step0.025V_02-23_16-48.h5",
    r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\v=1\10_6_Power_dependancy at 0B_angle117.5_4923_5s_v=1(2.2_2.2)_0.01V-0.95V_step0.025V_02-23_16-43.h5",
    r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\v=1\10_5_Power_dependancy at 0B_angle117.5_4923_5s_v=1(2.2_2.2)_0.01V-0.95V_step0.025V_02-23_16-38.h5",
    r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\v=1\10_4_Power_dependancy at 0B_angle117.5_4923_5s_v=1(2.2_2.2)_0.01V-0.95V_step0.025V_02-23_16-33.h5",
]
power_ranges = [
    (0.0008, 0.008),   # 10^8: 0.8 nW to 8 nW (μW)
    (0.008, 0.08),     # 10^7: 8 nW to 80 nW (μW)
    (0.08, 0.8),       # 10^6: 0.08 μW to 0.8 μW
    (0.8, 8),          # 10^5: 0.8 μW to 8 μW
    (8, 100),          # 10^4: 8 μW to 100 μW
]

all_spectra = []
all_power = []
wavelength = None
for i, f in enumerate(h5_paths):
    with h5py.File(f, "r") as h5:
        data = np.array(h5["/spectro_data"]).squeeze()
        wvl = np.array(h5["/spectro_wavelength"]).squeeze()
        n_spec = data.shape[0]
        min_power, max_power = power_ranges[i]
        power_axis = np.linspace(min_power, max_power, n_spec)
        all_spectra.append(data)
        all_power.append(power_axis)
        if wavelength is None:
            wavelength = wvl

combined_spectra = np.vstack(all_spectra)
combined_power = np.concatenate(all_power)

# Save combined data to new .h5 file
out_path = r"C:\Users\ti2006\OneDrive - Heriot-Watt University\Downloads\v=1\Combined_Powerdependence_v1.h5"
with h5py.File(out_path, "w") as h5:
    h5.create_dataset("spectro_data", data=combined_spectra)
    h5.create_dataset("spectro_wavelength", data=wavelength)
    h5.create_dataset("power_axis", data=combined_power)
print(f"Combined file saved to: {out_path}")
