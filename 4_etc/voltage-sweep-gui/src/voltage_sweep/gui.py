from tkinter import Tk, filedialog, messagebox
import os
import sys
from voltage_sweep.Voltage_sweep import main_functionality  # Adjust this import based on the actual function name

def choose_files():
    root = Tk()
    root.withdraw()  # Hide the root window
    file_list = filedialog.askopenfilenames(
        title="Select HDF5 files for analysis",
        filetypes=(("HDF5 files", "*.h5"), ("All files", "*.*"))
    )
    root.destroy()
    return file_list

def main():
    files = choose_files()
    if not files:
        messagebox.showwarning("No files selected", "Please select at least one HDF5 file.")
        sys.exit()

    # Call the main functionality from Voltage_sweep.py with the selected files
    main_functionality(files)

if __name__ == "__main__":
    main()