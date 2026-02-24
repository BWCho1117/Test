import tkinter as tk
from tkinter import filedialog
from voltage_sweep.Voltage_sweep import main_functionality  # Assuming main_functionality is the function to run the analysis

def choose_files():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_list = filedialog.askopenfilenames(
        title="Select HDF5 files for analysis",
        filetypes=(("HDF5 files", "*.h5"), ("All files", "*.*"))
    )
    return file_list

def main():
    files = choose_files()
    if files:
        main_functionality(files)  # Call the main functionality with the selected files

if __name__ == "__main__":
    main()