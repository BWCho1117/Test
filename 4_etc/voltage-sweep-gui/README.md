# Voltage Sweep GUI

This project provides a graphical user interface (GUI) for selecting and analyzing HDF5 files related to voltage sweep data. The application allows users to choose files and then processes them to generate various plots and analyses.

## Project Structure

```
voltage-sweep-gui
├── src
│   ├── main.py               # Entry point of the application
│   └── voltage_sweep
│       ├── __init__.py       # Marks the voltage_sweep directory as a package
│       ├── gui.py            # GUI implementation for file selection
│       └── Voltage_sweep.py   # Main functionality for processing files
├── requirements.txt           # Lists project dependencies
├── pyproject.toml            # Project configuration and metadata
└── README.md                 # Project documentation
```

## Requirements

To run this project, you need to install the following dependencies:

- h5py
- numpy
- matplotlib
- tkinter

You can install the required packages using pip:

```
pip install -r requirements.txt
```

## Usage

1. Run the application by executing the `main.py` file:

   ```
   python src/main.py
   ```

2. A GUI will appear, allowing you to select HDF5 files for analysis.

3. After selecting the files, the application will process the data and generate the corresponding plots.

### Differential Reflectance Analysis (Raman reflectance text files)

You can compute differential reflectance spectra from pairs of plain text files containing two columns (x, reflectance). Provide a *sample* file (e.g. `1L.txt`) and a *substrate/background* file (e.g. `1L_sub.txt`). The differential reflectance is computed as:

\[ \Delta R / R_{sub} = (R_{sample} - R_{sub}) / R_{sub} \]

The script auto-detects pairs in a directory using the naming pattern `<stem>.txt` and `<stem>_sub.txt`.

Run from PowerShell:

```powershell
python -m src.diff_reflectance --data-dir "C:\\path\\to\\20231123_PtS2_R" --xunit auto --outdir outputs\\diff-reflectance
```

Or explicitly list pairs:

```powershell
python -m src.diff_reflectance --pair "C:\\...\\1L.txt=C:\\...\\1L_sub.txt" --pair "C:\\...\\2L.txt=C:\\...\\2L_sub.txt" --xunit nm
```

Arguments:
- `--data-dir` Directory containing files to auto-pair.
- `--pair` Explicit `sample=substrate` specification (repeatable).
- `--xunit` One of `auto`, `nm`, `cm-1`, `eV` (default `auto`).
- `--outdir` Output directory (default `outputs/diff-reflectance`).
- `--show` Show interactive matplotlib windows.

Outputs:
- `differential_reflectance_all.png` combined plot
- One CSV per pair: `<label>_diff_reflectance.csv` (Energy_eV, DeltaR_over_Rsub)
- Individual PNG plots per pair

If auto unit guessing is wrong (e.g., wavelength mistaken for wavenumber), force the correct unit with `--xunit`.

Edge cases handled:
- Division by near-zero substrate values is masked as NaN.
- Non-overlapping energy regions omitted from plots.
- Out-of-range interpolation avoided (NaN outside source range).

## License

This project is licensed under the MIT License. See the LICENSE file for more details.