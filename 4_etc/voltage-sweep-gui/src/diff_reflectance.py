"""
Differential reflectance computation and plotting.

Given pairs of reflectance spectra (sample and substrate), compute
  DR = (R_sample - R_substrate) / R_substrate
and plot/export in the energy domain (eV).

Usage examples (PowerShell):
  # Auto-pair files in a directory (expects names like 1L.txt and 1L_sub.txt)
  python -m src.diff_reflectance --data-dir "C:\\path\\to\\folder" --xunit auto --outdir outputs\\diff-reflectance

  # Explicit pairs
  python -m src.diff_reflectance --pair "C:\\...\\1L.txt=C:\\...\\1L_sub.txt" --pair "C:\\...\\2L.txt=C:\\...\\2L_sub.txt"

Assumptions:
- Input text files contain two columns: x (wavelength nm, wavenumber cm^-1, or energy eV), y (reflectance).
- Delimiters can be spaces or tabs. Lines starting with '#' are ignored.

If unit detection is ambiguous, pass --xunit explicitly.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# Types and small utilities
# -------------------------

@dataclass
class Spectrum:
    x_raw: np.ndarray  # original x-axis
    y: np.ndarray      # reflectance values
    xunit: str         # 'nm' | 'cm-1' | 'eV'
    path: str

    def to_energy_sorted(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert x axis to energy (eV) and return arrays sorted ascending in energy.

        Returns
        -------
        E_sorted : np.ndarray
            Energy in eV, ascending order.
        y_sorted : np.ndarray
            y values corresponding to E_sorted.
        """
        E = convert_to_energy(self.x_raw, self.xunit)
        # Sort ascending by E
        order = np.argsort(E)
        return E[order], self.y[order]


def convert_to_energy(x: np.ndarray, xunit: str) -> np.ndarray:
    """Convert x-axis to energy in eV for given unit.

    Supported units:
      - 'nm'   : E[eV] = 1240 / λ[nm]
      - 'cm-1' : E[eV] = ν~[cm^-1] / 8065.54429
      - 'eV'   : identity
    """
    unit = xunit.lower().strip()
    if unit in ("ev",):
        return np.asarray(x, dtype=float)
    if unit in ("nm",):
        with np.errstate(divide="ignore", invalid="ignore"):
            E = 1240.0 / np.asarray(x, dtype=float)
        return E
    if unit in ("cm-1", "cm^-1", "wavenumber"):
        return np.asarray(x, dtype=float) / 8065.54429
    raise ValueError(f"Unsupported xunit: {xunit}")


def guess_xunit(x: np.ndarray) -> str:
    """Heuristically guess x unit from value ranges.

    Rules (crude but practical):
      - If median is between 0.5 and 6.0 and max < 10 -> 'eV'
      - Else if all values in [100, 4000] -> 'nm' (typical UV-VIS-NIR wavelength)
      - Else if max > 5000 or median in [50, 10000] -> 'cm-1'
      - Fallback: if max <= 12 -> 'eV', else 'nm'
    """
    x = np.asarray(x, dtype=float)
    xs = x[np.isfinite(x)]
    if xs.size == 0:
        return "eV"
    med = float(np.median(xs))
    x_max = float(np.nanmax(xs))
    x_min = float(np.nanmin(xs))

    if 0.5 <= med <= 6.0 and x_max < 10.0:
        return "eV"
    if 100.0 <= x_min and x_max <= 4000.0:
        return "nm"
    if x_max > 5000.0 or 50.0 <= med <= 10000.0:
        return "cm-1"
    return "eV" if x_max <= 12.0 else "nm"


def load_txt_spectrum(path: str, xunit: str = "auto") -> Spectrum:
    """Load a two-column text spectrum.

    Ignores comment lines starting with '#'. Accepts space or tab delimiters.
    """
    data = np.loadtxt(path, comments="#", dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"File {path} does not appear to have at least two columns.")
    x = data[:, 0]
    y = data[:, 1]
    unit = guess_xunit(x) if xunit == "auto" else xunit
    return Spectrum(x_raw=x, y=y, xunit=unit, path=path)


def safe_interp(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    """Interpolate y_src(x_src) onto x_tgt using linear interpolation.

    Values outside the source range will be set to NaN to avoid extrapolation artifacts.
    """
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_tgt = np.asarray(x_tgt, dtype=float)

    # Mask out-of-range
    y_out = np.full_like(x_tgt, np.nan, dtype=float)
    in_range = (x_tgt >= np.nanmin(x_src)) & (x_tgt <= np.nanmax(x_src))
    y_out[in_range] = np.interp(x_tgt[in_range], x_src, y_src)
    return y_out


def compute_differential(sample: Spectrum, substrate: Spectrum, denom_eps: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """Compute differential reflectance in eV grid of the sample.

    Returns energy (eV) and DR array with NaN outside overlap or where substrate ~ 0.
    """
    Es, Ys = sample.to_energy_sorted()
    Eu, Yu = substrate.to_energy_sorted()

    Yu_on_Es = safe_interp(Eu, Yu, Es)
    # Avoid division by tiny numbers
    denom = Yu_on_Es.copy()
    denom[np.isfinite(denom) & (np.abs(denom) < denom_eps)] = np.nan
    DR = (Ys - Yu_on_Es) / denom
    return Es, DR


def infer_pair_label(sample_path: str) -> str:
    """Infer a short label like '1L' from filename.

    Looks for tokens ending with 'L' (e.g., 1L, 2L). Fallback to basename without extension.
    """
    base = os.path.splitext(os.path.basename(sample_path))[0]
    m = re.search(r"(\d+L)", base, flags=re.IGNORECASE)
    return m.group(1) if m else base


def find_pairs_in_dir(data_dir: str) -> List[Tuple[str, str]]:
    """Find (sample, substrate) pairs in a directory by filename convention '*_sub.*'.

    For every '<stem>.txt' and '<stem>_sub.txt', produce a pair (sample, substrate).
    Case-insensitive for '_sub'.
    """
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    # Map stem (lowercased) -> variants
    stems: dict[str, dict[str, str]] = {}
    for fname in files:
        name_lower = fname.lower()
        if name_lower.endswith(".txt"):
            if name_lower.endswith("_sub.txt"):
                stem = name_lower[:-8]  # remove '_sub.txt'
                stems.setdefault(stem, {})['sub'] = fname
            else:
                stem = name_lower[:-4]  # remove '.txt'
                stems.setdefault(stem, {})['sample'] = fname

    pairs: List[Tuple[str, str]] = []
    for stem, parts in stems.items():
        if 'sample' in parts and 'sub' in parts:
            pairs.append(
                (
                    os.path.join(data_dir, parts['sample']),
                    os.path.join(data_dir, parts['sub']),
                )
            )
    return pairs


def plot_and_save(pairs_results: List[Tuple[str, np.ndarray, np.ndarray]], outdir: str, show: bool = False) -> None:
    """Plot all pairs on one figure and also save individual plots and CSVs.

    pairs_results: list of (label, E_eV, DR)
    """
    os.makedirs(outdir, exist_ok=True)

    # Combined figure
    plt.figure(figsize=(8, 5))
    for label, E, DR in pairs_results:
        mask = np.isfinite(E) & np.isfinite(DR)
        if np.any(mask):
            plt.plot(E[mask], DR[mask], label=label)
    plt.xlabel("Energy (eV)")
    plt.ylabel("Differential Reflectance (ΔR/R_sub)")
    plt.title("Differential Reflectance Spectra")
    plt.legend()
    plt.tight_layout()
    combined_png = os.path.join(outdir, "differential_reflectance_all.png")
    plt.savefig(combined_png, dpi=200)
    if show:
        plt.show()
    plt.close()

    # Individual saves
    for label, E, DR in pairs_results:
        mask = np.isfinite(E) & np.isfinite(DR)
        if not np.any(mask):
            continue
        arr = np.column_stack([E[mask], DR[mask]])
        csv_path = os.path.join(outdir, f"{label}_diff_reflectance.csv")
        np.savetxt(csv_path, arr, delimiter=",", header="Energy_eV,DeltaR_over_Rsub", comments="")

        plt.figure(figsize=(7, 4))
        plt.plot(E[mask], DR[mask], lw=1.2)
        plt.xlabel("Energy (eV)")
        plt.ylabel("Differential Reflectance (ΔR/R_sub)")
        plt.title(f"Differential Reflectance: {label}")
        plt.tight_layout()
        png_path = os.path.join(outdir, f"{label}_diff_reflectance.png")
        plt.savefig(png_path, dpi=200)
        plt.close()


def parse_pair_arg(pair_str: str) -> Tuple[str, str]:
    if "=" not in pair_str:
        raise argparse.ArgumentTypeError("--pair must be in the form 'sample_path=substrate_path'")
    sample, sub = pair_str.split("=", 1)
    return sample.strip().strip('"'), sub.strip().strip('"')


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute differential reflectance spectra (energy domain)")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory to auto-detect pairs like <stem>.txt and <stem>_sub.txt")
    parser.add_argument("--pair", type=parse_pair_arg, action="append", default=None, help="Explicit pair 'sample=sub'. Can be repeated.")
    parser.add_argument("--xunit", type=str, default="auto", choices=["auto", "nm", "cm-1", "eV"], help="Unit of x-axis in input files")
    parser.add_argument("--outdir", type=str, default=os.path.join("outputs", "diff-reflectance"), help="Output directory for plots and CSVs")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    args = parser.parse_args(argv)

    pairs: List[Tuple[str, str]] = []
    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            raise SystemExit(f"--data-dir not found: {args.data_dir}")
        pairs.extend(find_pairs_in_dir(args.data_dir))
    if args.pair:
        pairs.extend(args.pair)

    if not pairs:
        parser.error("No (sample, substrate) pairs found. Provide --data-dir or --pair.")

    results: List[Tuple[str, np.ndarray, np.ndarray]] = []
    for sample_path, sub_path in pairs:
        sample_spec = load_txt_spectrum(sample_path, xunit=args.xunit)
        sub_spec = load_txt_spectrum(sub_path, xunit=args.xunit)
        E, DR = compute_differential(sample_spec, sub_spec)
        label = infer_pair_label(sample_path)
        results.append((label, E, DR))

    plot_and_save(results, args.outdir, show=args.show)
    print(f"Saved outputs to: {os.path.abspath(args.outdir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
