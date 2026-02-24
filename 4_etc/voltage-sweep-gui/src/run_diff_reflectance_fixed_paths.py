"""Quick script to compute differential reflectance for the provided 1L-4L PtS2 files.

Edit the FILES list below if paths change. Requires numpy & matplotlib.

Run (PowerShell):
  python src/run_diff_reflectance_fixed_paths.py
"""

from __future__ import annotations

import os
from typing import List, Tuple

from diff_reflectance import load_txt_spectrum, compute_differential, infer_pair_label, plot_and_save


# Raw absolute paths provided by user (adjust if capitalization differs)
FILES: List[Tuple[str, str]] = [
    (r"C:\\Users\\bwcho\\OneDrive\\Finished\\3. VP low frequency Raman\\251110_duhee_ULF raman data\\Byeongwook\\20231123_PtS2_R\\1L.txt",
     r"C:\\Users\\bwcho\\OneDrive\\Finished\\3. VP low frequency Raman\\251110_duhee_ULF raman data\\Byeongwook\\20231123_PtS2_R\\1L_sub.txt"),
    (r"C:\\Users\\bwcho\\OneDrive\\Finished\\3. VP low frequency Raman\\251110_duhee_ULF raman data\\Byeongwook\\20231123_PtS2_R\\2L.txt",
     r"C:\\Users\\bwcho\\OneDrive\\Finished\\3. VP low frequency Raman\\251110_duhee_ULF raman data\\Byeongwook\\20231123_PtS2_R\\2L_sub.txt"),
    (r"C:\\Users\\bwcho\\OneDrive\\Finished\\3. VP low frequency Raman\\251110_duhee_ULF raman data\\Byeongwook\\20231123_PtS2_R\\3L.txt",
     r"C:\\Users\\bwcho\\OneDrive\\Finished\\3. VP low frequency Raman\\251110_duhee_ULF raman data\\Byeongwook\\20231123_PtS2_R\\3L_sub.txt"),
    (r"C:\\Users\\bwcho\\OneDrive\\Finished\\3. VP low frequency Raman\\251110_duhee_ULF raman data\\Byeongwook\\20231123_PtS2_R\\4L.txt",
     r"C:\\Users\\bwcho\\OneDrive\\Finished\\3. VP low frequency Raman\\251110_duhee_ULF raman data\\Byeongwook\\20231123_PtS2_R\\4L_Sub.txt"),
]


def main() -> None:
    results = []
    for sample_path, sub_path in FILES:
        if not (os.path.isfile(sample_path) and os.path.isfile(sub_path)):
            print(f"Skipping pair (missing file):\n  {sample_path}\n  {sub_path}")
            continue
        sample_spec = load_txt_spectrum(sample_path, xunit="auto")
        sub_spec = load_txt_spectrum(sub_path, xunit="auto")
        E, DR = compute_differential(sample_spec, sub_spec)
        label = infer_pair_label(sample_path)
        results.append((label, E, DR))

    if not results:
        print("No valid pairs processed. Check file paths.")
        return

    outdir = os.path.join("outputs", "diff-reflectance", "fixed-paths")
    plot_and_save(results, outdir, show=False)
    print(f"Saved differential reflectance outputs to: {os.path.abspath(outdir)}")


if __name__ == "__main__":
    main()
