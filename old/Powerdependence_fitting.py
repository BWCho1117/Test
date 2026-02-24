import os, re, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# 재사용: SPE 로더만 기존 파일에서 import (PyQt 의존 없음)
from Powerdependence_20250916 import SPE3read

# --------- line shapes ---------
def gaussian(x, amp, cen, wid):  # wid = sigma
    return amp * np.exp(-((x - cen) ** 2) / (2 * wid ** 2))

def lorentzian(x, amp, cen, wid):  # wid = HWHM
    return amp * (wid**2) / ((x - cen) ** 2 + wid**2)

def multi_model(x, params, model, offset):
    fn = gaussian if model.lower().startswith("g") else lorentzian
    y = np.full_like(x, float(offset), dtype=float)
    for i in range(0, len(params), 3):
        amp, cen, wid = params[i:i+3]
        y += fn(x, amp, cen, wid)
    return y

# --------- helpers ---------
def parse_power_and_exposure(path):
    base = os.path.splitext(os.path.basename(path))[0]
    parts = base.split('_')
    power_txt = parts[0]
    if power_txt.endswith('uW'):
        try: p_uW = float(power_txt[:-2])
        except: p_uW = 0.0
    elif power_txt.endswith('nW'):
        try: p_uW = float(power_txt[:-2]) / 1000.0
        except: p_uW = 0.0
    else:
        try: p_uW = float(power_txt)
        except: p_uW = 0.0
    expo = None
    if len(parts) >= 2 and parts[1].endswith('s'):
        try: expo = float(parts[1][:-1])
        except: expo = None
    return p_uW, expo, power_txt

def load_spectrum(path):
    spe = SPE3read(path)
    if spe.data is None or spe.data.size == 0:
        raise RuntimeError(f"No data in {path}")
    y = spe.data[0, 0, :].astype(float)
    if spe.wavelength is not None and np.all(np.asarray(spe.wavelength) > 0):
        x = 1240.0 / np.asarray(spe.wavelength)  # eV
        is_energy = True
    else:
        x = np.arange(y.size, dtype=float)
        is_energy = False
    return x, y, is_energy

def auto_initial_guesses(x, y, n_peaks, model, bg_offset=0.0):
    y_use = y - bg_offset
    y_s = gaussian_filter1d(y_use, sigma=max(1, int(len(y)*0.003)))  # 약간 스무딩
    # prominence 기준으로 상위 n개 피크
    peaks, props = find_peaks(y_s, prominence=np.max(y_s)*0.02 if np.max(y_s) > 0 else 0.0)
    if peaks.size == 0:
        # 단조 데이터면 중앙에 하나
        peaks = np.array([np.argmax(y_s)])
    # width 추정(FWHM)
    widths_res = peak_widths(y_s, peaks, rel_height=0.5)
    fwhm_pts = widths_res[0]
    # 상위 n개
    order = np.argsort(y_s[peaks])[::-1][:max(1, n_peaks)]
    peaks = peaks[order]
    fwhm_pts = fwhm_pts[order]

    # x 간격 -> FWHM(eV 또는 index)
    dx = np.mean(np.diff(x)) if len(x) > 1 else 1.0
    fwhm = np.clip(fwhm_pts * dx, dx*1.2, (x.max()-x.min())*0.3)

    p0 = []
    bounds_lo, bounds_hi = [], []
    x_rng = x.max() - x.min()
    for i, pk in enumerate(peaks):
        amp0 = max(y_s[pk], 0.0)
        cen0 = float(x[pk])
        wid0 = float(fwhm[i] / (2.354820045)) if model.lower().startswith("g") else float(fwhm[i]/2.0)
        if wid0 <= 0 or not np.isfinite(wid0):
            wid0 = x_rng * 0.005
        p0.extend([amp0, cen0, wid0])

        bounds_lo.extend([0.0, cen0 - x_rng*0.05, max(wid0*0.2, dx*0.2)])
        bounds_hi.extend([amp0*5.0 + 1e-12, cen0 + x_rng*0.05, wid0*5.0])

    return np.array(p0, float), (np.array(bounds_lo, float), np.array(bounds_hi, float))

def fit_one(x, y, model="Gaussian", n_peaks=1, bg_offset=0.0):
    p0, bounds = auto_initial_guesses(x, y, n_peaks, model, bg_offset)
    try:
        popt, _ = curve_fit(lambda _x, *pp: multi_model(_x, pp, model, bg_offset),
                            x, y, p0=p0, bounds=bounds, maxfev=20000)
    except Exception as e:
        return None, str(e)

    res = []
    is_gauss = model.lower().startswith("g")
    for i in range(0, len(popt), 3):
        amp, cen, wid = popt[i:i+3]
        if is_gauss:
            fwhm = 2.354820045 * wid
            integ = amp * wid * np.sqrt(2*np.pi)
        else:
            fwhm = 2.0 * wid
            integ = amp * np.pi * wid
        res.append(dict(peak_id=i//3 + 1, position=float(cen), fwhm=float(fwhm), integrated_intensity=float(integ)))
    return res, None

def process_files(pattern, model="Gaussian", n_peaks=1, bg=0.0, normalize=False, correction=0.0,
                  smooth_sigma=0.0, out_csv="autofit_results.csv", save_plots=None):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files matched: {pattern}")

    rows = []
    for f in files:
        try:
            power_uW, expo, power_label = parse_power_and_exposure(f)
            x, y, is_energy = load_spectrum(f)

            y2 = y.astype(float).copy()
            if bg:
                y2 -= bg
            if normalize and expo and expo > 0:
                y2 = y2 / expo
            if correction:
                y2 = y2 + correction
            if smooth_sigma and smooth_sigma > 0:
                y2 = gaussian_filter1d(y2, sigma=smooth_sigma)

            results, err = fit_one(x, y2, model=model, n_peaks=n_peaks, bg_offset=0.0)
            if err:
                print(f"[WARN] fit failed for {os.path.basename(f)}: {err}")
                continue

            for r in results:
                rows.append({
                    "File": os.path.basename(f),
                    "Power (uW)": power_uW,
                    "Peak_ID": r["peak_id"],
                    "Position (eV)" if is_energy else "Position (px)": r["position"],
                    "FWHM (eV)" if is_energy else "FWHM (px)": r["fwhm"],
                    "Integrated_Intensity": r["integrated_intensity"],
                })

            if save_plots:
                plt.figure(figsize=(7,4))
                plt.plot(x, y2, 'k-', lw=1, label="data")
                # 재현 곡선
                pp = []
                for r in results:
                    if model.lower().startswith("g"):
                        wid = r["fwhm"]/2.354820045
                        yp = gaussian(x, r["integrated_intensity"]/(wid*np.sqrt(2*np.pi)), r["position"], wid)  # amp 복원 어려움 → 합성 곡선만
                    else:
                        wid = r["fwhm"]/2.0
                        amp = r["integrated_intensity"]/(np.pi*wid)
                        yp = lorentzian(x, amp, r["position"], wid)
                    plt.plot(x, yp, '--', lw=1)
                plt.title(f"{os.path.basename(f)}  [{model}]")
                plt.xlabel("Energy (eV)" if is_energy else "Pixel")
                plt.ylabel("Intensity (a.u.)")
                os.makedirs(save_plots, exist_ok=True)
                plt.tight_layout()
                plt.savefig(os.path.join(save_plots, os.path.splitext(os.path.basename(f))[0] + ".png"), dpi=150)
                plt.close()

        except Exception as e:
            print(f"[ERROR] {os.path.basename(f)}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df.sort_values(["Power (uW)", "Peak_ID"], inplace=True)
        df.to_csv(out_csv, index=False, float_format="%.6f")
        print(f"Saved: {os.path.abspath(out_csv)}  (rows={len(df)})")
    else:
        print("No successful fits.")

def main():
    ap = argparse.ArgumentParser(description="Batch auto-fitting for SPE spectra.")
    ap.add_argument("--input", required=True, help="Glob pattern for SPE files, e.g. 'D:/data/*.spe'")
    ap.add_argument("--model", choices=["Gaussian","Lorentzian"], default="Gaussian")
    ap.add_argument("--peaks", type=int, default=1, help="Number of peaks to fit per spectrum")
    ap.add_argument("--bg", type=float, default=0.0, help="Background (counts) to subtract before fit")
    ap.add_argument("--normalize", action="store_true", help="Normalize by exposure time if encoded in filename")
    ap.add_argument("--correction", type=float, default=0.0, help="Add constant correction after normalization")
    ap.add_argument("--smooth", type=float, default=0.0, help="Gaussian sigma for 1D smoothing before peak finding")
    ap.add_argument("--out", default="autofit_results.csv", help="Output CSV path")
    ap.add_argument("--save-plots", default=None, help="Folder to save per-spectrum PNGs")
    args = ap.parse_args()

    process_files(args.input, model=args.model, n_peaks=args.peaks, bg=args.bg,
                  normalize=args.normalize, correction=args.correction,
                  smooth_sigma=args.smooth, out_csv=args.out, save_plots=args.save_plots)

if __name__ == "__main__":
    main()