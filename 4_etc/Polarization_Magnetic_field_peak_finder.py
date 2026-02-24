from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

HC_EV_NM = 1239.841984  # eV*nm
MAX_PEAKS = 4


# =============================
# Data load
# =============================
def load_h5_spectro(h5_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      B_T: (N_B,)
      E_eV: (N_E,) ascending
      I: (N_B, N_E)
    """
    with h5py.File(h5_path, "r") as f:
        if "spectro_wavelength" not in f or "spectro_data" not in f:
            raise KeyError(f"Missing spectro_wavelength/spectro_data. Keys={list(f.keys())}")

        wl_nm = np.asarray(f["spectro_wavelength"][:]).squeeze()
        data = np.asarray(f["spectro_data"][:])

        if data.ndim == 3:
            I = data[:, 0, :]
        elif data.ndim == 2:
            I = data
        else:
            raise ValueError(f"Unexpected spectro_data shape: {data.shape}")

        if "yLims" in f and "yStep" in f:
            yLims = np.asarray(f["yLims"][:]).squeeze()
            yStep = float(np.asarray(f["yStep"][()]).squeeze())
            N = I.shape[0]
            B = float(yLims[0]) + np.arange(N, dtype=float) * yStep
        elif "yPositions" in f:
            B = np.asarray(f["yPositions"][:]).squeeze()
        else:
            raise KeyError("No yLims/yStep or yPositions found for B axis.")

    E = HC_EV_NM / wl_nm.astype(float)
    idx = np.argsort(E)
    E = E[idx]
    I = I[:, idx]
    return B.astype(float), E.astype(float), I.astype(float)


# =============================
# Helpers (local peak finding)
# =============================
def _odd_clamp(n: int, lo: int, hi: int) -> int:
    n = int(max(lo, min(hi, n)))
    if n % 2 == 0:
        n = n - 1 if n > lo else n + 1
    return int(max(lo, min(hi, n)))


def clamp_to_bounds(p0: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    p0 = np.asarray(p0, float)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    span = np.maximum(ub - lb, 1e-12)
    eps = 1e-9 * span
    return np.minimum(np.maximum(p0, lb + eps), ub - eps)


def gaussian_c0(x: np.ndarray, A: float, mu: float, sigma: float, c0: float) -> np.ndarray:
    x = np.asarray(x, float)
    sigma = float(sigma)
    return float(c0) + float(A) * np.exp(-0.5 * ((x - float(mu)) / sigma) ** 2)


def polyfit_bg_from_edges(x: np.ndarray, y: np.ndarray, order: int, edge_frac: float = 0.2) -> list[float]:
    """
    Fit polynomial background using only the left/right edges of the window.
    Returns coeffs [c0,c1,c2...] such that bg(x)=sum c_k x^k
    """
    if order < 0:
        return []
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < 8:
        return [float(np.median(y))] + [0.0] * order

    edge_frac = float(edge_frac)
    edge_frac = max(0.05, min(0.45, edge_frac))
    m = max(4, int(edge_frac * n))
    idx = np.r_[0:m, n - m : n]

    xx = x[idx]
    yy = y[idx]
    deg = int(order)

    coeff_hi = np.polyfit(xx, yy, deg=deg)  # highest power first
    # convert to [c0,c1,c2...]
    coeff_lo = list(reversed([float(c) for c in coeff_hi]))
    # pad to exact length
    while len(coeff_lo) < deg + 1:
        coeff_lo.append(0.0)
    return coeff_lo[: deg + 1]


def eval_bg(x: np.ndarray, coeffs: list[float]) -> np.ndarray:
    x = np.asarray(x, float)
    b = np.zeros_like(x, float)
    for pwr, c in enumerate(coeffs):
        b = b + float(c) * (x**pwr)
    return b


def local_peak_mu(
    x: np.ndarray,
    y: np.ndarray,
    *,
    polarity: str = "max",  # "max" or "min"
    smooth_win: int = 11,
    smooth_poly: int = 3,
    refine: str = "parabola",  # "none" | "parabola" | "gaussian"
    gauss_halfwidth_pts: int = 10,
) -> float:
    """
    Find peak position (mu) within a window.
    - smoothing (Savitzky-Golay) optional
    - peak = argmax/argmin depending on polarity
    - refine by 3-point parabola or very local gaussian+c0 fit
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 5:
        raise ValueError("window too small")

    pol = polarity.strip().lower()
    if pol not in ("max", "min"):
        raise ValueError("polarity must be 'max' or 'min'")

    ref = refine.strip().lower()
    if ref not in ("none", "parabola", "gaussian"):
        raise ValueError("refine must be one of: none, parabola, gaussian")

    # smoothing
    win = _odd_clamp(smooth_win, lo=5, hi=int(x.size if x.size % 2 == 1 else x.size - 1))
    poly = int(min(max(2, smooth_poly), win - 2))
    try:
        ys = savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
    except Exception:
        ys = y

    i = int(np.argmax(ys) if pol == "max" else np.argmin(ys))
    mu0 = float(x[i])

    if ref == "none":
        return mu0

    if ref == "parabola":
        if i <= 0 or i >= x.size - 1:
            return mu0
        xx = x[i - 1 : i + 2]
        yy = ys[i - 1 : i + 2]
        a, b, _c = np.polyfit(xx, yy, deg=2)
        if abs(a) < 1e-30:
            return mu0
        muv = -b / (2 * a)
        return float(np.clip(muv, min(xx[0], xx[-1]), max(xx[0], xx[-1])))

    # gaussian refine (very local)
    hw = int(max(3, gauss_halfwidth_pts))
    j0 = max(0, i - hw)
    j1 = min(x.size, i + hw + 1)
    xf = x[j0:j1]
    yf = y[j0:j1]
    if xf.size < 7:
        return mu0

    # for polarity=min, flip so we still fit a "positive" gaussian amplitude robustly
    if pol == "min":
        yf_fit = -yf
    else:
        yf_fit = yf

    A0 = float(np.max(yf_fit) - np.min(yf_fit))
    c00 = float(np.median(yf_fit))
    sig0 = float(max((xf[-1] - xf[0]) / 6.0, 5e-5))

    lb = np.array([0.0, xf[0], 1e-5, np.min(yf_fit) - 10.0 * abs(A0 + 1e-12)], float)
    ub = np.array(
        [abs(A0) * 80.0 + 1e-12, xf[-1], (xf[-1] - xf[0]) * 2.0 + 1e-12, np.max(yf_fit) + 10.0 * abs(A0 + 1e-12)],
        float,
    )
    p0 = np.array([max(A0, 1e-9), mu0, sig0, c00], float)
    p0 = clamp_to_bounds(p0, lb, ub)

    try:
        popt, _ = curve_fit(gaussian_c0, xf, yf_fit, p0=p0, bounds=(lb, ub), maxfev=20000)
        return float(popt[1])
    except Exception:
        # fallback: parabola
        return local_peak_mu(x, y, polarity=polarity, smooth_win=smooth_win, smooth_poly=smooth_poly, refine="parabola")


# =============================
# GUI App (peak positions only)
# =============================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Magnetic-field peak finder (positions only)")
        self.geometry("1250x860")

        # data
        self.h5_path: Path | None = None
        self.B: np.ndarray | None = None
        self.E: np.ndarray | None = None
        self.I: np.ndarray | None = None

        self.i0: int | None = None  # index closest to 0T
        self.E_roi: np.ndarray | None = None
        self.I_roi: np.ndarray | None = None

        # results
        self.B_sorted: np.ndarray | None = None
        self.local_mu_track: np.ndarray | None = None  # (N_B, n_peaks)
        self.local_ok: np.ndarray | None = None        # (N_B,)

        # UI vars
        self.npeaks_var = tk.IntVar(value=3)

        self.roi_emin = tk.StringVar(value="")
        self.roi_emax = tk.StringVar(value="")

        self.pk_emin_vars = [tk.StringVar(value="") for _ in range(MAX_PEAKS)]
        self.pk_emax_vars = [tk.StringVar(value="") for _ in range(MAX_PEAKS)]
        self.mu_out_vars = [tk.StringVar(value="") for _ in range(MAX_PEAKS)]  # output only

        # local options
        self.local_polarity_var = tk.StringVar(value="max")  # max|min
        self.local_smooth_win_var = tk.StringVar(value="11")
        self.local_smooth_poly_var = tk.StringVar(value="3")
        self.local_refine_var = tk.StringVar(value="parabola")  # none|parabola|gaussian
        self.local_gauss_hw_var = tk.StringVar(value="10")

        # baseline options (per-peak window)
        self.bg_model_var = tk.StringVar(value="Linear")  # None|Constant|Linear|Quadratic
        self.bg_edge_frac_var = tk.StringVar(value="0.20")

        self.status = tk.StringVar(value="Open an .h5 file to start.")

        self._build_ui()
        self._update_peak_entry_state()

    # ---------- UI ----------
    def _build_ui(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        # LEFT: scrollable controls
        left_container = ttk.Frame(outer)
        left_container.pack(side="left", fill="y")

        left_canvas = tk.Canvas(left_container, width=460, highlightthickness=0, borderwidth=0)
        left_scroll = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)

        left_scroll.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="y", expand=False)

        left = ttk.Frame(left_canvas, padding=8)
        left_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_left_configure(_event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _on_canvas_configure(event):
            left_canvas.itemconfigure(left_window, width=event.width)

        left.bind("<Configure>", _on_left_configure)
        left_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # RIGHT: plot
        right = ttk.Frame(outer, padding=8)
        right.pack(side="right", fill="both", expand=True)

        # -------- Controls --------
        ttk.Label(left, text="File").pack(anchor="w")
        file_row = ttk.Frame(left)
        file_row.pack(fill="x", pady=(0, 8))
        self.file_lbl = ttk.Label(file_row, text="(no file)", width=52)
        self.file_lbl.pack(side="left", fill="x", expand=True)
        ttk.Button(file_row, text="Open .h5", command=self.on_open).pack(side="left", padx=(6, 0))

        ttk.Separator(left).pack(fill="x", pady=6)

        ttk.Label(left, text="ROI (eV)").pack(anchor="w")
        roi_row = ttk.Frame(left)
        roi_row.pack(fill="x", pady=(0, 8))
        ttk.Label(roi_row, text="Emin").pack(side="left")
        ttk.Entry(roi_row, textvariable=self.roi_emin, width=10).pack(side="left", padx=(4, 10))
        ttk.Label(roi_row, text="Emax").pack(side="left")
        ttk.Entry(roi_row, textvariable=self.roi_emax, width=10).pack(side="left", padx=(4, 0))
        ttk.Button(left, text="Plot 0T (ROI)", command=self.on_plot_0t).pack(fill="x", pady=(0, 8))

        ttk.Separator(left).pack(fill="x", pady=6)

        ttk.Label(left, text="Peak windows (set Emin/Emax for each peak)").pack(anchor="w")
        set_row = ttk.Frame(left)
        set_row.pack(fill="x", pady=(2, 6))
        ttk.Label(set_row, text="#peaks").pack(side="left")
        npeaks = ttk.Combobox(set_row, values=[1, 2, 3, 4], textvariable=self.npeaks_var, width=3, state="readonly")
        npeaks.pack(side="left", padx=(6, 0))
        npeaks.bind("<<ComboboxSelected>>", lambda _e: self._update_peak_entry_state())

        for i in range(MAX_PEAKS):
            row = ttk.Frame(left)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"peak {i+1}").pack(side="left", padx=(0, 6))
            ttk.Label(row, text="Emin").pack(side="left")
            ent1 = ttk.Entry(row, textvariable=self.pk_emin_vars[i], width=9)
            ent1.pack(side="left", padx=(4, 8))
            ttk.Label(row, text="Emax").pack(side="left")
            ent2 = ttk.Entry(row, textvariable=self.pk_emax_vars[i], width=9)
            ent2.pack(side="left", padx=(4, 8))
            ttk.Label(row, text="mu").pack(side="left")
            out = ttk.Entry(row, textvariable=self.mu_out_vars[i], width=12, state="readonly")
            out.pack(side="left", padx=(4, 0))

            setattr(self, f"pk_emin_ent_{i+1}", ent1)
            setattr(self, f"pk_emax_ent_{i+1}", ent2)

        ttk.Separator(left).pack(fill="x", pady=8)

        # Baseline (per window)
        bg_box = ttk.LabelFrame(left, text="Local baseline (per peak window)")
        bg_box.pack(fill="x", pady=(0, 8))

        rbg1 = ttk.Frame(bg_box)
        rbg1.pack(fill="x", pady=2)
        ttk.Label(rbg1, text="Model").pack(side="left")
        ttk.Combobox(
            rbg1,
            values=["None", "Constant", "Linear", "Quadratic"],
            textvariable=self.bg_model_var,
            width=10,
            state="readonly",
        ).pack(side="left", padx=(6, 10))
        ttk.Label(rbg1, text="edge frac").pack(side="left")
        ttk.Entry(rbg1, textvariable=self.bg_edge_frac_var, width=8).pack(side="left", padx=(6, 0))
        ttk.Label(bg_box, text="(baseline is fit using window edges only)", foreground="#444").pack(anchor="w", pady=(2, 0))

        # Local peak finder options
        local_box = ttk.LabelFrame(left, text="Local peak finder options")
        local_box.pack(fill="x", pady=(0, 8))

        r1 = ttk.Frame(local_box)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="polarity").pack(side="left")
        ttk.Combobox(r1, values=["max", "min"], textvariable=self.local_polarity_var, width=6, state="readonly").pack(
            side="left", padx=(6, 12)
        )
        ttk.Label(r1, text="smooth win").pack(side="left")
        ttk.Entry(r1, textvariable=self.local_smooth_win_var, width=6).pack(side="left", padx=(6, 12))
        ttk.Label(r1, text="poly").pack(side="left")
        ttk.Entry(r1, textvariable=self.local_smooth_poly_var, width=4).pack(side="left", padx=(6, 0))

        r2 = ttk.Frame(local_box)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="refine").pack(side="left")
        ttk.Combobox(r2, values=["none", "parabola", "gaussian"], textvariable=self.local_refine_var, width=10, state="readonly").pack(
            side="left", padx=(6, 12)
        )
        ttk.Label(r2, text="gauss HW pts").pack(side="left")
        ttk.Entry(r2, textvariable=self.local_gauss_hw_var, width=6).pack(side="left", padx=(6, 0))

        # Actions
        act = ttk.Frame(left)
        act.pack(fill="x", pady=(4, 8))
        ttk.Button(act, text="Find peaks (0T)", command=self.on_find_peaks_0t_local).pack(side="left", fill="x", expand=True)
        ttk.Button(act, text="Track peaks (all B)", command=self.on_track_peaks_local).pack(side="left", fill="x", expand=True, padx=(6, 0))

        ttk.Button(left, text="Save CSV (local peaks)", command=self.on_save_local_csv).pack(fill="x", pady=(0, 6))

        ttk.Label(left, textvariable=self.status, wraplength=440, foreground="#333").pack(anchor="w", pady=(6, 0))

        # -------- Plot area --------
        self.fig = plt.Figure(figsize=(7.8, 6.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("0T spectrum (ROI)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()

    # ---------- internal reads ----------
    def _require_data(self) -> bool:
        if self.B is None or self.E is None or self.I is None:
            messagebox.showwarning("No data", "Open an .h5 file first.")
            return False
        return True

    def _bg_order(self) -> int:
        m = self.bg_model_var.get().strip()
        return {"None": -1, "Constant": 0, "Linear": 1, "Quadratic": 2}.get(m, 1)

    def _read_peak_ranges(self, n: int) -> list[tuple[float, float]]:
        ranges: list[tuple[float, float]] = []
        for i in range(n):
            s1 = self.pk_emin_vars[i].get().strip()
            s2 = self.pk_emax_vars[i].get().strip()
            if not s1 or not s2:
                raise ValueError(f"peak {i+1}: Emin/Emax required.")
            Emin, Emax = float(s1), float(s2)
            if Emin > Emax:
                Emin, Emax = Emax, Emin
            ranges.append((Emin, Emax))
        return ranges

    def _read_local_opts(self) -> tuple[str, int, int, str, int]:
        pol = self.local_polarity_var.get().strip().lower()
        ref = self.local_refine_var.get().strip().lower()
        try:
            win = int(float(self.local_smooth_win_var.get().strip()))
            poly = int(float(self.local_smooth_poly_var.get().strip()))
            hw = int(float(self.local_gauss_hw_var.get().strip()))
        except Exception:
            raise ValueError("Local options must be numeric: smooth win/poly, gauss HW pts.")
        return pol, win, poly, ref, hw

    def _read_edge_frac(self) -> float:
        try:
            v = float(self.bg_edge_frac_var.get().strip())
        except Exception:
            raise ValueError("edge frac must be numeric (e.g., 0.2).")
        return float(max(0.05, min(0.45, v)))

    def _get_roi_mask(self) -> np.ndarray:
        assert self.E is not None
        Emin_s = self.roi_emin.get().strip()
        Emax_s = self.roi_emax.get().strip()

        if not Emin_s and not Emax_s:
            Emin, Emax = float(self.E.min()), float(self.E.max())
        else:
            Emin = float(Emin_s) if Emin_s else float(self.E.min())
            Emax = float(Emax_s) if Emax_s else float(self.E.max())
            if Emin > Emax:
                Emin, Emax = Emax, Emin

        roi = (self.E >= Emin) & (self.E <= Emax)
        if np.count_nonzero(roi) < 10:
            raise ValueError("ROI too small/empty. Widen Emin/Emax.")
        return roi

    def _update_peak_entry_state(self):
        n = int(self.npeaks_var.get())
        for i in range(MAX_PEAKS):
            state = "normal" if i < n else "disabled"
            getattr(self, f"pk_emin_ent_{i+1}").configure(state=state)
            getattr(self, f"pk_emax_ent_{i+1}").configure(state=state)
            if i >= n:
                self.mu_out_vars[i].set("")

    # ---------- actions ----------
    def on_open(self):
        file_path = filedialog.askopenfilename(
            title="Select an HDF5 file",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")],
        )
        if not file_path:
            return

        try:
            self.h5_path = Path(file_path)
            self.B, self.E, self.I = load_h5_spectro(self.h5_path)
            self.i0 = int(np.argmin(np.abs(self.B)))

            self.E_roi = None
            self.I_roi = None
            self.B_sorted = None
            self.local_mu_track = None
            self.local_ok = None

            for v in self.mu_out_vars:
                v.set("")
        except Exception as e:
            messagebox.showerror("Load error", f"{type(e).__name__}: {e}")
            return

        self.file_lbl.configure(text=self.h5_path.name)
        self.status.set(f"Loaded. N_B={self.I.shape[0]}, N_E={self.I.shape[1]}, B0≈{self.B[self.i0]:.6g} T.")
        self.on_plot_0t()

    def on_plot_0t(self):
        if not self._require_data():
            return
        assert self.B is not None and self.E is not None and self.I is not None and self.i0 is not None

        try:
            roi = self._get_roi_mask()
        except Exception as e:
            messagebox.showerror("ROI error", str(e))
            return

        self.E_roi = self.E[roi]
        self.I_roi = self.I[:, roi]
        y0 = self.I_roi[self.i0, :]

        self.ax.clear()
        self.ax.plot(self.E_roi, y0, lw=1.2, label="0T ROI")
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title(f"0T spectrum (closest B={self.B[self.i0]:.4g} T)")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

        self.status.set("0T plotted. Set per-peak windows, then Find peaks (0T) or Track peaks (all B).")

    def _find_peaks_in_spectrum(self, y: np.ndarray) -> tuple[list[float], list[np.ndarray]]:
        """
        Returns:
          mus_found: list of mu per peak
          bgs_used:  list of bg arrays (each window's bg evaluated on that window grid)
        """
        if self.E_roi is None:
            raise ValueError("ROI missing")
        E = self.E_roi

        n = int(self.npeaks_var.get())
        if n not in (1, 2, 3, 4):
            raise ValueError("n_peaks must be 1..4")

        ranges = self._read_peak_ranges(n)
        bg_order = self._bg_order()
        edge_frac = self._read_edge_frac()
        pol, win, poly, ref, hw = self._read_local_opts()

        mus_found: list[float] = []
        bgs_used: list[np.ndarray] = []

        for i in range(n):
            Emin, Emax = ranges[i]
            m = (E >= Emin) & (E <= Emax)
            if np.count_nonzero(m) < 7:
                raise ValueError(f"peak {i+1}: window too small/empty inside ROI.")

            xw = E[m]
            yw = y[m]

            coeffs = polyfit_bg_from_edges(xw, yw, order=bg_order, edge_frac=edge_frac) if bg_order >= 0 else []
            bgw = eval_bg(xw, coeffs) if bg_order >= 0 else np.zeros_like(xw, float)
            ycorr = yw - bgw

            mu = local_peak_mu(
                xw,
                ycorr,
                polarity=pol,
                smooth_win=win,
                smooth_poly=poly,
                refine=ref,
                gauss_halfwidth_pts=hw,
            )

            mus_found.append(mu)
            bgs_used.append(bgw)

        return mus_found, bgs_used

    def on_find_peaks_0t_local(self):
        if not self._require_data():
            return
        if self.E_roi is None or self.I_roi is None or self.i0 is None:
            messagebox.showwarning("ROI missing", "Plot 0T first.")
            return

        y0 = self.I_roi[self.i0, :]

        try:
            mus, _bgs = self._find_peaks_in_spectrum(y0)
        except Exception as e:
            messagebox.showerror("Find peaks error", str(e))
            return

        for i, mu in enumerate(mus):
            self.mu_out_vars[i].set(f"{mu:.6f}")

        # Plot markers on 0T
        self.ax.clear()
        self.ax.plot(self.E_roi, y0, lw=1.2, label="0T ROI")
        for i, mu in enumerate(mus):
            self.ax.axvline(mu, lw=1.2, ls="--", label=f"peak{i+1} mu={mu:.6f}")
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("0T peak positions (local, windowed)")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

        self.status.set("0T peak positions found (mu updated).")

    def on_track_peaks_local(self):
        if not self._require_data():
            return
        if self.E_roi is None or self.I_roi is None or self.B is None:
            messagebox.showwarning("ROI missing", "Plot 0T first.")
            return

        n = int(self.npeaks_var.get())
        if n not in (1, 2, 3, 4):
            messagebox.showerror("Error", "n_peaks must be 1..4.")
            return

        idxB = np.argsort(self.B)
        B_sorted = self.B[idxB]
        I_sorted = self.I_roi[idxB, :]

        mu_track = np.full((len(B_sorted), n), np.nan, float)
        ok = np.ones(len(B_sorted), dtype=bool)

        self.status.set("Tracking local peaks... (positions only)")
        self.update_idletasks()

        for k in range(len(B_sorted)):
            y = I_sorted[k, :]
            try:
                mus, _bgs = self._find_peaks_in_spectrum(y)
                for i in range(n):
                    mu_track[k, i] = mus[i]
            except Exception:
                ok[k] = False

        self.B_sorted = B_sorted
        self.local_mu_track = mu_track
        self.local_ok = ok

        # Quick plot (outside Tk)
        plt.figure()
        for i in range(n):
            plt.plot(B_sorted, mu_track[:, i], "o-", ms=3, lw=1, label=f"peak {i+1}")
        plt.xlabel("Magnetic field B (T)")
        plt.ylabel("Peak position mu (eV)")
        plt.title("Local peak positions vs B")
        plt.legend()
        plt.tight_layout()
        plt.show()

        self.status.set(f"Tracking done. Success: {int(np.sum(ok))}/{len(ok)} points.")

    def on_save_local_csv(self):
        if self.h5_path is None or self.B_sorted is None or self.local_mu_track is None or self.local_ok is None:
            messagebox.showwarning("Nothing to save", "Run Track peaks (all B) first.")
            return

        n = self.local_mu_track.shape[1]
        default_name = self.h5_path.stem + f"_localPeaks{n}.csv"
        save_path = filedialog.asksaveasfilename(
            title="Save CSV (local peak positions)",
            initialdir=str(self.h5_path.parent),
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not save_path:
            return

        cols = ["B_T", "ok"] + [f"mu{i+1}_eV" for i in range(n)]
        mat = np.column_stack([self.B_sorted, self.local_ok.astype(int), self.local_mu_track])
        np.savetxt(save_path, mat, delimiter=",", header=",".join(cols), comments="")
        self.status.set(f"Saved: {save_path}")


if __name__ == "__main__":
    app = App()
    app.mainloop()