from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.optimize import curve_fit

HC_EV_NM = 1239.841984  # eV*nm
MAX_PEAKS = 4


# =============================
# Models / helpers
# =============================
def gaussian(x: np.ndarray, A: float, mu: float, sigma: float) -> np.ndarray:
    x = np.asarray(x, float)
    sigma = float(sigma)
    return float(A) * np.exp(-0.5 * ((x - float(mu)) / sigma) ** 2)


def make_model(n_peaks: int, bg_order: int):
    """
    bg_order:
      -1: no background
       0: constant c0
       1: linear   c0 + c1*x
       2: quad     c0 + c1*x + c2*x^2
    params: [A1,mu1,s1,  A2,mu2,s2, ... , c0,c1,c2...]
    """
    if n_peaks < 1 or n_peaks > MAX_PEAKS:
        raise ValueError("n_peaks must be 1..4")
    if bg_order not in (-1, 0, 1, 2):
        raise ValueError("bg_order must be -1,0,1,2")

    n_bg = 0 if bg_order < 0 else (bg_order + 1)
    n_params = 3 * n_peaks + n_bg

    def f(x: np.ndarray, *params) -> np.ndarray:
        x = np.asarray(x, float)
        y = np.zeros_like(x, float)

        for i in range(n_peaks):
            A = params[3 * i + 0]
            mu = params[3 * i + 1]
            sig = params[3 * i + 2]
            y = y + gaussian(x, A, mu, sig)

        if bg_order >= 0:
            base = 3 * n_peaks
            bg = params[base : base + n_bg]
            for pwr, c in enumerate(bg):
                y = y + float(c) * (x**pwr)

        return y

    return f, n_params


def estimate_bg_poly(E_roi: np.ndarray, y_roi: np.ndarray, bg_order: int) -> list[float]:
    """Estimate polynomial background from ROI edges (robust enough for init)."""
    if bg_order < 0:
        return []

    E_roi = np.asarray(E_roi, float)
    y_roi = np.asarray(y_roi, float)
    n = len(E_roi)
    if n < 8:
        return [float(np.median(y_roi))] + [0.0] * bg_order

    m = max(6, int(0.12 * n))
    idx = np.r_[0:m, n - m : n]
    x = E_roi[idx]
    y = y_roi[idx]

    coeff_hi = np.polyfit(x, y, deg=int(bg_order))
    # polyfit returns highest power first -> convert to [c0,c1,c2]
    if bg_order == 0:
        return [float(coeff_hi[0])]
    if bg_order == 1:
        a1, a0 = coeff_hi
        return [float(a0), float(a1)]
    if bg_order == 2:
        a2, a1, a0 = coeff_hi
        return [float(a0), float(a1), float(a2)]
    raise ValueError("Unsupported bg_order")


def eval_bg(x: np.ndarray, bg_coeffs: list[float]) -> np.ndarray:
    x = np.asarray(x, float)
    b = np.zeros_like(x, float)
    for pwr, c in enumerate(bg_coeffs):
        b = b + float(c) * (x**pwr)
    return b


def make_bounds(E_roi: np.ndarray, y_roi: np.ndarray, n_peaks: int, bg_order: int) -> tuple[np.ndarray, np.ndarray]:
    Emin, Emax = float(np.min(E_roi)), float(np.max(E_roi))
    roi_w = float(Emax - Emin) + 1e-12

    yr = float(np.max(y_roi) - np.min(y_roi) + 1e-12)

    sig_min = max(roi_w / 5000.0, 5e-5)
    sig_max = max(roi_w / 8.0, sig_min * 20)

    lb, ub = [], []
    for _ in range(n_peaks):
        lb += [0.0, Emin, sig_min]
        ub += [yr * 200.0, Emax, sig_max]

    # background bounds
    if bg_order >= 0:
        lb += [float(np.min(y_roi) - 10.0 * yr)]
        ub += [float(np.max(y_roi) + 10.0 * yr)]
    if bg_order >= 1:
        slope_scale = yr / roi_w
        lb += [-300.0 * abs(slope_scale)]
        ub += [+300.0 * abs(slope_scale)]
    if bg_order >= 2:
        quad_scale = yr / (roi_w * roi_w)
        lb += [-800.0 * abs(quad_scale)]
        ub += [+800.0 * abs(quad_scale)]

    return np.array(lb, float), np.array(ub, float)


def clamp_to_bounds(p0: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    p0 = np.asarray(p0, float)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    if p0.shape != lb.shape or p0.shape != ub.shape:
        raise ValueError("clamp_to_bounds: shape mismatch")
    span = np.maximum(ub - lb, 1e-12)
    eps = 1e-9 * span
    return np.minimum(np.maximum(p0, lb + eps), ub - eps)


def build_p0_from_user(
    E_roi: np.ndarray,
    y_roi: np.ndarray,
    mus: list[float],
    sigmas: list[float],
    bg_order: int,
    bg_user: list[float] | None,
    amps_user: list[float | None] | None = None,
) -> np.ndarray:
    """Build initial parameter vector from UI values."""
    E_roi = np.asarray(E_roi, float)
    y_roi = np.asarray(y_roi, float)

    # background init
    if bg_order < 0:
        bg0: list[float] = []
        y_flat = y_roi
    else:
        if bg_user is not None and len(bg_user) == (bg_order + 1):
            bg0 = [float(v) for v in bg_user]
        else:
            bg0 = estimate_bg_poly(E_roi, y_roi, bg_order=bg_order)
        y_flat = y_roi - eval_bg(E_roi, bg0)

    if amps_user is None:
        amps_user = [None] * len(mus)

    p0: list[float] = []
    for mu, sig, A_in in zip(mus, sigmas, amps_user):
        mu = float(mu)
        sig = float(sig)
        j = int(np.argmin(np.abs(E_roi - mu)))
        A_auto = float(max(y_flat[j], np.max(y_flat) * 0.2, 1e-9))
        A = float(A_in) if (A_in is not None) else A_auto
        p0 += [A, mu, sig]

    p0 += bg0
    return np.array(p0, float)


def unpack_params(popt: np.ndarray, n_peaks: int, bg_order: int):
    amps, mus, sigs = [], [], []
    for i in range(n_peaks):
        amps.append(float(popt[3 * i + 0]))
        mus.append(float(popt[3 * i + 1]))
        sigs.append(float(popt[3 * i + 2]))

    if bg_order < 0:
        bg = []
    else:
        bg = [float(v) for v in popt[3 * n_peaks : 3 * n_peaks + (bg_order + 1)]]
    return amps, mus, sigs, bg


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
# GUI App
# =============================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Magnetic-field spectra fitting (Gaussian, eV)")
        self.geometry("1250x860")

        # data
        self.h5_path: Path | None = None
        self.B: np.ndarray | None = None
        self.E: np.ndarray | None = None
        self.I: np.ndarray | None = None

        self.i0: int | None = None  # index closest to 0T
        self.E_roi: np.ndarray | None = None
        self.I_roi: np.ndarray | None = None

        # fit results
        self.popt0: np.ndarray | None = None
        self.track_params: np.ndarray | None = None
        self.track_ok: np.ndarray | None = None
        self.B_sorted: np.ndarray | None = None
        self.last_npeaks: int | None = None
        self.last_bg_order: int | None = None

        # locks / options
        self.lock_mu_vars = [tk.BooleanVar(value=False) for _ in range(MAX_PEAKS)]
        self.lock_sig_vars = [tk.BooleanVar(value=False) for _ in range(MAX_PEAKS)]
        self.lock_sigma0_var = tk.BooleanVar(value=False)

        self.fix_bg_var = tk.BooleanVar(value=False)
        self.apply_sigma_lock_tracking_var = tk.BooleanVar(value=True)

        # per-peak vars
        self.mu_vars = [tk.StringVar(value="") for _ in range(MAX_PEAKS)]
        self.sig_vars = [tk.StringVar(value="0.002") for _ in range(MAX_PEAKS)]
        self.A_vars = [tk.StringVar(value="") for _ in range(MAX_PEAKS)]
        self.pk_emin_vars = [tk.StringVar(value="") for _ in range(MAX_PEAKS)]
        self.pk_emax_vars = [tk.StringVar(value="") for _ in range(MAX_PEAKS)]

        self._build_ui()
        self._update_peak_entry_state()
        self._update_bg_entry_state()

    # ---------- UI ----------
    def _build_ui(self):
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        # LEFT: scrollable controls
        left_container = ttk.Frame(outer)
        left_container.pack(side="left", fill="y")

        left_canvas = tk.Canvas(left_container, width=420, highlightthickness=0, borderwidth=0)
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
            # Windows: delta=120 per notch
            left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # RIGHT: plot
        right = ttk.Frame(outer, padding=8)
        right.pack(side="right", fill="both", expand=True)

        # -------- Controls --------
        ttk.Label(left, text="File").pack(anchor="w")
        file_row = ttk.Frame(left)
        file_row.pack(fill="x", pady=(0, 8))
        self.file_lbl = ttk.Label(file_row, text="(no file)", width=44)
        self.file_lbl.pack(side="left", fill="x", expand=True)
        ttk.Button(file_row, text="Open .h5", command=self.on_open).pack(side="left", padx=(6, 0))

        ttk.Separator(left).pack(fill="x", pady=6)

        ttk.Label(left, text="ROI (eV)").pack(anchor="w")
        roi_row = ttk.Frame(left)
        roi_row.pack(fill="x", pady=(0, 8))
        self.roi_emin = tk.StringVar(value="")
        self.roi_emax = tk.StringVar(value="")
        ttk.Label(roi_row, text="Emin").pack(side="left")
        ttk.Entry(roi_row, textvariable=self.roi_emin, width=10).pack(side="left", padx=(4, 10))
        ttk.Label(roi_row, text="Emax").pack(side="left")
        ttk.Entry(roi_row, textvariable=self.roi_emax, width=10).pack(side="left", padx=(4, 0))
        ttk.Button(left, text="Plot 0T", command=self.on_plot_0t).pack(fill="x", pady=(0, 6))

        ttk.Separator(left).pack(fill="x", pady=6)

        ttk.Label(left, text="Fit settings").pack(anchor="w")
        set_row = ttk.Frame(left)
        set_row.pack(fill="x", pady=(0, 6))
        ttk.Label(set_row, text="#peaks").pack(side="left")
        self.npeaks_var = tk.IntVar(value=3)
        npeaks = ttk.Combobox(set_row, values=[1, 2, 3, 4], textvariable=self.npeaks_var, width=3, state="readonly")
        npeaks.pack(side="left", padx=(6, 12))
        npeaks.bind("<<ComboboxSelected>>", lambda e: self._update_peak_entry_state())

        self.same_sigma = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="same sigma for all peaks", variable=self.same_sigma, command=self._update_peak_entry_state).pack(
            anchor="w", pady=(0, 6)
        )

        self.sigma0_var = tk.StringVar(value="0.002")
        sig0_row = ttk.Frame(left)
        sig0_row.pack(fill="x", pady=(0, 10))
        ttk.Label(sig0_row, text="sigma0 (eV)").pack(side="left")
        self.sigma0_ent = ttk.Entry(sig0_row, textvariable=self.sigma0_var, width=10)
        self.sigma0_ent.pack(side="left", padx=(6, 0))

        ttk.Label(left, text="Initial peak centers mu_i (eV)").pack(anchor="w")
        for i in range(MAX_PEAKS):
            row = ttk.Frame(left)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"mu{i+1}").pack(side="left", padx=(0, 6))
            ent = ttk.Entry(row, textvariable=self.mu_vars[i], width=14)
            ent.pack(side="left", fill="x", expand=True)
            setattr(self, f"mu_ent_{i+1}", ent)

        ttk.Label(left, text="(optional) initial amplitudes A_i").pack(anchor="w", pady=(8, 0))
        for i in range(MAX_PEAKS):
            row = ttk.Frame(left)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"A{i+1}").pack(side="left", padx=(0, 6))
            ent = ttk.Entry(row, textvariable=self.A_vars[i], width=14)
            ent.pack(side="left", fill="x", expand=True)
            setattr(self, f"A_ent_{i+1}", ent)

        ttk.Label(left, text="(optional) individual sigma_i (eV)").pack(anchor="w", pady=(10, 0))
        for i in range(MAX_PEAKS):
            row = ttk.Frame(left)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"sigma{i+1}").pack(side="left", padx=(0, 6))
            ent = ttk.Entry(row, textvariable=self.sig_vars[i], width=14)
            ent.pack(side="left", fill="x", expand=True)
            setattr(self, f"sig_ent_{i+1}", ent)

        ttk.Label(left, text="Per-peak fitting ranges (eV) for stepwise fit").pack(anchor="w", pady=(10, 0))
        for i in range(MAX_PEAKS):
            row = ttk.Frame(left)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"peak {i+1}").pack(side="left", padx=(0, 6))
            ttk.Label(row, text="Emin").pack(side="left")
            ent1 = ttk.Entry(row, textvariable=self.pk_emin_vars[i], width=8)
            ent1.pack(side="left", padx=(4, 8))
            ttk.Label(row, text="Emax").pack(side="left")
            ent2 = ttk.Entry(row, textvariable=self.pk_emax_vars[i], width=8)
            ent2.pack(side="left", padx=(4, 0))
            setattr(self, f"pk_emin_ent_{i+1}", ent1)
            setattr(self, f"pk_emax_ent_{i+1}", ent2)

        lock_box = ttk.LabelFrame(left, text="Lock during fit (fix parameter)")
        lock_box.pack(fill="x", pady=(8, 8))
        for i in range(MAX_PEAKS):
            r = ttk.Frame(lock_box)
            r.pack(fill="x", pady=1)
            ttk.Label(r, text=f"peak {i+1}").pack(side="left", padx=(0, 6))
            ttk.Checkbutton(r, text="lock mu", variable=self.lock_mu_vars[i]).pack(side="left")
            ttk.Checkbutton(r, text="lock sigma", variable=self.lock_sig_vars[i]).pack(side="left", padx=(8, 0))

        ttk.Checkbutton(lock_box, text="(same sigma mode) lock sigma0 for all peaks", variable=self.lock_sigma0_var).pack(
            anchor="w", pady=(4, 0)
        )
        ttk.Checkbutton(
            lock_box, text="Apply sigma locks during tracking", variable=self.apply_sigma_lock_tracking_var
        ).pack(anchor="w", pady=(4, 0))

        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", pady=(2, 6))
        ttk.Button(btn_row, text="Preview (no fit)", command=self.on_preview).pack(side="left", fill="x", expand=True)
        ttk.Button(btn_row, text="Use fit → init", command=self.on_use_fit_as_init).pack(
            side="left", fill="x", expand=True, padx=(6, 0)
        )

        ttk.Button(left, text="Fit peaks (0T, stepwise)", command=self.on_fit_peaks_stepwise).pack(fill="x", pady=(0, 6))

        # Background
        ttk.Separator(left).pack(fill="x", pady=8)
        ttk.Label(left, text="Background").pack(anchor="w")

        bg_row = ttk.Frame(left)
        bg_row.pack(fill="x", pady=(2, 6))
        ttk.Label(bg_row, text="Model").pack(side="left")
        self.bg_model_var = tk.StringVar(value="Linear")
        self.bg_combo = ttk.Combobox(
            bg_row, values=["None", "Constant", "Linear", "Quadratic"], textvariable=self.bg_model_var, width=10, state="readonly"
        )
        self.bg_combo.pack(side="left", padx=(6, 0))
        self.bg_combo.bind("<<ComboboxSelected>>", lambda e: self._update_bg_entry_state())

        self.bg_c0_var = tk.StringVar(value="")
        self.bg_c1_var = tk.StringVar(value="")
        self.bg_c2_var = tk.StringVar(value="")

        for name, var in [("c0", self.bg_c0_var), ("c1", self.bg_c1_var), ("c2", self.bg_c2_var)]:
            row = ttk.Frame(left)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=name).pack(side="left", padx=(0, 6))
            ent = ttk.Entry(row, textvariable=var, width=16)
            ent.pack(side="left", fill="x", expand=True)
            setattr(self, f"bg_{name}_ent", ent)

        ttk.Label(left, text="(leave c's empty to auto-estimate from ROI edges)", foreground="#444").pack(
            anchor="w", pady=(4, 6)
        )

        bg_fix_row = ttk.Frame(left)
        bg_fix_row.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(bg_fix_row, text="Fix background (keep c0/c1/c2 constant)", variable=self.fix_bg_var).pack(
            side="left", fill="x", expand=True
        )
        ttk.Button(bg_fix_row, text="Freeze bg from 0T fit", command=self.on_freeze_bg_from_fit).pack(side="left", padx=(6, 0))

        ttk.Button(left, text="Fit 0T (global)", command=self.on_fit_0t).pack(fill="x", pady=(6, 6))
        ttk.Button(left, text="Run tracking (all B)", command=self.on_tracking).pack(fill="x", pady=(0, 6))
        ttk.Button(left, text="Save CSV", command=self.on_save).pack(fill="x", pady=(0, 10))

        self.status = tk.StringVar(value="Open an .h5 file to start.")
        ttk.Label(left, textvariable=self.status, wraplength=400, foreground="#333").pack(anchor="w", pady=(6, 0))

        # -------- Plot area --------
        self.fig = plt.Figure(figsize=(7.8, 6.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("0T spectrum / fit")

        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()

    # ---------- Helper methods ----------
    def _update_peak_entry_state(self):
        n = int(self.npeaks_var.get())
        same = bool(self.same_sigma.get())

        for i in range(MAX_PEAKS):
            getattr(self, f"mu_ent_{i+1}").configure(state=("normal" if i < n else "disabled"))
            getattr(self, f"A_ent_{i+1}").configure(state=("normal" if i < n else "disabled"))
            getattr(self, f"pk_emin_ent_{i+1}").configure(state=("normal" if i < n else "disabled"))
            getattr(self, f"pk_emax_ent_{i+1}").configure(state=("normal" if i < n else "disabled"))
            getattr(self, f"sig_ent_{i+1}").configure(state=("disabled" if same else ("normal" if i < n else "disabled")))

        self.sigma0_ent.configure(state=("normal" if same else "disabled"))

    def _bg_order(self) -> int:
        m = self.bg_model_var.get().strip()
        return {"None": -1, "Constant": 0, "Linear": 1, "Quadratic": 2}.get(m, 1)

    def _update_bg_entry_state(self):
        order = self._bg_order()
        self.bg_c0_ent.configure(state=("normal" if order >= 0 else "disabled"))
        self.bg_c1_ent.configure(state=("normal" if order >= 1 else "disabled"))
        self.bg_c2_ent.configure(state=("normal" if order >= 2 else "disabled"))

    def _require_data(self) -> bool:
        if self.B is None or self.E is None or self.I is None:
            messagebox.showwarning("No data", "Open an .h5 file first.")
            return False
        return True

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

    def _read_initials(self) -> tuple[int, list[float], list[float]]:
        n = int(self.npeaks_var.get())
        if n not in (1, 2, 3, 4):
            raise ValueError("n_peaks must be 1..4.")

        mus: list[float] = []
        for i in range(n):
            s = self.mu_vars[i].get().strip()
            if not s:
                raise ValueError(f"mu{i+1} is required.")
            mus.append(float(s))

        if bool(self.same_sigma.get()):
            s0 = self.sigma0_var.get().strip()
            if not s0:
                raise ValueError("sigma0 is required (same sigma mode).")
            sigmas = [float(s0)] * n
        else:
            sigmas: list[float] = []
            for i in range(n):
                s = self.sig_vars[i].get().strip()
                if not s:
                    raise ValueError(f"sigma{i+1} is required (individual sigma mode).")
                sigmas.append(float(s))

        return n, mus, sigmas

    def _read_amps_user(self, n: int) -> list[float | None]:
        out: list[float | None] = []
        for i in range(n):
            s = self.A_vars[i].get().strip()
            out.append(None if s == "" else float(s))
        return out

    def _read_peak_ranges(self, n: int) -> list[tuple[float, float]]:
        ranges: list[tuple[float, float]] = []
        for i in range(n):
            s1 = self.pk_emin_vars[i].get().strip()
            s2 = self.pk_emax_vars[i].get().strip()
            if not s1 or not s2:
                raise ValueError(f"peak {i+1} range Emin/Emax must be set for stepwise fit.")
            Emin, Emax = float(s1), float(s2)
            if Emin > Emax:
                Emin, Emax = Emax, Emin
            ranges.append((Emin, Emax))
        return ranges

    def _read_bg_user(self, bg_order: int) -> list[float] | None:
        """Return bg coeffs if user filled them, else None meaning auto-estimate."""
        if bg_order < 0:
            return None

        s0 = self.bg_c0_var.get().strip()
        s1 = self.bg_c1_var.get().strip()
        s2 = self.bg_c2_var.get().strip()

        if (not s0) and (not s1) and (not s2):
            return None

        vals: list[float] = []
        if bg_order >= 0:
            if not s0:
                raise ValueError("c0 required for selected background model.")
            vals.append(float(s0))
        if bg_order >= 1:
            if not s1:
                raise ValueError("c1 required for selected background model.")
            vals.append(float(s1))
        if bg_order >= 2:
            if not s2:
                raise ValueError("c2 required for selected background model.")
            vals.append(float(s2))
        return vals

    def _read_bg_fixed(self, bg_order: int) -> list[float]:
        """When Fix background is ON, required coeffs must be present."""
        if bg_order < 0:
            return []

        def req(name: str, s: str) -> float:
            s = s.strip()
            if not s:
                raise ValueError(f"{name} is required when background is fixed.")
            return float(s)

        out: list[float] = []
        if bg_order >= 0:
            out.append(req("c0", self.bg_c0_var.get()))
        if bg_order >= 1:
            out.append(req("c1", self.bg_c1_var.get()))
        if bg_order >= 2:
            out.append(req("c2", self.bg_c2_var.get()))
        return out

    def _lock_background_in_bounds(self, n_peaks: int, bg_order: int, lb: np.ndarray, ub: np.ndarray, bg_vals: list[float]):
        """Tighten bounds so bg stays fixed."""
        if bg_order < 0:
            return
        n_bg = bg_order + 1
        if len(bg_vals) != n_bg:
            raise ValueError("Internal error: bg_vals length mismatch.")

        base = 3 * n_peaks

        def eps(v: float) -> float:
            return max(1e-12, 1e-6 * abs(v) + 1e-12)

        for k in range(n_bg):
            v = float(bg_vals[k])
            e = eps(v)
            lb[base + k] = v - e
            ub[base + k] = v + e

    def _apply_locks_to_bounds(
        self,
        n: int,
        lb: np.ndarray,
        ub: np.ndarray,
        mus: list[float],
        sigmas: list[float],
        *,
        allow_mu_lock: bool,
    ):
        """Tighten bounds for locked mu/sigma."""
        def eps(val: float) -> float:
            return max(1e-12, 1e-6 * abs(val) + 1e-12)

        if allow_mu_lock:
            for i in range(n):
                if bool(self.lock_mu_vars[i].get()):
                    mu = float(mus[i])
                    j = 3 * i + 1
                    e = eps(mu)
                    lb[j] = mu - e
                    ub[j] = mu + e

        # sigma locks
        if bool(self.same_sigma.get()):
            if bool(self.lock_sigma0_var.get()):
                s0 = float(sigmas[0])
                for i in range(n):
                    j = 3 * i + 2
                    e = eps(s0)
                    lb[j] = s0 - e
                    ub[j] = s0 + e
        else:
            for i in range(n):
                if bool(self.lock_sig_vars[i].get()):
                    s = float(sigmas[i])
                    j = 3 * i + 2
                    e = eps(s)
                    lb[j] = s - e
                    ub[j] = s + e

    # ---------- Actions ----------
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

            # reset results
            self.popt0 = None
            self.track_params = None
            self.track_ok = None
            self.B_sorted = None
            self.E_roi = None
            self.I_roi = None
            self.last_npeaks = None
            self.last_bg_order = None
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
        self.ax.plot(self.E, self.I[self.i0, :], lw=1, alpha=0.35, label="0T full")
        self.ax.plot(self.E_roi, y0, lw=1.2, label="0T ROI")
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title(f"0T spectrum (closest B={self.B[self.i0]:.4g} T)")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

        self.status.set("0T plotted. Set peaks/background, then Preview / Stepwise / Fit 0T.")

    def on_preview(self):
        """No optimization: draw current manual parameters on ROI."""
        if not self._require_data():
            return
        if self.E_roi is None or self.I_roi is None or self.i0 is None:
            messagebox.showwarning("ROI missing", "Plot 0T first.")
            return

        try:
            n, mus, sigmas = self._read_initials()
            amps_user = self._read_amps_user(n)
            bg_order = self._bg_order()
            bg_user = self._read_bg_fixed(bg_order) if bool(self.fix_bg_var.get()) else self._read_bg_user(bg_order)
        except Exception as e:
            messagebox.showerror("Preview error", str(e))
            return

        y0 = self.I_roi[self.i0, :]
        model, n_params = make_model(n_peaks=n, bg_order=bg_order)

        p0 = build_p0_from_user(self.E_roi, y0, mus=mus, sigmas=sigmas, bg_order=bg_order, bg_user=bg_user, amps_user=amps_user)
        if len(p0) != n_params:
            messagebox.showerror("Preview error", "Internal error: parameter length mismatch.")
            return

        y_hat = model(self.E_roi, *p0)

        self.ax.clear()
        self.ax.plot(self.E_roi, y0, lw=1.2, label="data (0T ROI)")
        self.ax.plot(self.E_roi, y_hat, lw=1.6, label="preview model")
        self.ax.plot(self.E_roi, y0 - y_hat, lw=1.0, label="residual")
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("Preview (no fit) — adjust values then Stepwise / Fit 0T")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

        self.status.set("Preview updated (no fit).")

    def on_fit_peaks_stepwise(self):
        """
        Stepwise 0T fit:
          - choose ONE background (fixed entries if Fix background else auto-estimate from ROI edges)
          - for each peak i: fit Gaussian on its own [Emin,Emax] window at 0T, background fixed
          - write A/mu/sigma back to the UI
        """
        if not self._require_data():
            return
        if self.i0 is None or self.E is None or self.I is None:
            messagebox.showwarning("No data", "Open an .h5 file first.")
            return
        if self.E_roi is None or self.I_roi is None:
            messagebox.showwarning("ROI missing", "Plot 0T first.")
            return

        try:
            n, mus0, sig0 = self._read_initials()
            ranges = self._read_peak_ranges(n)
            bg_order = self._bg_order()
        except Exception as e:
            messagebox.showerror("Stepwise setup error", str(e))
            return

        E_full = self.E
        y_full = self.I[self.i0, :]

        # background (fixed during stepwise)
        try:
            if bg_order < 0:
                bg_coeffs: list[float] = []
            elif bool(self.fix_bg_var.get()):
                bg_coeffs = self._read_bg_fixed(bg_order)
            else:
                y0_roi = self.I_roi[self.i0, :]
                bg_coeffs = estimate_bg_poly(self.E_roi, y0_roi, bg_order=bg_order)
        except Exception as e:
            messagebox.showerror("Background error", str(e))
            return

        results: list[tuple[float, float, float]] = []

        for i in range(n):
            Emin, Emax = ranges[i]
            m = (E_full >= Emin) & (E_full <= Emax)
            if np.count_nonzero(m) < 10:
                messagebox.showerror("Range error", f"peak {i+1}: window too small/empty.")
                return

            xw = E_full[m]
            yw = y_full[m] - eval_bg(xw, bg_coeffs)

            mu_init = float(mus0[i])
            sig_init = float(sig0[i])

            A_init_s = self.A_vars[i].get().strip()
            A_init = float(A_init_s) if A_init_s else float(max(np.max(yw), 1e-9))

            # local bounds
            yr = float(np.max(yw) - np.min(yw) + 1e-12)
            w = float(Emax - Emin) + 1e-12
            sig_min = max(w / 5000.0, 5e-5)
            sig_max = max(w / 3.0, sig_min * 10)

            lock_mu = bool(self.lock_mu_vars[i].get())
            lock_sig = bool(self.lock_sig_vars[i].get())
            if bool(self.same_sigma.get()) and bool(self.lock_sigma0_var.get()):
                lock_sig = True

            try:
                if lock_mu and lock_sig:
                    A_fit = float(max(np.max(yw), 1e-9))
                    results.append((A_fit, mu_init, sig_init))

                elif lock_mu:
                    def g(x, A, sigma):
                        return gaussian(x, A, mu_init, sigma)

                    lb = np.array([0.0, sig_min], float)
                    ub = np.array([yr * 300.0, sig_max], float)
                    p0 = clamp_to_bounds(np.array([A_init, sig_init], float), lb, ub)
                    popt, _ = curve_fit(g, xw, yw, p0=p0, bounds=(lb, ub), maxfev=20000)
                    results.append((float(popt[0]), mu_init, float(popt[1])))

                elif lock_sig:
                    def g(x, A, mu):
                        return gaussian(x, A, mu, sig_init)

                    lb = np.array([0.0, Emin], float)
                    ub = np.array([yr * 300.0, Emax], float)
                    p0 = clamp_to_bounds(np.array([A_init, mu_init], float), lb, ub)
                    popt, _ = curve_fit(g, xw, yw, p0=p0, bounds=(lb, ub), maxfev=20000)
                    results.append((float(popt[0]), float(popt[1]), sig_init))

                else:
                    lb = np.array([0.0, Emin, sig_min], float)
                    ub = np.array([yr * 300.0, Emax, sig_max], float)
                    p0 = clamp_to_bounds(np.array([A_init, mu_init, sig_init], float), lb, ub)
                    popt, _ = curve_fit(gaussian, xw, yw, p0=p0, bounds=(lb, ub), maxfev=20000)
                    results.append((float(popt[0]), float(popt[1]), float(popt[2])))

            except Exception as e:
                messagebox.showerror("Stepwise fit failed", f"peak {i+1}: {type(e).__name__}: {e}")
                return

        # write back
        for i, (A_fit, mu_fit, sig_fit) in enumerate(results):
            self.A_vars[i].set(f"{A_fit:.6g}")
            self.mu_vars[i].set(f"{mu_fit:.6f}")
            self.sig_vars[i].set(f"{sig_fit:.6g}")

        if bool(self.same_sigma.get()) and results:
            self.sigma0_var.set(f"{float(np.median([r[2] for r in results])):.6g}")

        self.status.set("Stepwise peak fits done. Now run 'Fit 0T (global)'.")

    def on_fit_0t(self):
        if not self._require_data():
            return
        if self.E_roi is None or self.I_roi is None or self.i0 is None:
            messagebox.showwarning("ROI missing", "Plot 0T first.")
            return

        try:
            n, mus, sigmas = self._read_initials()
            amps_user = self._read_amps_user(n)
            bg_order = self._bg_order()
            bg_user = self._read_bg_fixed(bg_order) if bool(self.fix_bg_var.get()) else self._read_bg_user(bg_order)
        except Exception as e:
            messagebox.showerror("Initial guess error", str(e))
            return

        y0 = self.I_roi[self.i0, :]
        try:
            model, n_params = make_model(n_peaks=n, bg_order=bg_order)
            lb, ub = make_bounds(self.E_roi, y0, n_peaks=n, bg_order=bg_order)

            p0 = build_p0_from_user(self.E_roi, y0, mus=mus, sigmas=sigmas, bg_order=bg_order, bg_user=bg_user, amps_user=amps_user)
            if len(p0) != n_params:
                raise ValueError("Internal error: parameter length mismatch.")

            # apply locks
            self._apply_locks_to_bounds(n, lb, ub, mus=mus, sigmas=sigmas, allow_mu_lock=True)
            if bool(self.fix_bg_var.get()) and bg_order >= 0:
                bg_fixed = self._read_bg_fixed(bg_order)
                self._lock_background_in_bounds(n, bg_order, lb, ub, bg_fixed)

            p0 = clamp_to_bounds(p0, lb, ub)

            popt0, _ = curve_fit(
                lambda x, *p: model(x, *p),
                self.E_roi,
                y0,
                p0=p0,
                bounds=(lb, ub),
                maxfev=60000,
            )
        except Exception as e:
            messagebox.showerror("Fit failed", f"{type(e).__name__}: {e}")
            return

        self.popt0 = popt0
        self.last_npeaks = n
        self.last_bg_order = bg_order

        amps, mus_fit, sigs_fit, bg = unpack_params(popt0, n_peaks=n, bg_order=bg_order)
        bg_txt = "bg: None" if bg_order < 0 else "bg: " + ", ".join([f"c{i}={bg[i]:.6g}" for i in range(len(bg))])
        msg = (
            "0T global fit OK.\n"
            + "\n".join([f"peak{i+1}: mu={mus_fit[i]:.6f} eV, sigma={sigs_fit[i]:.6g} eV, A={amps[i]:.6g}" for i in range(n)])
            + f"\n{bg_txt}"
        )
        self.status.set(msg)

        fit = model(self.E_roi, *popt0)
        self.ax.clear()
        self.ax.plot(self.E_roi, y0, lw=1.2, label="data (0T ROI)")
        self.ax.plot(self.E_roi, fit, lw=1.6, label="fit")
        self.ax.plot(self.E_roi, y0 - fit, lw=1.0, label="residual")
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.set_title("0T fit (ROI)")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()

    def on_use_fit_as_init(self):
        if self.popt0 is None or self.last_npeaks is None or self.last_bg_order is None:
            messagebox.showwarning("No fit", "Fit 0T first.")
            return

        n = int(self.last_npeaks)
        bg_order = int(self.last_bg_order)

        for i in range(n):
            self.A_vars[i].set(f"{float(self.popt0[3*i+0]):.6g}")
            self.mu_vars[i].set(f"{float(self.popt0[3*i+1]):.6f}")
            self.sig_vars[i].set(f"{float(self.popt0[3*i+2]):.6g}")

        if bool(self.same_sigma.get()) and n >= 1:
            self.sigma0_var.set(self.sig_vars[0].get())

        if bg_order < 0:
            self.bg_c0_var.set("")
            self.bg_c1_var.set("")
            self.bg_c2_var.set("")
        else:
            bg = self.popt0[3*n : 3*n + (bg_order + 1)]
            self.bg_c0_var.set(f"{float(bg[0]):.6g}")
            self.bg_c1_var.set(f"{float(bg[1]):.6g}" if bg_order >= 1 else "")
            self.bg_c2_var.set(f"{float(bg[2]):.6g}" if bg_order >= 2 else "")

        self.status.set("Copied 0T fit → init. Now tweak values and Preview/Stepwise/Fit again.")

    def on_freeze_bg_from_fit(self):
        if self.popt0 is None or self.last_npeaks is None or self.last_bg_order is None:
            messagebox.showwarning("No fit", "Fit 0T first.")
            return

        n = int(self.last_npeaks)
        bg_order = int(self.last_bg_order)
        if bg_order < 0:
            messagebox.showwarning("No background", "Background model is None.")
            return

        bg = self.popt0[3*n : 3*n + (bg_order + 1)]
        self.bg_c0_var.set(f"{float(bg[0]):.6g}")
        self.bg_c1_var.set(f"{float(bg[1]):.6g}" if bg_order >= 1 else "")
        self.bg_c2_var.set(f"{float(bg[2]):.6g}" if bg_order >= 2 else "")
        self.fix_bg_var.set(True)
        self.status.set("Background frozen from 0T fit. It will stay fixed for subsequent fits/tracking.")

    def on_tracking(self):
        if not self._require_data():
            return
        if self.popt0 is None or self.last_npeaks is None or self.last_bg_order is None:
            messagebox.showwarning("Fit needed", "Fit 0T first.")
            return
        if self.E_roi is None or self.I_roi is None or self.i0 is None or self.B is None:
            messagebox.showwarning("ROI missing", "Plot 0T first.")
            return

        n = int(self.last_npeaks)
        bg_order = int(self.last_bg_order)
        model, _ = make_model(n_peaks=n, bg_order=bg_order)

        y0 = self.I_roi[self.i0, :]
        lb_base, ub_base = make_bounds(self.E_roi, y0, n_peaks=n, bg_order=bg_order)

        bg_fixed_vals: list[float] = []
        if bool(self.fix_bg_var.get()) and bg_order >= 0:
            try:
                bg_fixed_vals = self._read_bg_fixed(bg_order)
            except Exception as e:
                messagebox.showerror("Background fixed", str(e))
                return

        idxB = np.argsort(self.B)
        self.B_sorted = self.B[idxB]
        I_sorted = self.I_roi[idxB, :]
        k0 = int(np.where(idxB == self.i0)[0][0])

        params = np.full((len(self.B_sorted), len(self.popt0)), np.nan, float)
        ok = np.zeros(len(self.B_sorted), dtype=bool)
        params[k0, :] = self.popt0
        ok[k0] = True

        self.status.set("Tracking... (may take a bit)")
        self.update_idletasks()

        def fit_one(y: np.ndarray, p_start: np.ndarray) -> tuple[bool, np.ndarray]:
            lb = lb_base.copy()
            ub = ub_base.copy()

            # lock background if fixed
            if bool(self.fix_bg_var.get()) and bg_order >= 0:
                self._lock_background_in_bounds(n, bg_order, lb, ub, bg_fixed_vals)

            # optionally lock sigma during tracking (mu lock intentionally NOT applied)
            if bool(self.apply_sigma_lock_tracking_var.get()):
                mus_now = [float(p_start[3*i+1]) for i in range(n)]
                sig_now = [float(p_start[3*i+2]) for i in range(n)]
                self._apply_locks_to_bounds(n, lb, ub, mus=mus_now, sigmas=sig_now, allow_mu_lock=False)

            p0 = clamp_to_bounds(p_start, lb, ub)
            try:
                popt, _ = curve_fit(
                    lambda x, *p: model(x, *p),
                    self.E_roi,
                    y,
                    p0=p0,
                    bounds=(lb, ub),
                    maxfev=60000,
                )
                return True, popt
            except Exception:
                return False, np.full_like(p_start, np.nan)

        # forward
        for k in range(k0 + 1, len(self.B_sorted)):
            prev = params[k - 1, :]
            if not np.all(np.isfinite(prev)):
                prev = self.popt0.copy()
            ok[k], params[k, :] = fit_one(I_sorted[k, :], prev.copy())

        # backward
        for k in range(k0 - 1, -1, -1):
            prev = params[k + 1, :]
            if not np.all(np.isfinite(prev)):
                prev = self.popt0.copy()
            ok[k], params[k, :] = fit_one(I_sorted[k, :], prev.copy())

        self.track_params = params
        self.track_ok = ok

        # quick plot: mu vs B
        mus_all = np.full((len(self.B_sorted), n), np.nan, float)
        for i in range(n):
            mus_all[:, i] = params[:, 3 * i + 1]

        plt.figure()
        for i in range(n):
            plt.plot(self.B_sorted, mus_all[:, i], "o-", ms=3, lw=1, label=f"peak {i+1}")
        plt.xlabel("Magnetic field B (T)")
        plt.ylabel("Peak energy (eV)")
        plt.title("Peak energies vs B (tracking fits)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        self.status.set(f"Tracking done. Success: {int(np.sum(ok))}/{len(ok)} points.")

    def on_save(self):
        if self.h5_path is None or self.track_params is None or self.track_ok is None or self.B_sorted is None:
            messagebox.showwarning("Nothing to save", "Run tracking first.")
            return
        if self.last_npeaks is None or self.last_bg_order is None:
            messagebox.showwarning("Unknown model", "Fit 0T first.")
            return

        n = int(self.last_npeaks)
        bg_order = int(self.last_bg_order)
        n_bg = 0 if bg_order < 0 else (bg_order + 1)

        # collect
        A_all = np.full((len(self.B_sorted), n), np.nan, float)
        mu_all = np.full((len(self.B_sorted), n), np.nan, float)
        sig_all = np.full((len(self.B_sorted), n), np.nan, float)
        for i in range(n):
            A_all[:, i] = self.track_params[:, 3 * i + 0]
            mu_all[:, i] = self.track_params[:, 3 * i + 1]
            sig_all[:, i] = self.track_params[:, 3 * i + 2]

        bg_c0 = np.full(len(self.B_sorted), np.nan, float)
        bg_c1 = np.full(len(self.B_sorted), np.nan, float)
        bg_c2 = np.full(len(self.B_sorted), np.nan, float)
        if n_bg > 0:
            bg_vals = self.track_params[:, 3 * n : 3 * n + n_bg]
            bg_c0[:] = bg_vals[:, 0]
            if n_bg >= 2:
                bg_c1[:] = bg_vals[:, 1]
            if n_bg >= 3:
                bg_c2[:] = bg_vals[:, 2]

        cols = ["B_T", "ok"]
        for i in range(n):
            cols += [f"A{i+1}", f"mu{i+1}_eV", f"sigma{i+1}_eV"]
        cols += ["bg_c0", "bg_c1", "bg_c2"]

        out = [self.B_sorted, self.track_ok.astype(int)]
        for i in range(n):
            out += [A_all[:, i], mu_all[:, i], sig_all[:, i]]
        out += [bg_c0, bg_c1, bg_c2]
        mat = np.column_stack(out)

        default_name = self.h5_path.stem + f"_gauss{n}_bg{bg_order}_tracking.csv"
        save_path = filedialog.asksaveasfilename(
            title="Save CSV",
            initialdir=str(self.h5_path.parent),
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not save_path:
            return

        np.savetxt(save_path, mat, delimiter=",", header=",".join(cols), comments="")
        self.status.set(f"Saved: {save_path}")


if __name__ == "__main__":
    app = App()
    app.mainloop()