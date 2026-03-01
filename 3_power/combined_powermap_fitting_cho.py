import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import h5py

from scipy.optimize import curve_fit
from scipy.ndimage import median_filter

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QFileDialog, QLabel, QLineEdit, QCheckBox,
    QSpinBox, QDoubleSpinBox,
    QTableWidget, QTableWidgetItem,
    QMessageBox, QProgressBar, QComboBox, QSplitter,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# -----------------------------
# Models
# -----------------------------
def gauss(x, amp, cen, wid):
    return amp * np.exp(-0.5 * ((x - cen) / wid) ** 2)


def multi_gauss_fixed_bg(x, *params, bg_fixed=0.0):
    y = np.full_like(x, bg_fixed, dtype=float)
    n = len(params) // 3
    for i in range(n):
        amp = params[3 * i + 0]
        cen = params[3 * i + 1]
        wid = params[3 * i + 2]
        y += gauss(x, amp, cen, wid)
    return y


# -----------------------------
# Cosmic ray removal
# -----------------------------
def remove_cosmic_rays_1d(y: np.ndarray, kernel_size: int = 7, sigma_threshold: float = 6.0) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    baseline = median_filter(y, size=kernel_size, mode="nearest")
    diff = y - baseline
    mad = np.median(np.abs(diff - np.median(diff)))
    if mad < 1e-12:
        return y.copy()
    sigma = 1.4826 * mad
    mask = np.abs(diff) > (sigma_threshold * sigma)
    out = y.copy()
    out[mask] = baseline[mask]
    return out


# -----------------------------
# Helpers
# -----------------------------
def wavelength_to_energy_ev(wvl_nm: np.ndarray) -> np.ndarray:
    return 1240.0 / np.asarray(wvl_nm, dtype=float)


def sort_by_energy(energy: np.ndarray, data_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(energy)
    return energy[idx], data_2d[:, idx]


def safe_get(h5: h5py.File, key: str):
    if key in h5:
        return h5[key][...]
    if key.startswith("/") and key[1:] in h5:
        return h5[key[1:]][...]
    raise KeyError(f"Dataset not found: {key} (available: {list(h5.keys())})")


def parse_float(text: str, default: float) -> float:
    t = (text or "").strip()
    if t == "":
        return default
    try:
        return float(t)
    except Exception:
        return default


def compute_bg_fixed(y_fit: np.ndarray, method: str, percentile: float, manual_bg: float) -> float:
    if method == "median":
        return float(np.median(y_fit))
    if method == "percentile":
        return float(np.percentile(y_fit, percentile))
    if method == "manual":
        return float(manual_bg)
    return float(np.median(y_fit))


def fwhm_from_sigma(wid: float) -> float:
    return 2.0 * np.sqrt(2.0 * np.log(2.0)) * float(wid)


def area_gauss(amp: float, wid: float) -> float:
    # integral of amp * exp(-0.5*((x-c)/wid)^2) dx = amp*wid*sqrt(2*pi)
    return float(amp) * float(wid) * np.sqrt(2.0 * np.pi)


# -----------------------------
# Rule definition
# -----------------------------
@dataclass
class PeakBound:
    amp_min: float = 0.0
    amp_max: float = 1e12
    cen_min: float = 0.0
    cen_max: float = 10.0
    wid_min: float = 0.0005
    wid_max: float = 0.03


@dataclass
class PowerRule:
    pmin: float
    pmax: float
    npeaks: int
    peak_bounds: List[PeakBound] = field(default_factory=list)


# -----------------------------
# Guess + bounds builders
# -----------------------------
def initial_guess_from_bounds(
    x: np.ndarray,
    y: np.ndarray,
    bg_fixed: float,
    pbs: List[PeakBound],
) -> List[float]:
    """
    initial guess uses bound centers midpoints + amplitude from local y
    """
    params = []
    for pb in pbs:
        cen0 = 0.5 * (pb.cen_min + pb.cen_max)
        wid0 = max(0.004, pb.wid_min) if pb.wid_max >= 0.004 else 0.5 * (pb.wid_min + pb.wid_max)
        amp0 = float(np.interp(cen0, x, y) - bg_fixed)
        amp0 = np.clip(amp0, pb.amp_min if pb.amp_min >= 0 else 0.0, pb.amp_max if np.isfinite(pb.amp_max) else amp0 + 1.0)
        amp0 = max(amp0, 1.0)
        params += [amp0, cen0, wid0]
    return params


def bounds_from_peak_bounds(pbs: List[PeakBound]) -> Tuple[List[float], List[float]]:
    lb, ub = [], []
    for pb in pbs:
        lb += [pb.amp_min, pb.cen_min, pb.wid_min]
        ub += [pb.amp_max, pb.cen_max, pb.wid_max]
    return lb, ub


# -----------------------------
# Matplotlib canvas
# -----------------------------
class MplCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure(figsize=(7, 5))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)


# -----------------------------
# Main GUI
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Power-Range Gaussian Fitting (H5) - bg fixed + peak bounds")
        self.resize(1550, 850)

        # Data
        self.h5_path: Optional[str] = None
        self.data_raw: Optional[np.ndarray] = None  # (n_spec, n_pts) energy-sorted and power-sorted
        self.energy: Optional[np.ndarray] = None    # (n_pts,) sorted ascending
        self.power: Optional[np.ndarray] = None     # (n_spec,) sorted ascending
        self.n_spec = 0
        self.n_pts = 0

        # Rules
        self.rules: List[PowerRule] = []
        self.selected_rule_row: int = -1

        # Fit results per spectrum
        self.fit_results: Dict[int, Dict[str, Any]] = {}

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # LEFT
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left.setMaximumWidth(520)

        row_load = QHBoxLayout()
        self.btn_load = QPushButton("Load H5")
        self.btn_load.clicked.connect(self.on_load_h5)
        self.lbl_path = QLabel("(no file)")
        self.lbl_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row_load.addWidget(self.btn_load)
        row_load.addWidget(self.lbl_path, 1)
        left_layout.addLayout(row_load)

        # Controls grid
        grid = QGridLayout()
        grid.addWidget(QLabel("Energy min (eV)"), 0, 0)
        self.ed_e_min = QLineEdit("1.38")
        grid.addWidget(self.ed_e_min, 0, 1)

        grid.addWidget(QLabel("Energy max (eV)"), 1, 0)
        self.ed_e_max = QLineEdit("1.48")
        grid.addWidget(self.ed_e_max, 1, 1)

        self.cb_cosmic = QCheckBox("Cosmic removal")
        self.cb_cosmic.setChecked(False)
        grid.addWidget(self.cb_cosmic, 2, 0, 1, 2)

        grid.addWidget(QLabel("kernel"), 3, 0)
        self.sp_kernel = QSpinBox()
        self.sp_kernel.setRange(3, 51)
        self.sp_kernel.setSingleStep(2)
        self.sp_kernel.setValue(7)
        grid.addWidget(self.sp_kernel, 3, 1)

        grid.addWidget(QLabel("sigma_th"), 4, 0)
        self.ds_sigma = QDoubleSpinBox()
        self.ds_sigma.setRange(1.0, 30.0)
        self.ds_sigma.setDecimals(2)
        self.ds_sigma.setValue(6.0)
        grid.addWidget(self.ds_sigma, 4, 1)

        grid.addWidget(QLabel("Preview spectrum idx"), 5, 0)
        self.sp_preview = QSpinBox()
        self.sp_preview.setRange(0, 0)
        self.sp_preview.valueChanged.connect(self.update_preview_plot)
        grid.addWidget(self.sp_preview, 5, 1)

        self.cb_show_recon = QCheckBox("Show reconstructed (if fitted)")
        self.cb_show_recon.setChecked(True)
        self.cb_show_recon.stateChanged.connect(self.update_preview_plot)
        grid.addWidget(self.cb_show_recon, 6, 0, 1, 2)

        # bg fixed
        grid.addWidget(QLabel("BG fixed method"), 7, 0)
        self.cb_bg_method = QComboBox()
        self.cb_bg_method.addItems(["median", "percentile", "manual"])
        self.cb_bg_method.setCurrentText("median")
        grid.addWidget(self.cb_bg_method, 7, 1)

        grid.addWidget(QLabel("percentile"), 8, 0)
        self.ds_bg_pct = QDoubleSpinBox()
        self.ds_bg_pct.setRange(0.0, 50.0)
        self.ds_bg_pct.setDecimals(1)
        self.ds_bg_pct.setValue(10.0)
        grid.addWidget(self.ds_bg_pct, 8, 1)

        grid.addWidget(QLabel("manual bg"), 9, 0)
        self.ed_bg_manual = QLineEdit("0")
        grid.addWidget(self.ed_bg_manual, 9, 1)

        left_layout.addLayout(grid)

        # Rule table
        left_layout.addWidget(QLabel("Power rules (click a rule row -> edit peak bounds below)"))
        self.tbl_rules = QTableWidget(0, 3)
        self.tbl_rules.setHorizontalHeaderLabels(["pmin(uW)", "pmax(uW)", "npeaks"])
        self.tbl_rules.horizontalHeader().setStretchLastSection(True)
        self.tbl_rules.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_rules.itemSelectionChanged.connect(self.on_rule_selected)
        left_layout.addWidget(self.tbl_rules)

        row_rule_btn = QHBoxLayout()
        self.btn_add_rule = QPushButton("Add rule")
        self.btn_add_rule.clicked.connect(self.on_add_rule)
        self.btn_del_rule = QPushButton("Delete rule")
        self.btn_del_rule.clicked.connect(self.on_del_rule)
        row_rule_btn.addWidget(self.btn_add_rule)
        row_rule_btn.addWidget(self.btn_del_rule)
        left_layout.addLayout(row_rule_btn)

        self.btn_insert_example = QPushButton("Insert example rules (10^8→10^4)")
        self.btn_insert_example.clicked.connect(self.on_insert_example_rules)
        left_layout.addWidget(self.btn_insert_example)

        # Peak bounds table
        left_layout.addWidget(QLabel("Peak bounds for selected rule (amp/cen/wid)"))
        self.tbl_bounds = QTableWidget(0, 6)
        self.tbl_bounds.setHorizontalHeaderLabels(["amp_min", "amp_max", "cen_min(eV)", "cen_max(eV)", "wid_min(eV)", "wid_max(eV)"])
        self.tbl_bounds.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.tbl_bounds, 1)

        self.btn_apply_bounds = QPushButton("Apply bounds to selected rule")
        self.btn_apply_bounds.clicked.connect(self.on_apply_bounds_to_rule)
        left_layout.addWidget(self.btn_apply_bounds)

        # Fit controls
        row_fit = QHBoxLayout()
        self.btn_fit_all = QPushButton("Batch Fit (all spectra)")
        self.btn_fit_all.clicked.connect(self.on_fit_all)
        row_fit.addWidget(self.btn_fit_all)
        left_layout.addLayout(row_fit)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        left_layout.addWidget(self.progress)

        self.btn_save_csv = QPushButton("Save peak results CSV (long format)")
        self.btn_save_csv.clicked.connect(self.on_save_csv_long)
        left_layout.addWidget(self.btn_save_csv)

        left_layout.addStretch(1)

        # RIGHT
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.canvas = MplCanvas()
        right_layout.addWidget(self.canvas, 2)

        right_layout.addWidget(QLabel("Fit results (click a row -> spectrum+fit)"))
        # results table (one row per spectrum)
        self.tbl_results = QTableWidget(0, 7)
        self.tbl_results.setHorizontalHeaderLabels(["idx", "power(uW)", "npeaks", "bg", "success", "note", "rule"])
        self.tbl_results.horizontalHeader().setStretchLastSection(True)
        self.tbl_results.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_results.itemSelectionChanged.connect(self.on_result_row_selected)
        right_layout.addWidget(self.tbl_results, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([520, 1030])

    # ---------------- Data loading ----------------
    def on_load_h5(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select H5 file", "", "H5 files (*.h5 *.hdf5);;All files (*.*)")
        if not path:
            return
        try:
            self.load_h5(path)
            self.h5_path = path
            self.lbl_path.setText(path)
            self.fit_results.clear()
            self.refresh_results_table()
            self.update_preview_plot()
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def load_h5(self, path: str):
        with h5py.File(path, "r") as f:
            data = safe_get(f, "spectro_data")
            wvl = safe_get(f, "spectro_wavelength")
            power = safe_get(f, "power_axis")

        data = np.asarray(data).squeeze()
        wvl = np.asarray(wvl).squeeze()
        power = np.asarray(power).squeeze()

        if data.ndim != 2:
            raise ValueError(f"spectro_data must be 2D. Got {data.shape}")
        if wvl.ndim != 1 or power.ndim != 1:
            raise ValueError("spectro_wavelength and power_axis must be 1D.")
        if data.shape[1] != wvl.shape[0]:
            raise ValueError("data points != wavelength length")
        if data.shape[0] != power.shape[0]:
            raise ValueError("n_spectra != power length")

        energy = wavelength_to_energy_ev(wvl)
        energy_sorted, data_sorted = sort_by_energy(energy, data)

        order = np.argsort(power.astype(float))
        self.power = power[order].astype(float)
        self.data_raw = data_sorted[order, :].astype(float)
        self.energy = energy_sorted.astype(float)

        self.n_spec, self.n_pts = self.data_raw.shape
        self.sp_preview.setRange(0, max(0, self.n_spec - 1))
        self.sp_preview.setValue(0)

    # ---------------- Rules ----------------
    def on_add_rule(self):
        r = self.tbl_rules.rowCount()
        self.tbl_rules.insertRow(r)
        self.tbl_rules.setItem(r, 0, QTableWidgetItem("0.1"))
        self.tbl_rules.setItem(r, 1, QTableWidgetItem("1.0"))
        self.tbl_rules.setItem(r, 2, QTableWidgetItem("1"))
        # also add to internal list (lazy build later)

    def on_del_rule(self):
        rows = sorted({i.row() for i in self.tbl_rules.selectedIndexes()}, reverse=True)
        for r in rows:
            self.tbl_rules.removeRow(r)
        self.selected_rule_row = -1
        self.tbl_bounds.setRowCount(0)

    def on_insert_example_rules(self):
        rules = [
            (0.0008, 0.008, 1),
            (0.008,  0.08,  1),
            (0.08,   0.8,   2),
            (0.8,    8.0,   2),
            (8.0,  110.0,   3),
        ]
        self.tbl_rules.setRowCount(0)
        for pmin, pmax, npk in rules:
            r = self.tbl_rules.rowCount()
            self.tbl_rules.insertRow(r)
            self.tbl_rules.setItem(r, 0, QTableWidgetItem(f"{pmin}"))
            self.tbl_rules.setItem(r, 1, QTableWidgetItem(f"{pmax}"))
            self.tbl_rules.setItem(r, 2, QTableWidgetItem(f"{npk}"))

    def build_rules_from_ui(self) -> List[PowerRule]:
        rules: List[PowerRule] = []
        for r in range(self.tbl_rules.rowCount()):
            try:
                pmin = float(self.tbl_rules.item(r, 0).text())
                pmax = float(self.tbl_rules.item(r, 1).text())
                npeaks = int(float(self.tbl_rules.item(r, 2).text()))
                if pmax < pmin:
                    pmin, pmax = pmax, pmin
                npeaks = max(1, npeaks)

                # if bounds table currently corresponds to this row and applied, it will be stored in self.rules_store
                # We keep a per-row store:
                pb_list = self._get_saved_bounds_for_row(r, npeaks)
                rules.append(PowerRule(pmin, pmax, npeaks, pb_list))
            except Exception:
                continue
        rules.sort(key=lambda rr: rr.pmin)
        return rules

    def _ensure_bounds_store(self):
        if not hasattr(self, "_bounds_store"):
            self._bounds_store: Dict[int, List[PeakBound]] = {}

    def _get_saved_bounds_for_row(self, row: int, npeaks: int) -> List[PeakBound]:
        self._ensure_bounds_store()
        if row in self._bounds_store and len(self._bounds_store[row]) == npeaks:
            return self._bounds_store[row]

        # default bounds if none saved
        emin = parse_float(self.ed_e_min.text(), 1.38)
        emax = parse_float(self.ed_e_max.text(), 1.48)
        if emax < emin:
            emin, emax = emax, emin
        # spread peaks across window as default centers
        centers = np.linspace(emin, emax, npeaks + 2)[1:-1]
        out = []
        for c in centers:
            out.append(PeakBound(
                amp_min=0.0, amp_max=1e12,
                cen_min=float(c - 0.01), cen_max=float(c + 0.01),
                wid_min=0.0005, wid_max=0.03
            ))
        self._bounds_store[row] = out
        return out

    def on_rule_selected(self):
        sel = self.tbl_rules.selectedItems()
        if not sel:
            return
        row = sel[0].row()
        self.selected_rule_row = row

        # determine npeaks from rule row
        try:
            npeaks = int(float(self.tbl_rules.item(row, 2).text()))
            npeaks = max(1, npeaks)
        except Exception:
            npeaks = 1

        pbs = self._get_saved_bounds_for_row(row, npeaks)
        self.populate_bounds_table(pbs)

    def populate_bounds_table(self, pbs: List[PeakBound]):
        self.tbl_bounds.setRowCount(len(pbs))
        for i, pb in enumerate(pbs):
            self.tbl_bounds.setVerticalHeaderItem(i, QTableWidgetItem(f"Peak {i+1}"))
            vals = [pb.amp_min, pb.amp_max, pb.cen_min, pb.cen_max, pb.wid_min, pb.wid_max]
            for c, v in enumerate(vals):
                self.tbl_bounds.setItem(i, c, QTableWidgetItem(f"{v:.6g}"))

    def on_apply_bounds_to_rule(self):
        if self.selected_rule_row < 0:
            QMessageBox.warning(self, "No rule", "Select a rule row first.")
            return
        # read npeaks
        try:
            npeaks = int(float(self.tbl_rules.item(self.selected_rule_row, 2).text()))
            npeaks = max(1, npeaks)
        except Exception:
            npeaks = 1

        if self.tbl_bounds.rowCount() != npeaks:
            QMessageBox.warning(self, "Mismatch", "Bounds rows must match npeaks. Re-select rule to refresh.")
            return

        pbs: List[PeakBound] = []
        for i in range(npeaks):
            try:
                amp_min = float(self.tbl_bounds.item(i, 0).text())
                amp_max = float(self.tbl_bounds.item(i, 1).text())
                cen_min = float(self.tbl_bounds.item(i, 2).text())
                cen_max = float(self.tbl_bounds.item(i, 3).text())
                wid_min = float(self.tbl_bounds.item(i, 4).text())
                wid_max = float(self.tbl_bounds.item(i, 5).text())
                if amp_max < amp_min: amp_min, amp_max = amp_max, amp_min
                if cen_max < cen_min: cen_min, cen_max = cen_max, cen_min
                if wid_max < wid_min: wid_min, wid_max = wid_max, wid_min
                pbs.append(PeakBound(amp_min, amp_max, cen_min, cen_max, wid_min, wid_max))
            except Exception as e:
                QMessageBox.warning(self, "Input error", f"Bad bounds at peak {i+1}: {e}")
                return

        self._ensure_bounds_store()
        self._bounds_store[self.selected_rule_row] = pbs
        QMessageBox.information(self, "Saved", "Bounds applied to this rule (in memory).")

    def rule_for_power(self, p: float, rules: List[PowerRule]) -> Optional[PowerRule]:
        for rr in rules:
            if rr.pmin <= p <= rr.pmax:
                return rr
        return None

    # ---------------- Fitting ----------------
    def current_fit_window_mask(self) -> np.ndarray:
        if self.energy is None:
            return np.array([], dtype=bool)
        emin = parse_float(self.ed_e_min.text(), float(self.energy.min()))
        emax = parse_float(self.ed_e_max.text(), float(self.energy.max()))
        if emax < emin:
            emin, emax = emax, emin
        return (self.energy >= emin) & (self.energy <= emax)

    def preprocess_spectrum(self, y: np.ndarray) -> np.ndarray:
        if self.cb_cosmic.isChecked():
            return remove_cosmic_rays_1d(y, kernel_size=int(self.sp_kernel.value()), sigma_threshold=float(self.ds_sigma.value()))
        return y.copy()

    def on_fit_all(self):
        if self.data_raw is None or self.energy is None or self.power is None:
            QMessageBox.warning(self, "No data", "Load an H5 file first.")
            return

        rules = self.build_rules_from_ui()
        if not rules:
            QMessageBox.warning(self, "No rules", "Add at least one power rule.")
            return

        mask = self.current_fit_window_mask()
        if mask.size == 0 or mask.sum() < 10:
            QMessageBox.warning(self, "Window error", "Energy window too small.")
            return

        method = self.cb_bg_method.currentText()
        pct = float(self.ds_bg_pct.value())
        manual_bg = parse_float(self.ed_bg_manual.text(), 0.0)

        self.fit_results.clear()
        self.progress.setValue(0)

        xw = self.energy[mask]

        for i in range(self.n_spec):
            p = float(self.power[i])
            rr = self.rule_for_power(p, rules)
            if rr is None:
                continue

            y = self.preprocess_spectrum(self.data_raw[i])
            yw = y[mask]

            bg_fixed = compute_bg_fixed(yw, method=method, percentile=pct, manual_bg=manual_bg)

            # bounds per peak
            pbs = rr.peak_bounds
            if len(pbs) != rr.npeaks:
                # safety: rebuild default
                pbs = self._get_saved_bounds_for_row(self.find_rule_row(rr), rr.npeaks)

            p0 = initial_guess_from_bounds(xw, yw, bg_fixed, pbs)
            lb, ub = bounds_from_peak_bounds(pbs)

            try:
                popt, _ = curve_fit(
                    lambda xx, *params: multi_gauss_fixed_bg(xx, *params, bg_fixed=bg_fixed),
                    xw, yw, p0=p0, bounds=(lb, ub), maxfev=30000
                )
                self.fit_results[i] = {
                    "idx": i,
                    "power": p,
                    "npeaks": rr.npeaks,
                    "bg": bg_fixed,
                    "params": [float(v) for v in popt],
                    "success": True,
                    "rule": f"{rr.pmin:g}-{rr.pmax:g}uW",
                }
            except Exception as e:
                self.fit_results[i] = {
                    "idx": i,
                    "power": p,
                    "npeaks": rr.npeaks,
                    "bg": bg_fixed,
                    "params": [],
                    "success": False,
                    "error": str(e),
                    "rule": f"{rr.pmin:g}-{rr.pmax:g}uW",
                }

            self.progress.setValue(int(100 * (i + 1) / self.n_spec))
            QApplication.processEvents()

        self.refresh_results_table()
        QMessageBox.information(self, "Done", f"Fitted {sum(v['success'] for v in self.fit_results.values())} spectra.")

    def find_rule_row(self, rr: PowerRule) -> int:
        # best-effort to match rule row; fallback 0
        for r in range(self.tbl_rules.rowCount()):
            try:
                pmin = float(self.tbl_rules.item(r, 0).text())
                pmax = float(self.tbl_rules.item(r, 1).text())
                npeaks = int(float(self.tbl_rules.item(r, 2).text()))
                if abs(pmin - rr.pmin) < 1e-15 and abs(pmax - rr.pmax) < 1e-15 and npeaks == rr.npeaks:
                    return r
            except Exception:
                pass
        return 0

    # ---------------- Results table + plot ----------------
    def refresh_results_table(self):
        rows = sorted(self.fit_results.keys())
        self.tbl_results.setRowCount(len(rows))
        for r, idx in enumerate(rows):
            res = self.fit_results[idx]
            self.tbl_results.setItem(r, 0, QTableWidgetItem(str(res["idx"])))
            self.tbl_results.setItem(r, 1, QTableWidgetItem(f"{res['power']:.6g}"))
            self.tbl_results.setItem(r, 2, QTableWidgetItem(str(res["npeaks"])))
            self.tbl_results.setItem(r, 3, QTableWidgetItem(f"{res['bg']:.6g}"))
            self.tbl_results.setItem(r, 4, QTableWidgetItem("1" if res.get("success", False) else "0"))
            self.tbl_results.setItem(r, 5, QTableWidgetItem("" if res.get("success", False) else res.get("error", "")[:60]))
            self.tbl_results.setItem(r, 6, QTableWidgetItem(res.get("rule", "")))

            if not res.get("success", False):
                for c in range(self.tbl_results.columnCount()):
                    it = self.tbl_results.item(r, c)
                    if it is not None:
                        it.setForeground(Qt.gray)

        if len(rows) > 0:
            self.tbl_results.selectRow(0)

    def selected_result_idx(self) -> Optional[int]:
        sel = self.tbl_results.selectedItems()
        if not sel:
            return None
        row = sel[0].row()
        item = self.tbl_results.item(row, 0)
        if item is None:
            return None
        try:
            return int(item.text())
        except Exception:
            return None

    def on_result_row_selected(self):
        idx = self.selected_result_idx()
        if idx is None:
            return
        self.sp_preview.blockSignals(True)
        self.sp_preview.setValue(idx)
        self.sp_preview.blockSignals(False)
        self.update_preview_plot()

    def update_preview_plot(self):
        if self.data_raw is None or self.energy is None or self.power is None:
            self.canvas.ax.clear()
            self.canvas.draw()
            return

        idx = int(self.sp_preview.value())
        x = self.energy
        p = float(self.power[idx])
        y = self.preprocess_spectrum(self.data_raw[idx])

        emin = parse_float(self.ed_e_min.text(), float(x.min()))
        emax = parse_float(self.ed_e_max.text(), float(x.max()))
        if emax < emin:
            emin, emax = emax, emin

        self.canvas.ax.clear()
        self.canvas.ax.plot(x, y, lw=1.2, label=f"raw/cleaned idx={idx}, P={p:.4g} uW")
        self.canvas.ax.set_xlim(emin, emax)
        self.canvas.ax.set_xlabel("Energy (eV)")
        self.canvas.ax.set_ylabel("Intensity")
        self.canvas.ax.grid(True, alpha=0.25)

        if self.cb_show_recon.isChecked() and idx in self.fit_results and self.fit_results[idx].get("success", False):
            res = self.fit_results[idx]
            bg = float(res["bg"])
            params = res["params"]
            yfit = multi_gauss_fixed_bg(x, *params, bg_fixed=bg)
            self.canvas.ax.plot(x, yfit, lw=2.0, label="fit (total)")
            npeaks = int(res["npeaks"])
            for k in range(npeaks):
                if len(params) >= 3 * (k + 1):
                    amp, cen, wid = params[3 * k: 3 * k + 3]
                    self.canvas.ax.plot(x, gauss(x, amp, cen, wid) + bg, ls="--", lw=1.0, label=f"peak{k+1}")
                    self.canvas.ax.axvline(cen, ls=":", lw=0.9, alpha=0.5)
            self.canvas.ax.axhline(bg, ls=":", lw=1.0, alpha=0.7)

        self.canvas.ax.legend(fontsize=9, loc="upper right")
        self.canvas.draw()

    # ---------------- CSV (long format) ----------------
    def on_save_csv_long(self):
        if not self.fit_results:
            QMessageBox.warning(self, "No results", "Run Batch Fit first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV files (*.csv)")
        if not path:
            return

        # long format: one row per peak
        headers = [
            "idx", "power_uW", "rule", "success",
            "bg_fixed",
            "peak_id",
            "amp", "cen_eV", "wid_eV",
            "fwhm_eV",
            "integrated_intensity",
        ]
        lines = [",".join(headers)]

        for idx in sorted(self.fit_results.keys()):
            res = self.fit_results[idx]
            success = bool(res.get("success", False))
            bg = float(res.get("bg", np.nan))
            rule = res.get("rule", "")
            p = float(res.get("power", np.nan))
            npeaks = int(res.get("npeaks", 0))
            params = res.get("params", [])

            if not success or len(params) < 3:
                # still write a row marking failure (peak_id empty)
                row = [
                    str(idx),
                    f"{p:.10g}",
                    rule,
                    "0",
                    f"{bg:.10g}",
                    "",
                    "", "", "",
                    "", ""
                ]
                lines.append(",".join(row))
                continue

            for k in range(npeaks):
                amp, cen, wid = params[3 * k: 3 * k + 3]
                fwhm = fwhm_from_sigma(wid)
                area = area_gauss(amp, wid)
                row = [
                    str(idx),
                    f"{p:.10g}",
                    rule,
                    "1",
                    f"{bg:.10g}",
                    str(k + 1),
                    f"{amp:.10g}",
                    f"{cen:.10g}",
                    f"{wid:.10g}",
                    f"{fwhm:.10g}",
                    f"{area:.10g}",
                ]
                lines.append(",".join(row))

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            QMessageBox.information(self, "Saved", path)
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())