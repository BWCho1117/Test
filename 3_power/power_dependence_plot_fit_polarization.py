# polarization_gui.py
# GUI to compute polarization or sigma+-sigma- difference vs power per peak from two fit-result CSV files
#
# polarization = (I_sigma+ - I_sigma-) / (I_sigma+ + I_sigma-)
# delta       = (Y_sigma+ - Y_sigma-)   where Y is the selected column (e.g., cen_eV, integrated_intensity, fwhm_eV)
#
# Rules:
# - CSV filename containing "72.5"  => sigma+
# - CSV filename containing "117.5" => sigma-
# - Compute ONLY when (same peak_id) AND (same power after rounding) exist in BOTH files.

import os
import sys
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QFileDialog, QLabel, QLineEdit,
    QMessageBox, QListWidget, QListWidgetItem,
    QCheckBox, QComboBox, QSpinBox, QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ------------------------------
# Matplotlib canvas
# ------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, width=8, height=5.5):
        self.fig = Figure(figsize=(width, height))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def resize_figure(self, w, h):
        self.fig.set_size_inches(w, h, forward=True)
        self.draw_idle()


# ------------------------------
# Main GUI
# ------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Polarization / Δ(σ+−σ−) vs Power (from two fit CSVs)")
        self.resize(1500, 850)

        self.df_plus = None
        self.df_minus = None
        self.path_plus = ""
        self.path_minus = ""

        self.peak_label_edits = {}

        self._build_ui()

    # --------------------------
    # UI
    # --------------------------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        left = QWidget()
        L = QVBoxLayout(left)

        row = QHBoxLayout()
        self.btn_load1 = QPushButton("Load CSV #1")
        self.btn_load2 = QPushButton("Load CSV #2")
        self.btn_load1.clicked.connect(lambda: self.load_csv(which=1))
        self.btn_load2.clicked.connect(lambda: self.load_csv(which=2))
        row.addWidget(self.btn_load1)
        row.addWidget(self.btn_load2)
        L.addLayout(row)

        self.lbl_info = QLabel("Load two CSV files.\nFilename containing 72.5 => σ+, 117.5 => σ−.")
        self.lbl_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        L.addWidget(self.lbl_info)

        # column selector
        L.addWidget(QLabel("Column to use (Y)"))
        self.cb_ycol = QComboBox()
        self.cb_ycol.addItems(["integrated_intensity", "amp", "cen_eV", "fwhm_eV"])
        self.cb_ycol.currentIndexChanged.connect(self._update_yaxis_hint_label)
        L.addWidget(self.cb_ycol)

        self.lbl_yhint = QLabel("")
        self.lbl_yhint.setTextInteractionFlags(Qt.TextSelectableByMouse)
        L.addWidget(self.lbl_yhint)

        # plot mode
        L.addWidget(QLabel("Plot mode"))
        self.cb_mode = QComboBox()
        self.cb_mode.addItems([
            "Polarization: (σ+−σ−)/(σ++σ−)",
            "Delta: (σ+−σ−) of selected column"
        ])
        self.cb_mode.currentIndexChanged.connect(self._update_yaxis_hint_label)
        L.addWidget(self.cb_mode)

        # matching
        grid = QGridLayout()
        grid.addWidget(QLabel("Power rounding decimals (matching key)"), 0, 0)
        self.sp_round = QSpinBox()
        self.sp_round.setRange(0, 12)
        self.sp_round.setValue(6)
        grid.addWidget(self.sp_round, 0, 1)

        grid.addWidget(QLabel("Power min (µW)"), 1, 0)
        self.ed_pmin = QLineEdit("")
        grid.addWidget(self.ed_pmin, 1, 1)

        grid.addWidget(QLabel("Power max (µW)"), 2, 0)
        self.ed_pmax = QLineEdit("")
        grid.addWidget(self.ed_pmax, 2, 1)
        L.addLayout(grid)

        # peak selector
        L.addWidget(QLabel("Select peaks (peak_id)"))
        self.peak_list = QListWidget()
        self.peak_list.setSelectionMode(QListWidget.MultiSelection)
        L.addWidget(self.peak_list, 2)

        # labels
        self.group_labels = QGroupBox("Legend labels (Peak ID → Name)")
        gl = QVBoxLayout(self.group_labels)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_widget)
        self.scroll.setWidget(self.scroll_widget)
        gl.addWidget(self.scroll)
        L.addWidget(self.group_labels, 2)

        # plot options
        self.cb_show_points = QCheckBox("Show points")
        self.cb_show_points.setChecked(True)
        self.cb_logx = QCheckBox("Log X (power)")
        self.cb_logx.setChecked(False)
        self.cb_show_raw = QCheckBox("Also plot σ+ and σ− raw Y (same axes)")
        self.cb_show_raw.setChecked(False)

        L.addWidget(self.cb_show_points)
        L.addWidget(self.cb_logx)
        L.addWidget(self.cb_show_raw)

        # formatting
        fmt_box = QGroupBox("Plot formatting")
        fmt = QGridLayout(fmt_box)

        fmt.addWidget(QLabel("Fig W"), 0, 0)
        self.ed_w = QLineEdit("6")
        fmt.addWidget(self.ed_w, 0, 1)
        fmt.addWidget(QLabel("Fig H"), 0, 2)
        self.ed_h = QLineEdit("4")
        fmt.addWidget(self.ed_h, 0, 3)

        fmt.addWidget(QLabel("Left"), 1, 0)
        self.ed_left = QLineEdit("0.14")
        fmt.addWidget(self.ed_left, 1, 1)
        fmt.addWidget(QLabel("Right"), 1, 2)
        self.ed_right = QLineEdit("0.98")
        fmt.addWidget(self.ed_right, 1, 3)

        fmt.addWidget(QLabel("Bottom"), 2, 0)
        self.ed_bottom = QLineEdit("0.16")
        fmt.addWidget(self.ed_bottom, 2, 1)
        fmt.addWidget(QLabel("Top"), 2, 2)
        self.ed_top = QLineEdit("0.95")
        fmt.addWidget(self.ed_top, 2, 3)

        fmt.addWidget(QLabel("Label font"), 3, 0)
        self.ed_lab_fs = QLineEdit("11")
        fmt.addWidget(self.ed_lab_fs, 3, 1)
        fmt.addWidget(QLabel("Tick font"), 3, 2)
        self.ed_tick_fs = QLineEdit("10")
        fmt.addWidget(self.ed_tick_fs, 3, 3)

        fmt.addWidget(QLabel("Legend loc"), 4, 0)
        self.cb_leg_loc = QComboBox()
        self.cb_leg_loc.addItems(["best", "upper right", "upper left", "lower right", "lower left", "center right", "center left"])
        self.cb_leg_loc.setCurrentText("best")
        fmt.addWidget(self.cb_leg_loc, 4, 1)

        fmt.addWidget(QLabel("Legend font"), 4, 2)
        self.ed_leg_fs = QLineEdit("9")
        fmt.addWidget(self.ed_leg_fs, 4, 3)

        self.cb_ylim = QCheckBox("Set Y limits")
        self.cb_ylim.setChecked(False)
        fmt.addWidget(self.cb_ylim, 5, 0)
        self.ed_ymin = QLineEdit("")
        self.ed_ymax = QLineEdit("")
        fmt.addWidget(self.ed_ymin, 5, 1)
        fmt.addWidget(self.ed_ymax, 5, 2)

        L.addWidget(fmt_box)

        # actions
        row2 = QHBoxLayout()
        self.btn_plot = QPushButton("Plot")
        self.btn_plot.clicked.connect(self.plot_quantity)
        self.btn_save = QPushButton("Save computed CSV")
        self.btn_save.clicked.connect(self.save_computed_csv)
        row2.addWidget(self.btn_plot)
        row2.addWidget(self.btn_save)
        L.addLayout(row2)

        self.lbl_status = QLabel("")
        self.lbl_status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        L.addWidget(self.lbl_status)
        L.addStretch(1)

        self.canvas = MplCanvas()
        main.addWidget(left, 1)
        main.addWidget(self.canvas, 3)

        self._update_yaxis_hint_label()

    # --------------------------
    # Helpers
    # --------------------------
    def _parse_float(self, txt: str, default=None):
        t = (txt or "").strip()
        if t == "":
            return default
        try:
            return float(t)
        except Exception:
            return default

    def _parse_int(self, txt: str, default=None):
        t = (txt or "").strip()
        if t == "":
            return default
        try:
            return int(float(t))
        except Exception:
            return default

    def get_peak_label(self, pid: int) -> str:
        ed = self.peak_label_edits.get(pid)
        if ed is None:
            return f"Peak {pid}"
        s = ed.text().strip()
        return s if s else f"Peak {pid}"

    def _unit_for_column(self, col: str) -> str:
        # crude but works for your fit CSV naming
        if col.endswith("_eV") or col in ("cen_eV", "fwhm_eV", "wid_eV"):
            return "eV"
        if col == "power_uW":
            return "µW"
        # intensities etc
        return "a.u."

    def _update_yaxis_hint_label(self):
        col = self.cb_ycol.currentText()
        mode = self.cb_mode.currentIndex()
        unit = self._unit_for_column(col)

        if mode == 0:
            self.lbl_yhint.setText("Y-axis: polarization (dimensionless, -1 to 1). Uses selected column as I.")
        else:
            self.lbl_yhint.setText(f"Y-axis: Δ = ({col})σ+ − ({col})σ−  (unit: {unit})")

    # --------------------------
    # Loading
    # --------------------------
    def load_csv(self, which: int):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV files (*.csv)")
        if not path:
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Read error", str(e))
            return

        fname = os.path.basename(path)
        is_plus = ("72.5" in fname)
        is_minus = ("117.5" in fname)
        if not (is_plus or is_minus):
            QMessageBox.warning(self, "Filename rule not matched",
                                f"Filename must contain '72.5' (σ+) or '117.5' (σ−).\nGot: {fname}")
            return

        required = {"power_uW", "peak_id"}
        if not required.issubset(set(df.columns)):
            QMessageBox.critical(self, "Missing columns",
                                 f"CSV must contain columns {required}.\nFound: {list(df.columns)}")
            return

        if "success" in df.columns:
            df = df[df["success"] == 1].copy()

        df["power_uW"] = pd.to_numeric(df["power_uW"], errors="coerce")
        df["peak_id"] = pd.to_numeric(df["peak_id"], errors="coerce")
        df = df.dropna(subset=["power_uW", "peak_id"]).copy()
        df["peak_id"] = df["peak_id"].astype(int)

        if is_plus:
            self.df_plus = df
            self.path_plus = path
        if is_minus:
            self.df_minus = df
            self.path_minus = path

        self.update_ui_after_load()

    def update_ui_after_load(self):
        parts = []
        if self.df_plus is not None:
            parts.append(f"σ+ loaded: {os.path.basename(self.path_plus)} (rows={len(self.df_plus)})")
        if self.df_minus is not None:
            parts.append(f"σ− loaded: {os.path.basename(self.path_minus)} (rows={len(self.df_minus)})")
        self.lbl_status.setText("\n".join(parts))

        # update selectable columns = intersection of columns in both
        cols = None
        if self.df_plus is not None:
            cols = set(self.df_plus.columns)
        if self.df_minus is not None:
            cols = set(self.df_minus.columns) if cols is None else (cols & set(self.df_minus.columns))

        if cols:
            preferred = ["integrated_intensity", "amp", "cen_eV", "fwhm_eV", "wid_eV"]
            candidates = [c for c in preferred if c in cols] + sorted([c for c in cols if c not in {"power_uW", "peak_id", "success"}])
            seen = set()
            candidates = [c for c in candidates if not (c in seen or seen.add(c))]
            if candidates:
                cur = self.cb_ycol.currentText()
                self.cb_ycol.clear()
                self.cb_ycol.addItems(candidates)
                if cur in candidates:
                    self.cb_ycol.setCurrentText(cur)

        if self.df_plus is None or self.df_minus is None:
            return

        peaks_common = sorted(set(self.df_plus["peak_id"].unique()) & set(self.df_minus["peak_id"].unique()))
        self.peak_list.clear()
        for pid in peaks_common:
            it = QListWidgetItem(f"Peak {pid}")
            it.setData(Qt.UserRole, int(pid))
            self.peak_list.addItem(it)

        self._populate_label_controls(peaks_common)

        # auto-select 1-3
        for i in range(self.peak_list.count()):
            pid = self.peak_list.item(i).data(Qt.UserRole)
            if pid in (1, 2, 3):
                self.peak_list.item(i).setSelected(True)

        self._update_yaxis_hint_label()

    def _populate_label_controls(self, peaks_common):
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.peak_label_edits.clear()

        for row, pid in enumerate(peaks_common):
            lab = QLabel(f"Peak {pid}")
            ed = QLineEdit(f"Peak {pid}")
            self.scroll_layout.addWidget(lab, row, 0)
            self.scroll_layout.addWidget(ed, row, 1)
            self.peak_label_edits[int(pid)] = ed

    # --------------------------
    # Compute (long format)
    # --------------------------
    def compute_long(self) -> pd.DataFrame:
        """
        Return long-format dataframe with matching on (peak_id, power_key).
        Always includes:
          peak_id, power_uW, Y_plus, Y_minus, polarization, delta
        Polarization uses selected column as "I" (as you requested).
        """
        if self.df_plus is None or self.df_minus is None:
            raise ValueError("Load both σ+ and σ− CSVs first.")

        col = self.cb_ycol.currentText()
        if col not in self.df_plus.columns or col not in self.df_minus.columns:
            raise ValueError(f"Selected column '{col}' must exist in BOTH CSVs.")

        dp = self.df_plus[["power_uW", "peak_id", col]].copy().rename(columns={col: "Y_plus"})
        dm = self.df_minus[["power_uW", "peak_id", col]].copy().rename(columns={col: "Y_minus"})

        ndec = int(self.sp_round.value())
        dp["power_key"] = dp["power_uW"].round(ndec)
        dm["power_key"] = dm["power_uW"].round(ndec)

        merged = pd.merge(dp, dm, on=["peak_id", "power_key"], how="inner")
        merged["power_uW"] = merged["power_key"]

        # power range
        pmin = self._parse_float(self.ed_pmin.text(), None)
        pmax = self._parse_float(self.ed_pmax.text(), None)
        if pmin is not None:
            merged = merged[merged["power_uW"] >= pmin]
        if pmax is not None:
            merged = merged[merged["power_uW"] <= pmax]

        # delta
        merged["delta"] = (merged["Y_plus"].astype(float) - merged["Y_minus"].astype(float))

        # polarization (treat selected column as intensity I)
        denom = (merged["Y_plus"].astype(float) + merged["Y_minus"].astype(float))
        numer = (merged["Y_plus"].astype(float) - merged["Y_minus"].astype(float))
        merged["polarization"] = np.where(np.abs(denom) > 0, numer / denom, np.nan)

        out = merged[["peak_id", "power_uW", "Y_plus", "Y_minus", "polarization", "delta"]].copy()
        out = out.dropna(subset=["power_uW"]).sort_values(["peak_id", "power_uW"])
        return out

    # --------------------------
    # Plot
    # --------------------------
    def plot_quantity(self):
        if self.df_plus is None or self.df_minus is None:
            QMessageBox.warning(self, "Missing input", "Load two CSV files first (σ+ and σ−).")
            return

        items = self.peak_list.selectedItems()
        if not items:
            QMessageBox.warning(self, "No peaks", "Select at least one peak_id.")
            return
        peaks = [int(it.data(Qt.UserRole)) for it in items]

        try:
            df = self.compute_long()
        except Exception as e:
            QMessageBox.critical(self, "Compute error", str(e))
            return

        fw = self._parse_float(self.ed_w.text(), 6.0)
        fh = self._parse_float(self.ed_h.text(), 4.0)
        self.canvas.resize_figure(fw, fh)

        ax = self.canvas.ax
        ax.clear()

        show_points = self.cb_show_points.isChecked()
        show_raw = self.cb_show_raw.isChecked()

        mode = self.cb_mode.currentIndex()
        col = self.cb_ycol.currentText()
        unit = self._unit_for_column(col)

        ykey = "polarization" if mode == 0 else "delta"
        ylabel = "Polarization (dimensionless)" if mode == 0 else f"Δ({col}) = ({col})σ+ − ({col})σ−  ({unit})"

        plotted_any = False
        for pid in peaks:
            sub = df[df["peak_id"] == pid]
            if sub.empty:
                continue

            x = sub["power_uW"].values
            y = sub[ykey].values
            name = self.get_peak_label(pid)

            if show_points:
                ax.plot(x, y, "o", label=f"{name}")
            else:
                ax.plot(x, y, "-", label=f"{name}")

            if show_raw:
                ax.plot(x, sub["Y_plus"].values, "--", label=f"{name} σ+ ({col})")
                ax.plot(x, sub["Y_minus"].values, "--", label=f"{name} σ− ({col})")

            plotted_any = True

        if not plotted_any:
            QMessageBox.warning(self, "No matched data",
                                "No matched (peak_id, power) pairs found after rounding/range filter.\n"
                                "Try reducing rounding decimals or expanding power range.")
            ax.clear()
            self.canvas.draw_idle()
            return

        if self.cb_logx.isChecked():
            ax.set_xscale("log")

        lab_fs = self._parse_int(self.ed_lab_fs.text(), 11)
        tick_fs = self._parse_int(self.ed_tick_fs.text(), 10)

        ax.set_xlabel("Power (µW)", fontsize=lab_fs)
        ax.set_ylabel(ylabel, fontsize=lab_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.grid(True, alpha=0.25)

        if self.cb_ylim.isChecked():
            ymin = self._parse_float(self.ed_ymin.text(), None)
            ymax = self._parse_float(self.ed_ymax.text(), None)
            if ymin is not None and ymax is not None and ymax > ymin:
                ax.set_ylim(ymin, ymax)

        # x padding
        try:
            sub_all = df[df["peak_id"].isin(peaks)]
            xmin = sub_all["power_uW"].min()
            xmax = sub_all["power_uW"].max()
            if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
                if self.cb_logx.isChecked():
                    ax.set_xlim(xmin / 1.1, xmax * 1.1)
                else:
                    pad = 0.03 * (xmax - xmin)
                    ax.set_xlim(xmin - pad, xmax + pad)
        except Exception:
            pass

        leg_loc = self.cb_leg_loc.currentText()
        leg_fs = self._parse_int(self.ed_leg_fs.text(), 9)
        ax.legend(fontsize=leg_fs, loc=leg_loc)

        left = self._parse_float(self.ed_left.text(), 0.14)
        right = self._parse_float(self.ed_right.text(), 0.98)
        bottom = self._parse_float(self.ed_bottom.text(), 0.16)
        top = self._parse_float(self.ed_top.text(), 0.95)
        self.canvas.fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

        self.canvas.draw_idle()

        nrows = len(df[df["peak_id"].isin(peaks)])
        parts = []
        parts.append(f"Mode: {'polarization' if mode == 0 else 'delta'}; column={col}")
        parts.append(f"Matched rows plotted: {nrows}")
        parts.append(f"Matching: power rounded to {self.sp_round.value()} decimals; inner-join on (peak_id, power)")
        self.lbl_status.setText("\n".join(parts))

    # --------------------------
    # Save
    # --------------------------
    def save_computed_csv(self):
        if self.df_plus is None or self.df_minus is None:
            QMessageBox.warning(self, "Missing input", "Load two CSV files first (σ+ and σ−).")
            return

        try:
            df = self.compute_long()
        except Exception as e:
            QMessageBox.critical(self, "Compute error", str(e))
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save computed CSV", "", "CSV files (*.csv)")
        if not path:
            return

        try:
            df.to_csv(path, index=False)
            QMessageBox.information(self, "Saved", path)
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())