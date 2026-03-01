# analyze_fit_gui.py

import sys
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QListWidget, QListWidgetItem,
    QCheckBox, QLineEdit, QMessageBox, QGridLayout,
    QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ------------------------------
# Matplotlib canvas
# ------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, width=7, height=5):
        self.fig = Figure(figsize=(width, height))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def resize_figure(self, width, height):
        self.fig.set_size_inches(width, height)
        self.draw()


# ------------------------------
# Main GUI
# ------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fit Result Analyzer GUI")
        self.resize(1300, 750)

        self.df = None
        self.peak_label_edits = {}   # peak_id -> QLineEdit

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        # LEFT PANEL
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.btn_load = QPushButton("Load Fit CSV")
        self.btn_load.clicked.connect(self.load_csv)
        left_layout.addWidget(self.btn_load)

        left_layout.addWidget(QLabel("Select Peaks"))
        self.peak_list = QListWidget()
        self.peak_list.setSelectionMode(QListWidget.MultiSelection)
        left_layout.addWidget(self.peak_list)

        left_layout.addWidget(QLabel("Select Quantity"))
        self.cb_position = QCheckBox("Peak Position (eV)")
        self.cb_linewidth = QCheckBox("Linewidth (FWHM)")
        self.cb_intensity = QCheckBox("Integrated Intensity")

        left_layout.addWidget(self.cb_position)
        left_layout.addWidget(self.cb_linewidth)
        left_layout.addWidget(self.cb_intensity)

        # Power range
        grid = QGridLayout()
        grid.addWidget(QLabel("Fit Power Min"), 0, 0)
        self.ed_pmin = QLineEdit("")
        grid.addWidget(self.ed_pmin, 0, 1)

        grid.addWidget(QLabel("Fit Power Max"), 1, 0)
        self.ed_pmax = QLineEdit("")
        grid.addWidget(self.ed_pmax, 1, 1)
        left_layout.addLayout(grid)

        # Fit toggle
        self.cb_enable_fit = QCheckBox("Enable Power-law Fit (I ~ P^x)")
        self.cb_enable_fit.setChecked(True)
        left_layout.addWidget(self.cb_enable_fit)

        # Axis scale
        self.cb_logx = QCheckBox("Log X")
        self.cb_logy = QCheckBox("Log Y")
        self.cb_logx.setChecked(True)
        self.cb_logy.setChecked(True)
        left_layout.addWidget(self.cb_logx)
        left_layout.addWidget(self.cb_logy)

        # Figure size
        left_layout.addWidget(QLabel("Figure Size"))
        size_grid = QGridLayout()
        size_grid.addWidget(QLabel("Width"), 0, 0)
        self.ed_width = QLineEdit("7")
        size_grid.addWidget(self.ed_width, 0, 1)

        size_grid.addWidget(QLabel("Height"), 1, 0)
        self.ed_height = QLineEdit("5")
        size_grid.addWidget(self.ed_height, 1, 1)
        left_layout.addLayout(size_grid)

        # ---------------- Legend label control ----------------
        self.group_labels = QGroupBox("Legend Labels (Peak ID → Name)")
        labels_layout = QVBoxLayout(self.group_labels)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_widget)
        self.scroll.setWidget(self.scroll_widget)

        labels_layout.addWidget(self.scroll)
        left_layout.addWidget(self.group_labels)

        # Plot
        self.btn_plot = QPushButton("Plot")
        self.btn_plot.clicked.connect(self.plot_selected)
        left_layout.addWidget(self.btn_plot)

        self.lbl_fit_result = QLabel("Power-law exponent x: ")
        left_layout.addWidget(self.lbl_fit_result)

        left_layout.addStretch()

        # RIGHT PANEL
        self.canvas = MplCanvas()
        main_layout.addWidget(left, 1)
        main_layout.addWidget(self.canvas, 3)

    # --------------------------
    # Load CSV
    # --------------------------
    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV files (*.csv)")
        if not path:
            return

        df = pd.read_csv(path)

        required = {"power_uW", "peak_id"}
        if not required.issubset(df.columns):
            QMessageBox.critical(self, "Error", "CSV missing required columns.")
            return

        if "success" in df.columns:
            df = df[df["success"] == 1]

        self.df = df
        self.populate_peaks()
        self.populate_label_controls()

    def populate_peaks(self):
        self.peak_list.clear()
        peaks = sorted(self.df["peak_id"].dropna().unique())
        for p in peaks:
            item = QListWidgetItem(f"Peak {int(p)}")
            item.setData(Qt.UserRole, int(p))
            self.peak_list.addItem(item)

    def populate_label_controls(self):
        # clear old widgets
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self.peak_label_edits.clear()

        peaks = sorted(self.df["peak_id"].dropna().unique())
        for row, pid in enumerate(peaks):
            pid = int(pid)
            self.scroll_layout.addWidget(QLabel(f"Peak {pid}"), row, 0)
            ed = QLineEdit(f"Peak {pid}")  # default label
            self.scroll_layout.addWidget(ed, row, 1)
            self.peak_label_edits[pid] = ed

    def get_peak_label(self, pid: int) -> str:
        ed = self.peak_label_edits.get(pid)
        if ed is None:
            return f"Peak {pid}"
        txt = ed.text().strip()
        return txt if txt else f"Peak {pid}"

    # --------------------------
    # Plot
    # --------------------------
    # analyze_fit_gui.py

import sys
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QListWidget, QListWidgetItem,
    QCheckBox, QLineEdit, QMessageBox, QGridLayout,
    QGroupBox, QScrollArea
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ------------------------------
# Matplotlib canvas
# ------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, width=7, height=5):
        self.fig = Figure(figsize=(width, height))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def resize_figure(self, width, height):
        self.fig.set_size_inches(width, height)
        self.draw()


# ------------------------------
# Main GUI
# ------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fit Result Analyzer GUI")
        self.resize(1300, 750)

        self.df = None
        self.peak_label_edits = {}   # peak_id -> QLineEdit

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QHBoxLayout(root)

        # LEFT PANEL
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.btn_load = QPushButton("Load Fit CSV")
        self.btn_load.clicked.connect(self.load_csv)
        left_layout.addWidget(self.btn_load)

        left_layout.addWidget(QLabel("Select Peaks"))
        self.peak_list = QListWidget()
        self.peak_list.setSelectionMode(QListWidget.MultiSelection)
        left_layout.addWidget(self.peak_list)

        left_layout.addWidget(QLabel("Select Quantity"))
        self.cb_position = QCheckBox("Peak Position (eV)")
        self.cb_linewidth = QCheckBox("Linewidth (FWHM)")
        self.cb_intensity = QCheckBox("Integrated Intensity")

        left_layout.addWidget(self.cb_position)
        left_layout.addWidget(self.cb_linewidth)
        left_layout.addWidget(self.cb_intensity)

        # Power range
        grid = QGridLayout()
        grid.addWidget(QLabel("Fit Power Min"), 0, 0)
        self.ed_pmin = QLineEdit("")
        grid.addWidget(self.ed_pmin, 0, 1)

        grid.addWidget(QLabel("Fit Power Max"), 1, 0)
        self.ed_pmax = QLineEdit("")
        grid.addWidget(self.ed_pmax, 1, 1)
        left_layout.addLayout(grid)

        # Fit toggle
        self.cb_enable_fit = QCheckBox("Enable Power-law Fit (I ~ P^x)")
        self.cb_enable_fit.setChecked(True)
        left_layout.addWidget(self.cb_enable_fit)

        # Axis scale
        self.cb_logx = QCheckBox("Log X")
        self.cb_logy = QCheckBox("Log Y")
        self.cb_logx.setChecked(True)
        self.cb_logy.setChecked(True)
        left_layout.addWidget(self.cb_logx)
        left_layout.addWidget(self.cb_logy)

        # Figure size
        left_layout.addWidget(QLabel("Figure Size"))
        size_grid = QGridLayout()
        size_grid.addWidget(QLabel("Width"), 0, 0)
        self.ed_width = QLineEdit("7")
        size_grid.addWidget(self.ed_width, 0, 1)

        size_grid.addWidget(QLabel("Height"), 1, 0)
        self.ed_height = QLineEdit("5")
        size_grid.addWidget(self.ed_height, 1, 1)
        left_layout.addLayout(size_grid)

        # ---------------- Legend label control ----------------
        self.group_labels = QGroupBox("Legend Labels (Peak ID → Name)")
        labels_layout = QVBoxLayout(self.group_labels)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_widget)
        self.scroll.setWidget(self.scroll_widget)

        labels_layout.addWidget(self.scroll)
        left_layout.addWidget(self.group_labels)

        # Plot
        self.btn_plot = QPushButton("Plot")
        self.btn_plot.clicked.connect(self.plot_selected)
        left_layout.addWidget(self.btn_plot)

        self.lbl_fit_result = QLabel("Power-law exponent x: ")
        left_layout.addWidget(self.lbl_fit_result)

        left_layout.addStretch()

        # RIGHT PANEL
        self.canvas = MplCanvas()
        main_layout.addWidget(left, 1)
        main_layout.addWidget(self.canvas, 3)

    # --------------------------
    # Load CSV
    # --------------------------
    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV files (*.csv)")
        if not path:
            return

        df = pd.read_csv(path)

        required = {"power_uW", "peak_id"}
        if not required.issubset(df.columns):
            QMessageBox.critical(self, "Error", "CSV missing required columns.")
            return

        if "success" in df.columns:
            df = df[df["success"] == 1]

        self.df = df
        self.populate_peaks()
        self.populate_label_controls()

    def populate_peaks(self):
        self.peak_list.clear()
        peaks = sorted(self.df["peak_id"].dropna().unique())
        for p in peaks:
            item = QListWidgetItem(f"Peak {int(p)}")
            item.setData(Qt.UserRole, int(p))
            self.peak_list.addItem(item)

    def populate_label_controls(self):
        # clear old widgets
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self.peak_label_edits.clear()

        peaks = sorted(self.df["peak_id"].dropna().unique())
        for row, pid in enumerate(peaks):
            pid = int(pid)
            self.scroll_layout.addWidget(QLabel(f"Peak {pid}"), row, 0)
            ed = QLineEdit(f"Peak {pid}")  # default label
            self.scroll_layout.addWidget(ed, row, 1)
            self.peak_label_edits[pid] = ed

    def get_peak_label(self, pid: int) -> str:
        ed = self.peak_label_edits.get(pid)
        if ed is None:
            return f"Peak {pid}"
        txt = ed.text().strip()
        return txt if txt else f"Peak {pid}"

    # --------------------------
    # Plot
    # --------------------------
    def plot_selected(self):
        if self.df is None:
            return

        # Resize figure
        try:
            w = float(self.ed_width.text())
            h = float(self.ed_height.text())
            self.canvas.resize_figure(w, h)
        except:
            pass

        selected_items = self.peak_list.selectedItems()
        if not selected_items:
            return

        peaks = [item.data(Qt.UserRole) for item in selected_items]

        self.canvas.ax.clear()
        self.lbl_fit_result.setText("Power-law exponent x: ")

        power = self.df["power_uW"]

        try:
            pmin = float(self.ed_pmin.text()) if self.ed_pmin.text() else power.min()
            pmax = float(self.ed_pmax.text()) if self.ed_pmax.text() else power.max()
        except:
            pmin, pmax = power.min(), power.max()

        mask_range = (power >= pmin) & (power <= pmax)

        fit_results = []

        for pid in peaks:
            sub = self.df[(self.df["peak_id"] == pid) & mask_range]
            if sub.empty:
                continue

            label_base = self.get_peak_label(pid)

            if self.cb_position.isChecked():
                self.canvas.ax.plot(sub["power_uW"], sub["cen_eV"], "o",
                                    label=f"{label_base} position")

            if self.cb_linewidth.isChecked():
                self.canvas.ax.plot(sub["power_uW"], sub["fwhm_eV"], "o",
                                    label=f"{label_base} FWHM")

            if self.cb_intensity.isChecked():
                self.canvas.ax.plot(sub["power_uW"], sub["integrated_intensity"], "o",
                                    label=f"{label_base}")

                # power-law fit (optional)
                if self.cb_enable_fit.isChecked() and len(sub) > 2:
                    P = sub["power_uW"].values
                    I = sub["integrated_intensity"].values
                    valid = (P > 0) & (I > 0)
                    P, I = P[valid], I[valid]

                    if len(P) > 2:
                        logP = np.log10(P)
                        logI = np.log10(I)

                        coeff = np.polyfit(logP, logI, 1)
                        slope = coeff[0]
                        fit_I = 10 ** coeff[1] * P ** slope

                        self.canvas.ax.plot(P, fit_I, "--",
                                            label=f"{label_base} fit (x={slope:.3f})")

                        fit_results.append(f"{label_base}: x≈{slope:.3f}")

        if fit_results:
            self.lbl_fit_result.setText(" | ".join(fit_results))

        if self.cb_logx.isChecked():
            self.canvas.ax.set_xscale("log")
        if self.cb_logy.isChecked():
            self.canvas.ax.set_yscale("log")

        self.canvas.ax.set_xlabel("Power (µW)")
        self.canvas.ax.legend()
        self.canvas.draw()


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())