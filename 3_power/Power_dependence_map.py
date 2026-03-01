import sys
import os
import glob
import itertools
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QFileDialog, QListWidget, QListWidgetItem, QDialog, 
                             QGroupBox, QLabel, QLineEdit, QGridLayout, QComboBox, QCheckBox,
                             QMessageBox, QSplitter, QRadioButton, QTableWidget, QTableWidgetItem,
                             QStyle, QShortcut)  # <- 추가: QStyle, QShortcut
from PyQt5.QtGui import QFont, QKeySequence, QImage  # <- 추가: QKeySequence, QImage
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.transforms import Bbox  # <- 추가

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
from datetime import datetime
import matplotlib.colors as colors
import xmltodict
import pylab as pl

import xmltodict
import pylab as pl

# SPE3read 클래스를 새 버전(SPE3map 기반)으로 교체
class SPE3read:
    """
    Class which handles the reading of SPE3 files, based on SPE3map.
    Reads metadata from the XML footer to determine wavelength and other parameters.
    """
    SPEVERSIONOFFSET = 1992
    XMLFOOTEROFFSETPOS = 678
    DATAOFFSET = 4100

    def __init__(self, fname=None):
        self._fid = None
        self.fname = fname
        self.wavelength = None
        self.data = None
        self.SPEversion = 0
        self._footerInfo = {}
        self.nbOfFrames = 0
        self.dataType = np.uint16
        self.frameSize = 0
        self.frameStride = 0
        self.regionSize = None
        self.width = 0
        self.height = 0

        if fname is not None:
            self.openFile(fname)

        if self._fid:
            try:
                self.readData()
            except Exception as e:
                print(f"Error processing file {self.fname}: {e}")
            finally:
                self._fid.close()

    def openFile(self, fname):
        self._fname = fname
        self._fid = open(fname, "rb")

    def _readAtNumpy(self, pos, size, ntype):
        self._fid.seek(pos)
        return pl.fromfile(self._fid, ntype, int(size))

    def readData(self):
        self._read_legacy_header() # Read basic info first for fallback
        try:
            self._readSPEversion()
            if self.SPEversion < 3:
                raise ValueError(f"SPE version is {self.SPEversion}, which is not supported by XML footer reading.")
            
            self._readXMLfooter()
            self._readRegionSize() # Read region size before wavelength for polynomial calculation
            self._readWavelengths()
            self._readFramesInfo()
            self._readArray()
        except Exception as e:
            print(f"Info: Could not read via XML footer ({e}). Falling back to legacy header reading.")
            self._read_legacy_data()
            self.wavelength = None # No wavelength info in legacy mode

    def _readSPEversion(self):
        self.SPEversion = self._readAtNumpy(self.SPEVERSIONOFFSET, 1, pl.float32)[0]

    def _readXMLfooter(self):
        XMLfooterPos = self._readAtNumpy(self.XMLFOOTEROFFSETPOS, 1, pl.uint64)[0]
        if XMLfooterPos == 0:
            raise ValueError("XML footer position is zero.")
        self._fid.seek(XMLfooterPos)
        xml_bytes = self._fid.read()
        # Clean up non-printable characters that can break parsing
        xml_string = ''.join(char for char in xml_bytes.decode('utf-8', errors='ignore') if char.isprintable())
        if not xml_string.strip().endswith('>'):
             raise ValueError("Incomplete XML footer.")
        self._footerInfo = xmltodict.parse(xml_string)

    def _readWavelengths(self):
        try:
            # First, try direct wavelength text
            wavelengthStr = self._footerInfo['SpeFormat']['Calibrations']['WavelengthMapping']['Wavelength']['#text']
            self.wavelength = pl.array([float(w) for w in wavelengthStr.split(',')])
            print(f"Wavelength data loaded directly for {os.path.basename(self.fname)}.")
        except (KeyError, TypeError):
            # If that fails, try polynomial calculation
            try:
                calib = self._footerInfo['SpeFormat']['Calibrations']['WavelengthMapping']['Wavelength']
                poly_coeffs_text = calib['Polynomial']['Coefficient']['#text']
                coeffs = np.fromstring(poly_coeffs_text, sep=',')
                pixel_indices = np.arange(self.width)
                # The polynomial is evaluated as c0 + c1*x + c2*x^2 ...
                self.wavelength = np.polyval(coeffs[::-1], pixel_indices)
                print(f"Wavelength data calculated from polynomial for {os.path.basename(self.fname)}.")
            except Exception as e:
                print(f"Warning: Could not extract wavelength from XML for {self.fname}. {e}. Using pixel indices.")
                self.wavelength = None

    def _readFramesInfo(self):
        block = self._footerInfo['SpeFormat']['DataFormat']['DataBlock']
        self.nbOfFrames = int(block['@count'])
        dataTypeName = block['@pixelFormat']
        possibleDataTypes = {
            'MonochromeUnsigned16': np.uint16,
            'MonochromeUnsigned32': np.uint32,
            'MonochromeFloat32': np.float32
        }
        self.dataType = possibleDataTypes.get(dataTypeName, np.float32)
        self.frameSize = int(block['@size'])
        self.frameStride = int(block['@stride'])

    def _readRegionSize(self):
        roi_data = self._footerInfo['SpeFormat']['DataFormat']['DataBlock']['DataBlock']
        if isinstance(roi_data, list):
            # If multiple ROIs, use the first one
            self.height = int(roi_data[0]['@height'])
            self.width = int(roi_data[0]['@width'])
        else:
            self.height = int(roi_data['@height'])
            self.width = int(roi_data['@width'])
        self.regionSize = (self.height, self.width)

    def _readArray(self):
        self.data = []
        bytes_per_pixel = np.dtype(self.dataType).itemsize
        num_pixels_per_frame = self.width * self.height
        
        for frameNb in range(self.nbOfFrames):
            pos = self.DATAOFFSET + frameNb * self.frameStride
            frameData = self._readAtNumpy(pos, num_pixels_per_frame, self.dataType)
            
            if frameData.size == num_pixels_per_frame:
                self.data.append(frameData.reshape(self.regionSize))
            else:
                print(f"Warning: Frame {frameNb} data size mismatch. Skipping.")
        self.data = np.array(self.data)

    def _read_legacy_header(self):
        self.width = self._readAtNumpy(42, 1, np.uint16)[0]
        self.height = self._readAtNumpy(656, 1, np.uint16)[0]
        self.nbOfFrames = self._readAtNumpy(1446, 1, np.int32)[0]
        data_type_code = self._readAtNumpy(108, 1, np.int16)[0]
        dtype_map = {0: np.float32, 1: np.int32, 2: np.int16, 3: np.uint16, 8: np.uint32}
        self.dataType = dtype_map.get(data_type_code, np.float32)
        self.regionSize = (self.height, self.width)

    def _read_legacy_data(self):
        data_size = self.width * self.height * self.nbOfFrames
        self.data = self._readAtNumpy(self.DATAOFFSET, data_size, self.dataType)
        if self.nbOfFrames > 0 and self.data.size > 0:
            self.data = self.data.reshape(self.nbOfFrames, self.height, self.width)
        else:
            self.data = np.array([])


class FittingDialog(QDialog):
    # Signal to send fit results back to the main app
    fit_results_updated = pyqtSignal(float, list)

    def __init__(self, power_val, spectrum_data, energy_axis, is_energy_axis, image_save_dir, parent=None):
        super().__init__(parent)
        self.power_val = power_val
        self.spectrum_data = spectrum_data
        self.energy_axis = energy_axis
        self.is_energy_axis = is_energy_axis
        self.image_save_dir = image_save_dir
        
        self.setWindowTitle(f"Fitting for {power_val:.4f} uW")
        self.setGeometry(200, 200, 900, 800)

        self.num_peaks = 1
        self.initial_guesses = []
        self.is_picking_mode = False

        self.init_ui()
        self.plot_spectrum()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Vertical)

        # Top part: Plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

        # Bottom part: Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # --- Top Controls ---
        top_controls_layout = QGridLayout()
        top_controls_layout.addWidget(QLabel("Number of Peaks:"), 0, 0)
        self.num_peaks_spinbox = QComboBox()
        self.num_peaks_spinbox.addItems([str(i) for i in range(1, 11)])
        self.num_peaks_spinbox.currentIndexChanged.connect(self.update_num_peaks)
        top_controls_layout.addWidget(self.num_peaks_spinbox, 0, 1)

        top_controls_layout.addWidget(QLabel("Background Offset:"), 1, 0)
        self.bg_offset_edit = QLineEdit("0")
        top_controls_layout.addWidget(self.bg_offset_edit, 1, 1)

        self.start_picking_btn = QPushButton("Set Initial Guesses by Clicking")
        self.start_picking_btn.clicked.connect(self.start_picking_mode)
        top_controls_layout.addWidget(self.start_picking_btn, 0, 2)

        self.fit_button = QPushButton("Perform Fit")
        self.fit_button.clicked.connect(self.perform_fit)
        self.fit_button.setEnabled(False)
        top_controls_layout.addWidget(self.fit_button, 1, 2)
        
        self.instruction_label = QLabel("Instructions: 1. Set number of peaks. 2. Click 'Set Initial Guesses'. 3. Click on the plot for each peak's center. 4. Click 'Perform Fit'.")
        self.instruction_label.setWordWrap(True)
        
        controls_layout.addLayout(top_controls_layout)
        controls_layout.addWidget(self.instruction_label)

        # --- Bounds Table ---
        self.bounds_table = QTableWidget()
        self.bounds_table.setColumnCount(6)
        self.bounds_table.setHorizontalHeaderLabels(['Amp Min', 'Amp Max', 'Cen Min', 'Cen Max', 'Wid Min', 'Wid Max'])
        self.update_num_peaks(0) # Initialize for 1 peak
        controls_layout.addWidget(QLabel("Fit Parameter Bounds:"))
        controls_layout.addWidget(self.bounds_table)

        splitter.addWidget(plot_widget)
        splitter.addWidget(controls_widget)
        splitter.setSizes([450, 350])
        main_layout.addWidget(splitter)

    def update_num_peaks(self, index):
        self.num_peaks = int(self.num_peaks_spinbox.currentText())
        self.bounds_table.setRowCount(self.num_peaks)
        self.bounds_table.setVerticalHeaderLabels([f"Peak {i+1}" for i in range(self.num_peaks)])
        self.initial_guesses = []
        self.fit_button.setEnabled(False)
        self.instruction_label.setText(f"Click 'Set Initial Guesses' then click on the plot {self.num_peaks} time(s).")
        self.plot_spectrum() # Redraw to clear old markers

    def start_picking_mode(self):
        self.is_picking_mode = True
        self.initial_guesses = []
        self.plot_spectrum() # Redraw to clear old markers
        self.instruction_label.setText(f"Click on the plot to mark the center of {self.num_peaks} peak(s). {len(self.initial_guesses)}/{self.num_peaks} selected.")

    def on_plot_click(self, event):
        if not (self.is_picking_mode and event.inaxes and len(self.initial_guesses) < self.num_peaks):
            return

        x_clicked, y_clicked = event.xdata, event.ydata
        current_peak_index = len(self.initial_guesses)
        self.initial_guesses.append({'center': x_clicked, 'amplitude': y_clicked})
        
        self.instruction_label.setText(f"Click on the plot to mark the center of {self.num_peaks} peak(s). {len(self.initial_guesses)}/{self.num_peaks} selected.")
        
        # Mark the selected point and populate bounds
        self.ax.plot(x_clicked, y_clicked, 'r+', markersize=10)
        self.canvas.draw()
        self.populate_default_bounds(current_peak_index, x_clicked, y_clicked)

        if len(self.initial_guesses) == self.num_peaks:
            self.is_picking_mode = False
            self.fit_button.setEnabled(True)
            self.instruction_label.setText("Initial guesses are set. You can now 'Perform Fit'.")

    def populate_default_bounds(self, peak_index, cen_guess, amp_guess_raw):
        try:
            bg_offset = float(self.bg_offset_edit.text())
        except ValueError:
            bg_offset = 0
        
        amp_guess = amp_guess_raw - bg_offset
        energy_range = self.energy_axis.max() - self.energy_axis.min()

        # [Amp Min, Amp Max, Cen Min, Cen Max, Wid Min, Wid Max]
        defaults = [
            0,                  # Amp Min
            amp_guess * 2,      # Amp Max
            cen_guess - energy_range * 0.05, # Cen Min
            cen_guess + energy_range * 0.05, # Cen Max
            0.0001,             # Wid Min
            energy_range * 0.2  # Wid Max
        ]

        for col, value in enumerate(defaults):
            self.bounds_table.setItem(peak_index, col, QTableWidgetItem(f"{value:.4f}"))

    def plot_spectrum(self, fit_results=None):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.plot(self.energy_axis, self.spectrum_data, label=f"Data ({self.power_val:.4f} uW)")
        
        if fit_results:
            self.ax.plot(self.energy_axis, fit_results['total_fit'], 'r-', label='Total Fit')
            for i, peak_curve in enumerate(fit_results['individual_peaks']):
                self.ax.plot(self.energy_axis, peak_curve, '--', label=f'Peak {i+1}')

        if self.is_energy_axis:
            self.ax.set_xlabel("Energy (eV)")
        else:
            self.ax.set_xlabel("Pixel Index")
        self.ax.set_ylabel("Intensity")
        self.ax.set_title(f"Spectrum at {self.power_val:.4f} uW")
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    @staticmethod
    def gaussian(x, amp, cen, wid):
        return amp * np.exp(-((x - cen)**2) / (2 * wid**2))

    def multi_gaussian(self, x, *params):
        offset = float(self.bg_offset_edit.text())
        result = np.full_like(x, offset, dtype=float)
        num_peaks = len(params) // 3
        for i in range(num_peaks):
            amp, cen, wid = params[i*3], params[i*3+1], params[i*3+2]
            result += self.gaussian(x, amp, cen, wid)
        return result

    def perform_fit(self):
        if len(self.initial_guesses) != self.num_peaks:
            QMessageBox.warning(self, "Fit Error", "Please set initial guesses for all peaks first.")
            return

        try:
            bg_offset = float(self.bg_offset_edit.text())
            
            # Read bounds from the table
            lower_bounds = []
            upper_bounds = []
            for i in range(self.num_peaks):
                # Amp (min, max), Cen (min, max), Wid (min, max)
                l_amp = float(self.bounds_table.item(i, 0).text())
                u_amp = float(self.bounds_table.item(i, 1).text())
                l_cen = float(self.bounds_table.item(i, 2).text())
                u_cen = float(self.bounds_table.item(i, 3).text())
                l_wid = float(self.bounds_table.item(i, 4).text())
                u_wid = float(self.bounds_table.item(i, 5).text())
                lower_bounds.extend([l_amp, l_cen, l_wid])
                upper_bounds.extend([u_amp, u_cen, u_wid])
            bounds = (lower_bounds, upper_bounds)

        except (ValueError, AttributeError) as e:
            QMessageBox.warning(self, "Input Error", f"Please ensure all bounds are valid numbers. Error: {e}")
            return

        # Flatten initial guesses
        p0 = []
        for guess in self.initial_guesses:
            p0.extend([guess['amplitude'] - bg_offset, guess['center'], 0.005])

        try:
            popt, pcov = curve_fit(self.multi_gaussian, self.energy_axis, self.spectrum_data, p0=p0, bounds=bounds, maxfev=10000)
            
            total_fit_curve = self.multi_gaussian(self.energy_axis, *popt)
            individual_peaks_curves = []
            calculated_results = []

            for i in range(self.num_peaks):
                params = popt[i*3 : (i+1)*3]
                amp, cen, wid = params[0], params[1], params[2]
                peak_curve = self.gaussian(self.energy_axis, amp, cen, wid) + bg_offset
                individual_peaks_curves.append(peak_curve)
                fwhm = 2 * np.sqrt(2 * np.log(2)) * wid
                integrated_intensity = amp * wid * np.sqrt(2 * np.pi)
                calculated_results.append({'peak_id': i + 1, 'position': cen, 'fwhm': fwhm, 'integrated_intensity': integrated_intensity})

            fit_plot_data = {'total_fit': total_fit_curve, 'individual_peaks': individual_peaks_curves}
            self.plot_spectrum(fit_plot_data)
            
            success_message = "Gaussian fit completed successfully."
            try:
                if self.image_save_dir:
                    image_filename = f"fit_{self.power_val:.4f}uW.png"
                    save_path = os.path.join(self.image_save_dir, image_filename)
                    self.figure.savefig(save_path, dpi=150)
                    full_path = os.path.abspath(self.image_save_dir)
                    success_message += f"\n\nImage saved in:\n{full_path}"
            except Exception as e:
                print(f"Warning: Could not save fit graph image. {e}")

            QMessageBox.information(self, "Fit Success", success_message)
            self.fit_results_updated.emit(self.power_val, calculated_results)

        except RuntimeError as e:
            QMessageBox.warning(self, "Fit Error", f"Could not find optimal parameters: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Fit Error", f"An unexpected error occurred during fitting: {e}")


class PowerDependenceApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Power Dependence Analyzer')
        self.setGeometry(100, 100, 1300, 800)

        self.spectra_data = []
        self.power_values = []
        self.energy_axis = None
        self.is_energy_axis = False
        self.all_fit_results = {}
        self.recovery_file = 'fit_recovery.json'
        self.current_plot_mode = '2d_map'

        # 내부 플롯 복사용 참조
        self.current_axes = None
        self.current_colorbar = None

        # NEW: exposure times and background/normalize flags
        self.exposure_times = []        # list of exposure times (seconds) per spectrum (None if absent)
        self.background_counts = 0.0    # scalar background (counts) to subtract
        self.normalize_by_exposure = False

        # 날짜 기준 이미지 저장 폴더 이름 생성
        datestamp = datetime.now().strftime("%Y%m%d")
        self.fit_image_dir = f'fit_images_{datestamp}'
        os.makedirs(self.fit_image_dir, exist_ok=True)

        self.init_ui()
        self.check_for_recovery()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        main_layout = QHBoxLayout(main_widget)
        
        # Left panel for controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(350)

        # Controls Group
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_group)
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        controls_layout.addWidget(self.load_button)

        # NEW: Background input (single constant counts) and normalize checkbox
        bg_row = QWidget()
        bg_layout = QHBoxLayout(bg_row)
        bg_layout.setContentsMargins(0,0,0,0)
        bg_layout.addWidget(QLabel("Background (counts):"))
        self.background_input = QLineEdit()
        self.background_input.setFixedWidth(80)
        self.background_input.setPlaceholderText("0")
        bg_layout.addWidget(self.background_input)
        controls_layout.addWidget(bg_row)

        self.normalize_cb = QCheckBox("Normalize by exposure ( (I-bg)/t )")
        controls_layout.addWidget(self.normalize_cb)

        # NEW: Correction term input (counts/sec) with default 100 (user editable)
        corr_row = QWidget()
        corr_layout = QHBoxLayout(corr_row)
        corr_layout.setContentsMargins(0,0,0,0)
        corr_layout.addWidget(QLabel("Correction (count/s):"))
        self.correction_input = QLineEdit()
        self.correction_input.setFixedWidth(80)
        self.correction_input.setText("100")   # default (사용자가 바꿀 수 있음)
        self.correction_input.setToolTip("normalized = (I - background) / exposure + correction")
        corr_layout.addWidget(self.correction_input)
        controls_layout.addWidget(corr_row)

        left_layout.addWidget(controls_group)

        # Spectra list
        spectra_group = QGroupBox("Spectra List for Fitting")
        spectra_layout = QVBoxLayout(spectra_group)
        self.spectra_list_widget = QListWidget()
        self.spectra_list_widget.itemDoubleClicked.connect(self.open_fitting_by_index)
        spectra_layout.addWidget(self.spectra_list_widget)
        left_layout.addWidget(spectra_group)

        # 2D Map Controls
        map_controls_group = QGroupBox("2D Map Controls")
        map_controls_layout = QGridLayout(map_controls_group)
        
        map_controls_layout.addWidget(QLabel("Color Scale:"), 0, 0)
        self.color_linear_rb = QRadioButton("Linear")
        self.color_log_rb = QRadioButton("Log")
        self.color_log_rb.setChecked(True)
        map_controls_layout.addWidget(self.color_linear_rb, 0, 1)
        map_controls_layout.addWidget(self.color_log_rb, 0, 2)

        map_controls_layout.addWidget(QLabel("Color Range (R):"), 1, 0)
        self.color_min_edit = QLineEdit()
        self.color_max_edit = QLineEdit()
        map_controls_layout.addWidget(QLabel("Min:"), 2, 0)
        map_controls_layout.addWidget(self.color_min_edit, 2, 1, 1, 2)
        map_controls_layout.addWidget(QLabel("Max:"), 3, 0)
        map_controls_layout.addWidget(self.color_max_edit, 3, 1, 1, 2)

        # Smoothing controls (sigma along power axis, sigma along energy axis)
        map_controls_layout.addWidget(QLabel("Smoothing σ (power, energy):"), 4, 0)
        smooth_row = QWidget()
        smooth_layout = QHBoxLayout(smooth_row)
        smooth_layout.setContentsMargins(0,0,0,0)
        self.smooth_power_edit = QLineEdit("1.0")   # default vertical smoothing
        self.smooth_power_edit.setFixedWidth(50)
        self.smooth_energy_edit = QLineEdit("3.0")  # default horizontal smoothing
        self.smooth_energy_edit.setFixedWidth(50)
        smooth_layout.addWidget(self.smooth_power_edit)
        smooth_layout.addWidget(QLabel(","))
        smooth_layout.addWidget(self.smooth_energy_edit)
        map_controls_layout.addWidget(smooth_row, 4, 1, 1, 2)
        left_layout.addWidget(map_controls_group)

        # Axes Control
        axes_control_group = QGroupBox("Axes Control")
        axes_control_layout = QGridLayout(axes_control_group)

        axes_control_layout.addWidget(QLabel("Y-Axis Scale (Power):"), 0, 0)
        self.y_axis_linear_rb = QRadioButton("Linear")
        self.y_axis_linear_rb.setChecked(True)
        self.y_axis_log_rb = QRadioButton("Log")
        axes_control_layout.addWidget(self.y_axis_linear_rb, 0, 1)
        axes_control_layout.addWidget(self.y_axis_log_rb, 0, 2)

        axes_control_layout.addWidget(QLabel("X-Axis Range (Energy):"), 1, 0, 1, 3)
        self.x_axis_min_edit = QLineEdit()
        self.x_axis_max_edit = QLineEdit()
        axes_control_layout.addWidget(QLabel("Min:"), 2, 0)
        axes_control_layout.addWidget(self.x_axis_min_edit, 2, 1, 1, 2)
        axes_control_layout.addWidget(QLabel("Max:"), 3, 0)
        axes_control_layout.addWidget(self.x_axis_max_edit, 3, 1, 1, 2)

        axes_control_layout.addWidget(QLabel("Y-Axis Range (Power):"), 4, 0, 1, 3)
        self.y_axis_min_edit = QLineEdit()
        self.y_axis_max_edit = QLineEdit()
        axes_control_layout.addWidget(QLabel("Min:"), 5, 0)
        axes_control_layout.addWidget(self.y_axis_min_edit, 5, 1, 1, 2)
        axes_control_layout.addWidget(QLabel("Max:"), 6, 0)
        axes_control_layout.addWidget(self.y_axis_max_edit, 6, 1, 1, 2)

        self.update_plot_btn = QPushButton("Update Plot")
        self.update_plot_btn.clicked.connect(self.plot_2d_map)
        self.full_scale_btn = QPushButton("Full Scale")
        self.full_scale_btn.clicked.connect(self.reset_plot_view)
        axes_control_layout.addWidget(self.update_plot_btn, 7, 0, 1, 2)
        axes_control_layout.addWidget(self.full_scale_btn, 7, 2, 1, 2)
        left_layout.addWidget(axes_control_group)

        # Fit Results Analysis
        fit_results_group = QGroupBox("Fit Results Analysis")
        fit_results_layout = QVBoxLayout(fit_results_group)

        # Analysis Type Radio Buttons
        analysis_type_group = QGroupBox("Analysis Type")
        analysis_type_layout = QGridLayout(analysis_type_group)
        self.plot_type_intensity_rb = QRadioButton("Intensity vs Power")
        self.plot_type_intensity_rb.setChecked(True)
        self.plot_type_position_rb = QRadioButton("Position vs Power")
        self.plot_type_fwhm_rb = QRadioButton("FWHM vs Power")
        self.plot_type_energy_diff_rb = QRadioButton("Energy Diff vs Power")  # 추가
        analysis_type_layout.addWidget(self.plot_type_intensity_rb, 0, 0)
        analysis_type_layout.addWidget(self.plot_type_position_rb, 0, 1)
        analysis_type_layout.addWidget(self.plot_type_fwhm_rb, 1, 0)
        analysis_type_layout.addWidget(self.plot_type_energy_diff_rb, 1, 1)  # 추가
        fit_results_layout.addWidget(analysis_type_group)

        # Peak Selection Group
        self.peak_selection_group = QGroupBox("Peak Selection")
        self.peak_selection_layout = QHBoxLayout(self.peak_selection_group)
        self.peak_checkboxes = {} # peak_id -> QCheckBox
        fit_results_layout.addWidget(self.peak_selection_group)

        self.plot_results_button = QPushButton("Plot Fit Results")
        self.plot_results_button.clicked.connect(self.plot_fit_results)
        fit_results_layout.addWidget(self.plot_results_button)

        # File I/O buttons
        file_io_layout = QHBoxLayout()
        self.load_csv_button = QPushButton("Load Fit Results from CSV")
        self.load_csv_button.clicked.connect(self.load_results_from_csv)
        self.save_csv_button = QPushButton("Save Fit Results to CSV")
        self.save_csv_button.clicked.connect(self.save_results_to_csv)
        file_io_layout.addWidget(self.load_csv_button)
        file_io_layout.addWidget(self.save_csv_button)
        fit_results_layout.addLayout(file_io_layout)
        
        left_layout.addWidget(fit_results_group)

        # Power Law Fit Group
        self.power_law_group = QGroupBox("Power Law Fit (I ~ P^x)")
        power_law_layout = QVBoxLayout(self.power_law_group)
        self.power_law_table = QTableWidget()
        self.power_law_table.setColumnCount(3)
        self.power_law_table.setHorizontalHeaderLabels(['Peak ID', 'Power Min (uW)', 'Power Max (uW)'])
        power_law_layout.addWidget(self.power_law_table)
        self.fit_power_law_button = QPushButton("Fit Power Law")
        self.fit_power_law_button.clicked.connect(self.fit_power_law)
        power_law_layout.addWidget(self.fit_power_law_button)
        self.power_law_group.setVisible(False) # Initially hidden
        left_layout.addWidget(self.power_law_group)

        left_layout.addStretch()

        # Right panel for the 2D plot
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        
        # Add the NavigationToolbar back
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        right_layout.addWidget(self.toolbar)

        # 추가: 버튼과 단축키(a)
        self.copy_ax_action = self.toolbar.addAction(self.style().standardIcon(QStyle.SP_DialogSaveButton),
                                             "Copy Axes+Colorbar")
        self.copy_ax_action.setToolTip("Copy only axes area with colorbar (shortcut: a)")
        self.copy_ax_action.triggered.connect(
            lambda: copy_axes_plus_colorbar_to_clipboard(getattr(self, "current_axes", None),
                                                         getattr(self, "current_colorbar", None))
        )
        QShortcut(QKeySequence("a"), self, activated=lambda:
            copy_axes_plus_colorbar_to_clipboard(getattr(self, "current_axes", None),
                                                 getattr(self, "current_colorbar", None))
        )
        
        right_layout.addWidget(self.canvas)
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])

        main_layout.addWidget(splitter)

    def on_plot_click(self, event):
        # 현재 플롯 모드가 '2d_map'이 아니면 아무 동작도 하지 않음 (확대/축소 가능)
        if self.current_plot_mode != '2d_map':
            return

        if event.inaxes is None or not self.spectra_data or event.ydata is None:
            return

        clicked_power = event.ydata
        power_array = np.asarray(self.power_values)

        # y축 스케일에 따라 가장 가까운 인덱스를 찾는 방식 변경
        if self.y_axis_log_rb.isChecked():
            # 로그 스케일일 경우, 값들의 로그 값과 클릭된 y좌표의 로그 값의 차이가 가장 작은 것을 찾음
            # 0 이하의 파워 값은 로그 변환 시 오류를 유발하므로 제외
            valid_powers = power_array[power_array > 0]
            if len(valid_powers) == 0: return
            
            log_powers = np.log(valid_powers)
            log_clicked_power = np.log(clicked_power)
            
            # 원래 power_array에서 해당 인덱스를 찾기 위해 np.where 사용
            closest_power_val = valid_powers[(np.abs(log_powers - log_clicked_power)).argmin()]
            idx = np.where(power_array == closest_power_val)[0][0]

        else:
            # 선형 스케일일 경우, 값의 차이가 가장 작은 것을 찾음
            idx = (np.abs(power_array - clicked_power)).argmin()
        
        selected_power = self.power_values[idx]
        selected_spectrum = self.spectra_data[idx]
        
        self.show_spectrum_dialog(selected_power, selected_spectrum)

    def update_peak_selection_ui(self):
        # Clear existing checkboxes and stretchers safely
        while self.peak_selection_layout.count():
            item = self.peak_selection_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self.peak_checkboxes.clear()

        if not self.all_fit_results:
            self.peak_selection_group.setVisible(False)
            return

        # Find all unique peak IDs
        unique_peak_ids = set()
        for results in self.all_fit_results.values():
            for peak_data in results:
                unique_peak_ids.add(peak_data['peak_id'])
        
        if not unique_peak_ids:
            self.peak_selection_group.setVisible(False)
            return

        self.peak_selection_group.setVisible(True)
        for peak_id in sorted(list(unique_peak_ids)):
            checkbox = QCheckBox(f"Peak {peak_id}")
            checkbox.setChecked(True)
            self.peak_selection_layout.addWidget(checkbox)
            self.peak_checkboxes[peak_id] = checkbox
        self.peak_selection_layout.addStretch()


    def show_spectrum_dialog(self, power_val, spectrum_data):
        # The dialog will be garbage collected if not held in a variable.
        dialog = FittingDialog(power_val, spectrum_data, self.energy_axis, self.is_energy_axis, self.fit_image_dir, self)
        # Connect the signal from the dialog to a slot in the main app
        dialog.fit_results_updated.connect(self.add_fit_result)
        dialog.exec_()

    def load_data(self):
        spe_files, _ = QFileDialog.getOpenFileNames(self,
                                                    "Select SPE Files to Load",
                                                    "",
                                                    "SPE Files (*.spe);;All Files (*.*)")
        if not spe_files:
            return

        # 데이터를 임시 리스트에 저장 (파일 경로도 포함)
        temp_data_list = []
        for f in spe_files:
            try:
                basename = os.path.basename(f)
                base_no_ext = os.path.splitext(basename)[0]
                parts = base_no_ext.split('_')

                # parse power exactly as written (keep unit text for display, convert to uW for numeric sorting)
                power_part = parts[0] if len(parts) >= 1 else base_no_ext
                # numeric conversion for sorting: handle nW/uW if present
                power_val_uW = None
                if power_part.endswith('uW'):
                    try:
                        power_val_uW = float(power_part[:-2])
                    except Exception:
                        power_val_uW = 0.0
                elif power_part.endswith('nW'):
                    try:
                        power_val_uW = float(power_part[:-2]) / 1000.0
                    except Exception:
                        power_val_uW = 0.0
                else:
                    try:
                        power_val_uW = float(power_part)  # assume already uW
                    except Exception:
                        power_val_uW = 0.0

                # parse exposure (if present)
                exposure_time = None
                if len(parts) >= 2 and parts[1].endswith('s'):
                    try:
                        exposure_time = float(parts[1][:-1])
                    except Exception:
                        exposure_time = None

                spe = SPE3read(f)
                if spe.data is not None and spe.data.size > 0:
                    spectrum = spe.data[0, 0, :].astype(float)
                    # store: (power_val_uW, spectrum, wavelength, width, filepath, exposure_time, power_label)
                    temp_data_list.append((power_val_uW, spectrum, spe.wavelength, spe.width, f, exposure_time, power_part))
                else:
                    print(f"Warning: No data loaded for file {f}")

            except Exception as e:
                print(f"Could not process file {f}: {e}")

        # 파워 값을 기준으로 데이터 정렬 (numeric uW used)
        temp_data_list.sort(key=lambda x: x[0])

        # 정렬된 데이터로 클래스 변수들을 채움
        self.spectra_data = [item[1] for item in temp_data_list]
        self.power_values = [item[0] for item in temp_data_list]
        self.exposure_times = [item[5] for item in temp_data_list]   # NEW

        # 리스트에 표시할 라벨 생성 (원본 파일명 기반, exposure가 있으면 괄호로 표시)
        self.spectra_display_labels = []
        self.spectra_list_widget.clear()
        for item in temp_data_list:
            filepath = item[4]
            power_label = item[6]  # original power part (e.g., '0.1022nW')
            exposure_part = item[5]
            if exposure_part is not None:
                display_label = f"{power_label} ({int(exposure_part)}s)"
            else:
                display_label = f"{power_label}"
            self.spectra_display_labels.append(display_label)
            self.spectra_list_widget.addItem(QListWidgetItem(display_label))

        # X축 (에너지/픽셀) 설정
        self.energy_axis = None
        self.is_energy_axis = False
        if self.spectra_data:
            first_wavelength = temp_data_list[0][2]
            first_width = temp_data_list[0][3]

            if first_wavelength is not None and len(first_wavelength) > 1 and np.all(first_wavelength > 0):
                self.energy_axis = 1240 / first_wavelength
                self.is_energy_axis = True
                print("Successfully loaded wavelength data. X-axis will be in Energy (eV).")
            else:
                self.energy_axis = np.arange(first_width)
                self.is_energy_axis = False
                print("Could not load wavelength data or wavelength is invalid. X-axis will be in Pixel Index.")

        self.reset_plot_view()
        self.update_list_with_fit_status()
        self.update_peak_selection_ui()

    def plot_2d_map(self):
        # ensure we have data
        if not getattr(self, 'spectra_data', None):
            return

        # build 2D data (float)
        try:
            data_2d = np.array([s.astype(float).copy() for s in self.spectra_data])
        except Exception as e:
            QMessageBox.warning(self, "Data Error", f"Could not prepare data array: {e}")
            return

        # use actual power values (no historical /2.0)
        pw = np.asarray(self.power_values, dtype=float)

        # length match and sort by power once
        if pw.shape[0] != data_2d.shape[0]:
            if pw.shape[0] < data_2d.shape[0]:
                pad_val = pw[-1] if pw.size else 0.0
                pw = np.concatenate([pw, np.full(data_2d.shape[0] - pw.shape[0], pad_val)])
            else:
                pw = pw[:data_2d.shape[0]]
        order = np.argsort(pw)
        pw = pw[order]
        data_2d = data_2d[order, :]

        # background/normalization
        bg_val = 0.0
        try:
            t = self.background_input.text().strip()
            if t != '':
                bg_val = float(t)
        except Exception:
            QMessageBox.warning(self, "Input Error", "Background must be a number. Using 0.")
            bg_val = 0.0
        if bg_val != 0.0:
            data_2d = data_2d - bg_val

        normalize_flag = bool(self.normalize_cb.isChecked())
        corr_val = 0.0
        try:
            t = self.correction_input.text().strip()
            if t != '':
                corr_val = float(t)
        except Exception:
            QMessageBox.warning(self, "Input Error", "Correction must be a number. Using 0.")
            corr_val = 0.0
        if normalize_flag and getattr(self, 'exposure_times', None):
            for i, expo in enumerate(self.exposure_times):
                tt = expo if (expo is not None and expo > 0) else 1.0
                data_2d[i, :] = data_2d[i, :] / tt
        if corr_val != 0.0:
            data_2d = data_2d + corr_val

        # --- Resample to uniform power grid before smoothing ---
        # helper: resample each energy column along power axis to uniform grid
        def _resample_power_grid(pw_in: np.ndarray, Z_in: np.ndarray, mode: str = 'log', Ny: int = None):
            pw_in = np.asarray(pw_in, dtype=float)
            Z_in = np.asarray(Z_in, dtype=float)  # shape: (Ny_in, Nx)
            Ny_in, Nx = Z_in.shape
            if Ny is None:
                Ny = max(200, Ny_in)  # target rows

            if mode == 'log':
                # use only positive powers
                mask_pos = pw_in > 0
                pw = pw_in[mask_pos]
                Z = Z_in[mask_pos, :]
                # log-uniform grid
                y_new = np.geomspace(pw.min(), pw.max(), Ny)
                x_old = np.log(pw)
                x_new = np.log(y_new)
            else:
                pw = pw_in
                Z = Z_in
                y_new = np.linspace(pw.min(), pw.max(), Ny)
                x_old = pw
                x_new = y_new

            # interpolate each energy column on the chosen grid
            out = np.empty((Ny, Nx), dtype=float)
            order = np.argsort(x_old)
            x_sorted = x_old[order]
            for j in range(Nx):
                col = Z[:, j][order]
                # remove NaN/Inf and duplicate x
                m = np.isfinite(col)
                xs = x_sorted[m]
                ys = col[m]
                if xs.size < 2:
                    out[:, j] = np.nan
                    continue
                # drop duplicate xs to make np.interp stable
                xsu, idx = np.unique(xs, return_index=True)
                ysu = ys[idx]
                out[:, j] = np.interp(x_new, xsu, ysu)
            return y_new, out

        # choose resampling mode based on Y-axis scale toggle
        resample_mode = 'log' if self.y_axis_log_rb.isChecked() else 'linear'
        # Z_in: 배경/정규화까지 적용된 data_2d
        y_uniform, Z_uniform = _resample_power_grid(pw, data_2d, mode=resample_mode, Ny=max(300, len(pw)))

        # --- Smoothing on the uniform grid ---
        try:
            s_pow = float(self.smooth_power_edit.text().strip())
        except Exception:
            s_pow = 1.0
        try:
            s_eng = float(self.smooth_energy_edit.text().strip())
        except Exception:
            s_eng = 3.0

        epsilon = 1e-12
        if self.color_log_rb.isChecked():
            safe = np.where(Z_uniform <= epsilon, epsilon, Z_uniform)
            logdat = np.log10(safe)
            smoothed_log = gaussian_filter(logdat, sigma=(s_pow, s_eng))
            Z = np.power(10.0, smoothed_log)
            Z = np.where(Z <= epsilon, epsilon, Z)
        else:
            Z = gaussian_filter(Z_uniform, sigma=(s_pow, s_eng))

        # --- Color normalization as before ---
        try:
            c_min = float(self.color_min_edit.text()) if self.color_min_edit.text().strip() else None
        except Exception:
            c_min = None
        try:
            c_max = float(self.color_max_edit.text()) if self.color_max_edit.text().strip() else None
        except Exception:
            c_max = None
        if self.color_log_rb.isChecked():
            if c_min is None or c_min <= 0:
                c_min = max(1e-12, np.nanmin(Z[Z > 0]))
            norm = colors.LogNorm(vmin=c_min, vmax=c_max)
        else:
            norm = colors.Normalize(vmin=c_min, vmax=c_max)

        # --- Axis ranges from UI (x_base as before) ---
        try:
            x_min_text = self.x_axis_min_edit.text().strip()
            x_max_text = self.x_axis_max_edit.text().strip()
            y_min_text = self.y_axis_min_edit.text().strip()
            y_max_text = self.y_axis_max_edit.text().strip()

            if self.is_energy_axis:
                x_base = self.energy_axis
                x_min = float(x_min_text) if x_min_text else float(np.min(x_base))
                x_max = float(x_max_text) if x_max_text else float(np.max(x_base))
            else:
                x_base = np.arange(Z.shape[1], dtype=float)
                x_min = float(x_min_text) if x_min_text else float(x_base.min())
                x_max = float(x_max_text) if x_max_text else float(x_base.max())

            y_min = float(y_min_text) if y_min_text else float(y_uniform.min())
            y_max = float(y_max_text) if y_max_text else float(y_uniform.max())
            if x_min >= x_max: x_min, x_max = min(x_min, x_max), max(x_min, x_max + 1e-6)
            if y_min >= y_max: y_min, y_max = min(y_min, y_max), max(y_min, y_max + 1e-6)
        except Exception:
            x_base = self.energy_axis if self.is_energy_axis else np.arange(Z.shape[1], dtype=float)
            x_min, x_max = float(x_base.min()), float(x_base.max())
            y_min, y_max = float(y_uniform.min()), float(y_uniform.max())

        # --- Compute cell edges and draw ---
        def _edges(vals: np.ndarray) -> np.ndarray:
            vals = np.asarray(vals, dtype=float)
            if vals.size == 1:
                return np.array([vals[0]-0.5, vals[0]+0.5], dtype=float)
            mids = 0.5*(vals[:-1] + vals[1:])
            first = vals[0] - (mids[0] - vals[0])
            last  = vals[-1] + (vals[-1] - mids[-1])
            return np.concatenate(([first], mids, [last]))

        Xe = _edges(x_base)
        Ye = _edges(y_uniform)

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        im = ax.pcolormesh(Xe, Ye, Z, cmap='RdBu_r', norm=norm, shading='auto')
        cbar = self.figure.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label('Log Scale Intensity' if self.color_log_rb.isChecked() else 'Linear Scale Intensity')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel('Power (uW)')
        ax.set_xlabel('Energy (eV)' if self.is_energy_axis else 'Pixel Index')
        if self.y_axis_log_rb.isChecked():
            ax.set_yscale('log')

        # 추가: 복사용 참조 저장
        self.current_axes = ax
        self.current_colorbar = cbar

        self.current_plot_mode = '2d_map'
        self.canvas.draw()
        return

    def reset_plot_view(self):
        if not self.spectra_data:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        data_2d = np.array(self.spectra_data)
        smoothed_data = gaussian_filter(data_2d, sigma=(1, 5))

        norm = colors.LogNorm() if self.color_log_rb.isChecked() else colors.Normalize()
        
        # is_energy_axis 플래그에 따라 축 범위와 라벨 설정
        if self.is_energy_axis:
            ax.set_xlabel('Energy (eV)')
            extent = [self.energy_axis[0], self.energy_axis[-1], min(self.power_values), max(self.power_values)]
        else:
            ax.set_xlabel('Pixel Index')
            # energy_axis가 실제로는 픽셀 인덱스일 경우
            extent = [self.energy_axis.min(), self.energy_axis.max(), min(self.power_values), max(self.power_values)]

        im = ax.imshow(smoothed_data, aspect='auto', origin='lower', extent=extent, norm=norm, cmap='RdBu_r')

        ax.set_ylabel('Power (uW)')
        if self.y_axis_log_rb.isChecked():
            ax.set_yscale('log')

        if self.color_log_rb.isChecked():
            cbar = self.figure.colorbar(im, ax=ax, orientation='vertical')
            cbar.set_label('Log Scale Intensity')
        else:
            cbar = self.figure.colorbar(im, ax=ax, orientation='vertical')
            cbar.set_label('Linear Scale Intensity')

        # X, Y 축 범위 설정
        if self.is_energy_axis:
            ax.set_xlim(self.energy_axis.min(), self.energy_axis.max())
        else:
            ax.set_xlim(self.energy_axis.min(), self.energy_axis.max())
        ax.set_ylim(min(self.power_values), max(self.power_values))

        # 추가: 복사용 참조 저장
        self.current_axes = ax
        self.current_colorbar = cbar

        self.current_plot_mode = '2d_map' 
        self.canvas.draw()

    def open_fitting_by_index(self, item):
        index = self.spectra_list_widget.row(item)
        if index >= 0 and index < len(self.power_values):
            power_val = self.power_values[index]
            spectrum_data = self.spectra_data[index]
            self.show_spectrum_dialog(power_val, spectrum_data)

    def load_results_from_csv(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Fit Results from CSV", "", "CSV Files (*.csv);;All Files (*.*)", options=options)
        if not file_name:
            return

        try:
            df = pd.read_csv(file_name)
            # 컬럼 이름에서 공백 제거
            df.columns = df.columns.str.strip()
            
            self.all_fit_results = {}
            # 'Power (uW)'로 그룹화하여 각 파워에 대한 피크 리스트 생성
            for power, group in df.groupby('Power (uW)'):
                peaks_list = []
                for _, row in group.iterrows():
                    peaks_list.append({
                        'peak_id': int(row['Peak_ID']),
                        'position': row['Position (eV)'],
                        'fwhm': row['FWHM (eV)'],
                        'integrated_intensity': row['Integrated_Intensity']
                    })
                self.all_fit_results[power] = peaks_list

            QMessageBox.information(self, "Load Success", "Fit results loaded successfully from CSV.")
            self.update_list_with_fit_status() # 목록 상태 업데이트
            self.update_peak_selection_ui() # 피크 선택 UI 업데이트
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Failed to load or parse CSV file: {e}")

    def plot_fit_results(self):
        if not self.all_fit_results:
            QMessageBox.warning(self, "No Data", "No fit results available to plot.")
            return

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

        # 어떤 데이터를 플롯할지 결정
        is_intensity_plot = self.plot_type_intensity_rb.isChecked()
        is_position_plot = self.plot_type_position_rb.isChecked()
        is_fwhm_plot = self.plot_type_fwhm_rb.isChecked()
        is_energy_diff_plot = self.plot_type_energy_diff_rb.isChecked()

        if is_energy_diff_plot:
            self.plot_energy_diff_vs_power()
            return

        # 데이터를 파워와 피크별로 정리
        self.plot_data = {}  # {peak_id: {'powers': [], 'y_values': []}}
        sorted_powers = sorted(self.all_fit_results.keys())

        for power in sorted_powers:
            peaks = self.all_fit_results[power]
            for peak_data in peaks:
                peak_id = peak_data['peak_id']
                if peak_id not in self.plot_data:
                    self.plot_data[peak_id] = {'powers': [], 'y_values': []}
                
                self.plot_data[peak_id]['powers'].append(power)
                self.plot_data[peak_id]['y_values'].append(peak_data[y_key])

        # 선택된 피크에 대해서만 플롯
        selected_peak_ids = {pid for pid, checkbox in self.peak_checkboxes.items() if checkbox.isChecked()}

        for peak_id, data in self.plot_data.items():
            if peak_id in selected_peak_ids and data['powers']:
                self.ax.plot(data['powers'], data['y_values'], marker='o', linestyle='none', label=f'Peak {peak_id}')

        self.ax.set_xlabel('Power (uW)')
        self.ax.set_ylabel(y_label)
        self.ax.set_title(title)
        self.ax.set_xscale('log')
        self.ax.set_yscale(y_scale)
        self.ax.legend(fontsize='large')
        self.ax.grid(True, which="both", ls="--")

        # Power Law Fit UI 업데이트
        if is_intensity_plot:
            self.power_law_group.setVisible(True)
            self.power_law_table.setRowCount(len(self.plot_data))
            for i, peak_id in enumerate(sorted(self.plot_data.keys())):
                id_item = QTableWidgetItem(str(peak_id))
                id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable) # Make ID non-editable
                self.power_law_table.setItem(i, 0, id_item)
                self.power_law_table.setItem(i, 1, QTableWidgetItem("")) # Power Min
                self.power_law_table.setItem(i, 2, QTableWidgetItem("")) # Power Max
        else:
            self.power_law_group.setVisible(False)

        self.current_plot_mode = 'analysis' # 플롯 모드 설정
        self.canvas.draw()

    def fit_power_law(self):
        if not hasattr(self, 'ax') or self.current_plot_mode != 'analysis' or not self.plot_type_intensity_rb.isChecked():
            QMessageBox.warning(self, "Warning", "Please plot 'Intensity vs Power' first.")
            return

        # 이전에 그려진 Power Law Fit 라인들만 제거
        lines_to_remove = [line for line in self.ax.lines if line.get_linestyle() == '--']
        for line in lines_to_remove:
            line.remove()

        # 기존 범례 핸들과 라벨 가져오기 (데이터 포인트에 대한 것)
        original_handles = [h for h in self.ax.get_legend().legend_handles if h.get_linestyle() != '--']
        original_labels = [f'Peak {pid}' for pid, cb in self.peak_checkboxes.items() if cb.isChecked()]
        
        new_handles = original_handles.copy()
        new_labels = original_labels.copy()

        for i in range(self.power_law_table.rowCount()):
            try:
                peak_id_item = self.power_law_table.item(i, 0)
                p_min_item = self.power_law_table.item(i, 1)
                p_max_item = self.power_law_table.item(i, 2)

                if not (peak_id_item and p_min_item and p_max_item and p_min_item.text() and p_max_item.text()):
                    continue

                peak_id = int(peak_id_item.text())
                
                # 현재 선택된(체크된) 피크가 아니면 건너뛰기
                if not self.peak_checkboxes.get(peak_id) or not self.peak_checkboxes[peak_id].isChecked():
                    continue

                p_min = float(p_min_item.text())
                p_max = float(p_max_item.text())

                if p_min <= 0:
                    p_min = 1e-9

                peak_data = self.plot_data[peak_id]
                powers = np.array(peak_data['powers'])
                intensities = np.array(peak_data['y_values'])

                mask = (powers >= p_min) & (powers <= p_max) & (powers > 0) & (intensities > 0)
                if np.sum(mask) < 2:
                    continue

                fit_powers = powers[mask]
                fit_intensities = intensities[mask]

                log_p = np.log10(fit_powers)
                log_i = np.log10(fit_intensities)
                
                slope, intercept = np.polyfit(log_p, log_i, 1)

                fit_line_p = np.logspace(np.log10(p_min), np.log10(p_max), 50)
                fit_line_i = 10**(slope * np.log10(fit_line_p) + intercept)

                # 원본 데이터 포인트의 색상 찾기
                original_line_color = None
                original_label_text = f'Peak {peak_id}'
                
                # new_labels에서 인덱스를 찾아 해당 핸들의 색상을 사용
                try:
                    label_index = new_labels.index(original_label_text)
                    original_line_color = new_handles[label_index].get_color()
                except ValueError:
                    continue # 원본 라벨을 찾을 수 없으면 건너뜀

                # 그래프에 피팅 라인 그리기
                fit_line, = self.ax.plot(fit_line_p, fit_line_i, '--', color=original_line_color)
                
                # 해당 피크의 라벨을 업데이트
                new_labels[label_index] = f'Peak {peak_id} (x={slope:.2f})'

            except (ValueError, TypeError) as e:
                QMessageBox.warning(self, "Input Error", f"Invalid input for Peak {peak_id_item.text() if peak_id_item else 'N/A'}. Please use numbers.\nError: {e}")
                return
        
        # 새 라벨로 범례 다시 생성
        self.ax.legend(handles=new_handles, labels=new_labels, fontsize='large')
        self.canvas.draw()

    def save_results_to_csv(self):
        if not self.all_fit_results:
            QMessageBox.warning(self, "No Data", "No fit results available to save.")
            return

        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Fit Results to CSV", "", "CSV Files (*.csv);;All Files (*.*)", options=options)
        if not file_name:
            return

        try:
            results_list = []
            for power, peaks in self.all_fit_results.items():
                for peak_data in peaks:
                    results_list.append({
                        'Power (uW)': power,
                        'Peak_ID': peak_data['peak_id'],
                        'Position (eV)': peak_data['position'],
                        'FWHM (eV)': peak_data['fwhm'],
                        'Integrated_Intensity': peak_data['integrated_intensity']
                    })
            
            df = pd.DataFrame(results_list)
            df.to_csv(file_name, index=False, float_format='%.6f')

            QMessageBox.information(self, "Save Success", "Fit results saved successfully to CSV.")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Failed to save fit results to CSV: {e}")

    def check_for_recovery(self):
        if os.path.exists(self.recovery_file):
            try:
                with open(self.recovery_file, 'r') as f:
                    recovery_data = json.load(f)
                # JSON은 float 키를 str으로 저장하므로, 다시 float으로 변환
                self.all_fit_results = {float(k): v for k, v in recovery_data.items()}
                QMessageBox.information(self, "Recovery Found", f"{len(self.all_fit_results)} previous fit results have been loaded.")
                self.update_peak_selection_ui() # 피크 선택 UI 업데이트
            except Exception as e:
                print(f"Warning: Could not read or parse recovery file: {e}")

    def add_fit_result(self, power, results):
        """Slot to receive fit results from FittingDialog."""
        self.all_fit_results[power] = results
        self.save_recovery_data() # Save to recovery file as well
        self.update_list_with_fit_status()
        self.update_peak_selection_ui() # 피크 선택 UI 업데이트
        print(f"Fit results for {power:.4f} uW updated and stored.")

    def update_list_with_fit_status(self):
        for i in range(self.spectra_list_widget.count()):
            item = self.spectra_list_widget.item(i)
            power = self.power_values[i]
            label = self.spectra_display_labels[i] if i < len(self.spectra_display_labels) else f"{power:.4f} uW"
            if power in self.all_fit_results:
                item.setText(f"{label} (Fitted)")
                item.setFont(QFont('Arial', 9, QFont.Bold))
            else:
                item.setText(label)
                item.setFont(QFont('Arial', 9, QFont.Normal))

    def save_recovery_data(self):
        try:
            with open(self.recovery_file, 'w') as f:
                json.dump(self.all_fit_results, f)
        except Exception as e:
            print(f"Error saving recovery data: {e}")

    def plot_energy_diff_vs_power(self):
        # 선택된 피크만 추출
        selected_peak_ids = sorted([pid for pid, cb in self.peak_checkboxes.items() if cb.isChecked()])
        if len(selected_peak_ids) < 2:
            self.ax.set_title("Select at least two peaks")
            self.canvas.draw()
            return

        from itertools import combinations
        combis = list(combinations(selected_peak_ids, 2))  # (i, j) where i < j

        sorted_powers = sorted(self.all_fit_results.keys())
        diff_data = {f"P{j}-P{i}": {'powers': [], 'diffs': []} for i, j in combis}

        for power in sorted_powers:
            peaks = self.all_fit_results[power]
            peaks_by_id = {p['peak_id']: p for p in peaks}
            for i, j in combis:
                if i in peaks_by_id and j in peaks_by_id:
                    # position 차이 계산 (eV 단위)
                    diff = (peaks_by_id[j]['position'] - peaks_by_id[i]['position']) * 1000  # meV
                    diff_data[f"P{j}-P{i}"]['powers'].append(power)
                    diff_data[f"P{j}-P{i}"]['diffs'].append(diff)

        for label, data in diff_data.items():
            if data['powers']:
                self.ax.plot(data['powers'], data['diffs'], marker='o', label=label)

        self.ax.set_xlabel('Power (uW)')
        self.ax.set_ylabel('Energy Difference (meV)')
        self.ax.set_title('Power vs. Energy Difference')
        self.ax.set_xscale('log')
        self.ax.set_yscale('linear')
        self.ax.legend(fontsize='large')
        self.ax.grid(True, which="both", ls="--")
        self.canvas.draw()

def copy_axes_plus_colorbar_to_clipboard(ax, cbar=None):
    """현재 Axes와 Colorbar 영역만 정확히 잘라서 클립보드에 PNG로 복사."""
    if ax is None:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Axes + (있으면) Colorbar의 타이트 BBox 합집합
    bboxes = [ax.get_tightbbox(renderer)]
    if cbar is not None and getattr(cbar, "ax", None) is not None:
        bboxes.append(cbar.ax.get_tightbbox(renderer))
    union = Bbox.union(bboxes).expanded(1.02, 1.05)  # 살짝 여유

    # 캔버스 RGBA 버퍼 → QImage (좌상 원점)
    w, h = fig.canvas.get_width_height()
    buf = fig.canvas.buffer_rgba()
    qimg_full = QImage(buf, w, h, QImage.Format_RGBA8888)

    # display 좌표(좌하 원점) → QImage 좌표(좌상 원점) 변환 후 크롭
    x0 = int(np.floor(union.x0))
    y0 = int(np.floor(union.y0))
    x1 = int(np.ceil(union.x1))
    y1 = int(np.ceil(union.y1))
    left = max(0, x0)
    top = max(0, h - y1)
    width = max(1, min(w - left, x1 - x0))
    height = max(1, min(h - top, y1 - y0))

    region = qimg_full.copy(left, top, width, height)
    QApplication.clipboard().setImage(region)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = PowerDependenceApp()
    main_app.show()
    sys.exit(app.exec_())