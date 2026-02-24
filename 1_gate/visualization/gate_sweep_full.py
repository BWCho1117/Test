# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:30:59 2020

@author: maubr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:40:28 2016.
@author: raphaelproux
Reads spatial 2D hyperspectral maps done using:
- Lightfield and SPE3 files
OR
- WinSpec and SPE2 files
and not spatial (just a stack of spectra, useful for polar maps for example)
Version 2 - 27/07/2016 R. Proux
Version 3 - 29/06/2017 R. Proux
Version 3.1 - 17/10/2017 RProux - added Git repo architecture support in python 3
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# 이제 from tools.smallFuncs3 import * 가능
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Try to import generated Python UI module first; if missing, try to load UiRanges.ui at runtime
try:
    from UiRanges import Ui_ranges
except Exception:
    # fallback: try to load .ui file dynamically using PyQt5.uic
    ui_path = os.path.join(script_dir, "UiRanges.ui")
    if os.path.exists(ui_path):
        from PyQt5 import uic
        Ui_ranges, _ = uic.loadUiType(ui_path)
        print("Loaded UiRanges from UiRanges.ui via PyQt5.uic")
    else:
        # final fallback: provide a minimal stub to avoid immediate crash (replace with real UI later)
        from PyQt5.QtWidgets import QWidget
        class Ui_ranges(object):
            def setupUi(self, parent):
                parent.setWindowTitle("Missing UiRanges - placeholder")
        print("Warning: UiRanges.py and UiRanges.ui not found. Using placeholder Ui_ranges; replace with real UI.")

from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtCore import QTimer
import matplotlib

matplotlib.use("Qt5Agg")
import pylab as pl
import numpy as np
import matplotlib.colors

import json

from tools.smallFuncs3 import *
from tools.instruments.princeton3.SPE3read import SPE3map
from tools.instruments.princeton3.SPE2read import SPE2map
from tools.arrayProcessing import range_to_edge

# For working with python, pythonw and ipython
import sys, os

if sys.executable.endswith(
    "pythonw.exe"
):  # this allows pythonw not to quit immediately
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(
        os.path.join(os.getenv("TEMP"), "stderr-" + os.path.basename(sys.argv[0])), "w"
    )
else:
    try:  # if under ipython, needs to choose external window for live updating figure
        from IPython import get_ipython

        ipython = get_ipython()
        ipython.magic("matplotlib qt5")
    except:
        pass

sys._excepthook = sys.excepthook


def exception_hook(exctype, value, traceback):
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


sys.excepthook = exception_hook


class SpaceMapWindow(QWidget):
    """
    Main QDialog class with the measurement and plot properties form.
    Also generates the matplotlib windows for displaying data
    """

    def __init__(self, app):
        super().__init__()                     # <-- FIX: was self.dialog = QWidget.__init__(self)
        self.app = app

        # Set up the user interface from Designer.
        self.ui = Ui_ranges()
        self.ui.setupUi(self)

        self.setMinimumWidth(500)
        self.setMinimumHeight(700)
        self.resize(820, 1000)  # 필요시 더 크게 조정 가능

        self.cmapList = [
            colormap for colormap in pl.colormaps() if not colormap.endswith("_r")
        ]
        self.ui.colorMap.addItems(self.cmapList)
        # set initial colormap to 'RdBu' if available, fallback to index 60
        try:
            initial_idx = self.cmapList.index("RdBu")
        except ValueError:
            initial_idx = 60
        self.ui.colorMap.setCurrentIndex(initial_idx)

        # flags
        self.f_mapOpen = False
        self.f_spOpen = False
        self.f_fileOpen = False
        self.f_firstOpening = False
        self.f_cacheRecovered = False

        self.measParams = {}
        self.plotParams = {}
        self.plotSpPos = None
        self.spMapPos = None
        self.spVerticalBarCounter = 0
        self.spWaveLimits = []
        self.countsIndex = None
        # Mauro's change:
        self.bckParams = {}

        # Connect up the buttons.
        self.ui.squareMeas.toggled.connect(self.squareMeasToggled)
        self.ui.xMeasMin.valueChanged.connect(self.measRangeUpdate)
        self.ui.xMeasMax.valueChanged.connect(self.measRangeUpdate)
        self.ui.xMeasStep.valueChanged.connect(self.measRangeUpdate)
        self.ui.colorPlotFullscale.clicked.connect(self.colorPlotFullscale)
        self.ui.colorPlotAutoscale.clicked.connect(self.colorPlotAutoscale)
        self.ui.xPlotFullscale.clicked.connect(self.xPlotFullscale)
        self.ui.yPlotFullscale.clicked.connect(self.yPlotFullscale)
        self.ui.wavePlotFullscale.clicked.connect(self.wavePlotFullscale)
        self.ui.plotUpdate.clicked.connect(self.updatePlots)
        self.ui.selectFileButton.clicked.connect(self.selectFile)
        self.ui.isSpaceMap.toggled.connect(self.isSpaceMapToggled)
        self.ui.spPlotAutoscaleCheckbox.toggled.connect(
            self.spPlotAutoscaleCheckboxToggled
        )
        self.ui.saveSpectrum.clicked.connect(self.saveSpectrumClicked)

        # Mauro's change:
        self.ui.isDifRef.toggled.connect(self.isDifRefToggled)
        self.ui.isDifRef.clicked.connect(self.isDifRefClicked)
        self.ui.xBck.valueChanged.connect(self.measRangeUpdate)
        self.ui.yBck.valueChanged.connect(self.measRangeUpdate)

        # --- 이 줄을 추가하세요 ---
        self.ui.plot_angular_linecut_button.clicked.connect(self.plot_angular_linecut)

        QTimer.singleShot(100, self.selectFile)  # select file at start

        # --- Dynamically (re)create Angular Linecut inputs if missing ---
        from PyQt5.QtWidgets import QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton
        if not hasattr(self.ui, "linecut_group"):
            self.ui.linecut_group = QGroupBox("Angular Linecut", self)
            self.ui.linecut_form = QFormLayout(self.ui.linecut_group)

            # Center X
            self.ui.linecut_center_x = QDoubleSpinBox()
            self.ui.linecut_center_x.setDecimals(3)
            self.ui.linecut_center_x.setRange(-1e3, 1e3)
            self.ui.linecut_center_x.setSingleStep(0.05)
            self.ui.linecut_form.addRow("Center X (V):", self.ui.linecut_center_x)

            # Center Y
            self.ui.linecut_center_y = QDoubleSpinBox()
            self.ui.linecut_center_y.setDecimals(3)
            self.ui.linecut_center_y.setRange(-1e3, 1e3)
            self.ui.linecut_center_y.setSingleStep(0.05)
            self.ui.linecut_form.addRow("Center Y (V):", self.ui.linecut_center_y)

            # Angle
            self.ui.linecut_angle = QDoubleSpinBox()
            self.ui.linecut_angle.setRange(0.0, 360.0)
            self.ui.linecut_angle.setSingleStep(1.0)
            self.ui.linecut_angle.setWrapping(True)
            self.ui.linecut_form.addRow("Angle (deg):", self.ui.linecut_angle)

            # Number of Points (create only if absent)
            if not hasattr(self.ui, "linecut_points"):
                self.ui.linecut_points = QSpinBox()
                self.ui.linecut_points.setRange(2, 2000)
                self.ui.linecut_points.setValue(100)
                self.ui.linecut_form.addRow("Number of Points:", self.ui.linecut_points)

            # Display mode
            if not hasattr(self.ui, "linecut_mode"):
                self.ui.linecut_mode = QComboBox()
                self.ui.linecut_mode.addItems(["Stacked", "Map"])
                self.ui.linecut_form.addRow("Display:", self.ui.linecut_mode)

            # NEW: Scale selector (independent of main plot)
            if not hasattr(self.ui, "linecut_scale"):
                self.ui.linecut_scale = QComboBox()
                self.ui.linecut_scale.addItems(["linear", "log"])
                self.ui.linecut_form.addRow("Scale:", self.ui.linecut_scale)

            # Plot button
            if not hasattr(self.ui, "plot_angular_linecut_button"):
                self.ui.plot_angular_linecut_button = QPushButton("Plot Angular Linecut")
                self.ui.linecut_form.addRow(self.ui.plot_angular_linecut_button)

            # Insert group at bottom of main layout
            # If your root layout is not accessible, fall back to adding as a child widget
            try:
                # Try to append to existing main layout (vertical)
                self.layout().addWidget(self.ui.linecut_group)
            except Exception:
                self.ui.linecut_group.setParent(self)

        # Keep last-used linecut settings
        self.last_linecut = {
            "center_x": 0.0,
            "center_y": 0.0,
            "angle": 0.0
        }

    def selectFile(self, event=0):
        """ 
        Generates the selection dialog window until cancelled or valid selected file.
        This function calls self.openFile() to open the file and validate it.
        """
        # ask for SPE file until valid file or user cancel
        while True:

            self.filename = QFileDialog.getOpenFileName(self, "Open 2D map SPE file")[0]
            #            print( repr(self.filename))
            if self.filename == "":  # user cancelled
                break
            elif self.openFile(self.filename):  # open selected file
                #                print(self.filename)
                self.ui.filePath.setText(self.filename)
                self.f_fileOpen = True
                if self.f_cacheRecovered:
                    self.update()
                break
            else:  # not valid file, display an error box
                errorMessageWindow(
                    self,
                    "File is not valid",
                    "Failed to read the file.\nPlease specify an SPE2 or SPE3 file recorded with WinSpec or Lightfield.",
                )

    def openFile(self, filename):
        """ 
        Opens data SPE3 file provided the filename from self.selectFile().
        Also opens if existing cache file
        """
        # read cache file if it is there
        cacheUnreadable = False
        self.f_cacheRecovered = True
        try:
            with open(cacheName(filename)) as infile:
                cacheData = json.load(infile)
        except IOError:
            self.f_cacheRecovered = False
        except json.JSONDecodeError:
            cacheUnreadable = True
            self.f_cacheRecovered = False
        else:
            try:
                self.measParams = cacheData["measParams"]
                self.plotParams = cacheData["plotParams"]
                # Mauro:
                self.bckParams = cacheData["bckParams"]
                #
                self.updateForm()
            except:
                cacheUnreadable = True
                self.f_cacheRecovered = False

        #        if cacheUnreadable:
        #            errorMessageWindow(self, "Failed to read cache file",
        #                                     "Failed to read the cache file used to store the parameters. Please set the settings again.")

        # read the SPE3 file if possible
        try:
            try:
                self.data = SPE3map(self.filename)
                print("SPE3 read")
            except:
                try:
                    self.data = SPE2map(self.filename)
                    print("SPE2 read")
                except:
                    raise
            self.f_firstOpening = True
            # directory = r'E:\Dropbox new (HW account)\Dropbox (Heriot-Watt University Team)\RES_EPS_Quantum_Photonics_Lab\Experiments\Current Experiments\Dichromatic Pulse RF\20190429 - PL map Dot searrch')
            # self.voltagefile = np.loadtxt(os.path.join(directory,'Spacemap_powerInVolts_2019-05-16_09-01-03.txt'))
            self.wavelength = self.data.wavelength
            self.wavelengthRangeMeas = (float(self.wavelength[0]), float(self.wavelength[-1]))  # for wavePlotFullscale
            # Precompute energy axis (eV). 1239.841984 eV*nm = hc
            self.energy_eV = HC_EV_NM / self.wavelength  # energy decreases if wavelength increases
            # For plotting with increasing energy to the right we will reverse both energy and spectra later.

            self.nbOfFrames = self.data.nbOfFrames
            self.counts = pl.array([dataRow[0] for dataRow in self.data.data])

            self.counts = pl.array([dataRow[0] for dataRow in self.data.data])
            self.bckParams["isDifRef"] = self.ui.isDifRef.isChecked()

            # out = []
            # for j in range(len(self.counts)):
            #     out.append(self.counts[j]/self.voltagefile[j])

            # self.counts = pl.array(out)
            self.exposureTime = self.data.exposureTime  # exposure time in ms
            
            # Initialize default measurement parameters for -4 to +4 V range
            if not self.f_cacheRecovered:
                self.initializeDefaultParams()

            # Initialize angular linecut spinboxes with map center if they exist
            try:
                x0 = (self.measParams["xRange"][0] + self.measParams["xRange"][1]) / 2.0
                y0 = (self.measParams["yRange"][0] + self.measParams["yRange"][1]) / 2.0
                if hasattr(self.ui, "linecut_center_x"):
                    self.ui.linecut_center_x.setValue(x0)
                if hasattr(self.ui, "linecut_center_y"):
                    self.ui.linecut_center_y.setValue(y0)
                if hasattr(self.ui, "linecut_angle"):
                    self.ui.linecut_angle.setValue(self.last_linecut.get("angle", 0.0))
                self.last_linecut.update({"center_x": x0, "center_y": y0})
            except Exception:
                pass
        except:
            return False
        else:
            return True

    def initializeDefaultParams(self):
        """
        Initialize default measurement parameters for -6 to +6 V range
        """
        # Default measurement parameters for your specific range
        self.measParams = {
            "xRange": [-6.0, 6.0],  # -4.0 → -6.0
            "yRange": [-6.0, 6.0],  # -4.0 → -6.0
            "xStep": 0.05,
            "yStep": 0.05,
            "isSpaceMap": True
        }
        
        # Default plot parameters
        self.plotParams = {
            "xRange": [-6.0, 6.0],
            "yRange": [-6.0, 6.0],
            "colorRange": [0.0, 1000.0],  # Will be auto-scaled
            "waveRange": [self.wavelength[0], self.wavelength[-1]],
            "mode": 0,  # 0=sum, 1=max
            "colorMap": 60,  # viridis-like
            "colorMapReversed": False,
            "keepAspRatio": True,
            "scale": 1,  # 0=log, 1=linear
            "spAutoscale": True
        }
        
        # Default background parameters
        self.bckParams = {
            "xBackground": 0.0,
            "yBackground": 0.0,
            "isDifRef": False
        }
        
        print("Initialized default parameters for -6 to +6 V range")
        print(f"X Range: {self.measParams['xRange']}")
        print(f"Y Range: {self.measParams['yRange']}")
        print(f"Step size: {self.measParams['xStep']} V")
        
        # IMMEDIATELY update the UI elements with correct values
        self.updateForm()
        print("Updated GUI elements with -6 to +6 V ranges")

    def update(self, event=0):
        self.updateProcessedCounts()
        self.f_firstOpening = False

        self.plotFigure(self.processedCounts, self.xVoltage, self.yVoltage)
        self.ui.plotUpdate.setText("Update")

    def updatePlots(self, event=0):
        """Recalculate processedCounts then (re)draw map figure."""
        if not self.f_fileOpen:
            print("[updatePlots] No file open yet.")
            return
        print("[updatePlots] start")
        try:
            self.updateProcessedCounts()
        except Exception as e:
            import traceback
            print("[updatePlots] ERROR in updateProcessedCounts:", e)
            traceback.print_exc()
            return

        if hasattr(self, "processedCounts"):
            print("[updatePlots] calling plotFigure")
            self.plotFigure(self.processedCounts, self.xVoltage, self.yVoltage)
        else:
            print("[updatePlots] processedCounts missing; skip plotFigure")
        self.ui.plotUpdate.setText("Update")
        print("[updatePlots] done")

    def updateProcessedCounts(self):

        # recover data and validate form
        if self.f_cacheRecovered and self.f_firstOpening:
            self.updateForm()
            validForm = self.recoverForm()
        else:
            validForm = self.recoverForm()

        self.updateForm()  # will sort the min/max if necessary

        if not (validForm):
            errorMessageWindow(
                self, "Form is not valid", "Please check no min/max values are equal."
            )

            self.f_cacheRecovered = False
            return

        # save cache file with new information
        try:
            cacheData = {
                "measParams": self.measParams,
                "plotParams": self.plotParams,
                "bckParams": self.bckParams,
            }
            #            print(cacheData)
            with open(cacheName(self.filename), "w") as outfile:
                json.dump(cacheData, outfile)

        except Exception as excep:
            print("Error saving cache file: {}".format(excep))
            pass

        xNbOfSteps = int(
            pl.floor(
                abs(self.measParams["xRange"][1] - self.measParams["xRange"][0])
                / abs(self.measParams["xStep"])
            )
            + 1
        )
        yNbOfSteps = int(
            pl.floor(
                abs(self.measParams["yRange"][1] - self.measParams["yRange"][0])
                / abs(self.measParams["yStep"])
            )
            + 1
        )

        # if space map
        if self.measParams["isSpaceMap"]:
            # Calculate expected number of points
            expected_points = xNbOfSteps * yNbOfSteps
            
            if expected_points != self.nbOfFrames:
                print(f"Warning: Expected {expected_points} points but got {self.nbOfFrames} frames")
                print(f"X steps: {xNbOfSteps}, Y steps: {yNbOfSteps}")
                print(f"X range: {self.measParams['xRange']}, step: {self.measParams['xStep']}")
                print(f"Y range: {self.measParams['yRange']}, step: {self.measParams['yStep']}")
                
                # Try to auto-correct the step size based on actual data
                if self.nbOfFrames > 0:
                    # Assume square grid for auto-correction
                    estimated_side = int(np.sqrt(self.nbOfFrames))
                    if estimated_side * estimated_side == self.nbOfFrames:
                        # Perfect square
                        x_range = self.measParams["xRange"][1] - self.measParams["xRange"][0]
                        y_range = self.measParams["yRange"][1] - self.measParams["yRange"][0]
                        
                        new_x_step = x_range / (estimated_side - 1)
                        new_y_step = y_range / (estimated_side - 1)
                        
                        print(f"Auto-correcting step sizes:")
                        print(f"  New X step: {new_x_step:.4f} V (was {self.measParams['xStep']:.4f})")
                        print(f"  New Y step: {new_y_step:.4f} V (was {self.measParams['yStep']:.4f})")
                        
                        self.measParams["xStep"] = new_x_step
                        self.measParams["yStep"] = new_y_step
                        
                        # Recalculate steps
                        xNbOfSteps = estimated_side
                        yNbOfSteps = estimated_side
                        
                        self.updateForm()  # Update UI with corrected values
                
                if xNbOfSteps * yNbOfSteps != self.nbOfFrames:
                    errorMessageWindow(
                        self,
                        "Step issue", 
                        f"Calculated steps ({xNbOfSteps}×{yNbOfSteps}={xNbOfSteps*yNbOfSteps}) don't match frames ({self.nbOfFrames}).\n\n"
                        f"Please adjust the range or step size:\n"
                        f"X: {self.measParams['xRange'][0]} to {self.measParams['xRange'][1]} V, step {self.measParams['xStep']} V\n"
                        f"Y: {self.measParams['yRange'][0]} to {self.measParams['yRange'][1]} V, step {self.measParams['yStep']} V"
                    )
                    self.f_cacheRecovered = False
                    return

            self.xVoltage = pl.linspace(
                min(self.measParams["xRange"]),
                max(self.measParams["xRange"]),
                xNbOfSteps,
            )
            self.yVoltage = pl.linspace(
                min(self.measParams["yRange"]),
                max(self.measParams["yRange"]),
                yNbOfSteps,
            )

            self.processedCounts = self.calculateCountArray(xNbOfSteps, yNbOfSteps)

            # print("updateProcessedCounts")

        # if not space map
        else:
            # called Voltage but actually not voltage
            self.xVoltage = self.wavelength
            self.yVoltage = pl.array(range(self.nbOfFrames))

            self.processedCounts = self.counts

        if not self.f_cacheRecovered and self.f_firstOpening:
            self.updateForm()  # Update UI with default parameters
            self.colorPlotFullscale(onlySet=True)
            self.xPlotFullscale(onlySet=True)
            self.yPlotFullscale(onlySet=True)
            self.wavePlotFullscale(onlySet=True)
            self.recoverForm()

    def recoverForm(self):
        try:
            # FORCE correct values for space maps BEFORE reading from UI
            if hasattr(self, 'measParams') and self.measParams.get("isSpaceMap", True):
                # Override UI values with correct -6 to +6 V range
                self.ui.xMeasMin.setValue(-6.0)  # -4.0 → -6.0
                self.ui.xMeasMax.setValue(6.0)   # 4.0 → 6.0
                self.ui.yMeasMin.setValue(-6.0)  # -4.0 → -6.0
                self.ui.yMeasMax.setValue(6.0)   # 4.0 → 6.0
                self.ui.xMeasStep.setValue(0.05)
                self.ui.yMeasStep.setValue(0.05)
                print("FORCED GUI measurement values to -6 to +6 V range")
            
            self.measParams["xRange"] = sorted(
                (self.ui.xMeasMin.value(), self.ui.xMeasMax.value())
            )
            self.measParams["xStep"] = self.ui.xMeasStep.value()
            self.measParams["yRange"] = sorted(
                (self.ui.yMeasMin.value(), self.ui.yMeasMax.value())
            )
            self.measParams["yStep"] = self.ui.yMeasStep.value()

            # Mauro's change:
            self.bckParams["xBackground"] = self.ui.xBck.value()
            self.bckParams["yBackground"] = self.ui.yBck.value()
            # self.bckParams['isDifRef'] = self.ui.isDifRef.isChecked()
            #####

            self.measParams["isSpaceMap"] = self.ui.isSpaceMap.isChecked()

            self.plotParams["xRange"] = sorted(
                (self.ui.xPlotMin.value(), self.ui.xPlotMax.value())
            )
            self.plotParams["yRange"] = sorted(
                (self.ui.yPlotMin.value(), self.ui.yPlotMax.value())
            )
            self.plotParams["colorRange"] = sorted(
                (self.ui.colorPlotMin.value(), self.ui.colorPlotMax.value())
            )
            self.plotParams["waveRange"] = sorted(
                (self.ui.wavePlotMin.value(), self.ui.wavePlotMax.value())
            )
            self.plotParams["mode"] = self.ui.plotMode.currentIndex()
            self.plotParams["colorMap"] = self.ui.colorMap.currentIndex()
            self.plotParams["colorMapReversed"] = self.ui.colorMapReversed.isChecked()
            self.plotParams["keepAspRatio"] = self.ui.keepAspRatio.isChecked()
            self.plotParams["scale"] = self.ui.scaleMode.currentIndex()

            self.plotParams["spAutoscale"] = self.ui.spPlotAutoscaleCheckbox.isChecked()

            if not (
                increasingTuples(
                    [
                        self.measParams["xRange"],
                        self.measParams["yRange"],
                        self.plotParams["xRange"],
                        self.plotParams["yRange"],
                        self.plotParams["colorRange"],
                        self.plotParams["waveRange"],
                    ]
                )
            ):
                raise ValueError

        except ValueError:
            raise
            return False
        else:
            return True

    def updateForm(self):
        # FORCE correct measurement parameters for space maps
        if self.measParams.get("isSpaceMap", True):
            self.measParams["xRange"] = [-6.0, 6.0]
            self.measParams["yRange"] = [-6.0, 6.0]
            self.measParams["xStep"] = 0.05
            self.measParams["yStep"] = 0.05
            print("FORCED measurement parameters to -6 to +6 V in updateForm")

        self.ui.xMeasMin.setValue(self.measParams["xRange"][0])
        self.ui.xMeasMax.setValue(self.measParams["xRange"][1])
        self.ui.xMeasStep.setValue(self.measParams["xStep"])
        self.ui.yMeasMin.setValue(self.measParams["yRange"][0])
        self.ui.yMeasMax.setValue(self.measParams["yRange"][1])
        self.ui.yMeasStep.setValue(self.measParams["yStep"])
        self.ui.isSpaceMap.setChecked(self.measParams["isSpaceMap"])

        self.ui.xPlotMin.setValue(self.plotParams["xRange"][0])
        self.ui.xPlotMax.setValue(self.plotParams["xRange"][1])
        self.ui.yPlotMin.setValue(self.plotParams["yRange"][0])
        self.ui.yPlotMax.setValue(self.plotParams["yRange"][1])
        self.ui.colorPlotMin.setValue(self.plotParams["colorRange"][0])
        self.ui.colorPlotMax.setValue(self.plotParams["colorRange"][1])
        self.ui.wavePlotMin.setValue(self.plotParams["waveRange"][0])
        self.ui.wavePlotMax.setValue(self.plotParams["waveRange"][1])

        self.ui.plotMode.setCurrentIndex(self.plotParams["mode"])
        self.ui.colorMap.setCurrentIndex(self.plotParams["colorMap"])
        self.ui.colorMapReversed.setChecked(self.plotParams["colorMapReversed"])
        self.ui.keepAspRatio.setChecked(self.plotParams["keepAspRatio"])
        self.ui.scaleMode.setCurrentIndex(self.plotParams["scale"])
        self.ui.spPlotAutoscaleCheckbox.setChecked(self.plotParams["spAutoscale"])

        # Mauro's change:
        self.ui.xBck.setValue(self.bckParams["xBackground"])
        self.ui.yBck.setValue(self.bckParams["yBackground"])
        self.ui.isDifRef.setChecked(self.bckParams["isDifRef"])

    def squareMeasToggled(self, event):
        squareMeasurement = self.ui.squareMeas.isChecked()
        if squareMeasurement:  # box was checked
            self.measParams["xRange"] = (
                self.ui.xMeasMin.value(),
                self.ui.xMeasMax.value(),
            )
            self.measParams["xStep"] = float(self.ui.xMeasStep.text())

            self.ui.yMeasMin.setValue(self.measParams["xRange"][0])
            self.ui.yMeasMax.setValue(self.measParams["xRange"][1])
            self.ui.yMeasStep.setValue(self.measParams["xStep"])
            self.ui.yMeasMin.setEnabled(False)
            self.ui.yMeasMax.setEnabled(False)
            self.ui.yMeasStep.setEnabled(False)

        else:  # box was unchecked
            self.ui.yMeasMin.setEnabled(True)
            self.ui.yMeasMax.setEnabled(True)
            self.ui.yMeasStep.setEnabled(True)

    def isSpaceMapToggled(self, event):
        self.measParams["isSpaceMap"] = self.ui.isSpaceMap.isChecked()
        if not self.measParams["isSpaceMap"]:  # box was unchecked
            self.ui.squareMeas.setEnabled(False)
            self.ui.xMeasMin.setEnabled(False)
            self.ui.xMeasMax.setEnabled(False)
            self.ui.xMeasStep.setEnabled(False)
            self.ui.yMeasMin.setEnabled(False)
            self.ui.yMeasMax.setEnabled(False)  
            self.ui.yMeasStep.setEnabled(False)

        else:  # box was checked
            self.ui.squareMeas.setEnabled(True)
            self.ui.xMeasMin.setEnabled(True)
            self.ui.xMeasMax.setEnabled(True)
            self.ui.xMeasStep.setEnabled(True)
            self.ui.yMeasMin.setEnabled(True)
            self.ui.yMeasMax.setEnabled(True)
            self.ui.yMeasStep.setEnabled(True)
        self.xPlotFullscale(onlySet=True)
        self.yPlotFullscale(onlySet=True)
        self.updateProcessedCounts()
        self.colorPlotFullscale(onlySet=True)

    def isDifRefToggled(self, event):
        self.bckParams["isDifRef"] = self.ui.isDifRef.isChecked()
        if not self.bckParams["isDifRef"]:  # box was unchecked
            self.ui.xBck.setEnabled(False)
            self.ui.yBck.setEnabled(False)

        else:  # box was checked
            self.ui.xBck.setEnabled(True)
            self.ui.yBck.setEnabled(True)

    def measRangeUpdate(self, event=0):
        if self.ui.squareMeas.isChecked():
            self.ui.yMeasMin.setValue(self.ui.xMeasMin.value())
            self.ui.yMeasMax.setValue(self.ui.xMeasMax.value())
            self.ui.yMeasStep.setValue(self.ui.xMeasStep.value())

    #        if self.ui.isDifRef.isChecked():
    #            self.ui.xBck.setValue(self.ui.xBck.value())
    #            self.ui.yBck.setValue(self.ui.yBck.value())

    def calculateCountArray(self, xNbOfSteps, yNbOfSteps):
        """
        Main function that calculates the data to display given the parameters
        """
        indexWaveRange = self.findWaveIndex(self.plotParams["waveRange"])

        #        if self.plotParams['mode'] == 0:
        #            countVector = pl.sum(self.counts[:, int(indexWaveRange[0]):int(indexWaveRange[1]+1)], axis=1)
        #            resultCounts = countVector.reshape(yNbOfSteps, xNbOfSteps)
        #        else:
        #            countVector = self.counts[:, int(indexWaveRange[0]):int(indexWaveRange[1]+1)].max(axis=1)
        #            resultCounts = countVector.reshape(yNbOfSteps, xNbOfSteps)

        # Mauro's change:

        # self.counts = pl.array([dataRow[0] for dataRow in self.data.data])
        # self.bckParams['isDifRef'] = self.ui.isDifRef.isChecked()
        if self.bckParams["isDifRef"]:
            # bckIndex = self.determineWhichSpectrum((self.bckParams['xBackground'],self.bckParams['yBackground']), indexMove=(self.bckParams['xBackground'],self.bckParams['yBackground']))
            bckIndex = self.determineWhichSpectrum(
                (self.bckParams["xBackground"], self.bckParams["yBackground"]),
                indexMove=(0, 0),
            )
            bckIndex = bckIndex[0]
            # refmap = -(self.counts - self.counts[bckIndex,:])/self.counts
            refmap = (0.0001 * self.counts - 0.0001 * self.counts[bckIndex, :]) / (
                0.0001 * self.counts[bckIndex, :]
            )
            #            refmap = (self.counts[bckIndex,: - self.counts])/self.counts[bckIndex,:]

            if self.plotParams["mode"] == 0:
                countVector = pl.sum(
                    refmap[:, int(indexWaveRange[0]) : int(indexWaveRange[1] + 1)],
                    axis=1,
                )
                resultCounts = countVector.reshape(yNbOfSteps, xNbOfSteps)

                # Mauro's change
                resultCounts = resultCounts + abs(pl.amin(resultCounts)) + 1

            else:
                countVector = refmap[
                    :, int(indexWaveRange[0]) : int(indexWaveRange[1] + 1)
                ].max(axis=1)
                resultCounts = countVector.reshape(yNbOfSteps, xNbOfSteps)
        else:

            if self.plotParams["mode"] == 0:
                countVector = pl.sum(
                    self.counts[:, int(indexWaveRange[0]) : int(indexWaveRange[1] + 1)],
                    axis=1,
                )
                resultCounts = countVector.reshape(yNbOfSteps, xNbOfSteps)
            else:
                countVector = self.counts[
                    :, int(indexWaveRange[0]) : int(indexWaveRange[1] + 1)
                ].max(axis=1)
                resultCounts = countVector.reshape(yNbOfSteps, xNbOfSteps)

        ##############################

        #        if self.plotParams['mode'] == 0:
        #            countVector = pl.sum(self.counts[:, int(indexWaveRange[0]):int(indexWaveRange[1]+1)], axis=1)
        #            resultCounts = countVector.reshape(yNbOfSteps, xNbOfSteps)
        #        else:
        #            countVector = self.counts[:, int(indexWaveRange[0]):int(indexWaveRange[1]+1)].max(axis=1)
        #            resultCounts = countVector.reshape(yNbOfSteps, xNbOfSteps)
        #

        return resultCounts

    def findWaveIndex(self, waveRange):
        """
        Find the indices of each element of 2-tuple waveRange
        Returns the tuple of indices
        """
        foundMinIndex = False
        foundMaxIndex = False
        minIndex = 0
        for i, wave in enumerate(self.wavelength):
            if (
                waveRange[0] is not None
                and wave >= waveRange[0]
                and not (foundMinIndex)
            ):
                foundMinIndex = True
                minIndex = i
            if waveRange[1] is not None and wave >= waveRange[1]:
                foundMaxIndex = True
                indexWaveRange = (minIndex, i)
                break
        if not (foundMaxIndex):
            indexWaveRange = (minIndex, len(self.wavelength))

        return indexWaveRange

    def plotFigure(self, counts, xVoltage, yVoltage):

        # if figure did not exist, let's create it
        if not self.f_mapOpen:
            self.fig = pl.figure(figsize=(8.0 * 4.0 / 3.0 * 0.8, 8.0 * 0.8), dpi=100)
            self.mainAxes = self.fig.add_axes([0.1, 0.1, 0.7, 0.8])
            self.colorBarAxes = self.fig.add_axes([0.82, 0.1, 0.03, 0.8])
            self.f_mapOpen = True
        else:
            self.mainAxes.clear()  # fresh start for replotting
            self.colorBarAxes.clear()

        if self.plotParams["scale"] == 0:
            normFunc = matplotlib.colors.LogNorm(
                vmin=self.plotParams["colorRange"][0],
                vmax=self.plotParams["colorRange"][1]
            )
        else:
            normFunc = matplotlib.colors.Normalize(
                vmin=self.plotParams["colorRange"][0],
                vmax=self.plotParams["colorRange"][1]
            )

        cmapName = colorMapName(
            self.plotParams["colorMap"],
            self.plotParams["colorMapReversed"],
            self.cmapList,
        )

        # draw the map
        self.pcf = self.mainAxes.pcolorfast(
            range_to_edge(xVoltage),
            range_to_edge(yVoltage),
            counts,
            norm=normFunc,
            cmap=cmapName,
            zorder=1,
        )

        # plot position of spectrum if displayed
        if self.f_spOpen:
            self.plotSpPos, = self.mainAxes.plot(
                [self.spMapPos[0]], [self.spMapPos[1]], "xb", markersize=20
            )

        self.mainAxes.set_title(os.path.basename(self.filename))
        if self.measParams["isSpaceMap"]:
            self.mainAxes.get_xaxis().set_label_text("Voltage (V)")
            self.mainAxes.get_yaxis().set_label_text("Voltage (V)")
        else:
            self.mainAxes.get_xaxis().set_label_text("Wavelength (nm)")
            self.mainAxes.get_yaxis().set_label_text("Index")
        self.mainAxes.get_xaxis().get_major_formatter().set_useOffset(False)
        self.mainAxes.get_yaxis().get_major_formatter().set_useOffset(False)

        self.mainAxes.set_xlim(self.plotParams["xRange"])
        self.mainAxes.set_ylim(self.plotParams["yRange"])
        if self.plotParams["keepAspRatio"]:
            self.mainAxes.set_aspect("equal")
        else:
            self.mainAxes.set_aspect("auto")

        # --- (3) 이벤트 연결 교체 ---
        if hasattr(self, "_click_cid"):
            self.fig.canvas.mpl_disconnect(self._click_cid)
        self._click_cid = self.fig.canvas.mpl_connect("button_press_event", self.on_map_click)

        self.fig.subplots_adjust(left=0.10, right=0.92)  # before colorbar

        cb = self.fig.colorbar(mappable=self.pcf, cax=self.colorBarAxes)
        cb.set_label("Counts in {:.0f} ms".format(self.exposureTime))
        cb.ax.minorticks_on()

        self.fig.canvas.draw()
        self.fig.canvas.show()

    # --- (4) 새 메서드 추가 (클래스 내부 아무 위치) ---
    def on_map_click(self, event):
        """Map 클릭 처리: mainAxes 내부 좌클릭일 때만 스펙트럼 표시."""
        if event.inaxes is not self.mainAxes:
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        if DEBUG_MAP_CLICK:
            print(f"[on_map_click] click @ ({event.xdata:.3f}, {event.ydata:.3f})")
        try:
            self.displaySpectrum(event)
        except Exception as e:
            import traceback
            print("[on_map_click] ERROR:", e)
            traceback.print_exc()

    # --- (5) displaySpectrum 전체 교체 ---
    def displaySpectrum(self, event=0, moveX=0, moveY=0):
        """
        Spectrum window:
          Bottom: Energy (eV)
          Top   : Wavelength (nm)
        """
        # (a) 위치 결정
        if event != 0 and hasattr(event, "xdata") and event.xdata is not None:
            pos = (float(event.xdata), float(event.ydata))
        else:
            if self.spMapPos is None:
                if DEBUG_MAP_CLICK:
                    print("[displaySpectrum] No previous position.")
                return
            pos = self.spMapPos

        # (b) 인덱스
        try:
            self.countsIndex, self.spMapPos = self.determineWhichSpectrum(
                pos=pos, indexMove=(moveX, moveY)
            )
        except Exception as e:
            import traceback
            print("[displaySpectrum] determineWhichSpectrum ERROR:", e)
            traceback.print_exc()
            return
        if self.countsIndex is None or not (0 <= self.countsIndex < len(self.counts)):
            if DEBUG_MAP_CLICK:
                print("[displaySpectrum] Invalid countsIndex:", self.countsIndex)
            return

        # (c) diff ref
        if self.bckParams.get("isDifRef", False):
            try:
                bckIndex = self.determineWhichSpectrum(
                    (self.bckParams["xBackground"], self.bckParams["yBackground"]), indexMove=(0, 0)
                )[0]
                refmap = (0.0001 * self.counts - 0.0001 * self.counts[bckIndex, :]) / (
                    0.0001 * self.counts[bckIndex, :]
                )
                counts_all = refmap
            except Exception as e:
                if DEBUG_MAP_CLICK:
                    print("[displaySpectrum] diff ref fallback:", e)
                counts_all = self.counts
        else:
            counts_all = self.counts

        # (d) 에너지 축 만들기 (오름차순)
        energy_raw = self.energy_eV
        if energy_raw[0] > energy_raw[-1]:
            energy_axis = energy_raw[::-1]
            spec_y = counts_all[self.countsIndex][::-1]
        else:
            energy_axis = energy_raw
            spec_y = counts_all[self.countsIndex]

        # (e) Y limit
        ymin, ymax = float(spec_y.min()), float(spec_y.max())
        pad = 0.05 * (ymax - ymin + 1e-12)
        if pad <= 0:
            pad = 1.0
        ylims = (ymin - pad, ymax + pad)

        # (f) 맵 마커
        if self.f_mapOpen:
            try:
                if self.plotSpPos is not None:
                    self.plotSpPos.remove()
            except:
                pass
            (self.plotSpPos,) = self.mainAxes.plot(
                [self.spMapPos[0]], [self.spMapPos[1]], "xb", markersize=20
            )
            self.fig.canvas.draw_idle()

        # (g) Figure 생성/재사용
        new_fig = not self.f_spOpen
        if new_fig:
            self.f_spOpen = True
            self.spFig = pl.figure(figsize=SPECTRUM_FIG_SIZE)
            self.spFigNumber = self.spFig.number
            self.spAxes = self.spFig.add_axes(SPECTRUM_AX_RECT)
        else:
            self.spAxes.clear()

        title = f"Spectrum at X = {self.spMapPos[0]:.3f} V, Y = {self.spMapPos[1]:.3f} V"
        (self.spPlot,) = self.spAxes.plot(energy_axis, spec_y, color="tab:blue", lw=1.05)
        self.spAxes.set_title(title)
        self.spAxes.set_xlabel("Energy (eV)")
        self.spAxes.set_ylabel(f"Counts in {self.exposureTime:.0f} ms")
        self.spAxes.set_xlim(energy_axis[0], energy_axis[-1])
        if self.plotParams.get("spAutoscale", True):
            self.spAxes.set_ylim(*ylims)

        # (h) 위쪽 보조축 (안전 변환 함수)
        def E_to_wl(E):
            E = np.asarray(E)
            return HC_EV_NM / np.where(E <= 0, np.nan, E)
        def wl_to_E(wl):
            wl = np.asarray(wl)
            return HC_EV_NM / np.where(wl <= 0, np.nan, wl)

        try:
            sec = self.spAxes.secondary_xaxis("top", functions=(E_to_wl, wl_to_E))
            sec.set_xlabel("Wavelength (nm)")
        except Exception as e:
            if DEBUG_MAP_CLICK:
                print("[displaySpectrum] secondary_xaxis fallback:", e)
            sec = self.spAxes.twiny()
            sec.set_xlim(self.spAxes.get_xlim())
            Eticks = self.spAxes.get_xticks()
            with np.errstate(divide="ignore"):
                wl_ticks = HC_EV_NM / np.where(Eticks <= 0, np.nan, Eticks)
            labels = [f"{w:.0f}" if np.isfinite(w) else "" for w in wl_ticks]
            sec.set_xticks(Eticks)
            sec.set_xticklabels(labels)
            sec.set_xlabel("Wavelength (nm) (approx)")

        self.spAxes.figure.canvas.draw_idle()

        # Copy 버튼 (이미 설치 안됐으면)
        try:
            self.add_spectrum_copy_buttons(energy_axis, spec_y)
        except Exception as _e:
            if DEBUG_MAP_CLICK:
                print("[add_spectrum_copy_buttons] skip:", _e)

        if new_fig:
            self.spFig.canvas.mpl_connect("button_press_event", self.mousSpectrumClick)
            self.spFig.canvas.mpl_connect("close_event", self.spectrumClose)
            self.fig.canvas.mpl_connect("key_press_event", self.keyboardChangeSpectrum)
            self.spFig.canvas.mpl_connect("key_press_event", self.keyboardChangeSpectrum)

        if hasattr(self.ui, "linecut_center_x"):
            self.ui.linecut_center_x.setValue(self.spMapPos[0])
        if hasattr(self.ui, "linecut_center_y"):
            self.ui.linecut_center_y.setValue(self.spMapPos[1])
        self.last_linecut.update({"center_x": self.spMapPos[0], "center_y": self.spMapPos[1]})

    def determineWhichSpectrum(self, pos, indexMove):
        # determine spectrum index and closest position for spectrum designated by
        #   position pos (X,Y) in graph coordinates and after a move of indexMove (DX, DY) in index (nb of positions)
        # -> pos and indexMove should be 2-tuples

        xIndex = argNearest(self.xVoltage, pos[0])
        if 0 <= xIndex + indexMove[0] < len(self.xVoltage):
            xIndex = xIndex + indexMove[0]
        yIndex = argNearest(self.yVoltage, pos[1])
        if 0 <= yIndex + indexMove[1] < len(self.yVoltage):
            yIndex = yIndex + indexMove[1]

        if self.measParams["isSpaceMap"]:
            countsIndex = yIndex * len(self.xVoltage) + xIndex
        else:
            countsIndex = yIndex

        return countsIndex, (self.xVoltage[xIndex], self.yVoltage[yIndex])

    def spPlotAutoscaleCheckboxToggled(self, event):
        self.plotParams["spAutoscale"] = self.ui.spPlotAutoscaleCheckbox.isChecked()

    def keyboardChangeSpectrum(self, event):
        if event.key == "right":
            self.displaySpectrum(moveX=1, moveY=0)
        elif event.key == "left":
            self.displaySpectrum(moveX=-1, moveY=0)
        elif event.key == "up":
            self.displaySpectrum(moveX=0, moveY=1)
        elif event.key == "down":
            self.displaySpectrum(moveX=0, moveY=-1)

    def mousSpectrumClick(self, event):
        """
        Energy 축에서 영역 선택 (두 번 = waveRange). Right click = clear.
        """
        if event.inaxes is not self.spAxes:
            return
        if event.xdata is None:
            return

        # Right click -> clear
        if event.button == 3:
            if self.spVerticalBarCounter == 0 and len(self.spAxes.lines) > 2:
                self.spAxes.lines[-1].remove(); self.spAxes.lines[-1].remove()
            elif self.spVerticalBarCounter == 1:
                self.spAxes.lines[-1].remove()
            self.spFig.canvas.draw_idle()
            self.wavePlotFullscale(); self.colorPlotFullscale()
            return

        if event.button != 1:
            return

        E = float(event.xdata)
        if E <= 0:
            return
        wl = HC_EV_NM / E

        # 새 쌍 시작 시 기존 막대 제거
        if self.spVerticalBarCounter == 0 and len(self.spAxes.lines) > 2:
            self.spAxes.lines[-1].remove(); self.spAxes.lines[-1].remove()
            self.spWaveLimits = []

        plotVerticalBar(self.spAxes, E, "r")
        self.spFig.canvas.draw_idle()

        self.spWaveLimits.append(wl)
        self.spVerticalBarCounter = (self.spVerticalBarCounter + 1) % 2

        if self.spVerticalBarCounter == 0:
            self.plotParams["waveRange"] = sorted(self.spWaveLimits)
            if hasattr(self.ui, "wavePlotMin"):
                self.ui.wavePlotMin.setValue(self.plotParams["waveRange"][0])
            if hasattr(self.ui, "wavePlotMax"):
                self.ui.wavePlotMax.setValue(self.plotParams["waveRange"][1])
            self.update()
            self.colorPlotFullscale()

    def mapClose(self, event):
        self.f_mapOpen = False

    def spectrumClose(self, event):
        self.f_spOpen = False
        self.plotSpPos = None
        if self.f_mapOpen:
            self.update()

    def updateFormDraw(self, event=0):
        """Updates Entry boxes after redrawing PL map"""
        self.plotParams["xRange"] = self.mainAxes.get_xlim()
        self.plotParams["yRange"] = self.mainAxes.get_ylim()
        self.ui.xPlotMin.setValue(self.plotParams["xRange"][0])
        self.ui.xPlotMax.setValue(self.plotParams["xRange"][1])
        self.ui.yPlotMin.setValue(self.plotParams["yRange"][0])
        self.ui.yPlotMax.setValue(self.plotParams["yRange"][1])

    def colorPlotFullscale(self, onlySet=False):
        try:
            self.ui.colorPlotMin.setValue(self.processedCounts.min())
            self.ui.colorPlotMax.setValue(self.processedCounts.max())

        finally:
            if not onlySet:
                self.updatePlots()

    def colorPlotAutoscale(self, onlySet=False):
        try:
            histo = pl.histogram(window.processedCounts, bins=2 ** 8)[0]
            histo = histo.astype(float) / sum(histo)
            intPower = 0
            iMin = 0
            for i, histBin in enumerate(histo):
                intPower += histBin
                if intPower <= 0.6:
                    iMin = i
                if intPower > 0.99:
                    break
            minValData = self.processedCounts.min()
            maxValData = self.processedCounts.max()
            minVal = float(iMin) / 2 ** 8 * (maxValData - minValData) + minValData
            maxVal = (
                float(max(0, i - 1)) / 2 ** 8 * (maxValData - minValData) + minValData
            )

            # Mauro's change
            # minVal = float(iMin) / 2**8 * (maxValData - minValData)
            # maxVal = float(max(0,i-1)) / 2**8 * (maxValData - minValData)

            self.ui.colorPlotMin.setValue(minVal)
            self.ui.colorPlotMax.setValue(maxVal)

            # Mauro's change:
        #            self.ui.colorPlotMin.setValue(minValData)
        #            self.ui.colorPlotMax.setValue(maxValData)

        finally:
            if not onlySet:
                self.updatePlots()

    def xPlotFullscale(self, onlySet=False):

        try:
            if self.measParams["isSpaceMap"]:
                self.ui.xPlotMin.setValue(self.measParams["xRange"][0])
                self.ui.xPlotMax.setValue(self.measParams["xRange"][1])
            else:
                self.ui.xPlotMin.setValue(pl.amin(self.wavelength))
                self.ui.xPlotMax.setValue(pl.amax(self.wavelength))
        finally:
            if not onlySet:
                self.updatePlots()

    def yPlotFullscale(self, onlySet=False):
        try:
            if self.measParams["isSpaceMap"]:
                self.ui.yPlotMin.setValue(self.measParams["yRange"][0])
                self.ui.yPlotMax.setValue(self.measParams["yRange"][1])
            else:
                self.ui.yPlotMin.setValue(0.0)
                self.ui.yPlotMax.setValue(len(self.counts))

        finally:
            if not onlySet:
                self.updatePlots()

    def wavePlotFullscale(self, onlySet=False):
        try:
            self.ui.wavePlotMin.setValue(self.wavelengthRangeMeas[0])
            self.ui.wavePlotMax.setValue(self.wavelengthRangeMeas[1])

        finally:
            if not onlySet:
                self.updatePlots()

    def saveSpectrumClicked(self):
        if self.countsIndex is not None:
            # ask for SPE file until valid file or user cancel
            while True:

                filename = QFileDialog.getSaveFileName(
                    self, "Save spectrum as text file..."
                )[0]
                if filename == "":  # user cancelled
                    break
                else:  # try to save selected file
                    try:
                        saveArray = pl.array(
                            [self.wavelength, self.counts[self.countsIndex]]
                        ).transpose()
                        pl.savetxt(filename, saveArray, delimiter="\t")
                        break
                    except:  # not valid file, display an error box
                        errorMessageWindow(
                            self,
                            "Failed to save",
                            "Failed to save the file.\nPlease try again (maybe with a different name).",
                        )

    # Mauro's change:
    def isDifRefClicked(self):
        self.ui.xBck.setValue(self.ui.xBck.value())
        self.ui.yBck.setValue(self.ui.yBck.value())

    # --- 아래의 새로운 메서드 전체를 여기에 추가하세요 ---
    def plot_angular_linecut(self):
        """
        Calculates and plots spectra along a line defined by a center point and an angle.
        Supports two display modes: Stacked (offset spectra) or Map (distance vs energy).
        """
        if not self.f_fileOpen:
            errorMessageWindow(self, "No File", "Please open a data file first.")
            return

        # Acquire UI parameters with fallback
        if hasattr(self.ui, "linecut_center_x"):
            center_x = self.ui.linecut_center_x.value()
        else:
            center_x = self.last_linecut.get("center_x", 0.0)

        if hasattr(self.ui, "linecut_center_y"):
            center_y = self.ui.linecut_center_y.value()
        else:
            center_y = self.last_linecut.get("center_y", 0.0)

        if hasattr(self.ui, "linecut_angle"):
            angle_deg = self.ui.linecut_angle.value()
        else:
            angle_deg = self.last_linecut.get("angle", 0.0)

        if hasattr(self.ui, "linecut_points"):
            num_points = self.ui.linecut_points.value()
        else:
            num_points = 100

        if hasattr(self.ui, "linecut_mode"):
            mode_display = self.ui.linecut_mode.currentText()
        else:
            mode_display = "Stacked"

        # Persist latest values
        self.last_linecut.update({"center_x": center_x, "center_y": center_y, "angle": angle_deg})

        angle_rad = np.deg2rad(angle_deg)

        # --- OLD ---
        # x_range = self.plotParams["xRange"]
        # y_range = self.plotParams["yRange"]
        # max_dist = np.sqrt((x_range[1] - x_range[0])**2 + (y_range[1] - y_range[0])**2)
        # distances = np.linspace(-max_dist/2, max_dist/2, num_points)
        # line_x = center_x + distances * np.cos(angle_rad)
        # line_y = center_y + distances * np.sin(angle_rad)

        # --- NEW: 전체 데이터 사각 경계(x/y range)와 직선의 교점을 이용해 맵 끝까지 라인 생성 ---
        x_min, x_max = self.plotParams["xRange"]
        y_min, y_max = self.plotParams["yRange"]

        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        ts = []
        tol = 1e-9

        # 교점 후보 (수직 경계)
        if abs(dx) > 1e-12:
            t_left = (x_min - center_x) / dx
            y_at_left = center_y + t_left * dy
            if y_min - tol <= y_at_left <= y_max + tol:
                ts.append(t_left)

            t_right = (x_max - center_x) / dx
            y_at_right = center_y + t_right * dy
            if y_min - tol <= y_at_right <= y_max + tol:
                ts.append(t_right)

        # 교점 후보 (수평 경계)
        if abs(dy) > 1e-12:
            t_bottom = (y_min - center_y) / dy
            x_at_bottom = center_x + t_bottom * dx
            if x_min - tol <= x_at_bottom <= x_max + tol:
                ts.append(t_bottom)

            t_top = (y_max - center_y) / dy
            x_at_top = center_x + t_top * dx
            if x_min - tol <= x_at_top <= x_max + tol:
                ts.append(t_top)

        if len(ts) < 2:
            # 극단적 경우(라인이 경계 계산 실패) 대비: 기존 fallback
            span = max(x_max - x_min, y_max - y_min)
            ts = [-span, span]

        t_min = min(ts)
        t_max = max(ts)

        # 끝이 딱 잘려 보이지 않도록 살짝 연장
        expand = 0.01 * (t_max - t_min + 1e-12)
        t_min -= expand
        t_max += expand

        distances = np.linspace(t_min, t_max, num_points)
        line_x = center_x + distances * dx
        line_y = center_y + distances * dy

        spectra_list = []
        dist_list = []
        x_list = []
        y_list = []
        used_idx = set()

        for d, px, py in zip(distances, line_x, line_y):
            counts_idx, _ = self.determineWhichSpectrum(pos=(px, py), indexMove=(0, 0))
            if counts_idx not in used_idx:
                used_idx.add(counts_idx)
                spectra_list.append(self.counts[counts_idx])
                dist_list.append(d)
                x_list.append(px)
                y_list.append(py)

        if not spectra_list:
            errorMessageWindow(self, "No Data", "No spectra found along the specified line.")
            return

        spectra_array = np.array(spectra_list)
        dist_array = np.array(dist_list)
        x_arr = np.array(x_list)
        y_arr = np.array(y_list)

        # Energy axis (ensure increasing)
        energy_axis = self.energy_eV
        if energy_axis[0] > energy_axis[-1]:
            energy_plot = energy_axis[::-1]
            spectra_array = spectra_array[:, ::-1]
        else:
            energy_plot = energy_axis

        # First order by original param distance (to keep a stable baseline ordering)
        order = np.argsort(dist_array)
        spectra_sorted = spectra_array[order]
        x_sorted = x_arr[order]
        y_sorted = y_arr[order]

        # === NEW: Physical axis selection based on angle ===
        # angle >= 90° : Electric field dependence -> use difference Vx - Vy
        # angle  < 90° : Doping dependence        -> use sum       Vx + Vy
        if angle_deg >= 90.0:
            axis_y = x_sorted - y_sorted
            axis_label = "Electric field (Vx - Vy) (V)"
        else:
            axis_y = x_sorted + y_sorted
            axis_label = "Doping axis (Vx + Vy) (V)"

        # Ensure strictly monotonic increasing along y-axis; reorder if needed
        if not np.all(np.diff(axis_y) >= 0):
            o2 = np.argsort(axis_y)
            axis_y = axis_y[o2]
            spectra_sorted = spectra_sorted[o2]

        # Figure size (fallback if widgets missing)
        try:
            fig_w = self.ui.linecut_fig_w.value()
            fig_h = self.ui.linecut_fig_h.value()
        except Exception:
            fig_w, fig_h = 7.0, 6.0

        if mode_display == "Stacked":
            # (Stacked part unchanged – optional: also make it fixed like Map if needed)
            fig, ax = pl.subplots(figsize=(fig_w, fig_h))
            max_val = np.max(spectra_sorted) if np.max(spectra_sorted) > 0 else 1
            offset = 0.5 * max_val
            for i, spec in enumerate(spectra_sorted):
                ax.plot(energy_plot, spec + i * offset, linewidth=1.0)
            ax.set_title(f"Angular Linecut: Center=({center_x:.2f}, {center_y:.2f}), Angle={angle_deg:.1f}° (Stacked)")
            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel(axis_label + " (order only)")
            ax.get_yaxis().set_ticks([])
            if not FIXED_LINECUT_LAYOUT:
                fig.tight_layout()
            fig.show()

            # --- Add copy-to-clipboard helpers (place right before/after fig.show()) ---
            self.add_copy_buttons(fig)

            fig.show()
        else:
            # ================= FIXED MAP LAYOUT BRANCH =================
            scale_choice = getattr(self.ui, "linecut_scale", None)
            scale_choice = scale_choice.currentText() if scale_choice else "linear"
            cmap_linecut = "RdBu_r"
            data_map = spectra_sorted.astype(float)

            # Build norm
            if scale_choice == "log":
                finite_mask = np.isfinite(data_map)
                positive_mask = data_map > 0
                if not np.any(positive_mask):
                    data_map = data_map - np.min(data_map[finite_mask]) + 1e-12
                    positive_mask = data_map > 0
                min_pos = np.min(data_map[positive_mask])
                eps = max(min_pos * 1e-6, 1e-12)
                if np.min(data_map) <= 0:
                    data_map = data_map - np.min(data_map) + eps
                vmin = max(np.min(data_map[positive_mask]), eps)
                vmax = np.max(data_map)
                norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
                cbar_label = "Counts (log)"
            else:
                norm = matplotlib.colors.Normalize(vmin=np.min(data_map), vmax=np.max(data_map))
                cbar_label = "Counts"

            if FIXED_LINECUT_LAYOUT:
                fig_w_fix, fig_h_fix = LINECUT_LAYOUT["fig_size"]
                fig = pl.figure(figsize=(fig_w_fix, fig_h_fix))
                L = LINECUT_LAYOUT
                main_w = L["right"] - L["left"]
                main_h = L["top"] - L["bottom"]
                ax = fig.add_axes([L["left"], L["bottom"], main_w, main_h])
            else:
                fig, ax = pl.subplots(figsize=(fig_w, fig_h))

            im = ax.pcolormesh(
                energy_plot,
                axis_y,
                data_map,
                shading="auto",
                cmap=cmap_linecut,
                norm=norm
            )

            ax.set_title(
                f"Angular Linecut: Center=({center_x:.2f}, {center_y:.2f}), "
                f"Angle={angle_deg:.1f}° (Map {scale_choice})"
            )
            ax.set_xlabel("Energy (eV)")
            ax.set_ylabel(axis_label)

            # Colorbar fixed position
            if FIXED_LINECUT_LAYOUT:
                L = LINECUT_LAYOUT
                cbar_x = L["right"] + L["cbar_pad"]
                cbar_rect = [cbar_x, L["bottom"], L["cbar_width"], main_h]
                cax = fig.add_axes(cbar_rect)
                cbar = fig.colorbar(im, cax=cax)
            else:
                cbar = fig.colorbar(im, ax=ax)

            # Uniform tick control to stabilize width
            from matplotlib import ticker
            if scale_choice == "log":
                from matplotlib.ticker import LogFormatterMathtext, LogLocator
                cbar.locator = LogLocator(base=10, numticks=5)
                cbar.formatter = LogFormatterMathtext()
            else:
                cbar.locator = ticker.MaxNLocator(5, prune=None)
            cbar.update_ticks()
            cbar.set_label(cbar_label)

            # (Removed invalid toolbar.set_active call)
            # If you just want to be sure no pan/zoom mode is active you can optionally do:
            toolbar = getattr(fig.canvas.manager, "toolbar", None)
            if toolbar and hasattr(toolbar, "mode"):
                toolbar.mode = ''  # clear any active tool (safe)

            fig.show()

            # --- Add copy-to-clipboard helpers (place right before/after fig.show()) ---
            self.add_copy_buttons(fig)

            fig.show()

        # Overlay line on main map (unchanged)
        if self.f_mapOpen:
            for line in list(self.mainAxes.lines):
                if line.get_label() == '_linecut':
                    line.remove()
            self.mainAxes.plot(line_x, line_y, 'w--', linewidth=2, label='_linecut')
            self.fig.canvas.draw()

    def add_copy_buttons(self, fig):
        """
        Injects 'Copy PNG' and 'Copy SVG' buttons + shortcuts into a Matplotlib Qt figure window.
        Safe to call multiple times (will not duplicate).
        """
        try:
            from PyQt5.QtWidgets import QPushButton, QApplication
            from PyQt5.QtGui import QImage
            import io
        except Exception:
            return  # PyQt not available

        win = getattr(fig.canvas.manager, "window", None)
        if win is None or hasattr(win, "_copy_buttons_installed"):
            return

        def copy_png(dpi=300):
            import io
            from PyQt5.QtGui import QImage
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            buf.seek(0)
            qimg = QImage.fromData(buf.getvalue(), "PNG")
            QApplication.clipboard().setImage(qimg)
            win.statusBar().showMessage(f"Figure copied to clipboard as {dpi} DPI PNG", 2500)

        def copy_svg():
            import io
            buf = io.StringIO()
            fig.savefig(buf, format="svg", bbox_inches="tight")
            QApplication.clipboard().setText(buf.getvalue())
            win.statusBar().showMessage("Figure SVG text copied to clipboard", 2500)

        # Toolbar zone (create a tiny horizontal layout via an extra toolbar)
        tb = win.addToolBar("CopyTools")
        btn_png = QPushButton("Copy PNG")
        btn_svg = QPushButton("Copy SVG")
        btn_png.clicked.connect(lambda: copy_png(300))
        btn_svg.clicked.connect(copy_svg)
        tb.addWidget(btn_png)
        tb.addWidget(btn_svg)

        # Keyboard shortcuts
        def _on_key(event):
            # Matplotlib key events (e.key) are lowercase + modifiers separate; easier via Qt shortcut
            pass  # handled by Qt below

        fig.canvas.mpl_connect("key_press_event", _on_key)

        # Qt shortcuts
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        QShortcut(QKeySequence("Ctrl+C"), win, activated=lambda: copy_png(300))
        QShortcut(QKeySequence("Ctrl+Shift+C"), win, activated=copy_svg)

        win._copy_buttons_installed = True

    def add_spectrum_copy_buttons(self, energy_axis, counts_y):
        """
        Add Copy PNG / SVG / Data buttons to spectrum figure (only once).
        Data copied as TSV with columns: Energy_eV, Wavelength_nm, Counts.
        """
        if not self.f_spOpen or not hasattr(self, "spFig"):
            return
        win = getattr(self.spFig.canvas.manager, "window", None)
        if win is None or hasattr(win, "_spec_copy_installed"):
            return

        from PyQt5.QtWidgets import QPushButton, QApplication, QShortcut
        from PyQt5.QtGui import QImage, QKeySequence
        import io, numpy as np

        wavelengths_nm = HC_EV_NM / energy_axis  # (top axis inverse mapping)

        def copy_png(dpi=300):
            buf = io.BytesIO()
            self.spFig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            buf.seek(0)
            qimg = QImage.fromData(buf.getvalue(), "PNG")
            QApplication.clipboard().setImage(qimg)
            if hasattr(win, "statusBar"):
                win.statusBar().showMessage(f"Spectrum PNG copied ({dpi} DPI)", 2000)

        def copy_svg():
            buf = io.StringIO()
            self.spFig.savefig(buf, format="svg", bbox_inches="tight")
            QApplication.clipboard().setText(buf.getvalue())
            if hasattr(win, "statusBar"):
                win.statusBar().showMessage("Spectrum SVG copied", 2000)

        def copy_data():
            arr = np.column_stack([energy_axis, wavelengths_nm, counts_y])
            # Header + TSV
            header = "Energy_eV\tWavelength_nm\tCounts"
            lines = [header] + [f"{e:.6f}\t{w:.2f}\t{c:.6f}" for e, w, c in arr]
            txt = "\n".join(lines)
            QApplication.clipboard().setText(txt)
            if hasattr(win, "statusBar"):
                win.statusBar().showMessage("Spectrum data (TSV) copied", 2500)

        tb = win.addToolBar("SpecCopy")
        b_png = QPushButton("Copy PNG")
        b_svg = QPushButton("Copy SVG")
        b_dat = QPushButton("Copy Data")
        b_png.clicked.connect(lambda: copy_png(300))
        b_svg.clicked.connect(copy_svg)
        b_dat.clicked.connect(copy_data)
        tb.addWidget(b_png); tb.addWidget(b_svg); tb.addWidget(b_dat)

        # Shortcuts
        QShortcut(QKeySequence("Ctrl+C"), win, activated=lambda: copy_png(300))
        QShortcut(QKeySequence("Ctrl+Shift+C"), win, activated=copy_svg)
        QShortcut(QKeySequence("Ctrl+Alt+C"), win, activated=copy_data)

        win._spec_copy_installed = True


def errorMessageWindow(parent, title, message):
    from PyQt5.QtWidgets import QMessageBox

    msgBox = QMessageBox(parent)
    msgBox.setWindowTitle(title)
    msgBox.setText(message)
    msgBox.setIcon(QMessageBox.Warning)
    msgBox.setStandardButtons(QMessageBox.Ok)
    msgBox.exec_()


def plotVerticalBar(ax, xPos, *args, **kwargs):
    """
    Displays a vertical bar in axes ax at x position xPos
    """
    yLims = ax.get_ylim()
    return ax.plot([xPos, xPos], [yLims[0], yLims[1]], *args, **kwargs)


def colorMapName(index, isReversed, cmapList):

    if isReversed:
        name = "{}_r".format(cmapList[index])
    else:
        name = cmapList[index]
    return name


# ---- Add these constants for fixed linecut map layout ----
FIXED_LINECUT_LAYOUT = True
LINECUT_LAYOUT = dict(
    fig_size=(7, 6),   # (width, height) inches – 여기에 원하는 고정 크기
    left=0.15,         # main axes left
    right=0.80,        # main axes right (끝 위치)
    bottom=0.12,
    top=0.94,
    cbar_pad=0.015,    # gap between main axes and colorbar
    cbar_width=0.045   # colorbar width (fraction of figure width)
)

SPECTRUM_FIG_SIZE = (7, 4)  # (width, height) inches  기존 6.4,3.2 보다 큼
SPECTRUM_AX_RECT = [0.10, 0.10, 0.85, 0.7]  # 필요시 여백 조정

# --- (1) 상단 import 아래 아무 곳에 추가 (중복 정의 없을 때) ---
HC_EV_NM = 1239.841984  # eV*nm
DEBUG_MAP_CLICK = True

# === Add near top (after imports) ===
if __name__ == "__main__":

    # interactive mode for matplotlib
    pl.ion()

    app = QApplication(sys.argv)
    app.aboutToQuit.connect(app.deleteLater)

    window = SpaceMapWindow(app)

    window.show()
    try:
        app.exec_()
    except:
        print("exiting")
