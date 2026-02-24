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
import os.path
import sys
 
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtCore import QTimer
from UiRanges import Ui_ranges
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
 
 
def _vector_to_grid(vec, xN, yN):
    """
    Map a 1D scan vector (row-major order) into a (yN, xN) grid.
    Missing tail (or any absent frames) remain NaN.
    """
    grid = np.full((yN, xN), np.nan, dtype=float)
    n = min(len(vec), xN * yN)
    if n > 0:
        xs = np.arange(n) % xN
        ys = np.arange(n) // xN
        grid[ys, xs] = vec[:n]
    return grid
 
 
class SpaceMapWindow(QWidget):
    """
    Main QDialog class with the measurement and plot properties form.
    Also generates the matplotlib windows for displaying data
    """
 
    def __init__(self, app):
        self.dialog = QWidget.__init__(self)
        self.app = app
 
        # Set up the user interface from Designer.
        self.ui = Ui_ranges()
        self.ui.setupUi(self)
 
        self.cmapList = [
            colormap for colormap in pl.colormaps() if not colormap.endswith("_r")
        ]
        self.ui.colorMap.addItems(self.cmapList)
        self.ui.colorMap.setCurrentIndex(60)
 
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
        self.ui.plotUpdate.clicked.connect(self.update)
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
 
        QTimer.singleShot(100, self.selectFile)  # select file at start
 
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
 
        # read the SPE3/SPE2 file if possible
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
            self.wavelength = self.data.wavelength
            self.wavelengthRangeMeas = (self.wavelength[0], self.wavelength[-1])
            self.nbOfFrames = self.data.nbOfFrames
            self.counts = pl.array([dataRow[0] for dataRow in self.data.data])
 
            self.counts = pl.array([dataRow[0] for dataRow in self.data.data])
            self.bckParams["isDifRef"] = self.ui.isDifRef.isChecked()
 
            self.exposureTime = self.data.exposureTime  # exposure time in ms
 
        except:
            return False
        else:
            return True
 
    def update(self, event=0):
        self.updateProcessedCounts()
        self.f_firstOpening = False
 
        self.plotFigure(self.processedCounts, self.xVoltage, self.yVoltage)
        self.ui.plotUpdate.setText("Update")
 
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
            totalGrid = xNbOfSteps * yNbOfSteps
            if totalGrid != self.nbOfFrames:
                # Warn but continue – we will place what we have and leave NaNs for missing frames
                print(
                    f"[WARN] Expected {totalGrid} frames from grid, found {self.nbOfFrames}. "
                    f"Proceeding with NaN gaps for {totalGrid - self.nbOfFrames} missing frames."
                )
 
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
 
        # if not space map
        else:
            # called Voltage but actually not voltage
            self.xVoltage = self.wavelength
            self.yVoltage = pl.array(range(self.nbOfFrames))
 
            self.processedCounts = self.counts
 
        if not self.f_cacheRecovered and self.f_firstOpening:
            self.colorPlotFullscale(onlySet=True)
            self.xPlotFullscale(onlySet=True)
            self.yPlotFullscale(onlySet=True)
            self.wavePlotFullscale(onlySet=True)
            self.recoverForm()
 
    def recoverForm(self):
        try:
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
 
    def calculateCountArray(self, xNbOfSteps, yNbOfSteps):
        """
        Main function that calculates the data to display given the parameters
        """
        indexWaveRange = self.findWaveIndex(self.plotParams["waveRange"])
 
        if self.bckParams["isDifRef"]:
            bckIndex = self.determineWhichSpectrum(
                (self.bckParams["xBackground"], self.bckParams["yBackground"]),
                indexMove=(0, 0),
            )
            bckIndex = bckIndex[0]
            refmap = (0.0001 * self.counts - 0.0001 * self.counts[bckIndex, :]) / (
                0.0001 * self.counts[bckIndex, :]
            )
 
            if self.plotParams["mode"] == 0:
                countVector = pl.sum(
                    refmap[:, int(indexWaveRange[0]) : int(indexWaveRange[1] + 1)],
                    axis=1,
                )
                resultCounts = _vector_to_grid(countVector, xNbOfSteps, yNbOfSteps)
 
                # Mauro's change
                resultCounts = resultCounts + abs(pl.amin(resultCounts[np.isfinite(resultCounts)])) + 1
 
            else:
                countVector = refmap[
                    :, int(indexWaveRange[0]) : int(indexWaveRange[1] + 1)
                ].max(axis=1)
                resultCounts = _vector_to_grid(countVector, xNbOfSteps, yNbOfSteps)
        else:
 
            if self.plotParams["mode"] == 0:
                countVector = pl.sum(
                    self.counts[:, int(indexWaveRange[0]) : int(indexWaveRange[1] + 1)],
                    axis=1,
                )
                resultCounts = _vector_to_grid(countVector, xNbOfSteps, yNbOfSteps)
            else:
                countVector = self.counts[
                    :, int(indexWaveRange[0]) : int(indexWaveRange[1] + 1)
                ].max(axis=1)
                resultCounts = _vector_to_grid(countVector, xNbOfSteps, yNbOfSteps)
 
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
 
        # Build a proper norm instance (no callable confusion)
        if self.plotParams["scale"] == 0:
            norm = matplotlib.colors.LogNorm()
        else:
            # Linear scale; respect UI color range
            norm = matplotlib.colors.Normalize(
                vmin=self.plotParams["colorRange"][0],
                vmax=self.plotParams["colorRange"][1],
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
            norm=norm,
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
 
        self.fig.canvas.mpl_connect("draw_event", self.updateFormDraw)
        self.fig.canvas.mpl_connect("button_release_event", self.displaySpectrum)
 
        self.fig.canvas.mpl_connect("close_event", self.mapClose)
 
        self.fig.subplots_adjust(left=0.10, right=0.92)  # before colorbar
 
        cb = self.fig.colorbar(mappable=self.pcf, cax=self.colorBarAxes)
        cb.set_label("Counts in {:.0f} ms".format(self.exposureTime))
        cb.ax.minorticks_on()
 
        self.fig.canvas.draw()
        self.fig.canvas.show()
 
    def displaySpectrum(self, event=0, moveX=0, moveY=0):
 
        # get the spectrum to display
        try:
            if event != 0:
                pos = (event.xdata, event.ydata)
            else:
                pos = self.spMapPos
                assert pos is not None
        except:
            pos = (0, 0)
        self.countsIndex, self.spMapPos = self.determineWhichSpectrum(
            pos=pos, indexMove=(moveX, moveY)
        )
 
        # Clamp to last available spectrum if clicked in a NaN (missing) grid cell
        if self.countsIndex >= self.nbOfFrames:
            self.countsIndex = self.nbOfFrames - 1
 
        if self.measParams["isSpaceMap"]:
            figTitle = "Spectrum at X = {:.3f} V, Y = {:.3f} V".format(
                self.spMapPos[0], self.spMapPos[1]
            )
        else:
            figTitle = "Spectrum at index {}".format(self.countsIndex)
 
        if self.bckParams["isDifRef"]:
            bckIndex = self.determineWhichSpectrum(
                (self.bckParams["xBackground"], self.bckParams["yBackground"]),
                indexMove=(0, 0),
            )
            bckIndex = bckIndex[0]
            refmap = (0.0001 * self.counts - 0.0001 * self.counts[bckIndex, :]) / (
                0.0001 * self.counts[bckIndex, :]
            )
            countsPlot = refmap
        else:
            countsPlot = self.counts
 
        xlims = (self.wavelength[0], self.wavelength[-1])
        ymin = pl.amin(countsPlot[self.countsIndex])
        ymax = pl.amax(countsPlot[self.countsIndex])
        ydelta = ymax - ymin
        ylims = (ymin - 0.05 * ydelta, ymax + 0.05 * ydelta)
 
        # update blue cross on map
        if self.f_mapOpen:
            if self.plotSpPos is not None:
                self.plotSpPos.remove()
            self.plotSpPos, = self.mainAxes.plot(
                [self.spMapPos[0]], [self.spMapPos[1]], "xb", markersize=20
            )
            self.fig.canvas.draw()
            self.fig.canvas.show()
 
        # if figure did not exist, let's create it
        if not self.f_spOpen:
            self.f_spOpen = True
 
            # create new figure for spectrum
            self.spFig = pl.figure(figsize=(8.0 * 4.0 / 3.0 * 0.8, 8.0 * 0.4))
            self.spFigNumber = self.spFig.number
            self.spAxes = self.spFig.add_axes([0.1, 0.16, 0.85, 0.74])
 
            # plot spectrum
            self.spPlot, = self.spAxes.plot(
                self.wavelength, countsPlot[self.countsIndex]
            )
            self.spAxes.set_title(figTitle)
            self.spAxes.get_xaxis().set_label_text("Wavelength (nm)")
            self.spAxes.get_yaxis().set_label_text(
                "Counts in {:.0f} ms".format(self.exposureTime)
            )
            self.spAxes.set_xlim(xlims)
            self.spAxes.set_ylim(ylims)
 
            # connect events for choosing wavelength range
            self.spFig.canvas.mpl_connect(
                "button_release_event", self.mousSpectrumClick
            )
            self.spFig.canvas.mpl_connect("close_event", self.spectrumClose)
            self.fig.canvas.mpl_connect("key_press_event", self.keyboardChangeSpectrum)
            self.spFig.canvas.mpl_connect(
                "key_press_event", self.keyboardChangeSpectrum
            )
            self.spFig.canvas.draw()
            self.spFig.canvas.show()
        else:
            # update spectrum
            self.spPlot.set_data(self.wavelength, countsPlot[self.countsIndex])
            self.spAxes.set_title(figTitle)
            self.spAxes.set_xlim(xlims)
            if self.plotParams["spAutoscale"]:
                self.spAxes.set_ylim(ylims)
            self.spFig.canvas.draw()
            self.spFig.canvas.show()
 
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
        if event.button == 3:  # right click
            if self.spVerticalBarCounter == 0 and len(self.spAxes.lines) > 2:
                self.spAxes.lines[-1].remove()
                self.spAxes.lines[-1].remove()
            elif self.spVerticalBarCounter == 1:
                self.spAxes.lines[-1].remove()
 
            self.spFig.canvas.draw()
            self.wavePlotFullscale()
            self.colorPlotFullscale()
 
        elif event.button == 1:  # left click
 
            if self.spVerticalBarCounter == 0 and len(self.spAxes.lines) > 2:
                self.spAxes.lines[-1].remove()
                self.spAxes.lines[-1].remove()
                self.spWaveLimits = []
 
            plotVerticalBar(self.spAxes, event.xdata, "r")
            self.spFig.canvas.draw()
            self.spWaveLimits.append(float(event.xdata))
            self.spVerticalBarCounter = (self.spVerticalBarCounter + 1) % 2
 
            if self.spVerticalBarCounter == 0:
                self.plotParams["waveRange"] = sorted(self.spWaveLimits)
                self.ui.wavePlotMin.setValue(self.plotParams["waveRange"][0])
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
            self.ui.colorPlotMin.setValue(np.nanmin(self.processedCounts))
            self.ui.colorPlotMax.setValue(np.nanmax(self.processedCounts))
 
        finally:
            if not onlySet:
                self.update()
 
    def colorPlotAutoscale(self, onlySet=False):
        try:
            finite_vals = self.processedCounts[np.isfinite(self.processedCounts)]
            if finite_vals.size == 0:
                # nothing finite; fall back to (0,1)
                minValData, maxValData = 0.0, 1.0
                histo = np.array([1.0])
            else:
                histo = pl.histogram(finite_vals, bins=2 ** 8)[0]
                histo = histo.astype(float) / max(1, sum(histo))
                minValData = np.nanmin(self.processedCounts)
                maxValData = np.nanmax(self.processedCounts)
 
            intPower = 0
            iMin = 0
            i = 0
            for i, histBin in enumerate(histo):
                intPower += histBin
                if intPower <= 0.6:
                    iMin = i
                if intPower > 0.99:
                    break
            minVal = float(iMin) / 2 ** 8 * (maxValData - minValData) + minValData
            maxVal = float(max(0, i - 1)) / 2 ** 8 * (maxValData - minValData) + minValData
 
            self.ui.colorPlotMin.setValue(minVal)
            self.ui.colorPlotMax.setValue(maxVal)
 
        finally:
            if not onlySet:
                self.update()
 
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
                self.update()
 
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
                self.update()
 
    def wavePlotFullscale(self, onlySet=False):
        try:
            self.ui.wavePlotMin.setValue(self.wavelengthRangeMeas[0])
            self.ui.wavePlotMax.setValue(self.wavelengthRangeMeas[1])
 
        finally:
            if not onlySet:
                self.update()
 
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