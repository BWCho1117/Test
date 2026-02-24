
import os, sys, time
from datetime import datetime
import random
import pylab as plt
import numpy as np
import h5py
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets

class QPLZoomableGUI(QtWidgets.QMainWindow):

    def __init__(self, ui_panel):

        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.ui = ui_panel
        self.ui.setupUi(self)
         
        # connect to mouse events for zoom
        self.ui.canvas.mpl_connect('button_press_event', self.mouseClick)
        self.ui.canvas.mpl_connect('motion_notify_event', self.mouseMove)
        self.ui.canvas.mpl_connect('button_release_event', self.mouseRelease)
        self.mouse_clicked = False
        self.cursor_clicked = False

    def _update_figure (self):
        self.ui.canvas.update_figure()

    def _update_view(self):
        x0 = min (self.x0, self.x1)
        x1 = max (self.x0, self.x1)
        y0 = min (self.y0, self.y1)
        y1 = max (self.y0, self.y1)
        self.ui.canvas.set_range (axis1= [x0, x1], axis2 = [y0, y1])

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        print ("Close window...")
        ce.accept()

    def mouseClick(self, event):
        x = event.xdata
        if (x != None):
            self.cursor_clicked = self._is_click_near_cursor(x)
            if (self.cursor_clicked == False):
                self.x0 = event.xdata
                self.y0 = event.ydata
                self.mouse_clicked = True

    def mouseMove(self, event):
        pass

    def mouseRelease(self, event):
        if (event.xdata != None):
            x = event.xdata

            c = self.cursor_clicked
            if (c == False):
                self.x1 = x
                self.y1 = event.ydata
                if self.mouse_clicked:
                    self.mouse_clicked = False
                    self.ui.canvas.set_range(axis1= [self.x0, self.x1], 
                                                        axis2 = [self.y0, self.y1])
            else:
                self._set_cursor (cursor = c, position =x)

        self._update_view()

    def _zoom_out (self):
        d0, d1 = self._zoom_funct (alpha=1.)
        self.set_view_range (d0, d1)
        self._update_view()

    def _zoom_in (self):
        d0, d1 = self._zoom_funct (alpha=0.25)
        self.set_view_range (d0, d1)
        self._update()

