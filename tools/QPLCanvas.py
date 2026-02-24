

from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import h5py
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.colors as colors


class MplCanvas(Canvas):

    def __init__(self, parent=None, width=2, height=2, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.tick_params(labelsize=15)
        self._w_inches = width
        self._h_inches = height
        self._dpi = dpi
        self.fig.subplots_adjust(left=0.15,right=0.95,
                            bottom=0.15,top=0.95,
                            hspace=0,wspace=0)

        Canvas.__init__(self, self.fig)
        self.setParent(parent)

        Canvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

    def update_figure (self):
        pass

    def set_tick_fontsize (self, fontsize):
        self.axes.tick_params(labelsize=fontsize)
        
    def resize_canvas (self, w, h):
        self._w_inches = w/float(self._dpi)
        self._h_inches = h/float(self._dpi)
        self.fig.set_size_inches (self._w_inches, self._h_inches)
        self.draw()

    def clear (self):
        self.axes.clear()


class ZoomableCanvas (MplCanvas):

    def __init__ (self, *args, **kwargs):
        MplCanvas.__init__ (self, *args, **kwargs)
        self.cursor_clicked = False

    def _is_click_near_cursor (self, x):
        pass

    def set_cursor (self, cursor, position):
        pass

    def _draw_cursors (self):
        pass        

    def draw_zoom_rect (self, x0, y0, x1, y1):
        pass

    def close_zoom_rect (self):
        pass

    def set_range (self, x0, y0, x1, y1):
        pass

    def mouseClick(self, event):
        x = event.xdata
        if (x != None):
            self.cursor_clicked = self._is_click_near_cursor(x)
            if (self.cursor_clicked == False):
                self._mouse_x0 = event.xdata
                self._mouse_y0 = event.ydata
                self.mouse_clicked = True

    def mouseMove(self, event):
        if self.mouse_clicked:
            if (event.xdata != None):
                self.x1 = event.xdata
                # make sure you call the correct zoom rectangle, whether it's x-only or x/y
                self.draw_zoom_rect (self.x0, self.y0, self.x1, self.y1)

    def mouseRelease(self, event):
        if (event.xdata != None):
            x = event.xdata

            c = self.cursor_clicked
            if (c == False):
                self.x1 = x
                self.y1 = event.ydata
                if self.mouse_clicked:
                    self.mouse_clicked = False
                    self.close_zoom_rect()
                    self.set_range(self.x0, self.y0, self.x1, self.y1)
            else:
                self._set_cursor (cursor = c, position =x)

        self.update()


class QPLCanvas_multi1D(ZoomableCanvas):

    def __init__(self, *args, **kwargs):
        ZoomableCanvas.__init__(self, *args, **kwargs)

    def set_time_interval (self, t):
        self._t = t

    def set_range(self, t0, t1):
        self.axes.set_xlim ([t0, t1])
        self._cursor_x0 = t0
        self._scursor_x1 = t1
        self._draw_cursors()
        self.axes.figure.canvas.draw_idle()
        self.repaint()

    def reset_cursors(self):
        self._cursor_x0 = self._x_min + 5
        self._cursor_x1 = self._x_max - 5

        self._draw_cursor0()
        self._draw_cursor1()

    def enable_cursors (self, value):
        self._cursors_enabled = value

    def reset_canvas (self):
        self.fig.clf()
        
        self.axes = []
        self._nr_chans_in_view = int(sum(self._view_chs))

        for i in range(self._nr_chans_in_view):
            num = self._nr_chans_in_view*100+10+i+1
            self.axes.append(self.fig.add_subplot (num))
        self.fig.subplots_adjust(hspace=0)

    def set_cursor (self, cursor, position):
        position = int(position)
        if (cursor == "c0"):
            self._cursor_x0 = position
            self._draw_cursor0()
        elif (cursor == "c1"):
            self._cursor_x1 = position
            self._draw_cursor1()

    def get_cursors (self):
        return self._cursor_x0, self._cursor_x1

    def _draw_cursor0 (self):

        try:
            while (len(self._lines0)>0):
                self._lines0.pop().remove()
        except:
            pass

        if self._cursors_enabled:
            print ("drawing cursor 0!!")
            if (self._pdict == None):
                self._pdict = self._stream.get_plot_dict()

            i = 0
            for ind, ch in enumerate(self._stream.ch_list):

                if (int(self._view_chs[ind]) == 1):
                    self._lines0.append(self.axes[i].axvline(x=self._cursor_x0, color='yellow', linestyle='--'))
                    i+=1

        self.repaint()

    def _draw_cursor1 (self):

        try:
            while (len(self._lines1)>0):
                self._lines1.pop().remove()
        except:
            pass

        if self._cursors_enabled:
            print ("drawing cursor 0!!")

            if (self._pdict == None):
                self._pdict = self._stream.get_plot_dict()

            i = 0
            for ind, ch in enumerate(self._stream.ch_list):

                if (int(self._view_chs[ind]) == 1):
                    self._lines1.append(self.axes[i].axvline(x=self._cursor_x1, color='yellow', linestyle='--'))
                    i+=1

        self.repaint()

    def set_view_channels(self, channels_idx):
        self._view_chs = channels_idx
        self.reset_canvas ()

    def update_figure (self):
        self.clear()
        self._plot_channels()
        for i in range(self._nr_chans_in_view):
            self.axes[i].figure.canvas.draw_idle()
        self.repaint()

    def clear (self):
        for i in range(self._nr_chans_in_view):
            self.axes[i].cla()

    def set_time_range(self, t0, t1):
        self._x_min = t0
        self._x_max = t1
        for i in range(self._nr_chans_in_view):
            self.axes[i].set_xlim ([t0, t1])
            self.axes[i].figure.canvas.draw_idle()
        self.repaint()


class QPLCanvas_2D(ZoomableCanvas):

    def __init__(self, *args, **kwargs):
        ZoomableCanvas.__init__(self, *args, **kwargs)
        self._2D_matrix = []

    def reinitialize(self):
        self.axes.clear()

    def _recalculate (self, values):
        x0 =  values[0]
        x1 = values[-1]
        dx = values[1]-values[0]
        N = len(values)
        return np.linspace (x0-dx/2, x1+dx/2, N+1)

    def set_range (self, axis1, axis2):
        self.axes.set_xlim ([axis1[0], axis1[1]])
        self.axes.set_ylim ([axis2[0], axis2[1]])
        self.update_figure()
        
    def set_data (self, i1, i2, value):
        self._2D_matrix [i1, i2] = value
        self._update_figure()

    def set_format_axes_ticks (self, digits=1):
        self._digits = digits

    def _format (self, x):
        return [int(i*(10**self._digits))/(10**self._digits) for i in x]

    def set_2D_data (self, value, x=None, y=None, scan_units = 'V', color_scale = 'Lin', colormap = 'viridis'):
        if ((x is not None) and (y is not None)):
            [X, Y] = np.meshgrid (self._recalculate(x), self._recalculate(y))
            if colormap != None:
                cmap = plt.get_cmap(colormap)

            if color_scale == 'Lin':
                self.plotdata = self.axes.pcolor (X, Y, np.transpose(value), cmap=cmap)
            elif color_scale == 'Log':
                self.plotdata = self.axes.pcolor (X, Y, np.transpose(value), norm = colors.LogNorm(vmin=max(value[~np.isnan(value)].min(),1), vmax = max(value[~np.isnan(value)].max(),2)),cmap=cmap)
            else:
                raise NameError('Invalid color scale! Options are Lin or Log, as strings.')
            self.axes.xaxis.set_ticks (self._format(x))
            self.axes.yaxis.set_ticks (self._format(y))
            self.axes.set_xlabel ("X ("+str(scan_units)+')', fontsize = 20)
            self.axes.set_ylabel ("Y ("+str(scan_units)+')', fontsize = 20)

        else:
            self.update_figure()

    def update_figure (self):
        self.draw()
        self.repaint()
        

    def add_zoom_rect(self):
        pass

    def draw_1D_zoom_rect (self, x0, x1):
        pass

    def close_zoom_rect(self):
        pass


