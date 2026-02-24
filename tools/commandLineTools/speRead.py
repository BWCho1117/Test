#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Command-line tool to be called to plot a spectrum in SPE 2 or SPE 3 format (from Princeton)
# 
# Created by R. Proux, 18/01/2018
# Used code from jabaldonedo 19/01/2018
#    https://stackoverflow.com/questions/18390461/scroll-backwards-and-forwards-through-matplotlib-plots

import matplotlib
matplotlib.use('Qt5Agg')
import pylab as pl
import click
import os
import tools.instruments.princeton3.SPE2read as spe2
import tools.instruments.princeton3.SPE3read as spe3


@click.command()
@click.argument('filename', nargs=1, required=True, type=click.Path(exists=True))
@click.option('--export', '-o', type=click.Path(exists=False, writable=True), nargs=1, help="export as text file to this path.")
@click.option('--autoexport', '-a', is_flag=True, help="export as text file with automatic name (name of original file + frame number).")
@click.option('--index', '-i', type=click.INT, nargs=1, default=0, help="index of spectrum to export or to display first.")
def spe_read(filename, index=0, export=None, autoexport=False):
    """
    Reads, plots and exports an SPE2 or SPE3 file from Princeton Instruments (WinSpec and Lightfield).
    Handles multiple frames by using left and right arrows (try combining with alt and windows/cmd/super/ctrl to change scrolling speed).
    """
    if autoexport:
        filename_noext, _ = os.path.splitext(filename)
        export = '{}-frame{}.txt'.format(filename_noext, index)
    spectrum = SpeRead(filename, start_index=index, export=export)

class SpeRead():
    def __init__(self, filename, start_index=0, export=None, autoexport=False):
        """Object used to read SPE files, plot them and export them to a text file
        
        Args:
            filename (str): path to the SPE file
            start_index (int, optional): index of the frame you want to export or the plot to start.
            export (str, optional): path to the exported text file. If None, will plot the spectra, if not None, will export.
            autoexport (bool, optional): if True, will export with an automated 'filename + frame number' named text file.
        """
        self.data = self.open_spe(filename)

        self.curr_pos = start_index

        if export is None:
            self.init_plot(start_index)
        else:
            save_array = pl.array([self.data.wavelength, self.data.data[start_index][0]]).transpose()
            pl.savetxt(export, save_array)

    def open_spe(self, filename):
        """Opens the SPE file as an SPE3map or SPE2map (automatic selection) from the corresponding libraries.
        
        Args:
            filename (str): path to the SPE file to open
        
        Returns:
            SPE3map or SPE2map object: object to handle the file (access to wavelength vector, different frames, etc.)
        """
        try:
            data = spe3.SPE3map(filename)
        except:
            data = spe2.SPE2map(filename)

        return data

    def init_plot(self, start_index):
        """First plot, to call before plot_spe() which is meant to update the plot only.
        Note it will make the first call for plot_spe() so plot_spe() can be called by the user to then update the plot.
        
        Args:
            start_index (int): frame index for the first plot
        """
        self.fig = pl.figure()
        self.fig.canvas.mpl_connect('key_press_event', self.key_event)
        self.ax = self.fig.add_subplot(111)
        self.plot_spe(start_index)
        pl.show()

    def plot_spe(self, index):
        """Update the plot (should be called after init_plot())
        
        Args:
            index (int): frame index for the new plot
        """
        self.ax.cla()
        pl.plot(self.data.wavelength, self.data.data[index][0])
        pl.xlabel('Wavelength (nm)')
        pl.ylabel('Intensity (counts/{:.3f} ms)'.format(self.data.exposureTime))
        pl.title('Spectrum {}/{}'.format(self.curr_pos + 1, self.data.nbOfFrames))
        self.fig.canvas.draw()

    def key_event(self, e):
        """Callback called when a key is pressed in the matplotlib window.
        
        Args:
            e (event identifier (matplotlib)): is provided automatically by matplotlib when the callback is called.
        
        """
        if e.key == "right":
            self.curr_pos = self.curr_pos + 1
        elif e.key == "left":
            self.curr_pos = self.curr_pos - 1
        elif e.key == "alt+right":
            self.curr_pos = self.curr_pos + 25
        elif e.key == "alt+left":
            self.curr_pos = self.curr_pos - 25
        elif e.key == "alt+super+right" or e.key == "ctrl+alt+right":
            self.curr_pos = self.curr_pos + 250
        elif e.key == "alt+super+left" or  e.key == "ctrl+alt+left":
            self.curr_pos = self.curr_pos - 250
        else:
            return
        self.curr_pos = self.curr_pos % self.data.nbOfFrames

        self.plot_spe(self.curr_pos)

        

if __name__ == '__main__':
    # spe_read(r'/Users/raphaelproux/Desktop/kspace-experiment/DATA/InSe/180112-InSe-fresh-batch-S6GM1/spectra/SPOT-B/180112-S6GM1-FlakeB1-thin-INT-2s-step-glue-700-1100nm-EXC-532nm-0.20V1E3.spe', export='toto.txt')
    spectrum = SpeRead(r'/Users/raphaelproux/Desktop/kspace-experiment/DATA/InSe/180112-InSe-fresh-batch-S6GM1/spectra/SPOT-B/180112-S6GM1-FlakeB1-thin-INT-2s-step-glue-700-1100nm-EXC-532nm-0.20V1E3.spe', start_index=0, export='toto.txt')


