#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Command-line tool to plot a text file in a flexible universal way. 
# Will try different types of delimiters and skip automatically rows it cannot interpret at the beginning of the file.
# 
# Created by R. Proux, 24/01/2018

import matplotlib
matplotlib.use('Qt5Agg')
import pylab as pl
import click
import glob
import os

@click.command()
@click.argument('filenames', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--rows', '-r', 'force_shape', flag_value='rows', help="Force to interpret rows as the X/Y data vectors.")
@click.option('--columns', '-c', 'force_shape', flag_value='columns', help="Force to interpret columns as the X/Y data vectors.")
@click.option('--autoshape', '-a', 'force_shape', flag_value='auto', default=True, help="Use the longer side of the array to read the X/Y data vectors.")
@click.option('--yonly', '-y', is_flag=True, help="Generate X data based on index and consider all rows/columns to be Y data.")
def plot_file(filenames, force_shape='auto', yonly=False):
    """
    Reads and plots text files containing data in columns or lines.
    Will try different types of delimiters and skip automatically rows it cannot interpret at the beginning of the file.
    """
    plotfile(filenames, force_shape=force_shape, yonly=yonly)

def plotfile(filenames, force_shape='auto', yonly=False):
    """
    Args:
        filenames (tuple or list): list of filenames containing the data to plot (can be parsed with *)
        force_shape (str, optional): change the way the data is read from the file.
                                     Possible values are 'auto', 'columns' or 'rows' (if random, will go to 'rows')
                                     'auto' will use the longest dimension as X/Y data for plotting
                                     'columns' will use the columns as X/Y data
                                     'rows' will use the rows as X/Y data
        yonly (bool, optional): Plot all the vectors in the file as Y data (X will then be generated from the vector indices)
                                If False, the first column/row will be used for X data.
    """
    plotted_something = False
    fig = pl.figure()
    for filename_pattern in filenames:
        list_of_filenames = glob.glob(filename_pattern)
        for filename in list_of_filenames:
            try:
                file_array = loadtxt_gen_skiprows(filename)
            except IsADirectoryError:
                continue
            # print(file_array.shape)
            if len(file_array.shape) > 1 and ((file_array.shape[0] > file_array.shape[1] and force_shape == 'auto') or force_shape == 'columns'):
                file_array = file_array.transpose()
            if len(file_array.shape) == 1 or yonly:
                # print(len(file_array), 'YOLOOOOOO', file_array)
                file_array = pl.vstack([range(file_array.transpose().shape[0]), file_array])
            for y_vector in file_array[1:]:
                pl.plot(file_array[0], y_vector, label=os.path.basename(filename))
                if not plotted_something:
                    plotted_something = True
    if plotted_something:
        pl.legend()
        pl.show()
    else:
        pl.close(fig)


def loadtxt_gen_delimiter(filename, load_func=pl.loadtxt, **opt_dict):
    """Wrapper for a numpy.loadtxt style function which reads data from a text file, here trying
    different delimiters (None means tab or space, comma and semi-colon)
    
    Args:
        filename (str): Path to the file to read
        load_func (function, optional): Numpy loadtxt style function (could be another wrapper)
        **opt_dict: options to pass to load_func
    
    Returns:
        numpy array: numpy array read from the file
    """
    try:
        file_array = load_func(filename, **opt_dict)
    except ValueError:
        try:
            file_array = load_func(filename, delimiter=",", **opt_dict)
        except ValueError:
            file_array = load_func(filename, delimiter=";", **opt_dict)
    return file_array

def loadtxt_gen_skiprows(filename, load_func=loadtxt_gen_delimiter, **opt_dict):
    """Wrapper for a numpy.loadtxt style function which reads data from a text file, here trying
    different header sizes (number of skipped lines at the top of the file).
    
    Args:
        filename (str): Path to the file to read
        load_func (function, optional): Numpy loadtxt style function (could be another wrapper)
        **opt_dict: options to pass to load_func
    
    Returns:
        numpy array: numpy array read from the file
    """
    i = 0
    while True:
        try:
            return load_func(filename, skiprows=i, **opt_dict)
        except ValueError:
            i = i + 1
        except StopIteration:  # reached the end of the file!
            raise
        else:
            break

if __name__ == '__main__':
    
    plot_file([r'/Users/raphaelproux/Desktop/test-plotuni/test-*.txt'])


