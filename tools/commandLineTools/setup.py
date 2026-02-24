#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Command-line tool to be called to plot a spectrum in SPE 2 or SPE 3 format (from Princeton)
# 
# Created by R. Proux, 18/01/2018

# For more details, see https://kushaldas.in/posts/building-command-line-tools-in-python-with-click.html

from setuptools import setup

setup(
    name="qplCommandLineTools",
    version='0.1',
    py_modules=['qplCommandLineTools'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        speread=speRead:spe_read
        plottxt=univPlot:plot_file
    ''',
)