#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:06:12 2016

@author: raphael proux

Small functions used a bit everywhere!

/!\ Python 3 / PyQt 5 version
"""

from PyQt5.QtWidgets import QMessageBox

def cacheName(filename):
    """Simple function to obtain cache files name. 
    
    Args:
        filename (str): base filename
    
    Returns:
        str: base filename with '.cache' extension added
    """
    return (filename+'.cache')
    
def floatOrEmpty(testVar):
    """
    Converts testVar string to float or None if empty.
    
    Args:
        testVar (str): string to test (containing a float or being empty)
    
    Returns:
        float or None: float value if the string contains a float, None if the string is empty. 
        Raises a ValueError if float fails and string is not empty.
    """
    if testVar != '':
        try:
            return float(testVar)
        except ValueError:
            raise
    else:
        return None
        
def increasingTuples(listOfTuples):
    """
    Detects if 2-tuples are increasing strictly (useful to check max/min tuples from form)
    Ignores if one of the value is None
    Parameters: listOfTuples is a list of the 2-tuples to check
    Returns: True if all 2-tuples are strictly increasing (or if None in tuple), False otherwise
    
    Args:
        listOfTuples (list): list of 2-tuples containing values like (min, max)
    
    Returns:
        bool: True if all tuples are increasing strictly, False otherwise.
    """
    for tuple2 in listOfTuples:
        if not(None in tuple2) and tuple2[1] <= tuple2[0]:
            return False
    return True

def argNearest(arrayToSearch, value):
    """
    Find the index of the nearest value in an array
    
    Args:
        arrayToSearch (numpy array): array of numbers
        value (number): value to search in array
    
    Returns:
        int: index of element closest to value in arrayToSearch.
    """
    return (abs(value - arrayToSearch)).argmin()
    
def errorMessageWindow(parentWindow, winTitle, winText):
    """
    Displays a QT error message box, with title, text and OK button
    
    Args:
        parentWindow (QWidget): Parent widget used to display the error window
        winTitle (str): Text displayed as error window title
        winText (str): Text displayed as error message.
    """
    msg = QMessageBox(parentWindow)
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle(winTitle)
    msg.setText(winText)
    msg.exec_()
