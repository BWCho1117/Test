# Access to packages on a new computer (Windows)

Here is the procedure to follow to get the lab packages available on a new Windows 7 computer:

1. Right-click on "Computer" and select "Properties".
2. Go to the "Advanced settings" tab and click on the "Environment variables" button at the bottom.
3. You will see two lists of environment variables. In the top one, you should see in the first column names of variables like "PATH". Check if there is a variable named "PYTHONPATH". 
4. If there is a PYTHONPATH, edit it by clicking "Edit" and skip to step 6. If not, create a PYTHONPATH variable by clicking "New".
5. Name you new variable "PYTHONPATH".
6. Now input the path to the folder where the GIT repositories were cloned (typically "C:\Users\QPL\Desktop\LabSharedPrograms\"). If you are editing a preexisting path, add it at the end separated by a semi-colon (or check other variables to see which separation character you should use).

You can then check if you can import packages from the GIT repositories in Python, but remember to restart the terminal in which you are doing the test — just restarting the kernel might not be enough. 

# Configuration of the Paths for Console usage on Windows

To use Python from the terminal on Windows ([Cmder](http://cmder.net/) is recommended, or at least the Powershell which should be there by default on Windows 7), you need to add some paths to your system, which are associated to your Anaconda distribution:
```
C:\ProgramData\Anaconda3\Scripts;C:\ProgramData\Anaconda3;C:\ProgramData\Anaconda3\Library\bin
```

Add these paths to the Path variable in the System variables window (Right-click Computer > Properties > Advanced system settings > Advanced > Environment variables, bottom list: search for `Path` or `PATH`, etc.). 

If your Anaconda distribution is not installed at this path or if you are not using Anaconda, you should have similar paths leading to utilities like the python executable, iPython or other utilities like the batch file executing `pyuic` (to convert Qt .ui files into python code).


# Python packages to install

To make things easier, here is a list of Python packages that are needed by the different programs we have on Github:

- PyQt, version 4 on Python 2.7 and version 5 on Python >3.6 (classic, usually already there),
- Matplotlib to plot graphs (classic, usually already there),
- Numpy for any scientific calculations (classic, usually already there),
- Scipy is numpy with more (classic, usually already there),
- PyQtGraph to plot nice graphs in QT, very light and fast,
- PyVISA to communicate with instruments using the [VISA](https://en.wikipedia.org/wiki/Virtual_instrument_software_architecture) interface,
- PyDAQmx to specifically handle the National Instruments boxes like the NI-6341,
- PyUSB to handle directly the USB interface (might be able to replace it using PyVISA, to check for the future),
- XMLtoDict allows to import/export XML files from Python easily,
- for the Swabian Instruments correlation card, you need to download the library directly from their website.
- tinyRPC and gevent-websocket handles FPGA boxes like the Swabian Instruments Pulse generator
- tabulate to print nice tables in the console


A small copy-paste from below should make things easier for you:
```
conda install pyqt=5 matplotlib numpy scipy tinyrpc gevent-websocket tabulate
pip install pyqtgraph pyvisa pydaqmx pyusb xmltodict
```

__NOTE:__ for Python 2.7 environments, please replace `pyqt=5` by `pyqt=4` to install PyQt4 instead of PyQt5.

__NOTE 2:__ the default environment from Anaconda should already have PyQt, Numpy, Scipy and Matplotlib. Therefore, you only need to install the package on the _second_ line.
