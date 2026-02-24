# qplCommandLineTools – command-line tools for QPL

Set of useful command-line tools for the everyday lab activity.

## Installation
Go inside this folder with your terminal
```
cd /blablabla/git-repos/tools/commandLineTools
```

You can then install the command-line tools by simply executing
```
pip install --editable .
```
Note that you can add an `--editable` flag if you are developping a command-line tool and don't want to execute `pip install .` every time you want to test your modifications.

### Note for Windows users
You can use the Powershell to use these tools!!! Simply add the `C:\Path\To\Anaconda3\Scripts` path to your environment variable `Path`.

Reminder (Windows 7): you can modify the `Path` environment variable by clicking right on `Computer>Properties` in the start menu or the explorer, then `Advanced system settings` on the left menu, `Advanced` tab, `Environment variables...` and finally editing the `Path` variable in your System variables list.

## Add a new command-line tool

So you have this new tool that you would like to develop and share. These command-line tools are based on the [Click package](http://click.pocoo.org) which provides an easy way of implementing command-line tools and documenting them automatically. The basic usage is described below.

### First step: configure your program

Let's pretend you have a `plotTrace.py` file containing a `plot_trace(filename)` function that you would like to execute a command-line tool `plottrace PATH`. In your `plotTrace.py` python file, above the `plot_trace()` function, you have to add decorators to tell Click that you want this function to be called when the command is called:
```
@click.command()
@click.argument('filename', type=click.Path(exists=True))
def plot_trace(filename):
    # do something here
    pass
```

To check all the options and arguments possibilities, check the [documentation of the package](http://click.pocoo.org).

### Second step: tell your computer where to look for your script

To be able to call your Python function from your terminal, you need to modify the `setup.py` file in the `commandLineTools` folder. At the moment, this file contains the following:
```
from setuptools import setup

setup(
    name="qplCommandLineTools",
    version='0.1',
    py_modules=['speRead'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        speread=speRead:spe_read
    ''',
)
```

To add your own command `plottrace`, you simply need to modify the `entry_points` variable by adding a line at the end, as such:
```
entry_points='''
        [console_scripts]
        speread=speRead:spe_read
        plottrace=plotTrace:plot_trace
    '''
```

The syntax for each line is: `command=pythonFile:pythonFunction`.

You have to reinstall the `qplCommandLineTools` package to take your new command into account (see Installation section above, no uninstall required).

## Uninstallation

To uninstall, run the following from inside the `commandLineTools` folder:
```
pip uninstall .
```

