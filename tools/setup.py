
import sys
import os
import numpy as np
import h5py
import socket


user_settings = {}

user_settings ['DaleMac.local'] = {
    'description': 'Dale Mac',
    'folder': "/Users/dalescerri/Documents/GitHub/Spins/",
    }
user_settings['BAY3-HP'] = {
	'description': 'Bay3 pc in DB2.17',
	'folder': "C:/Research/",
	}
user_settings['cristian-mint'] = {
	'description': 'Cristian - old TU Delft laptop',
	'folder': "/home/cristian/Work/QPL-code/",
	}
user_settings['cristian-PC'] = {
	'description': 'Cristian - Thinkpad',
	'folder': "C:/Users/cristian/Research/QPL-code/",	
}	
user_settings['HWPC0526-EPS'] = {
	'description': 'Cristian - office pc',
	'folder': "H:/Research/QPL-code/",
	}


hostpc = socket.gethostname()
print 'Loaded settings for: ', user_settings[hostpc]['description']

folder = user_settings[hostpc]['folder']
sys.path.append (folder)
sys.path.append (folder+ '/analysis/')
sys.path.append (folder+ '/simulations/')
sys.path.append (folder+ '/measurements/')
sys.path.append (folder+ '/tools/')

os.chdir (folder)

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
mpl.rc('xtick', labelsize=18) 
mpl.rc('ytick', labelsize=18)

try:
	_viridis_data = np.load (folder+'/viridis_cmp.npy')
	viridis = ListedColormap(_viridis_data, name='viridis')
	plt.register_cmap(name='viridis', cmap=viridis)
except:
	pass
