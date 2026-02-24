
#!/usr/bin/env python2
# 
"""

@author: Cristian Bonato

Class that allows to save all variables in HDF5 files
One can also save the code in specific modules, listed in self._called_modules

Classes that need to load/save data from HDF5 files can inherit this class

"""
import numpy as np
import datetime, time
import h5py
import os, sys

class TimeStampObject ():

    def __init__ (self, data_folder):
        self._called_modules = []
        self._data_folder = data_folder

    def _fill_datetime_str (self, value):
        if (len(value) == 4):
        	value += '01'
        if (len(value) == 6):
        	value += '01'
        if (len(value) == 8):
        	value += '_'

        return value.ljust (15, '0')

    def parse_datetime_str (self, value, fill_void_in = True):
        '''
        Parses a string into a time_stamp (if valid, else returns None).
        The string is supposed to be in '%Y%m%d_%H%M%S' format. 
        If param fill_void_in is set to true, subsets of the format can be used and will be filled with zeros for the missing parts.

        Returns:
        time stamp
        '''

        if fill_void_in:
        	value = self._fill_datetime_str (value)

        try:
        	time_stamp = datetime.datetime.strptime (value, '%Y%m%d_%H%M%S')
        	return time_stamp
        except:
        	print ("String not valid as time tag")
        	return None

    def folder_is_valid (self, folder):
        '''
        This methods will be overloaded by the specific object
        and will tell whether the folder content fulfills 
        what is expected by the specific type of dataset

        Returns:
        	bool -- [description]
        '''
        return True


    def get_file_list (self, folder=None):

        '''
        Return list of files in folder

        Inputs:
        folder [str]: if None, it sets value in self._data_folder
        '''

        if (folder == None):
        	folder = self._data_folder

        file_list = [f for f in os.listdir(folder) 
        		if (os.path.isfile(os.path.join(folder, f)))]
        return file_list

    def get_folder_list (self, folder=None):
        '''
        Return list of sub-folders in folder

        Inputs:
        folder [str]: if None, it sets value in self._data_folder
        '''

        if (folder == None):
        	folder = self._data_folder

        folder_list = [f for f in os.listdir(folder) 
        		if (os.path.isdir(os.path.join(folder, f)))]
        return folder_list

    def get_data_folder_list (self, folder=None):

        '''
        Returns list of "valid" data_sub-folders in folder.
        Valid here means the folder has the correct time stamp.

        Inputs:
        folder [str]: if None, self._data_folder

        Outputs:
        list of data sub-folders, list of corresponding time stmaps

        [description]
        '''

        folder_list = self.get_folder_list (folder = folder)
        data_folder_list = []
        time_stamp_list = []

        for f in folder_list:
        	try:
        		tag = f[:15]
        		time_tag = self.parse_datetime_str (value = tag, fill_void_in = False)
        		valid_tag = True
        	except:
        		valid_tag = False

        	if valid_tag:
        		if self.folder_is_valid (f):
        			data_folder_list.append(f)
        			time_stamp_list.append(time_tag)

        return data_folder_list, time_stamp_list

    def get_data_with_tag (self, tag_list = [], logic_operator = 'all'):

        '''
        Returns list of valid data_folders which incluse "any" or "all" the tag specified in tag_list

        Inputs:
        tag_list: list of tags (str)
        logic_operator: 'all' or 'any'

        Outputs:
        List of data_folders satisfying the conditions
        '''

        tag_data_list = []

        if (logic_operator in ['all', 'any']):

            data_folder_list, time_stamp_list = self.get_data_folder_list ()

            for f in data_folder_list:

                if self.folder_is_valid(f):

                    bool_list = [(t in f) for t in tag_list]

                    if (logic_operator == 'all'):
                        valid = all(bool_list)
                    elif (logic_operator == 'any'):
                        valid = any (bool_list)

                    if valid:
                        tag_data_list.append (f)

        else:
            print ("Allowed logic operators: all, any")

        return tag_data_list



    def get_data_between_stamps (self, later_than, earlier_than):

        '''
        Returns list of valid data_folders with time_stamps between the values specified by "lather_than" and "earlier_than"

        Inputs:
        later_than [str]: time stamp
        earlier_than [str]: time stamp

        Outputs:
        List of data_folders satisfying the conditions
        '''

        f_list = []

        if ((later_than != None) and (earlier_than != None)):
        	t1 = self.parse_datetime_str (later_than, fill_void_in = True)
        	t2 = self.parse_datetime_str (earlier_than, fill_void_in = True)

        	f_list = []
        	t_list = []

        	if ((t1 != None) and (t2 != None)):

        		data_folder_list, time_stamp_list = self.get_data_folder_list ()

        		for idx, t in enumerate(time_stamp_list):

        			if ((t<=t2) and (t>=t1)):
        				f_list.append (data_folder_list[idx])

        return f_list


    def get_files_with_tag (self, tag_list = [], logic_operator = 'all', folder = None):

        '''
        Returns list of valid data_folders which incluse "any" or "all" the tag specified in tag_list

        Inputs:
        tag_list: list of tags (str)
        logic_operator: 'all' or 'any'

        Outputs:
        List of data_folders satisfying the conditions
        '''

        if folder is None:
            folder = self._data_folder

        tag_data_list = []

        if (logic_operator in ['all', 'any']):

            file_list = self.get_file_list (folder=folder)

            for f in file_list:

                    f_full = os.path.join (folder, f)
                    bool_list = [(t in f_full) for t in tag_list]

                    if (logic_operator == 'all'):
                        valid = all(bool_list)
                    elif (logic_operator == 'any'):
                        valid = any (bool_list)

                    if valid:
                        tag_data_list.append (f)

        else:
            print ("Allowed logic operators: all, any")

        return tag_data_list




class DataObjectHDF5 (TimeStampObject):

    def __init__(self, data_folder = None):
        super().__init__ (data_folder)
        self.data_dict = {}
        self._called_modules=[]
        self._folder = None

    def create_file (self, name, folder=None):
        if (folder != None):
            self._folder = self.create_folder (folder)
        if (self._folder != None):
            tstamp = time.strftime ('%Y%m%d_%H%M%S')
            if name != '':
                name = '_'+name
            fname = os.path.join (self._folder, tstamp + '_' + name + '.hdf5')
            f = h5py.File (fname, 'w')
            f.close()
        else:
            fname = False
            print ("No folder has been selected - file not created!")
        return fname

    def create_folder (self, folder, name = ''):
        tstamp = time.strftime ('%Y%m%d_%H%M%S')
        if name != '':
            name = '_'+name
        subf = os.path.join (folder, tstamp + '_'+ name + '/')
        if not(os.path.exists(subf)):
            os.mkdir(subf)
        self._folder = subf
        return subf        

    def save_dict_to_file(self, d, file_name, group_name = None):

        '''
        Saves a dictionary d to a file )

        Inputs:
        d:			dictionary
        file_name: 	name of existing hdf5 file
        group_name: name of new group (if you wnat to save data to a new group in the file)

        '''

        with h5py.File (file_name, 'a') as file_handle:

        	if (group_name == None):
        		handle = file_handle
        	else:
        		grp = file_handle.create_group (group_name)
        		handle = grp

        	for k in d.keys():
        		if isinstance(d[k], dict):
        			grp = handle.create_group(k)
        			self.save_dict_to_file (d = d[k], file_handle = grp)
        		elif (type (d[k]) in [int, float, str]):
        			handle.attrs[k] = d[k]
        		elif isinstance(d[k], np.int32):
        			handle.attrs[k] = d[k]
        		elif isinstance(d[k], np.float64):
        			handle.attrs[k] = d[k]
        		elif isinstance(d[k], np.ndarray):
        			handle.create_dataset (k, data = d[k])

    def load_all_attributes (self, obj, file_name):
        '''
        Loads all attributes (variables) in hdf5 file (file_handle) into an object (obj)

        Inputs:
        obj: object
        file_name
        '''
        with h5py.File (file_name, 'r') as file_handle:
        	for k in file_handle.attrs.keys():
        		setattr (obj, k, file_handle.attrs [k])

    def load_all_datasets (self, obj, file_name):
        '''
        Loads all datasets (arrays) in hdf5 file (file_handle) into an object (obj)

        Inputs:
        obj: object
        file_handle
        '''

        with h5py.File (file_name, 'r') as file_handle:
            all_h5_objs = []
            file_handle.visit(all_h5_objs.append)
            all_datasets = [ obj for obj in all_h5_objs if isinstance(file_handle[obj],h5py.Dataset)]
            for k in all_datasets:
                setattr (obj, k, file_handle[k].value)

    def load_all_into_object (self, obj, file_name):
        self.load_all_attributes (obj=obj, file_name = file_name)
        self.load_all_datasets (obj=obj, file_name = file_name)

    def find_groups_and_datasets (self, file_handle):

        all_h5_objs = []
        file_handle.visit(all_h5_objs.append)
        all_groups   = [ obj for obj in all_h5_objs if isinstance(file_handle[obj],h5py.Group) ]
        all_datasets = [ obj for obj in all_h5_objs if isinstance(file_handle[obj],h5py.Dataset)]
        return all_groups, all_datasets

    def store_function_code (self):
        self.data_dict['code'] = {}
        for i in self._called_modules:
        	try:
        		self.data_dict['code'][i] = inspect.getsource(getattr(self, i))
        	except:
        		print ("Non-existing function: ", i)
        return self.data_dict['code']

    def save_object_to_file (self, obj, file_name, group_name = None, arrays_to_txt = []):

        with h5py.File (file_name, 'a') as file_handle:

            if (group_name == None):
                handle = file_handle
            else:
                grp = file_handle.create_group (group_name)
                handle = grp

            for k in obj.__dict__.keys():

                if (type(obj.__dict__[k]) in [int, float, str]):
                    try:
                        handle.attrs[k] = getattr (obj, k)
                    except:
                        handle[k] = getattr (obj, k)
                elif isinstance(obj.__dict__[k], np.float64):
                    try:
                        handle.attrs[k] = getattr (obj, k)
                    except:
                        handle[k] = getattr (obj, k)
                elif isinstance(obj.__dict__[k], np.int32):
                    try:
                        handle.attrs[k] = getattr (obj, k)
                    except:
                        handle[k] = getattr (obj, k)
                elif isinstance(obj.__dict__[k], np.ndarray):
                    try:
                        handle.create_dataset (k, data = getattr (obj, k))
                    except:
                        # if there is a list of strings, we need to convert 
                        #unicode to ascii since it is not supported by hdf5
                        asciiList = [n.encode("ascii", "ignore") for n in getattr (obj, k)]
                        handle.create_dataset (k, data = asciiList)
                elif (type (obj.__dict__[k]) is list):
                    try:
                        # there's some problems when saving lists of strings
                        b = [type (s) for s in obj.__dict__[k]]
                        c = [s is str for s in b]
                        if (any(c)):
                            # if there is a list of strings, we need to convert unicode to ascii
                            # since unicode is not supported by hdf5
                            asciiList = [n.encode("ascii", "ignore") for n in getattr (obj, k)]
                            handle.create_dataset (k, data = asciiList)
                        else:
                            handle.create_dataset (k, data = getattr (obj, k))
                    except:
                        pass
        
        if len(arrays_to_txt)>0:
            fname_txt = file_name[:-5]+'.txt'
            np.savetxt (fname_txt, arrays_to_txt)

    def save_object_all_vars_to_file (self, obj, file_name, group_name = None):

        with h5py.File (file_name, 'a') as file_handle:

        	if (group_name == None):
        		handle = file_handle
        	else:
        		grp = file_handle.require_group (group_name)
        		handle = grp

        	for k in obj.__dict__.keys():
        		if (type(obj.__dict__[k]) in [int, float, str]):
        			try:
        				handle.attrs[k] = getattr (obj, k)
        			except:
        				handle[k] = getattr (obj, k)
        		elif isinstance(obj.__dict__[k], np.float64):
        			try:
        				handle.attrs[k] = getattr (obj, k)
        			except:
        				handle[k] = getattr (obj, k)
        		elif isinstance(obj.__dict__[k], np.int32):
        			try:
        				handle.attrs[k] = getattr (obj, k)
        			except:
        				handle[k] = getattr (obj, k)

    def save_object_params_list_to_file (self, obj, file_name, params_list, group_name = None):

        with h5py.File (file_name, 'a') as file_handle:

        	if (group_name == None):
        		handle = file_handle
        	else:
        		grp = file_handle.require_group (group_name)
        		handle = grp

        	for k in params_list:
        		try:
        			p = getattr (obj, k)
        			if (type(p) in [int, float, str]):
        				try:
        					handle.attrs[k] = p
        				except:
        					handle[k] = p
        			elif isinstance(p, np.float64):
        				try:
        					handle.attrs[k] = p
        				except:
        					handle[k] = p
        			elif isinstance(p, np.int32):
        				try:
        					handle.attrs[k] = p
        				except:
        					handle[k] = p
        			elif isinstance(p, np.ndarray):
        				handle.create_dataset (k, data = p)
        			elif (type (p) is list):
        				# there's some problems when saving lists of strings
        				b = [type (s) for s in p]
        				c = [s is str for s in b]
        				if (any(c)):
        					pass
        				else:
        					handle.create_dataset (k, data = p)
        		except Exception as e:
        			print ("Variable ", k, " cannot be saved. Exception: ", e)
