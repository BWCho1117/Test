"""
This script is for extracting data, e.g. name/surname, from a csv datasheet.

Note that the script can be anywhere on the system, as long as the directory is correctly specified. The code will extract the relevant entries from the database, and then save to the directory specified (currently the same as the database)

Author: TihanB, CBonato, January 2021
"""

import csv
import time, os
import h5py
import numpy as np
import tools.data_object as DO
import tools.QPLemail as QPLm
import pandas

#-----------------------#
#  Database Management  #
#-----------------------#

#Directories
database_directory = 'C:\\Users\\cb2017\\Documents\\ConferencesTalks\\MLQ2021\\Organisation\\AttendeeDatabase\\'
database_filename = '20201022_attendees.csv'

save_directory = database_directory
save_filename = 'Attendees_extracted'


#Create HDF5 file in save directory
#HDF5 = DataObjectHDF5(data_folder = save_directory)

#HDF5_filename = HDF5.create_file(name = save_filename, folder = save_directory)



class AttendeesManager (DO.DataObjectHDF5):

    def __init__ (self, work_folder):
        self._wfolder = work_folder
        self._fDict = None
        self._email_mngr = None
        self._nr_attendees = 0
        self.attendee_dict = {}
        self._countries_list = []

    def add_email_account (self, username, password):
        self._email_mngr = QPLm.EmailManager(username=username, password=password)

    def set_fields (self, fields_dict):
        self._fDict = fields_dict
        self._init_dict()

    def add_attendee (self, firstname, lastname , email, 
                title, sponsor, institution, country):

        attendee = {
            'firstname': firstname,
            'lastname': lastname,
            'email': email,
            'title': title,
            'sponsor': sponsor,
            'institution': institution,
            'country': country,
            }

        self._nr_attendees += 1
        self.attendee_dict[self._nr_attendees] = attendee
        self._countries_list.append(country)

    def set_fields_MLQ2021 (self):
        self._fDict = {
            'firstname': 3,
            'lastname': 4,
            'email': 5,
            'title': 2,
            'sponsor': 13,
            'institution': 6,
            'country': 8,
            'job': 9,
            'field': 10
            }
        self._init_dict()
        for k in self._fDict.keys():
            setattr (self, '_'+k+'_list', [])

    def _init_dict(self):
        for k in self._fDict.keys():
            setattr(self, '_'+k, [])

    def import_csv (self, file_name):

        self.df = pandas.read_csv(file_name)

        try:
            #Open and read attendees database
            workingfile = open(file_name)#(database_directory+database_filename)
            csv_f = csv.reader(workingfile)

            if self._fDict is not None:

                #Constants
                i=0

                #Extract from database to lists
                self.attendee_dict = {}
                for row in csv_f:

                    if i==0:
                        pass
                    else:
                        attendee = {}
                        for k in self._fDict.keys():
                            a = getattr(self, '_'+k)
                            a.append(row[self._fDict[k]])
                            setattr(self, '_'+k, a)
                            attendee[k] = row[self._fDict[k]]
                        self.attendee_dict[i] = attendee
                        if 'country' in self._fDict.keys():
                            self._country_list.append(attendee['country'])
                        if 'job' in self._fDict.keys():
                            self._job_list.append(attendee['job'])
                        if 'field' in self._fDict.keys():
                            self._field_list.append(attendee['field'])

                    i += 1
                self._nr_attendees = i-1
            else:
                print ("You need to specify a dictionary for the fields to be extracted")
        except:
            print('Could not convert to dict, using pandas instead')
            self._fDict = self.df.to_dict()

        return self.df


    def plot_pie(self):
        plot_field = self.df['Field:'].dropna().astype('str').value_counts().plot.pie(fontsize=36, figsize=(20, 20))
        fig = plot_field.get_figure()
        fig.savefig(self._wfolder + 'pie_field.png')
        plt.close(fig)

        plot_occupation = self.df['Occupation:'].dropna().astype('str').value_counts().plot.pie(fontsize=36, figsize=(20, 20))
        fig = plot_occupation.get_figure()
        fig.savefig(self._wfolder + 'pie_occupation.png')
        plt.close(fig)

        plot_country = self.df['Country'].dropna().astype('str').value_counts().plot.pie(fontsize=12, figsize=(20, 20))
        fig = plot_country.get_figure()
        fig.savefig(self._wfolder + 'pie_country.png')
        plt.close(fig)

    def save_dictionary (self, file_name):
        # this does not work
        fname = os.path.join(self._wfolder, file_name+'.hdf5')
        f = h5py.File (fname, 'w')
        f.close()
        self.save_dict_to_file(d=self._fDict, file_name=fname, group_name = 'Attendees')
        

    def format_dict_MLQ2021(self):
        #Populate dictionary of attendees from lists
        for j in range(0,i-1): #Note: i-1 to account for the last increment of i in the previous for loop
            
            #Check for special characters, then empose Capitalised to first and last names
            if any(not c.isalnum() for c in firstnames[j]+lastnames[j]):
            	pass
            else:
            	firstnames[j] = firstnames[j][0].upper()+firstnames[j][1:].lower()
            	lastnames[j] = lastnames[j][0].upper()+lastnames[j][1:].lower()

            attendees[f'{j+1}'] = {'title':title_attendee[j], 'firstname': firstnames[j], 'lastname': lastnames[j], 'email': emailaddress[j], 'institution': institution[j], 'country': country[j], 'sponsor_consent': sponsor_consent[j]}



        #Display data about attendees dictionary
        print('Number of attendees:', len(attendees))
        print('First attendee:', attendees['26'])

    def send_to_all (self, subject, message, replace_fields = True):

        for i in 1+np.arange(self._nr_attendees):
            msg = message
            if replace_fields:
                for k in self._fDict.keys():
                    msg = msg.replace('#'+k+'#', self.attendee_dict[i][k])
            self._email_mngr.send (to=[self.attendee_dict[i]['email'],'mlq2021.conference@gmail.com'], subject=subject, message=msg)

    def export_gmail_mailing_list_csv(self, file_name):
        
        fname = os.path.join(self._wfolder, file_name+'.csv')
        with open(fname, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            
            filewriter.writerow(['Name', 'Email'])
            for k in self.attendee_dict.keys():
                name = self.attendee_dict[k]['firstname']+' '+self.attendee_dict[k]['lastname']
                email = self.attendee_dict[k]['email']
                filewriter.writerow([name, email])
                print (name, email)




