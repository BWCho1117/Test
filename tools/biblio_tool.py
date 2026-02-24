
import os, os.path
import PyPDF2
import platform
import datetime

now = datetime.datetime.now()
curr_year = int(now.year)


def find_root_folder ():
    if (platform.system() == 'Linux'):
        root_folder = 'home/cbonato/CloudFolders/Notes&Literature'
    elif (platform.system() == 'Windows'):
        hostpc = os.getenv('computername')
        if (hostpc == 'HWLAP0134-EPS'):
            root_folder = 'C:/Users/cb76/Dropbox (Heriot-Watt University Team)/Cristian-personal/Literature/'
        else:
            print ('Unknown host pc. Please set root folder')
    else:
        print ("Unsupported operating system")

    return root_folder

def incorp_som (f):

    x = 0
    tag = '_SOM'
    if os.path.isfile (f):
        corpo_som = os.path.splitext (f)[0]
        ext = os.path.splitext (f)[1]
        corpo = corpo_som [:-4]
        main_file = corpo + ext
        som_file = corpo + tag + ext
        if ((os.path.isfile (som_file)) and (os.path.isfile (main_file))):
            tmp_file = corpo+'_tmp'+ext
            os.rename (main_file, tmp_file)
            merger = PyPDF2.PdfFileMerger()
            merger.append(PyPDF2.PdfFileReader(tmp_file, "rb"))
            merger.append(PyPDF2.PdfFileReader(som_file, "rb"))
            merger.write(main_file)
            os.remove(tmp_file)
            os.remove(som_file)
            print (main_file)
            x = 1

    return x

def incorporate_SOM ():

    root_folder = find_root_folder()
    print ('Root folder: ', root_folder)
    
    total = 0
    for base, subs, files in os.walk (root_folder):
        for f in files:
            if f.endswith("_SOM.pdf"):
                print ("Checking: ", f)
                try:
                    total = total + incorp_som (os.path.join(base, f))
                except Exception as e:
                    print ("FAILED! Exception: ", e)

    print ("Merged SOM for "+str(total)+ " papers")

def merge (f_list, file_name):

    merger = PyPDF2.PdfFileMerger()
    for f in f_list:
        merger.append(PyPDF2.PdfFileReader(f, "rb"))
    merger.write(file_name)

def change_year_position (base, f):
    b = f[:-4]
    year_str = b[-4:]
    rest_str = b[:-4]
    if (rest_str[-1] == '_'):
        rest_str = rest_str[:-1]

    success = 0
    try:
        num = int(year_str)
        if ((num<curr_year+1)&(num>1920)):
            new_name = str(num)+"_"+rest_str+'.pdf'
            os.rename (os.path.join(base, f), os.path.join(base, new_name))
            success = 1
    except:
        success = 0

    return success

def set_year_first ():

    root_folder = find_root_folder()

    total = 0
    for base, subs, files in os.walk (root_folder):
        for f in files:
            if f.endswith(".pdf"):
                try:
                    a = change_year_position(base, f)
                    total = total + a
                except:
                    pass

    print ("Changed year position for "+str(total)+" files.")

incorporate_SOM()
set_year_first()
