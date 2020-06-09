

    ### reads the data files from the folders and generates two files:
    #  db_Images.text: path to all files in a dictionary format with subject id
    #  db_labels_lookup.txt: subject id vs subject name dictionary
    #  use json to read and write
    ###

import os
import json

rootpath = r'E:\Work\Multi Modal Face Recognition\Image Databases\Sejong mmface db'
 
sub_folder_paths = []
files_dic = dict()
labels_dic = dict()

def find_Label(key):

    sub_name = key.split('_')[0]
    label_no = labels_dic[sub_name]

    return label_no


def read_Image_Names(folder_path):
    # folder_path = root...\VIS-TH_Database\001\TH
    # print(folder_path)
    subject = folder_path.split('\\')[-2]
    file_names = os.listdir(folder_path)
    for file in file_names:
        if file.endswith(".jpg"):
            file_path = os.path.join(folder_path,file) # join to create path to image file
            key =  subject + '_' + file                        # file name is the key to dict
            if key in files_dic:              # if the key already exist, put an incremental number at the end of it or print error
                print("Error")
            files_dic[key] = [file_path]      # add file path to dic
            label_no = find_Label(key)        # get subject label (filename) example (TH_019_1_01_NN)
            files_dic[key].append(label_no) 
            #now we have a dict with key = filename .tiff or .jpg
            # dict[0] = file path
            # dict[1] = label number (subject id)


def write_Json():

    with open('SejongDB_label_lookup.txt','w') as lookup:
        json.dump(labels_dic,lookup)
    
    with open('SejongDB_Images.txt','w') as images:
        json.dump(files_dic,images)



sub_folder_names = os.listdir(rootpath) #subject named folder

sub_label = 0
for sub_folder_name in sub_folder_names:
    sub_folder_paths.append(os.path.join(rootpath, sub_folder_name))
    labels_dic[sub_folder_name] = sub_label
    sub_label += 1

for sub_folder_path in sub_folder_paths:
    addonlist = os.listdir(sub_folder_path)
    for addon in addonlist:
        addon_path = os.path.join(sub_folder_path,addon)
        read_Image_Names(addon_path)
        # the_path = os.path.join(sub_folder_path,'TH')
        # read_Image_Names(the_path)


write_Json()

