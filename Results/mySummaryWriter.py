from os.path import join
import os
import numpy as np
from sklearn.metrics import accuracy_score
import csv
#read logs for training/validation acc/loss
#get predictions and calculate metrics acc/

global filepath
filepath = ''
########################## Set the path for output Directory ########################## 
def set_Outputpath():
    global filepath
    with open('Pathfile.txt', 'r') as myfile:
        filepath = myfile.read()
        filepath = filepath.split("\n")[0]

########################## Reads all history files loops to get data, returns a dic with their names[key] vs name[0] data[1-7] ##########################
def get_History(data_dic):
    for key in data_dic:
        hist_path = join(filepath,'History',key+'.csv')
        results = get_Historydata(hist_path)
        data_dic[key] = [*data_dic[key], *results]
    return data_dic

########################## Reads 1 history file and gets average data for last 10 rows, returns a list data[0-6] ##########################
def get_Historydata(hist_path):
    a = np.genfromtxt(hist_path, delimiter=',')
    lastN = a[-10:]
    data = np.average(lastN,axis=0)
    if 'myecm' in hist_path:
        data = [data[0], data[1], data[3],data[4], data[6], data[8]]
    else:
        data = [data[0], data[1], data[2],data[3], data[4], data[5]]
    return data

########################## Reads All model names in directory and returns a dic with their names[key] vs name[0] ##########################
def get_Modelnames():
    data_dic = dict()
    model_names = os.listdir(join(filepath,'Models'))
    for model_name in model_names:
        data_dic[model_name] = [model_name]
        print(model_name)

    return data_dic

########################## Reads the labels and predictions for all models, returns data_dic with metrics[7-] ##########################
def get_Testmetrics(data_dic):
    for key in data_dic:
        label_path = join(filepath,'TestData',key.split('_')[1]+ '_y_test.npy')
        pred_path = join(filepath,'Predictions',key + '.npy')
        labels = np.load(label_path)
        pred = np.load(pred_path, allow_pickle=True) 
        nonhot_labels = np.argmax(labels, axis=1)
        # print(pred[0].shape)
        if 'myecm' in key:
            nonhot_pred = np.argmax(pred[0],axis=1)
        else:
            nonhot_pred = np.argmax(pred,axis=1)
        acc = accuracy_score(nonhot_labels, nonhot_pred)
        
        data_dic[key] = [ *data_dic[key], acc]
    return data_dic

def write_Summary(data_dic):
    summary_path = os.path.join(filepath,'Summary.csv')
    f = open(summary_path,'w')
    with f:
        writer = csv.writer(f, lineterminator='\n')
        row = ['Model','DB','mod','Merge point','Mergetype','Epochs','Train Acc','Train Loss','lr', 'Val Acc', 'Val Loss', 'Test Acc']
        writer.writerow(row)
        i = 1
        for key in data_dic:
            temp = key.split('_')
            row = [key.split('_')[0], key.split('_')[1], key.split('_')[3], key.split('_')[4], key.split('_')[5], 
                    *[data_dic[key][i] for i in range(1,8)] ]
            writer.writerow(row) 
            i += 1

def write_cond_Summary(data_dic):
    summary_path = os.path.join(filepath,'Condensed Summary.csv')
    cond_data = dict()
    for key in data_dic:
        if key[-2:] == '_1':
            netwrk_key = key[:-2]
            key_2 = netwrk_key + '_2'
            key_3 = netwrk_key + '_3'
            # print(data_dic[key][7])
            # print(data_dic[key_2][7])
            # print(data_dic[key_3][7])
            max_key = np.argmax([data_dic[key][7],data_dic[key_2][7],data_dic[key_3][7]]) + 1
            max_network_key = key[:-2] + '_' + str(max_key)
            cond_data[max_network_key] = data_dic[max_network_key]
    
    summary_path = os.path.join(filepath,'Summary Top 1.csv')
    f = open(summary_path,'w')
    with f:
        writer = csv.writer(f, lineterminator='\n')
        row = ['Model','DB','mod','Merge point','Mergetype','Epochs','Train Acc','Train Loss','lr', 'Val Acc', 'Val Loss', 'Test Acc']
        writer.writerow(row)
        i = 1
        for key in cond_data:
                temp = key.split('_')
                row = [key.split('_')[0], key.split('_')[1], key.split('_')[3], key.split('_')[4], key.split('_')[5], 
                        *[cond_data[key][i] for i in range(1,8)] ]
                writer.writerow(row) 
                i += 1
    
    # f = open(summary_path,'w')
    #     # with f:
        



set_Outputpath()
data_dic = get_Modelnames()
data_dic = get_History(data_dic)
data_dic = get_Testmetrics(data_dic)
write_cond_Summary(data_dic)