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
    return data

########################## Reads All model names in directory and returns a dic with their names[key] vs name[0] ##########################
def get_Modelnames():
    data_dic = dict()
    model_names = os.listdir(join(filepath,'Models'))
    for model_name in model_names:
        data_dic[model_name] = [model_name] 

    return data_dic

########################## Reads the labels and predictions for all models, returns data_dic with metrics[7-] ##########################
def get_Testmetrics(data_dic):
    for key in data_dic:
        label_path = join(filepath,'TestData',key.split('_')[2]+ '_y_test.npy')
        pred_path = join(filepath,'Predictions',key[12:] + '.npy')
        labels = np.load(label_path)
        pred = np.load(pred_path) 
        nonhot_labels = np.argmax(labels, axis=1)
        nonhot_pred = np.argmax(pred,axis=1)
        acc = accuracy_score(nonhot_labels, nonhot_pred)
        data_dic[key] = [ *data_dic[key], acc]
    return data_dic

def write_Summary(data_dic):
    summary_path = os.path.join(filepath,'Summary.csv')
    f = open(summary_path,'w')
    with f:
        writer = csv.writer(f)
        row = ['Model','Epochs','Train Acc','Train Loss','lr', 'Val Acc', 'Val Loss', 'Test Acc']
        writer.writerow(row)
        for key in data_dic:
            row = [data_dic[key][0][12:],*[data_dic[key][i] for i in range(1,8)] ]
            writer.writerow(row) 



set_Outputpath()
data_dic = get_Modelnames()
data_dic = get_History(data_dic)
data_dic = get_Testmetrics(data_dic)
write_Summary(data_dic)