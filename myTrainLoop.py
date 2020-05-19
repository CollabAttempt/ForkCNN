import csv
import myRun

editparams = ['d1']

for editparam in editparams:
    with open('Run Networks.csv',newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            DataPath, Database, Modalities, Model, Stream, MergeAt, MergeWith, Epochs, Batch, Run, Test = row
            Modalities = list(Modalities.split(','))
            if Run == '1':
                # print(DataPath, Database, Modalities, Model, int(Stream), int(MergeAt), MergeWith, int(Epochs), int(Batch))
                myRun.train_model(DataPath,Database,Modalities,Model,int(Stream),int(MergeAt),MergeWith,int(Epochs),int(Batch), editparam )