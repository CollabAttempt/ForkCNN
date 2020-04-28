
################### LOADING A MODEL ###################
    # import tensorflow as tf
    #'''
    #model = tf.keras.models.load_model(r'E:\Work\Multi Modal Face Recognition\Code\Main ForkCNN-tf2\ForkCNN-tf2\Output\Models\vgg16_IRIS_1_Vis-_30_concatenate_13-30-31')
    #model.summary()
    #'''

################### LOADING CSV for Model Parameters ###################
    # import csv
    # import myRun

    # with open('Run Networks.csv',newline='') as csvfile:
    #     spamreader = csv.reader(csvfile, delimiter=',')
    #     for row in spamreader:
    #         DataPath, Database, Modalities, Model, Stream, MergeAt, MergeWith, Epochs, Batch, Run = row
    #         Modalities = list(Modalities.split(','))
    #         if Run == '1':
    #             # print(DataPath, Database, Modalities, Model, int(Stream), int(MergeAt), MergeWith, int(Epochs), int(Batch))
    #             myRun.train_model(DataPath,Database,Modalities,Model,int(Stream),int(MergeAt),MergeWith,int(Epochs),int(Batch) )

################### TESTING DATA TYPES FOR NUMPY ARRAYS ###################
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

iris = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data\\'+'IRIS The Images.npy')
visth = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\VISTH Data\\'+'VISTH Labels.npy')
img1 = iris[0]
cv2.imshow('IRIS',img1)
cv2.waitKey(0)
img2 = visth[0]
cv2.imshow('VISTH',img2)
cv2.waitKey(0)
print('Something')

