import Data_loader.my_Get_DB as getDb
import Data_loader.my_Generator as myGen
from tensorflow.keras import Sequential, layers

from

import os

########### DATA LOADING PARAMETERS ###########
# Data
# [Database: IRIS, I2BVSD, VISTH
# Modalities: Vis, The
data_path = r'/media/mobeen/work/face/ForkCNN-tf2/data'
database = 'IRIS'
modalities = ['Vis', 'The']


########### LOAD DATA ###########
data_dic = getDb.get_data(data_path, database, modalities)
########### DATA AUGMENTATION ###########
batch_size = 32
data_gen = myGen.create_Generator(data_dic, batch_size)

model = Sequential()
model.add(layers.Conv2D(32, 3))
model.add(layers.Conv2D(32, 3))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, 3))
model.add(layers.Conv2D(64, 3))
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dense(29, activation='softmax'))


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1' # GPU selection code
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(data_gen['Vis_img_train'],
          verbose=True, steps_per_epoch=len(data_dic['Vis_img_train']) / batch_size, epochs=50)

# data_dic['Vis_img_train']
# data_dic['The_img_train']

# Network [Archi, Structure, Parameters]
# Training Parameters [...]
# Training Metrics [Loss, Accuracy]

# FUNCTIONS
# Data Augmentation
# Network Fetching
# Network Weights Loading
# Network Training
# check saving history
# Network Saving
# Network Testing
# Saving Network Predictions
# Caluclating Metrics from saved predictions


# todo load data
