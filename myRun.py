import Data_loader.my_Get_DB as getDb
import Data_loader.my_Generator as myGen
import Models.Test_Model as test_Model
from tensorflow.keras import Sequential, layers
import os

########### DATA LOADING PARAMETERS ###########
# Data
# [Database: IRIS, I2BVSD, VISTH
# Modalities: Vis, The
data_path = r'E:\Work\Multi Modal Face Recognition\Numpy Data'
database = 'IRIS'
modalities = ['Vis']#, 'The']


########### LOAD DATA ###########
data_dic = getDb.get_data(data_path, database, modalities)

########### DATA AUGMENTATION ###########
batch_size = 32
data_gen_train, data_gen_val  = myGen.multistream_Generator(data_dic,batch_size)

########### GET MODEL ###########
model = test_Model.temp_1stream_model()

########### GET WEIGHTS ###########
#todo

########### MODEL CALLBACKS ###########
#todo 

########### COMPILE MODEL ###########
model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics = ['accuracy'])

########### FIT MODEL ###########
model.fit(data_gen_train, validation_data = data_gen_val, verbose=True, epochs=5)
########### SAVE HISTORY ###########

########### SAVE MODEL ###########


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