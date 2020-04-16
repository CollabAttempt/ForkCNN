import Data.my_Get_DB as getDb
import Data.my_Generator as myGen
import Models.Test_Model as test_Model
from tensorflow.keras import Sequential, layers

########### DATA LOADING PARAMETERS ###########
# Data
    # [Database: IRIS, I2BVSD, VISTH
    # Modalities: Vis, The
data_path = r'E:\Work\Multi Modal Face Recognition\Numpy Data'
database = 'IRIS'
modalities = ['Vis', 'The']


########### LOAD DATA ###########
data_dic= getDb.get_data(data_path, database, modalities)
########### DATA AUGMENTATION ###########
batch_size = 32
data_gen = myGen.create_Generator(data_dic,batch_size)
########### GET MODEL ###########
# model = Sequential()
# model.add(layers.Conv2D(64,3))
# model.add(layers.Conv2D(64,3))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(64,3))
# model.add(layers.Conv2D(64,3))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(128,3))
# model.add(layers.Conv2D(128,3))
# model.add(layers.MaxPooling2D())
# model.add(layers.Conv2D(256,3))
# model.add(layers.Conv2D(256,3))
# model.add(layers.MaxPooling2D())
# model.add(layers.Flatten())
# model.add(layers.Dense(1024,activation='relu'))
# model.add(layers.Dense(29,activation='softmax'))

model = test_Model.temp_1stream_model()

########### GET WEIGHTS ###########
#todo

########### MODEL CALLBACKS ###########
#todo 

########### COMPILE MODEL ###########

model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics = ['accuracy'])

########### FIT MODEL ###########
model.fit(data_gen['Vis_img_train'], 
                verbose=True,steps_per_epoch=len(data_dic['Vis_img_train']) / batch_size, epochs=5)

########### SAVE HISTORY ###########

########### SAVE MODEL ###########


# data_dic['Vis_img_train']
# data_dic['The_img_train']

    # Network [Archi, Structure, Parameters]
    # Training Parameters [...]
    # Training Metrics [Loss, Accuracy] 

#FUNCTIONS
    #Data Augmentation
    #Network Fetching
    #Network Weights Loading
    #Network Training
        #check saving history
    #Network Saving
    #Network Testing
    #Saving Network Predictions
    #Caluclating Metrics from saved predictions




