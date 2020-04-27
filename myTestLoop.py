import csv, os
import numpy as np
from tensorflow.keras import models

model_path = r'E:\Work\Multi Modal Face Recognition\Output\Models\\'
data_path = r'E:\Work\Multi Modal Face Recognition\Code\Main ForkCNN-tf2\ForkCNN-tf2\Output\TestData'

################################ LOADS 1 NEEDED MODEL FROM A PATH, PERFORMS MODEL.PREDICT ################################
def test_model(model_path,model_name):
    model = models.load_model(model_path)
    test_data, test_labels = get_TestData(model_name)
    print('model_name',model_name)
    results = model.evaluate(test_data,test_labels,batch_size= 32, verbose=1)
    test_pred = model.predict(test_data, batch_size= 32, verbose=0)
    save_predictions(test_labels,test_pred,model_name)

################################ SAVE PREDICITIONS WITH MODEL NAME TO OUTPUT\PREDICTIONS ################################
def save_predictions(test_labels,test_pred,model_name):
    pred_name = r'E:\\Work\\Multi Modal Face Recognition\\Output\\Predictions\\' + model_name + '.npy'
    np.save(pred_name,test_pred)
    return None

################################ LOADS SPECIFIC DATABASE NUMPY DATA FOR THE MODEL ################################
def get_TestData(model_name):
    #EXAMPLE: _vgg16_IRIS_1_Vis-_30_na
    db = model_name.split('_')[2]
    mods = model_name.split('_')[4].split('-')[:-1]
    test_data = []
    for mod in mods:
        data_filepath = os.path.join(data_path,db+mod+'_img_test.npy')
        test_data.append(np.load(data_filepath))
    test_labels = np.load(os.path.join(data_path,db+'_y_test.npy'))
    if len(test_data) ==1:
        return test_data[0], test_labels
    else:
        return test_data, test_labels

################################ LOAD ALL MODEL NAMES AVAILABLE FROM OUTPUT DIR ################################
def getall_models_paths():
    saved_model_names = os.listdir(model_path)
    saved_model_paths = [model_path + names for names in saved_model_names]
    return saved_model_paths

################################ LOAD MODELS TO BE TESTED FROM TRAIN CSV FILE ################################
def gettest_models():
    with open('Run Networks.csv',newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        model_names = []
        for row in spamreader:
            DataPath, Database, Modalities, Model, Stream, MergeAt, MergeWith, Epochs, Batch, Run, Test = row
            
            Modalities = list(Modalities.split(','))
            # print(Modalities)
            modalities = ''
            for modality in Modalities:
                modalities = modalities + modality + '-'
            # print(modalities)
            if Test == '1':
                u = '_'
                model_names.append(u+ Model +u+ Database +u+ Stream +u+ modalities +u+ MergeAt +u+ MergeWith)
    return model_names


################################ RUNNING THE LOOP FOR TESTING ################################
saved_model_paths = getall_models_paths()
model_names = gettest_models()

for model_dir in saved_model_paths:
    for model_name in model_names:
        if model_name in model_dir:
            # print('model_dir',model_dir)
            # print('model_name',model_name)
            test_model(model_dir,model_name)
