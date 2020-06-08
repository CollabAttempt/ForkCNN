import csv, os
import numpy as np
from tensorflow.keras import backend as k
from tensorflow.keras import models
from myUtils.my_Model_Utils import my_embd_loss

filepath = ''
with open('Pathfile.txt', 'r') as myfile:
    filepath = myfile.read()
    filepath = filepath.split("\n")[0]
model_path = os.path.join(filepath, 'Models')
data_path = os.path.join(filepath, 'TestData')


################################ LOADS 1 NEEDED MODEL FROM A PATH, PERFORMS MODEL.PREDICT ################################
def test_model(model_path, model_name):
    print('Loading ',model_path,': ... ')
    if 'myecm' in model_path:
        model = models.load_model(model_path, custom_objects={'my_embd_loss': my_embd_loss} )
    else:
        model = models.load_model(model_path)
    print('Loaded Model From:',model_path)

    test_data, test_labels = get_TestData(model_name)
    # results = model.evaluate(test_data,test_labels,batch_size= 32, verbose=1)
    print('Predicting..')
    test_pred = model.predict(test_data, batch_size= 32, verbose=1)
    # save_predictions(test_pred,model_name,model_path)
    k.clear_session()
    

################################ SAVE PREDICITIONS WITH MODEL NAME TO OUTPUT\PREDICTIONS ################################
def save_predictions(test_pred, model_name, model_path):

    # # name_str = model_path.split('_',1)[1]
    # name_str = model_path.split('/'or '\\')[-1]
    name_str = model_path + '.npy'
    name_str = os.path.split(name_str)[1]
    print('Saving Predictions as: ', name_str)
    pred_path = os.path.join(filepath, 'Predictions', name_str )
    np.save(pred_path, test_pred)
    print('Saved')
    return None


################################ LOADS SPECIFIC DATABASE NUMPY DATA FOR THE MODEL ################################
def get_TestData(model_name):
    # EXAMPLE: vgg16_IRIS_1_Vis-_30_na_1
    db = model_name.split('_')[1]
    mods = model_name.split('_')[3].split('-')[:-1]
    test_data = []
    print('Loading Data:', db, '  ', mods)
    for mod in mods:
        data_filepath = os.path.join(data_path, db + mod + '_img_test.npy')
        test_data.append(np.load(data_filepath))
    test_labels = np.load(os.path.join(data_path, db + '_y_test.npy'))
    if len(test_data) == 1:
        return test_data[0], test_labels
    else:
        return test_data, test_labels


################################ LOAD ALL MODEL NAMES AVAILABLE FROM OUTPUT DIR ################################
def getall_models_paths():
    saved_model_names = os.listdir(model_path)
    saved_model_paths = []
    for name in saved_model_names:
        if 'myecm' in name:
            saved_model_paths.append(os.path.join(model_path, name))
            # print( os.path.join(model_path, name) )
    return saved_model_paths


################################ LOAD MODELS TO BE TESTED FROM TRAIN CSV FILE ################################
def gettest_models():
    with open('Run Networks.csv', newline='') as csvfile:
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
                model_names.append(Model + u + Database + u + Stream + u + modalities + u + MergeAt + u + MergeWith)
                # print(Model + u + Database + u + Stream + u + modalities + u + MergeAt + u + MergeWith)
    return model_names


################################ RUNNING THE LOOP FOR TESTING ################################
saved_model_paths = getall_models_paths()
model_names = gettest_models()

for model_dir in saved_model_paths:
    for model_name in model_names:
        if model_name in model_dir:
            # print('model_dir:',model_dir)
            # print('model_name:',model_name)
            test_model(model_dir, model_name)
