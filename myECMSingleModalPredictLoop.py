from tensorflow.keras import models, Model, Sequential, utils, Input
from myUtils.ECM_Classifier_Block import classifier_ECM
import csv, os
import numpy as np
from tensorflow.keras import backend as k
from myUtils.my_Model_Utils import my_embd_loss

filepath = ''
with open('Pathfile.txt', 'r') as myfile:
    filepath = myfile.read()
    filepath = filepath.split("\n")[0]
model_path = os.path.join(filepath, 'Models')
data_path = os.path.join(filepath, 'TestData')


def get_classifier(main_model):
    input_shape = main_model.get_layer(name = 'conv' + str(4) + "_" + str(1) + "_1x1_reduce_" + 'clasify').input.shape
    output_tuple = main_model.outputs[0].shape[1]
    input_tuple = (input_shape[1], input_shape[2], input_shape[3])
    class_model = classifier_ECM(input_tuple,output_tuple)

    c_dict = dict()
    m_dict = dict()
    for i in range(0,len(class_model.layers)):
        c_dict[class_model.layers[i].name] = i

    for i in range(0,len(main_model.layers)):
        m_dict[main_model.layers[i].name] = i

    for key in c_dict:
        if ('multiply' not in key) and ('add' not in key):
            class_model.layers[c_dict[key]].set_weights(main_model.layers[m_dict[key]].get_weights())
    class_model.compile(optimizer='SGD', loss = 'categorical_crossentropy', metrics='acc')
    return class_model
        
def get_embdModel(base_model, stream):## Create embd_model

    # activation_35 for first stream, activation_71 for second stream
    if stream == 0:
        i = 0
        activ_name = 'activation_35'
    elif stream == 1:
        i = 1
        activ_name = 'activation_71'
    else:
        print('Stream Value invalid, got', stream, 'expected 1 or 2')
    
    embd_in = base_model.inputs[i]
    embd_out = base_model.get_layer(name = activ_name)
    embd_model = Model(embd_in,embd_out.output)
    embd_model.compile(optimizer='SGD', loss = my_embd_loss)
    return embd_model

################################ LOADS 1 NEEDED MODEL FROM A PATH, PERFORMS MODEL.PREDICT ################################
def test_model(model_path, model_name):
    print('Loading ',model_path,': ... ')
    if 'myecm' in model_path:
        model = models.load_model(model_path, custom_objects={'my_embd_loss': my_embd_loss} )
    else:
        model = models.load_model(model_path)
    # print('Loaded Model From:',model_path)

    test_data, test_labels = get_TestData(model_name)
    class_model = get_classifier(model)

    mods = model_name.split('_')[3].split('-')[:-1]
    embd_pred = []
    for i in range(len(mods)):
        embd_model = get_embdModel(model,i)
        embd_pred.append(embd_model.predict(test_data[i],batch_size=32,verbose=1))
        test_pred = class_model.predict(embd_pred[i],batch_size=32,verbose=1)
        save_predictions(test_pred,model_path + '_' + mods[i])

    avg_embd_pred = (embd_pred[0] + embd_pred[1])/2

    avg_pred = class_model.predict(avg_embd_pred,batch_size=32,verbose=1)
    save_predictions(avg_pred,model_path + '_embdavg')


    # results = model.evaluate(test_data,test_labels,batch_size= 32, verbose=1)
    # print('Predicting..')
    # test_pred = model.predict(test_data, batch_size= 32, verbose=1)
    # save_predictions(test_pred,model_name,model_path)
    k.clear_session()
    

################################ SAVE PREDICITIONS WITH MODEL NAME TO OUTPUT\PREDICTIONS ################################
def save_predictions(test_pred, model_path):

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
            print( os.path.join(model_path, name) )
    return saved_model_paths


################################ LOAD MODELS TO BE TESTED FROM TRAIN CSV FILE ################################
def gettest_models():
    with open('Run Networks Mobeen.csv', newline='') as csvfile:
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


# model_path = '/media/vip/Program/mobeen/face/Output/Models/myecm_SejongDB_2_Vis-Ir-_0_na_1'
# base_model = models.load_model(model_path, compile= False)
# embd_model = get_embdModel(base_model,1)
# class_model = get_classifier(base_model)

# data = np.load('/media/vip/Program/mobeen/face/Output/TestData/SejongDBVis_img_test.npy')
# label_data = np.load('/media/vip/Program/mobeen/face/Output/TestData/SejongDB_y_test.npy')
# embd_results = embd_model.predict(data,batch_size= 32, verbose=1)
# results = class_model.evaluate(embd_results, label_data, batch_size=32,verbose=1)
