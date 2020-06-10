from tensorflow.keras import models, Model, Sequential, utils, Input
from myUtils.ECM_Classifier_Block import classifier_ECM
# from myUtils.my_Model_Utils import my_embd_loss

def my_embd_loss(y_actual,y_pred):
    return y_pred

def extract_classifier(main_model, starting_layer_ix, ending_layer_ix):
    # create an empty model
    input_shape = Input(shape=(32,32,512))
    x = main_model.layers[starting_layer_ix](input_shape)
    for ix in range(starting_layer_ix+1, ending_layer_ix ):
        print(main_model.layers[ix].name)
        x = main_model.layers[ix](x)
        # copy this layer over to the new model
    new_model = Model(input_shape,x)
    return new_model

def create_classifier(main_model):
    input_shape = main_model.get_layer(name = 'conv' + str(4) + "_" + str(1) + "_1x1_reduce_" + 'clasify').input.shape
    output_tuple = main_model.outputs[0].shape[1]
    input_tuple = (input_shape[1], input_shape[2], input_shape[3])
    class_model = classifier_ECM(input_shape,output_tuple)
    # class_model.compile(optimizer='SGD', loss='categorical_crossentropy')
    c_dict = dict()
    m_dict = dict()

    for i in range(0,len(class_model.layers)):
        # print('class_model:' , i , class_model.layers[i].name)
        c_dict[class_model.layers[i].name] = i

    for i in range(0,len(main_model.layers)):
        # print('main_model:' , i , main_model.layers[i].name)
        m_dict[main_model.layers[i].name] = i

    for key in c_dict:
        # class_model.layers[c_dict[key]].weights = main_model.layers[m_dict[key]].weights
        # if key not in ['input_1', 'conv4_1_1x1_reduce_clasify']:
        # blacklist = ['multiply', 'add']
        # for x in blacklist:
        #     if x not in key:
        if ('multiply' not in key) and ('add' not in key):
            # print('Class: ', key, class_model.layers[c_dict[key]].name)
            # print('input_c: ', key,  class_model.layers[c_dict[key]].input.shape)
            # print('input_m: ', key, main_model.layers[m_dict[key]].input.shape)
            #
            # print('Main: ', key, main_model.layers[m_dict[key]].name)
            # print('output_c: ', key, class_model.layers[c_dict[key]].output.shape)
            # print('output_m: ', key, main_model.layers[m_dict[key]].output.shape)
            # print(' ')
            class_model.layers[c_dict[key]].set_weights(main_model.layers[m_dict[key]].get_weights())

    return class_model
            

model_path = '/media/vip/Program/mobeen/face/Output/Models/myecm_SejongDB_2_Vis-VisIr-_0_na_2'
base_model = models.load_model(model_path, compile= False)
print(base_model.summary())


## get average + 1 layer data
name = 'conv' + str(4) + "_" + str(1) + "_1x1_reduce_" + 'clasify'
for i in range(len(base_model.layers)):
    if name == base_model.layers[i].name:
        print('First Layer:', base_model.layers[i].name, ' ', i)


## Create embd_model
embd_in = base_model.inputs[0]
embd_out = base_model.get_layer(name = 'activation_35')
embd_model = Model(embd_in,embd_out.output)
embd_model.compile(optimizer='SGD', loss = my_embd_loss)
print('EMBD inputs', embd_model.inputs)
print('EMBD outputs',embd_model.outputs)


class_model = create_classifier(base_model)
class_model.compile(optimizer='SGD', loss = my_embd_loss)
print('Class inputs', class_model.inputs)
print('Class outputs', class_model.outputs)


# # utils.plot_model(class_model, 'Class_model.png', show_shapes=True)
# class_name = 'conv' + str(4) + "_" + str(1) + "_1x1_reduce_" + 'clasify'
# class_in = base_model.get_layer(name = class_name)    
# class_out = base_outputs[0]
# class_model = Model(class_in.input,class_out)
# class_model.compile(optimizer='SGD', loss = my_embd_loss)
