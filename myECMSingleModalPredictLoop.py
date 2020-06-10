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
    input_shape = main_model.get_layer(name = 'conv' + str(4) + "_" + str(1) + "_1x1_reduce_" + 'clasify').output.shape
    output_tuple = main_model.outputs[0].shape[1]
    input_tuple = (input_shape[1], input_shape[2], input_shape[3])
    class_model = classifier_ECM(input_tuple,output_tuple)
    
    for c_layer in class_model.layers:
        for m_layer in main_model.layers:
            c_name = c_layer.name
            if m_layer.name == c_name:
                print(c_name)

            

model_path = r'E:\myecm_I2BVSD_2_Vis-The-_0_na_3'
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
# class_model.compile(optimizer='SGD', loss = my_embd_loss)
# print('Class inputs', class_model.inputs)
# print('Class outputs', class_model.outputs)


# # utils.plot_model(class_model, 'Class_model.png', show_shapes=True)
# class_name = 'conv' + str(4) + "_" + str(1) + "_1x1_reduce_" + 'clasify'
# class_in = base_model.get_layer(name = class_name)    
# class_out = base_outputs[0]
# class_model = Model(class_in.input,class_out)
# class_model.compile(optimizer='SGD', loss = my_embd_loss)
