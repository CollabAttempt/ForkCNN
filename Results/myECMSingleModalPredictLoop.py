from tensorflow.keras import models, Model, Sequential, utils, Input
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



model_path = r'E:\Work\Multi Modal Face Recognition\Output\Models\myecm_SejongDB_2_Vis-VisIr-_0_na_3'
base_model = models.load_model(model_path, custom_objects={'my_embd_loss': my_embd_loss}, compile= False )
print(base_model.summary())
base_inputs = base_model.inputs
base_outputs = base_model.outputs
utils.plot_model(base_model, 'Base_model.png', show_shapes=True)
for i in range(len(base_model.layers)):
    name = 'conv' + str(4) + "_" + str(1) + "_1x1_reduce_" + 'clasify'
    if name == base_model.layers[i].name:
        print('average:', base_model.layers[i].name, ' ', i)
        print(base_model.layers[i].input)

baseavg_layer = base_model.get_layer(name='average')
print(baseavg_layer.input)
print(baseavg_layer.output)

embd_in = base_model.inputs[0]
embd_out = base_model.get_layer(name = 'activation_35')
embd_model = Model(embd_in,embd_out.output)
embd_model.compile(optimizer='SGD', loss = my_embd_loss)
print('EMBD inputs', embd_model.inputs)
print('EMBD outputs',embd_model.outputs)

class_model = extract_classifier(base_model, 257, len(base_model.layers))
class_model.compile(optimizer='SGD', loss = my_embd_loss)
# utils.plot_model(class_model, 'Class_model.png', show_shapes=True)
print('Class inputs', class_model.inputs)
print('Class outputs', class_model.outputs)

# class_name = 'conv' + str(4) + "_" + str(1) + "_1x1_reduce_" + 'clasify'
# class_in = base_model.get_layer(name = class_name)    
# class_out = base_outputs[0]
# class_model = Model(class_in.input,class_out)
# class_model.compile(optimizer='SGD', loss = my_embd_loss)


# for m_output in m_outputs:
#     print(m_output)
#     print(m_output.shape) 
