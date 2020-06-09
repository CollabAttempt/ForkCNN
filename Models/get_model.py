from __future__ import print_function
from Models.vgg import VGG16
from Models.resnet import RESNET50
from Models.senet import SENET50
from Models.my_Model import my_Model
from Models.my_ECM import my_ECModel
from Models.my_Model_A import my_Model_A
from Models.my_Model_B import my_Model_B
from Models.my_Model_C import my_Model_C



def get_model(include_top=False, model='vgg16', weights=None, stream=1,
                input_1_tensor=None, input_2_tensor=None, input_shape=None, pooling=None, classes=None, merge_point=None,
                merge_style=None):

    if model == 'vgg16':
        return VGG16(include_top=include_top, input_1_tensor=input_1_tensor, input_2_tensor=input_2_tensor,
                        stream=stream, merge_style=merge_style, merge_point=merge_point, input_shape=input_shape,
                        pooling=pooling, weights=weights, classes=classes)

    if model == 'resnet50':
        return RESNET50(include_top=include_top, input_1_tensor=input_1_tensor, input_2_tensor=input_2_tensor,
                        input_shape=input_shape, pooling=pooling, stream=stream, merge_style=merge_style,
                        merge_point=merge_point, weights=weights, classes=classes)

    if model == 'senet50':
        return SENET50(include_top=include_top, input_1_tensor=input_1_tensor, input_2_tensor=input_2_tensor,
                        input_shape=input_shape, pooling=pooling, stream=stream, merge_style=merge_style,
                        merge_point=merge_point, weights=weights, classes=classes)
    
    if model == 'myModel':
        return my_Model(input_shape = input_shape, merge_style = merge_style, merge_point = merge_point, 
                        classes = classes)

    if model == 'myecm':
        return my_ECModel(input_shape= input_shape, classes= classes)

    if model == 'modelA':
        return my_Model_A(input_shape, merge_style, merge_point, classes)
    
    if model == 'modelB':
        return my_Model_B(input_shape, merge_style, merge_point, classes)

    if model == 'modelC':
        return my_Model_C(input_shape, merge_style, merge_point, classes)