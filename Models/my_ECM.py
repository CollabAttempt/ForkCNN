from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, Reshape, Add, Concatenate, multiply, Flatten, Dense, Average
from tensorflow.keras.models import Model
import tensorflow.math as tf_math
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

def senet_se_block(input_tensor, stage, block, compress_rate=16, bias=False, name=None):
    conv1_down_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_down_" + name
    conv1_up_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_up_" + name

    num_channels = int(input_tensor.shape[-1])
    bottle_neck = int(num_channels // compress_rate)

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, num_channels))(se)
    se = Conv2D(bottle_neck, (1, 1), use_bias=bias,
                name=conv1_down_name)(se)
    se = Activation('relu')(se)
    se = Conv2D(num_channels, (1, 1), use_bias=bias,
                name=conv1_up_name)(se)
    se = Activation('sigmoid')(se)

    x = input_tensor
    x = multiply([x, se])
    return x

def senet_conv_block(input_tensor, kernel_size, filters,
                     stage, block, bias=False, strides=(2, 2), name=None):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    bn_eps = 0.0001

    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce_" + name
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase_" + name
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj_" + name
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3_" + name

    x = Conv2D(filters1, (1, 1), use_bias=bias, strides=strides,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn", epsilon=bn_eps)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn", epsilon=bn_eps)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn", epsilon=bn_eps)(x)

    se = senet_se_block(x, stage=stage, block=block, bias=True, name=name)

    shortcut = Conv2D(filters3, (1, 1), use_bias=bias, strides=strides,
                      name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis,
                                  name=conv1_proj_name + "/bn", epsilon=bn_eps)(shortcut)

    m = Add()([se, shortcut])
    m = Activation('relu')(m)
    return m

def senet_identity_block(input_tensor, kernel_size,
                         filters, stage, block, bias=False, name=None):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    bn_eps = 0.0001

    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce" + name
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase" + name
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3_" + name

    x = Conv2D(filters1, (1, 1), use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn", epsilon=bn_eps)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn", epsilon=bn_eps)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn", epsilon=bn_eps)(x)

    se = senet_se_block(x, stage=stage, block=block, bias=True, name=name)

    m = Add()([se, input_tensor])
    m = Activation('relu')(m)

    return m

def create_Model(input_image_1, input_image_2, bn_axis, bn_eps):
    x_1 = embbedding_block(input_image_1, bn_axis, bn_eps, 'emb_stream_1')
    x_2 = embbedding_block(input_image_2, bn_axis, bn_eps, 'emb_stream_2')

    output, latent = combine_stream(x_1, x_2)
    output = clasi_block(output, 'clasify')
    return output, latent

def embbedding_block(image_input, bn_axis, bn_eps, name):
    

    x = Conv2D(64, (7, 7), use_bias=False, strides=(2, 2), padding='same', name='conv1/7x7_s2_' + name)(image_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn_' + name, epsilon=bn_eps)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2,2))(x)

    x = senet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1), name=name)
    x = senet_identity_block(x, 3, [64, 64, 256], stage=2, block=2, name=name)
    x = senet_identity_block(x, 3, [64, 64, 256], stage=2, block=3, name=name)

    x = senet_conv_block(x, 3, [128, 128, 512], stage=3, block=1, name=name)
    
    filter3 = K.int_shape(x)[3]
    x = senet_identity_block(x, 3, [128, 128, filter3], stage=3, block=2, name=name)
    x = senet_identity_block(x, 3, [128, 128, filter3], stage=3, block=3, name=name)
    x = senet_identity_block(x, 3, [128, 128, filter3], stage=3, block=4, name=name)

    return x

def combine_stream(x_1, x_2):
    x1_latent = Flatten(name = 'stream1_latent')(x_1)
    x2_latent = Flatten(name = 'stream2_latent')(x_2)
    latent = tf_math.subtract(x1_latent, x2_latent, name = 'sub')
    latent = tf_math.square(latent)
    latent = tf_math.reduce_sum(latent)
    latent = tf_math.sqrt(latent)    
    # tf.sqrt(tf.reduce_sum(tf.square(w)))
    # latent = [x1_latent, x2_latent]
    x = Average()([x_1, x_2])
    return x, latent


def clasi_block(x, name):
    filter3 = K.int_shape(x)[3]
    x = senet_conv_block(x, 3, [256, 256, filter3], stage=4, block=1, name=name)
    x = senet_identity_block(x, 3, [256, 256, filter3], stage=4, block=2, name=name)
    x = senet_identity_block(x, 3, [256, 256, filter3], stage=4, block=3, name=name)

    filter3 = K.int_shape(x)[3]
    x = senet_identity_block(x, 3, [256, 256, filter3], stage=4, block=4, name=name)
    x = senet_identity_block(x, 3, [256, 256, filter3], stage=4, block=5, name=name)
    x = senet_identity_block(x, 3, [256, 256, filter3], stage=4, block=6, name=name)

    x = senet_conv_block(x, 3, [512, 512, filter3], stage=5, block=1, name=name)
    x = senet_identity_block(x, 3, [512, 512, filter3], stage=5, block=2, name=name)
    x = senet_identity_block(x, 3, [512, 512, filter3], stage=5, block=3, name=name)
    size = x.shape[1]
    x = AveragePooling2D((size - 1, size - 1), name='avg_pool')(x)
    return x


def my_ECModel(input_shape, classes):
    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)
    bn_axis = 3
    bn_eps = 0.0001
    model_name = 'myECM'

    inputs = [input_image_1, input_image_2]
    output, latent = create_Model(input_image_1, input_image_2, bn_axis, bn_eps)

    output = Flatten()(output)
    output = Dense(classes, activation='softmax', name='classifier')(output)

    # Create model.
    model = Model(inputs,[output, latent],  name=model_name)
    # model = Model(inputs, latent, name=model_name)
    # print(model.summary())
    return model

# def custom_loss(y_actual,y_pred):
#     # print(y_pred)
#     # print(y_pred.shape)
#     return y_pred


# model = my_ECModel((256, 256, 3), 64)

# losses = {
# 	        'classifier': "categorical_crossentropy",
# 	        'tf_op_layer_Sqrt': custom_loss,
#         }

# metrics = {
# 	        'classifier': 'acc',
# 	        'tf_op_layer_Sqrt': None,
#         }

# model.compile(optimizer= 'SGD', loss = losses, metrics = metrics)


# v_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\SejongDB Data\SejongDB Vis Images.npy')
# t_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\SejongDB Data\SejongDB The Images.npy')
# label =  to_categorical( np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\SejongDB Data\SejongDB Labels.npy') )

# model.fit(x = [v_data,t_data],y = label, batch_size=32, epochs = 50, verbose = 1)

# model.save(r'E:\Work\Multi Modal Face Recognition\Output\Models\My_ECM_SejongDB')


