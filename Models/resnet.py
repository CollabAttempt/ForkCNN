from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Add, Concatenate, multiply, Flatten, Dense, Dropout


from tensorflow.keras.utils import get_file, get_source_inputs
from tensorflow.keras import backend as K
from keras_vggface import utils
import warnings
from tensorflow.keras.models import Model


def combine_stream(x_1, x_2, merge):
    if merge == "concatenate":
        return Concatenate()([x_1, x_2])
    if merge == "addition":
        return Add()([x_1, x_2])


def bottom(image_input, bn_axis, name):
    x = Conv2D(64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
               name='conv1/7x7_s2_' + name)(image_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn_' + name)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1), name=name)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2, name=name)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3, name=name)
    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1, name=name)
    return x


def mid(x, name):
    filter3 = K.int_shape(x)[3]
    x = resnet_identity_block(x, 3, [128, 128, filter3], stage=3, block=2, name=name)
    x = resnet_identity_block(x, 3, [128, 128, filter3], stage=3, block=3, name=name)
    x = resnet_identity_block(x, 3, [128, 128, filter3], stage=3, block=4, name=name)
    return x


def midtop(x, name):
    filter3 = K.int_shape(x)[3]
    x = resnet_conv_block(x, 3, [256, 256, filter3], stage=4, block=1, name=name)
    x = resnet_identity_block(x, 3, [256, 256, filter3], stage=4, block=2, name=name)
    x = resnet_identity_block(x, 3, [256, 256, filter3], stage=4, block=3, name=name)
    return x


def top(x, name):
    filter3 = K.int_shape(x)[3]
    x = resnet_identity_block(x, 3, [256, 256, filter3], stage=4, block=4, name=name)
    x = resnet_identity_block(x, 3, [256, 256, filter3], stage=4, block=5, name=name)
    x = resnet_identity_block(x, 3, [256, 256, filter3], stage=4, block=6, name=name)

    x = resnet_conv_block(x, 3, [512, 512, filter3], stage=5, block=1, name=name)
    x = resnet_identity_block(x, 3, [512, 512, filter3], stage=5, block=2, name=name)
    x = resnet_identity_block(x, 3, [512, 512, filter3], stage=5, block=3, name=name)
    size = x.shape[1]
    x = AveragePooling2D((size-1, size-1), name='avg_pool')(x)
    return x


def resnet_identity_block(input_tensor, kernel_size, filters, stage, block,
                          bias=False, name=None):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce_" + name
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase_" + name
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3_" + name

    x = Conv2D(filters1, (1, 1), use_bias=bias, name=conv1_reduce_name)(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, use_bias=bias,
               padding='same', name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), use_bias=bias, name=conv1_increase_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x


def resnet_conv_block(input_tensor, kernel_size, filters, stage, block,
                      strides=(2, 2), bias=False, name=None):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce_" + name
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase_" + name
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj_" + name
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3_" + name

    x = Conv2D(filters1, (1, 1), strides=strides, use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=bias,
                      name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "/bn")(
        shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def RESNET50_vanilla(input_image_1, bn_axis):
    print("Using single stream ResNet50")
    output = bottom(input_image_1, bn_axis, 'single_stream')
    output = top(midtop(mid(output, 'single_stream'), 'single_stream'), 'single_stream')
    return output


def RESNET50_two_stream_30(input_image_1, input_image_2, bn_axis, merge_style):
    x_1 = bottom(input_image_1, bn_axis, 'visible_stream')
    x_2 = bottom(input_image_2, bn_axis, 'thermal_stream')
    output = combine_stream(x_1, x_2, merge_style)
    output = mid(output, 'merged')
    output = midtop(output, 'merged')
    return top(output, 'merged')
    # return top(midtop(mid(output)))


def RESNET50_two_stream_50(input_image_1, input_image_2, bn_axis, merge_style):
    x_1 = mid(bottom(input_image_1, bn_axis, 'visible_stream'), name='visible_stream')
    x_2 = mid(bottom(input_image_2, bn_axis, 'thermal_stream'), name='thermal_stream')
    output = combine_stream(x_1, x_2, merge_style)
    return top(midtop(output, 'merged'), 'merged')


def RESNET50_two_stream_70(input_image_1, input_image_2, bn_axis, merge_style):
    x_1 = midtop(mid(bottom(input_image_1, bn_axis, 'visible_stream'), name='visible_stream'), name='visible_stream')
    x_2 = midtop(mid(bottom(input_image_2, bn_axis, 'thermal_stream'), name='thermal_stream'), name='thermal_stream')
    output = combine_stream(x_1, x_2, merge_style)
    return top(output, 'merged')


def RESNET50(input_shape, include_top, input_1_tensor, input_2_tensor, stream, merge_style, merge_point, pooling,
             weights, classes):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)
    if stream == 1:
        inputs = input_image_1
        output = RESNET50_vanilla(input_image_1, bn_axis)

    if stream == 2:
        inputs = [input_image_1, input_image_2]
        if merge_point == 30:
            output = RESNET50_two_stream_30(input_image_1, input_image_2, bn_axis, merge_style)

        if merge_point == 50:
            output = RESNET50_two_stream_50(input_image_1, input_image_2, bn_axis, merge_style)

        if merge_point == 70:
            output = RESNET50_two_stream_70(input_image_1, input_image_2, bn_axis, merge_style)

    output = Flatten()(output)
    output = Dense(classes, activation='softmax', name='classifier')(output)


    # Create model.
    model = Model(inputs, output, name='vggface_resnet50')
    return model
