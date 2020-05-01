from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, Reshape, Add, Concatenate, multiply, Flatten, Dense
from tensorflow.keras.models import Model


def combine_stream(x_1, x_2, merge):
    if merge == "concatenate":
        return Concatenate()([x_1, x_2])
    if merge == "addition":
        return Add()([x_1, x_2])


def bottom(image_input, bn_axis, bn_eps, name):
    x = Conv2D(
        64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
        name='conv1/7x7_s2_' + name)(image_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn_' + name, epsilon=bn_eps)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = senet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1), name=name)
    x = senet_identity_block(x, 3, [64, 64, 256], stage=2, block=2, name=name)
    x = senet_identity_block(x, 3, [64, 64, 256], stage=2, block=3, name=name)

    x = senet_conv_block(x, 3, [128, 128, 512], stage=3, block=1, name=name)
    return x


def mid(x, name):
    filter3 = K.int_shape(x)[3]
    x = senet_identity_block(x, 3, [128, 128, filter3], stage=3, block=2, name=name)
    x = senet_identity_block(x, 3, [128, 128, filter3], stage=3, block=3, name=name)
    x = senet_identity_block(x, 3, [128, 128, filter3], stage=3, block=4, name=name)
    return x


def midtop(x, name):
    filter3 = K.int_shape(x)[3]
    x = senet_conv_block(x, 3, [256, 256, filter3], stage=4, block=1, name=name)
    x = senet_identity_block(x, 3, [256, 256, filter3], stage=4, block=2, name=name)
    x = senet_identity_block(x, 3, [256, 256, filter3], stage=4, block=3, name=name)
    return x


def top(x, name):
    filter3 = K.int_shape(x)[3]
    x = senet_identity_block(x, 3, [256, 256, filter3], stage=4, block=4, name=name)
    x = senet_identity_block(x, 3, [256, 256, filter3], stage=4, block=5, name=name)
    x = senet_identity_block(x, 3, [256, 256, filter3], stage=4, block=6, name=name)

    x = senet_conv_block(x, 3, [512, 512, filter3], stage=5, block=1, name=name)
    x = senet_identity_block(x, 3, [512, 512, filter3], stage=5, block=2, name=name)
    x = senet_identity_block(x, 3, [512, 512, filter3], stage=5, block=3, name=name)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    return x


def senet_se_block(input_tensor, stage, block, compress_rate=16, bias=False, name=None):
    conv1_down_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_down_" + name
    conv1_up_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_up_" + name

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


def SENET50_vanilla(image_input, bn_axis, bn_eps):
    output = bottom(image_input, bn_axis, bn_eps, 'single_stream')
    output = mid(output, 'single_stream')
    output = midtop(output, 'single_stream')
    output = top(output, 'single_stream')
    return output


def SENET50_two_stream_30(input_image_1, input_image_2, bn_axis, bn_eps, merge_style):
    x_1 = bottom(input_image_1, bn_axis, bn_eps, 'visible_stream')
    x_2 = bottom(input_image_2, bn_axis, bn_eps, 'thermal_stream')
    output = combine_stream(x_1, x_2, merge_style)
    output = mid(output, 'merged')
    output = top(midtop(output, 'merged'), 'merged')
    return output


def SENET50_two_stream_50(input_image_1, input_image_2, bn_axis, bn_eps, merge_style):
    x_1 = mid(bottom(input_image_1, bn_axis, bn_eps, 'visible_stream'), 'visible_stream')
    x_2 = mid(bottom(input_image_2, bn_axis, bn_eps, 'thermal_stream'), 'thermal_stream')
    output = combine_stream(x_1, x_2, merge_style)
    output = top(midtop(output, 'merged'), 'merged')
    return output


def SENET50_two_stream_70(input_image_1, input_image_2, bn_axis, bn_eps, merge_style):
    x_1 = midtop(mid(bottom(input_image_1, bn_axis, bn_eps, 'visible_stream'), 'visible_stream'), 'visible_stream')
    x_2 = midtop(mid(bottom(input_image_2, bn_axis, bn_eps, 'thermal_stream'), 'thermal_stream'), 'thermal_stream')
    output = combine_stream(x_1, x_2, merge_style)
    output = top(output, 'merged')
    return output


def SENET50(input_shape, include_top, input_1_tensor, input_2_tensor, stream, merge_style, merge_point, pooling,
            weights, classes):
    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    bn_eps = 0.0001

    if stream == 1:
        inputs = input_image_1
        output = SENET50_vanilla(input_image_1, bn_axis, bn_eps)
        model_name = 'vanilla_senet50'
    else:
        inputs = [input_image_1, input_image_2]
        if merge_point == 30:
            output = SENET50_two_stream_30(input_image_1, input_image_2, bn_axis, bn_eps, merge_style)
            model_name = 'two_stream_30_senet50'
        if merge_point == 50:
            output = SENET50_two_stream_50(input_image_1, input_image_2, bn_axis, bn_eps, merge_style)
            model_name = 'two_stream_50_senet50'
        if merge_point == 70:
            output = SENET50_two_stream_70(input_image_1, input_image_2, bn_axis, bn_eps, merge_style)
            model_name = 'two_stream_70_senet50'

    output = Flatten()(output)
    output = Dense(classes, activation='softmax', name='classifier')(output)

    # Create model.
    model = Model(inputs, output, name=model_name)
    return model
