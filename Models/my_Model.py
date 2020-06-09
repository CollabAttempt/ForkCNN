from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, Reshape, Add, Concatenate, multiply, Flatten, Dense
from tensorflow.keras.models import Model


def combine_stream(x_1, x_2, merge):
    if merge == "concatenate":
        x_1 = multi_filter_block(x_1, 'Concat_stream_1', 0.0001)
        x_2 = multi_filter_block(x_2, 'Concat_stream_2', 0.0001)
        return Concatenate()([x_1, x_2])
    if merge == "addition":
        return Add()([x_1, x_2])
    if merge == 'se_merge':
        return se_merge(x_1, x_2)
    if merge == 'con_se_merge':
        return concat_se_merge(x_1,x_2)

def concat_se_merge(x_1,x_2):
    x = Concatenate()([x_1, x_2])
    x = senet_se_block(x, 'merge', 'concat_sqex', compress_rate=16, bias=False, name='')
    return x


def multi_filter_block(input_img, name, bn_eps):
    filter3 = K.int_shape(input_img)[3]
    x3 = Conv2D(filter3, (3, 3), padding='same', name='MFB_Conv2D_3_1_' + name)(input_img)
    x3 = Conv2D(filter3, (3, 3), padding='same', name='MFB_Conv2D_3_2_' + name)(x3)
    x3 = Conv2D(filter3, (3, 3), padding='same', name='MFB_Conv2D_3_3_' + name)(x3)
    x3 = MaxPooling2D((2, 2), strides=(2, 2), name='MFB_MaxPool_3_' + name)(x3)

    x5 = Conv2D(filter3, (5, 5), padding='same', name='MFB_Conv2D_5_1_' + name)(input_img)
    x5 = Conv2D(filter3, (5, 5), padding='same', name='MFB_Conv2D_5_2_' + name)(x5)
    x5 = MaxPooling2D((2, 2), strides=(2, 2), name='MFB_MaxPool_5_' + name)(x5)

    x7 = Conv2D(filter3, (7, 7), padding='same', name='MFB_Conv2D_7_1_' + name)(input_img)
    x7 = MaxPooling2D((2, 2), strides=(2, 2), name='MFB_MaxPool_7_' + name)(x7)

    x = Concatenate(name='MFB_Concat_kernel_' + name)([x3, x5, x7])
    x = BatchNormalization(axis=3, name='MFB_BN_kernel_' + name, epsilon=bn_eps)(x)
    x = Conv2D(filter3, (1, 1), padding='same', name='MFB_Conv2D_1_1_' + name)(x)

    return x


def bottom(image_input, bn_axis, bn_eps, name):
    # x = multi_filter_block(image_input,name, bn_eps)
    x = Conv2D(64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
               name='conv1/7x7_s2_' + name)(image_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn_' + name, epsilon=bn_eps)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2,
                                      2))(x)

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
    size = x.shape[1]
    x = AveragePooling2D((size - 1, size - 1), name='avg_pool')(x)
    return x


def get_se_ex(input_tensor, stage='merge', block='', compress_rate=16, bias=False):
    conv1_down_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_down"
    conv1_up_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_up"

    num_channels = int(input_tensor.shape[-1])
    bottle_neck = int(num_channels // compress_rate)

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, num_channels))(se)
    se = Conv2D(bottle_neck, (1, 1), use_bias=bias, name=conv1_down_name)(se)
    se = Activation('relu')(se)
    se = Conv2D(num_channels, (1, 1), use_bias=bias, name=conv1_up_name)(se)
    se = Activation('sigmoid')(se)

    x = input_tensor
    return x, se


def se_merge(input_1, input_2):
    x_1, se_1 = get_se_ex(input_1, stage='sqex_merge_1', block='', compress_rate=16, bias=False)
    x_2, se_2 = get_se_ex(input_2, stage='sqex_merge_2', block='', compress_rate=16, bias=False)

    x = Concatenate()([x_1, x_2])
    se = Concatenate()([se_1, se_2])

    out_put = multiply([x, se])
    return out_put


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


def my_Model_two_stream_50(input_image_1, input_image_2, bn_axis, bn_eps, merge_style):
    x_1 = bottom(input_image_1, bn_axis, bn_eps, 'stream_1')
    x_1 = mid(x_1, 'stream_1')
    x_2 = bottom(input_image_1, bn_axis, bn_eps, 'stream_2')
    x_2 = mid(x_2, 'stream_2')

    output = combine_stream(x_1, x_2, merge_style)
    output = midtop(output, 'merged')
    output = top(output, 'merged')
    return output


def SENET50_two_stream_70(input_image_1, input_image_2, bn_axis, bn_eps, merge_style):
    x_1 = midtop(mid(bottom(input_image_1, bn_axis, bn_eps, 'visible_stream'), 'visible_stream'), 'visible_stream')
    x_2 = midtop(mid(bottom(input_image_2, bn_axis, bn_eps, 'thermal_stream'), 'thermal_stream'), 'thermal_stream')
    output = combine_stream(x_1, x_2, merge_style)
    output = top(output, 'merged')
    return output


def my_Model(input_shape, merge_style, merge_point, classes):
    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)
    bn_axis = 3
    bn_eps = 0.0001

    inputs = [input_image_1, input_image_2]
    if merge_point == 30:
        output = SENET50_two_stream_30(input_image_1, input_image_2, bn_axis, bn_eps, merge_style)
        model_name = 'two_stream_30_senet50'
    if merge_point == 50:
        output = my_Model_two_stream_50(input_image_1, input_image_2, bn_axis, bn_eps, merge_style)
        model_name = 'two_stream_50_myModel'
    if merge_point == 70:
        output = SENET50_two_stream_70(input_image_1, input_image_2, bn_axis, bn_eps, merge_style)
        model_name = 'two_stream_70_senet50'

    output = Flatten()(output)
    output = Dense(classes, activation='softmax', name='classifier')(output)

    # Create model.
    model = Model(inputs, output, name=model_name)
    return model

# model = my_Model((256, 256, 3), 'concatenate', 50, 25)

# model.summary()
