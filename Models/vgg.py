from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, \
    GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Add, Concatenate, multiply, Flatten, Dense, Dropout, Lambda

from tensorflow.keras.utils import get_file, get_source_inputs
from tensorflow.keras import backend as K
from keras_vggface import utils
import warnings
from tensorflow.keras.models import Model
# import tensorflow as tf

def bottom(img_input, name):  # 0.3077
    # Block 1
    # x = Lambda(show_output)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1_' + name)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2_' + name)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_' + name)(x)
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1_' + name)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2_' + name)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2_' + name)(x)
    return x


def mid(x, name):  # 0.1538 %     Approx. 50% of network (45% :D)
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1_' + name)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2_' + name)(x)
    return x


def midtop(x, name):  # 0.1538     Approx. 65% of network
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3_' + name)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3_' + name)(x)
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1_' + name)(x)
    return x


def top(x, name):  # 0.3846     100%
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2_' + name)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3_' + name)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4_' + name)(x)
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)
    z = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(x)
    return z


def VGG16_vanilla(input_image_1):
    # img_input = Input(shape=input_shape)
    # Normal VGG (single stream)
    print("Using single stream VGG16")
    output = top(midtop(mid(bottom(input_image_1, 'visible_stream'), 'visible_stream'), 'visible_stream'),
                 'visible_stream')
    return output


def VGG16_two_stream_30(input_image_1, input_image_2, merge):
    x_1 = bottom(input_image_1, 'visible_stream')
    x_2 = bottom(input_image_2, 'thermal_stream')
    x = combine_stream(x_1, x_2, merge)
    print("Using two streams (VGG16) with merging strings after 30% of conv layers")

    # Network is single streamed
    return top(midtop(mid(x, 'merged'), 'merged'), 'merged')


def VGG16_two_stream_50(input_image_1, input_image_2, merge):
    x_1 = mid(bottom(input_image_1, 'visible_stream'), 'visible_stream')
    x_2 = mid(bottom(input_image_2, 'thermal_stream'), 'thermal_stream')
    x = combine_stream(x_1, x_2, merge)
    print("Using two streams (VGG16) with merging strings after 50% of conv layers")

    # Network is single streamed
    return top(midtop(x, 'merged'), 'merged')


def VGG16_two_stream_70(input_image_1, input_image_2, merge):
    x_1 = midtop(mid(bottom(input_image_1, 'visible_stream'), 'visible_stream'), 'visible_stream')
    x_2 = midtop(mid(bottom(input_image_2, 'thermal_stream'), 'thermal_stream'), 'thermal_stream')
    x = combine_stream(x_1, x_2, merge)
    # Network is single streamed
    print("Using two streams (VGG16) with merging strings after 70% of conv layers")
    return top(x, 'merged')


def combine_stream(x_1, x_2, merge):
    if merge == "concatenate":
        return Concatenate()([x_1, x_2])
    elif merge == "addition":
        return Add()([x_1, x_2])
    else:
        print("NO MERGE STYLE GIVEN")
        exit(1)

def VGG16(input_shape, include_top, input_1_tensor, input_2_tensor, stream, merge_style,
          merge_point, pooling, weights, classes):
    input_image_1 = Input(shape=input_shape)
    input_image_2 = Input(shape=input_shape)

    if stream == 1:
        inputs = [input_image_1]
        output = VGG16_vanilla(input_image_1)
    if stream == 2:
        inputs = [input_image_1, input_image_2]

        if merge_point == 30:
            output = VGG16_two_stream_30(input_image_1, input_image_2, merge_style)
        elif merge_point == 50:
            output = VGG16_two_stream_50(input_image_1, input_image_2, merge_style)
        elif merge_point == 70:
            output = VGG16_two_stream_70(input_image_1, input_image_2, merge_style)
        else:
            print("DEFINE A MERGE POINT FOR MULTI STREAM")
            exit(1)

    output = Flatten(name='flatten')(output)
    output = Dense(4096, name='fc6')(output)
    output = Activation('relu', name='fc6/relu')(output)
    # output = Dropout(0.5)(output)
    output = Dense(4096, name='fc7')(output)
    output = Activation('relu', name='fc7/relu')(output)
    
    output = Dense(classes,name='fc8')(output)
    output = Activation('softmax', name='fc8/softmax')(output)
    
    model = Model(inputs, output, name='vggface_vgg16_2stream')
    return model

def show_output(x):
    # x_array = tf.make_ndarray(x) 
    tf.print(x.shape)
    tf.print(x.dtype)
    # tf.print(tf.math.reduce_sum(x))
    # print()
    input("Batch Done")
    return x