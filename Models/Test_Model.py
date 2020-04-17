from tensorflow.keras import Sequential, layers
import tensorflow as tf



def temp_streams(input,activation):
    next_input = layers.Conv2D(64, kernel_size=(3,3), activation=activation, padding='same')(input)
    next_input = layers.Conv2D(64, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

    next_input = layers.Conv2D(64, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = layers.Conv2D(64, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

    return next_input

def temp_2stream_model():
    input_shape1 = tf.keras.Input(shape=(256,256,3))
    input_shape2 = tf.keras.Input(shape=(256,256,3))
    activation = 'relu'


    thermal_output = temp_streams(input_shape1,activation)
    visible_output = temp_streams(input_shape2,activation)

    next_input = layers.Add()([thermal_output,visible_output])

    next_input = layers.Conv2D(128, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = layers.Conv2D(128, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

    next_input = layers.Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = layers.Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

    output = layers.Flatten()(next_input)
    output = layers.Dense(4096, activation=activation)(output)
    output = layers.Dropout(0.5)(output)
    output = layers.Dense(29, activation='softmax')(output)

    model = tf.keras.Model(inputs = [input_shape1,input_shape2], outputs=[output])

    return model

def temp_1stream_model():
    input_shape1 = tf.keras.Input(shape=(256,256,3))
    activation = 'relu'

    next_input = temp_streams(input_shape1,activation)

    next_input = layers.Conv2D(128, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = layers.Conv2D(128, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

    next_input = layers.Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = layers.Conv2D(512, kernel_size=(3,3), activation=activation, padding='same')(next_input)
    next_input = layers.MaxPooling2D(pool_size=(2,2), padding="same", strides=(2,2))(next_input)

    output = layers.Flatten()(next_input)
    output = layers.Dense(4096, activation=activation)(output)
    output = layers.Dropout(0.5)(output)
    output = layers.Dense(29, activation='softmax')(output)

    model = tf.keras.Model(inputs = input_shape1, outputs=[output])

    return model