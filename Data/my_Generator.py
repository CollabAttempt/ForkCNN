from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_Generator(data_dic,batchsize):
    
    my_gen_dic = dict() 
    for key in data_dic:
        if '_img_train' in key:
            tem_gen = new_Generator()
            tem_gen.fit(data_dic[key])
            temflow = tem_gen.flow(data_dic[key],data_dic['_y_train'],batch_size = batchsize, shuffle = True)
            my_gen_dic[key] = temflow
    
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    
    # datagen.fit(x_train)


    # fits the model on batches with real-time data augmentation:
    # dataflow = datagen.flow(x_train, y_train, batch_size=32),
                        # steps_per_epoch=len(x_train) / 32, epochs=epochs)

    return my_gen_dic

def new_Generator():
    
    datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=True,
    featurewise_std_normalization=False,
    samplewise_std_normalization=True,
    zca_epsilon= 1e-6,
    rotation_range=20, #degree of random rotations
    width_shift_range=0.2, # float: fraction of total width, if < 1,
    height_shift_range=0.2,# float: fraction of total heing, if < 1,
    brightness_range = [1.0,1.0], # Tuple or list of two floats. Range for picking a brightness shift value from
    shear_range = 0.0, # Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
    zoom_range = [1.0, 1.0], # Float or [lower, upper]. Range for random zoom
    vertical_flip=True,
    horizontal_flip=True,
    rescale = 0,
    validation_split = 0.2, # Float Fraction of images reserved for validation
    # preprocessing_function = function I guess
    # dataformat
    )

    return datagen
