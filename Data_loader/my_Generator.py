from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import numpy as np


def create_Generator(data_dic,batch_size):

    # create a data generator, fit it, flow (train,val), 
    mod_gen_train = dict()
    mod_gen_val = dict()
    for key in data_dic:
        if '_img_train' in key:
            temp_gen = new_Generator()
            temp_gen.fit(data_dic[key])
            temp_train = temp_gen.flow(data_dic[key],data_dic['_y_train'],batch_size = batch_size, shuffle = False,subset = 'training')
            temp_val = temp_gen.flow(data_dic[key],data_dic['_y_train'],batch_size = batch_size, shuffle = False,subset = 'validation')
            mod_gen_train[key] = temp_train
            mod_gen_val[key.replace('train','val')] = temp_val
    
    return mod_gen_train, mod_gen_val

def new_Generator():
    
    datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=True,
    featurewise_std_normalization=False,
    samplewise_std_normalization=True,
    zca_epsilon= 1e-6,
    rotation_range=15, #degree of random rotations
    width_shift_range=0.2, # float: fraction of total width, if < 1,
    height_shift_range=0.2,# float: fraction of total heing, if < 1,
    brightness_range = [0.8,1.2], # Tuple or list of two floats. Range for picking a brightness shift value from
    shear_range = 0.0, # Float. Shear Intensity (Shear angle in counter-clockwise direction in degrees)
    zoom_range = [0.8, 1.2], # Float or [lower, upper]. Range for random zoom
    vertical_flip=False,
    horizontal_flip=True,
    rescale = 0.5,
    validation_split = 0.2, # Float Fraction of images reserved for validation
    # preprocessing_function = function I guess
    # dataformat
    )

    return datagen

def multistream_Generator(data_dic,batch_size):
    
    mod_gen_train,mod_gen_val  = create_Generator(data_dic,batch_size)         # dictionary for all individual generators: modality, train, validate
    data_gen_train = JoinedGen(mod_gen_train)
    data_gen_val = JoinedGen(mod_gen_val)
    return data_gen_train, data_gen_val

class JoinedGen(Sequence):
    def __init__(self, multi_modal_gen):
        self.gen = multi_modal_gen

    def __len__(self):
        return len(self.gen[list(self.gen)[0]])

    def __getitem__(self, i):
        x_batch = []
        for key in self.gen:
            x, y = self.gen[key][i]
            x_batch.append(x)
        return x_batch, y
        '''
            # modalities = len(list(self.gen))
            # if modalities == 1:
            #     x1,y = self.gen[list(self.gen)[0]][i]
            #     x_batch = x1
            #     return x_batch, y
            # elif modalities == 2:
            #     x1,y = self.gen[list(self.gen)[0]][i]
            #     x2,y = self.gen[list(self.gen)[1]][i]
            #     x_batch = [x1, x2]
            #     return x_batch, y
            # elif modalities == 3:
            #     x1,y = self.gen[list(self.gen)[0]][i]
            #     x2,y = self.gen[list(self.gen)[1]][i]
            #     x3,y = self.gen[list(self.gen)[2]][i]
            #     x_batch = [x1, x2, x3]
            #     return x_batch, y
            # elif modalities == 4:
            #     x1,y = self.gen[list(self.gen)[0]][i]
            #     x2,y = self.gen[list(self.gen)[1]][i]
            #     x3,y = self.gen[list(self.gen)[2]][i]
            #     x4,y = self.gen[list(self.gen)[3]][i]
            #     x_batch = [x1, x2, x3, x4]
            #     return x_batch, y
        '''
    def on_epoch_end(self):
        for key in self.gen:
            self.gen[key].on_epoch_end()

