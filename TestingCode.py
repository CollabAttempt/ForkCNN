
################### LOADING A MODEL ###################
    # import tensorflow as tf
    #'''
    #model = tf.keras.models.load_model(r'E:\Work\Multi Modal Face Recognition\Code\Main ForkCNN-tf2\ForkCNN-tf2\Output\Models\vgg16_IRIS_1_Vis-_30_concatenate_13-30-31')
    #model.summary()
    #'''

################### LOADING CSV for Model Parameters ###################
    # import csv
    # import myRun

    # with open('Run Networks.csv',newline='') as csvfile:
    #     spamreader = csv.reader(csvfile, delimiter=',')
    #     for row in spamreader:
    #         DataPath, Database, Modalities, Model, Stream, MergeAt, MergeWith, Epochs, Batch, Run = row
    #         Modalities = list(Modalities.split(','))
    #         if Run == '1':
    #             # print(DataPath, Database, Modalities, Model, int(Stream), int(MergeAt), MergeWith, int(Epochs), int(Batch))
    #             myRun.train_model(DataPath,Database,Modalities,Model,int(Stream),int(MergeAt),MergeWith,int(Epochs),int(Batch) )

################### TESTING DATA TYPES FOR NUMPY ARRAYS ###################
    # import numpy as np
    # import cv2
    # import matplotlib.pyplot as plt
    # import matplotlib.image as mpimg

    # iris = np.load(r'E:\Work\Multi Modal Face Recognition\Output\TestData\\'+'I2BVSDVIS_img_test.npy')
    # visth = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data\\'+'IRIS Vis Images.npy')
    # img1 = iris[0]
    # cv2.imshow('IRIS',img1)
    # cv2.waitKey(0)
    # img2 = visth[0]
    # cv2.imshow('VISTH',img2)
    # cv2.waitKey(0)
    # print('Something')

################### LOADING A MODEL AND MODIFYING LAYERS FOR RETRAINING ###################
    # from tensorflow.keras import models, Model
    # from tensorflow.keras.layers import Dense, Activation
    # from tensorflow.keras.utils import to_categorical
    # import numpy as np

    # m_path = r'E:\Work\Multi Modal Face Recognition\Output\Models\202004250436_vgg16_IRIS_2_Vis-The-_30_concatenate'
    # o_model = models.load_model(m_path)
    # o_model.summary()
    # # n_model = Model(o_model.inputs,o_model.layers[-5].output)
    # # n_model.summary()
    # output = Dense(4096, name='fc7')(o_model.layers[-5].output)
    # output = Activation('relu', name='fc7/relu')(output)
    # output = Dense(75, name='fc8')(output)
    # output = Activation('softmax', name='fc8/softmax')(output)
    # n_model = Model(o_model.inputs,output)
    # n_model.compile(optimizer= 'SGD' ,loss='categorical_crossentropy',metrics = 'acc')
    # # print(n_model.get_layer(name = 'conv3_1_merged').get_weights())

    # v_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\I2BVSD Data\I2BVSD Vis Images.npy')
    # t_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\I2BVSD Data\I2BVSD The Images.npy')
    # label =  to_categorical( np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\I2BVSD Data\I2BVSD Labels.npy') )

    # n_model.fit(x = [v_data,t_data],y = label, batch_size=32, epochs = 5, verbose = 1)

################### LOADING A MODEL AND MODIFYING LAYERS To ADD TRIPLET LOSS ###################
    # from tensorflow.keras import models, Model
    # from tensorflow.keras.layers import Dense, Activation, Lambda
    # from tensorflow.keras.utils import to_categorical
    # import numpy as np
    # import tensorflow_addons as tfa
    # from tensorflow.math import l2_normalize
    # import io

    # m_path = r'E:\Work\Multi Modal Face Recognition\Output\Models\202004250436_vgg16_IRIS_2_Vis-The-_30_concatenate'
    # o_model = models.load_model(m_path)
    # output = Dense(4096, name='fc7')(o_model.layers[-5].output)
    # # output = Activation('relu', name='fc7/relu')(output)
    # output = Dense(1024, name='fc8')(output)
    # output = Lambda(lambda x: l2_normalize(x,axis=1))(output)
    # n_model = Model(o_model.inputs,output)
    # n_model.compile(optimizer= 'SGD' , loss = tfa.losses.TripletSemiHardLoss())
    # # print(n_model.get_layer(name = 'conv3_1_merged').get_weights())
    # n_model.summary()
    # v_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data\IRIS Vis Images.npy')
    # t_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data\IRIS The Images.npy')
    # label =  np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data\IRIS Labels.npy')

    # n_model.fit(x = [v_data,t_data],y = label, batch_size=32, epochs = 5, verbose = 1)

    # results = n_model.predict([v_data,t_data])
    # np.savetxt("vecs.tsv", results, delimiter='\t')
    # out_m = io.open('meta.tsv', 'w', encoding='utf-8')
    # for labels in label:
    #     out_m.write(str(labels) + "\n")
    # out_m.close()

################### VISUALIZING FILTERS AND MORE MAYBE ###################
    # from tensorflow.keras import models, Model
    # from tensorflow.keras.utils import to_categorical
    # import numpy as np
    # from matplotlib import pyplot

    # m_path = r'E:\Work\Multi Modal Face Recognition\Output\Models\202004260146_vgg16_IRIS_2_Vis-The-_70_concatenate'
    # o_model = models.load_model(m_path)
    # my_input = o_model.inputs
    # my_output = [o_model.get_layer('conv3_3_visible_stream').output, o_model.get_layer('conv3_3_thermal_stream').output]
    # # my_output = o_model.get_layer('conv4_2_merged').output
    # model = Model(my_input, my_output)

    # # model.summary()

    # v_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data\IRIS Vis Images.npy')
    # t_data = np.load(r'E:\Work\Multi Modal Face Recognition\Numpy Data\IRIS Data\IRIS The Images.npy')

    # img = [np.expand_dims(v_data[1],axis=0), np.expand_dims(t_data[1],axis=0)]

    # featuremaps = model.predict(img)
    # # (1 ,128,128 ,128)
    # # (1 ,128,128 ,128)

    # for fmap in featuremaps:
    #     # plot all 64 maps in an 8x8 squares
    #     # plot all 128 maps in 8x16 squares
    #     ix = 1
    #     for _ in range(16):
    #         for _ in range(16):
    #             # specify subplot and turn of axis
    #             ax = pyplot.subplot(16, 16, ix)
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    #             # plot filter channel in grayscale
    #             pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
    #             ix += 1 
    #             # show the figure
    #     pyplot.show()
    #     # pyplot.savefig('vis.png')

################### my ECM ###################

    # from Models.my_ECM import my_ECModel

    # model = my_ECModel((256, 256, 3), 10)
    # model.summary()




################### Test that the Testing Data is same every time ###################

# import Data_loader.my_Get_DB as getDb
# import Data_loader.my_Generator as myGen
# import matplotlib.pyplot as plt
# import numpy as np

# data_path = r'E:\\Work\\Multi Modal Face Recognition\\Numpy Data'
# database = 'SejongDB'
# modalities = ['Vis', 'The']
# batch_size = 32

# for i in range(5):
#     data_dic = getDb.get_data(data_path,database, modalities)
#     nb_classes = data_dic['_y_train'].shape[1]
#     ########### DATA AUGMENTATION ###########
#     data_gen_train, data_gen_val  = myGen.multistream_Generator(data_dic,batch_size)
#     getDb.save_test_Data(data_dic,database) # Saving test data after being standardized by Datagenerator

#     vis_img = data_dic['Vis_img_test'][0]
#     the_img = data_dic['The_img_test'][0]
#     vis_img_name = data_path + r'\\vis_img' + str(i) + '.jpg'
#     the_img_name = data_path + r'\\the_img' + str(i) + '.jpg'
#     plt.imsave(vis_img_name,vis_img.astype(np.uint8))
#     plt.imsave(the_img_name,the_img.astype(np.uint8))
#     print(i)    
#
# import numpy as np
# ecmpred = np.load(r'E:\Work\Multi Modal Face Recognition\Output\Predictions\myecm_I2BVSD_2_Vis-The-_0_na_1.npy', allow_pickle=True)
# respred = np.load(r'E:\Work\Multi Modal Face Recognition\Output\Predictions\resnet50_I2BVSD_1_Vis-_30_na_1.npy')
#
# print(ecmpred[1].shape)
# print(respred.shape)

from tensorflow.keras import Model, models

path = '/media/mobeen/work/face/ForkCNN/models/myecm_IRIS_2_Vis-The-_0_na_1'
