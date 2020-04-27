import Data_loader.my_Get_DB as getDb
import Data_loader.my_Generator as myGen
import Models.Test_Model as test_Model
import Models.get_model as get_model
import myUtils.my_Model_Utils as utils
import tensorflow as tf
import os

########### DATA LOADING PARAMETERS ###########
    # Data
    # [Database: IRIS, I2BVSD, VISTH
    # Modalities: Vis, The
    # data_path = r'E:\Work\Multi Modal Face Recognition\Numpy Data'
    # database = 'IRIS'
    # modalities = ['Vis', 'The']
    # model = 'vgg16'
    # stream = 1
    # merge_point = 30
    # merge_style='concatenate'
    # epochs = 1
    # batch_size = 32

def train_model(data_path,database,modalities,model,stream,merge_point,merge_style,epochs,batch_size, editparam):
    model_name = utils.get_name(database,modalities,model,stream,merge_point,merge_style,editparam)
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # exit(1)

    ########### LOAD DATA ###########
    data_dic = getDb.get_data(data_path, database, modalities)
    nb_classes = data_dic['_y_train'].shape[1]
    ########### DATA AUGMENTATION ###########
    data_gen_train, data_gen_val  = myGen.multistream_Generator(data_dic,batch_size)
    getDb.save_test_Data(data_dic,database) # Saving test data after being standardized by Datagenerator

    ########### DISTRIBUTE ###########  
    mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    with mirrored_strategy.scope():

        ########### MODEL PARAMETERS ###########
        metrics = utils.get_metrics()
        optimizer = utils.get_optimizer()
        callbacks = utils.get_callbacks(model_name)

        ########### GET MODEL ###########
        # model = test_Model.temp_1stream_model()
        his_model = get_model.get_model(include_top=False, model = model, weights = None, stream = stream,
                    input_1_tensor=None, input_2_tensor=None, input_shape=(256,256,3), pooling = None, classes = nb_classes, merge_point = merge_point,
                    merge_style = merge_style)
        
        ########### GET WEIGHTS ###########
        #todo

        ########### COMPILE MODEL ###########
        his_model.compile(optimizer= optimizer,loss='categorical_crossentropy',metrics = metrics) 

        ########### FIT MODEL ###########
        his_model.fit(data_gen_train, validation_data = data_gen_val, verbose=1, epochs=epochs, callbacks = callbacks)
        
        ########### SAVE MODEL ###########
        utils.save_model(model_name,his_model)

    tf.keras.backend.clear_session()
    print('Backend Cleared')

# ########### LOAD MODEL ###########
    # model = tf.keras.models.load_model(r'Output\Models\vgg16_IRIS_1_Vis-_30_concatenate')
    # model.summary()

    # ########### TEST MODEL ###########
    # eval_res = model.evaluate(data_dic['Vis_img_test'],data_dic['_y_test'])
    # print(eval_res)


# Network [Archi, Structure, Parameters]
# Training Parameters [...]
# Training Metrics [Loss, Accuracy]

# FUNCTIONS
# Data Augmentation
# Network Fetching
# Network Weights Loading
# Network Training
# check saving history
# Network Saving
# Network Testing
# Saving Network Predictions
# Caluclating Metrics from saved predictions



# x,y = data_gen_train.next(1)
# for i in range(0,3):
#     image = x[0][i].astype(int)
#     label = y[i]
#     print (label)
#     plt.imshow(image)
#     plt.show()