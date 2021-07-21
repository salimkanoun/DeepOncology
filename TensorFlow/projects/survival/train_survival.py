import os
from datetime import datetime
import math
import matplotlib.pyplot as plt  
import tensorflow as tf
import tensorflow_addons as tfa 
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard

from data_loader import DataGeneratorSurvival
from ...networks.Vnet_survival import *
from lib.data_manager import DataManagerSurvival
from lib.info_file import create_info_file
from ...loss.Loss import *
from ...loss.Metrics import *

""" 
    For now based on : 
    https://nbviewer.jupyter.org/github/sebp/survival-cnn-estimator/blob/master/tutorial_tf2.ipynb 
    https://github.com/chl8856/DeepHit
    
"""

print("##########################   TENSORFLOW ON GPU   ##########################")
print(tf.__version__) #here version 2.4.1
print("Num GPU available: ", len(tf.config.list_physical_devices('GPU')))
gpu= tf.config.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices([gpu[0]],'GPU') #use only first GPU
#tf.config.experimental.set_memory_growth(gpu[0], True) #limit gpu memory
#tf.debugging.set_log_device_placement(True) #information about GPU and CPU actions
print("###########################################################################")

##########################   FILES PARAMETERS  ##########################
 
base_path='../../FLIP_NIFTI_COMPLET/' # base path of the nifti files 
excel_path='../FLIP_v3.xlsx' # path to the excel file containing survival data +structured data
csv_path='../CSV_FLIP.csv' # path to create a new csv if don't exist or path to read the csv
create_csv = False 

#########################################################################
 
training_model_folder = '../model/' #base path to save info network, trained mode (.h5), ...
now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
training_model_folder = os.path.join(training_model_folder, now)  # '/path/to/folder'

if not os.path.exists(training_model_folder):
    os.makedirs(training_model_folder)
    #if the directory doesn't exist creates it

logdir = os.path.join(training_model_folder, 'logs') #path for logs of the trained model (tensorboard)
if not os.path.exists(logdir):
    os.makedirs(logdir)

##########################   PARAMETERS INPUT DATA   ##########################

mask= False  # is the mask one of the image inputs of the neural network 
survival = mask #do not change
mixed_data =True # include structured data with the scans (age, grade...)
dict_struct_data = {"age":9, "gender": 10, "flipi":11, "arbor": 12, "grade": 13}#columns associated to specific criteria (from excel file given above)
mixed_data_info=["age", "gender", "arbor", "grade", "flipi"  ] #the criterai chosen for the neural network if mixed data is True
mixed_data_columns = [] #automatic filling
for i in mixed_data_info: mixed_data_columns.append(dict_struct_data[i])

epochs = 600
reduced_size= 1 # needed if batch_size to big for GPU (divided images size by 2 => reduced_size == 0.5) => standard size: (256,128,128) (reduced_size == 0)
val_size = 0.20 #0.2 => 20% of the data is for the validation 
last_layer = "softmax" #last laayer of the neural network 

##########################   IMAGE PROCESSING PARAMETERS   ##########################

modalities = ('pet_img', 'ct_img') #input neural network ct and pet image
mode = ['binary', 'probs', 'mean_probs'][0]
method = ['otsu', 'absolute', 'relative', 'otsu_abs'][0]
#[if mode = binary & method = relative : t_val = 0.42
#if mode = binary & method = absolute : t_val = 2.5, 
#else : don't need tval]
tval = ''
target_direction = (1,0,0,0,1,0,0,0,1)
target_size = (128, 128, 256)
target_spacing = (4.0, 4.0, 4.0)
if reduced_size!=0:
    target_size = tuple(round(x*reduced_size) for x in target_size)
    target_spacing = tuple(round(x/reduced_size) for x in target_spacing)


##########################   DATA MANAGER  ##########################

DM = DataManagerSurvival(base_path, excel_path,csv_path, mask, mixed_data, mixed_data_columns)
data = DM.get_data_survival(create_csv, 30) #x: images path, y: (time, event)
if mixed_data:
    train_X_batch_struct, train_X_batch, train_Y_batch, val_X_batch_struct, val_X_batch, val_Y_batch = DM.dataset_survival(create_csv, modalities, survival, mode, method, tval, target_size, target_spacing, target_direction)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X_batch_struct, train_X_batch))
    train_Y_dataset = tf.data.Dataset.from_tensor_slices(train_Y_batch)
    train_dataset = tf.data.Dataset.zip((train_dataset,train_Y_dataset))

    val_dataset = tf.data.Dataset.from_tensor_slices((val_X_batch_struct, val_X_batch))
    val_Y_dataset = tf.data.Dataset.from_tensor_slices(val_Y_batch)
    val_dataset = tf.data.Dataset.zip((val_dataset,val_Y_dataset))
else: 
    train_X_batch, train_Y_batch,  val_X_batch, val_Y_batch = DM.dataset_survival(create_csv, modalities, survival, mode, method, tval, target_size, target_spacing, target_direction)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X_batch, train_Y_batch))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_X_batch, val_Y_batch))

batch_size_train = len(train_Y_batch)
batch_size_val = len(val_Y_batch)

train_dataset = train_dataset.shuffle(100).batch(batch_size_train).cache().repeat()
val_dataset = val_dataset.shuffle(100).batch(batch_size_val).cache().repeat()

##########################   MODEL LOSS METRICS OPTIMIZER  ##########################

time_horizon = int(max(max(train_Y_batch), max(val_Y_batch))*1.2) #number of neurons on the last layer 
np.sum(np.where(np.array(val_Y_batch)<0, 1, 0))
censure_val = np.sum(np.where(np.array(val_Y_batch)<0, 1, 0))/len(val_Y_batch)
print("censure val")
print(censure_val)
#time_horizon = 1

alpha = 0.6  #cross_entropy loss factor
beta = 0.4  # ranking loss factor
gamma = 0 #brier score factor

# definition of loss, optimizer and metrics
optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
loss = get_loss_survival(time_horizon_dim=time_horizon, alpha=alpha, beta=beta, gamma=gamma)
c_index= metric_cindex(time_horizon_dim=time_horizon)
metrics = [c_index] 
#loss = get_loss_cox
#metrics = [concordance_index_censored_cox]


#c_index_weighted= metric_cindex_weighted(time_horizon_dim=time_horizon,batch_size=batch_size, y_val=y_val)
#td_c_index = metric_td_c_index(time_horizon_dim=time_horizon, batch_size=batch_size)

##########################   CALLBACKS PARAMETERS  ##########################
patience = 10 
ReduceLROnPlateau1 = False
EarlyStopping1 = False 
ModelCheckpoint1 = True
TensorBoard1 = True

callbacks = []
if ReduceLROnPlateau1 == True :
    # reduces learning rate if no improvement are seen
    learning_rate_reduction = ReduceLROnPlateau(monitor= loss,
                                                patience=patience ,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.0000001)
    callbacks.append(learning_rate_reduction)

if EarlyStopping1 == True :
    # stop training if no improvements are seen
    early_stop = EarlyStopping(monitor="val_loss",
                                mode="min",
                                patience=int(patience // 2),
                                restore_best_weights=True)
    callbacks.append(early_stop)

if ModelCheckpoint1 == True :
    # saves model weights to file
    # 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
    checkpoint = ModelCheckpoint(os.path.join(training_model_folder, 'model_weights.h5'),
                                    monitor='val_loss',  #td_c_index ?
                                    verbose=1,
                                    save_best_only=True,
                                    mode='min',  # max
                                    save_weights_only=False)
    callbacks.append(checkpoint)

if TensorBoard1 == True :
    tensorboard_callback = TensorBoard(log_dir=logdir,
                                        histogram_freq=0,
                                        update_freq='epoch',
                                        write_graph=True,
                                        write_images=True)
    callbacks.append(tensorboard_callback)

##########################   DEFINITION MODEL  ##########################

dim_mlp= len(mixed_data_columns) #dim input layer structured data neural network 
in_channels= len(modalities)
if mask: 
    in_channels+=1
out_channels= 1
channels_last=True
keep_prob= 1
keep_prob_last_layer= 0.8 
kernel_size= (5, 5, 5)
num_channels= 12

num_levels= 4
num_convolutions= (1, 2, 1)
num_levels = len(num_convolutions)
bottom_convolutions= 1
activation= "relu"
image_shape= (256, 128, 128)
if reduced_size!=0:
    image_shape = tuple(round(x*reduced_size) for x in image_shape)


if mixed_data: 
    architecture = 'vnet_survival_mixed_data'
    model=create_mixed_data_network(dim_mlp,
            image_shape,
            in_channels,
            out_channels,
            time_horizon,
            channels_last,
            keep_prob,
            keep_prob_last_layer,
            kernel_size,
            num_channels,
            num_levels,
            num_convolutions,
            bottom_convolutions,
            last_layer,
            activation)
else: 
    architecture = 'vnet_survival'
    model = VnetSurvival(image_shape,
            in_channels,
            out_channels,
            time_horizon,
            channels_last,
            keep_prob,
            keep_prob_last_layer,
            kernel_size,
            num_channels,
            num_levels,
            num_convolutions,
            bottom_convolutions,
            activation).create_model()


model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

print(model.summary())

model_json = model.to_json()
with open(os.path.join(training_model_folder, 'architecture_{}_model_{}.json'.format(architecture, now)),
            "w") as json_file:
    json_file.write(model_json)
  
##########################   TRAINING MODEL  ##########################

history = model.fit(train_dataset,
                    steps_per_epoch=int(len(train_Y_batch)/batch_size_train),
                    validation_data=val_dataset,
                    validation_steps=int(len(val_Y_batch)/batch_size_val),
                    epochs=epochs,
                    callbacks=callbacks,  # initial_epoch=0,
                    verbose=1
                    )

###########################     RESULTS       ##########################
loss_and_metrics={}
censure_val = []
print(history.history.keys())
print(f'Loss :')
index_loss_val = np.argmin(history.history['val_loss'])

for i in history.history.keys():
    loss_and_metrics[i]=[]
    if history.history[i][index_loss_val] !=-1:
        loss_and_metrics[i].append(history.history[i][index_loss_val])
    print(i+" : "+str(history.history[i][index_loss_val])   )
    
x_axis = np.arange(0, len(history.history['loss']))

fig, (ax1, ax2) = plt.subplots(1,2)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['validation', 'training'])
fig.suptitle('Loss and accuracy plot')
ax1.plot(x_axis, history.history['val_loss'],x_axis, history.history['loss'])
ax2.plot(x_axis, history.history['val_cindex'], x_axis, history.history['cindex'])
#plot when loss and metrics are based on cox 
#ax1.plot(x_axis, history.history['val_loss'],x_axis, history.history['loss'])
#ax2.plot(x_axis, history.history['val_concordance_index_censored_cox'], x_axis, history.history['concordance_index_censored_cox'])
#plt.plot(x_axis, history.history['val_loss'],x_axis, history.history['loss'])
#plt.title('Loss and accuracy model no '+str(fold_no))
#fig.legend(['val_loss', 'loss', 'val_accuracy', 'accuracy'])
for i in history.history.keys():
        if history.history[i][index_loss_val] !=-1:
            loss_and_metrics[i].append(history.history[i][index_loss_val])
        print(i+" : "+str(history.history[i][index_loss_val])   )

censure_val.append(np.sum(np.where(np.array(val_Y_batch)<0, 1, 0))/len(val_Y_batch)*100)
create_info_file(loss_and_metrics, training_model_folder, mask, mixed_data, mixed_data_info, batch_size_train, batch_size_val, epochs, reduced_size, alpha, beta, gamma, censure_val, num_convolutions)
plt.savefig(training_model_folder+'/loss.png')
plt.show()  

##########################   ALTERNATIVE K-FOLD CROSS VALIDATION  ##########################

"""
from sklearn.model_selection import KFold

K=5
loss_and_metrics={}
kfold= KFold(n_splits=K, shuffle=True)
fold_no=1

time_horizon = math.ceil(max(abs(y))*1.2) #number of neurons on the output layer
#time_horizon = 1
max_time = math.ceil(max(abs(y)))
#y = y/max_time
last_layer = "softmax"
patience = 10 
ReduceLROnPlateau1 = True
EarlyStopping1 = False
ModelCheckpoint1 = True
TensorBoard1 = True
for train, test in kfold.split(x, y):

    #generate a print
    print('----------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    alpha=0.4 #cross_entropy loss factor
    beta=0.6  # ranking loss factor
    gamma=0 #brier score factor

    optimizer = tf.keras.optimizers.RMSprop(lr=0.0001)
    #optimizer = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-4)
    loss = get_loss_survival(time_horizon_dim=time_horizon, alpha=alpha, beta=beta, gamma=gamma)
    c_index= metric_cindex(time_horizon_dim=time_horizon)
    #brier_score= get_brier_loss(time_horizon)
    metrics = [c_index] 
    #loss = get_loss_cox
    #metrics = [concordance_index_censored_cox]
    #generate a print
    print('----------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    logdir_fold = logdir+'_fold_no_'+str(fold_no)
    if not os.path.exists(logdir_fold):
        os.makedirs(logdir_fold)

    callbacks = []
    if ReduceLROnPlateau1 == True :
        # reduces learning rate if no improvement are seen
        learning_rate_reduction = ReduceLROnPlateau(monitor= 'loss',
                                                    patience=patience ,
                                                    verbose=1,
                                                    factor=0.5,
                                                    min_lr=0.0000001)
        callbacks.append(learning_rate_reduction)

    if EarlyStopping1 == True :
        # stop training if no improvements are seen
        early_stop = EarlyStopping(monitor="val_loss",
                                    mode="min",
                                    patience=int(patience_ES),
                                    restore_best_weights=True)
        callbacks.append(early_stop)

    if ModelCheckpoint1 == True :
        # saves model weights to file
        # 'model_weights.{epoch:02d}-{val_loss:.2f}.hdf5'
        weights_file='model_weights_'+str(fold_no)+'.h5'
        checkpoint = ModelCheckpoint(os.path.join(training_model_folder, weights_file),
                                        monitor='val_loss',  #td_c_index ?
                                        verbose=1,
                                        save_best_only=True,
                                        mode='min',  # max
                                        save_weights_only=False)
        callbacks.append(checkpoint)

    if TensorBoard1 == True :
        tensorboard_callback = TensorBoard(log_dir=logdir_fold,
                                            histogram_freq=0,
                                            update_freq='epoch',
                                            write_graph=True,
                                            write_images=True)
        callbacks.append(tensorboard_callback)

    #c_index_weighted= metric_cindex_weighted(time_horizon_dim=time_horizon,batch_size=batch_size, y_val=y_val)
    #td_c_index = metric_td_c_index(time_horizon_dim=time_horizon, batch_size=batch_size)
    #IMAGE PROCESSING
    train_transforms = get_transform('train', modalities, survival, mode, method, tval, target_size, target_spacing, target_direction, None, data_augmentation = True, from_pp=False, cache_pp=False)
    val_transforms = get_transform('val', modalities, survival, mode, method, tval, target_size, target_spacing, target_direction, None,  data_augmentation = False, from_pp=False, cache_pp=False)

    x_train= [x[element] for element in train]
    x_test= [x[element] for element in test]
    y_train=[y[element] for element in train]
    y_test=[y[element] for element in test]
    
    censure_val.append(np.sum(np.where(np.array(y_test)<0, 1, 0))/len(y_test)*100)
    print("Nombre patients non censurés fold no ", fold_no, " training : ", np.sum(np.where(np.array(y_train)<0, 0, 1)))
    print("Censure fold no ", fold_no, " training :", np.sum(np.where(np.array(y_train)<0, 1, 0))/len(y_train), "%")
    print("Censure fold no ", fold_no, " validation :", np.sum(np.where(np.array(y_test)<0, 0, 1)))
    print("Censure fold no ", fold_no, " validation :", np.sum(np.where(np.array(y_test)<0, 1, 0))/len(y_test), "%")
    if mixed_data: 
        train_struct_data = [struct_data[element] for element in train]
        val_struct_data = [struct_data[element] for element in test]

    #print(x_train)
    train_images_paths_x, val_images_paths_x = DM.get_images_paths_train_val(x_train,x_test)
    #DATA GENERATOR
    batch_size_train = len(x_train)
    batch_size_val = len(x_test)
    train_X_batch=[]
    train_Y_batch = []
    val_X_batch = []
    val_Y_batch = []
    train_X_batch_struct = []
    val_X_batch_struct = []

    for i in range(len(train_images_paths_x)):
        train=train_transforms(train_images_paths_x[i])
        train_X_batch.append(train['input'])
        train_Y_batch.append(y_train[i])
        if mixed_data:
            train_X_batch_struct.append(train_struct_data[i])
    print(len(x_val))
    print(len(val_images_paths_x))
    print(len(y_val))
    for i in range(len(val_images_paths_x)):
        val=val_transforms(val_images_paths_x[i])
        val_X_batch.append(val['input'])
        val_Y_batch.append(y_val[i])
        if mixed_data: 
            val_X_batch_struct.append(val_struct_data[i])

    buffer_size= 100

    if mixed_data:
        train_dataset = tf.data.Dataset.from_tensor_slices((train_X_batch_struct, train_X_batch))
        train_Y_dataset = tf.data.Dataset.from_tensor_slices(train_Y_batch)
        train_dataset = tf.data.Dataset.zip((train_dataset,train_Y_dataset))

        val_dataset = tf.data.Dataset.from_tensor_slices((val_X_batch_struct, val_X_batch))
        val_Y_dataset = tf.data.Dataset.from_tensor_slices(val_Y_batch)
        val_dataset = tf.data.Dataset.zip((val_dataset,val_Y_dataset))
    else: 
        train_dataset = tf.data.Dataset.from_tensor_slices((train_X_batch, train_Y_batch))
        val_dataset = tf.data.Dataset.from_tensor_slices((val_X_batch, val_Y_batch))

    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size_train).cache().repeat()
    val_dataset = val_dataset.shuffle(buffer_size).batch(batch_size_val).cache().repeat()
    #fit data to model
    if mixed_data: 
        architecture = 'vnet_survival_mixed_data_simplified'
        model=create_mixed_data_network(dim_mlp,
                image_shape,
                in_channels,
                out_channels,
                time_horizon,
                channels_last,
                keep_prob,
                keep_prob_last_layer,
                kernel_size,
                num_channels,
                num_levels,
                num_convolutions,
                bottom_convolutions,
                last_layer,
                activation)
    else: 
        architecture = 'vnet_survival_simplified'
        model = VnetSurvival(image_shape,
                in_channels,
                out_channels,
                time_horizon,
                channels_last,
                keep_prob,
                keep_prob_last_layer,
                kernel_size,
                num_channels,
                num_levels,
                num_convolutions,
                bottom_convolutions,
                activation).create_model()
   
  
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    history = model.fit(train_dataset,
                    steps_per_epoch=int(len(train_images_paths_x)/batch_size_train),
                    validation_data=val_dataset,
                    validation_steps=int(len(val_images_paths_x)/batch_size_val),
                    epochs=epochs,
                    callbacks=callbacks,  # initial_epoch=0,
                    verbose=1
                    )
        
    print(history.history.keys())
    print(f'Loss for fold {fold_no}:')
    index_loss_val = np.argmin(history.history['val_loss'])

    if fold_no==1:
        for i in history.history.keys():
            loss_and_metrics[i]=[]

    for i in history.history.keys():
        if history.history[i][index_loss_val] !=-1:
            loss_and_metrics[i].append(history.history[i][index_loss_val])
        print(i+" : "+str(history.history[i][index_loss_val])   )
    x_axis = np.arange(0, len(history.history['loss']))


    fig, (ax1, ax2) = plt.subplots(1,2)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation', 'training'])
    fig.suptitle('Loss and accuracy plot')
    ax1.plot(x_axis, history.history['val_loss'],x_axis, history.history['loss'])
    ax2.plot(x_axis, history.history['val_cindex'], x_axis, history.history['cindex'])
    #ax1.plot(x_axis, history.history['val_loss'],x_axis, history.history['loss'])
    #ax2.plot(x_axis, history.history['val_concordance_index_censored_cox'], x_axis, history.history['concordance_index_censored_cox'])
    #plt.plot(x_axis, history.history['val_loss'],x_axis, history.history['loss'])
    #plt.title('Loss and accuracy model no '+str(fold_no))
    #fig.legend(['val_loss', 'loss', 'val_accuracy', 'accuracy'])
    
    plt.savefig(training_model_folder+'/loss_'+str(fold_no)+'.png')
    plt.show()
    fold_no+=1

print('-----------------------------------------------------------------')
print(f'Loss and metrics per fold')
for i in loss_and_metrics.keys():
    print('------------------------------------------------------------------')
    for j in range(len(loss_and_metrics[i])):
        print(f'> {i}: {loss_and_metrics[i][j]}')
    print('------------------------------------------------------------------')  

print(f'Loss and metrics per fold average and meean')
print('------------------------------------------------------------------')
for i in loss_and_metrics.keys():  
    print(f'> {i}: {np.mean(loss_and_metrics[i])}(+- {np.std(loss_and_metrics[i])})')
print('------------------------------------------------------------------')   
for i in censure_val:
    print("censure validation fold  : ", i)
create_info_file(loss_and_metrics, training_model_folder, mask, mixed_data, mixed_data_info, batch_size_train, batch_size_val, epochs, reduced_size, alpha, beta, gamma, censure_val)

"""