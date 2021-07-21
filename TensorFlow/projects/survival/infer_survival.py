import tensorflow as tf
from lib.DataManagerSurvival import DataManagerSurvival

##########################   FILES PARAMETERS  ##########################

trained_model_path = '../model/20210610-13:10:39/model_weights.h5'#If None, train from scratch 
base_path='../../FLIP_NIFTI_COMPLET/' # base path of the nifti files 
excel_path='../FLIP_v3.xlsx' # path to the excel file containing survival data +structured data
csv_path='../CSV_FLIP.csv' # path to create a new csv if don't exist or path to read the csv
create_csv = False 

##########################   CHOICES INPUT DATA   ##########################

mask= True  # is the mask one of the image inputs of the neural network 
survival = [True, mask] #do not change
dict_struct_data = {"age":9, "gender": 10, "flipi":11, "arbor": 12, "grade": 13}#column associated to specific criteria (from excel file given above)
mixed_data =True# include structured data with the scans (age, grade...)
mixed_data_info=["age", "gender", "grade"]
mixed_data_columns = [] #automatic filling
for i in mixed_data_info: mixed_data_columns.append(dict_struct_data[i])
shuffle = True 
reduced_size= 0.5 # needed if batch_size to big for GPU (divided images size by 2 => reduced_size == 0.5) => standard size: (256,128,128) (reduced_size == 0)
val_size = 0 #0.2 => 20% of the data is for the validation 

##########################   IMAGE PROCESSING PARAMETERS   ##########################

modalities = ('pet_img', 'ct_img') #input neural network ct and pet image
mode = ['binary', 'probs', 'mean_probs'][0]
method = ['otsu', 'absolute', 'relative', 'otsu_abs'][0]
tval = ''
target_direction = (1,0,0,0,1,0,0,0,1)
target_size = (128, 128, 256)
target_spacing = (4.0, 4.0, 4.0)
if reduced_size!=0:
    target_size = tuple(round(x*reduced_size) for x in target_size)
    target_spacing = tuple(round(x/reduced_size) for x in target_spacing)

##########################   DATA MANAGER  ##########################

DM = DataManagerSurvival(base_path, excel_path,csv_path, mask, mixed_data, mixed_data_columns)
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

train_dataset = train_dataset.shuffle(100).batch(batch_size_train).cache().repeat()

time_horizon = max(max(train_Y_batch), max(val_Y_batch))*1.2
i=10
test_X_batch = [train_X_batch[i]]
test_X_batch_struct = [train_X_batch_struct[i]]
test_Y_batch = [train_Y_batch[i]]
batch_size_test = len(test_Y_batch)
#######################      LOAD MODEL       ##########################

model= tf.keras.models.model_from_json(open('./../model/20210610-09:48:32/architecture_vnet_survival_mixed_data_simplified_model_20210610-09:48:32.json').read())
model.load_weights(trained_model_path)
print(model.summary())

#######################      PREDICTION       ##########################

result = model.predict([np.array(test_X_batch_struct), np.array(test_X_batch)],
                batch_size = batch_size_train, 
                steps = len(test_Y_batch)/batch_size_test,
                verbose = 1)[0]

for i in range(len(result)-1):
    result[i+1] +=result[i]

print(test_Y_batch)
print(result)
plt.plot(result)
plt.show()
