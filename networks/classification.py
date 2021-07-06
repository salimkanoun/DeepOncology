import numpy as np

from tensorflow.keras.layers import Input
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout 
from tensorflow.keras.models import Model


def classification(input_shape=(1024, 256, 1)):
    x_input = Input(input_shape)

    x = Conv2D(32, (7,7), name='conv1', activation='relu')(x_input)
    x = MaxPooling2D(pool_size=(2,2), name='max_pool1')(x)
    x = Conv2D(64, (3,3), name ='conv2', activation = 'relu')(x)
    x = MaxPooling2D(pool_size=(2,2), name='max_pool2')(x)
    #x = Dropout(rate = 0.5, name='dropout1')(x)

    x = Conv2D(64, (3,3), name='conv3', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2), name = 'max_pool3')(x)
    #x = Dropout(rate=0.5, name='dropout2')(x)


    x = Flatten(name ='flatten')(x)

    #dense layer and final output

    x = Dense(256, activation= 'relu', name='dense_1')(x)
    x = Dropout(rate = 0.5, name='dropout_1')(x) 
    x = Dense(128, activation= 'relu', name= 'dense_2')(x) 
    x = Dropout(rate = 0.5, name = 'dropout_2')(x)

    left_arm = Dense(2, activation='softmax', name='left_arm')(x)
    right_arm = Dense(2, activation='softmax', name = 'right_arm')(x)

    head = Dense(2, activation='softmax', name='head')(x)
    legs = Dense(3, activation='softmax', name='legs')(x)

    model = Model(inputs = x_input, outputs = [head, legs, right_arm, left_arm], name='classic_model')
    return model 