
import numpy as np
#import keras.backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Dense, Activation, Flatten
from tensorflow.keras.layers import Dropout, Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam,RMSprop


import time

def deepPETmodel2(input_shape, learning_rate):
    
    start_time = time.time()
    print ('Constructing Model ... ')
        
    no_angles = input_shape[2]
    img_size  = input_shape[1]
    model = Sequential()

    
    model.add(Conv2D(input_shape=(no_angles,img_size,1), filters=32, kernel_size=(7,7), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=32, kernel_size=(7,7), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
              
    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
                
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
              
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
   
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    '''
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    '''
    
    model.add(UpSampling2D((2, 2), interpolation='bilinear'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
              
    model.add(UpSampling2D((2, 2), interpolation='bilinear'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
              
    model.add(UpSampling2D((2, 2), interpolation='bilinear'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    #model.add(UpSampling2D((2, 2), interpolation='bilinear'))
    #model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same')) ###
    #model.add(BatchNormalization())
    model.add(Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same'))
    
    #model.add(Reshsape((img_size, img_size)))
    
    print ('Model constructed in {0} seconds'.format(time.time() - start_time))
    start_time = time.time()      
        
    #adam = Adam(learning_rate=learning_rate,epsilon=None, decay=0.00001)
    #model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
    
    rms = RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=rms, metrics=['mean_squared_error'])
    
    print ('Model compiled in {0} seconds'.format(time.time() - start_time))
    
    #filepath="weights.best.hdf5"
    #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    #callbacks_list = [checkpoint]

    return model