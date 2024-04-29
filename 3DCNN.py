# -*- coding: utf-8 -*-
## Import function needed
import tensorflow as tf
import keras.backend as K  #use tensorflow as calculation
import numpy as np
import h5py
import scipy.io as sio 

from keras.layers import Activation, Concatenate, Lambda, Input, Dense, BatchNormalization, TimeDistributed, Dropout, Reshape, LSTM, Conv3D, Conv2D, add, LeakyReLU, PReLU, MaxoutDense, Flatten, ReLU
from keras.models import Model, model_from_json, Sequential  
from keras import regularizers, optimizers
from keras.regularizers import l2
from keras.callbacks import TensorBoard, Callback, EarlyStopping
import scipy.io as sio 
import numpy as np
import math
import time
import random
import matplotlib.pyplot as plt
tf.reset_default_graph()  # reset graph for each time
np.random.seed(1337)  # for reproducibility

## Parameter setting
final_time = 9998
ITERATIONS = 1000
BATCH_SIZE = 100
num_unit = 2
accum_loss= []
length = 3
size1 = 50
size2 = 50
each_size = 16
lenth = 5
time_len = 21
## Network structure define
def extract_(in_,idx):
    return in_[idx,:,:,:]

def Pred_network(Input_1,num_unit):       
    
    def add_common_layers(y):
        y = BatchNormalization()(y)
        y = PReLU()(y)
        return y
    
    def residual_block(y):
        shortcut = y         
#        x = Conv2D(64, kernel_size=(4, 4), padding='same', data_format='channels_first')(y)
#        x = add_common_layers(x)
#        x = Conv2D(32, kernel_size=(4, 4), padding='same', data_format='channels_first')(y)
#        x = add_common_layers(x)
#        x = Conv2D(16*lenth, kernel_size=(4, 4), padding='same', data_format='channels_first')(y)
#        x = add_common_layers(x)
        x = Conv3D(8, kernel_size=(3, 3, 3), padding='same')(y)
        x = add_common_layers(x)
        x = Conv3D(4, kernel_size=(3, 3, 3), padding='same')(x)
        x = add_common_layers(x)
        x = Conv3D(2, kernel_size=(3, 3, 3), padding='same')(x)
        x = add_common_layers(x)
        x = Conv3D(1, kernel_size=(3, 3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = TimeDistributed(Flatten())(x) # Flatten the output for each time step
        x = LSTM(units=64)(x) 
        x = BatchNormalization()(x)
        g = add([shortcut, x])
#        g = PReLU()(g)
#        x = Flatten()(g)
#        x = Dense(1536,activation='relu')(x)
#        x = Dense(1024,activation='relu')(x)
        return g            
    
    # Two CNN model with relu
    for i in range(num_unit):
        Input_2 = residual_block(Input_1)
    
    x = Flatten()(g)
    x = Dense(2*5*50*50,activation='relu')(x)
    x = Dense(5*50*50,activation='relu')(x)
    Final_output = Activation('sigmoid')(Input_2)
    return Final_output



#def root_mean_squared_error(y_true, y_pred):
#    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
#rmse = RMSE = root_mean_squared_error
def R_loss(true,pred):
    RL = true*K.pow(K.log(K.maximum(pred,1.0e-9))/K.log(2.0),2)
    RL += -(1-true)*K.log(K.maximum(1-pred,1.0e-9))/K.log(2.0)
    return K.mean(RL)

adam = optimizers.Adam(lr=0.001)
InInput = Input(shape=(lenth,size1,size1,1))  # without batchsize: 3*50*50
Output = Pred_network(InInput,num_unit)
PredictModel = Model(inputs=InInput, outputs= Output)
PredictModel.compile(optimizer=adam,
                    loss=[R_loss],
                    loss_weights=[1])

print(PredictModel.summary()) 


## LOAD DATA INTO PYTHON
mat_1 = sio.loadmat('UEgrid50_non_1.mat') #200*(21*16)*50*50
GridUE_1 = mat_1['UEgrid50_non_1'] 
mat_2 = sio.loadmat('UEgrid50_non_2.mat') #200*(21*16)*50*50
GridUE_2 = mat_2['UEgrid50_non_2'] 
mat_3 = sio.loadmat('UEgrid50_non_3.mat') #200*(21*16)*50*50
GridUE_3 = mat_3['UEgrid50_non_3'] 
mat_4 = sio.loadmat('UEgrid50_non_4.mat') #200*(21*16)*50*50
GridUE_4 = mat_4['UEgrid50_non_4'] 
mat_5 = sio.loadmat('UEgrid50_non_5.mat') #200*(21*16)*50*50
GridUE_5 = mat_5['UEgrid50_non_5'] 


#data_orgall = np.concatenate([GridUE_1,GridUE_2,GridUE_3,GridUE_4,GridUE_5,GridUE_6,GridUE_7,GridUE_8,GridUE_9],axis = 0)
data_orgall = np.concatenate([GridUE_1,GridUE_2,GridUE_3,GridUE_4,GridUE_5],axis = 0)
data_all = np.reshape(data_orgall, (data_orgall.shape[0]*time_len,each_size,1,size1,size2))
#data_all = np.reshape(data_all_temp, (data_all_temp.shape[0],each_size,size1,size2))

## Preprocessing data 3 time slot
index = 0;
data_temp = [];
data_tempcon = [];
lenn = int(data_all.shape[0])-15
while index <=lenn:
    if (index+1)%time_len  == 0:
        index = index + 1
    elif (index+2)%time_len  == 0:
        index = index + 1 
    elif (index+3)%time_len  == 0:
        index = index + 1 
    elif (index+4)%time_len  == 0:
        index = index + 1 
    elif (index+5)%time_len  == 0:
        index = index + 1 
#    elif (index+5)%time_len  == 0:
#        index = index + 1 
#    elif (index+6)%time_len  == 0:
#        index = index + 1 
#    elif (index+7)%time_len  == 0:
#        index = index + 1 
#    elif (index+8)%time_len  == 0:
#        index = index + 1 
#    elif (index+9)%time_len  == 0:
#        index = index + 1    
#    elif (index+10)%time_len  == 0:
#        index = index + 1    
#    elif (index+11)%time_len  == 0:
#        index = index + 1    
    else:
        data_1 = data_all[index,:,:,:,:]
        data_2 = data_all[index+1,:,:,:,:]
        data_3 = data_all[index+2,:,:,:,:]
        data_4 = data_all[index+3,:,:,:,:]
        data_5 = data_all[index+4,:,:,:,:]
        data_6 = data_all[index+5,:,:,:,:]
        data_tempcon = np.concatenate([data_1,data_2,data_3,data_4,data_5,data_6],axis = 1)
        data_tempcon_tempp = np.reshape(data_tempcon, (each_size,lenth*2,size1,size2))
        data_tempcon_tempp = list(data_tempcon_tempp)
        data_temp.append(data_tempcon)  #
        index = index + 1
data_overall = np.array(data_temp)

#data_overall = np.random.permutation(data_four)
#data_overall_list = list(data_overall )


## Data variale
number_train = round(0.6 * data_overall.shape[0]) 
number_val = round(0.2 * data_overall.shape[0])
number_test = round(0.2 * data_overall.shape[0])
train_min = 0
train_max = int(number_train) -1
val_min = int(number_train)
val_max = int(number_train)+int(number_val)-1
test_min = int(number_train)+int(number_val)
test_max = int(number_train)+int(number_val)+int(number_test)-1

# Data as input
train_data = data_overall[train_min:train_max,:,0:lenth,:,:]
train_data = np.reshape(train_data, (train_data.shape[0]*each_size,lenth,size1,size2,1))
train_data_list = list(train_data)
val_data = data_overall[val_min:val_max,:,0:lenth,:,:]
val_data = np.reshape(val_data, (val_data.shape[0]*each_size,lenth,size1,size2,1))
test_data = data_overall[test_min:test_max,:,0:lenth,:,:]
test_data = np.reshape(test_data, (test_data.shape[0]*each_size,lenth,size1,size2,1))
test_data_A = list(test_data)

# Data as answer
train_ans = data_overall[train_min:train_max,:,lenth:lenth*2,:,:]
train_ans = np.reshape(train_ans, (train_ans.shape[0]*each_size,lenth,size1,size2,1))
train_ans_list = list(train_ans)
val_ans = data_overall[val_min:val_max,:,lenth:lenth*2,:,:]
val_ans = np.reshape(val_ans, (val_ans.shape[0]*each_size,lenth,size1,size2,1))
test_ans = data_overall[test_min:test_max,:,lenth:lenth*2,:,:]
test_ans = np.reshape(test_ans, (test_ans.shape[0]*each_size,lenth,size1,size2,1))
test_ans_A = test_ans[:,lenth-1,:,:]
test_ans_A = list(test_ans_A)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
#session = keras.backend.get_session()
init = tf.global_variables_initializer()
sess.run(init)       
#training
history = PredictModel.fit(train_data, train_ans,
          batch_size=BATCH_SIZE,
          epochs=ITERATIONS,
          verbose=1,
          validation_data= (val_data, val_ans)
          )



ResultCNN = PredictModel.predict(test_data)  #8982*1*50*50 --> 998*9*50*50
Test = int(ResultCNN.shape[0]/each_size)
Result=np.reshape(ResultCNN, (Test*each_size,lenth,size1,size2))
Result_A=Result[:,lenth-1,:,:]
Result_A = list(Result_A)
#test= np.reshape(test_ans, (test_ans.shape[0], 50, 50))
#ResultCNN = np.reshape(ResultCNN, (999, 9, 50, 50))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

import scipy.io as io
io.savemat('CNNans.mat',{'CNNans':test_ans})
io.savemat('InputCNN.mat',{'InputCNN':test_data})
io.savemat('ResultCNN.mat',{'ResultCNN':Result})
PredictModel.save_weights('CNN_weights.h5')