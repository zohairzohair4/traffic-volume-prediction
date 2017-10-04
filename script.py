# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, RepeatVector
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from numpy import nanmean
from keras.models import Sequential
from keras.layers import Dense,UpSampling1D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation,BatchNormalization,Input
from keras.layers.pooling import MaxPooling1D
from keras.layers import Conv2D, MaxPooling2D,Conv1D
from keras.optimizers import Adam,RMSprop,Adadelta
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import pywt
from sklearn.svm import SVR
from pre_processing import dataset_avg_vol_8_10_17_19, dataset_avg_vol_6_8_15_17

dataset = dataset_avg_vol_6_8_15_17
dataset_1_0_x = dataset.ix[(dataset['tollgate_id'] == 0.0) & (dataset['direction'] == 0.0)]
dataset_2_0_x = dataset.ix[(dataset['tollgate_id'] == 0.5) & (dataset['direction'] == 0.0)]
dataset_3_0_x = dataset.ix[(dataset['tollgate_id'] == 1.0) & (dataset['direction'] == 0.0)] 
dataset_1_1_x = dataset.ix[(dataset['tollgate_id'] == 0.0) & (dataset['direction'] == 1.0)]
dataset_3_1_x = dataset.ix[(dataset['tollgate_id'] == 1.0) & (dataset['direction'] == 1.0)] 

dataset = dataset_avg_vol_8_10_17_19
dataset_1_0_y = dataset.ix[(dataset['tollgate_id'] == 0.0) & (dataset['direction'] == 0.0)]
dataset_2_0_y = dataset.ix[(dataset['tollgate_id'] == 0.5) & (dataset['direction'] == 0.0)]
dataset_3_0_y = dataset.ix[(dataset['tollgate_id'] == 1.0) & (dataset['direction'] == 0.0)] 
dataset_1_1_y = dataset.ix[(dataset['tollgate_id'] == 0.0) & (dataset['direction'] == 1.0)]
dataset_3_1_y = dataset.ix[(dataset['tollgate_id'] == 1.0) & (dataset['direction'] == 1.0)] 

dataset_2_0_x = dataset_2_0_x.fillna(dataset_2_0_x['volume'].mean(),axis =1)
dataset_2_0_y = dataset_2_0_y.fillna(dataset_2_0_y['volume'].mean(),axis =1)


#2,4 volume min
roll = 7
def get_rolling_median(dataset):
    #dataset = dataset.sort_values(['direction', 'tollgate_id'])
    dataset.loc[:,'rolling_median'] = np.zeros(dataset.shape[0], dtype=float, order='C')
    #seperate data into 1-0 2-0 3-0 1-1 3-1
    
    dataset.loc[:,'rolling_median'] = dataset['volume'].rolling(roll).median().fillna(0)
    return dataset

def get_rolling_mean(dataset):
    #dataset = dataset.sort_values(['direction', 'tollgate_id'])
    dataset.loc[:,'rolling_mean'] = np.zeros(dataset.shape[0], dtype=float, order='C')
    #seperate data into 1-0 2-0 3-0 1-1 3-1
    
    dataset.loc[:,'rolling_mean']= dataset['volume'].rolling(roll).mean().fillna(0)
    return dataset 

def get_rolling_skew(dataset):
    #dataset = dataset.sort_values(['direction', 'tollgate_id'])
    dataset.loc[:,'rolling_skew'] = np.zeros(dataset.shape[0], dtype=float, order='C')
    #seperate data into 1-0 2-0 3-0 1-1 3-1
    dataset.loc[:,'rolling_skew'] = dataset['volume'].rolling(roll).max().fillna(0)
    return dataset     

def get_rolling_min(dataset):
    #dataset = dataset.sort_values(['direction', 'tollgate_id'])
    dataset.loc[:,'rolling_min'] = np.zeros(dataset.shape[0], dtype=float, order='C')
    #seperate data into 1-0 2-0 3-0 1-1 3-1
    dataset.loc[:,'rolling_min'] = dataset['volume'].rolling(roll).min().fillna(0)
    return dataset

def get_rolling_max(dataset):
    #dataset = dataset.sort_values(['direction', 'tollgate_id'])
    dataset.loc[:,'rolling_max'] = np.zeros(dataset.shape[0], dtype=float, order='C')
    #seperate data into 1-0 2-0 3-0 1-1 3-1
    dataset.loc[:,'rolling_max']= dataset['volume'].rolling(roll).max().fillna(0)
    return dataset

def get_diff(dataset):
    #dataset = dataset.sort_values(['direction', 'tollgate_id'])
    dataset.loc[:,'rolling_diff'] = np.zeros(dataset.shape[0], dtype=float, order='C')
    #seperate data into 1-0 2-0 3-0 1-1 3-1
    order = 1
    dataset.loc[:,'rolling_diff'] = dataset['volume'].diff(order).fillna(0)
    return dataset     



dataset_1_0_x  =   get_format(dataset_1_0_x)
dataset_2_0_x  = get_format(dataset_2_0_x)
dataset_3_0_x = get_format(dataset_3_0_x)
dataset_1_1_x = get_format(dataset_1_1_x)
dataset_3_1_x = get_format(dataset_3_1_x)

#get_format_tripple_length

dataset_2_0_x = np.delete(dataset_2_0_x,1,0)

data_set_combined_x = np.concatenate((dataset_1_0_x, dataset_2_0_x,dataset_3_0_x,dataset_1_1_x,dataset_3_1_x), axis=0)

#get_format_tripple_length


dataset_1_0_y = get_format(dataset_1_0_y)
dataset_2_0_y = get_format(dataset_2_0_y)
dataset_3_0_y = get_format(dataset_3_0_y)
dataset_1_1_y = get_format(dataset_1_1_y)
dataset_3_1_y = get_format(dataset_3_1_y)

dataset_2_0_y = np.delete(dataset_2_0_y,1,0)
dataset_2_0_y = np.delete(dataset_2_0_y,0,0)



data_set_combined_y = np.concatenate((dataset_1_0_y, dataset_2_0_y,dataset_3_0_y,dataset_1_1_y,dataset_3_1_y), axis=0)





###################################################################################################
##################################### get ytrain data      ##########################################



data_set_combined_y = get_y_data(data_set_combined_y)



x_train_1 = data_set_combined_x

y_train_1= data_set_combined_y

    


###################################################################################################
# divide into train test

print(x_train_1.shape)

x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(
    x_train_1, y_train_1, test_size=0.33, random_state=42)





###################################################################################################


look_back = 1
batch_size = 64
# reshape input to be [samples, time steps, features]
trainX_1 = np.reshape(x_train_1, (x_train_1.shape[0], x_train_1.shape[1],1))
testX_1 = np.reshape(x_test_1, (x_test_1.shape[0], x_test_1.shape[1], 1))

clf_1 = Sequential()
clf_1.add(Conv1D(64, kernel_size=4,strides=1,activation='relu',input_shape=(8, 1),kernel_initializer='glorot_normal',name='cnn'))
clf_1.add(MaxPooling1D())
#clf_1.add(LSTM(100,return_sequences=True,name='lstm_100'))
#clf_1.add(LSTM(70,name='lstm_70')) 
clf_1.add(Flatten())
clf_1.add(Dense(50,kernel_initializer='glorot_normal',name='dense_50'))
clf_1.add(Activation('relu'))
clf_1.add(Dense(30,kernel_initializer='glorot_normal',name='dense_30'))
clf_1.add(Activation('relu'))
clf_1.add(Dense(20,kernel_initializer='glorot_normal',name='dense_10'))
clf_1.add(Activation('relu')) 
clf_1.add(Dense(6,kernel_initializer='glorot_normal',name='dense_6'))
clf_1.add(Activation('linear'))
rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
clf_1.compile(loss='mean_absolute_error',optimizer=rmsprop)#
filepath_1_cnn = "/tmp/weights.best_1_cnn.hdf5"
checkpoint_1_cnn = ModelCheckpoint(filepath_1_cnn, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list_1_cnn = [checkpoint_1_cnn]
clf_1.fit(trainX_1, y_train_1,validation_data=(testX_1,y_test_1), epochs=100, batch_size=batch_size, verbose=2,callbacks=callbacks_list_1_cnn) # volume
#clf_1.fit(trainX_1, y_train_1, epochs=200,verbose=1, batch_size=batch_size) # volume
clf_1.load_weights("/tmp/weights.best_1_cnn.hdf5")
y_pred_1 = clf_1.predict(testX_1)


y_pred_1 = y_pred_1.flatten()

y_test_1 = y_test_1.flatten()



y_test_1[y_test_1 == 0] = 1
mape_1 = mean_absolute_percentage_error(y_test_1, y_pred_1)
print(mape_1)



###################### handling the submission files and removing the sample time and volume ######################
submission_avg_time,index_value_avg_time = get_submission(submission_avg_time,'avg_travel_time')
submission_avg_volume,index_value_avg_volume = get_submission(submission_avg_volume,'volume')
test_avg_time,_ = get_submission(test_avg_time,'avg_travel_time')
test_avg_vol,_ = get_submission(test_avg_vol,'volume')

#test_avg_vol.sort_values(['direction', 'tollgate_id'])
###################################################################################################
# taking important intervals out 

submission_avg_time = cat_to_int(submission_avg_time)
test_avg_time = cat_to_int(test_avg_time)

#submission_avg_time['intersection_id'] = submission_avg_time['intersection_id'].astype('category')
#submission_avg_time['intersection_id'] = submission_avg_time['intersection_id'].cat.codes 


test_avg_time = fill_missing_values_time(test_avg_time,time_range = ['6:00','8:00'])
test_avg_time = fill_missing_values_time(test_avg_time,time_range = ['15:00','17:00'])

test_avg_time_6_8  = test_avg_time.between_time('6:00','8:00',include_end=False)
test_avg_time_15_17  = test_avg_time.between_time('15:00','17:00')


vis_6_8  = test_avg_vol.between_time('6:00','8:00',include_end=False)
vis_15_17  = test_avg_vol.between_time('15:00','17:00',include_end=False)
vis_6_8 = vis_6_8.sort_values(['direction', 'tollgate_id'])
vis_15_17 = vis_15_17.sort_values(['direction', 'tollgate_id'])
vis = pd.concat([vis_6_8,vis_15_17 ], axis=0)
vis[['volume']] = scaler_vol.inverse_transform(vis[['volume']])

submission_avg_time_8_10 = submission_avg_time.between_time('8:00','10:00',include_end=False)
submission_avg_time_17_19 = submission_avg_time.between_time('17:00','19:00',include_end=False)

test_avg_time_6_8  =   get_format_time(test_avg_time_6_8)
test_avg_time_15_17  = get_format_time(test_avg_time_15_17)
submission_avg_time_8_10  =   get_format_submission_time(submission_avg_time_8_10)
submission_avg_time_17_19  = get_format_submission_time(submission_avg_time_17_19)

dataset = test_avg_vol
test_avg_vol_1_0_x = dataset.ix[(dataset['tollgate_id'] == 0.0) & (dataset['direction'] == 0.0)]
test_avg_vol_2_0_x = dataset.ix[(dataset['tollgate_id'] == 0.5) & (dataset['direction'] == 0.0)]
test_avg_vol_3_0_x = dataset.ix[(dataset['tollgate_id'] == 1.0) & (dataset['direction'] == 0.0)] 
test_avg_vol_1_1_x = dataset.ix[(dataset['tollgate_id'] == 0.0) & (dataset['direction'] == 1.0)]
test_avg_vol_3_1_x = dataset.ix[(dataset['tollgate_id'] == 1.0) & (dataset['direction'] == 1.0)] 


test_avg_vol_1_0_x  =   get_format(test_avg_vol_1_0_x)
test_avg_vol_2_0_x  = get_format(test_avg_vol_2_0_x)
test_avg_vol_3_0_x = get_format(test_avg_vol_3_0_x)
test_avg_vol_1_1_x = get_format(test_avg_vol_1_1_x)
test_avg_vol_3_1_x = get_format(test_avg_vol_3_1_x)

dataset = submission_avg_volume
submission_avg_volume_1_0_x = dataset.ix[(dataset['tollgate_id'] == 1) & (dataset['direction'] == 0)]
submission_avg_volume_2_0_x = dataset.ix[(dataset['tollgate_id'] == 2) & (dataset['direction'] == 0)]
submission_avg_volume_3_0_x = dataset.ix[(dataset['tollgate_id'] == 3) & (dataset['direction'] == 0)] 
submission_avg_volume_1_1_x = dataset.ix[(dataset['tollgate_id'] == 1) & (dataset['direction'] == 1)]
submission_avg_volume_3_1_x = dataset.ix[(dataset['tollgate_id'] == 3) & (dataset['direction'] == 1)] 

submission_avg_volume_1_0_x  =   get_format_submission(submission_avg_volume_1_0_x)
submission_avg_volume_2_0_x  =   get_format_submission(submission_avg_volume_2_0_x)
submission_avg_volume_3_0_x  =   get_format_submission(submission_avg_volume_3_0_x)
submission_avg_volume_1_1_x  =   get_format_submission(submission_avg_volume_1_1_x)
submission_avg_volume_3_1_x  =   get_format_submission(submission_avg_volume_3_1_x)



test_avg_vol_1_0_x = np.reshape(test_avg_vol_1_0_x, (test_avg_vol_1_0_x.shape[0], test_avg_vol_1_0_x.shape[1], 1))
test_avg_vol_2_0_x = np.reshape(test_avg_vol_2_0_x, (test_avg_vol_2_0_x.shape[0], test_avg_vol_2_0_x.shape[1], 1))
test_avg_vol_3_0_x = np.reshape(test_avg_vol_3_0_x, (test_avg_vol_3_0_x.shape[0], test_avg_vol_3_0_x.shape[1], 1))
test_avg_vol_1_1_x = np.reshape(test_avg_vol_1_1_x, (test_avg_vol_1_1_x.shape[0], test_avg_vol_1_1_x.shape[1], 1))
test_avg_vol_3_1_x = np.reshape(test_avg_vol_3_1_x, (test_avg_vol_3_1_x.shape[0], test_avg_vol_3_1_x.shape[1], 1))

test_set_combined_y = np.concatenate((test_avg_vol_1_0_x, test_avg_vol_2_0_x,test_avg_vol_3_0_x,test_avg_vol_1_1_x,test_avg_vol_3_1_x), axis=0)
submission_avg_volume = pd.concat([submission_avg_volume_1_0_x,submission_avg_volume_2_0_x,submission_avg_volume_3_0_x,submission_avg_volume_1_1_x,submission_avg_volume_3_1_x ], axis=0)


###################################################################################################
############################# prediction ##########################################################
batch_size = 1
print(test_set_combined_y.shape)
prediction_vol_1_0_x = clf_1.predict(test_set_combined_y)

prediction_vol_1_0_x = prediction_vol_1_0_x.flatten()


submission_avg_volume.loc[:, 'volume'] = prediction_vol_1_0_x
submission_avg_volume = submission_avg_volume.sort_values(['direction', 'tollgate_id'])


submission_avg_volume.index = pd.to_datetime(submission_avg_volume.index)

submission_avg_volume_8_10  = submission_avg_volume.between_time('8:00','10:00',include_end=False)
submission_avg_volume_17_19 = submission_avg_volume.between_time('17:00','19:00',include_end=False)
submission_avg_volume_8_10 = submission_avg_volume_8_10.sort_values(['direction', 'tollgate_id'])
submission_avg_volume_17_19 = submission_avg_volume_17_19.sort_values(['direction', 'tollgate_id'])

final_submission = pd.concat([submission_avg_volume_8_10,submission_avg_volume_17_19 ], axis=0)

#final_submission = final_submission.sort_index()
#final_submission = final_submission.sort_values(['direction', 'tollgate_id'])

final_submission[['volume']] = scaler_vol.inverse_transform(final_submission[['volume']])
#final_submission[['volume']].min(axis=0)
final_submission.to_csv(path_or_buf= 'final_submission.csv', index = False)

