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

pd.set_option('display.max_columns', 322)

#dataset_avg_time = pd.read_csv("DATA_WC3_COMPLETE.csv",  encoding = "ISO-8859-1")
dataset_avg_time = pd.read_csv("training_20min_avg_travel_time.csv")
dataset_avg_vol = pd.read_csv("training_20min_avg_volume.csv")
weather = pd.read_csv("weather (table 7)_training_update.csv")
submission_avg_time = pd.read_csv("submission_sample_travelTime.csv")
submission_avg_volume =pd.read_csv("submission_sample_volume.csv")
test_avg_time = pd.read_csv("test1_20min_avg_travel_time.csv")
test_avg_vol = pd.read_csv("test1_20min_avg_volume.csv")
test_avg_vol_2 = pd.read_csv("test2_20min_avg_volume.csv")
dataset_avg_vol_2 = pd.read_csv("training2_20min_avg_volume.csv")
dataset_avg_vol  = pd.concat([dataset_avg_vol,dataset_avg_vol_2],axis = 0)
#print(test_avg_vol_2.shape)




def transform_min_max_scaler_vol(dataset_avg_vol,test_avg_vol):
            # remove outlier
    #max_val = dataset_avg_vol['volume'].max()
    #dataset_avg_vol = dataset_avg_vol.ix[dataset_avg_vol['volume'] != max_val]
    #############
    transformed_vol = pd.concat([dataset_avg_vol,test_avg_vol],axis = 0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler_vol = MinMaxScaler(feature_range=(0, 1))
    transformed_vol[['tollgate_id','direction']] = scaler.fit_transform(transformed_vol[['tollgate_id','direction']])
    transformed_vol[['tollgate_id','direction']] = transformed_vol[['tollgate_id','direction']]
    transformed_vol[['volume']] = scaler_vol.fit_transform(transformed_vol[['volume']])
    dataset_avg_vol = transformed_vol.iloc[0:dataset_avg_vol.shape[0]]
    test_avg_vol = transformed_vol.iloc[dataset_avg_vol.shape[0]:dataset_avg_vol.shape[0]+test_avg_vol.shape[0]]
    return dataset_avg_vol,test_avg_vol,scaler_vol
    
    
    


######################handling the training files and adjusting the dates.######################

###################################################################################################
def date_format( dataset ):
    dataset['date_half'] = dataset['time_window'].astype(str).str[1:20]
    dataset['time_window'] =  pd.to_datetime(dataset['date_half'])
    dataset = dataset.drop('date_half', axis=1)
    dataset = dataset.sort_values(by='time_window')
    dataset = dataset.set_index('time_window')
    return dataset

def cat_to_int(dataset):
    dataset['intersection_id'].ix[dataset['intersection_id'] == 'A'] = 1
    dataset['intersection_id'].ix[dataset['intersection_id'] == 'B'] = 2
    dataset['intersection_id'].ix[dataset['intersection_id'] == 'C'] = 3
    return dataset
    
def int_to_cat(dataset):
    dataset['intersection_id'].ix[dataset['intersection_id'] == 1] = 'A'
    dataset['intersection_id'].ix[dataset['intersection_id'] == 2 ] = 'B'
    dataset['intersection_id'].ix[dataset['intersection_id'] == 3] = 'C'
    return dataset    
    
def get_y_data(dataset):
    
    y = np.zeros((dataset.shape[0],6), dtype=float, order='C')
    i = 0
    for x in dataset:
        y[i] = x[0:6]
        i = i+1
    #dataset = dataset.drop(col, axis=1) 
    return y 
    
def get_submission(dataset,col):
    dataset['date_half'] = dataset['time_window'].astype(str).str[1:20]
    dataset['date_half'] =  pd.to_datetime(dataset['date_half'])
    dataset = dataset.sort_values(by='date_half')
    index_values = dataset.index.values
    dataset = dataset.set_index('date_half')
    #dataset = dataset.drop('date_half', axis=1)
    #dataset = dataset.drop(col, axis=1)
    return dataset,index_values    
    


def chunk_of_30(dataset_x_y,dataset_y):
    x = 0
    dataX, dataY = [], []
    a = dataset_x_y.values.tolist()                        
    b = dataset_y.values.tolist()
    for i in range(int(len(a)/30)):
        dataX.append(a[x:x+30])
        dataY.append(b[x:x+30])
        x = x+30


def fill_missing_values_norm(dataset,time_range):

    x = dataset.groupby(pd.TimeGrouper(freq='20Min'))['volume'].count()
    x  = x.between_time(time_range[0],time_range[1],include_end=False)
    x = x.ix[x < 5]
    x = x.apply(lambda x: 5-x)
    y = dataset.loc[x.index.values,:]
    
    #y = y[(y['tollgate_id']== 0.0) & (y['direction'] == 0.0)]
    for index_val in x.index.values:
        yy = dataset_avg_vol.loc[index_val,:]
        columns=['tollgate_id','direction','volume']
        if(yy[(yy['tollgate_id']== 0.0) & (yy['direction'] == 0.0)].empty):
            temp_df = pd.DataFrame(index=[index_val], columns=columns)
            temp_df = temp_df.fillna(0)
            temp_df.loc[index_val,:] = [0.0,0.0,np.array(yy['volume'])[0]]
            dataset = pd.concat([dataset, temp_df ], axis=0)
        if(yy[(yy['tollgate_id']== 0.5) & (yy['direction'] == 0.0)].empty):
            temp_df = pd.DataFrame(index=[index_val], columns=columns)
            temp_df = temp_df.fillna(0)
            temp_df.loc[index_val,:] = [0.5,0.0,np.nan]
            dataset = pd.concat([dataset, temp_df ], axis=0)
        if(yy[(yy['tollgate_id']== 1.0) & (yy['direction'] == 0.0)].empty):
            temp_df = pd.DataFrame(index=[index_val], columns=columns)
            temp_df = temp_df.fillna(0)
            temp_df.loc[index_val,:] = [1.0,0.0,np.array(yy['volume'])[0]]
            dataset = pd.concat([dataset, temp_df ], axis=0)
        if(yy[(yy['tollgate_id']== 0.0) & (yy['direction'] == 1.0)].empty):
            temp_df = pd.DataFrame(index=[index_val], columns=columns)
            temp_df = temp_df.fillna(0)
            temp_df.loc[index_val,:] = [0.0,1.0,np.array(yy['volume'])[0]]
            dataset = pd.concat([dataset, temp_df ], axis=0)
        if(yy[(yy['tollgate_id']== 0.0) & (yy['direction'] == 1.0)].empty):
            temp_df = pd.DataFrame(index=[index_val], columns=columns)
            temp_df = temp_df.fillna(0)
            temp_df.loc[index_val,:] = [0.0,1.0,np.array(yy['volume'])[0]]
            dataset = pd.concat([dataset, temp_df ], axis=0)
            
def fill_missing_values(dataset,time_range):

    x = dataset.groupby(pd.TimeGrouper(freq='20Min'))['volume'].count()
    x  = x.between_time(time_range[0],time_range[1],include_end=False)
    x = x.ix[x < 5]
    if(x.empty): return dataset
    x = x.apply(lambda x: 5-x)
    
    y = dataset.loc[x.index.values,:]
    
    #y = y[(y['tollgate_id']== 1) & (y['direction'] == 1)]
    for index_val in x.index.values:
        yy = dataset_avg_vol.loc[index_val,:]
        columns=['tollgate_id','direction','volume']
        if(yy[(yy['tollgate_id']== 1) & (yy['direction'] == 1)].empty):
            temp_df = pd.DataFrame(index=[index_val], columns=columns)
            temp_df = temp_df.fillna(0)
            temp_df.loc[index_val,:] = [1,1,np.array(yy['volume'])[0]]
            dataset = pd.concat([dataset, temp_df ], axis=0)
        if(yy[(yy['tollgate_id']== 2) & (yy['direction'] == 1)].empty):
            temp_df = pd.DataFrame(index=[index_val], columns=columns)
            temp_df = temp_df.fillna(0)
            temp_df.loc[index_val,:] = [2,1,np.nan]
            dataset = pd.concat([dataset, temp_df ], axis=0)
        if(yy[(yy['tollgate_id']== 3) & (yy['direction'] == 1)].empty):
            temp_df = pd.DataFrame(index=[index_val], columns=columns)
            temp_df = temp_df.fillna(0)
            temp_df.loc[index_val,:] = [3,1,np.array(yy['volume'])[0]]
            dataset = pd.concat([dataset, temp_df ], axis=0)
        if(yy[(yy['tollgate_id']== 1) & (yy['direction'] == 3)].empty):
            temp_df = pd.DataFrame(index=[index_val], columns=columns)
            temp_df = temp_df.fillna(0)
            temp_df.loc[index_val,:] = [1,3,np.array(yy['volume'])[0]]
            dataset = pd.concat([dataset, temp_df ], axis=0)
        if(yy[(yy['tollgate_id']== 1) & (yy['direction'] == 3)].empty):
            temp_df = pd.DataFrame(index=[index_val], columns=columns)
            temp_df = temp_df.fillna(0)
            temp_df.loc[index_val,:] = [1,3,np.array(yy['volume'])[0]]
            dataset = pd.concat([dataset, temp_df ], axis=0)            
    
    
    #dataset = pd.concat([dataset, y ], axis=0)
    dataset.index = pd.to_datetime(dataset.index)
    dataset = dataset.sort_index()
    return dataset    



def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

def my_scorer():
    return make_scorer(mean_absolute_percentage_error, greater_is_better=True) # change for false if using MSE 

def get_format_x(dataset):

    DFList = [group[1] for group in dataset.groupby(dataset.index.day)]          
    i = 0          
    for x in DFList:
        DFList[i] = DFList[i].sort_values(['direction', 'tollgate_id'])
        i = i+1
    dflist = pd.DataFrame()  
    for x in DFList:
        dflist = pd.concat([dflist,pd.DataFrame(x)],axis = 0)

    direction = np.array(dflist['direction'])
    tollgate_id = np.array(dflist['tollgate_id'])
    volume = list(np.array(dflist['volume']))
    #volume_median = list(np.array(dflist['rolling_median']))
    #volume_diff = list(np.array(dflist['rolling_diff']))
    #volume_max = list(np.array(dflist['rolling_max'])) #2
    volume_min = list(np.array(dflist['rolling_min'])) #1
    #volume_mean = list(np.array(dflist['rolling_mean']))
    #xx2 = list()
    
    counter = 0 
    xx = np.zeros(14, dtype=float, order='C')
    xx2 = np.zeros((int(len(volume)/6),14), dtype=float, order='C')
    i = 0
    for x in range(int(len(volume)/6)):
        xx[0:6] = volume[counter:counter + 6] #+ direction[counter] + tollgate_id[counter]
        xx[6:12] = volume_min[counter:counter + 6]
        '''
        xx[12:18] = volume_diff[counter:counter + 6]
        xx[18:24] = volume_max[counter:counter + 6]
        xx[24:30] = volume_min[counter:counter + 6]
        xx[30:36] = volume_mean[counter:counter + 6]
        '''
        xx[12:13] = direction[counter]
        xx[13:14] = tollgate_id[counter]
        xx2[i] = xx
        i = i+1
        counter = counter +6
    return xx2   


def get_format(dataset):

    DFList = [group[1] for group in dataset.groupby(dataset.index.day)]          
    i = 0          
    for x in DFList:
        DFList[i] = DFList[i].sort_values(['direction', 'tollgate_id'])
        i = i+1
    dflist = pd.DataFrame()  
    for x in DFList:
        dflist = pd.concat([dflist,pd.DataFrame(x)],axis = 0)

    direction = np.array(dflist['direction'])
    tollgate_id = np.array(dflist['tollgate_id'])
    volume = list(np.array(dflist['volume']))
    #xx2 = list()
    counter = 0 
    xx = np.zeros(8, dtype=float, order='C')
    xx2 = np.zeros((int(len(volume)/6),8), dtype=float, order='C')
    i = 0
    for x in range(int(len(volume)/6)):
        xx[0:6] = volume[counter:counter + 6] #+ direction[counter] + tollgate_id[counter]
        xx[6:7] = direction[counter]
        xx[7:8] = tollgate_id[counter]
        xx2[i] = xx
        i = i+1
        counter = counter +6   
        
    return xx2   
    
    

    
    
def get_format_submission(dataset):

    DFList = [group[1] for group in dataset.groupby(dataset.index.day)]          
    i = 0          
    for x in DFList:
        DFList[i] = DFList[i].sort_values(['direction', 'tollgate_id'])
        i = i+1
    dflist = pd.DataFrame()  
    for x in DFList:
        dflist = pd.concat([dflist,pd.DataFrame(x)],axis = 0)
    return dflist       
        
def get_format_submission_time(dataset):

    DFList = [group[1] for group in dataset.groupby(dataset.index.day)]          
    i = 0          
    for x in DFList:
        DFList[i] = DFList[i].sort_values(['intersection_id', 'tollgate_id'])
        i = i+1
    dflist = pd.DataFrame()  
    for x in DFList:
        dflist = pd.concat([dflist,pd.DataFrame(x)],axis = 0)           
    return dflist       
    



dataset_avg_vol,test_avg_vol,scaler_vol = transform_min_max_scaler_vol(dataset_avg_vol,test_avg_vol)

    
dataset_avg_vol = date_format(dataset_avg_vol)

###################################################################################################
# taking important intervals out 


print(dataset_avg_vol.shape)
dataset_avg_vol = fill_missing_values(dataset_avg_vol,time_range = ['6:00','10:00'])
print(dataset_avg_vol.shape)
dataset_avg_vol = fill_missing_values(dataset_avg_vol,time_range = ['15:00','19:00'])
print(dataset_avg_vol.shape)



dataset_avg_vol_6_8  = dataset_avg_vol.between_time('6:00','8:00',include_end=False)  
dataset_avg_vol_8_10  = dataset_avg_vol.between_time('8:00','10:00',include_end=False)
dataset_avg_vol_15_17 = dataset_avg_vol.between_time('15:00','17:00',include_end=False)
dataset_avg_vol_17_19 = dataset_avg_vol.between_time('17:00','19:00',include_end=False)

dataset_avg_vol_6_8_15_17 = pd.concat([dataset_avg_vol_6_8,dataset_avg_vol_15_17 ], axis=0)
dataset_avg_vol_8_10_17_19 = pd.concat([dataset_avg_vol_8_10,dataset_avg_vol_17_19 ], axis=0)
