# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:26:45 2019

@author: iannis
"""

import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import csv


TRAIN_DIR = r'C:\Users\ianni\Desktop\robot career\career-con-2019'

import math
import keras
import tensorflow as tf
tf.reset_default_graph()
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, LSTM, RNN
from keras.backend import eval
from keras.optimizers import Adam


# =============================================================================
# =============================================================================
# =============================================================================
# # # ici debut de la zone traitement de données
# =============================================================================
# =============================================================================
# =============================================================================

#training_data[0]
#Out[24]: ['row_id,series_id,measurement_number,orientation_X,
#orientation_Y,orientation_Z,orientation_W,angular_velocity_X,angular_velocity_Y,angular_velocity_Z,
#linear_acceleration_X,linear_acceleration_Y,linear_acceleration_Z']

def process_train_dataX(measure='off'):
    training_data=[]
    with open(r'C:\Users\ianni\Desktop\robot career\career-con-2019\X_train.csv','rt') as csvfile:
        a = csv.reader(csvfile,delimiter=' ')
        for row in a:
            row=row[0].split(',')
            data=[]
            lenn=len(row)
#            degage la row 0 et 2
            for i in range(1,lenn):
                if i==2:
                    if measure!='off':
                        data.append(row[i])
                else:
                    data.append(row[i])
            training_data.append(data)
    training_data=training_data[1:]
    

    return training_data

#training_data[0]
#Out[21]: ['series_id,group_id,surface']
    
def process_train_dataY():
    training_data=[]
    with open(r'C:\Users\ianni\Desktop\robot_career\career-con-2019\y_train.csv','rt') as csvfile:
        a=csv.reader(csvfile,delimiter=' ')
        for row in a:
            row=row[0].split(',')
            training_data.append([row[0],row[2]])
    
    training_data=training_data[1:]
    return training_data

def mix_data(a,b):
    for i in a:
        i[0]=b[int(i[0])][1]
        
def process_to_float(a):
    lenn=len(a[0])
    for i in a:
        for j in range(1,lenn):
            i[j]=float(i[j])
            
def normalyze_factor_table(a):
    lenn=len(a[0])
    List=[[x,x] for x in a[0][1:]]

    for i in a:
        for j in range(1,lenn):
            if i[j]<List[j-1][0]:
                List[j-1][0]=i[j]
            if i[j]>List[j-1][1]:
                List[j-1][1]=i[j]
    return List

def factor_to_normalyze(List_):
    lenn=np.shape(List_)[0]
    factor=List_.copy()
    for i in range(lenn):
        factor[i][0]=-List_[i][0]
        factor[i][1]+=factor[i][0]
        
#    step1 add by factor[i][0]
#    step2 divide by factor[i][1]
#    step3 multiply by 2
#    step4 -1
        
def change_val_for_neural(factor,a):
    lenn=len(factor)
    for i in a:
        for j in range(1,lenn+1):
            i[j]=2*( (i[j]+factor[j-1][0])/(factor[j-1][1]) )-1
    return a
# =============================================================================
# =============================================================================
#             traitement avant load
# =============================================================================
# a=process_train_dataX()
# b=process_train_dataY()
# mix_data(a,b)
# process_to_float(a)
# List_=normalyze_factor_table(a)
# factor=factor_to_normalyze(List_)
# data_for_neural=change_val_for_neural(factor,a)
# =============================================================================

      
# =============================================================================
# =============================================================================
# =============================================================================
# # # ici debut de la zone neural
# =============================================================================
# =============================================================================
# =============================================================================
            
def neural_model(List=[64,128,256,128,64],drop=0.2,LR=0.0001,input_size=10):
    model = Sequential()
    
    model.add(Dense(List[0], input_shape=(input_size,),activation='relu'))
    model.add(Dropout(drop))
    
    for i in range(1,len(List)):
        model.add(Dense(List[i], activation='relu'))
        model.add(Dropout(drop))
    
    model.add(Dense(9, activation='softmax'))
    
    learning_rate = LR
    opt = keras.optimizers.adam(lr=learning_rate, decay=1e-10)
    
    model.compile(loss='mean_squared_error',optimizer=opt)
    
    return model
    
def split_XY(data):
    X=[x[1:] for x in data]
    Y=[y[0] for y in data]
    
    return X,Y

def give_val(Y):

    dico=[]
    for i in Y:
        if i in dico:
            pass
        else:
            dico.append(i)        
        
    return dico

def train_model(data,hm_epoch=5,train_size=0.8,batch=64):
    
    x = np.array([i[1:] for i in data])

    
    dico=give_val([i[0] for i in data])
    y=np.zeros([np.shape(x)[0],len(dico)])
    lenn=len(data)
    for i in range(lenn):
        val=dico.index(data[i][0])
        y[i][val]=1
    
    model.fit(x=x,
              y=y,
              epochs=hm_epoch,
              validation_split=1-train_size,
              batch_size=batch)

    return model
# =============================================================================
# =============================================================================
# =============================================================================
# # # ici debut de la zone de traitement
# =============================================================================
# =============================================================================
# =============================================================================
factor=np.load('factor.npy')
data=np.load('data_for_neural.npy')

model=None
# on peut add un shuffle ici
import random
random.shuffle(data)
data_=data[:int(0.8*len(data))]
test=data[int(0.8*len(data)):]
# on peut add un shuffle ici
#X,Y=split_XY(data)
#Y,dico=give_val(Y)
#X = np.array([[i] for i in X])
#Y = np.array([[i] for i in Y])
# =============================================================================
# =============================================================================
# # model=neural_model()
# # model=train_model(data_)
# =============================================================================
# =============================================================================

# model.save('model0')


# =============================================================================
# =============================================================================
# =============================================================================
# # # ici debut de la zone test
# =============================================================================
# =============================================================================
# =============================================================================

def test(model=model,test=test):
    if model==None:
        model=load_model('model0')
    lenn=len(test)
    compteur=0
    dico=give_val([i[0] for i in data])
    searcheur=[[0,0,0]]
    for i in range(8):
        searcheur.append([0,0,0])
    for i in range(lenn):
        result=np.argmax(model.predict(np.array([test[i][1:]])))
        name = dico[result]
        if test[i][0]==name:
            compteur+=1
            searcheur[result][0]+=1
        else:
            searcheur[result][1]+=1
        searcheur[dico.index(test[i][0])][2]+=1
        if i%10000==0:
            print(i)
            
    print(compteur)
    print(compteur/lenn)




























