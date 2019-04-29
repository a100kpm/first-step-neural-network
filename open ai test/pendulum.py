# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 19:41:17 2018

@author: iannis
"""

import keras  # Keras 2.1.2 and TF-GPU 1.9.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard

import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from numpy import arccos
from statistics import median
from math import sqrt,floor

tf.reset_default_graph()

#env = gym.make('Pendulum-v0').env

def play_random_game(step=200):
    env = gym.make('Pendulum-v0').env
    env.reset()
    
    
    for _ in range(step):
        env.render()
        
        action = env.action_space.sample()
        print('action=',action)
        
        observation, reward, done, info = env.step(action)
        print('obs={} , reward={} , done={} , info= {}'.format(observation,reward,done,info))
#        input()
        
    env.close()
        
# =============================================================================
# faire des data basé sur un reward moins nul a chaque itération, et des
# data basé sur un reward moins nul au bout de X itération
# =============================================================================
def create_random_data(step=200,hm_games=1000,moy=None):
    env= gym.make('Pendulum-v0')
    data=[]
    compteur=1
    for _ in range(hm_games):
        prev_observation=env.reset()
        
        memory=[]
        reward_tot=0
        for _ in range(step):
            
            action = env.action_space.sample()
            
            observation, reward, done, info = env.step(action)
            reward_tot+=reward
            memory.append([prev_observation,action])
            
            prev_observation=observation
        
        
        if moy==None:
            moy=reward_tot
            data.append(memory)
        elif moy<reward_tot:
            
            data.append(memory)
            moy=(moy*compteur+reward_tot)/(compteur+1)
            compteur+=1
    print(moy)
    return data

def create_educated_random_data(step=200,hm_games=1000):
    env = gym.make('Pendulum-v0')
    data=[]
    env.reset()

    
    for _ in range(hm_games):
        action = env.action_space.sample()
        prev_obs,prev_reward,done,info=env.step(action)
#        memory=[]
        
        for _ in range(step):
            action=env.action_space.sample()
            observation,reward,done,info=env.step(action)
            
            if reward>prev_reward:
                data.append([prev_obs,action])
            prev_reward=reward
            prev_obs=observation
            
    return data
        
    
def neural_model():
    model = Sequential()
    
    model.add(Dense(64, input_shape=(3,),activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
#    
#    model.add(Dense(128, activation='relu'))
#    model.add(Dropout(0.2))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='hard_sigmoid'))
    # torque entre -2 et 2 donc faire -0.5 puis *4
    
    learning_rate = 0.0001
    opt = keras.optimizers.adam(lr=learning_rate, decay=1e-6)
    
    model.compile(loss='mean_squared_error',
                  optimizer=opt)
#                  metrics=['mae'])
    return model
#tensorboard = TensorBoard(log_dir="logs/test")

#train_data_dir = "train_data"


#hm_epoch=10
#
#for i in range(hm_epochs):
#    current = 0
#    increment = 200
#    not_maximum = True
#    all_files = os.listdir(train_data_dir)
#    maximum = len(all_files)
#    random.shuffle(all_files)
#
#        test_size = 100
#        batch_size = 128
#
#        x_train = np.array([i[1] for i in train_data[:-test_size]]).reshape(-1, 176, 200, 3)
#        y_train = np.array([i[0] for i in train_data[:-test_size]])
#
#        x_test = np.array([i[1] for i in train_data[-test_size:]]).reshape(-1, 176, 200, 3)
#        y_test = np.array([i[0] for i in train_data[-test_size:]])
#
#        model.fit(x_train, y_train,
#                  batch_size=batch_size,
#                  validation_data=(x_test, y_test),
#                  shuffle=True,
#                  verbose=1, callbacks=[tensorboard])
#
#        model.save("BasicCNN-{}-epochs-{}-LR-STAGE1".format(hm_epochs, learning_rate))
#        current += increment
#        if current > maximum:
#            not_maximum = False
    
def normalize_shuffle_data(data):
    
    data=np.reshape(data,(np.shape(data)[0]*np.shape(data)[1],-1))
    lenn = np.shape(data)[0]
    for i in range(lenn):
        data[i][1]=max(0, min(1, data[i][1]*0.2 + 0.5))
#    np.random.shuffle(data)
    return data

def normalize_data(data):
    lenn=np.shape(data)[0]
    for i in range(lenn):
        data[i][1]=max(0, min(1,data[i][1]*0.2 + 0.5))
        
    return data


    
    

def train_model(data,model=None,hm_epoch=10,train_size=0.8,batch=32):
    if model==None:
        model=neural_model()
    
#    data=np.reshape(data,(np.shape(data)[0]*np.shape(data)[1],-1))
    x = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])
    
#    size=train_size*np.shape(y)
#    
#    x_train=x[:size]
#    y_train=y[:size]
#    
#    x_test=x[size:]
#    y_test=y[size:]
    
    
    
    model.fit(x=x,
              y=y,
              epochs=hm_epoch,
              validation_split=1-train_size,
              batch_size=batch)
    
    return model



def play_game(model,step=200,hm_games=10):
    env = gym.make('Pendulum-v0').env
    
    for i in range(hm_games):
        observation=env.reset()
        score=0
    
        for _ in range(step):
            env.render()
            
            action = model.predict(observation.reshape(-1,len(observation)))
            action = action -0.5
            action = action * 2
            observation, reward, done, info = env.step(action)
            score+=reward
    #        input()
        print('score={} for game {}'.format(score,i+1))
        
            
    env.close()