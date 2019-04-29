# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 02:05:47 2018

@author: iannis
"""

import keras  # Keras 2.1.2 and TF-GPU 1.9.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras import optimizers

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

def neural_model():
    model = Sequential()
    
    model.add(Dense(32, input_shape=(3,),activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
#    
#    model.add(Dense(64, activation='sigmoid'))
#    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    # torque entre -2 et 2 donc faire -0.5 puis *4
    
    learning_rate = 0.001
    opt = keras.optimizers.adam(lr=learning_rate, decay=0.1)
    
    model.compile(loss='mean_squared_error',
                  optimizer=opt)
    
    return model

#-(theta^2 + 0.1*theta_dt^2 + 0.001*action^2) REWARD


def play_game(reward_average=1,model=None,step=199,hm_games=1000,random_=True):
    env = gym.make('Pendulum-v0').env
    data_select=[]
    compteur=0
    for i in range(1,hm_games):
        reward_tot=0
        prev_obs=env.reset()
        prev_obs=np.array(prev_obs,dtype='float32')
        prev_obs=np.reshape(prev_obs,(3,1))
        if model==None:
            model=neural_model()
        data=[]
       
        
#        env.render()
        action=model.predict(prev_obs.reshape(-1,len(prev_obs)))
        obs,reward,done,info=env.step(action)
        
        data.append([prev_obs,action])
        prev_obs=obs
        prev_reward=float(reward)
        for _ in range(step):
#            env.render()
# =============================================================================
#             if random_==True:
#                 random_value=random.randrange(0,10)
#             else:
#                 random_value=0
#             if random_value<=7:
#                 action = model.predict(prev_obs.reshape(-1,len(prev_obs)))
#                 action=4*action-2 
# 
#             else:
#                 action = env.action_space.sample()
# =============================================================================
#            print('action=',action)
            action=model.predict(prev_obs.reshape(-1,len(prev_obs)))
            action=add_gauss_to_action(action)
            
            obs, reward, done, info = env.step(action)
#            print('obs={} , reward={} , done={} , info= {}'.format(obs,reward,done,info))

                
            data.append([prev_obs,action])

            prev_obs=obs
            prev_reward=float(reward)
            reward_tot+=prev_reward
            if done:
                break

        if reward_average==1:
            reward_average=reward_tot
            data_select.append(data)
            compteur=1
        elif reward_average<=reward_tot:
            
            reward_average=(compteur*reward_average+reward_tot)/(compteur+1)
            compteur+=1
            data_select.append(data)
#    X,Y=traite_data(data_select)
#    model=train_model(model,X,Y,hm_epochs=10,batch=128)
        
    env.close()
#    return data,model
    return data_select,reward_average

def traite_data(data):
    data=np.reshape(data,(np.shape(data)[0]*np.shape(data)[1],-1))
    X=np.array( [ i[0]       for i in data ] )
    X=np.reshape(X,(np.shape(X)[0],-1))
    Y=np.array( [ (i[1]+2)/4 for i in data ] )
    Y=np.reshape(Y,(np.shape(Y)[0],-1))

    return X,Y

def train_model(model,X,Y,hm_epochs=10,batch=32):
    
    model.fit(x=X,y=Y,
              epochs=hm_epochs,
              batch_size=batch,
              )
    
    return model
    
def add_gauss_to_action(action):
    factor=0.8
    scale_=0.3
    a=random.uniform(0, 1)
    if a<factor:
        number=np.random.normal(loc=action[0][0],scale=scale_)
    else:
        number=np.random.normal(loc=-action[0][0],scale=scale_)
    if number<-2:
        number=-2
    if number>2:
        number=2
    action[0][0]=number
    
    return action
    
    
    
    
def render_game(model,step=199):
    env = gym.make('Pendulum-v0').env
    reward_tot=0
    prev_obs=env.reset()
    prev_obs=np.array(prev_obs,dtype='float32')
    prev_obs=np.reshape(prev_obs,(3,1))
    env.render()
    action=model.predict(prev_obs.reshape(-1,len(prev_obs)))
    obs,reward,done,info=env.step(action)
    prev_obs=obs
    reward_tot+=reward
        
    for _ in range(step):
        env.render()
        action = model.predict(prev_obs.reshape(-1,len(prev_obs)))
        action=4*action-2
        action=add_gauss_to_action(action)
        obs, reward, done, info = env.step(action)

        prev_obs=obs
        reward_tot+=reward
        if done:
            break
        
        
    print(reward_tot)
    input()
    env.close()
    
    
    

    
#model = neural_model()
# =============================================================================
# reward_average=1
# for i in range(100):
#     data,reward_average=play_game(reward_average=reward_average,hm_games=1000)
#     print(reward_average)
#     if len(data)>0:
#         X,Y=traite_data(data)
#         model=train_model(model,X,Y,hm_epochs=10,batch=128)
#         print(reward_average)
#         
# #model.save('pendulum_base_model.model')
# =============================================================================
model=keras.models.load_model('pendulum_base_model.model')

def learn_specific(reward_average=1,model=model,range_=[-0.9,-1,0.5],hm_games=1000,step=199):
    env = gym.make('Pendulum-v0').env
    data_select=[]
    compteur=0
    games=0
    compteur1=0
    while games<hm_games:
        prev_obs=env.reset()
        if prev_obs[0]<range_[0] and prev_obs[0]>range_[1] and abs(prev_obs[2])<range_[2]:
            games+=1
            reward_tot=0
            prev_obs=np.array(prev_obs,dtype='float32')
            prev_obs=np.reshape(prev_obs,(3,1))
            action=model.predict(prev_obs.reshape(-1,len(prev_obs)))
            obs,reward,done,info=env.step(action)
            data=[]
            data.append([prev_obs,action])
            prev_obs=obs
            prev_reward=float(reward)
            for _ in range(step):
                action=model.predict(prev_obs.reshape(-1,len(prev_obs)))
                action=add_gauss_to_action(action)
            
                obs, reward, done, info = env.step(action)
               
                data.append([prev_obs,action])
                
                prev_obs=obs
                prev_reward=float(reward)
                reward_tot+=prev_reward
                if done:
                    break
            if reward_average==1:
                reward_average=reward_tot
                data_select.append(data)
                compteur=1
            elif reward_average<=reward_tot:
            
                reward_average=(compteur*reward_average+reward_tot)/(compteur+1)
                compteur+=1
                data_select.append(data)
        else:
            compteur1+=1
            if compteur1%1000==0:
                print(compteur1)
                print(games)
            
    return data_select,reward_average,compteur1


opt1 = keras.optimizers.adam(lr=0.1, decay=0.0001)
opt2 = keras.optimizers.adam(lr=0.01, decay=0.0001)
opt3 = keras.optimizers.adam(lr=0.001, decay=0.0001)
opt4 = keras.optimizers.adam(lr=0.0001, decay=0.0001)
reward_average=1
for i in range(100):
    data,reward_average,compteur1=learn_specific(reward_average=reward_average,range_=[-0.98,-1,0.2])
    print(reward_average)
    if len(data)>0:
        X,Y=traite_data(data)
        model.compile(loss='mean_squared_error',optimizer=opt1)
        model=train_model(model,X,Y,hm_epochs=10,batch=64)
        model.compile(loss='mean_squared_error',optimizer=opt2)
        model=train_model(model,X,Y,hm_epochs=10,batch=64)
        model.compile(loss='mean_squared_error',optimizer=opt3)
        model=train_model(model,X,Y,hm_epochs=10,batch=64)
        model.compile(loss='mean_squared_error',optimizer=opt4)
        model=train_model(model,X,Y,hm_epochs=10,batch=64)
        print(reward_average)

#model.save('pendulum_base_model.model')