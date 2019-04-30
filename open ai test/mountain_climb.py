# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 17:39:04 2018

@author: iannis
"""

import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


import tensorflow as tf
tf.reset_default_graph()

env = gym.make('MountainCar-v0').env


def some_random_games_first(n=1000):
    for episode in range(5):
        env.reset()
        maxx=-2
        maxy=0
        for t in range(n):
            env.render()
            #render pour voir (attention ralentit obviously)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if observation[0]>maxx:
                maxx=observation[0]
            if observation[1]>maxy:
                maxy=observation[1]
            if done: break
        print('waiting for input to stop')
        print("t=",t)
        print('maxx=',maxx)
        print('maxy=',maxy)
    env.close()
    
# 0 = freine, 1 = rien, 2 = accélère


            
def creation_initial_games_data2(n=30000,step=200,score_requis=0.025):
    data_init=[]
    for game in range(n):
        maxy=0
        prev_obs=[]
        memory=[]
        for t in range(step):
            action= random.randrange(0,3)
            observation, reward, done, info = env.step(action)
            
            if len(prev_obs)>0:
                memory.append([prev_obs,action])
            prev_obs=observation
            if maxy<observation[1]:
                maxy=observation[1]
        if maxy>score_requis:
            for data in memory:
                if data[1]==0:
                    output=[0,0,1]
                if data[1]==1:
                    output=[0,1,0]
                if data[1]==2:
                    output=[1,0,0]
                data_init.append([data[0],output])
        env.reset()
    return data_init
            
def creation_initial_games_data3(n=1000,step=200):
    data_init=[]
    for game in range(n):
        prev_obs=[]
        for t in range(step):
            action= random.randrange(0,3)
            observation, reward, done, info = env.step(action)
            
            if len(prev_obs)>0:
                if observation[1]>0 and action ==2:
                    data_init.append([prev_obs,action])
                elif observation[1]<0 and action ==0:
                    data_init.append([prev_obs,action])
            prev_obs=observation
        env.reset()
    for data in data_init:
        if data[1]==0:
            data[1]=[0,0,1]
        if data[1]==1:
            data[1]=[0,1,0]
        if data[1]==2:
            data[1]=[1,0,0]

    return data_init
                
#np.save('saved_mountain_0.npy', nouveau_data)            
            
def neural_model(input_size,keep_rate=0.8,LR=1e-3):
    network = input_data(shape = [None, input_size, 1], name='input')
    
    network = fully_connected(network,64,activation='relu')
    network = dropout(network, keep_rate)
    
#    network = fully_connected(network,128,activation='relu')
#    network = dropout(network, keep_rate)
#    
#    network = fully_connected(network,256,activation='relu')
#    network = dropout(network, keep_rate)
#    
#    network = fully_connected(network,512,activation='relu')
#    network = dropout(network, keep_rate)
#    
#    network = fully_connected(network,256,activation='relu')
#    network = dropout(network, keep_rate)
#    
#    network = fully_connected(network,128,activation='relu')
#    network = dropout(network, keep_rate)
#    

    
    
    network = fully_connected(network,3,activation='softmax')
    network = regression(network,learning_rate=LR,optimizer='adam',
                         name='targets',loss='categorical_crossentropy')
    
    model = tflearn.DNN(network,tensorboard_verbose=0,tensorboard_dir='log_climb')
    
    return model

def train_model(training_data,model=False,LR=1e-3,hm_epoch=5):
    
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y= [i[1] for i in training_data]    
    
    if not model:
        model = neural_model(input_size = len(X[0]),LR=LR)
    
    model.fit({'input':X},{'targets':y},n_epoch=hm_epoch, show_metric=True,
              run_id='DriveDriveDrive')
    
    return model

    
#train_data_saved=np.load('saved_mountain_1.npy')

#lenn=len(train_data_saved)
#lena=int(0.2*lenn)
#train_data_saved=train_data_saved[:lena]

#model.save('mountain01.model')
train_data_saved=creation_initial_games_data2()  
model = train_model(train_data_saved)
model.save('mountain01.model')


#env = gym.make('MountainCar-v0').env
min_i=200
conter = 0
for each_game in range(100):
    env.reset()
    prev_obs = []
    i=0
    for _ in range(200):
        i+=1
        env.render()
        if len(prev_obs) ==0:
            action = random.randrange(0,3)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            if action==2:
                action=0
            elif action==0:
                action=2
#            print('prev_obs=', prev_obs)
#            print('action=', action)
        obs, reward, done, info = env.step(action)
        prev_obs = obs
        if done:
            break
#    print('input')
#    input()
#    print(i)
    if i<min_i:
        min_i=i
    conter += i
print(conter)
env.close()
print('meilleur',min_i)