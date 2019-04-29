# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 16:09:16 2018

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
#nécessaire pour run plusieurs fois d'affilé le programme.

# =============================================================================
# env = gym.make('MountainCar-v0').env
# ouvre l'environnement du jeu mountain_car
# https://gym.openai.com/envs/MountainCar-v0/
# =============================================================================


def play_random_games(step=1500,hm_games=1,pause_after_game=True):
    env = gym.make('MountainCar-v0').env
    for game in range(hm_games):
        for t in range(step):
            env.render()
            action = env.action_space.sample()
            env.step(action)
        if pause_after_game==True:
            print('game {} out of {}'.format(game,hm_games))
            print('press enter to continue')
            input()
        env.reset()
    env.close()
        
def creation_data_set(step=200,hm_games=1000,saved=False):
    data=[]
    env = gym.make('MountainCar-v0').env
    for game in range(hm_games):
        prev_obs=[]
        for t in range(step):
            action= random.randrange(0,3)
            observation, reward, done, info = env.step(action)
            if len(prev_obs)>0:
                if ( observation[1]*(action-1) )>0:
                    data.append([prev_obs,action])
            prev_obs=observation
        env.reset()
    for data_set in data:
        if data_set[1]==0:
            data_set[1]=[1,0,0]
        if data_set[1]==1:
            data_set[1]=[0,1,0]
        if data_set[1]==2:
            data_set[1]=[0,0,1]
            
    env.close()
    if saved==True:
        np.save('saved_mountain.npy', data)
    return data


def neural_model(input_size=2,keep_rate=0.8,LR=1e-3,nbr_layer=1):
    LIST=[64,128,256,512,256,128]
    
    network = input_data(shape = [None, input_size, 1], name='input')
    
    for i in range(nbr_layer):        
        network = fully_connected(network,LIST[i],activation='relu')
        network = dropout(network, keep_rate)
        
    network = fully_connected(network,3,activation='softmax')
    network = regression(network,learning_rate=LR,optimizer='adam',
                         name='targets',loss='categorical_crossentropy')
    
    model = tflearn.DNN(network,tensorboard_verbose=0,tensorboard_dir='log_climb')
    
    return model
        
def train_model(training_data,model=False,LR=1e-3,hm_epoch=5,saved=False):    
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y= [i[1] for i in training_data]    
    
    if not model:
        model = neural_model(input_size = len(X[0]),LR=LR)
    
    model.fit({'input':X},{'targets':y},n_epoch=hm_epoch, show_metric=True,
              run_id='DriveDriveDrive')
    if saved:
        model.save('mountain.model')
        
    return model
    
def play_game(model=False,step=200,hm_games=1,render=True):
    tf.reset_default_graph()
    env = gym.make('MountainCar-v0').env    
    compteur=0
    if not model:
        model = neural_model(2,1e-3)
        
    for games in range(hm_games):
        prev_obs = []
        i=0
        for _ in range(step):
            i+=1
            if render==True:
                env.render()
            if len(prev_obs)==0:
                action=random.randrange(0,3)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
                
            obs, reward, done, info = env.step(action)
            prev_obs = obs
            
            if done:
                break
        env.reset()    
        compteur+=i
    
    compteur=compteur/hm_games
    print(compteur)
    env.close()
    
    
# =============================================================================
# def main():
#     data=creation_data_set()
#     model=train_model(data)
#     play_game(model=model,hm_games=5)
#   
# if __name__== "__main__":
#     main()  
# =============================================================================
    
    
data=creation_data_set()
model=train_model(data)
play_game(model=model,hm_games=1)


    
    
    
    
    
    
    