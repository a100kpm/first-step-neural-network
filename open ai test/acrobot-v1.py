# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:49:27 2018

@author: iannis
"""
# =============================================================================
# https://gym.openai.com/evaluations/eval_c6LRzrkyTKyHgl7k9SCuVA/
# =============================================================================

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

#env = gym.make('Acrobot-v1').env


def play_random_games(step=1500,hm_games=1,pause_after_game=True):
    env = gym.make('Acrobot-v1')
    for game in range(hm_games):
        env.reset()
        for t in range(step):
            env.render()
            action = env.action_space.sample()
            
            observation, reward, done, info = env.step(action)
            input()
            print(observation)
            print('-------------')
            print(angle_tot(observation[0:4]))
 
        if pause_after_game==True:
            print('game {} out of {}'.format(game+1,hm_games))
            print('press enter to continue')
            input()
    env.close()
    env = gym.make('Acrobot-v1')

def angle_tot(observation):
    if observation[1]>=0:
        angle1=arccos(observation[0])
    else:
        angle1=-arccos(observation[0])
    if observation[3]>=0:
        angle2=arccos(observation[2])
    else:
        angle2=-arccos(observation[2])
    
    return angle1+angle2
        
    
    

def creation_data_set_init(step=200,hm_games=1000,saved=False,score=0.25):
    data=[]
    env = gym.make('Acrobot-v1').env
    for game in range(hm_games):
        score_game=1
        env.reset()
        prev_obs=[]
        memory=[]
        for t in range(step):
            action= random.randrange(0,3)
            observation, reward, done, info = env.step(action)
            if observation[0]<score_game:
                score_game=observation[0]
            if len(prev_obs)>0:
                memory.append([prev_obs,action])
            prev_obs=observation
        
        if score_game<score:
            for memory_set in memory:
                if memory_set[1]==0:
                    memory_set[1]=[1,0,0]
                if memory_set[1]==1:
                    memory_set[1]=[0,1,0]
                if memory_set[1]==2:
                   memory_set[1]=[0,0,1]
                data.append(memory_set)
            
    env.close()
    if saved==True:
        np.save('saved_acrobot.npy', data)
    return data


def neural_model(input_size=6,keep_rate=0.8,LR=1e-3,nbr_layer=5):
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

def train_model(training_data,model=False,LR=1e-3,hm_epoch=3,saved=False):    
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y= [i[1] for i in training_data]    
    
    if not model:
        model = neural_model(input_size = len(X[0]),LR=LR)
    
    model.fit({'input':X},{'targets':y},n_epoch=hm_epoch, show_metric=True,
              run_id='DriveDriveDrive')
    if saved:
        model.save('acrobotV1.model')
        
    return model


#data=np.load('saved_acrobot.npy')
#model=train_model(data)
#model.save('acrobotV1.model')
#model = neural_network_model()
#model.load('acrobotV1.model')


def play_game(model,step=200,hm_games=1,render=True):
    tf.reset_default_graph()
    env = gym.make('Acrobot-v1').env    
    compteur=0
    meanss=[]
    minimum=200
    for games in range(hm_games):
        env.reset()
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
        if i<minimum:
            minimum=i
            
        compteur+=i
        meanss.append(i)
    compteur=compteur/hm_games
    print(compteur)
    env.close()
    return minimum,median(meanss)



#data=creation_data_set_init(hm_games=5000)
#print(np.shape(data)[0]/199)
#model=train_model(data)
#play_game(model,hm_games=100,render=False)



def creation_data_set_medianed(model,mediane,hm_games=1000,step=200):
    tf.reset_default_graph()
    
    data=[]
    env = gym.make('Acrobot-v1').env
    for game in range(hm_games):
        score_game=1
        env.reset()
        memory=[]
        action = random.randrange(0,3)
        prev_obs, reward, done, info = env.step(action)
        for _ in range(1,step):
            score_game+=1
            action= np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            observation, reward, done, info = env.step(action)
            
            
            memory.append([prev_obs,action])
            prev_obs=observation
                        
            if done:
                break
        
        if score_game<mediane:
            for memory_set in memory:
                if memory_set[1]==0:
                    memory_set[1]=[1,0,0]
                if memory_set[1]==1:
                    memory_set[1]=[0,1,0]
                if memory_set[1]==2:
                   memory_set[1]=[0,0,1]
                data.append(memory_set)
            
    env.close()
    
    return data

def creation_data_set_medianed2(model,mediane,longueur_data=60000,step=200,max_games=10000):
    tf.reset_default_graph()
    game_count=0
    data=[]
    env=gym.make('Acrobot-v1').env
    while np.shape(data)[0]<=longueur_data and game_count<max_games:
        game_count+=1
        score_game=1
        env.reset()
        memory=[]
        action=random.randrange(0,3)
        prev_obs,reward,done,info=env.step(action)
        for _ in range(1,step):
            score_game+=1
            action= np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            observation, reward, done, info = env.step(action)
            
            
            memory.append([prev_obs,action])
            prev_obs=observation
                        
            if done:
                break
        
        if score_game<mediane:
            for memory_set in memory:
                if memory_set[1]==0:
                    memory_set[1]=[1,0,0]
                if memory_set[1]==1:
                    memory_set[1]=[0,1,0]
                if memory_set[1]==2:
                   memory_set[1]=[0,0,1]
                data.append(memory_set)
    env.close()
    
    
    return data,game_count

def creation_data_set_medianed3(model,mediane,longueur_data=60000,step=200,max_games=10000):
    tf.reset_default_graph()
    game_count=0
    data=[]
    env=gym.make('Acrobot-v1').env
    while np.shape(data)[0]<=longueur_data and game_count<max_games:
        game_count+=1
        score_game=1
        env.reset()
        memory=[]
        action=random.randrange(0,3)
        prev_obs,reward,done,info=env.step(action)
        for _ in range(1,step):
            score_game+=1
            action= np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            observation, reward, done, info = env.step(action)
            
            
            memory.append([prev_obs,action])
            prev_obs=observation
                        
            if done:
                break
        
        if score_game<mediane:
            repete=floor(sqrt(5*(mediane-score_game)))
            for memory_set in memory:
                if memory_set[1]==0:
                    memory_set[1]=[1,0,0]
                if memory_set[1]==1:
                    memory_set[1]=[0,1,0]
                if memory_set[1]==2:
                   memory_set[1]=[0,0,1]
                for _ in range(repete):   
                    data.append(memory_set)
    env.close()
    
    
    return data,game_count
            

def creation_data_set_medianed4(model,mediane,factor,longueur_data=60000,step=200,max_games=10000):
    tf.reset_default_graph()
    game_count=0
    data=[]
    env=gym.make('Acrobot-v1').env
    while np.shape(data)[0]<=longueur_data and game_count<max_games:
        game_count+=1
        score_game=1
        env.reset()
        memory=[]
        action=random.randrange(0,3)
        prev_obs,reward,done,info=env.step(action)
        for _ in range(1,step):
            score_game+=1
            action= np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            observation, reward, done, info = env.step(action)
            
            
            memory.append([prev_obs,action])
            prev_obs=observation
                        
            if done:
                break
        
        if score_game<mediane:
            repete=floor(sqrt(factor*mediane-score_game))
            for memory_set in memory:
                if memory_set[1]==0:
                    memory_set[1]=[1,0,0]
                if memory_set[1]==1:
                    memory_set[1]=[0,1,0]
                if memory_set[1]==2:
                   memory_set[1]=[0,0,1]
                for _ in range(repete):   
                    data.append(memory_set)
    env.close()
    
    
    return data,game_count
    
def reinforcement_learn_acrobot_several(model,number,max_games=10000):
    minimum,mediane=play_game(model,hm_games=100,render=False)
    mediane=(5*mediane+3*minimum)/8
    stock_med=[]
    for game_nbr in range(number):
        data,game_count=creation_data_set_medianed2(model,mediane,max_games=max_games)
#        data=creation_data_set_medianed(model,mediane,hm_games=1000)
        if game_count>=0.99*max_games:
            print('trop dur')
            break
        model.save('gg.model')  
        tf.reset_default_graph()
        model=neural_model()

        model.load('gg.model')
        model=train_model(data,model=model,hm_epoch=3)
        print(game_count)
        play_game(model,hm_games=10)
        minimum,mediane=play_game(model,hm_games=100,render=False)
        print('mediane={} and minimum= {} in game {}'.format(mediane,minimum,game_nbr+1))
        stock_med.append(mediane)
        mediane=(5*mediane+3*minimum)/8
    print(stock_med)
    return model

def reinforcement_learn_acrobot_several_val(model,number,max_games=10000,factor=1.1):
    minimum,mediane=play_game(model,hm_games=100,render=False)
    mediane=floor(minimum*factor)
    stock_med=[]
    for game_nbr in range(number):
        data,game_count=creation_data_set_medianed4(model,mediane,factor,max_games=max_games)
#        data=creation_data_set_medianed(model,mediane,hm_games=1000)
        if game_count>=0.99*max_games:
            print('trop dur')
            break
        
        model=train_model(data,hm_epoch=3)
        print(game_count)
        play_game(model,hm_games=10)
        minimum,mediane=play_game(model,hm_games=100,render=False)
        print('mediane={} and minimum= {} in game {}'.format(mediane,minimum,game_nbr+1))
        stock_med.append(mediane)
        mediane=floor(minimum*factor)
    print(stock_med)
    return model

        

def reinforcement_early(model,start_step=0,end_step=200,hm_games=100,score=0.8):
    data=[]
    env = gym.make('Acrobot-v1').env
#    best_score=1
    for game in range(hm_games):
        score_game=1
        env.reset()
        prev_obs=[]
        memory=[]
        for t in range(end_step):
            if len(prev_obs)==0:
                action= random.randrange(0,3)
            else:
                action= np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            observation, reward, done, info = env.step(action)
#            if observation[0]<best_score:
#                best_score=observation[0]
            if observation[0]<score:
                score_game=observation[0]
            if len(prev_obs)>0:
                memory.append([prev_obs,action])
            prev_obs=observation
        
        if score_game<score:
            for memory_set in memory:
                if memory_set[1]==0:
                    memory_set[1]=[1,0,0]
                if memory_set[1]==1:
                    memory_set[1]=[0,1,0]
                if memory_set[1]==2:
                   memory_set[1]=[0,0,1]
                data.append(memory_set)
            
    env.close()
#    return best_score   
    return data
    
# =============================================================================
# 
# model.save('gg.model')  
# tf.reset_default_graph()
# model=neural_model(LR=1e-4)
# #data=np.load('saved_acrobot.npy')=
# #model.load('acrobotV1_31_model')
# #model.load('test1.model')
# #model.load('test2.model') seems cann't go better with only a 2 hidden layer
# #model.load('teest_5hidlayer0.model')
# #model.load('test_alzeimer_acrobot.model')
# # keep rate = 0.3 pour celui la
# model.load('gg.model')
# #data=np.load('gg.npy')
# model=train_model(data,model,hm_epoch=500)
# =============================================================================

env = gym.make('Acrobot-v1')
#env.close()

        


def creation_early(step=20,hm_games=10000,score=0.7):
    data=[]
    for game in range(hm_games):
        score_game=1
        env.reset()
        prev_obs=[]
        memory=[]
        for t in range(step):
            action= random.randrange(0,3)
            observation, reward, done, info = env.step(action)
            if observation[0]<score_game:
                score_game=observation[0]
            if len(prev_obs)>0:
                memory.append([prev_obs,action])
            prev_obs=observation
        
        if score_game<score:
            for memory_set in memory:
                if memory_set[1]==0:
                    memory_set[1]=[1,0,0]
                if memory_set[1]==1:
                    memory_set[1]=[0,1,0]
                if memory_set[1]==2:
                   memory_set[1]=[0,0,1]
                data.append(memory_set)
            
    return data
            
        
def get_last_val(model,hm_games=1000,step=200,last=20):
    tf.reset_default_graph()
    
    data=[]
    env = gym.make('Acrobot-v1').env
    for game in range(hm_games):
        score_game=1
        env.reset()
        memory=[]
        action = random.randrange(0,3)
        prev_obs, reward, done, info = env.step(action)
        for _ in range(1,step):
            score_game+=1
            action= np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            observation, reward, done, info = env.step(action)
            
            
            memory.append([prev_obs,action])
            prev_obs=observation
                        
            if done:
                break
        memory=memory[len(memory)-last:]
    
        for memory_set in memory:
            if memory_set[1]==0:
                memory_set[1]=[1,0,0]
            if memory_set[1]==1:
                memory_set[1]=[0,1,0]
            if memory_set[1]==2:
               memory_set[1]=[0,0,1]
            data.append(memory_set)
            
    env.close()
    
    return data    


def human_player(step=200):
    data=[]
    score=0
    env.reset()
    prev_obs=[]
    memory=[]
    for t in range(step):
        score+=1
        env.render()
        
        a=input()
        if a=='4':
            action=0
        if a=='6':
            action=2
        else:
            action=1
            
        observation, reward, done, info = env.step(action)
        print(observation)
        if len(prev_obs)>0:
            memory.append([prev_obs,action])
        prev_obs=observation
        
        if done:
            break
        
        env.render()
        

