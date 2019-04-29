# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 16:55:06 2018

@author: iannis
"""

import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from statistics import median


tf.reset_default_graph()


def creation_data_set_init(step=200,hm_games=1000,saved=False,score=0.25):
    data=[]
    env = gym.make('Acrobot-v1').env
    for game in range(hm_games):
        score_game=1
        env.reset()
        prev_obs=[]
        prev_obs2=[]
        memory=[]
        for t in range(step):
            action= random.randrange(0,3)
            observation, reward, done, info = env.step(action)
            if observation[0]<score_game:
                score_game=observation[0]
            if len(prev_obs2)>0:
                memory.append([np.concatenate((prev_obs,prev_obs2)),action])
            prev_obs2=prev_obs
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



def neural_model(input_size=12,keep_rate=0.8,LR=1e-3,nbr_layer=5):
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






def play_game(model,step=200,hm_games=1,render=True):
    tf.reset_default_graph()
    env = gym.make('Acrobot-v1').env    
    compteur=0
    meanss=[]
    minimum=200
    for games in range(hm_games):
#        env.reset()
#        prev_obs = []
#        prev_obs2 =[]
        prev_obs=env.reset()
        prev_obs2=prev_obs
        i=0
        for _ in range(step):
            i+=1
            if render==True:
                env.render()
            if len(prev_obs2)==0:
                action=random.randrange(0,3)
            else:
                knowledge=np.concatenate((prev_obs,prev_obs2))
                action = np.argmax(model.predict(knowledge.reshape(-1,len(knowledge),1))[0])
                
            obs, reward, done, info = env.step(action)
            prev_obs2= prev_obs
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



def creation_data_set_medianed(model,mediane,hm_games=1000,step=200):
    tf.reset_default_graph()
    
    data=[]
    env = gym.make('Acrobot-v1').env
    for game in range(hm_games):
        score_game=0
        prev_obs=env.reset()
        prev_obs2=env.reset()
        memory=[]
        for _ in range(step):
            score_game+=1
            knowledge=np.concatenate((prev_obs,prev_obs2))
            action= np.argmax(model.predict(knowledge.reshape(-1,len(knowledge),1))[0])
            observation, reward, done, info = env.step(action)
            
            
            memory.append([knowledge,action])
            prev_obs2=prev_obs
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


def reinforcement_learn_acrobot_several(model,number,max_games=3000,mediane=77,minimum=63):
#    minimum,mediane=play_game(model,hm_games=200,render=False)
    mediane_val=(5*mediane+3*minimum)/8
    stock_med=[]
#    
    for game_nbr in range(number):
        nbr_epoch=0
        data=creation_data_set_medianed(model,mediane_val,hm_games=max_games)
#        data=creation_data_set_medianed(model,mediane,hm_games=1000)
        if np.shape(data)[0]<60000:
            nbr_epoch=120000/np.shape(data)[0]
            nbr_epoch=int(nbr_epoch)
            
        model.save('acro_vitesse.model')  
        tf.reset_default_graph()
        model=neural_model()

        model.load('acro_vitesse.model')
        model=train_model(data,model=model,hm_epoch=3+nbr_epoch)
        play_game(model,hm_games=10)
        minimum,mediane_n=play_game(model,hm_games=200,render=False)
        print('mediane={} , new_mediane={} and minimum= {} in game {}'.format(mediane,mediane_n,minimum,game_nbr+1))
        
        if mediane_n<mediane:
            
            stock_med.append(mediane_n)
            mediane_val=(5*mediane_n+3*minimum)/8
            model.save('acro_vitesse.model')
            mediane=mediane_n
            
        else:
            stock_med.append(-1)
            tf.reset_default_graph()
            model=neural_model()
            model.load('acro_vitesse.model')
            
    print(stock_med)
    return model,mediane


def reinforcement_learn_acrobot_several_mean(model,number,max_games=3000,mean=90,minimum=63):
#    minimum,mediane=play_game(model,hm_games=200,render=False)
    moy=mean
    stock_med=[]
#    
    for game_nbr in range(number):
        nbr_epoch=0
        data=creation_data_set_moyened(model,moy,hm_games=max_games)
#        data=creation_data_set_medianed(model,mediane,hm_games=1000)
        if np.shape(data)[0]<60000:
            nbr_epoch=120000/np.shape(data)[0]
            nbr_epoch=int(nbr_epoch)
            
        model.save('acro_vitesse.model')  
        tf.reset_default_graph()
        model=neural_model()

        model.load('acro_vitesse.model')
        model=train_model(data,model=model,hm_epoch=3+nbr_epoch)
        play_game2(model,hm_games=10)
        minimum,moy_n=play_game2(model,hm_games=100,render=False)
        print('moy={} , new_m={} and minimum= {} in game {}'.format(moy,moy_n,minimum,game_nbr+1))
        
        if moy_n<moy:
            
            stock_med.append(moy_n)
            
            model.save('acro_vitesse.model')
            moy=moy_n
            
        else:
            stock_med.append(-1)
            tf.reset_default_graph()
            model=neural_model()
            model.load('acro_vitesse.model')
            
    print(stock_med)
    return model,moy




def creation_data_set_moyened(model,moy,hm_games=1000,step=200):
    tf.reset_default_graph()
    
    data=[]
    env = gym.make('Acrobot-v1').env
    for game in range(hm_games):
        score_game=0
        prev_obs=env.reset()
        prev_obs2=env.reset()
        memory=[]
        for _ in range(step):
            score_game+=1
            knowledge=np.concatenate((prev_obs,prev_obs2))
            action= np.argmax(model.predict(knowledge.reshape(-1,len(knowledge),1))[0])
            observation, reward, done, info = env.step(action)
            
            
            memory.append([knowledge,action])
            prev_obs2=prev_obs
            prev_obs=observation
                        
            if done:
                break
        
        if score_game<moy:
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


def play_game2(model,step=200,hm_games=1,render=True):
    tf.reset_default_graph()
    env = gym.make('Acrobot-v1').env    
    compteur=0
    meanss=[]
    minimum=200
    for games in range(hm_games):
#        env.reset()
#        prev_obs = []
#        prev_obs2 =[]
        prev_obs=env.reset()
        prev_obs2=prev_obs
        i=0
        for _ in range(step):
            i+=1
            if render==True:
                env.render()
            if len(prev_obs2)==0:
                action=random.randrange(0,3)
            else:
                knowledge=np.concatenate((prev_obs,prev_obs2))
                action = np.argmax(model.predict(knowledge.reshape(-1,len(knowledge),1))[0])
                
            obs, reward, done, info = env.step(action)
            prev_obs2= prev_obs
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
    return minimum,compteur










model,mediane=reinforcement_learn_acrobot_several(model,10,max_games=500,mediane=med)
model,mediane=reinforcement_learn_acrobot_several_mean(model,10,max_games=500,mean=moy)

play_game(model,hm_games=100,render=False)