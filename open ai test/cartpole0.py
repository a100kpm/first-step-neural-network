import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

import tensorflow as tf
tf.reset_default_graph()
#learing rate
LR = 1e-3
env = gym.make('CartPole-v0').env
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000



def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            #render pour voir (attention ralentit obviously)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done: break
    print('waiting for input to stop')
    input()
    env.close()
        
#some_random_games_first()
            

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)
            
            if len(prev_observation) > 0:
                game_memory.append([prev_observation,action])
            prev_observation = observation
            score += reward
            if done:
                break
            
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                    
                training_data.append([data[0],output])
                
        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved01.npy', training_data_save)
    
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data

tra=initial_population()
model = train_model(tra)

# =============================================================================
# https://www.youtube.com/watch?v=G-KvpNGudLw&index=61&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
# Rappels d'explication !
# =============================================================================
def neural_network_model(input_size):
    network = input_data(shape = [None, input_size, 1], name='input')
    
    keep_rate=0.8
    network = fully_connected(network,128,activation='relu')
    network = dropout(network, keep_rate)
    
    network = fully_connected(network,256,activation='relu')
    network = dropout(network, keep_rate)
    
    network = fully_connected(network,512,activation='relu')
    network = dropout(network, keep_rate)
    
    network = fully_connected(network,256,activation='relu')
    network = dropout(network, keep_rate)
    
    network = fully_connected(network,128,activation='relu')
    network = dropout(network, keep_rate)
#    
#    le "2" correspond au nombre d'action possible, donc au nombre
#    d'élément de sortie
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate = LR,
                         loss='categorical_crossentropy',name='targets')
    model = tflearn.DNN(network,tensorboard_dir='log')
    
    return model

def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y= [i[1] for i in training_data]
    
    if not model:
        model = neural_network_model(input_size = len(X[0]))
    
    model.fit({'input':X},{'targets':y},n_epoch=3, snapshot_step=500, show_metric=True,
              run_id='openaistuff')
    
    return model

            

def reinforcement_l(model,nbr=10,goal_steps=500,score_requirement=200,prev_data='saved.npy'):
#    load_data=np.load('saved.npy')
    load_data=np.load(prev_data)
    training_data_prev=load_data.tolist()
#    if not model:
#        model = neural_network_model(input_size = 4)
#        model.load('195.model')
    training_data=[]
    scores=[]
    accepted_scores = []
    
    for i in range(nbr):
        if i%10==0:
            print(i)
        score = 0
        game_memory = []
        prev_obs = []        
        for _ in range(goal_steps):
            if len(prev_obs) == 0:
                action = random.randrange(0,2)
            else:
                action =np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
            observation, reward, done, info = env.step(action)
            if len(prev_obs) > 0:
                game_memory.append([prev_obs,action])
            prev_obs = observation
            score += reward
            if done:
                break
            
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0,1]
                elif data[1] == 0:
                    output = [1,0]
                    
                training_data.append([data[0],output])
        env.reset()
        scores.append(score)  
    print('Average accepted score:', mean(accepted_scores))
    print('Median accepted score:', median(accepted_scores))
    print('---------------------------------')
    print('Starting to learn for recent data')
    print('---------------------------------')
#    input()
    if nbr<5000:
        data=training_data_prev+training_data
    else:
        data=training_data
#    print('lol')
#    input()
    data=training_data
#    print('mdr')
#    input()
    new_model=train_model(data, model)


            
    return new_model,data

            


# =============================================================================
# =============================================================================
# # model = neural_network_model(input_size = 4)
# # #model.load('195.model')
# # model.load('test1.model')
# =============================================================================
# =============================================================================
        
        
    # =============================================================================
# training_data = initial_population()
# model = train_model(training_data)
# =============================================================================

# RAPPEL!!!! IL FAUT CHARGER UN MODELE MÊME VIDE AVANT DE LOAD
#model.save('195.model')
#model=model.load(195.model)
# RAPPEL!!!! IL FAUT CHARGER UN MODELE MÊME VIDE AVANT DE LOAD

# =============================================================================
# =============================================================================
# # scores=[]
# # choices =[]
# # 
# # for each_game in range(10):
# #     score = 0
# #     game_memory = []
# #     prev_obs = []
# #     env.reset()
# #     for _ in range(goal_steps):
# # #        env.render()
# #         if len(prev_obs) == 0:
# #             action = random.randrange(0,2)
# #         else:
# # #            np.argmax permet de changer le "[0,1] en 1 ou [1,0] en 0
# # #            (hotmax = true)
# #             action =np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])
# #         
# #         choices.append(action)
# #         new_observation, reward, done ,info = env.step(action)
# #         prev_obs = new_observation
# #         game_memory.append([new_observation, action])
# # #        pour  retrain ^^
# #         score += reward
# # #        print(score)
# #         if done:
# #             break
# #     scores.append(score)
# #     
# # print('Average Score', sum(scores)/len(scores))
# # print('Choice 1: {}, Choice 2: {}'.format(choices.count(1)/len(choices),
# #       choices.count(0)/len(choices)))
# # 
# # env.close()
# =============================================================================
# =============================================================================

#328 -->195
#341 -->solo
#348 -->first0
#407 -->test1.model // saved2.npy


