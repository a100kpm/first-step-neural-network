import numpy as np
import csv
import math

import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,CuDNNLSTM,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

def preprocess_data_collected(collected,collected_y,factor):
#    mettre collected_y = '0' si l'on veut traiter les données a prédict !
    lenn=len(collected)
    lenn2=len(factor)
    data=[]
    data_temp=[]
    val=0
    for i in range(lenn):
        val_temp=int(collected[i][0].split('_')[0])
        if val==val_temp:
            new_to_add=[float(x) for x in collected[i][3:]]
            for j in range(lenn2):
                new_to_add[j]+=factor[j][0]
                new_to_add[j]=new_to_add[j]/factor[j][1]
                new_to_add[j]=new_to_add[j]*2-1
            data_temp.append(new_to_add)
        else:
#            
            if collected_y!='0':
#                
                data_temp.append(collected_y[val])
            val=val_temp
            data.append(data_temp)
            data_temp=[]
            new_to_add=[float(x) for x in collected[i][3:]]
            for j in range(lenn2):
                new_to_add[j]+=factor[j][0]
                new_to_add[j]=new_to_add[j]/factor[j][1]
                new_to_add[j]=new_to_add[j]*2-1
            data_temp.append(new_to_add)
#    
    if collected_y!='0':
#
        data_temp.append(collected_y[val])
    data.append(data_temp)
    
    return data

def find_proportion(target):
# ici le code a pour target la variable b
    List=[0]*9
    for i in target:
        if i=='fine_concrete':
            List[0]+=1
        elif i=='concrete':
            List[1]+=1
        elif i=='soft_tiles':
            List[2]+=1
        elif i=='tiled':
            List[3]+=1
        elif i=='soft_pvc':
            List[4]+=1
        elif i=='hard_tiles_large_space':
            List[5]+=1
        elif i=='carpet':
            List[6]+=1
        elif i=='hard_tiles':
            List[7]+=1
        elif i=='wood':
            List[8]+=1
        
    return List

def class_data(data,target):
    i=0
    lenn=len(target)
    fine_concrete=[]
    concrete=[]
    soft_tiles=[]
    tiled=[]
    soft_pvc=[]
    hard_tiles_large_space=[]
    carpet=[]
    hard_tiles=[]
    wood=[]
    
    for i in range(lenn):
        if target[i]=='fine_concrete':
            fine_concrete.append(data[i*128:i*128+127])
        elif target[i]=='concrete':
            concrete.append(data[i*128:i*128+127])
        elif target[i]=='soft_tiles':
            soft_tiles.append(data[i*128:i*128+127])
        elif target[i]=='tiled':
            tiled.append(data[i*128:i*128+127])
        elif target[i]=='soft_pvc':
            soft_pvc.append(data[i*128:i*128+127])
        elif target[i]=='hard_tiles_large_space':
            hard_tiles_large_space.append(data[i*128:i*128+127])
        elif target[i]=='carpet':
            carpet.append(data[i*128:i*128+127])
        elif target[i]=='hard_tiles':
            hard_tiles.append(data[i*128:i*128+127])
        elif target[i]=='wood':
            wood.append(data[i*128:i*128+127])

    return fine_concrete,concrete,soft_tiles,tiled,soft_pvc,hard_tiles_large_space,carpet,hard_tiles,wood

def create_equilibre_data(data_separe,List_proportion,new_size=100):
# data_separe est la list contenant tout les sous listes des différents types de sol
# fine_concrete,concrete,soft_tiles,tiled,soft_pvc,hard_tiles,carpet,hard_tiles,wood
# en provenance de class_data
#    taille_max=min(List_proportion)
    
    for i in data_separe:
        random.shuffle(i)
        
    fine_concrete,concrete,soft_tiles,tiled,soft_pvc,hard_tiles_large_space,carpet,hard_tiles,wood=[],[],[],[],[],[],[],[],[]
    
    for i in range(np.shape(data_separe)[0]):
        for j in range(len(data_separe[i])):
            for k in range(0,128-new_size):
                if i==0:
                    fine_concrete.append(data_separe[i][j][k:k+new_size])
                if i==1:
                    concrete.append(data_separe[i][j][k:k+new_size])
                if i==2:
                    soft_tiles.append(data_separe[i][j][k:k+new_size])
                if i==3:
                    tiled.append(data_separe[i][j][k:k+new_size])
                if i==4:
                    soft_pvc.append(data_separe[i][j][k:k+new_size])
                if i==5:
                    hard_tiles_large_space.append(data_separe[i][j][k:k+new_size])
                if i==6:
                    carpet.append(data_separe[i][j][k:k+new_size])
                if i==7:
                    hard_tiles.append(data_separe[i][j][k:k+new_size])
                if i==8:
                    wood.append(data_separe[i][j][k:k+new_size])
        
    random.shuffle(fine_concrete)
    random.shuffle(concrete)
    random.shuffle(soft_tiles)
    random.shuffle(tiled)
    random.shuffle(soft_pvc)
    random.shuffle(hard_tiles_large_space)
    random.shuffle(carpet)
    random.shuffle(hard_tiles)
    random.shuffle(wood)
    
    nombre_pris=min(np.shape(fine_concrete)[0],np.shape(concrete)[0],np.shape(soft_tiles)[0],np.shape(tiled)[0],np.shape(soft_pvc)[0],
                    np.shape(hard_tiles_large_space)[0],np.shape(carpet)[0],np.shape(hard_tiles)[0],np.shape(wood)[0])
    
    
    fine_concrete=fine_concrete[:nombre_pris]
    concrete=concrete[:nombre_pris]
    soft_tiles=soft_tiles[:nombre_pris]
    tiled=tiled[:nombre_pris]
    soft_pvc=soft_pvc[:nombre_pris]
    hard_tiles_large_space=hard_tiles_large_space[:nombre_pris]
    carpet=carpet[:nombre_pris]
    hard_tiles=hard_tiles[:nombre_pris]
    wood=wood[:nombre_pris]

    return fine_concrete,concrete,soft_tiles,tiled,soft_pvc,hard_tiles_large_space,carpet,hard_tiles,wood        
        

def collecte(path=r'C:\Users\ianni\Desktop\robot_career\career-con-2019\X_train.csv'):
    training_data=[]
    with open(path,'rt') as csvfile:
        a = csv.reader(csvfile,delimiter=' ')
        for row in a:
            row=row[0].split(',')
            data=[]
            lenn=len(row)
            for i in range(0,lenn):
                data.append(row[i])
            training_data.append(data)
    training_data=training_data[1:]

    return training_data

def collecte_y():
    training_data=[]
    with open(r'C:\Users\ianni\Desktop\robot_career\career-con-2019\y_train.csv','rt') as csvfile:
        a=csv.reader(csvfile,delimiter=' ')
        for row in a:
            row=row[0].split(',')
            training_data.append(row[2])
    
    training_data=training_data[1:]
    return training_data
    

def normalyze_factor_table(a,start_col=3):
    lenn=len(a[0])
    List=[[float(x),float(x)] for x in a[0][start_col:]]

    for i in a:
        for j in range(start_col,lenn):
            if float(i[j])<float(List[j-start_col][0]):
                List[j-start_col][0]=float(i[j])
            if float(i[j])>float(List[j-start_col][1]):
                List[j-start_col][1]=float(i[j])
    return List

def factor_to_normalyze(List_):
#    step1 add by factor[i][0]
#    step2 divide by factor[i][1]
#    step3 multiply by 2
#    step4 -1    
    lenn=np.shape(List_)[0]
    factor=List_.copy()
    for i in range(lenn):
        factor[i][0]=-List_[i][0]
        factor[i][1]+=factor[i][0]
        
    return factor

def normalyze_data(data_separe,factor):
    lenn=np.shape(data_separe)[0]
    lenn2=len(factor)
    for i in range(lenn):       
        for j in range(np.shape(data_separe[i])[0]):
            for k in range(np.shape(data_separe[i][j])[0]):
                data_separe[i][j][k]=data_separe[i][j][k][3:]
                for l in range(lenn2):
                    data_separe[i][j][k][l]=float(data_separe[i][j][k][l])
                    data_separe[i][j][k][l]+=factor[l][0]
                    data_separe[i][j][k][l]/=factor[l][1]
                    data_separe[i][j][k][l]=data_separe[i][j][k][l]*2-1
                    
    return data_separe

# =============================================================================
new_size_=2
a=collecte()
List=normalyze_factor_table(a)
factor=factor_to_normalyze(List)
b=collecte_y()
List_proportion=find_proportion(b)
data_separe=class_data(a,b)
data_separe=normalyze_data(data_separe,factor)
data_no_label=create_equilibre_data(data_separe,List_proportion,new_size=new_size_)
# =============================================================================

def process_y(target):
    dictio=dict()
    lenn=len(target)
    compteur=0
    for i in range(lenn):
        if target[i] not in dictio:
            dictio[target[i]]=compteur
            compteur+=1
        
    return dictio,len(dictio)


def add_label(data_no_label,dict_target,len_target):
    lenn=np.shape(data_no_label)[0]
    List=[0]*lenn
    result = []
    for i in range(np.shape(data_no_label)[0]):
        List_temp=List.copy()
        List_temp[i]=1
        for j in range(np.shape(data_no_label[i])[0]):
            data_no_label[i][j].append(List_temp)
            result.append(data_no_label[i][j])
    
    random.shuffle(result)
    return result

        
# =============================================================================
dict_target,len_target=process_y(b)
data_label=add_label(data_no_label,dict_target,len_target)
# =============================================================================
        

# =============================================================================
# =============================================================================
# =============================================================================
# # # remember about : dict_target,len_target,factor and d
# =============================================================================
# =============================================================================
# =============================================================================
        
# =============================================================================
# =============================================================================
# =============================================================================
# # # zone shuffle and type(data) -on commence sans faire de separation entre les classes !-
# =============================================================================
# =============================================================================
# =============================================================================
def separate_data(data,pourcent=10):
    random.shuffle(data)
    lenn=len(data)
    separation=math.floor(lenn*pourcent/100)
    train=data[separation:]
    validation=data[:separation]
    
    random.shuffle(train)
    random.shuffle(validation)
    
    x_train=[]
    y_train=[]
    
    x_validation=[]
    y_validation=[]
    
    for line in train:
        x_train.append(line[:-1])
        y_train.append(line[-1])
        
    for line in validation:
        x_validation.append(line[:-1])
        y_validation.append(line[-1])
        
    return np.array(x_train),y_train,np.array(x_validation),y_validation
        
    
    
    
    


# =============================================================================
# =============================================================================
# =============================================================================
# # # zone model
# =============================================================================
# =============================================================================
# =============================================================================
#EPOCHS = 400
train_x,train_y,valid_x,valid_y=separate_data(data_label)
BATCH_SIZE = 64
NAME = f"PRED-{int(time.time())}"



model = Sequential()
model.add(CuDNNLSTM(64, input_shape=(np.shape(train_x)[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(64, input_shape=(np.shape(train_x)[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))


model.add(Dense(np.shape(train_y)[1],activation="softmax"))
opt = tf.keras.optimizers.Adam(lr=0.001,decay=1e-4)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



EPOCHS = 1000
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath= "RNN_FINALE-{epoch:02d}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath,monitor='val_acc',verbose=1, save_best_only=True,mode='max'))



history = model.fit(np.array(train_x),np.array(train_y),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(np.array(valid_x),np.array(valid_y)),
                    callbacks=[tensorboard,checkpoint])



# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # # # # ici on commence la zone pour renvoyer un resultat chez kaggle
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================

def submit():
    submit_data=collecte(r'C:\Users\ianni\Desktop\robot_career\career-con-2019\X_test.csv')
    submit_data1=preprocess_data_collected(submit_data,'0',factor)
    data=[]
    for line in submit_data1:
        data.append(line)
    data1=np.array(data)
    
    List_result=[]
    nbr_iter=np.shape(data1)[0]
    
    for i in range(nbr_iter):
        result=np.argmax(model.predict(np.array(data1[i:i+1])))
        for name,val in dict_target.items():
            if val==result:
                printer=name
        List_result.append(printer)
        
        
    List_result_submit=[]
    List_result_submit.append(['series_id','surface'])
    for i in range(nbr_iter):
        List_result_submit.append([str(i),List_result[i]])
        
    with open('sample_submission.csv','w',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(List_result_submit)
        
    csvFile.close()

#model.load_weights(r'C:\Users\ianni\Desktop\robot_career\career-con-2019\models\RNN_FINALE-410-0.853.model')
#tensorboard --logdir="path"



def submit2():
    true_size=np.shape(data_label)[1]-1
    confirmation=(128-true_size)

    submit_data=collecte(r'C:\Users\ianni\Desktop\robot_career\career-con-2019\X_test.csv')
    submit_data1=preprocess_data_collected(submit_data,'0',factor)
    data=[]
    for line in submit_data1:
        for j in range(128-true_size):
            data.append(line[j:j+true_size])
    data1=np.array(data)
    compteur_choix=0
    List_result=[]
    nbr_iter=int(np.shape(data1)[0]/confirmation)

    for i in range(nbr_iter):
        List=[0]*len_target
        List_raw=[]
        for j in range(128-true_size):
#            print(j,j+true_size)
            result_raw=model.predict(np.array(data1[i*confirmation+j:i*confirmation+j+1]))
            result=np.argmax(result_raw)
            List[result]+=1
            List_raw.append(result_raw)
            
#        compteur=0
#        pos=np.argmax(List)
#        for j in range(len_target):
#            if List[j]==List[pos]:
#                compteur+=1
#        if compteur==1:
#            for name,val in dict_target.items():
#                if val==pos:
#                    printer=name
#            List_result.append(printer)
#        else:
#            compteur_choix+=1
        New_decider=[1]*len_target
        for j in range(np.shape(List_raw)[0]):
            for k in range(np.shape(List_raw)[2]):
                New_decider[k]=New_decider[k]*(1.01-List_raw[j][0][k])
        pos=np.argmin(New_decider)
        for name,val in dict_target.items():
            if val==pos:
                printer=name
        List_result.append(printer)
            
            
    List_result_submit=[]
    List_result_submit.append(['series_id','surface'])
    for i in range(nbr_iter):
        List_result_submit.append([str(i),List_result[i]])
        
    with open('sample_submission.csv','w',newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(List_result_submit)
        
    csvFile.close()
    print('done')
    print('nombre compteur choix=',compteur_choix)
            
            
        




























