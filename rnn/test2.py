import numpy as np
import csv
import math

import random
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,CuDNNLSTM,BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint




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
#                new_to_add[j]=new_to_add[j]*2-1
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
#                new_to_add[j]=new_to_add[j]*2-1
            data_temp.append(new_to_add)
#    
    if collected_y!='0':
#
        data_temp.append(collected_y[val])
    data.append(data_temp)
    
    return data

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
        


# =============================================================================
a=collecte()
List=normalyze_factor_table(a)
factor=factor_to_normalyze(List)
b=collecte_y()
c=preprocess_data_collected(a,b,factor)
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

def preprocess_data_collected_with_tag(data,dict_target,len_target):
    target_base=[0]*len_target
    lenn=len(data)
    for i in range(lenn):
        pos=dict_target[data[i][-1]]
        data[i][-1]=target_base.copy()
        data[i][-1][pos]=1
                
    return data
        
        
# =============================================================================
dict_target,len_target=process_y(b)
d=preprocess_data_collected_with_tag(c,dict_target,len_target)
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
        
    
    
    
    

train_x,train_y,valid_x,valid_y=separate_data(d)

# =============================================================================
# =============================================================================
# =============================================================================
# # # zone model
# =============================================================================
# =============================================================================
# =============================================================================
#EPOCHS = 400
BATCH_SIZE = 64
NAME = f"PRED-{int(time.time())}"



model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(np.shape(train_x)[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(np.shape(train_x)[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(np.shape(train_x)[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(64,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(128,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(64,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(np.shape(train_y)[1],activation="softmax"))
opt = tf.keras.optimizers.Adam(lr=0.001,decay=1e-5)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



EPOCHS = 2000
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath= "RNN_FINALE-{epoch:02d}-{val_acc:.3f}"
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

# =============================================================================
# submit_data=collecte(r'C:\Users\ianni\Desktop\robot_career\career-con-2019\X_test.csv')
# submit_data1=preprocess_data_collected(submit_data,'0',factor)
# data=[]
# for line in submit_data1:
#     data.append(line)
# data1=np.array(data)
# 
# List_result=[]
# nbr_iter=np.shape(data1)[0]
# 
# for i in range(nbr_iter):
#     result=np.argmax(model.predict(np.array(data1[i:i+1])))
#     for name,val in dict_target.items():
#         if val==result:
#             printer=name
#     List_result.append(printer)
#     
#     
# List_result_submit=[]
# List_result_submit.append(['series_id','surface'])
# for i in range(nbr_iter):
#     List_result_submit.append([str(i),List_result[i]])
#     
# with open('sample_submission.csv','w',newline='') as csvFile:
#     writer = csv.writer(csvFile)
#     writer.writerows(List_result_submit)
#     
# csvFile.close()
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



