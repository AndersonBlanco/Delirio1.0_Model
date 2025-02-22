#model development
#model will be constructed and its weights and biases will be saved from here as well

import tensorflow as tf 
import tf_keras as keras 
import sklearn as sklrn 
import keras 
from keras._tf_keras.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os 

def getX_getY(path, label):
    x = np.array([])
    y = np.array([])
    count = 0 
    paths = []
 
    for video_folder in os.listdir(path):
        y = np.append(y, label) 
        for angles in os.listdir(path):
            p = path + "/" + angles
            try:
                if ".npy" in p: 
                    #print("Path: ", p)
                    data = np.load(p, "r")
                    x = np.append(x, data) 
            except Exception:
                print(Exception) 


    x.resize((1, 40, 8))
    y.resize((1, 8 , 2)) #one set per every 40 set of angles  
    #40 sets of 8 elements (8 differernt states), of elements (punch type true or false: 0 for false 1 for true, quality = 0 for bad, 1 for good)
    return x, y



input_shape = (40, 8) # 1 global array --> 40 arrays --> 3 values each 
gru_units = 64
mmy_odel = keras.Sequential()
mmy_odel.add(keras.Input((40,8)))
mmy_odel.add(keras.layers.GRU(units = gru_units, recurrent_activation="tanh", return_sequences=True, return_state=True,))
#model.add(keras.layers.Concatenate(axis = -1))
mmy_odel.add(keras.layers.Lambda(lambda x : x[1]))
mmy_odel.add(keras.layers.Dense(units=16))
mmy_odel.add(keras.layers.Reshape((8, 2)))

mmy_odel.compile(optimizer="SGD", loss=keras.losses.CategoricalCrossentropy(from_logits=True), metrics = ['categorical_accuracy', "accuracy"])

print(mmy_odel.summary())


#labels hit-encodings: 
#[ [type of punch , quality] ] --> quality: 1 = good, bad = 0 
#jab -> [[0,0],[1,1],[0,0]]
#upper cut -> [[1,1],[0,0],[0,0]]
#rest -> [[0,0],[0,0],[1,1]]

#good jab = 0
#bad jab = 1

#good rest = 3
#bad rest = 4
 
def trainModel():
    x, y = getX_getY(path="./Data/past/jab/good/angles", label= [[0,0],[1,1],[0,0]])
    #x = x.reshape((40, 8))
    print('X size: ', x.shape)
    print('Y shape: ', y.shape)

    try: 
        mmy_odel.fit(x, y, epochs=10)
    except Exception as e: 
        print(e)
    mmy_odel.save('model.h5')
    #print(x) 
    #print(y) 
    #print(x.shape)
    #print(y.shape)
    #print(len(y))

trainModel()

#synsasensta technologies 