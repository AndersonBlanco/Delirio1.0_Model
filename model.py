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
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

checkpoint_path = "./model_storage/model1.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5),
]

def getX_getY(path, label):
    x = []
    y = []
    counter = 0

    for vid_folders in os.listdir(path):
        counter +=1
        y.append(label)

        for angles in os.listdir(path + '/' + vid_folders):
            data = np.load( (path + '/' + vid_folders +'/' + str(angles)) )

            #y stores the label of the video
            #x is a 40 files containing 40 angles of a persons movement
            #we could try to optimize further by having min val and max val instead of 0 to 180 angles
            temp = []
            for i in range(8):
                temp.append(data[i]/180)
            x.append(np.array(temp))
            
    xp= np.array(x)
    yp = np.array(y)

    xp.resize(counter,40, 8)
    return xp, yp





input_shape = (40, 8) # 1 global array --> 40 arrays --> 3 values each 
gru_units = 64
mmy_odel = keras.Sequential([
    keras.Input((40, 8)),
    keras.layers.GRU(128, return_sequences=False),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])


mmy_odel.compile(optimizer="adam", loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

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
    print("TRAINING!!!")
    xgj, ygj = getX_getY(path="./Data/past/jab/good/angles", label= 0)
    print('X size: ', xgj.shape)
    print('Y shape: ', ygj.shape)

    xbj, ybj = getX_getY(path="./Data/past/jab/bad/angles/knee_lvl_lack", label= 1)
    print('X size: ', xbj.shape)
    print('Y shape: ', ybj.shape)

    xbj2, ybj2 = getX_getY(path="./Data/past/jab/bad/angles/rotation_lack", label= 2)
    print('X size: ', xbj2.shape)
    print('Y shape: ', ybj2.shape)

    xgu, ygu = getX_getY(path="./Data/past/uppercut/good/angles", label= 3)
    print("4")

    xbu, ybu = getX_getY(path="./Data/past/uppercut/bad/angles/upper_knee_lvl_lack", label= 4)
    print("5")


    xbu2, ybu2 = getX_getY(path="./Data/past/uppercut/bad/angles/upper_rotation_lack", label= 5)
    print("6")


    xgr, ygr = getX_getY(path="./Data/past/rest/good/angles", label= 6)
    print("7")


    xbr, ybr = getX_getY(path="./Data/past/rest/bad/angles", label= 7)
    print("8")


    xgs, ygs = getX_getY(path="./Data/past/straight_right/good/angles", label= 8)
    print("9")



    xbs, ybs = getX_getY(path="./Data/past/straight_right/bad/angles/straight_defence_lack", label= 9)
    print("10")


    



    data = np.concatenate((xgj, xbj, xbj2, xgu, xbu, xbu2, xgr, xbr, xgs, xbs), axis = 0)
    labels = np.concatenate((ygj, ybj, ybj2, ygu, ybu, ybu2, ygr, ybr, ygs, ybs), axis = 0)

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)
    print(y_test)

    mmy_odel.fit(x_train, y_train, epochs=50, batch_size=16, callbacks=callbacks)

    print("done fiting, lets test!")

    y_pred = mmy_odel.predict(x_test)

    y_pred_labels = np.argmax(y_pred, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    report = classification_report(y_true_labels, y_pred_labels)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")

    mmy_odel.save_weights('./model_storage/GRU1_weight.weights.h5')
    mmy_odel.save('GRU1.keras')



trainModel()
