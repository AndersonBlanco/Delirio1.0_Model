#model development
#model will be constructed and its weights and biases will be saved from here as well

import tensorflow as tf 
#import tf_keras as keras 
import sklearn as sklrn 
import keras 
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os 
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import to_categorical
import coremltools as crml

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from chainedLayers_model import punchCalssification_model, createModel
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./punchClassification_checkpoint.ckpt",save_weights_only=True, verbose=1)

def loadModel():
    m = createModel()
    m.compile(optimizer=keras.optimizers.SGD(learning_rate=1.0), loss =keras.losses.CategoricalCrossentropy(), metrics =['accuracy'])
    m.load_weights("./punchClassification.weights.h5")
    return m


print('punch classification model loaded and compiled..')

"""
Data classification: 
Punch types: 
- Jab
- StrightRight 
- Upper Cut
- Hook 

punch classification label architecture: [jab, straightRight, upperCut, hook, rest]
"""
def getX_getY(path, label):
    xp = []
    y = []
    counter = 0

    for vid_folders in os.listdir(path):
        #print(vid_folders)
        counter +=1
        y.append(label)

        data = np.load(path + "/" + vid_folders)
       # print(data)

        x  = []
        for i in range(40):
            temp = []
            for j in range(8):
                temp.append(data[i][j]/180)
            x.append(temp)
        
        xp.append(x)

        """
        for angles in os.listdir(path + '/' + vid_folders):
            data = np.load( (path + '/' + vid_folders +'/' + str(angles)) )

            #y stores the label of the video
            #x is a 40 files containing 40 angles of a persons movement
            #we could try to optimize further by having min val and max val instead of 0 to 180 angles
            temp = []
            for i in range(8):
                temp.append(data[i]/180)
            x.append(np.array(temp))
        """
            
    xp= np.array(xp)
    yp = np.array(y)

    
    xp.resize(counter,40, 8)
    #print("X SHAPE:", xp.shape)
    return xp, yp

x_jabs_good, y_jabs_good = getX_getY('../newData/jab/good', label=[1, 0,0,0, 0])
x_jabs_bad, y_jabs_bad = getX_getY('../newData/jab/bad', label=[1, 0,0,0, 0])

x_straightRight_good, y_straightRight_good = getX_getY('../newData/straightRight/good', label=[0, 1,0,0, 0])
x_straightRight_bad, y_straightRight_bad = getX_getY('../newData/straightRight/bad', label=[0, 1,0,0, 0])

x_rest_good, y_rest_good = getX_getY('../newData/rest/good', label=[0, 0,0,0,1])
x_rest_bad, y_rest_bad = getX_getY('../newData/rest/bad', label=[0, 0,0,0,1])

#x_upperCut_good, y_upperCut_good = getX_getY('../Data/past/uppercut/good/angles', label=[0, 0,1,0,0])
#x_upperCut_bad, y_upperCut_bad = getX_getY('../Data/past/uppercut/bad/angles/upper_knee_lvl_lack', label=[0, 0,1,0,0])

print("X jab good shape: ", x_jabs_good.shape)
print("Y jab good shape: ", y_jabs_good.shape)

print("X jab bad shape: ", x_jabs_bad.shape)
print("Y jab bad shape: ", y_jabs_bad.shape)

print("X straightRight good shape: ", x_straightRight_good.shape)
print("Y straightRight good shape: ", y_straightRight_good.shape)

print("X straightRight bad shape: ", x_straightRight_bad.shape)
print("Y straightRight bad shape: ", y_straightRight_bad.shape)


def convertModel(model):
    try:
        m = crml.convert(model, source = "tensorflow")
        m.save("./punchClassification_coreml.mlpackage")
        print('model saved!!')
        return m
    except Exception as e:
        print(e)
        return e 
    


def fit_and_test_model_with_loadedWeights(x_train,y_train, x_test, y_test):
    loadedModel = loadModel()
    loadedModel.load_weights("./punchClassification.weights.h5")
    
    print("fitting..")
    loadedModel.fit(x_train,y_train, epochs=25, batch_size=2) 
    print("fitting complete..")

    print("testing.....")
    loadedModel.evaluate(x_test, y_test, verbose='auto')
    print('testing complete...')


    print('saving model....')
    loadedModel.save("./punchClassification.keras")
    loadedModel.save_weights("./punchClassification.weights.h5")
    print('model saved....')

    print('converting to coreml model file..')
    crmlModel = convertModel(loadedModel)
    print('coreml model file converted')

def fit_and_test_bare_model(x_train,y_train, x_test, y_test):
    print("fitting..")
    punchCalssification_model.fit(x_train,y_train, epochs=25, batch_size=2) 
    print("fitting complete..")

    print("testing.....")
    punchCalssification_model.evaluate(x_test, y_test, verbose='auto')
    print('testing complete...')


    print('saving model....')
    punchCalssification_model.save("./punchClassification.keras")
    punchCalssification_model.save_weights("./punchClassification.weights.h5")
    print('model saved....')


    print('converting to coreml model file..')
    crmlModel = convertModel(punchCalssification_model)
    print('coreml model file converted')

x_jabs = np.add(x_jabs_good, x_jabs_bad)
y_jabs = np.add(y_jabs_good, y_jabs_bad)

x_jabs_train, y_jabs_train, x_jabs_test, y_jabs_test = train_test_split(x_jabs, y_jabs, test_size=0.2, random_state=1)
print(x_jabs.shape, y_jabs.shape)

#jabs_x_set = np.concatenate((x_jabs_good, x_jabs_bad), axis=2)
#jabs_y_set = np.concatenate((y_jabs_good, y_jabs_bad), axis = 2)

#straightRight_x_set = np.concatenate((x_straightRight_good, x_straightRight_bad), axis =2)
#straightRight_x_set = np.concatenate((y_straightRight_good, y_straightRight_bad), axis =2)

trainingSet_x = np.concatenate((x_jabs_good, x_straightRight_good,  x_rest_good), axis = 0)
trainingSet_y = np.concatenate((y_jabs_good, y_straightRight_good,  y_rest_good), axis = 0)

testSet_x = np.concatenate((x_jabs_bad, x_straightRight_bad, x_rest_bad), axis = 0)
testSet_y = np.concatenate((y_jabs_bad, y_straightRight_bad, y_rest_bad), axis = 0)

print(trainingSet_x.shape,trainingSet_y.shape)
print(testSet_x.shape,testSet_y.shape)

fit_and_test_bare_model(trainingSet_x, trainingSet_y, testSet_x, testSet_y)
#fit_and_test_model_with_loadedWeights(x_jabs_good, y_jabs_good, x_jabs_bad, y_jabs_bad)
#fit_and_test_model_with_loadedWeights(x_rest_good, y_rest_good, x_rest_bad, y_rest_bad)
