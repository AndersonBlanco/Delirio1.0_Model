# Content in this file is mutable and there is no need to keep it consistent. 
# Serves no importance, testing, creative idea development, problem solving and troubleshooting purposes only 

import tensorflow as tf
from tensorflow import keras 
from keras.callbacks import ModelCheckpoint
import sklearn
import os
import numpy as np
import cv2 
import winsound
from vision import drawSkeleton

GRU1 = tf.keras.models.load_model('GRU1.keras')

num_videos = 1
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2040)  # set camera as wide as possible
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)




val_to_pred = ["good jab", "bad jab - knee level lack", "bad jab - rotation lack",
               "good uppercut", "bad uppercut - knee level lack", "bad uppercut - rotation lack",
               "good resting", "bad resting", "good straight", "bad straight - lack of defence"]

def label(angles):
    pred_y = np.array(GRU1.predict(angles))
    idx = pred_y[0].argmax(axis = 0)

    return val_to_pred[idx]


f = 0
a = []
p = "Put whole body into frame"
while True:
    ret, frame = cap.read()


    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break


    try:
        angles, newFrame = drawSkeleton(frame)
        if f != 40:
            f += 1
            a.append(angles)
        else:
            winsound.Beep(1000,500)
            a = np.array(a)
            a.resize(1,40,8)
            try:
                p = label(a)
                print(p)
            except:
                print("Try again")
                p = "Try again"

            
            a = []
            f = 0
            
        cv2.putText(frame, p, (200,100), cv2.FONT_HERSHEY_DUPLEX, 3, (0,0,0), 2, cv2.LINE_AA)

        cv2.imshow('frame', newFrame)
    except:
        cv2.rectangle(frame, (200, 200), (1000, 200), (255,255,255), -1) 
        cv2.putText(frame, "Put whole body into frame", (10,10), cv2.FONT_HERSHEY_DUPLEX, 2, (11,16,36), 2, cv2.LINE_AA)



    if cv2.waitKey(1) == ord('q'):
        break
    
 
cap.release()
cv2.destroyAllWindows()