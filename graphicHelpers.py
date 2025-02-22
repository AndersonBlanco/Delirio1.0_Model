#helper functions for graphic purposes like drawing user skeleton and extracting angles from frames 

import cv2
import mediapipe as mp 
import numpy as np
import os
import tkinter
from keras._tf_keras.keras.callbacks import ModelCheckpoint
import keras 
from model import mmy_odel

mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose 

pose = mp_pose.Pose(
    min_detection_confidence = .5,
    min_tracking_confidence = .5)

def draw(frame):
    p1 = (0,0)
    p2 = (10,10)
    updtFrame = cv2.line(frame, p1, p2, color=(255, 255, 0), thickness=5) 
    return updtFrame

def calculateAngle(p1, p2, p3):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)

    rads = np.arctan2(p3[1] - p2[1], p3[0] - p2[0]) - np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    angle = np.round(abs(rads * (180/np.pi)))

    if angle > 180:
        return 360 - angle 
    else:
        return angle 

def drawSkeleton(frame):
    #frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[0], frame.shape[1]
    res = pose.process(rgb)
    lmPose = mp_pose.PoseLandmark
    lm = res.pose_landmarks

   
    #frame = np.zeros(frame.shape, np.uint8)
    mp_drawing.draw_landmarks(
        frame, lm, mp_pose.POSE_CONNECTIONS) 
      
           #index will show the limb / joing cordinate 

    landmarks = res.pose_landmarks.landmark

       

    #right hemi
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        
    rightElbow_angle = calculateAngle(right_wrist, right_elbow, right_shoulder)

    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    rightShoulder_angle = calculateAngle(right_elbow, right_shoulder, right_hip)

    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    rightHip_angle = calculateAngle(right_shoulder, right_hip, right_knee)

    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    rightKnee_angle = calculateAngle(right_hip, right_knee, right_ankle)

    

    #left hemi
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        
    leftElbow_angle = calculateAngle(left_wrist, left_elbow, left_shoulder)

    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    leftShoulder_angle = calculateAngle(left_elbow, left_shoulder, left_hip)

    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    leftHip_angle = calculateAngle(left_shoulder, left_hip, left_knee)

    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    leftKnee_angle = calculateAngle(left_hip, left_knee, left_ankle)

####################################################################################

    right_elbow_cords= ( int(lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW].x * w), int(lm.landmark[lmPose.RIGHT_ELBOW].y * h) )
    left_elbow_cords = ( int(lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].x*w), int(lm.landmark[lmPose.LEFT_ELBOW].y*h) ) 

    right_shoulder_cords= ( int(lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].x * w), int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h) )
    left_shoulder_cords = ( int(lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].x*w), int(lm.landmark[lmPose.LEFT_SHOULDER].y*h) ) 

    right_hip_cords = ( int(lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP].x * w), int(lm.landmark[lmPose.RIGHT_HIP].y * h) )
    left_hip_cords = ( int(lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP].x*w), int(lm.landmark[lmPose.LEFT_HIP].y*h) ) 

    right_knee_cords = ( int(lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE].x * w), int(lm.landmark[lmPose.RIGHT_KNEE].y * h) )
    left_knee_cords = ( int(lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE].x*w), int(lm.landmark[lmPose.LEFT_KNEE].y*h) ) 


    right_wrist_cords = ( int(lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST].x * w), int(lm.landmark[lmPose.RIGHT_WRIST].y * h) )
    left_wrist_cords = ( int(lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST].x*w), int(lm.landmark[lmPose.LEFT_WRIST].y*h) ) 

    right_ankle_cords = ( int(lm.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE].x * w), int(lm.landmark[lmPose.RIGHT_ANKLE].y * h) )
    left_ankle_cords = ( int(lm.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE].x*w), int(lm.landmark[lmPose.LEFT_ANKLE].y*h) ) 


    #frame = cv2.flip(frame, 1) # delete to return re-alignment of drawing and user
    
    newFrame = cv2.putText(frame, str(rightElbow_angle), ( right_elbow_cords[0] - 75, right_elbow_cords[1] ), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,0), 2)
    newFrame =  cv2.putText(frame, str(leftElbow_angle), ( left_elbow_cords), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,0), 2)

    newFrame = cv2.putText(frame, str(rightShoulder_angle), ( right_shoulder_cords ), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,0), 2)
    newFrame =  cv2.putText(frame, str(leftShoulder_angle), ( left_shoulder_cords), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,0), 2)

    newFrame = cv2.putText(frame, str(rightHip_angle), ( right_hip_cords ), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,0), 2)
    newFrame =  cv2.putText(frame, str(leftHip_angle), ( left_hip_cords), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,0), 2)

    newFrame = cv2.putText(frame, str(rightKnee_angle), ( right_knee_cords ), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,0), 2)
    newFrame =  cv2.putText(frame, str(leftKnee_angle), ( left_knee_cords), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,0), 2)
        
        

    #out = cv2.VideoWriter('lessons/jab/v1.avi', fourcc, 20, (640, 480))
    #out.release()

    angles = [
        rightElbow_angle, 
        rightShoulder_angle, 
        rightHip_angle, 
        rightKnee_angle, 
        leftElbow_angle, 
        leftShoulder_angle,
        leftHip_angle,
        leftKnee_angle
        ]
    
    anglesDictionary = { #right hemisphere limbs are maped to left hemishphere in order to solve the nuance of inverted cordinate values during processing
           "rightElbow_angle": leftElbow_angle, 
        "rightShoulder_angle": leftShoulder_angle,
        "rightHip_angle": leftHip_angle,
        "rightKnee_angle": leftKnee_angle ,

        "leftElbow_angle": rightElbow_angle, 
        "leftShoulder_angle": rightShoulder_angle,
        "leftHip_angle": rightHip_angle,
        "leftKnee_angle": rightKnee_angle,
    }

    
    #evaluatios = jab_angle_conditions(angles)

    def getColor(boolean):
        if boolean == 1:
            return (0, 255, 0)
        else:
            return (0, 0, 255)
        
    rightArm_color = (0, 255, 0)
    leftArm_color = (0, 255, 0)
    rightLats_color = (0, 255, 0)
    leftLats_color = (0, 255, 0)
    rightLeg_color = (0, 255, 0)
    leftLeg_color = (0, 255, 0)


#Predictions HotFix 
    userEvaluateAngleCorrectness = [[0,0], [0,0], [0,0]] #(left elbow/arm, right elbow/arm), (left knee/leg, right knee/leg), (left shoulder, right shoulder| 1 == incorrect, 0 = correct
#elbows conditions - jab
    if np.abs(rightElbow_angle) not in validRanges["jab"]["leftElbow_angle"]: #right hemisphere limbs are maped to left hemishphere in order to solve the nuance of inverted cordinate values during processing
        rightArm_color = (0, 0, 255)
        userEvaluateAngleCorrectness[0][0] = 1
    else:
        rightArm_color = (0, 255, 0)
        userEvaluateAngleCorrectness[0][0] = 0
 
        

    if np.abs(leftElbow_angle) not in validRanges['jab']['rightElbow_angle']:#right hemisphere limbs are maped to left hemishphere in order to solve the nuance of inverted cordinate values during processing
        leftArm_color = (0, 0, 255)
        userEvaluateAngleCorrectness[0][1] = 1
    else:
        leftArm_color = (0, 255, 0)
        userEvaluateAngleCorrectness[0][1] = 0

 

#knee conditions
    if np.abs(rightKnee_angle) not in validRanges['jab']['leftKnee_angle']:#right hemisphere limbs are maped to left hemishphere in order to solve the nuance of inverted cordinate values during processing
        rightLeg_color = (0, 0, 255)
        userEvaluateAngleCorrectness[1][0] = 1
    else:
        rightLeg_color = (0, 255, 0)
        userEvaluateAngleCorrectness[1][0] = 0


    if np.abs(leftKnee_angle) not in validRanges['jab']['rightKnee_angle']:#right hemisphere limbs are maped to left hemishphere in order to solve the nuance of inverted cordinate values during processing
        leftLeg_color = (0, 0, 255)
        userEvaluateAngleCorrectness[1][1] = 1
    else:
        leftLeg_color = (0, 255, 0)
        userEvaluateAngleCorrectness[1][1] = 0
    


#shoulders conditions
    if np.abs(rightShoulder_angle) not in validRanges['jab']['leftShoulder_angle']:#right hemisphere limbs are maped to left hemishphere in order to solve the nuance of inverted cordinate values during processing
        rightArm_color = (0, 0, 255)
        userEvaluateAngleCorrectness[2][0] = 1
    else:
        rightArm_color = (0, 255, 0)
        userEvaluateAngleCorrectness[2][0] = 0


    if np.abs(leftShoulder_angle) not in validRanges['jab']['rightShoulder_angle']:#right hemisphere limbs are maped to left hemishphere in order to solve the nuance of inverted cordinate values during processing
        leftArm_color = (0, 0, 255)
        userEvaluateAngleCorrectness[2][1] = 1
    else:
        leftArm_color = (0, 255, 0)
        userEvaluateAngleCorrectness[2][1] = 0


  
 


 

 

    

#shoulders conditions - jab

    #if np.abs(rightShoulder_angle - 170) > 10:
    #    leftArm_color = (0, 0, 255)
    #else:
    #    leftArm_color = (0, 255, 0)

   # if np.abs(leftShoulder_angle - 170) > 10:
   #    leftArm_color = (0, 0, 255)
   # else:
   #    leftArm_color = (0, 255, 0)
        
        
   
    #highlight right arm : 
   

    newFrame = cv2.line(newFrame, (right_elbow_cords), (right_shoulder_cords), rightArm_color, 5)
    newFrame = cv2.line(newFrame, (right_wrist_cords), (right_elbow_cords), rightArm_color, 5)

    #highlight left arm : 
        
    newFrame = cv2.line(newFrame, (left_elbow_cords), (left_shoulder_cords), leftArm_color, 5)
    newFrame = cv2.line(newFrame, (left_wrist_cords), (left_elbow_cords), leftArm_color, 5)

    #highlight right lats: 
        
    newFrame = cv2.line(newFrame, (right_hip_cords), (right_shoulder_cords), rightLats_color, 5)

    #highlight left lats: 
        
    newFrame = cv2.line(newFrame, (left_hip_cords), (left_shoulder_cords), leftLats_color, 5)

    #highlight right leg : 
       
    newFrame = cv2.line(newFrame, (right_knee_cords), (right_hip_cords), rightLeg_color, 5)
    newFrame = cv2.line(newFrame, (right_ankle_cords), (right_knee_cords), rightLeg_color, 5)

    #highlight left leg:
        
    newFrame = cv2.line(newFrame, (left_knee_cords), (left_hip_cords), leftLeg_color, 5)
    newFrame = cv2.line(newFrame, (left_ankle_cords), (left_knee_cords), leftLeg_color, 5)

    #add jab angles
    #jabStates.add(rightElbow_angle)# right elbow, for left elbow, because the joint were processed veversed in hemishphere of body!


    #newFrame =  cv2.putText(newFrame, (user_pose_state), (10,10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,0), 2)

    #print(user_pose_state)
        
    #[rightAm, leftArm, rightLats, leftLats, rightLeg, leftLeg]
    _booleans = [1,1,1,1,1,1]
    #ewFrame  = cv2.flip(newFrame, 1)



    #cv2.imshow('Feed', newFrame)
    #cv2.resize(newFrame,( 1000, 1000), cv2.INTER_AREA)

    return angles, newFrame, userEvaluateAngleCorrectness


"""
rightElbow_angle, -> idx 0
rightShoulder_angle, -> idx 1
rightHip_angle, -> idx 2
rightKnee_angle, -> idx 3

leftElbow_angle, 
leftShoulder_angle,
leftHip_angle,
leftKnee_angle
"""
validRanges = {
    "jab": { 
        "rightElbow_angle": range(0, 55),
        "rightShoulder_angle": range(0, 25),
        "rightHip_angle": range(0,158),
        "rightKnee_angle": range(90, 172) ,

        "leftElbow_angle": range(150, 180), 
        "leftShoulder_angle": range(87,115),
        "leftHip_angle": range(0,360),
        "leftKnee_angle": range(90, 172) ,
    }
}
 

root = tkinter.Tk()
monitorResolution = (root.winfo_screenheight()+100, root.winfo_screenheight()-200) 

def label(correctNessValues, frame): #jab oriented for now, no other movement supported
    #correctNessValues -->  (left elbow/arm, right elbow/arm), (left knee/leg, right knee/leg) | 1 == incorrect, 0 = correct
    newFrame = frame

    #arms
    if  correctNessValues[0][0]:
        newFrame = cv2.putText(frame, "jab extension", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1.25, (0,0,255))
        #print("Extend your jab more")

    if  correctNessValues[0][1]:
        newFrame = cv2.putText(frame, "guard down", (50,100), cv2.FONT_HERSHEY_COMPLEX, 1.25, (0,0,255))
        print("Bring right arm closer to jaw, dont let your guard down")


    #legs
    if  correctNessValues[1][0]:
        newFrame = cv2.putText(frame, "left knee", (50,150), cv2.FONT_HERSHEY_COMPLEX, 1.25, (0,0,255))
        print("Bend your left knee more")

    if  correctNessValues[1][1]:
        newFrame = cv2.putText(frame, "rigtht knee", (50,200), cv2.FONT_HERSHEY_COMPLEX, 1.25, (0,0,255))
        print("Bend your right knee more")


    #shoulders
    if  correctNessValues[2][0]:
        newFrame = cv2.putText(frame, "left shoulder", (50,250), cv2.FONT_HERSHEY_COMPLEX, 1.25, (0,0,255))
        print("Keep your upper arm raised to eye level")


    if  correctNessValues[2][1]:
        newFrame = cv2.putText(frame, "right shoulder", (50,270), cv2.FONT_HERSHEY_COMPLEX, 1.25, (0,0,255))
        print("Keep your right elbows near your hips")


    return newFrame




        
        
#model = keras.saving.load_model("model.h5")

#test mediapipe only
def mediapipeTest():
    cap = cv2.VideoCapture(0)
    count = 0 
    data = []
    while True: 
        ret, frame = cap.read()
        frame = cv2.resize(frame, monitorResolution)

        try:
            #print(count)
            angles, newFrame, userEvaluateAngleCorrectness = drawSkeleton(cv2.flip(frame,1))
            count += 1
            #newFrame = label(userEvaluateAngleCorrectness, newFrame)
            cv2.imshow("Feed", newFrame)
            if count < 40: 
                data.append(angles)
            elif count == 41: 
                try: 
                    p = mmy_odel.predict(data)
                    print("Predicitng")
                    print("Prediction: ", p)
                except Exception as e: 
                    print(e)
            else:
                data = []
                count = 0

        
            #print(np.array(data).shape) 
            
        except Exception as e:
            print(e)
            cv2.imshow("Feed", frame)


        if cv2.waitKey(1) == ord("q"): 
            break
    cap.release()
    cv2.destroyAllWindows()

 

mediapipeTest()