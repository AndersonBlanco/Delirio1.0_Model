import time
import cv2 
from graphicHelpers import drawSkeleton
import numpy as np 
import winsound

import os 

def extract_from_video(video, num_0f_sets, saveToPath = None):
    allAngles = []
    cap = cv2.VideoCapture(video)
    for x in range(num_0f_sets):
        print(f"Data extraction {x} will start in 2 seconds.....")
        cv2.waitKey(2000)
        winsound.Beep(2000, 500)
        print("Video: ", x)
        for y in range(num_0f_sets):
            ret, frame = cap.read()
            
            try:
                angles, newFrame = drawSkeleton(frame)
                allAngles.append(angles) 
                newFrame = pasteText(f"Video: {x}", newFrame, (50,50))
                np.save(f"{saveToPath}/set_{y}.npy", angles)
                cv2.imshow("Feed", newFrame)
            except Exception as e:
                print(e)
                cv2.imshow("Feed", frame)
            
            if cv2.waitKey(1) == ord("q"): 
                break
        winsound.Beep(1000, 500) 
        #np.savetxt(f"set_txt_{x}.txt", allAngles)
        print(f"Data extraction {x} ended")

    cap.release()
    cv2.destroyAllWindows()

def extractFromMany(videosPath):
    count = 1
    for video in os.listdir(videosPath):
        path = (videosPath + '/' + video)
        
        try:
            saveToPath = "./testDataSet" + '/' + f'vid_{count}'
            if not os.path.exists(saveToPath): 
                os.mkdir(saveToPath) 

            extract_from_video(path, 40, saveToPath= saveToPath)
            count +=1 
        except Exception as e:
            print(e) 
            print("Crashed on: ", path)



#extract_from_video("./Data/past/jab/vids/video_1.avi", 40)
#extractFromMany("./Data/past/jab/vids")

def pasteText(text, frame, cords):
    return cv2.putText(frame, text, cords, cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,0), 2)

step = 50
def live_extract(num_0f_sets, num_of_frames):
    allAngles = []
    cap = cv2.VideoCapture(0)
    for x in range(num_0f_sets):
        print(f"Data extraction {x+step} will start in 2 seconds.....")
        cv2.waitKey(1000)
        winsound.Beep(2000, 500)
        print("Video: ", x+step)
        for y in range(num_of_frames):
            ret, frame = cap.read()
            angles, newFrame = drawSkeleton(frame)
            allAngles.append(angles) 

            newFrame = pasteText(f"Video: {x+step}", newFrame, (50,50))
            cv2.imshow("Feed", newFrame)
            if cv2.waitKey(1) == ord("q"): 
                break
        winsound.Beep(1000, 500) 
        np.save(f"./newdata/kick/bad/set_{x+step}.npy", allAngles)
        #np.savetxt(f"set_txt_{x}.txt", allAngles)
        print(f"Data extraction {x+step} ended")

    cap.release()
    cv2.destroyAllWindows()
 

live_extract(50,40)