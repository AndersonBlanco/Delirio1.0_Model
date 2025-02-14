import time
import cv2 
from graphicHelpers import drawSkeleton
import numpy as np 
import winsound

def extract_from_video(video):
    cap = cv2.VideoCapture(video)
    all_angles = []
    while True: 
        ret, frame = cap.read()

        angles_i, newFrame = drawSkeleton(frame) 
        all_angles.append(angles_i) 

        if cv2.waitKey(0) == ord("q"):
            break; 
    
    cap.release()
    cv2.destroyAllWindows()
    return all_angles


def pasteText(text, frame, cords):
    return cv2.putText(frame, text, cords, cv2.FONT_HERSHEY_COMPLEX, 0.9, (255,255,0), 2)


def live_extract(num_0f_sets, num_of_frames):
    allAngles = []
    cap = cv2.VideoCapture(0)
    for x in range(num_0f_sets):
        print(f"Data extraction {x} will start in 2 seconds.....")
        cv2.waitKey(2000)
        winsound.Beep(2000, 500)
        print("Video: ", x)
        for y in range(num_of_frames):
            ret, frame = cap.read()
            angles, newFrame = drawSkeleton(frame)
            allAngles.append(angles) 

            newFrame = pasteText(f"Video: {x}", newFrame, (50,50))
            cv2.imshow("Feed", newFrame)
            if cv2.waitKey(1) == ord("q"): 
                break
        winsound.Beep(1000, 500) 
        np.save(f"set_{x}.npy", allAngles)
        np.savetxt(f"set_txt_{x}.txt", allAngles)
        print(f"Data extraction {x} ended")

    cap.release()
    cv2.destroyAllWindows()
 

live_extract(2,40)