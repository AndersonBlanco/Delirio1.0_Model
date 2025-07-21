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
import tkinter
import multiprocessing
#print(os.path.isfile("./punchClassification.keras"))

GRU1 = tf.keras.models.load_model("ChainedGRU_Arch/CHAINED_MODEL.keras")#('GRU2.keras')
root = tkinter.Tk()
monitorResolution = (root.winfo_screenheight()+100, root.winfo_screenheight()) 

num_videos = 1

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2040)  # set camera as wide as possible
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)



"""val_to_pred = ["good jab", "bad jab - knee level lack", "bad jab - rotation lack",
               "good uppercut", "bad uppercut - rotation lack",
               "good resting", "bad resting", "good straight", "bad straight - lack of defence"]"""



val_pred = ["good jab", "bad jab - knee level lack", 
            "good straight", "bad straight, lack of rotation"," good rest", "bad rest, wrong stance",
            "good kick", "bad kick, don't lounge leg out"]

val_punchClassification_labels = ['jab', 'straightRight', 'upperCut', 'hook', 'rest']
chained_model_labels = ["jab_lack_of_rotation", "jab_correct", "straight_right_lack_of_rotation", "straight_right_correct", "rest_bad_stance", "rest_correct" ]
def label_punchClassification(angles):
    pred_y = np.array(GRU1.predict(angles))
    #print("ANGLES:", angles)
    #print("PREDICTION: ", pred_y)
    idx = pred_y[0].argmax(axis = 0)
    p = chained_model_labels[idx]
    print("Prediciton label: ", p)
    print("Raw prediction hot-on eencoding: ", pred_y[0])
    return p


def label(angles):
    pred_y = np.array(GRU1.predict(angles))
    #print("ANGLES:", angles)
    #print("PREDICTION: ", pred_y)
    idx = pred_y[0].argmax(axis = 0)

    return val_pred[idx]


def init_capture_window(path):
    right_hemi_40set = []
    left_hemi_40set = []
    cap = cv2.VideoCapture("jul_20_2025_trainingVids/" + path)
    counter = 0
    a = []
    statement = "Put whole body into frame"
    while True:
        ret, frame = cap.read()
        #frame = cv2.resize(frame, monitorResolution)
        frame = cv2.flip(frame, 0)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        try:
            height, width, _ = frame.shape

            left_half_frame = frame[:,width//5:width//2] #etseban

            right_half_frame = frame[:,width//2:] # anderson
            angles_right_hemi, right_angled_frame = drawSkeleton(right_half_frame)
            angles_left_hemi, left_angled_frame = drawSkeleton(left_half_frame)
           

            right_hemi_40set.append(angles_right_hemi)
            left_hemi_40set.append(angles_left_hemi)

            '''
        if counter == 40:
            counter = 0
            numpy_a = np.array(a)
            #print("BOTTOM SECTION: ", numpy_a)
            numpy_a.resize(1,40,8)
            print("numpy_a shape: ", numpy_a.shape)
            statement = label_punchClassification(numpy_a)
            a=[]
    
        else:

            frame = cv2.circle(frame, (300,300), 40, (0,0,255), -1)
            for i in range(8):
                angles[i] = angles[i]/180
            a.append(angles)
            counter += 1
            '''
        except Exception as e:
            print(e) 
            counter = 0
            a=[]
            statement = "Put body in frame"


        #cv2.putText(frame, statement, (50,50), cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,0), 4 ,cv2.LINE_AA)
        cv2.imshow('frame', left_angled_frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
 
    cap.release()
    cv2.destroyAllWindows()

    return right_hemi_40set, left_hemi_40set

def automate_extract():
    file_names = os.listdir("./jul_20_2025_trainingVids")
    movement_name = []
    movement_type = []
    for i in range(len(file_names)):
        right_hemi_40set, left_hemi_40set = init_capture_window(file_names[i])

        np.save(f"./jul_20_2025_dataExtract/{movement_name[i]}/{movement_type[i]}/a{i}.npy",right_hemi_40set)
        np.save(f"./jul_20_2025_dataExtract/{movement_name[i]}/{movement_type[i]}/a{i+1}.npy",right_hemi_40set)

    print("data extraction and storing complete")



video_punch_titles = ['rest', 'rest', 'rest', 'jab', 'jab', 'jab', 'jab', 'straight', 'straight', 'straight', 'straight', 'upper_cut', 'straight', 'experimental/rest', 'experimental/straight', 'experimental/upper_cut'] #name of technique 
video_punch_types = ['good', 'bad/low_guard','bad/curved_back', 'good', 'bad/lack_knee_lvl', 'bad/lack_end_guard', 'bad/lack_opposite_guard', 'good', 'good', "bad/lack_opposite_guard", 'bad/lack_hip_rotation', 'good', 'bad/lack_hip_rotation', 'bad/lack_knee_lvl_curved_back_low_guard', 'bad/lack_end_guard_opposite_guard', 'bad/over_committ']#good / bad or correct/incorrect evaluation result / quality of punch  

#caution: 
#IMG_3467.MOV and IMG_3468.MOV seem to be both straight_right punct title and of same quality (good)
def save_data_to_directory(video_title, base_path, punch_title, punch_type):
    save_data_to_directory = base_path + punch_title + '/' + punch_type
    try:

        right_hemi_40set, left_hemi_40set = init_capture_window(video_title)
    
        a_upper_bound = (len(right_hemi_40set) - (len(right_hemi_40set) % 40))
        b_upper_bound = (len(left_hemi_40set) - (len(left_hemi_40set) % 40))

        _a = np.array(right_hemi_40set[: a_upper_bound ])
        a = _a.reshape((int(a_upper_bound/40), 40,8))
        _b = np.array(left_hemi_40set[: b_upper_bound])
        b = _b.reshape((int(b_upper_bound/40), 40,8))
    
        print('Final cv2 angle extraction results (rigt hemi): ', np.array(a).shape, len(a))
        print('Final cv2 angle extraction results (left hemi): ', np.array(b).shape, len(b))

        if not os.path.exists(base_path + punch_title):
            os.mkdir(base_path+punch_title)
            os.mkdir(save_data_to_directory)

        last_i = 0
        for i in range(min([len(a), len(b)])):
            a_file_path=f"{save_data_to_directory}/a{i}.npy"
            np.save(a_file_path, a[i])
            last_i+=1
           

        for x in range(min([len(a), len(b)]), 2*min([len(a), len(b)])):
            b_file_path=f"{save_data_to_directory}/a{x}.npy"
            np.save(b_file_path, b[x - min([len(a), len(b)]) ])


    except Exception as e:
        print('vidoe footage error', e)



#save_data_to_directory("IMG_3457.MOV", "./jul_20_2025_dataExtract/rest/good/")

videos_titles = os.listdir('./jul_20_2025_trainingVids')
for x in range(len(videos_titles)):
    full_dir = (f"./july_20_2025_trainingVids/{videos_titles[x]}")
    punch_title = video_punch_titles[x]
    punch_type = video_punch_types[x]
    save_data_to_directory(videos_titles[x], f"./jul_20_2025_dataExtract/", punch_title, punch_type)
