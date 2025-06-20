import modal 
#import mediapipe as mp 


#test-app:

#mediapipe present only with python @3.12.10 as far as our knwoledge goes 
Image = modal.Image.from_registry("whole/mediapipe:latest").run_commands(
    
    "pip install numpy",
    "pip install opencv-python",
    #"apt-get install libgl1",
    #"apt-get install libglib2.0-0",
 

)
app = modal.App(name = "test-app", image = Image)
@app.function()
def log():
    import mediapipe as mp
    import cv2 
    print("Hello Universe")
    print(help(cv2))
    return "Helllo Universe"


@app.local_entrypoint()
def main():
    res = log.remote()
    print("Entry point accessed", res)


#uplaod files: 

#Create volume (storage unit, like a container housing your files) & handle uploading
"""
Terminal: 
create -> modal volume create {volume_name} 
upload -> modal volume put {file_path}
"""

"""
In scripts: 

volume = modal.Volume.from_name("model-volume")
loc_path = "" #the local path on computer 
remote_path = "" #the path directory in the cloud within which the file will live, 
#no dashes required if in global directory (just '.' will do it) | can also be empty to upload to global directory of volume 

@app.local_entrypoint()
def main():
    with volume.batch_upload() as upload: 
        upload.put_directory(loc_path, remote_path); 

"""


#model function
"""
app = modal.app("model", create_if_missing = True)
@app.function()
def DrawSkeleton():
    import 
"""


"""


import modal

# 1) Create a Modal App

package = modal.Image.debian_slim(python_version="3.12.10").run_commands(
    "apt-get update",
    "pip install numpy",
    "pip install tensorflow",
    "pip install tensorflow_hub"
)
app = modal.App("example-get-started", image = package)
# 2) Add this decorator to run in the cloud
@app.function()
def square(x=2):
    import numpy as mp
    import tensorflow as tf 
    import tensorflow_hub as hub
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
    
    
    #print(help(mp))
    print(f"The square of {x} is {x**2}")  # This runs on a remote worker!



"""