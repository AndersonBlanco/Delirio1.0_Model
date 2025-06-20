import modal 
import cv2
#example remote call to cloud function in deployed app on modal: 
"""
func = modal.Function.from_name("test-app","log")
res = func.remote()
print(res) 
"""
print(help(cv2))