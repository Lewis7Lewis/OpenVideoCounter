import cv2
from tkinter import filedialog
import time


filepath = filedialog.askopenfilename(title="Video",filetypes=(("Video","*.mp4"),))
cam = cv2.VideoCapture(filepath)

i = 0
frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
framerate = int(cam.get(cv2.CAP_PROP_FPS))
s = time.perf_counter()
while cam.isOpened():
    ret ,  img = cam.read()

    if ret :
        i+=1
        if i%(framerate*20) == 0:
            t = time.perf_counter()
            fps = (framerate*20)/(t-s)
            s=t
            print(f"image : {i}, {(i//framerate)//60} m {(i//framerate)%60} s (fps: {fps:.2f})")
        
        

    else :
        print("ret error")
        break
