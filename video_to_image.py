import cv2
import sys
import numpy as np

'''
Returns an array of frames and the frame count
'''
def video_to_image(filename):
    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()
    frame_count = 0
    success = True
    frames = []
    while success:
        success, image = vidcap.read()
        if (success):
            frames.append(image)
            # cv2.imwrite("frames/frame%d.jpg" % frame_count, image)     # save frame as JPEG file
            frame_count += 1
        
    return np.array(frames), frame_count


