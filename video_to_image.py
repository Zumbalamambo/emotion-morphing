import cv2
import sys
import numpy as np

'''
Returns an array of frames and the frame count
'''

def crop_image(I):
    (tr, tc, _) = I.shape
    ntc = int(tc / 2.0)
    offset = int(ntc / 2.0)
    cropped_frame = np.zeros((tr, ntc, 3))
    for i in range(ntc):
        cropped_frame[:, i] = I[:, i + offset]
    return cropped_frame

def video_to_image(filename, label):
    vidcap = cv2.VideoCapture(filename)
    success, image = vidcap.read()
    frame_count = 0
    success = True
    frames = []
    while success:
        success, image = vidcap.read()
        if (success):
            if label == "src":
                image = crop_image(image)
            frames.append(image)
            cv2.imwrite("frames/" + label + "-frame%d.jpg" % frame_count, image)     # save frame as JPEG file
            frame_count += 1
        
    return np.array(frames), frame_count


