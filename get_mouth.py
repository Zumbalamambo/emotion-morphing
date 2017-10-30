from PIL import Image, ImageDraw
import face_recognition
import numpy as np

'''
Function that takes in a frame and returns the points of the top and bottom lips
Input: 
    Frame: (n x m)
Output: 
    Top_lip: (12 x 2)
    Bottom_lip: (12 x 2)
'''

bb_dimensions = (75, 75)

def get_bounding_box(top, bottom):
    data = top + bottom
    min_x = min(data, key = lambda t: t[0])[0]
    max_x = max(data, key = lambda t: t[0])[0]
    min_y = min(data, key = lambda t: t[1])[1]
    max_y = max(data, key = lambda t: t[1])[1]

    offset_x = int((bb_dimensions[0] - (max_x - min_x)) / 2.0)
    offset_y = int((bb_dimensions[1] - (max_y - min_y)) / 2.0)

    bb_tl = (min_x - offset_x, min_y - offset_y)
    bb_bl = (min_x - offset_x, max_y + offset_y)
    bb_tr = (max_x + offset_x, min_y - offset_y)
    bb_br = (max_x + offset_x, max_y + offset_y)

    return [bb_tl, bb_tr, bb_br, bb_bl]


def get_pixels(I, bb):
    top_left = bb[0]
    pixels = np.zeros((bb_dimensions[0], bb_dimensions[1], 3), dtype=np.uint8)
    for i in range(bb_dimensions[0]):
        for j in range(bb_dimensions[1]):
            pixels[j, i, :] = I[top_left[1] + j, top_left[0] + i, :]
    return pixels


def adjust_points_to_bounding_box(mouth, bb):
    top_left = bb[0]
    adjusted = []
    for point in mouth:
        adjusted.append((point[0] - top_left[0], point[1] - top_left[1]))
    return adjusted


def get_mouth(filename, frame, idx):
    frame = frame.astype(np.uint8)
    # Load the jpg file into a numpy array
    print("Processing frame " + str(idx))
    
    # Find all facial features in all the faces in the image
    # we are only considering the first facial feature
    face_landmarks = face_recognition.face_landmarks(frame)[0]
    
    ## get the mouth
    top_lip, bottom_lip = face_landmarks['top_lip'], face_landmarks['bottom_lip']
    bounding_box = get_bounding_box(top_lip, bottom_lip)

    pixels = get_pixels(frame, bounding_box)
    mouth = bounding_box + top_lip + bottom_lip
    adjusted = adjust_points_to_bounding_box(mouth, bounding_box)

    # DEBUG TO DRAW ANNOTATIONS
    pil_image = Image.fromarray(pixels)
    d = ImageDraw.Draw(pil_image)
    d.line(adjust_points_to_bounding_box(top_lip, bounding_box))
    d.line(adjust_points_to_bounding_box(bottom_lip, bounding_box))
    d.line(adjust_points_to_bounding_box(bounding_box, bounding_box))
    pil_image.save("annotated/" + filename + "frame-" + str(idx) + ".jpg")

    return pixels, np.array(adjusted), bounding_box[0]

    