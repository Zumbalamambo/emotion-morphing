from get_mouth import get_mouth
from video_to_image import video_to_image
import numpy as np
from PIL import Image
import glob

def create_frames(src, tgt):
    print("Creating source frames")
    src_frames, _ = video_to_image(src, 'src')
    print("Creating target frames")
    tgt_frames, _ = video_to_image(tgt, 'tgt')
    return src_frames, tgt_frames

def read_frames(label):
    print("Reading in " + label + " frames")
    image_list = []
    for filename in glob.glob('frames/' + label + '*.jpg'):
        im = Image.open(filename).convert('RGB')
        im = np.array(im, dtype=np.uint8)
        image_list.append(im)
    return np.array(image_list)


# main code
# src_frames, tgt_frames = create_frames('videos/trump.mp4', 'videos/obama.mp4')
src_frames = read_frames('src')
tgt_frames = read_frames('tgt')

count = 0
for frame in src_frames:    
    pixels, mouth, o = get_mouth("src", frame, count)
    count += 1

count = 0
for frame in tgt_frames:
    pixels, mouth, o = get_mouth("tgt", frame, count)
    count += 1


