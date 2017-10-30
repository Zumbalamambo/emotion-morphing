from get_mouth import get_mouth
from video_to_image import video_to_image
from blending import seamlessCloningPoisson
from morph_tri import morph_tri
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt

WARP_FRAC = 0.5
DISSOLVE_FRAC = 0.5

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

final_video = []
count = 0
for f_ct in range(min(len(src_frames), len(tgt_frames))):
    src_mouth, src_cc, _ = get_mouth("src", src_frames[f_ct], count)  
    tgt_mouth, tgt_cc, tgt_offset = get_mouth("tgt", tgt_frames[f_ct], count)
    count += 1

    morphed_frame = morph_tri(src_mouth, tgt_mouth, src_cc, tgt_cc, [WARP_FRAC], [DISSOLVE_FRAC])

    ## blend this back into the original image.
    mask = np.ones((75, 75))
    result_frame = seamlessCloningPoisson(morphed_frame[0], tgt_frames[f_ct], mask, tgt_offset[0], tgt_offset[1])
    final_video.append(result_frame)
    break

plt.imshow(result_frame[0])
plt.show()


