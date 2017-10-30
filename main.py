from get_mouth import get_mouth
from video_to_image import video_to_image
import numpy as np
from PIL import Image
import glob

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

count = 0
src_mouths, src_cc = [], []
for frame in src_frames:    
    mouth, cc, offset = get_mouth("src", frame, count)
    src_mouths.append(mouth)
    src_cc.append(cc)
    count += 1

count = 0
tgt_mouths, tgt_cc, tgt_offset = [], [], []
for frame in tgt_frames:
    mouth, cc, offset = get_mouth("tgt", frame, count)
    tgt_mouths.append(mouth)
    tgt_cc.append(cc)
    tgt_offset.append(offset)
    count += 1

final_video = []
for f_ct in xrange(min(len(src_mouths, tgt_mouths))):
    morphed_frame = morph_tri(src_mouths[f_ct], tgt_mouths[f_ct], src_cc[f_ct], tgt_cc[f_ct], WARP_FRAC, DISSOLVE_FRAC)

    break

    ## blend this back into the original image.
    result_frame = seamlessCloningPoisson(morphed_frame, tgt_frames[f_ct], tgt_offset[f_ct])
    final_video.append(result_frame)

plt.imshow(morphed_frame[0])
plt.show()


