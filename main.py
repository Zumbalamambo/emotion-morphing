from get_mouth import get_mouth
from video_to_image import video_to_image
from blending import seamlessCloningPoisson
from morph_tri import morph_tri
import numpy as np
import cv2
from PIL import Image, ImageDraw
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

def read_frames(label, offset=0):
    print("Reading in " + label + " frames")
    image_list = []
    count = 0
    for filename in glob.glob('frames/' + label + '*.jpg'):
        if count >= offset:
            im = Image.open(filename).convert('RGB')
            im = np.array(im, dtype=np.uint8)
            image_list.append(im)
        count += 1
    return np.array(image_list)

def create_video(name):
    print("Reading in result frames")
    image_list = []
    for filename in glob.glob('results/*.jpg'):
        im = cv2.imread(filename)
        print(filename)
        image_list.append(im)
    print("Creating video for " + str(len(image_list)))
    (h, w, _)  = image_list[0].shape
    out = cv2.VideoWriter(name + '.avi', cv2.VideoWriter_fourcc(*"MJPG"), 30, (w, h))
    for r in image_list:
        out.write(r)
        cv2.imshow('frame', r)
    out.release()
    cv2.destroyAllWindows()

# main code
# src_frames, tgt_frames = create_frames('videos/trump.mp4', 'videos/obama.mp4')

def generate_video_frames():
    src_frames = read_frames('src', 96)
    tgt_frames = read_frames('tgt')

    # diff = len(src_frames) - len(tgt_frames) 
    # print(len(src_frames), len(tgt_frames))
    # if (diff > 0):
    #     print("Smartly increasing target frames by " + str(diff))
    #     flip = reversed(tgt_frames)
    #     for i in range(diff):
    #         tgt_frames.append(flip[i])

    count = 0
    for f_ct in range(min(len(src_frames), len(tgt_frames))):
        print("Processing frame " + str(f_ct))
        src_mouth, src_cc, _ = get_mouth("src", src_frames[f_ct], count)  
        tgt_mouth, tgt_cc, tgt_offset = get_mouth("tgt", tgt_frames[f_ct], count)
        count += 1

        warp_33_frac = [0]
        dissolve_33_frac = [1] * len(warp_33_frac)
        morphed_frame, warped = morph_tri(src_mouth, tgt_mouth, src_cc, tgt_cc, warp_33_frac, dissolve_33_frac)

        ## blend this back into the original image.
        # masked = Image.new('L', (75, 75), 0)
        # points = [(warped[0][0], warped[0][1])]
        # for i in range(4, 28):
        #     if i < 11:
        #         points.append((warped[i][0], warped[i][1]))
        #     if i >= 16 and i < 23:
        #          points.append((warped[i][0], warped[i][1]))

        # ImageDraw.Draw(masked).polygon(points, outline=1, fill=1)
        # mask = np.array(masked)

        mask = np.ones((60, 60))
        print("Created " + str(len(morphed_frame)) + " morphed frames")
        for mf_idx in range(len(morphed_frame)):
            pil_image = Image.fromarray(morphed_frame[mf_idx])
            pil_image.save("morphed/frame-%03d-%03d.jpg" % (f_ct, mf_idx))
            result_frame = seamlessCloningPoisson(morphed_frame[mf_idx], tgt_frames[f_ct], mask, tgt_offset[0], tgt_offset[1])
            pil_image = Image.fromarray(result_frame)
            pil_image.save("results/frame-%03d-%03d.jpg" % (f_ct, mf_idx))





generate_video_frames()
create_video('output-02-warp')

# plt.imshow(final_video[0])
# plt.show()


