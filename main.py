from get_mouth import get_mouth
from video_to_image import video_to_image
from blending import seamlessCloningPoisson
from morph_tri import morph_tri
from matplotlib.animation import FuncAnimation

import numpy as np
import cv2
from PIL import Image, ImageDraw
import glob
import matplotlib.pyplot as plt

video = []

def update(frame):
  plt.imshow(video[0][frame])
  plt.axis('off')

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

        warp_33_frac = [0.1]
        dissolve_33_frac = [0.1] * len(warp_33_frac)
        morphed_frame, warped = morph_tri(src_mouth, tgt_mouth, src_cc, tgt_cc, warp_33_frac, dissolve_33_frac)

        masked = Image.new('L', (60, 60), 0)
        points = [(warped[0][0], warped[0][1])]
        for i in range(4, 28):
            point = (warped[i][0], warped[i][1])
            if i < 11:
                points.append(point)
            if i >= 16 and i < 23:
                 points.append(point)

        ImageDraw.Draw(masked).polygon(points, outline=1, fill=1)
        mask1 = np.array(masked)

        masked = Image.new('L', (60, 60), 0)
        EXPANSION_OFFSET_X = 15
        EXPANSION_OFFSET_Y = 3
        points = [(tgt_cc[0][0], tgt_cc[0][1])]
        for i in range(4, 28):
            warped_point = (warped[i][0], warped[i][1])
            point = (tgt_cc[i][0], tgt_cc[i][1])
            if i < 11:
                if i == 4:
                    if warped_point[0] - point[0] > 10:
                        points.append((point[0] - (EXPANSION_OFFSET_X * 2), point[1]))
                    else:
                        points.append((point[0] - EXPANSION_OFFSET_X, point[1]))
                elif i == 10:
                    if point[0] - warped_point[0] > 10:
                        points.append((point[0] + (EXPANSION_OFFSET_X * 2), point[1]))
                    else:
                        points.append((point[0] + EXPANSION_OFFSET_X, point[1]))
                else:
                    points.append((point[0], point[1] - EXPANSION_OFFSET_Y))
            elif i >= 16 and i < 23:
                if i == 16:
                    if warped_point[0] - point[0] > 10:
                        points.append((point[0] - (EXPANSION_OFFSET_X * 2), point[1]))
                    else:
                        points.append((point[0] - EXPANSION_OFFSET_X, point[1]))
                elif i == 22:
                    if point[0] - warped_point[0] > 10:
                        points.append((point[0] + (EXPANSION_OFFSET_X * 2), point[1]))
                    else:
                        points.append((point[0] + EXPANSION_OFFSET_X, point[1]))
                else:
                    points.append((point[0], point[1] + EXPANSION_OFFSET_Y))
        ImageDraw.Draw(masked).polygon(points, outline=1, fill=1)
        mask2 = np.array(masked)


        union_mask = mask1 | mask2

        # mask = np.ones((60, 60))
        # blend this back into the original image.
        print("Created " + str(len(morphed_frame)) + " morphed frames")
        for mf_idx in range(len(morphed_frame)):
            pil_image = Image.fromarray(morphed_frame[mf_idx])
            pil_image.save("morphed/frame-%03d-%03d.jpg" % (f_ct, mf_idx))
            result_frame = seamlessCloningPoisson(morphed_frame[mf_idx], tgt_frames[f_ct], union_mask, tgt_offset[0], tgt_offset[1])
            pil_image = Image.fromarray(result_frame)
            pil_image.save("results/frame-%03d-%03d.jpg" % (f_ct, mf_idx))



generate_video_frames()
create_video('output-02-union-k-adaptive-warp')

# video.append(final_video)
# fig = plt.figure()
# anim = FuncAnimation(fig, update, frames=np.arange(0, len(src_frames)), interval=3)
# anim.save('result.gif', writer='imagemagick')


