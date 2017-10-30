from get_mouth import get_mouth
from video_to_image import video_to_image
import numpy as np

# convert the trump video to frames
frames, frame_count = video_to_image('videos/trump.mp4')
print(frame_count)

count = 0
(tr, tc, _) = frames[0].shape
ntc = int(tc / 2.0)
offset = int(ntc / 2.0)
for frame in frames:
    # resize each image to 
    cropped_frame = np.zeros((tr, ntc, 3))
    for i in range(ntc):
        cropped_frame[:, i] = frame[:, i + offset]
    mouth, bb = get_mouth("trump", cropped_frame, count)
    count += 1

# convert the obama video to frames
frames, frame_count = video_to_image('videos/obama.mp4')
print(frame_count)

count = 0
for frame in frames:
    mouth, bb = get_mouth("obama", frame, count)
    count += 1




