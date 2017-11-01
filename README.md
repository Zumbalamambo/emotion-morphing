#CIS 581 Project 3 Part B
#Video Image 

Krishna Bharathala (kbhara@seas.upenn.edu)

Devesh Dayal (deveshd@seas.upenn.edu)

##Abstract

In this project we attempted to create a video where one person speaks the words of another person, by morphing the two mouth shapes and blending it into the target image. We used facial detection provided by dlib and our final result video can be found at this Youtube link.

##Introduction

“I saw it with my own two eyes.” Before the age of photoshop, this statement was used as a way to demonstrate the validity of some occurrence. However, the age of the computer has eroded our trust in what we see. A more common saying these days would be “never trust what you read on the internet.” One field of the internet that has remained relatively untouched, however, is video. Early attempts at faking videos involved splicing clips together to make a different video. One humorous example can be found here. 

In recent years, the ability to “photoshop” video has gotten increasingly better. It is now possible to insert content into videos like text and images. It is possible to stretch and distort among other things. As time goes on, even video will not be something that can be trusted at face-value on the internet. Our goal is to use the blending and morphing techniques that we have learned over the semester to create a fake video clip. The clip takes two video clips, eg. Obama and Trump, and convincingly creates a composite clip with Obama saying Trump’s speech. 

##Related Work

University of Washington Obama: Lip Sync
In this paper, the authors use RNNs to synthesize a high quality video. The RNN maps raw audio to mouth shapes, which can then be composited with other texture information to create a full image. This work differs from what we are trying to do in two ways. One specific drawback in this study is that their research method has trained only on Obama. Any other video clip that is passed through will not work without a fully re-trained model.

University of California, Berkeley: Video Rewrite
In this paper, the authors use computer vision techniques to automatically create a new video where the person speaks words they have never spoken before. This technique works by training on old audio tracks and decomposing the video into specific mouth shapes. Then it uses morphing to properly create the full video clip that syncs with the audio. All mouth labeling is done automatically in this system.

##Implemented Methodology

At a high level we planned to implement our technique with the following algorithm:
Split the video into images
Extract mouths from images [Feature Detection]
Morph mouths into intermediate shape [Morphing]
Blend morphed mouth into target image [Blending]
Create the video from the frames
Add the audio to the video

###Feature Detection

In order to extract the mouths from the images we used dlib, a C++ library for image processing. One of dlib’s submodules is face_recognition which provides us with the face_landmarks function. Using this function we are given an array of pixels that correspond specifically to any facial feature in a given image according to this chart. 

Once given the pixels corresponding to the mouth, the next task is to find the bounding box that surrounds them. This bounding box is then passed into the morphing function in order to find the intermediate shape. 

We used a fairly trivial algorithm to create our bounding box. We found the maximum and minimum values for x and y across the different pixels and then we create the box around that. After that there was some additional work required to shift the pixels such that they were in relation to the bounding box and not the original image. 

We also needed to get a mask of the mouth to use during the blending stage. We created the mask in a similar way to the bounding box, however we were able to take advantage of the Polygon function in the PIL ImageDraw library. This function takes in a set of points and an image and returns a binary matrix of which points are within the polygon that is created by the points. We again used the chart above to determine exactly which pixels were in the outer portion of the mouth, that we wanted to feed into the polygon function. This gives us the information in the format we need for the blending function that we wrote in Project 2.

###Morphing

The next stage in our technique is to morph the two mouths together into the mouth shape that we want to blend into our target image.

Since FaceMorphing provides us with fixed points on the mouth ("mouth", (48, 68)) we can pair these two pixel sets directly. Then we can apply the morphing code that we wrote for Project 2, and take out a specific frame based on the warp factor and dissolve factor. The best value for the warp factor and dissolve factor will be experimentally determined through trial and error. Future work could be to figure out if this can be automatically determined.

###Blending

When blending the morphed mouth into the target image, we need to create a mask around the lips that determines what pixels should be blended into the target. This proved to be a significant challenge since the shape of the morphed mouth does not exactly overlap the shape of the pre-blend mouth. Thus, even after the blend, there is a significant amount of the old mouth that is still in the video. 

Instead, we took two polygons: 1 for the original mouth and 1 for the blended mouth and took the union of their bit masks. This created a slightly larger mask with which we blended the two images together. As you will see in the results section, this created a much closer image. In further iterations, we tried to create content-aware masks, that used the difference in target lip descriptors to create augmented union masks that encapsulated a buffer area. This buffer served to overwrite values in the target image that were not overwritten by the morph, such as in the case where an ‘O’ source mouth shape is blended into a target smile - the corners of the smile will not be blended unless an augmented union mask is used.

###Adding Audio 

Since we only extracted the latter 96 frames of the source Trump video, we only needed to extract a portion of the original audio segment. Since we polled frames at a rate of 30 frames per second, this meant we needed an audio offset of 3.2 seconds. We extracted the audio from the original video using the following FFMPEG command:

ffmpeg -i trump.mp4 -ss 00:00:03.200 -q:a 0 -map a trump.mp3

We then attached the ripped audio to the converted video to create our final result with the following command:

ffmpeg -i test.avi -i videos/trump.mp3 -codec copy final.avi      

##Comparison

We initially planned to follow the same pipeline but to morph full expressions from one face to another. We reduced the scope of the problem given limited time to just being mouth shapes and implemented the same pipeline. We spent a long time iterating and deliberating on the parameters and inputs to our morphing and blending routines, since tweaking these parameters directly affected the quality of our final output. For instance, dynamically creating a tight mask for a polygon shape for each frame of the source video required us to think outside the box, since integration of routines into a single pipeline is a non-trivial task. We spent a considerable amount of time familiarizing ourselves with OpenCV and dlib, both of which proved to be crucial for the project but did not factor heavily into our initial proposed solution.

##Results

Our goal was to extract the audio from this clip and blend it into this clip.

Over the course of the week we implemented many different types of algorithms that resulted in different types of videos. Below I will present screenshots from each video and the problem that we encountered, and then how we fixed it moving forward.

###Version 1:  Mask size too big and too small

In the first image, the mask that we set for the blend stage was too large, this led to all of pixels outside of the mouth being blended in. To fix this we changed it such that the mask represented just the polygon around the pixels on the lips themselves.

However, in the second image, although a little tough to see, the mask becomes too small. We are able to see Obama’s lips underneath the blended in Trump lips. This created some really weird, but amusing videos.

###Version 2: We then blended two polygons: 1 for the original mouth and 1 for the blended mouth and took the union of their bit masks.

There was definitely a significant improvement here, but sometimes the lips would still be too small or we would have patches of white randomly appear.

###Final Version: 

As mentioned above, we finally tried to create content-aware masks, that used the difference in target lip descriptors to create augmented union masks that encapsulated a buffer area.

This ended up creating the best image, and is what we eventually decided on for our final video.

Our final video can be found at this Youtube link with synced audio.

##Future Work

There are two areas for improvement within our technique:

1) Live video face blending
Currently, our algorithm runs rather slowly, taking upwards of 3 minutes for a 5 second video (reading at 30 frames per second) to execute on an image of size 640 by 720px. We would hope to optimize this by vectorizing our morphing and blending code and potentially investigate speed differences between Numpy/Scipy and MATLAB implementations of the same. If we manage to lower the latency in our face blending pipeline we can experiment with live video face blending or using a webcam capture feed.

2) Parameter estimation heuristics
Our current implementation uses pre-determined parameter values that are set as constant across every video frame processed. In the sample provided, we work with a warp factor of 0.1 and a dissolve factor of 0.1, along with a fixed shape bounding box rectangle for each detected facial feature area. However, if we could change these values using a content-aware heuristic, each frame could then morph and blend an optimal warped pixel area to the target image, resulting in a much smoother final outcome sequence.

##References

###Research Papers
https://grail.cs.washington.edu/projects/AudioToObama/siggraph17_obama.pdf
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/human/bregler-sig97.pdf

###Libraries
https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
http://dlib.net/imaging.html

###Youtube Videos
https://www.youtube.com/watch?v=hX1YVzdnpEc
https://www.youtube.com/watch?v=bhFB7-w2MY8
https://www.youtube.com/watch?v=-qCaLgRadwA

## How to Run
Execute the main.py script with Python 3.

##Contributions

###Krishna Bharathala
I wrote the original morphing and blending code and modified them slightly in order to fit our use case. I also wrote the original shell for interfacing morphing / blending with our code to get the facial features. I also spent time researching techniques to tighten the mask on the mouth and came up with the technique for taking the union of the two lip masks which we used in our final implementation.

###Devesh Dayal
I worked on the OpenCV part of the pipeline - reading in frames from both input videos, converting them to images, extracting out audio via FFMPEG and finally creating a combined video from the morphed and blended results with audio from the source video. I also worked on creating an input mask to our blend routines by using an adaptive content aware union mask of the warped correspondent points and the matched target correspondences.
