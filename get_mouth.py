from PIL import Image, ImageDraw
import face_recognition

'''
Function that takes in a frame and returns the points of the top and bottom lips
Input: 
    Frame: (n x m)
Output: 
    Top_lip: (12 x 2)
    Bottom_lip: (12 x 2)
'''
def get_mouth(frame):

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(frame)
    
    # Find all facial features in all the faces in the image
    # we are only considering the first facial feature
    face_landmarks = face_recognition.face_landmarks(image)[0]
    
    ## get the mouth
    top_lip, bottom_lip = face_landmarks['top_lip'], face_landmarks['bottom_lip']
    pil_image = Image.fromarray(frame)
    d = ImageDraw.Draw(pil_image)

    d.line(top_lip)
    d.line(bottom_lip)

    



    
    
    
    
    # for face_landmarks in face_landmarks_list:
    
    #     # Print the location of each facial feature in this image
    #     facial_features = [
    #         'chin',
    #         'left_eyebrow',
    #         'right_eyebrow',
    #         'nose_bridge',
    #         'nose_tip',
    #         'left_eye',
    #         'right_eye',
    #         'top_lip',
    #         'bottom_lip'
    #     ]
    
    #     # Let's trace out each facial feature in the image with a line!
    #     pil_image = Image.fromarray(image)
    #     d = ImageDraw.Draw(pil_image)
    
    #     # for facial_feature in facial_features:
    #     #     d.line(face_landmarks[facial_feature], width=1)
    #     d.line(face_landmarks['top_lip'])
    #     d.line(face_landmarks['bottom_lip'])
    
    #     pil_image.show()
        
    