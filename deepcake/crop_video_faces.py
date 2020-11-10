import os 

import  cv2

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import shutil
import matplotlib.pyplot as plt
import face_alignment
from .umeyama import umeyama


"""
Loading haarcascades 
"""
current_folder = os.path.dirname(__file__)
haarcascade_path = current_folder + "/" + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier( haarcascade_path)


"""
Loading ideal face alignment
"""
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
aligned_image_path = current_folder + "/alignment_reference/reference.jpg"
aligned_image = cv2.resize(cv2.imread(aligned_image_path), (64,64))
ideal_landmarks = fa.get_landmarks(aligned_image)[0]
print("Alignment landmarks loaded from: ", aligned_image_path)


def trim_video(source_path, start_time, end_time, target_path):
    ffmpeg_extract_subclip(source_path, start_time, end_time, targetname=target_path)



def crop_video_faces(video_path, new_folder_path, size = (64,64), pad = 20):
    frames_paths = []
    try:
        os.mkdir(new_folder_path)
    except:
        shutil.rmtree(new_folder_path)
        os.mkdir(new_folder_path) 
    vidObj = cv2.VideoCapture(video_path)   
    success = 1
    count = 0

    while success: 
        success, image = vidObj.read() 
        fmt_name = new_folder_path + "/" + str(count)+ ".jpg"
        try:

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray,5,1,1)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(50,50))

            for (x, y, w, h) in faces:
                image = image[y - pad :y+h + pad , x-pad:x+w+ pad, :]
            
            image = cv2.resize(image,size)

            landmarks = fa.get_landmarks(image)[0]

            transforms = umeyama(landmarks, ideal_landmarks, True)[:2]

            aligned_image =cv2.warpAffine(image, transforms, (64, 64))
            

            if len(faces) != 0 and landmarks is not None:
                cv2.imwrite(fmt_name, aligned_image)
                frames_paths.append(fmt_name)

                count += 1
                if count % 100 == 0:
                    print(count)
        except:
            pass 
    print ("saved ", count-1, " frames at ", new_folder_path)
    return frames_paths


#print("main")

#trim_video(source_path = "datasets/obama.mp4", start_time = 60 + 52, end_time = 120 + 5, target_path = "datasets/obama_trimmed.mp4")
#video_to_frames("datasets/obama_trimmed.mp4", "datasets/obama")

# trim_video(source_path = "datasets/trump.mp4", start_time = 2, end_time = 60, target_path = "datasets/trump_trimmed.mp4")
# video_to_frames("datasets/trump_trimmed.mp4", "datasets/trump"