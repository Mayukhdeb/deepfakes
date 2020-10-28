import os 

import  cv2

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import shutil
import matplotlib.pyplot as plt


current_folder = os.path.dirname(__file__)
haarcascade_path = current_folder + "/" + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier( haarcascade_path)

def trim_video(source_path, start_time, end_time, target_path):
    ffmpeg_extract_subclip(source_path, start_time, end_time, targetname=target_path)



def crop_video_faces(video_path, new_folder_path, size = (64,64)):
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
                image = image[y:y+h, x:x+w, :]
            
            image = cv2.resize(image,size)
            
            if len(faces) != 0:
                cv2.imwrite(fmt_name, image)
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