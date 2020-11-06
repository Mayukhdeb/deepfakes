import os 
import  cv2
import shutil
import matplotlib.pyplot as plt
import face_alignment


def find_landmarks(image_np):
    preds = fa.get_landmarks(image_np)[0]
    x = preds[:,0]
    y = preds[:,1]
    
    d = {
        "x": x,
        "y": y
    }
    return d

current_folder = os.path.dirname(__file__)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device = "cuda")

def trim_video(source_path, start_time, end_time, target_path):
    ffmpeg_extract_subclip(source_path, start_time, end_time, targetname=target_path)



def crop_video_faces(video_path, new_folder_path, size = (64,64), pad = 0):

    print("padding set to ", pad)
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

        if success == 0:
            pass
        else:
            fmt_name = new_folder_path + "/" + str(count)+ ".jpg"

            try:
                land_marks = find_landmarks(image)

                if land_marks is not None:

                    x = int(land_marks["x"].min())
                    y = int(land_marks["y"].min())

                    w  = int((land_marks["x"].max() - land_marks["x"].min()))

                    h  = int((land_marks["y"].max() - land_marks["y"].min()))

                    image = image[y - pad :y+h + pad, x - pad :x+w + pad , :]

                    image = cv2.resize(image, size)

                    cv2.imwrite(fmt_name, image)
                    count += 1
                    if count % 100 == 0:
                        print(count)

            except KeyboardInterrupt:
                break

    print ("saved ", count-1, " frames at ", new_folder_path)