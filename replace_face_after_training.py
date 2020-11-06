from deepcake import autoencoder
from deepcake import training_utils
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil
import numpy as np


model = autoencoder.Autoencoder()
inf = training_utils.deepfake_generator(model_class= model, checkpoint_path = "models/model.pth")
print("on: ", inf.device)


haarcascade_path = "deepcake/" + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier( haarcascade_path)

image_folder = "data/cropped_frames/elon"

image_paths = [image_folder + "/" + str(i) + ".jpg" for i in range(len(os.listdir(image_folder)))]


shutil.rmtree("fakes")
os.mkdir("fakes")


def write_video_from_image_list(save_name, all_images_np, framerate, size):
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    out = cv2.VideoWriter(save_name ,fourcc, framerate, size)

    for i in range(all_images_np.shape[0]):
        
        frame = all_images_np[i].astype(np.uint8)

        out.write(frame)
    out.release()



all_images = []
for i in tqdm(range(len(image_paths[:1000]))):
    path = image_paths[i]

    original_img_1 = cv2.resize(cv2.imread(path), (64,64))
    
    img_b = inf.inference( image_bgr = original_img_1 , decoder = "B")

    name = "fakes/" + str(i) + ".jpg"

    
    img_b_1= ((255*img_b).astype(np.uint8) *0.8 + original_img_1*0.3).astype(np.uint8)


    path = image_paths[i+1000]


    original_img_2 = cv2.resize(cv2.imread(path), (64,64))
    
    img_b = inf.inference( image_bgr = original_img_2 , decoder = "B")

    name = "fakes/" + str(i) + ".jpg"

    
    img_b_2 = ((255*img_b).astype(np.uint8) *0.8 + original_img_2*0.3).astype(np.uint8)


    img_c_1 = cv2.hconcat([original_img_1, img_b_1])

    img_c_2 = cv2.hconcat([original_img_2, img_b_2])

    img_f = cv2.vconcat([img_c_1, img_c_2])
    all_images.append(img_f)

all_images = np.array(all_images)
write_video_from_image_list("output/fake.mp4", all_images, framerate = 20, size = (128,128))
