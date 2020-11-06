from deepcake import autoencoder
from deepcake import training_utils
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil


model = autoencoder.Autoencoder()
inf = training_utils.deepfake_generator(model_class= model, checkpoint_path = "models/model.pth")
print("on: ", inf.device)


haarcascade_path = "deepcake/" + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier( haarcascade_path)

image_folder = "data/cropped_frames/elon"

image_paths = [image_folder + "/" + str(i) + ".jpg" for i in range(len(os.listdir(image_folder)))]


shutil.rmtree("fakes")
os.mkdir("fakes")


for i in tqdm(range(len(image_paths[:1000]))):
    path = image_paths[i]
    original_img = cv2.resize(cv2.imread(path), (64,64))
    
    img_b = inf.inference( image_bgr = original_img , decoder = "B")

    name = "fakes/" + str(i) + ".jpg"

    cv2.imwrite(name, 255*img_b)

cv2.destroyAllWindows()

