import cv2
from deepcake import training_utils
import os 
from deepcake import autoencoder
import matplotlib.pyplot as plt
import numpy as np
import face_alignment

plt.rcParams['figure.figsize'] = 7, 3

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

def find_landmarks(image_np):
    preds = fa.get_landmarks(image_np)[0]
    x = preds[:,0]
    y = preds[:,1]
    
    d = {
        "x": x,
        "y": y
    }
    return d

model = autoencoder.Autoencoder()
inf = training_utils.deepfake_generator(model_class= model, checkpoint_path = "model.pth")

preds = []
target_folder = "data/cropped_frames/elon"


for i in range(10, 200, 30):
    path = target_folder +  "/" + os.listdir(target_folder)[i]
    original_img = cv2.resize(cv2.imread(path), (64,64))
    
    img_b = inf.inference( image_bgr = original_img , decoder = "B")
    img_a = inf.inference(image_bgr = original_img , decoder = "A")

    l = fa.get_landmarks(img_b)  ## need to get landmarks
    # img_c  = (original_img/original_img.max()).astype(np.float32)

    # img_c[12:-12, 12:-12 , :] = cv2.resize(img_b, (40,40))

    fin = cv2.vconcat([ (original_img/original_img.max()).astype(np.float32),img_b, img_a])
    preds.append(fin)


all_preds = cv2.cvtColor(cv2.hconcat(preds), cv2.COLOR_BGR2RGB)
plt.imshow(all_preds)
plt.axis("off")
plt.savefig("output/preds.jpg")
plt.show()
