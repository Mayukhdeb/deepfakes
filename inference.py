import cv2
from deepcake import training_utils
import os 
from deepcake import autoencoder
import matplotlib.pyplot as plt
import numpy as np
import face_alignment

plt.rcParams['figure.figsize'] = 7, 3

# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

# def find_landmarks(image_np):
#     preds = fa.get_landmarks(image_np)[0]
#     x = preds[:,0]
#     y = preds[:,1]
    
#     d = {
#         "x": x,
#         "y": y
#     }
#     return d

model = autoencoder.Autoencoder()
inf = training_utils.deepfake_generator(model_class= model, checkpoint_path = "models/model.pth")

preds = []
target_folder = "data/cropped_frames/elon"


for i in range(10, 400, 40):
    path = target_folder +  "/" + os.listdir(target_folder)[i]
    original_img = cv2.resize(cv2.imread(path), (64,64))
    
    img_b = inf.inference( image_bgr = original_img , decoder = "B")
    img_c = (original_img/original_img.max()) * 0.2 + img_b * 0.8

    fin = cv2.vconcat([ (original_img/original_img.max()).astype(np.float32),img_b])# img_c.astype(np.float32)])
    preds.append(fin)


all_preds = cv2.cvtColor(cv2.hconcat(preds), cv2.COLOR_BGR2RGB)
plt.imshow(all_preds)
plt.axis("off")
# plt.title("I bring to you: obama musk")
plt.savefig("output/preds_aligned.jpg")
plt.show()
