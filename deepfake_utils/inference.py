import cv2
from deepcake import training_utils
import os 
from deepcake import autoencoder
import matplotlib.pyplot as plt
import numpy as np

model = autoencoder.Autoencoder()
inf = training_utils.deepfake_generator(model_class= model, checkpoint_path = "model.pth")

preds = []
target_folder = "data/cropped_frames/elon"


for i in range(10, 30, 2):
    path = target_folder +  "/" + os.listdir(target_folder)[i]
    original_img = cv2.resize(cv2.imread(path), (64,64))
    
    img_b = inf.inference(image_path = path , decoder = "B")
    img_a = inf.inference(image_path = path , decoder = "A")



    fin = cv2.vconcat([original_img.astype(np.float32)/original_img.max(), img_b, img_a])
    preds.append(fin)


all_preds = cv2.cvtColor(cv2.hconcat(preds), cv2.COLOR_BGR2RGB)
plt.imshow(all_preds)
plt.axis("off")
plt.savefig("output/preds_2.jpg")
plt.show()