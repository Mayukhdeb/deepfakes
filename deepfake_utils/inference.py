import cv2
import train_utils
import os 
import autoencoder
import matplotlib.pyplot as plt
import numpy as np

model = autoencoder.Autoencoder()
inf = train_utils.deepfake_generator(model_class= model, checkpoint_path = "model.pth")

preds = []

for i in range(10, 30, 1):
    path = "datasets/large/A/" + os.listdir("datasets/large/A")[i]
    original_img = cv2.resize(cv2.imread(path), (64,64))
    
    img_b = inf.inference(image_path = path , decoder = "B")
    img_a = inf.inference(image_path = path , decoder = "A")


    fin = cv2.vconcat([original_img.astype(np.float32)/original_img.max(), img_a, img_b])
    preds.append(fin)


all_preds = cv2.cvtColor(cv2.hconcat(preds), cv2.COLOR_BGR2RGB)
plt.imshow(all_preds)
plt.axis("off")
# plt.ylabel("A, A to A, A to B")
plt.savefig("output/preds.jpg")
plt.show()