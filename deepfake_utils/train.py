import autoencoder
import train_loader_util 
import train_utils

import os 
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import nn, optim


# train_loader_a = train_loader_util.create_dataloader(image_folder = "datasets/A", batch_size = 5)
# train_loader_b = train_loader_util.create_dataloader(image_folder = "datasets/B", batch_size= 5)


model = autoencoder.Autoencoder()


# optimizer_a = optim.Adam([{'params': model.encoder.parameters()},
#                           {'params': model.decoder_A.parameters()}]
#                          , lr=5e-5, betas=(0.5, 0.999))
# optimizer_b = optim.Adam([{'params': model.encoder.parameters()},
#                           {'params': model.decoder_B.parameters()}]
#                          , lr=5e-5, betas=(0.5, 0.999))

# trainer =  train_utils.deepfake_trainer(
#     model = model, 
#     train_loader_a = train_loader_a,
#     train_loader_b = train_loader_b,
#     optimizer_a = optimizer_a,
#     optimizer_b = optimizer_b  
# )

# trainer.train(
#     num_steps = 100
# )

# trainer.train(
#     num_steps = 10,
#     checkpoint_path = "model.pth"
# )

path = "datasets/A/" + os.listdir("datasets/A")[0]

orignal_img = cv2.imread(path)

inf = train_utils.deepfake_generator(model_class= model, checkpoint_path = "model.pth")

plt.imshow(orignal_img)
plt.show()

for i in range(5):
    path = "datasets/A/" + os.listdir("datasets/A")[i]

    img = inf.inference(image_path = path , decoder = "A")

    plt.imshow(img)
    plt.show()
