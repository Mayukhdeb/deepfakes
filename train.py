from deepcake import autoencoder
from deepcake import train_loader_utils
from deepcake import training_utils

import os 
import cv2

import torch
import torch.nn as nn
from torch import nn, optim

batch_size = 32

train_loader_a = train_loader_utils.create_dataloader(image_folder = "data/cropped_frames/elon", batch_size = batch_size)
train_loader_b = train_loader_utils.create_dataloader(image_folder = "data/cropped_frames/obama", batch_size= batch_size)  

# print(len(train_loader_b))
# print(len(train_loader_a))

model = autoencoder.Autoencoder()


optimizer_a = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoder_A.parameters()}]
                         , lr=1e-5, betas=(0.5, 0.999))
optimizer_b = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoder_B.parameters()}]
                         , lr = 1e-5, betas=(0.5, 0.999))

trainer =  training_utils.deepfake_trainer(
    model = model, 
    train_loader_a = train_loader_a,
    train_loader_b = train_loader_b,
    optimizer_a = optimizer_a,
    optimizer_b = optimizer_b  
)

# trainer.train(
#     num_steps = 1000,
#     save_path= "models/model.pth"
# )

trainer.train(
    num_steps = 2000,
    checkpoint_path = "models/model.pth",
    save_path= "models/model.pth"
)


