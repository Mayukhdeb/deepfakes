import autoencoder
import train_loader_util 
import train_utils

import os 
import cv2

import torch
import torch.nn as nn
from torch import nn, optim

batch_size = 3
train_loader_a = train_loader_util.create_dataloader(image_folder = "datasets/large/A", batch_size = batch_size)
train_loader_b = train_loader_util.create_dataloader(image_folder = "datasets/large/B", batch_size= batch_size)


model = autoencoder.Autoencoder()


optimizer_a = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoder_A.parameters()}]
                         , lr=1e-5, betas=(0.5, 0.999))
optimizer_b = optim.Adam([{'params': model.encoder.parameters()},
                          {'params': model.decoder_B.parameters()}]
                         , lr=1e-5, betas=(0.5, 0.999))

trainer =  train_utils.deepfake_trainer(
    model = model, 
    train_loader_a = train_loader_a,
    train_loader_b = train_loader_b,
    optimizer_a = optimizer_a,
    optimizer_b = optimizer_b  
)

# trainer.train(
#     num_steps = 100
# )

trainer.train(
    num_steps = 350,
    checkpoint_path = "model.pth"
)


