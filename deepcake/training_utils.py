import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms


import cv2
import numpy as np 


from ._utils import load_images
from ._utils import random_warp

import matplotlib.pyplot as plt

def var_to_np(img_var):
    return img_var.data.cpu().numpy()

mini_transforms = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((64,64 ),Image.BILINEAR),
                            transforms.ToTensor()
                        ])
    
class deepfake_trainer():
    def __init__(self, model, train_loader_a, train_loader_b, optimizer_a, optimizer_b, checkpoint_path = None):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        

        if checkpoint_path is not None:
            print("loading checkpoint: ", checkpoint_path)
            self.model.load_state_dict(torch.load(checkpoint_path))
    
        self.train_loader_a = train_loader_a
        self.train_loader_b = train_loader_b

        self.optimizer_a = optimizer_a
        self.optimizer_b = optimizer_b

        self.criterion = nn.L1Loss()
        print("deepcake trainer initiated on device:", self.device)

    def _train_single_step(self):
        batch_a = next(iter(self.train_loader_a))
        batch_b = next(iter(self.train_loader_b))

        warped_a ,target_a = batch_a["x"].to(self.device), batch_a["y"].to(self.device)
        warped_b, target_b = batch_b["x"].to(self.device), batch_b["y"].to(self.device)

        self.optimizer_a.zero_grad()
        self.optimizer_b.zero_grad()

        out_a = self.model.forward(warped_a, decoder = "A")
        out_b = self.model.forward(warped_b, decoder = "B")
        loss_a = self.criterion(out_a, target_a)
        loss_b = self.criterion(out_b, target_b)

        loss_a.backward()
        loss_b.backward()

        self.optimizer_a.step()
        self.optimizer_b.step()
        
        return loss_a.item(), loss_b.item()
    def train(self, num_steps, checkpoint_path = None, save_path = "model.pth"):

        if checkpoint_path is not None:
            print("loading checkpoint: ", checkpoint_path)

            self.model.load_state_dict(torch.load(checkpoint_path))

        self.model.to(self.device)

        for step in tqdm(range(num_steps)):
            # print("step: ", step)

            try:
                loss_a, loss_b = self._train_single_step()

            except KeyboardInterrupt:
                print("\nungracefully stopping...")
                break

        print('lossA:{}, lossB:{}'.format( loss_a, loss_b))

        torch.save(self.model.state_dict(), save_path)
        print("saved: ", save_path)


class deepfake_trainer_with_landmarks():
    def __init__(self, model, train_loader_a, train_loader_b, optimizer_a, optimizer_b, checkpoint_path = None):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("on device:", self.device)

        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))
    
        self.train_loader_a = train_loader_a
        self.train_loader_b = train_loader_b

        self.optimizer_a = optimizer_a
        self.optimizer_b = optimizer_b

        self.criterion = nn.L1Loss()



    def _train_single_step(self):
        batch_a = next(iter(self.train_loader_a))
        batch_b = next(iter(self.train_loader_b))

        warped_a ,target_a, l_x_a, l_y_a = batch_a["x"].to(self.device), batch_a["y"].to(self.device), batch_a["l_x"].to(self.device), batch_a["l_y"].to(self.device)
        warped_b, target_b, l_x_b, l_y_b = batch_b["x"].to(self.device), batch_b["y"].to(self.device), batch_b["l_x"].to(self.device), batch_b["l_y"].to(self.device)

        pack_a = {
            "x": warped_a,
            "l_x":  l_x_a,
            "l_y": l_y_a

        }

        pack_b = {
            "x": warped_b,
            "l_x":  l_x_b,
            "l_y": l_y_b

        }

        self.optimizer_a.zero_grad()
        self.optimizer_b.zero_grad()

        out_a = self.model.forward( pack_a, decoder = "A")
        out_b = self.model.forward(pack_b ,decoder = "B")
        loss_a = self.criterion(out_a, target_a)
        loss_b = self.criterion(out_b, target_b)

        loss_a.backward()
        loss_b.backward()

        self.optimizer_a.step()
        self.optimizer_b.step()
        
        return loss_a.item(), loss_b.item()
    def train(self, num_steps, checkpoint_path = None, save_path = "model.pth"):

        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))

        self.model.to(self.device)

        for step in tqdm(range(num_steps)):
            # print("step: ", step)

            try:
                loss_a, loss_b = self._train_single_step()

            except KeyboardInterrupt:
                print("ungracefully stopping...")
                break

        print('lossA:{}, lossB:{}'.format( loss_a, loss_b))

        torch.save(self.model.state_dict(), save_path)
        print("saved: ", save_path)



class deepfake_generator():
    def __init__(self, model_class, checkpoint_path):
        self.model = model_class
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.to(self.device)

        
    def inference(self, image_bgr, decoder, crop = 48):

        image = cv2.resize(image_bgr, (256,256))

        if crop is not None:
            image = image[crop:-crop, crop:-crop, :]
            image = cv2.resize(image, (256,256))


        # x, y = random_warp(image)

        # plt.imshow(x)

        # plt.show()

        inp = mini_transforms(image).unsqueeze(0).to(self.device)

        pred = self.model.forward(inp, decoder = decoder).cpu().squeeze(0)

        pred_np = pred.detach().cpu().permute(1,2,0).numpy()

        return pred_np