import torchvision
from torch.utils.data import Dataset, TensorDataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import torch
import cv2
from PIL import Image
import numpy as np 
import face_alignment
import matplotlib.pyplot as plt

from ._utils import get_image_paths
from ._utils import load_images
from ._utils import random_warp


class image_dataset(Dataset):
    """custom"""

    def __init__(self, image_folder):

        self.image_paths = get_image_paths(image_folder)

        self.all_images = load_images(self.image_paths) / 255.0

        print(self.all_images.max(), self.all_images.min())


        print(self.all_images.shape)
        self.transforms = transforms.Compose([
                            # transforms.ToPILImage(),
                            # transforms.Resize((64,64 ),Image.BILINEAR),
                            # transforms.RandomHorizontalFlip(0.5),
                            # transforms.RandomAffine( 10, translate= (0.05,0.05), scale=(0.05,0.05), shear=None, resample=0, fillcolor=0),
                            transforms.ToTensor()
                        ])

        
    def __getitem__(self, idx): 
        image = self.all_images[idx] 
        

        warped_img, target_img = random_warp(image)

        # print(warped_img.shape, target_img.shape)

        # plt.imshow(target_img)
        # plt.show()

        # plt.imshow(warped_img)
        # plt.show()
        

        im_tensor_x = self.transforms(warped_img)
        im_tensor_y = self.transforms(target_img)



        ret  = {
                "x": im_tensor_x.float(),
                "y": im_tensor_y.float()
        }
        return ret
        
    def __len__(self):
        return len(self.image_paths)


def create_dataloader(image_folder, batch_size, shuffle= True , crop = None):
    
    train_dataset = image_dataset(image_folder = image_folder)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader