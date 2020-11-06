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

import albumentations as A


class image_dataset(Dataset):
    """custom"""

    def __init__(self, image_folder):

        self.image_paths = get_image_paths(image_folder)
        print(len(self.image_paths), " images found in ",image_folder)

        



        # print(self.all_images.shape)
        self.transforms = transforms.Compose([
                            # transforms.ToPILImage(),
                            # transforms.Resize((64,64 ),Image.BILINEAR),
                            # transforms.RandomHorizontalFlip(0.5),
                            # transforms.RandomAffine( 10, translate= (0.05,0.05), scale=(0.05,0.05), shear=None, resample=0, fillcolor=0),
                            transforms.ToTensor()
                        ])

        self.augmentations =  A.Compose([
            A.HorizontalFlip(p=0.5)
        ])

        
    def __getitem__(self, idx): 
        image_path = self.image_paths[idx] 

        image = cv2.imread(image_path)/255.0

        image = cv2.resize(image, (256,256))
        
        
        image, label = random_warp(image)


        augmented = self.augmentations(image = image, label = label)

        warped_img, target_img = augmented["image"], augmented["label"]
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



    '''

    capsicum   200 
    bean   200
    gajar  4-5
    tomat  500
    saag 
    chaa pata
    boost 
    butter check 

    '''

