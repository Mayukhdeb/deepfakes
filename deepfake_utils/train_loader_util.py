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

device = "cuda" if torch.cuda.is_available() else "cpu"
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device= device)

def find_landmarks(image_np):
    preds = fa.get_landmarks(image_np)[0]
    x = preds[:,0]
    y = preds[:,1]
    
    d = {
        "x": x,
        "y": y
    }
    return d

class image_dataset(Dataset):
    """custom"""

    def __init__(self, image_folder, crop = None):

        self.image_paths = np.sort(np.array([image_folder + "/" + i for i in os.listdir(image_folder)]))
        self.transforms_input = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((64,64 ),Image.BILINEAR),
                            # transforms.RandomHorizontalFlip(p=0.5),
                            # transforms.RandomAffine(1, translate=(0.05, 0.05), scale=None, shear=0.05, resample=False, fillcolor=0),
                            transforms.ToTensor()
                        ])

        self.transforms_target = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((64,64 ),Image.BILINEAR),
                            transforms.ToTensor()
                        ])
        self.crop = crop
        
    def __getitem__(self, idx): 
        image = cv2.imread(self.image_paths[idx])

        if self.crop is not None:
            image = image[self.crop:-self.crop, self.crop:-self.crop]


        ret  = {
                "x": self.transforms_input(image),
                "y": self.transforms_target(image)
        }
        return ret
        
    def __len__(self):
        return len(self.image_paths)

class image_dataset_with_landmarks(Dataset):
    """custom"""

    def __init__(self, image_folder, crop = None):

        self.image_paths = np.sort(np.array([image_folder + "/" + i for i in os.listdir(image_folder)]))
        self.transforms_input = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((64,64 ),Image.BILINEAR),
                            # transforms.RandomHorizontalFlip(p=0.5),
                            # transforms.RandomAffine(1, translate=(0.05, 0.05), scale=None, shear=0.05, resample=False, fillcolor=0),
                            transforms.ToTensor()
                        ])

        self.transforms_target = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((64,64 ),Image.BILINEAR),
                            transforms.ToTensor()
                        ])
        self.crop = crop
        
    def __getitem__(self, idx): 
        image = cv2.imread(self.image_paths[idx])
        
        if self.crop is not None:
            image = image[self.crop:-self.crop, self.crop:-self.crop]

        landmarks = find_landmarks(image)

        l_x, l_y = landmarks["x"], landmarks["y"]
        

        ret  = {
                "x": self.transforms_input(image),
                "y": self.transforms_target(image),
                "l_x": torch.tensor(l_x),
                "l_y": torch.tensor(l_y)
        }


        return ret
        
    def __len__(self):
        return len(self.image_paths)


def create_dataloader(image_folder, batch_size, shuffle= True , crop = None, landmarks = False):
    if landmarks == True:
        train_dataset = image_dataset_with_landmarks(image_folder = image_folder, crop = crop )

    else:
        train_dataset = image_dataset(image_folder = image_folder, crop = crop )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader