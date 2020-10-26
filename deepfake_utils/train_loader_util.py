import torchvision
from torch.utils.data import Dataset, TensorDataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import cv2
from PIL import Image
import numpy as np 

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

def create_dataloader(image_folder, batch_size, shuffle= True , crop = None):
    train_dataset = image_dataset(image_folder = image_folder, crop = crop )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader