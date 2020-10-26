import torchvision
from torch.utils.data import Dataset, TensorDataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import cv2
from PIL import Image

class image_dataset(Dataset):
    """custom"""

    def __init__(self, image_folder):

        self.image_paths = [image_folder + "/" + i for i in os.listdir(image_folder)]

        self.transforms_input = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((64,64 ),Image.BILINEAR),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomAffine(10, translate=(0.05, 0.05), scale=None, shear=0.05, resample=False, fillcolor=0),
                            transforms.ToTensor()
                        ])

        self.transforms_target = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((64,64 ),Image.BILINEAR),
                            transforms.ToTensor()
                        ])
        
    def __getitem__(self, idx): 
        image = cv2.imread(self.image_paths[idx])
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow("ssss",image)
        # cv2.waitKey()

        ret  = {
                "x": self.transforms_input(image),
                "y": self.transforms_target(image)
        }
        return ret
        
    def __len__(self):
        return len(self.image_paths)

def create_dataloader(image_folder, batch_size, shuffle= True):
    train_dataset = image_dataset(image_folder = image_folder )
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader