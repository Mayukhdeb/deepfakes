import torchvision
from torch.utils.data import Dataset, TensorDataset
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

class C_Dataset(Dataset):
    """custom"""

    def __init__(self, image_folder_a, image_folder_b, transforms):

        self.image_paths_a = [image_folder_a + i for i in os.listdir(image_folder_a)]
        self.image_paths_b = [image_folder_b + i for i in os.listdir(image_folder_b)]

        
    def __getitem__(self, idx): 


    def __len__(self):
        return 
      
        
