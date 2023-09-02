import torch
from PIL import Image
import torchvision 
import torchvision.transforms as T
import numpy as np 

class AbstractArtDataset(torch.utils.data.Dataset):
    def __init__(self,list_ids):
        self.list_ids=list_ids
    
    def __len__(self):
        return len(self.list_ids)
    
    def __getitem__(self,index):
        id=self.list_ids[index]

        sample=Image.open("./Data/Abstract_gallery/Abstract_gallery/Abstract_image_"+str(id)+".jpg")

        composed_transforms=T.Compose([T.Resize(size=(256,256)),T.ToTensor(),T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        sample=composed_transforms(sample)

        return sample