import torch 
import torchvision 
import torchvision.transforms as T 
import numpy as np 
import matplotlib.pyplot as plt
from generator import Generator

from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter
from skimage.filters import median
from skimage.morphology import disk

if __name__=='__main__':
    device=torch.device('mps')

    model=Generator().to(device=device)
    model.load_state_dict(torch.load("./models/abstract_art/modified_arch/generator/generator60.pth"))

    latent_space=torch.randn((1,100)).to(device=device)

    pred=model(latent_space)
 
    pred=pred.to(device='cpu')
    pred=pred.detach().numpy()

    pred_np=pred.transpose(0,3,2,1)
    
    #apply gaussian filter
    image_filtered=gaussian_filter(pred_np[0],sigma=0.03,mode='mirror')
    #print(image_filtered)

    #apply median filter
    #footprint_function=disk
    #fp=footprint_function(radius=2)
    #image_median_filter=median(pred_np[0])


    plt.imshow(image_filtered)
    plt.show()