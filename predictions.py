import torch 
import torchvision 
import torchvision.transforms as T 
import numpy as np 
import matplotlib.pyplot as plt
from generator import Generator

if __name__=='__main__':
    device=torch.device('mps')

    model=Generator().to(device=device)
    model.load_state_dict(torch.load("./models/abstract_art/generator850.pth"))

    latent_space=torch.randn((1,100)).to(device=device)

    pred=model(latent_space)

    pred=pred.to(device='cpu')
    pred=pred.detach().numpy()

    pred_np=pred.transpose(0,3,2,1)
    print(pred_np.shape)

    #visualize
    plt.imshow(pred_np[0])
    plt.show()