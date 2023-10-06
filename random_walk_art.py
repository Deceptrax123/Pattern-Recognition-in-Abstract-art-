#random walk into the latent space from the DCGAN paper

import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import torchvision 
from generator import Generator
import torchvision.transforms as T
from PIL import Image

if __name__=='__main__':

    device=torch.device('mps')

    model=Generator().to(device=device)

    model.eval()
    model.load_state_dict(torch.load("./models/abstract_art/resized/generator/stable/generator250.pth"))

    z1=torch.randn((1,100)).to(device=device) #noise
    z2=torch.randn((1,100)).to(device=device) #noise
    z3=torch.randn((1,100)).to(device=device) #noise

    z4=torch.randn((1,100)).to(device=device) #noise
    z5=torch.randn((1,100)).to(device=device) #noise
    z6=torch.randn((1,100)).to(device=device) #noise

    vec1=model(z1)
    vec2=model(z2) 
    vec3=model(z3)
    vec4=model(z4)
    vec5=model(z5)
    vec6=model(z6)

    vec1=(vec2+vec4)/2
    vec2=(vec1+vec3)/2
    vec3=(vec5+vec6)/2

    #vector arithmetic
    v=(vec3+vec2-vec1)

    v2=(vec1+vec2-vec3)

    v3=(vec1-vec2+vec3)


    vec1=vec1.to(device='cpu')
    vec2=vec2.to(device='cpu')
    vec3=vec3.to(device='cpu')
    v=v.to(device='cpu')
    v2=v2.to(device='cpu')
    v3=v3.to(device='cpu')

    v=v.detach().numpy()
    v2=v2.detach().numpy()
    vec1=vec1.detach().numpy()
    vec2=vec2.detach().numpy()
    vec3=vec3.detach().numpy()
    v3=v3.detach().numpy()


    #transpose
    v=v.transpose(0,3,2,1)
    v2=v2.transpose(0,3,2,1)
    vec1=vec1.transpose(0,3,2,1)
    vec2=vec2.transpose(0,3,2,1)
    vec3=vec3.transpose(0,3,2,1)
    v3=v3.transpose(0,3,2,1)

    #plt.imshow(v3[0])

    #plot
    fig=plt.figure(figsize=(10,10))
 
    ax1=fig.add_subplot(2,3,1)
    ax1.imshow(vec1[0])
    ax1.set_title("V1")

    ax2=fig.add_subplot(2,3,2)
    ax2.imshow(vec2[0])
    ax2.set_title("V2")

    ax3=fig.add_subplot(2,3,3)
    ax3.imshow(vec3[0])
    ax3.set_title("V3")

    fig.suptitle("A random walk to explore the Latent Space")

    ax4=fig.add_subplot(2,3,4)
    ax4.imshow(v[0])
    ax4.set_title("V2+V3-V1")

    ax5=fig.add_subplot(2,3,5)
    ax5.imshow(v2[0])
    ax5.set_title("V1+V2-V3")

    ax6=fig.add_subplot(2,3,6)
    ax6.imshow(v3[0])
    ax6.set_title("V1-V2+V3")

    plt.show()
