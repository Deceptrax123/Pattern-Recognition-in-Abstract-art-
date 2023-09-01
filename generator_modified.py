import torch
from torch import nn
import torchvision

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import ConvTranspose2d
from torch.nn import BatchNorm2d,Linear,Flatten

from prettytable import PrettyTable
from torchsummary import summary

class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()

        #project noise of latent space to a 4d stack
        self.linear=Linear(in_features=100,out_features=16384)

        #create blocks for each convolution inspired by U-Net
        self.bconv11=Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=1)
        self.bconv12=Conv2d(in_channels=512,out_channels=512,kernel_size=(3,3),stride=1,padding=1)

        self.bconv21=Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=1,padding=1)
        self.bconv22=Conv2d(in_channels=256,out_channels=256,kernel_size=(3,3),stride=1,padding=1)

        self.bconv31=Conv2d(in_channels=128,out_channels=128,stride=1,padding=1,kernel_size=(3,3))
        self.bconv32=Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),stride=1,padding=1)

        self.bconv41=Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),padding=1,stride=1)
        self.bconv42=Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),padding=1,stride=1)

        self.bconv51=Conv2d(in_channels=32,out_channels=32,padding=1,kernel_size=(3,3),stride=1)
        self.bconv52=Conv2d(in_channels=32,out_channels=32,padding=1,kernel_size=(3,3),stride=1)

        #activations for blocks
        self.brelu11=ReLU()
        self.brelu12=ReLU()

        self.brelu21=ReLU()
        self.brelu22=ReLU()

        self.brelu31=ReLU()
        self.brelu32=ReLU()

        self.brelu41=ReLU()
        self.brelu42=ReLU()

        self.brelu51=ReLU()
        self.brelu52=ReLU()

        #batch norms for blocks
        self.bn11=BatchNorm2d(512)
        self.bn12=BatchNorm2d(512)

        self.bn21=BatchNorm2d(256)
        self.bn22=BatchNorm2d(256)

        self.bn31=BatchNorm2d(128)
        self.bn32=BatchNorm2d(128)

        self.bn41=BatchNorm2d(64)
        self.bn42=BatchNorm2d(64)

        self.bn51=BatchNorm2d(32)
        self.bn52=BatchNorm2d(32)


        #projection blocks
        self.conv1 = ConvTranspose2d(in_channels=1024, out_channels=512,
                            kernel_size=(4, 4), stride=2,padding=1)
        self.bn1=BatchNorm2d(512)
        self.relu1 = ReLU()

        self.conv2 = ConvTranspose2d(in_channels=512, out_channels=256,
                            kernel_size=(4, 4), stride=2,padding=1)
        self.bn2 = BatchNorm2d(num_features=256)
        self.relu2 = ReLU()

        self.conv3 = ConvTranspose2d(in_channels=256, out_channels=128,
                            kernel_size=(4, 4), stride=2,padding=1)
        self.bn3 = BatchNorm2d(num_features=128)
        self.relu3 = ReLU()

        self.conv4=ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(4,4),stride=2,padding=1)
        self.bn4=BatchNorm2d(64)
        self.relu4=ReLU()

        self.conv5=ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=(4,4),stride=2,padding=1)
        self.bn5=BatchNorm2d(32)
        self.relu5=ReLU()
        
        self.conv6=ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=(4,4),padding=1,stride=2)
        self.tanh = Tanh()

    def forward(self, x):
        x=self.linear(x)
        x=x.view(x.size(0),1024,4,4)

        #first upsample
        x = self.conv1(x)
        x = self.bn1(x)
        x= self.relu1(x)

        #block 1
        x=self.bconv11(x)
        x=self.bn11(x)
        x=self.brelu11(x)
        x=self.bconv12(x)
        x=self.bn12(x)
        x=self.brelu12(x)

        #second upsample
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        #Block 2
        x=self.bconv21(x)
        x=self.bn21(x)
        x=self.brelu21(x)
        x=self.bconv22(x)
        x=self.bn22(x)
        x=self.bconv22(x)

        #Upsample 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        #Block 3
        x=self.bconv31(x)
        x=self.bn31(x)
        x=self.brelu31(x)
        x=self.bconv32(x)
        x=self.bn32(x)
        x=self.brelu32(x)

        #Upsample 4
        x=self.conv4(x)
        x=self.bn4(x)
        x=self.relu4(x)

        #Block 4
        x=self.bconv41(x)
        x=self.bn41(x)
        x=self.brelu41(x)
        x=self.bconv42(x)
        x=self.bn42(x)
        x=self.brelu42(x)

        #upsample 5
        x=self.conv5(x)
        x=self.bn5(x)
        x=self.relu5(x)

        #block 5
        x=self.bconv51(x)
        x=self.bn51(x)
        x=self.brelu51(x)
        x=self.bconv52(x)
        x=self.bn52(x)
        x=self.brelu52(x)
    

        x=self.conv6(x)
        x=self.tanh(x)

        return x


# def count_parameters(model):
#     table = PrettyTable(['Modules', 'Parameters'])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params += params
#     print(table)
#     print(f'Total Trainable Params: {total_params}')
#     return total_params


model = Generator()
summary(model,input_size=(100,),batch_size=8,device='cpu')