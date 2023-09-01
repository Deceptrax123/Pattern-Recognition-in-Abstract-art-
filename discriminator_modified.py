import torch 
from torch import nn 

from torch.nn import Module 
from torch.nn import Conv2d,BatchNorm2d,Flatten,Sigmoid,LeakyReLU,ConvTranspose2d,Linear,Dropout2d,ReLU
from prettytable import PrettyTable

from torchsummary import summary

class Discriminator(Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.bconv21=Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),stride=1,padding=1)
        self.bconv22=Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),stride=1,padding=1)

        self.bconv31=Conv2d(in_channels=32,out_channels=32,stride=1,padding=1,kernel_size=(3,3))
        self.bconv32=Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),stride=1,padding=1)

        self.bconv41=Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),padding=1,stride=1)
        self.bconv42=Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),padding=1,stride=1)

        self.bconv51=Conv2d(in_channels=128,out_channels=128,padding=1,kernel_size=(3,3),stride=1)
        self.bconv52=Conv2d(in_channels=128,out_channels=128,padding=1,kernel_size=(3,3),stride=1)

        #activations for blocks

        self.brelu21=ReLU()
        self.brelu22=ReLU()

        self.brelu31=ReLU()
        self.brelu32=ReLU()

        self.brelu41=ReLU()
        self.brelu42=ReLU()

        self.brelu51=ReLU()
        self.brelu52=ReLU()

        #batch norms for blocks

        self.bn21=BatchNorm2d(16)
        self.bn22=BatchNorm2d(16)

        self.bn31=BatchNorm2d(32)
        self.bn32=BatchNorm2d(32)

        self.bn41=BatchNorm2d(64)
        self.bn42=BatchNorm2d(64)

        self.bn51=BatchNorm2d(128)
        self.bn52=BatchNorm2d(128)


        self.conv1=Conv2d(in_channels=3,out_channels=8,stride=2,kernel_size=(3,3),padding=1)
        self.bn1=BatchNorm2d(8)
        self.dp1=Dropout2d(p=0.2)
        self.lr1=LeakyReLU(negative_slope=0.2)

        self.conv2=Conv2d(in_channels=8,out_channels=16,stride=2,kernel_size=(3,3),padding=1)
        self.dp2=Dropout2d(p=0.2)
        self.bn2=BatchNorm2d(num_features=16)
        self.lr2=LeakyReLU(negative_slope=0.2)

        self.conv3=Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=2,padding=1)
        self.dp3=Dropout2d(p=0.2)
        self.bn3=BatchNorm2d(32)
        self.lr3=LeakyReLU(negative_slope=0.2)

        self.conv4=Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=2,padding=1)
        self.dp4=Dropout2d(p=0.2)
        self.bn4=BatchNorm2d(64)
        self.lr4=LeakyReLU(negative_slope=0.2)

        self.conv5=Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=2,padding=1)
        self.dp5=Dropout2d(p=0.2)
        self.bn5=BatchNorm2d(128)
        self.lr5=LeakyReLU(negative_slope=0.2)

        self.conv6=Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=2,padding=1)
        self.dp6=Dropout2d(p=0.2)
        self.bn6=BatchNorm2d(256)
        self.lr6=LeakyReLU(negative_slope=0.2)

        self.conv7=Conv2d(in_channels=256,out_channels=1,kernel_size=(4,4),stride=1)

    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.dp1(x)
        x=self.lr1(x)

        x=self.conv2(x)
        x=self.dp2(x)
        x=self.bn2(x)
        x=self.lr2(x)

        #Block 2
        x=self.bconv21(x)
        x=self.bn21(x)
        x=self.brelu21(x)
        x=self.bconv22(x)
        x=self.bn22(x)
        x=self.brelu22(x)

        x=self.conv3(x)
        x=self.dp3(x)
        x=self.bn3(x)
        x=self.lr3(x)

        #block 3
        x=self.bconv31(x)
        x=self.bn31(x)
        x=self.brelu31(x)
        x=self.bconv32(x)
        x=self.bn32(x)
        x=self.brelu32(x)

        x=self.conv4(x)
        x=self.dp4(x)
        x=self.bn4(x)
        x=self.lr4(x)

        #block 4
        x=self.bconv41(x)
        x=self.bn41(x)
        x=self.brelu41(x)
        x=self.bconv42(x)
        x=self.bn42(x)
        x=self.brelu42(x)

        x=self.conv5(x)
        x=self.dp5(x)
        x=self.bn5(x)
        x=self.lr5(x)
        
        #block 5
        x=self.bconv51(x)
        x=self.bn51(x)
        x=self.brelu51(x)
        x=self.bconv52(x)
        x=self.bn52(x)
        x=self.brelu52(x)

        x=self.conv6(x)
        x=self.dp6(x)
        x=self.bn6(x)
        x=self.lr6(x)

        x=self.conv7(x)

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

#model = Discriminator()
# # # # # count_parameters(model)
#summary(model,input_size=(3,256,256),batch_size=8,device='cpu')