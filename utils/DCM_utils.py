import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rigid_utils import exp_se3

class ReshapeConv(nn.Module):
    def __init__(self):
        super(ReshapeConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = F.normalize(x)
        x = self.activation(x)
        return x
        
class NorConv(nn.Module):
    def __init__(self):
        super(NorConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = F.normalize(x)
        x = self.activation(x)
        return x
        
        
class NorConv1(nn.Module):
    def __init__(self):
        super(NorConv1, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = F.normalize(x)
        x = self.activation(x)
        return x

class DCMNetwork(nn.Module):
    def __init__(self):  # t_multires 6 for D-NeRF; 10 for HyperNeRF
        super(DCMNetwork, self).__init__()
        self.DCM = ReshapeConv()
        self.DCM1 = NorConv()
        self.DCM2 = NorConv()


    def forward(self, x):
        img_tensor = x[0]
        #import pdb;pdb.set_trace()
        grad_x = torch.gradient(img_tensor, dim=1)
        grad_y = torch.gradient(img_tensor, dim=0)
        #import pdb;pdb.set_trace()


        grad_x_x = torch.gradient(grad_x[0], dim=1)
        grad_y_y = torch.gradient(grad_y[0], dim=0)
        #import pdb;pdb.set_trace()

        second_order_edge = torch.abs(grad_x_x[0]) + torch.abs(grad_y_y[0])
        #second_order_edge = grad_x_x[0] + grad_y_y[0]
        second_order_edge = torch.unsqueeze(second_order_edge, dim=0)
        
        x1 = torch.cat((torch.unsqueeze(x[0], dim=0), second_order_edge), dim=0)
        #import pdb;pdb.set_trace()
        x2 = self.DCM(x1) 
        x3 = self.DCM1(x2)
        rad = self.DCM2(x3)
        
        #x2 = x[0] + rad

        return torch.cat([rad] * 3, dim=0)
        