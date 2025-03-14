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

class TCMNetwork(nn.Module):
    def __init__(self):  # t_multires 6 for D-NeRF; 10 for HyperNeRF
        super(TCMNetwork, self).__init__()
        self.TCM = ReshapeConv()
        self.TCM1 = NorConv()
        self.TCM2 = NorConv()


    def forward(self, x):
        img_tensor = x[0]
        grad_x = torch.gradient(img_tensor, dim=1)
        grad_y = torch.gradient(img_tensor, dim=0)


        grad_x_x = torch.gradient(grad_x[0], dim=1)
        grad_y_y = torch.gradient(grad_y[0], dim=0)

        second_order_edge = torch.abs(grad_x_x[0]) + torch.abs(grad_y_y[0])
        second_order_edge = torch.unsqueeze(second_order_edge, dim=0)
        
        x1 = torch.cat((torch.unsqueeze(x[0], dim=0), second_order_edge), dim=0)

        # For nn.Conv2d() with broadcasting mechanism, used in the original paper
        x2 = self.TCM(x1) 
        x3 = self.TCM1(x2)
        rad = self.TCM2(x3)


        # For nn.Conv2d() without broadcasting mechanism
        # X1 = x1.unsqueeze(0)
        # x2 = self.TCM(x1) 
        # x3 = self.TCM1(x2)
        # rad = self.TCM2(x3).squeeze(0)
        

        return torch.cat([rad] * 3, dim=0)
        
