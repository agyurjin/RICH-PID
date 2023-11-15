import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        
        self.conv_net1 = nn.Sequential(
            nn.Conv2d(1,3,3,1,1),
#             nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d((4,4)),
            nn.Conv2d(3,5,5,1,2),
#             nn.BatchNorm2d(5)
            nn.ReLU()
        )

        self.conv_net2 = nn.Sequential(
            nn.MaxPool2d((2,2)),
            nn.Conv2d(5,2,5,1,2),
#             nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.MaxPool2d((4,4))
        )
        self.fcn = nn.Sequential(
            nn.Linear(input_size+2*6*7, 100),
            nn.ReLU(),
#             nn.BatchNorm1d(100),
            nn.Linear(100, 20),
            nn.ReLU(),
            nn.Linear(20, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, kin_vector, hit_map):
        x = self.conv_net1(hit_map)
        x = F.pad(x,(0,0,1,1))
        x = self.conv_net2(x)
        x = torch.hstack((kin_vector, x.flatten(1)))
        x = x.type(torch.float)
        x = self.fcn(x)
        return x
