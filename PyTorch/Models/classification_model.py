import torch 
from torch import nn 
import torch.nn.functional as F

class net_classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(1, 32, kernel_size=7)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2d2 = nn.Conv2d(32, 64,kernel_size=3)        
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2d3 = nn.Conv2d(64,64,kernel_size=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.dense = nn.Linear(64*125*29,256)
        self.drop= nn.Dropout(0.5)
        self.dense2 = nn.Linear(256,128)

        self.arm = nn.Linear(128, 2)
        self.head = nn.Linear(128,2)
        self.leg = nn.Linear(128, 3)
        self.softmax2 = nn.Softmax()
        self.softmax3 = nn.Softmax(3)
       
    def forward(self, x):
        x = self.conv2d(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.conv2d2(x)
        x = F.relu(x)
        x = self.maxpool2(x)
        x = self.conv2d3(x)
        x = F.relu(x)
        x = self.maxpool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dense(x)
        x = self.drop(x)
        x = self.dense2(x)
        x = self.drop(x)
        head = self.head(x)
        #head = self.softmax2(x)
        right_arm = self.arm(x)
        #right_arm = self.softmax2(x)
        left_arm = self.arm(x)
        #left_arm = self.softmax2(x)
        leg =self.leg(x)
        #leg = self.softmax2(x)
        return head, leg, right_arm, left_arm