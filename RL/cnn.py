import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import gym
import collections
import random
import torch.nn as nn
import torch.nn.functional as F

class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class dqn(nn.Module):
    def __init__(self):
        super(dqn, self).__init__()
        self.conv1=torch.nn.Conv2d(4, 16, (8, 8), stride=4)
        self.conv2=torch.nn.Conv2d(16, 32, (4, 4), stride=2)
        self.fc1=torch.nn.Linear(2592, 256)
        self.fc2=torch.nn.Linear(256, 2)
    def foward(self, x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
composed = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                           torchvision.transforms.Grayscale(1),
                                           torchvision.transforms.Resize(
                                               (110, 84)),
                                           torchvision.transforms.CenterCrop(
                                               (84, 84)),
                                           torchvision.transforms.ToTensor()])


modela = torch.nn.Sequential(
    torch.nn.Conv2d(4, 16, (8, 8), stride=4),
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 32, (4, 4), stride=2),
    torch.nn.ReLU(),
    Flatten(),
    torch.nn.Linear(2592, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 2)
)
model=dqn()
optimizer = torch.optim.RMSprop(model.parameters(), lr=.00025, momentum=.9)
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
