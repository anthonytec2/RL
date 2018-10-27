import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import gym
import collections
import random


class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


composed = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                           torchvision.transforms.Grayscale(1),
                                           torchvision.transforms.Resize(
                                               (110, 84)),
                                           torchvision.transforms.CenterCrop(
                                               (84, 84)),
                                           torchvision.transforms.ToTensor()])


model = torch.nn.Sequential(
    torch.nn.Conv2d(4, 16, (8, 8), stride=4),
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 32, (4, 4), stride=2),
    torch.nn.ReLU(),
    Flatten(),
    torch.nn.Linear(2592, 2)
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
