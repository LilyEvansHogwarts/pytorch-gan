import torch
from torch import nn

"""
Discriminator and Generator implementation for a simple GAN
"""

class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
                nn.Linear(image_dim, 128),
                nn.LeakyReLU(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid(),
                )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
                nn.Linear(z_dim, 256),
                nn.LeakyReLU(0.1),
                nn.Linear(256, image_dim),
                nn.Tanh(),
                )

    def forward(self, x):
        return self.gen(x)
