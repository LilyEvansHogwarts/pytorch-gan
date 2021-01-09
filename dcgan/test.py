import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import pylab
from dcgan import Generator
from utils import load_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channels_img = 1
features_gen = 64
image_size = 64
z_dim = 100
gen_file = 'parameters/dcgan_mnist/generator.pth'
gen = Generator(z_dim, channels_img, features_gen)
load_model(gen, gen_file, device)

noise = torch.randn(32, z_dim, 1, 1).to(device)
fake = gen(noise).reshape(-1, 1, image_size, image_size)

for idx, image in enumerate(fake):
    pylab.subplot(4, 8, idx+1)
    pylab.imshow(image[0].detach().numpy(), cmap='gray')
    pylab.axis('off')
pylab.tight_layout()
pylab.show()
