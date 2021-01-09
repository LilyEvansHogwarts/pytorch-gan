import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
import pylab
from gan import Generator
from utils import load_model


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channels_img = 1
image_size =28
image_dim = image_size * image_size * channels_img
z_dim = 64
gen_file = 'parameters/gan_mnist/generator.pth'
gen = Generator(z_dim, image_dim)
load_model(gen, gen_file, device)

noise = torch.randn(32, z_dim).to(device)
fake = gen(noise).reshape(-1, 1, image_size, image_size)

for idx, image in enumerate(fake):
    pylab.subplot(4, 8, idx+1)
    pylab.imshow(image[0].detach().numpy(), cmap='gray')
    pylab.axis('off')
pylab.tight_layout()
pylab.show()
