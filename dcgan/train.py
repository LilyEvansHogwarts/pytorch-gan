import torch
from torch import nn
from torch import optim
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from dcgan import Discriminator, Generator
from utils import load_model, save_model

"""
Training of DCGAN network on MNIST dataset with Discriminator and Generator
"""

# hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 2e-4
batch_size = 128
image_size = 64
channels_img = 1
z_dim = 100
num_epochs = 30
features_disc = 64
features_gen = 64
disc_file = 'parameters/dcgan_mnist/discriminator.pth'
gen_file = 'parameters/dcgan_mnist/generator.pth'

transforms = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]),
            ]
        )

dataset = datasets.MNIST(root='../datasets/', train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
gen = Generator(z_dim, channels_img, features_gen).to(device)
disc = Discriminator(channels_img, features_disc).to(device)
load_model(disc, disc_file, device)
load_model(gen, gen_file, device)

opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
writer_real = SummaryWriter('runs/dcgan_mnist/real')
writer_fake = SummaryWriter('runs/dcgan_mnist/fake')
step = 0

gen.train()
disc.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = gen(noise)

        # Training Discriminator max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).view(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))

        disc_fake = disc(fake).view(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # Training Generator min log(1 - D(G(z))) -> max log(D(G(z)))
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 200 == 0:
            print("Epoch {}, Loss Discriminator: {}, Loss Generator: {}".format(epoch, loss_disc, loss_gen))

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, image_size, image_size)
                data = real.reshape(-1, 1, image_size, image_size)
                image_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                image_grid_real = torchvision.utils.make_grid(real, normalize=True)
                writer_fake.add_image(
                        "MNIST Fake Images", image_grid_fake, global_step=step
                        )
                writer_real.add_image(
                        "MNIST Real Images", image_grid_real, global_step=step
                        )
                step += 1

    save_model(disc, disc_file)
    save_model(gen, gen_file)

