import torch
from torch import nn
from torch import optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from gan import Discriminator, Generator
from utils import load_model, save_model

# GAN is extremely sensitive to hyperparameters
# Things to try:
# 1. what happens if you use larger networks?
# 2. better normalization with Batch Normalization algorithm
# 3. different learning rates (is there a better one?)
# 4. change architecture to a CNN


# Hyperparameters etc.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 3e-4
z_dim = 64 # 128, 256
image_dim = 28 * 28 * 1 # 784
batch_size = 32
num_epochs = 200
disc_model_file = 'parameters/gan_mnist/discriminator.pth'
gen_model_file = 'parameters/gan_mnist/generator.pth'

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
load_model(disc, disc_model_file, device)
load_model(gen, gen_model_file, device)

fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )
dataset = datasets.MNIST(root='../datasets/', transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784)# .to(device)
        batch_size = real.shape[0]

        # Train Discriminator: max log(D(real)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim)# .to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) -> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx % 200 == 0:
            print("Epoch {}, Loss D: {}, Loss G: {}".format(epoch, lossD, lossG))

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                image_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                image_grid_real = torchvision.utils.make_grid(data, normalize=True)
                writer_fake.add_image(
                        "MNIST Fake Images", image_grid_fake, global_step=step
                        )
                writer_real.add_image(
                        "MNIST Real Images", image_grid_real, global_step=step
                        )
                step += 1

    save_model(disc, disc_model_file)
    save_model(gen, gen_model_file)


