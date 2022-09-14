"""Generative adversarial nets in PyTorch"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

# pylint: disable=invalid-name, too-many-locals, too-many-arguments, too-many-instance-attributes

class Generator(nn.Module):
    """Generator network"""
    def __init__(self, latent_dim, img_shape, hidden_dim):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, int(np.prod(img_shape)))
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x.view(-1, *self.img_shape)
        return x

class Discriminator(nn.Module):
    """Discriminator network"""
    def __init__(self, img_shape, hidden_dim):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.fc1 = nn.Linear(int(np.prod(img_shape)), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass"""
        x = x.view(-1, int(np.prod(self.img_shape)))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class GAN:
    """Generative adversarial network"""
    def __init__(self, latent_dim, img_shape, hidden_dim, lr, betas):
        self.img_shape = img_shape
        self.generator = Generator(latent_dim, img_shape, hidden_dim).to(device)
        self.discriminator = Discriminator(img_shape, hidden_dim).to(device)
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=lr, betas=betas)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr, betas=betas)
        self.BceLoss = nn.BCELoss()

    def train(self, dataloader, epochs):
        """Train"""
        for epoch in range(epochs):
            for i, (imgs, _) in enumerate(dataloader):
                batch_size = imgs.shape[0]
                imgs = imgs.view(batch_size, -1).to(device)

                # Adversarial ground truths
                valid = torch.ones(batch_size, 1).to(device)
                fake = torch.zeros(batch_size, 1).to(device)

                # -----------------
                #  Train Generator
                # -----------------

                self.g_optimizer.zero_grad()

                # Sample noise as generator input
                z = torch.randn(batch_size, latent_dim).to(device)

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.BceLoss(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                self.g_optimizer.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.d_optimizer.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.BceLoss(self.discriminator(imgs), valid)
                fake_loss = self.BceLoss(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.d_optimizer.step()

                if i % 100 == 0:
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                    )

                batches_done = epoch * len(dataloader) + i
                if batches_done % 400 == 0:
                    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generative adversarial nets")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--hidden_dim", type=int, default=128, help="dimensionality of the hidden layer")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
    args = parser.parse_args()
    print(args)

    img_shape = (args.channels, args.img_size, args.img_size)

    os.makedirs("images", exist_ok=True)

    cuda = True if torch.cuda.is_available() else False

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(args.latent_dim, img_shape, args.hidden_dim)
    discriminator = Discriminator(img_shape, args.hidden_dim)

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------
    for epoch in range(args.epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, args.epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

            batches_done = epoch * len(dataloader) + i
            if batches_done % args.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                
    # Save the model
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')

# The code above is a modified version of the code from the PyTorch tutorial on GANs. The only difference is that I added the code to save the model. The code to load the model is as follows:
