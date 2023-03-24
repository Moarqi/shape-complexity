import os
from typing import Callable
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import Tensor, tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from torchvision.utils import make_grid, save_image
from torch.optim.adam import Adam


class VAE(ABC, nn.Module):
    device = None
    use_std_in_recon = False
    beta = 1

    @abstractmethod
    def encode(self, x):
        pass

    def decode(self, z: Tensor):
        z = self.decode_linear(z)
        z = z.view((-1, 64, 6, 6))
        # z = F.max_unpool2d(z, idx4, (2, 2))
        # z = self.decode4(z)
        z = self.decode3(z)
        z = self.decode2(z)
        z = self.decode1(z)
        # z = z.view(-1, 128, 1, 1)
        # return self.decode_conv(z)
        return z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) if self.training or self.use_std_in_recon else 0
        return mu + eps * std

    def reparameterize_std(self, mu, std):
        eps = torch.randn_like(std) if self.training or self.use_std_in_recon else 0
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def decode_range(self, x: tensor, delta_range: list, index_function: Callable):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        std = torch.exp(0.5 * logvar)
        modifier_index = index_function(std)

        reconstruction_range = torch.zeros((len(delta_range) + 1, 1, 64, 64))
        reconstruction_range[0] = self.decode(z).view(-1, 64, 64)

        for i, delta in enumerate(delta_range):
            altered_std = torch.clone(std)
            altered_std[0, modifier_index] = delta  # NOTE: explicit replace
            z = self.reparameterize_std(mu, altered_std)
            reconstruction_range[i + 1] = self.decode(z).view(-1, 64, 64)

        return (
            make_grid(reconstruction_range.cpu(), nrow=len(delta_range) + 1, padding=0),
            std[0, modifier_index],
            std,
        )

    def loss(self, recon_x, x, mu, logvar):
        """https://github.com/pytorch/examples/blob/main/vae/main.py"""
        BCE = F.binary_cross_entropy(recon_x, x, reduction="sum")

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + self.beta * KLD

    def to(self, device=None):
        if device is not None:
            self.device = device

        return super().to(self.device)

    def train_loop(self, epoch, data_loader, log_interval=2):
        self.train()
        train_loss = 0

        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(self.device)

            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.forward(data)
            loss = self.loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(data_loader.dataset),
                        100.0 * batch_idx / len(data_loader),
                        loss.item() / len(data),
                    )
                )

        return train_loss / len(data_loader.dataset)


class CONVVAE(VAE):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __init__(self, bottleneck=2, lr=1.5e-3, beta=1.0):
        super(CONVVAE, self).__init__()

        self.beta = beta
        self.bottleneck = bottleneck
        self.feature_dim = 6 * 6 * 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> 30x30x16
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> 14x14x32
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # -> 6x6x64
        )

        self.encode_mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, self.bottleneck),
        )
        self.encode_logvar = nn.Sequential(
            nn.Flatten(), nn.Linear(self.feature_dim, self.bottleneck)
        )

        self.decode_linear = nn.Linear(self.bottleneck, self.feature_dim)

        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, stride=2),  # -> 12x12x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3),  # -> 14x14x32
            nn.ReLU(),
        )
        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 2, stride=2),  # -> 28x28x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3),  # -> 30x30x16
            nn.ReLU(),
        )
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 2, stride=2),  # -> 60x60x16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 5),  # -> 64x64x1
            nn.Sigmoid(),
        )

        self.optimizer = Adam(self.parameters(), lr=lr)

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x, idx4 = self.conv4(x)
        mu = self.encode_mu(x)
        logvar = self.encode_logvar(x)

        return mu, logvar


def load_models(load_pretrained=False, beta=1):
    bottlenecks = [16, 64]
    models = {bn: CONVVAE(bottleneck=bn, beta=beta).to() for bn in bottlenecks}

    if load_pretrained:
        for bn, model in models.items():
            model.load_state_dict(
                torch.load(f"trained/{model.__class__.__name__}{bn}_beta{beta}.pth")
            )
            model.eval()

    return list(models.values())


def test(epoch, models: list[VAE], dataset, save_results=False):
    for model in models:
        model.eval()
    test_loss = [0 for _ in models]

    test_batch_size = 32
    sampler = RandomSampler(dataset, replacement=True, num_samples=64)
    test_loader = DataLoader(dataset, batch_size=test_batch_size, sampler=sampler)
    comp_data = None

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(model.device)

            for j, model in enumerate(models):
                recon_batch, mu, logvar = model(data)
                test_loss[j] += model.loss(recon_batch, data, mu, logvar).item()

                if i == 0:
                    n = min(data.size(0), 20)
                    if comp_data == None:
                        comp_data = data[:n]
                    comp_data = torch.cat(
                        [comp_data, recon_batch.view(test_batch_size, 1, 64, 64)[:n]]
                    )

            if i == 0 and save_results:
                if not os.path.exists("results"):
                    os.makedirs("results")
                save_image(
                    comp_data.cpu(),
                    "results/reconstruction_" + str(epoch) + ".png",
                    nrow=min(data.size(0), 20),
                )

    for i, model in enumerate(models):
        test_loss[i] /= len(test_loader.dataset)
        print(f"====> Test set loss model {model.bottleneck}: {test_loss[i]:.4f}")   
