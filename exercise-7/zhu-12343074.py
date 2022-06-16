import numpy as np
import pylab as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):

    def __init__(self, num_channels=1, num_classes=10, latent_dim=2, embed_dim=16):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embed_dim)

        self.encoder = nn.ModuleList([
            nn.Conv2d(in_channels=num_channels, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
        ])

        self.decoder = nn.ModuleList([
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3),
            nn.Conv2d(in_channels=8, out_channels=num_channels, kernel_size=3, padding=1),
        ])
        self.fc_latent = nn.Linear(in_features=latent_dim + embed_dim, out_features=512)

        self.fc_mean = nn.Linear(in_features=512 + embed_dim, out_features=latent_dim)
        self.fc_var = nn.Linear(in_features=512 + embed_dim, out_features=latent_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
        Args:
            x (tensor): Image(s) of shape [B, C, H, W].
            y (tensor): Class label(s) of shape [B,].

        Returns:
            x_recon (tensor): Reconstructed image(s) of shape [B, C, H, W].
            mean (tensor): Mean of shape [B, latent_dim].
            log_var (tensor): Log variance of shape [B, latent_dim].
        """
        mean, log_var = self.encode(x, y)
        # Reparameterization Trick
        eps = torch.randn(log_var.shape, device=log_var.device)
        z = mean + torch.exp(log_var * 0.5) * eps
        x_recon = self.decode(z, y)
        return x_recon, mean, log_var

    def encode(self, x, y):
        """
        Args:
            x (tensor): Image(s) of shape [B, C, H, W].
            y (tensor): Class label(s) of shape [B,].

        Returns:
            mean (tensor): Mean of shape [B, latent_dim].
            log_var (tensor): Log variance of shape [B, latent_dim].
        """
        for layer in self.encoder:
            x = layer(x)
            x = self.leaky_relu(x)
        x = torch.reshape(x, (x.shape[0], -1))
        class_embed = self.embedding(y)
        # Concat class information
        mean = self.fc_mean(torch.cat((x, class_embed), dim=1))
        log_var = self.fc_var(torch.cat((x, class_embed), dim=1))
        return mean, log_var

    def decode(self, z, y):
        """
        Args:
            z (tensor): Latent variable(s) of shape [B, latent_dim].
            y (tensor): Class label(s) of shape [B,].

        Returns:
            x (tensor): Reconstructed image(s) of shape [B, C, H, W].
        """
        class_embed = self.embedding(y)
        # Concat class information
        x = self.fc_latent(torch.cat((z, class_embed), dim=1))
        x = torch.reshape(x, (-1, 32, 4, 4))
        for layer in self.decoder:
            x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = self.leaky_relu(x)
            x = layer(x)
        x = self.sigmoid(x)
        return x

    def sample(self, y, device):
        """
        Args:
            y (int): Class label.
            device (torch.device): Which device to use (cuda or cpu).

        Returns:
            (tensor): Image of shape [1, C, H, W].
        """
        z = torch.randn((1, self.latent_dim), device=device)
        return self.decode(z, torch.tensor([y], device=device))

    def sample_latent(self, x, y):
        """
        Args:
            x (tensor): Image(s) of shape [B, C, H, W].
            y (tensor): Class label(s) of shape [B,].

        Returns:
            z (tensor): Latent variable(s) of shape [B, latent_dim].
        """
        mean, log_var = self.encode(x, y)
        # Reparameterization Trick
        eps = torch.randn(log_var.shape, device=log_var.device)
        z = mean + torch.exp(log_var * 0.5) * eps
        return z


def data_loader():
    TrainData = datasets.MNIST(root="../data", train=True, download=True, transform=transforms.ToTensor())
    TestData = datasets.MNIST(root="../data", train=False, download=True, transform=transforms.ToTensor())
    train_dataloader = DataLoader(TrainData, batch_size=128, shuffle=True)
    test_dataloader = DataLoader(TestData, batch_size=128, shuffle=True)
    return train_dataloader, test_dataloader


def model_training(n_epoch, train_dataloader):
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for epoch in range(n_epoch):
        loss_batch = []
        for step, [x, label] in enumerate(tqdm(train_dataloader, desc='Batch')):
            # forward
            x, label = x.to(device), label.to(device)
            x_recon, mean, log_var = vae(x, label)
            MSE = mse(x_recon, x)
            KL = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            loss = MSE + KL * 0.001
            # save the loss in each step
            loss_batch.append(loss.item())
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1) % 100 == 0:
                print("\nEpoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                      .format(epoch + 1, n_epoch, step + 1, len(train_dataloader), MSE.item(), KL.item()))
        # loss_average = np.sum(loss_batch) / dataloader.batch_size
        plt.plot(np.asarray(loss_batch))
        plt.xlabel('Step')
        plt.ylabel('loss')
        plt.title("Average loss per step trained in Epoch-" + str(epoch + 1))
        plt.show()

        for i in range(10):
            plt.subplot(5, 2, i + 1)
            img = vae.sample(i, device)
            img = img[0].permute(1, 2, 0).cpu().detach().numpy()
            plt.imshow(img, cmap='gray')
        plt.show()


def test_3_1(n_epoch, train_dataloader, test_dataloader):
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for epoch in range(n_epoch):
        for step, [x, label] in enumerate(tqdm(train_dataloader, desc='Batch')):
            # forward
            x, label = x.to(device), label.to(device)
            x_recon, mean, log_var = vae(x, label)
            MSE = mse(x_recon, x)
            KL = 0.5 * torch.sum(log_var.exp() + mean.pow(2) - 1. - log_var)
            loss = MSE
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    embed = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    for idx, [test_ex, test_l] in enumerate(tqdm(test_dataloader, desc='Test')):
        test_ex = test_ex.to(device)
        test_l = test_l.to(device)
        z = vae.sample_latent(test_ex, test_l)
        embed = torch.cat((embed, z), 0)
        labels = torch.cat((labels, test_l), 0)
    x_point = embed[:, :1].cpu().detach().numpy()
    y_point = embed[:, 1:].cpu().detach().numpy()
    colorList = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    labels = labels.cpu().detach().numpy()
    for i in range(10):
        indices = np.argwhere(labels == i).ravel()
        plt.scatter(x_point[indices], y_point[indices], color=colorList[i], marker="o", s=5, label=str(i))
    plt.legend()
    plt.title("Embed the MNIST test set(Only MSE)")
    plt.show()


def test_3_2(n_epoch, train_dataloader, test_dataloader):
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for epoch in range(n_epoch):
        for step, [x, label] in enumerate(tqdm(train_dataloader, desc='Batch')):
            # forward
            x, label = x.to(device), label.to(device)
            x_recon, mean, log_var = vae(x, label)
            MSE = mse(x_recon, x)
            KL = 0.5 * torch.sum(log_var.exp() + mean.pow(2) - 1. - log_var)
            loss = KL
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    embed = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    for idx, [test_ex, test_l] in enumerate(tqdm(test_dataloader, desc='Test')):
        test_ex = test_ex.to(device)
        test_l = test_l.to(device)
        z = vae.sample_latent(test_ex, test_l)
        embed = torch.cat((embed, z), 0)
        labels = torch.cat((labels, test_l), 0)
    x_point = embed[:, :1].cpu().detach().numpy()
    y_point = embed[:, 1:].cpu().detach().numpy()
    colorList = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    labels = labels.cpu().detach().numpy()
    for i in range(10):
        indices = np.argwhere(labels == i).ravel()
        plt.scatter(x_point[indices], y_point[indices], color=colorList[i], marker="o", s=5, label=str(i))
    plt.legend()
    plt.title("Embed the MNIST test set(Only KL)")
    plt.show()


def test_3_3(n_epoch, train_dataloader, test_dataloader):
    vae = VAE().to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    for epoch in range(n_epoch):
        for step, [x, label] in enumerate(tqdm(train_dataloader, desc='Batch')):
            # forward
            x, label = x.to(device), label.to(device)
            x_recon, mean, log_var = vae(x, label)
            MSE = mse(x_recon, x)
            KL = 0.5 * torch.sum(log_var.exp() + mean.pow(2) - 1. - log_var) * 0.0001
            loss = MSE + KL
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    embed = torch.tensor([]).to(device)
    labels = torch.tensor([]).to(device)
    for idx, [test_ex, test_l] in enumerate(tqdm(test_dataloader, desc='Test')):
        test_ex = test_ex.to(device)
        test_l = test_l.to(device)
        z = vae.sample_latent(test_ex, test_l)
        embed = torch.cat((embed, z), 0)
        labels = torch.cat((labels, test_l), 0)
    x_point = embed[:, :1].cpu().detach().numpy()
    y_point = embed[:, 1:].cpu().detach().numpy()
    colorList = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    labels = labels.cpu().detach().numpy()
    for i in range(10):
        indices = np.argwhere(labels == i).ravel()
        plt.scatter(x_point[indices], y_point[indices], color=colorList[i], marker="o", s=5, label=str(i))
    plt.legend()
    plt.title("Embed the MNIST test set(Fully loss)")
    plt.show()


if __name__ == '__main__':
    train_dataloader, test_dataloader = data_loader()
    # task-2
    model_training(5, train_dataloader)
    # task-3
    test_3_1(10, train_dataloader, test_dataloader)
    test_3_2(10, train_dataloader, test_dataloader)
    test_3_3(10, train_dataloader, test_dataloader)
