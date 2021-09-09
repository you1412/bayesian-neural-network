import math
import os
import random

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms
from tqdm import tqdm, trange

from bnn.models_BS import BayesianNetwork, BayesianResNet14

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(net, optimizer, epoch, trainLoader, batchSize, nSamples, T):
    net.train()
    num_batches_train = len(trainLoader)

    for batch_idx, (data, target) in enumerate(tqdm(trainLoader)):
        data, target = data.to(DEVICE), target.to(DEVICE)

        net.zero_grad()
        (
            loss,
            log_prior,
            log_variational_posterior,
            negative_log_likelihood,
            corrects,
        ) = net.free_energy(data, target, batchSize, num_batches_train, nSamples, T)
        loss.backward()
        optimizer.step()

        accuracy = corrects / batchSize

    return accuracy, loss


@click.command()
@click.option("--data-set", default="MNIST", help="Dataset used, MNIST or CIFAR-10 or CIFAR-100")
@click.option("--output-dir", required=True, help="Output directory")
def main(**args):
    data_set = args["data_set"]
    output_dir = args["output_dir"]

    if data_set == "MNIST":
        training_data = datasets.MNIST(
            root=".data", train=True, download=True, transform=transforms.ToTensor()
        )
        test_data = datasets.MNIST(
            root=".data", train=False, download=True, transform=transforms.ToTensor()
        )
        training_loader = torch.utils.data.DataLoader(
            training_data, batch_size=128, shuffle=True, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=128, shuffle=True, drop_last=True
        )

        n_units = 400
        epochs = 50
        batch_size = 128
        T = 1
        sigma = torch.tensor(1).to(DEVICE)
        n_samples = 5

        net = BayesianNetwork(n_units, sigma, T).to(DEVICE)
        optimizer = optim.Adam(net.parameters())

        for epoch in range(epochs):
            trainAcc, trainLoss = train(net, optimizer, epoch, training_loader, batch_size, 5, T)
        with open(os.path.join(output_dir, "BS_net.pt"), "wb") as f:
            torch.save(net.state_dict(), f)

    elif data_set == "CIFAR-10":
        training_data = datasets.CIFAR10(
            root=".data", train=True, download=True, transform=transforms.ToTensor()
        )
        test_data = datasets.CIFAR10(
            root=".data", train=False, download=True, transform=transforms.ToTensor()
        )
        training_loader = torch.utils.data.DataLoader(
            training_data, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True, drop_last=True
        )

        batch_size = 128
        n_samples = 5
        T = 1
        sigma = torch.tensor(1).to(DEVICE)
        epochs = 500

        net = BayesianResNet14(ResidualBlock, sigma).to(DEVICE)
        optimizer = optim.Adam(net.parameters())

        for epoch in range(epochs):
            trainAcc, trainLoss = train(
                net, optimizer, epoch, training_loader, batch_size, n_samples, T
            )
        with open(os.path.join(output_dir, "BS_net.pt"), "wb") as f:
            torch.save(net.state_dict(), f)

    else:
        training_data = datasets.CIFAR100(
            root=".data", train=True, download=True, transform=transforms.ToTensor()
        )
        test_data = datasets.CIFAR100(
            root=".data", train=False, download=True, transform=transforms.ToTensor()
        )
        training_loader = torch.utils.data.DataLoader(
            training_data, batch_size=batch_size, shuffle=True, drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True, drop_last=True
        )

        batch_size = 128
        n_samples = 5
        T = 1
        sigma = torch.tensor(1).to(DEVICE)
        epochs = 500
        max_lr = 0.001
        curr_lr = 0.001

        net = BayesianResNet14(ResidualBlock, sigma, num_class=100).to(DEVICE)
        optimizer = optim.Adam(net.parameters(), lr=curr_lr)

        for epoch in range(epochs):
            trainAcc, trainLoss = train(
                net, optimizer, epoch, training_loader, batch_size, n_samples, T
            )
            curr_lr = max_lr / 2 * (1 + math.cos((epoch) / epochs * math.pi))
            update_lr(optimizer, curr_lr)
        with open(os.path.join(output_dir, "BS_net.pt"), "wb") as f:
            torch.save(net.state_dict(), f)


if __name__ == "__main__":
    main()
