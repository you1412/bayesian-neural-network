import logging
import math
import os

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms
from tqdm import tqdm, trange

from bnn.metrics import OOD_test, OOD_test_MCBB, test, test_MCBB
from bnn.models import BayesianNetwork, BayesianResNet14, myResNet14

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def reset_net(net, pretrained_net):
    net.conv.w_mu.data.copy_(pretrained_net.conv.w_mu.data)
    net.block1.conv1.w_mu.data.copy_(pretrained_net.block1.conv1.w_mu.data)
    net.block1.conv2.w_mu.data.copy_(pretrained_net.block1.conv2.w_mu.data)
    net.block2.conv1.w_mu.data.copy_(pretrained_net.block2.conv1.w_mu.data)
    net.block2.conv2.w_mu.data.copy_(pretrained_net.block2.conv2.w_mu.data)
    net.block3.conv1.w_mu.data.copy_(pretrained_net.block3.conv1.w_mu.data)
    net.block3.conv2.w_mu.data.copy_(pretrained_net.block3.conv2.w_mu.data)
    net.block4.conv1.w_mu.data.copy_(pretrained_net.block4.conv1.w_mu.data)
    net.block4.conv2.w_mu.data.copy_(pretrained_net.block4.conv2.w_mu.data)
    net.block5.conv1.w_mu.data.copy_(pretrained_net.block5.conv1.w_mu.data)
    net.block5.conv2.w_mu.data.copy_(pretrained_net.block5.conv2.w_mu.data)
    net.block6.conv1.w_mu.data.copy_(pretrained_net.block6.conv1.w_mu.data)
    net.block6.conv2.w_mu.data.copy_(pretrained_net.block6.conv2.w_mu.data)
    net.fc.w_mu.data.copy_(pretrained_net.fc.w_mu.data)
    net.fc.b_mu.data.copy_(pretrained_net.fc.b_mu.data)
    return net


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@click.command()
@click.option("--output-dir", required=True, help="Output directory")
@click.option("--pretrained-dir", required=True, help="Directory for pretrained network")
@click.option("--data-set", default="MNIST", help="Dataset used, MNIST or CIFAR-10 or CIFAR-100")
@click.option("--model", default="CBB", help="Models: CBB (default) or MCBB")
def main(**args):
    data_set = args["data_set"]
    output_dir = args["output_dir"]
    model = args["model"]
    pretrained_dir = args["pretrained_dir"]

    if data_set == "MNIST":
        # downloading data
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


        # out-of-distribution dataset: using FashionMNIST instead of notMNIST for simplicity.
        fmnist = datasets.FashionMNIST(
            root=".data", train=True, download=True, transform=transforms.ToTensor()
        )
        ood_loader = torch.utils.data.DataLoader(
            fmnist, batch_size=128, shuffle=True, drop_last=True
        )

        if model == "CBB":
            # hyper-parameters
            n_units = 400
            epochs = 50
            batch_size = 128
            T_list = torch.pow(10, -1 * torch.tensor(range(0, 45, 5)) / 10).to(DEVICE)
            sigma_list = torch.tensor([0.2, 0.4, 0.6, 0.8, 1]).to(DEVICE)
            n_samples = 1

            testECE = torch.zeros([9, 5]).to(DEVICE)
            testMCE = torch.zeros([9, 5]).to(DEVICE)
            test_accuracy = torch.zeros([9, 5]).to(DEVICE)
            test_loss = torch.zeros([9, 5]).to(DEVICE)
            test_ROCAUC = torch.zeros([9, 5]).to(DEVICE)
            entropy_ave = torch.zeros([9, 5]).to(DEVICE)
            entropy_std = torch.zeros([9, 5]).to(DEVICE)
            L2D = torch.zeros([9, 5]).to(DEVICE)
            ood_ROCAUC1 = torch.zeros([9, 5]).to(DEVICE)
            ood_ROCAUC2 = torch.zeros([9, 5]).to(DEVICE)

            for j, sigma in enumerate(sigma_list):
                for i, T in enumerate(T_list):
                    net = BayesianNetwork(n_units, sigma, T).to(DEVICE)
                    optimizer = optim.Adam(net.parameters())
                    for epoch in range(epochs):
                        _, _ = train(
                            net,
                            optimizer,
                            epoch,
                            training_loader,
                            batch_size,
                            n_samples,
                            T,
                        )
                    (
                        test_accuracy[i, j],
                        test_loss[i, j],
                        testECE[i, j],
                        testMCE[i, j],
                        test_ROCAUC[i, j],
                        out,
                    ) = test(net, test_loader, batch_size, 50, T)
                    (
                        entropy_ave[i, j],
                        entropy_std[i, j],
                        L2D[i, j],
                        ood_ROCAUC1[i, j],
                        ood_ROCAUC2[i, j],
                    ) = OOD_test(net, ood_loader, out, batch_size, 50, T)

                    with open(os.path.join(output_dir, f"net{i}{j}.pt"), "wb") as f:
                        torch.save(net.state_dict(), f)

        else:
            # hyper-parameter setting
            n_units = 400
            epochs = 50
            batch_size = 128
            T_list = torch.pow(10, -1 * torch.tensor(range(0, 45, 5)) / 10).to(DEVICE)
            sigma = torch.tensor(1).to(DEVICE)
            n_samples = 1

            testECE = torch.zeros(
                [
                    9,
                ]
            ).to(DEVICE)
            testMCE = torch.zeros(
                [
                    9,
                ]
            ).to(DEVICE)
            test_accuracy = torch.zeros(
                [
                    9,
                ]
            ).to(DEVICE)
            test_loss = torch.zeros(
                [
                    9,
                ]
            ).to(DEVICE)
            test_ROCAUC = torch.zeros(
                [
                    9,
                ]
            ).to(DEVICE)
            entropy_ave = torch.zeros(
                [
                    9,
                ]
            ).to(DEVICE)
            entropy_std = torch.zeros(
                [
                    9,
                ]
            ).to(DEVICE)
            L2D = torch.zeros(
                [
                    9,
                ]
            ).to(DEVICE)
            ood_ROCAUC1 = torch.zeros(
                [
                    9,
                ]
            ).to(DEVICE)
            ood_ROCAUC2 = torch.zeros(
                [
                    9,
                ]
            ).to(DEVICE)

            for t, T in enumerate(T_list):
                MoG_net = []
                for i in range(3):
                    net = BayesianNetwork(n_units, sigma, T).to(DEVICE)
                    optimizer = optim.Adam(net.parameters())
                    for epoch in range(epochs):
                        _, _ = train(
                            net,
                            optimizer,
                            epoch,
                            training_loader,
                            batch_size,
                            n_samples,
                            T,
                        )
                    with open(os.path.join(output_dir, f"nets/net{t}{i}.pt"), "wb") as f:
                        torch.save(net.state_dict(), f)
                    MoG_net.append(net)
                (
                    test_accuracy[t],
                    test_loss[t],
                    testECE[t],
                    _,
                    test_ROCAUC[t],
                    out,
                ) = test_MoG(MoG_net, test_loader, batch_size, 17, T)
                (
                    entropy_ave[t],
                    entropy_std[t],
                    L2D[t],
                    ood_ROCAUC1[t],
                    ood_ROCAUC2[t],
                ) = OOD_test_MoG(MoG_net, ood_loader, out, batch_size, 17, T)

    elif data_set == "CIFAR-10":
        # download data
        batch_size = 128
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

        # ood dataset
        svhn_dataset = datasets.SVHN(
            root=".data", split="test", transform=transforms.ToTensor(), download=True
        )
        svhn_loader = torch.utils.data.DataLoader(
            svhn_dataset, batch_size=batch_size, drop_last=True
        )

        # CBB for CIFAR-10
        if models == "CBB":
            pretrained_net = myResNet14(1).to(DEVICE)
            with open(os.path.join(pretrained_dir, "trained10_net0.pkl", "rb")) as f:
                pretrained_net.load_state_dict(torch.load(f))

            # hyper-parameters
            T_list = torch.pow(10, -1 * torch.tensor(range(0, 35, 5)) / 10).to(DEVICE)
            sigma_list = torch.tensor([0.2, 0.4, 0.6, 0.8, 1]).to(DEVICE)
            n_samples = 1
            epochs = 300

            testECE = torch.zeros([7, 5]).to(DEVICE)
            testMCE = torch.zeros([7, 5]).to(DEVICE)
            test_accuracy = torch.zeros([7, 5]).to(DEVICE)
            test_loss = torch.zeros([7, 5]).to(DEVICE)
            test_ROCAUC = torch.zeros([7, 5]).to(DEVICE)
            entropy_ave = torch.zeros([7, 5]).to(DEVICE)
            entropy_std = torch.zeros([7, 5]).to(DEVICE)
            L2D = torch.zeros([7, 5]).to(DEVICE)
            ood_ROCAUC1 = torch.zeros([7, 5]).to(DEVICE)
            ood_ROCAUC2 = torch.zeros([7, 5]).to(DEVICE)

            for j, sigma in enumerate(sigma_list):
                for i, T in enumerate(T_list):
                    net = BayesianResNet14(ResidualBlock, sigma).to(DEVICE)
                    net = reset_net(net, pretrained_net)
                    max_lr = 0.0001
                    curr_lr = 0.0001
                    optimizer = optim.Adam(net.parameters(), lr=curr_lr)

                    for epoch in range(epochs):
                        _, _ = train(
                            net,
                            optimizer,
                            epoch,
                            training_loader,
                            batch_size,
                            n_samples,
                            T,
                        )
                        # cosine step size
                        curr_lr = max_lr / 2 * (1 + math.cos((epoch) / epochs * math.pi))
                        update_lr(optimizer, curr_lr)

                    with open(os.path.join(output_dir, f"nets/net{i}{j}.pkl"), "wb") as f:
                        torch.save(net.state_dict(), f)

                    (
                        test_accuracy[i, j],
                        test_loss[i, j],
                        testECE[i, j],
                        testMCE[i, j],
                        test_ROCAUC[i, j],
                        inDis_output,
                    ) = test(net, test_loader, batch_size, 50, T)
                    (
                        entropy_ave[i, j],
                        entropy_std[i, j],
                        L2D[i, j],
                        ood_ROCAUC1[i, j],
                        ood_ROCAUC2[i, j],
                    ) = OOD_test(net, svhn_loader, inDis_output, batch_size, 50, T)

        # MCBB for CIFAR-10
        else:
            # hyper-parameter setting
            batch_size = 128
            n_samples = 1
            T_list = torch.pow(10, -1 * torch.tensor(range(0, 35, 5)) / 10).to(DEVICE)
            sigma = torch.sqrt(torch.tensor(1))
            epochs = 300

            testECE = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            testMCE = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            test_accuracy = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            test_loss = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            test_ROCAUC = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            entropy_ave = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            entropy_std = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            L2D = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            ood_ROCAUC1 = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            ood_ROCAUC2 = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)

            for t, T in enumerate(T_list):
                MoG_net = []
                for i in range(3):
                    net = BayesianResNet14(ResidualBlock, sigma).to(DEVICE)
                    pretrained_net = myResNet14(1).to(DEVICE)
                    with open(os.path.join(pretrained_dir, f"trained10_net{i}.pkl"), "rb") as f:
                        pretrained_net.load_state_dict(torch.load(f))
                    net = reset_net(net, pretrained_net)
                    max_lr = 0.0001
                    curr_lr = 0.0001
                    optimizer = optim.Adam(net.parameters(), lr=curr_lr)

                    for epoch in range(epochs):
                        _, _ = train(
                            net,
                            optimizer,
                            epoch,
                            training_loader,
                            batch_size,
                            n_samples,
                            T,
                        )
                        # cosine step size
                        curr_lr = max_lr / 2 * (1 + math.cos((epoch) / epochs * math.pi))
                        update_lr(optimizer, curr_lr)

                    with open(os.path.join(output_dir, f"nets/net{t}{i}.pkl"), "wb") as f:
                        torch.save(net.state_dict(), f)

                (
                    test_accuracy[t],
                    test_loss[t],
                    testECE[t],
                    _,
                    test_ROCAUC[t],
                    out,
                ) = test_MoG(MoG_net, test_loader, batch_size, 17, T)
                (
                    entropy_ave[t],
                    entropy_std[t],
                    L2D[t],
                    ood_ROCAUC1[t],
                    ood_ROCAUC2[t],
                ) = OOD_test_MoG(MoG_net, svhn_loader, out, batch_size, 17, T)

    else:
        # download data
        batch_size = 128
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

        # ood dataset
        svhn_dataset = datasets.SVHN(
            root=".data", split="test", transform=transforms.ToTensor(), download=True
        )
        svhn_loader = torch.utils.data.DataLoader(
            svhn_dataset, batch_size=batch_size, drop_last=True
        )

        # CBB for CIFAR-100
        if model == "CBB":
            pretrained_net = myResNet14(1, num_class=100).to(DEVICE)
            with open(
                os.path.join(pretrained_dir, "trained100_net0.pkl"), "rb"
            ) as f:  # determine the name
                pretrained_net.load_state_dict(torch.load(f))

            # hyper-parameters
            T_list = torch.pow(10, -1 * torch.tensor(range(0, 35, 5)) / 10).to(DEVICE)
            sigma_list = torch.tensor([0.2, 0.4, 0.6, 0.8, 1]).to(DEVICE)
            n_samples = 1
            epochs = 300

            testECE = torch.zeros([7, 5]).to(DEVICE)
            testMCE = torch.zeros([7, 5]).to(DEVICE)
            test_accuracy = torch.zeros([7, 5]).to(DEVICE)
            test_loss = torch.zeros([7, 5]).to(DEVICE)
            test_ROCAUC = torch.zeros([7, 5]).to(DEVICE)
            entropy_ave = torch.zeros([7, 5]).to(DEVICE)
            entropy_std = torch.zeros([7, 5]).to(DEVICE)
            L2D = torch.zeros([7, 5]).to(DEVICE)
            ood_ROCAUC1 = torch.zeros([7, 5]).to(DEVICE)
            ood_ROCAUC2 = torch.zeros([7, 5]).to(DEVICE)

            for j, sigma in enumerate(sigma_list):
                for i, T in enumerate(T_list):
                    net = BayesianResNet14(ResidualBlock, sigma, num_class=100).to(DEVICE)
                    net = reset_net(net, pretrained_net)
                    max_lr = 0.0001
                    curr_lr = 0.0001
                    optimizer = optim.Adam(net.parameters(), lr=curr_lr)

                    for epoch in range(epochs):
                        _, _ = train(
                            net,
                            optimizer,
                            epoch,
                            training_loader,
                            batch_size,
                            n_samples,
                            T,
                        )
                        # cosine step size
                        curr_lr = max_lr / 2 * (1 + math.cos((epoch) / epochs * math.pi))
                        update_lr(optimizer, curr_lr)

                    with open(os.path.join(output_dir, f"nets/net{i}{j}.pkl"), "wb") as f:
                        torch.save(net.state_dict(), f)

                    (
                        test_accuracy[i, j],
                        test_loss[i, j],
                        testECE[i, j],
                        testMCE[i, j],
                        test_ROCAUC[i, j],
                        inDis_output,
                    ) = test(net, test_loader, batch_size, 50, T, num_class=100)
                    (
                        entropy_ave[i, j],
                        entropy_std[i, j],
                        L2D[i, j],
                        ood_ROCAUC1[i, j],
                        ood_ROCAUC2[i, j],
                    ) = OOD_test(net, svhn_loader, inDis_output, batch_size, 50, T, num_class=100)

        # MCBB for CIFAR-100
        else:
            # hyper-parameter setting
            batch_size = 128
            n_samples = 1
            T_list = torch.pow(10, -1 * torch.tensor(range(0, 35, 5)) / 10).to(DEVICE)
            sigma = torch.sqrt(torch.tensor(1))
            epochs = 300

            testECE = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            testMCE = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            test_accuracy = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            test_loss = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            test_ROCAUC = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            entropy_ave = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            entropy_std = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            L2D = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            ood_ROCAUC1 = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)
            ood_ROCAUC2 = torch.zeros(
                [
                    7,
                ]
            ).to(DEVICE)

            for t, T in enumerate(T_list):
                MoG_net = []
                for i in range(3):
                    net = BayesianResNet14(ResidualBlock, sigma, num_class=100).to(DEVICE)
                    pretrained_net = myResNet14(1, num_class=100).to(DEVICE)
                    with open(os.path.join(pretrained_dir, f"trained100_net{i}.pkl"), "rb") as f:
                        pretrained_net.load_state_dict(torch.load(f))
                    net = reset_net(net, pretrained_net)
                    max_lr = 0.0001
                    curr_lr = 0.0001
                    optimizer = optim.Adam(net.parameters(), lr=curr_lr)

                    for epoch in range(epochs):
                        _, _ = train(
                            net,
                            optimizer,
                            epoch,
                            training_loader,
                            batch_size,
                            n_samples,
                            T,
                        )
                        # cosine step size
                        curr_lr = max_lr / 2 * (1 + math.cos((epoch) / epochs * math.pi))
                        update_lr(optimizer, curr_lr)

                    with open(os.path.join(output_dir, f"nets/net{t}{i}.pkl"), "wb") as f:
                        torch.save(net.state_dict(), f)

                (
                    test_accuracy[t],
                    test_loss[t],
                    testECE[t],
                    _,
                    test_ROCAUC[t],
                    out,
                ) = test_MoG(MoG_net, test_loader, batch_size, 17, T, num_class=100)
                (
                    entropy_ave[t],
                    entropy_std[t],
                    L2D[t],
                    ood_ROCAUC1[t],
                    ood_ROCAUC2[t],
                ) = OOD_test_MoG(MoG_net, svhn_loader, out, batch_size, 17, T, num_class=100)

    with open(os.path.join(output_dir, "results/test_accuracy.pt"), "wb") as f:
        torch.save(test_accuracy.cpu(), f)

    with open(os.path.join(output_dir, "results/test_loss.pt"), "wb") as f:
        torch.save(test_loss.cpu(), f)

    with open(os.path.join(output_dir, "results/testECE.pt"), "wb") as f:
        torch.save(testECE.cpu(), f)

    with open(os.path.join(output_dir, "results/testMCE.pt"), "wb") as f:
        torch.save(testMCE.cpu(), f)

    with open(os.path.join(output_dir, "results/entropy_ave.pt"), "wb") as f:
        torch.save(entropy_ave.cpu(), f)

    with open(os.path.join(output_dir, "results/entropy_std.pt"), "wb") as f:
        torch.save(entropy_std.cpu(), f)

    with open(os.path.join(output_dir, "results/L2D.pt"), "wb") as f:
        torch.save(L2D.cpu(), f)

    with open(os.path.join(output_dir, "results/test_ROCAUC.pt"), "wb") as f:
        torch.save(test_ROCAUC.cpu(), f)

    with open(os.path.join(output_dir, "results/ood_ROCAUC1.pt"), "wb") as f:
        torch.save(ood_ROCAUC1.cpu(), f)

    with open(os.path.join(output_dir, "results/ood_ROCAUC2.pt"), "wb") as f:
        torch.save(ood_ROCAUC2.cpu(), f)


if __name__ == "__main__":
    main()
