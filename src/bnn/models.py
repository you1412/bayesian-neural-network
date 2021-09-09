import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms
from tqdm import tqdm, trange

# for CBB and MCBB

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-20


class Gaussian:
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (
            -math.log(math.sqrt(2 * math.pi))
            - torch.log(self.sigma + eps)
            - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)
        ).sum()


class GaussianPrior:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def log_prob(self, input):
        return (
            -math.log(math.sqrt(2 * math.pi))
            - torch.log(self.sigma)
            - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)
        ).sum()


class BayesianLinear(nn.Module):
    """
    TODO: refact initialization of parameter rho
    """

    def __init__(self, n_input, n_output, sigma1, lower_bound, upper_bounnd):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output

        self.w_mu = nn.Parameter(
            torch.Tensor(n_output, n_input).normal_(0, math.sqrt(2 / n_input))
        )  # todo
        self.w_rho = nn.Parameter(
            torch.Tensor(n_output, n_input).uniform_(lower_bound, upper_bounnd)
        )
        # self.w_rho = nn.Parameter(torch.Tensor(n_output, n_input).uniform_(-2.253,-2.252))
        self.w = Gaussian(self.w_mu, self.w_rho)

        self.b_mu = nn.Parameter(torch.Tensor(n_output).normal_(0, math.sqrt(2 / n_input)))
        # self.b_rho = nn.Parameter(torch.Tensor(n_output).uniform_(-5,-4))
        self.b_rho = nn.Parameter(torch.Tensor(n_output).uniform_(lower_bound, upper_bounnd))
        self.b = Gaussian(self.b_mu, self.b_rho)

        # Prior: Gaussian
        self.w_prior = GaussianPrior(0, sigma1)
        self.b_prior = GaussianPrior(0, sigma1)
        self.log_prior = 0
        self.log_variational_posterior = 0
        self.sigma_mean = 0
        self.sigma_std = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            w = self.w.sample()
            b = self.b.sample()
        else:
            w = self.w_mu
            b = self.b_mu

        self.log_prior = self.w_prior.log_prob(w) + self.b_prior.log_prob(b)
        self.log_variational_posterior = self.w.log_prob(w) + self.b.log_prob(b)

        self.sigma_mean = self.w.sigma.mean()
        self.sigma_std = self.w.sigma.std()

        return F.linear(input, w, b)


class BayesianConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, sigma1, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.w_mu = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).normal_(
                0,
                math.sqrt(2 / (out_channels * in_channels * kernel_size * kernel_size)),
            )
        )
        # self.w_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).uniform_(-5,-4))
        self.w_rho = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size).uniform_(
                -2.253, -2.252
            )
        )
        self.w = Gaussian(self.w_mu, self.w_rho)

        # prior: Gaussian
        self.w_prior = GaussianPrior(0, sigma1)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            w = self.w.sample()
        else:
            w = self.w_mu

        self.log_prior = self.w_prior.log_prob(w)
        self.log_variational_porsterior = self.w.log_prob(w)
        return F.conv2d(input, w, bias=None, stride=self.stride, padding=self.padding)


def BayesianConv3x3(in_channels, out_channels, sigma1, stride=1):
    return BayesianConv2D(
        in_channels, out_channels, sigma1, kernel_size=3, stride=stride, padding=1
    )


class TLU(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.tau)
        # nn.init.zeros_(self.tau)

    def forward(self, x):
        return torch.max(x, self.tau)


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_eps_learnable=False):
        super().__init__()
        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_learnable = is_eps_learnable

        self.weight = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True
        )
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_learnable:
            self.eps = nn.Parameter(torch.Tensor(1))
        else:
            self.eps = torch.tensor(eps)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        nn.init.kaiming_normal_(self.bias)
        # nn.init.ones_(self.weight)
        # nn.init.zeros_(self.bias)
        if self.is_eps_learnable:
            nn.init.constant_(self.eps, self.init_eps)

    def forward(self, x):
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        x = x * torch.rsqrt(nu2 + self.eps.abs())
        x = self.weight * x + self.bias
        return x


class ResidualBlock(nn.Module):
    """
    distinguish between batch normalisation and filter response normalisation
    """

    def __init__(self, in_channels, out_channels, sigma1, FRN, stride=1, downsample=None):
        super().__init__()
        self.conv1 = BayesianConv3x3(in_channels, out_channels, sigma1, stride)
        self.conv2 = BayesianConv3x3(out_channels, out_channels, sigma1)
        if FRN:
            self.frn1 = FRN(out_channels)
            self.tlu1 = TLU(out_channels)
            self.frn2 = FRN(out_channels)
            self.tlu2 = TLU(out_channels)
        else:
            self.frn1 = nn.BatchNorm2d(out_channels)
            self.tlu1 = nn.ReLU(inplace=True)
            self.frn2 = nn.BatchNorm2d(out_channels)
            self.tlu2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.log_prior = 0
        self.log_variational_posterior = 0
        self.sigma_mean = 0
        self.sigma_std = 0

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.frn1(out)
        out = self.tlu1(out)
        out = self.conv2(out)
        out = self.frn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.tlu2(out)
        self.log_prior = self.conv1.log_prior + self.conv2.log_prior
        self.log_variational_posterior = (
            self.conv1.log_variational_posterior + self.conv2.log_variational_posterior
        )
        para = torch.cat((self.conv1.w.sigma.flatten(), self.conv2.w.sigma.flatten()))
        self.sigma_mean = para.mean()
        self.sigma_std = para.std()
        return out


class BayesianResNet14(nn.Module):
    """
    distinguish between batch normalisation and filter response normalisation
    """

    def __init__(self, block, sigma1, FRN=False, num_class=10):
        super().__init__()
        self.in_channels = 16
        self.conv = BayesianConv3x3(3, 16, sigma1)
        if FRN:
            self.frn = FRN(out_channels)
            self.tlu = TLU(out_channels)
            downsample1 = nn.Sequential(BayesianConv3x3(16, 32, sigma1, FRN, 2), FRN(32))
            downsample2 = nn.Sequential(BayesianConv3x3(32, 64, sigma1, FRN, 2), FRN(64))

        else:
            self.frn = nn.BatchNorm2d(16)
            self.tlu = nn.ReLU(inplace=True)
            downsample1 = nn.Sequential(BayesianConv3x3(16, 32, sigma1, FRN, 2), nn.BatchNorm2d(32))
            downsample2 = nn.Sequential(BayesianConv3x3(32, 64, sigma1, FRN, 2), nn.BatchNorm2d(64))

        self.block1 = ResidualBlock(16, 16, sigma1, FRN)
        self.block2 = ResidualBlock(16, 16, sigma1, FRN)

        self.block3 = ResidualBlock(16, 32, sigma1, FRN, 2, downsample1)
        self.block4 = ResidualBlock(32, 32, sigma1)

        self.block5 = ResidualBlock(32, 64, sigma1, FRN, 2, downsample2)
        self.block6 = ResidualBlock(64, 64, sigma1, FRN)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = BayesianLinear(64, num_class, sigma1, -2.253, -2.252)

    def forward(self, x, sample=False):
        out = self.conv(x)
        out = self.frn(out)
        out = self.tlu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out, sample))
        return out

    def log_prior(self):
        return (
            self.conv.log_prior
            + self.block1.log_prior
            + self.block2.log_prior
            + self.block3.log_prior
            + self.block4.log_prior
            + self.block5.log_prior
            + self.block6.log_prior
            + self.fc.log_prior
        )

    def log_variational_posterior(self):
        return (
            self.conv.log_variational_posterior
            + self.block1.log_variational_posterior
            + self.block2.log_variational_posterior
            + self.block3.log_variational_posterior
            + self.block4.log_variational_posterior
            + self.block5.log_variational_posterior
            + self.block6.log_variational_posterior
            + self.fc.log_variational_posterior
        )

    def free_energy(self, input, target, batch_size, num_batches, n_samples, T):
        outputs = torch.zeros(batch_size, 10).to(DEVICE)
        log_prior = torch.zeros(1).to(DEVICE)
        log_variational_posterior = torch.zeros(1).to(DEVICE)
        negative_log_likelihood = torch.zeros(1).to(DEVICE)
        for i in range(n_samples):
            output = self(input, sample=True)
            outputs += output / n_samples
            log_prior += self.log_prior() / n_samples
            log_variational_posterior += self.log_variational_posterior() / n_samples
            negative_log_likelihood += (
                F.nll_loss(torch.log(output + eps), target, size_average=False) / n_samples
            )

        # new target function, not absorb T into prior
        loss = (
            log_variational_posterior - log_prior / T
        ) + negative_log_likelihood / T * num_batches

        corrects = outputs.argmax(dim=1).eq(target).sum().item()

        return (
            loss,
            log_prior,
            log_variational_posterior,
            negative_log_likelihood,
            corrects,
        )


class BayesianNetwork(nn.Module):
    """
    plain network
    """

    def __init__(self, n_units, sigma1, T):
        super().__init__()
        self.l1 = BayesianLinear(28 * 28, n_units, sigma1, -5, -4)
        self.l2 = BayesianLinear(n_units, n_units, sigma1, -5, -4)
        self.l3 = BayesianLinear(n_units, 10, sigma1, -5, -4)

    def forward(self, x, sample=False):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.l1(x, sample), inplace=False)
        x = F.relu(self.l2(x, sample), inplace=False)
        x = F.softmax(self.l3(x, sample))
        return x

    def log_prior(self):
        return self.l1.log_prior + self.l2.log_prior + self.l3.log_prior

    def log_variational_posterior(self):
        return (
            self.l1.log_variational_posterior
            + self.l2.log_variational_posterior
            + self.l3.log_variational_posterior
        )

    def free_energy(self, input, target, batch_size, num_batches, n_samples, T):
        outputs = torch.zeros(batch_size, 10).to(DEVICE)
        log_prior = torch.zeros(1).to(DEVICE)
        log_variational_posterior = torch.zeros(1).to(DEVICE)
        negative_log_likelihood = torch.zeros(1).to(DEVICE)
        for i in range(n_samples):
            output = self(input, sample=True)
            outputs += output / n_samples
            log_prior += self.log_prior() / n_samples
            log_variational_posterior += self.log_variational_posterior() / n_samples
            negative_log_likelihood += (
                F.nll_loss(torch.log(output + eps), target, size_average=False) / n_samples
            )

        # new target function, not absorb T into prior
        loss = (
            log_variational_posterior - log_prior / T
        ) + negative_log_likelihood / T * num_batches

        corrects = outputs.argmax(dim=1).eq(target).sum().item()

        return (
            loss,
            log_prior,
            log_variational_posterior,
            negative_log_likelihood,
            corrects,
        )


# nonBayesian Network
class myLinear(nn.Module):
    def __init__(self, n_input, n_output, sigma1):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output

        self.w_mu = nn.Parameter(torch.Tensor(n_output, n_input).normal_(0, math.sqrt(2 / n_input)))

        self.b_mu = nn.Parameter(torch.Tensor(n_output).normal_(0, math.sqrt(2 / n_input)))

    def forward(self, input, sample=False):

        w = self.w_mu
        b = self.b_mu

        return F.linear(input, w, b)


class myConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, sigma1, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.w_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.reset_para()

    def reset_para(self):
        nn.init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))

    def forward(self, input, sample=False):

        w = self.w_mu

        return F.conv2d(input, w, bias=None, stride=self.stride, padding=self.padding)


def myConv3x3(in_channels, out_channels, sigma1, stride=1):
    return myConv2D(in_channels, out_channels, sigma1, kernel_size=3, stride=stride, padding=1)


class myResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sigma1, stride=1, downsample=None):
        super().__init__()
        self.conv1 = myConv3x3(in_channels, out_channels, sigma1, stride)
        self.frn1 = nn.BatchNorm2d(out_channels)
        self.tlu1 = nn.ReLU(inplace=True)
        self.conv2 = myConv3x3(out_channels, out_channels, sigma1)
        self.frn2 = nn.BatchNorm2d(out_channels)
        self.tlu2 = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.frn1(out)
        out = self.tlu1(out)
        out = self.conv2(out)
        out = self.frn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.tlu2(out)
        return out


class myResNet14(nn.Module):
    def __init__(self, sigma1, num_class=10):
        super().__init__()
        self.in_channels = 16
        self.conv = myConv3x3(3, 16, sigma1)
        self.frn = nn.BatchNorm2d(16)
        self.tlu = nn.ReLU(inplace=True)

        self.block1 = myResidualBlock(16, 16, sigma1)
        self.block2 = myResidualBlock(16, 16, sigma1)

        downsample1 = nn.Sequential(myConv3x3(16, 32, sigma1, 2), nn.BatchNorm2d(32))
        self.block3 = myResidualBlock(16, 32, sigma1, 2, downsample1)
        self.block4 = myResidualBlock(32, 32, sigma1)

        downsample2 = nn.Sequential(myConv3x3(32, 64, sigma1, 2), nn.BatchNorm2d(64))
        self.block5 = myResidualBlock(32, 64, sigma1, 2, downsample2)
        self.block6 = myResidualBlock(64, 64, sigma1)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc = myLinear(64, num_class, sigma1)

    def forward(self, x, sample=False):
        out = self.conv(x)
        out = self.frn(out)
        out = self.tlu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out, sample))
        return out

    def free_energy(self, input, target, batch_size, num_batches, n_samples, T):
        negative_log_likelihood = torch.zeros(1).to(DEVICE)
        for i in range(n_samples):
            output = self(input, sample=True)
            negative_log_likelihood += (
                F.nll_loss(torch.log(output + eps), target, size_average=False) / n_samples
            )

        # new target function, not absorb T into prior
        loss = negative_log_likelihood / T * num_batches

        corrects = output.argmax(dim=1).eq(target).sum().item()

        return loss, corrects, 0, 0, 0
