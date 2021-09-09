import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms
from tqdm import tqdm, trange


def test(net, testLoader, batchSize, nSamples, T, num_class=10):
    """
    test for CBB and BS
    """
    net.eval()
    accuracy = 0
    n_corrects = 0
    Loss = 0
    num_batches_test = len(testLoader)
    n_test = batchSize * num_batches_test
    outputs = torch.zeros(n_test, num_class).to(DEVICE)
    correct = torch.zeros(n_test).to(DEVICE)
    target_all = torch.zeros(n_test).to(DEVICE)

    M = 10
    boundary = ((torch.tensor(range(0, M)) + 1) / 10).view(1, -1)
    boundary = boundary.repeat(batchSize, 1).to(DEVICE)

    acc_Bm_sum = torch.zeros(M).to(DEVICE)
    conf_Bm_sum = torch.zeros(M).to(DEVICE)
    Bm = torch.zeros(M).to(DEVICE)

    with torch.no_grad():
        for i, (data, target) in enumerate(testLoader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            target_all[i * batchSize : batchSize * (i + 1)] = target
            for j in range(nSamples):
                output = net(data, sample=True)
                outputs[i * batchSize : batchSize * (i + 1), :] += output / nSamples
                Loss += F.nll_loss(torch.log(output), target, size_average=False) / nSamples
                # loss is log likelihood

            correct[i * batchSize : batchSize * (i + 1)] = (
                (outputs[i * batchSize : batchSize * (i + 1), :]).argmax(1).eq(target)
            )

            otemp = outputs[i * batchSize : batchSize * (i + 1), :]
            p_i, _ = otemp.max(dim=1, keepdims=True)
            B = (p_i.le(boundary) * 1).argmax(dim=1)

            acc_i = otemp.argmax(1).eq(target)
            for m in range(M):
                is_m = B.eq(m)
                Bm[m] += is_m.sum()
                acc_Bm_sum[m] += torch.sum(acc_i * is_m)
                conf_Bm_sum[m] += torch.sum(p_i.flatten() * is_m)

        accuracy = correct.mean()

    ROCAUC = roc_auc_score(target_all.cpu(), outputs.cpu(), multi_class="ovr")

    ECE = (acc_Bm_sum - conf_Bm_sum).abs().sum() / (n_test)

    temp = (acc_Bm_sum - conf_Bm_sum) / Bm
    temp[temp != temp] = 0
    MCE, _ = temp.abs().max(0)

    return accuracy, Loss, ECE, MCE, ROCAUC, output


def cal_entropy(p):
    logP = p.clone()
    logP[p == 0] = 1
    logP = torch.log(logP)
    return (-logP * p).sum(dim=1)


def OOD_test(net, oodLoader, inDis_output, batchSize, nSamples, T, num_class=10):
    net.eval()
    num_batches_test = len(oodLoader)
    n_test = batchSize * num_batches_test
    n_inDis = len(inDis_output)

    outputs = torch.zeros(n_test, num_class).to(DEVICE)

    target_all = torch.zeros(n_test + n_inDis)
    target_all[n_test:] = 1

    score1 = torch.zeros(n_test + n_inDis)
    score2 = torch.zeros(n_test + n_inDis)

    with torch.no_grad():
        for i, (data, target) in enumerate(oodLoader):
            data = data.to(DEVICE)

            for j in range(nSamples):
                output = net(data, sample=True)
                outputs[i * batchSize : batchSize * (i + 1), :] += output / nSamples
    entropy = cal_entropy(outputs)
    entropy_ave = entropy.mean()
    entropy_std = entropy.std()

    score1[:n_test], _ = outputs.max(dim=1)
    score1[n_test:], _ = inDis_output.max(dim=1)

    score2[:n_test] = entropy_ave
    score2[n_test:] = cal_entropy(inDis_output).mean()

    L2D = (torch.square(outputs - 0.1).sum(dim=1)).mean()
    ROCAUC1 = roc_auc_score(target_all, score1, multi_class="ovr", average="weighted")
    ROCAUC2 = roc_auc_score(target_all, score2, multi_class="ovr", average="weighted")
    return entropy_ave, entropy_std, L2D, ROCAUC1, ROCAUC2


def test_MCBB(net_list, testLoader, batchSize, nSamples, T, num_class=10):
    """
    test for MCBB
    """
    for net in net_list:
        net.eval()
    accuracy = 0
    n_corrects = 0
    Loss = 0
    num_batches_test = len(testLoader)
    n_test = batchSize * num_batches_test
    outputs = torch.zeros(n_test, num_class).to(DEVICE)
    correct = torch.zeros(n_test).to(DEVICE)
    target_all = torch.zeros(n_test).to(DEVICE)
    n_list = len(net_list)

    M = 10
    boundary = ((torch.tensor(range(0, M)) + 1) / 10).view(1, -1)
    boundary = boundary.repeat(batchSize, 1).to(DEVICE)

    acc_Bm_sum = torch.zeros(M).to(DEVICE)
    conf_Bm_sum = torch.zeros(M).to(DEVICE)
    Bm = torch.zeros(M).to(DEVICE)

    with torch.no_grad():
        for i, (data, target) in enumerate(testLoader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            target_all[i * batchSize : batchSize * (i + 1)] = target
            for k, net in enumerate(net_list):
                for j in range(nSamples):
                    output = net(data, sample=True)
                    outputs[i * batchSize : batchSize * (i + 1), :] += output / (nSamples * n_list)
                    Loss += F.nll_loss(torch.log(output), target, size_average=False) / (
                        nSamples * n_list
                    )
                    # loss is log likelihood

            correct[i * batchSize : batchSize * (i + 1)] = (
                (outputs[i * batchSize : batchSize * (i + 1), :]).argmax(1).eq(target)
            )

            otemp = outputs[i * batchSize : batchSize * (i + 1), :]
            p_i, _ = otemp.max(dim=1, keepdims=True)
            B = (p_i.le(boundary) * 1).argmax(dim=1)

            acc_i = otemp.argmax(1).eq(target)
            for m in range(M):
                is_m = B.eq(m)
                Bm[m] += is_m.sum()
                acc_Bm_sum[m] += torch.sum(acc_i * is_m)
                conf_Bm_sum[m] += torch.sum(p_i.flatten() * is_m)

        accuracy = correct.mean()

    ROCAUC = roc_auc_score(target_all.cpu(), outputs.cpu(), multi_class="ovr")

    ECE = (acc_Bm_sum - conf_Bm_sum).abs().sum() / (n_test)

    temp = (acc_Bm_sum - conf_Bm_sum) / Bm
    temp[temp != temp] = 0
    MCE, _ = temp.abs().max(0)

    return accuracy, Loss, ECE, MCE, ROCAUC, output


def OOD_test_MCBB(net_list, oodLoader, inDis_output, batchSize, nSamples, T, num_class=10):
    for net in net_list:
        net.eval()
    num_batches_test = len(oodLoader)
    n_test = batchSize * num_batches_test
    n_inDis = len(inDis_output)
    n_list = len(net_list)

    outputs = torch.zeros(n_test, num_class).to(DEVICE)

    target_all = torch.zeros(n_test + n_inDis)
    target_all[n_test:] = 1

    score1 = torch.zeros(n_test + n_inDis)
    score2 = torch.zeros(n_test + n_inDis)

    with torch.no_grad():
        for i, (data, target) in enumerate(oodLoader):
            data = data.to(DEVICE)

            for k, net in enumerate(net_list):
                for j in range(nSamples):
                    output = net(data, sample=True)
                    outputs[i * batchSize : batchSize * (i + 1), :] += output / (nSamples * n_list)
    entropy = cal_entropy(outputs)
    entropy_ave = entropy.mean()
    entropy_std = entropy.std()

    score1[:n_test], _ = outputs.max(dim=1)
    score1[n_test:], _ = inDis_output.max(dim=1)

    score2[:n_test] = entropy_ave
    score2[n_test:] = cal_entropy(inDis_output).mean()

    L2D = (torch.square(outputs - 0.1).sum(dim=1)).mean()
    ROCAUC1 = roc_auc_score(target_all, score1, multi_class="ovr", average="weighted")
    ROCAUC2 = roc_auc_score(target_all, score2, multi_class="ovr", average="weighted")
    return entropy_ave, entropy_std, L2D, ROCAUC1, ROCAUC2
