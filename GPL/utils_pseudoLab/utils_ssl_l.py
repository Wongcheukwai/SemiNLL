from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import scipy.stats as stats
import math
import numpy as np
from matplotlib import pyplot as plt
from utils.AverageMeter import AverageMeter
from utils.criterion import *
import ramps
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing as preprocessing
import sys
from math import pi
from math import cos

##############################################################################
############################# TRAINING LOSSSES ###############################
##############################################################################

def loss_soft_reg_ep(preds, labels, soft_labels, device, args):
    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)
    p = torch.ones(args.num_classes).to(device) / args.num_classes

    L_c = -torch.mean(torch.sum(soft_labels * F.log_softmax(preds, dim=1), dim=1))   # Soft labels
    L_p = -torch.sum(torch.log(prob_avg) * p)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))

    loss = L_c + args.reg1 * L_p + args.reg2 * L_e
    return prob, loss

##############################################################################
def mixup_data(x, y, alpha=1.0, device='cuda', addtion=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        if addtion:
            lam = max(lam, 1 - lam)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def loss_mixup_reg_ep(preds, labels, targets_a, targets_b, device, lam, args):
    prob = F.softmax(preds, dim=1) # [100,10]
    prob_avg = torch.mean(prob, dim=0)
    p = torch.ones(args.num_classes).to(device) / args.num_classes

    mixup_loss_a = -torch.mean(torch.sum(targets_a * F.log_softmax(preds, dim=1), dim=1))
    mixup_loss_b = -torch.mean(torch.sum(targets_b * F.log_softmax(preds, dim=1), dim=1))
    mixup_loss = lam * mixup_loss_a + (1 - lam) * mixup_loss_b

    L_p = -torch.sum(torch.log(prob_avg) * p)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))

    loss = mixup_loss + args.reg1 * L_p + args.reg2 * L_e
    return prob, loss


##############################################################################

def train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, unlabeled_indexes, corrupted_labels, clean_labels):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    loss_per_batch = []
    acc_train_per_batch = []

    end = time.time()

    results = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32) # 用来更新unlaleled data的

    if args.loss_term == "Reg_ep":
        alpha = None
    elif args.loss_term == "MixUp_ep":
        #print("Training with Mixup and regularization for soft labels and for predicting different classes (MixUp_ep)")
        alpha = args.Mixup_Alpha
        #print("Mixup alpha value:{}".format(alpha))

    count_true_1 = 0
    counter = 1
    for i, (imgs, img_pslab, labels, soft_labels, index, unlabeled_labels, unlabeled_soft_labels, img1, img2) in enumerate(train_loader):

        images, labels, soft_labels, unlabeled_labels, unlabeled_soft_labels = imgs.to(device), \
                                                                               labels.to(device), soft_labels.to(device),\
                                                                               unlabeled_labels.to(device), unlabeled_soft_labels.to(device)

        if len(unlabeled_indexes) != 0:  # 正式训练的时候，0的时候是warmup
            model.eval()

            results_2 = torch.zeros(len(index), args.num_classes).float().to(device)

            if args.argument == 'yes':
                with torch.no_grad():
                    #for _, _, _, _, index, _, _, img1, img2 in train_loader:
                    img1, img2 = img1.to(device), img2.to(device)
                    output_img1, output_img2 = model(img1), model(img2)
                    # compute output
                    pu = (F.softmax(output_img1, dim=1) + F.softmax(output_img2, dim=1)) / 2  # 64,10
                    ptu = pu ** (1 / args.T)  # temparature sharpening 64,10
                    targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                    targets_u = targets_u.detach()  # 64,10
                    #save
                    #results_2[index.detach().numpy().tolist()] = targets_u.cpu().detach().numpy().tolist()
                    results_2 = targets_u.data.clone().detach()
            elif args.argument == 'no':
                with torch.no_grad():
                    # for _, _, _, _, index, _, _, img1, img2 in train_loader:
                    img1 = img1.to(device)
                    output_img1 = model(img1)
                    results_2 = output_img1.data.clone().detach()

            if args.dataset == 'cifar100' :
                _, predicted = results_2.topk(5, dim=1, largest=True, sorted=False)
                predicted = predicted.cpu().numpy().tolist()

                count_labeled = 0
                for h in range(len(index)):
                    if not corrupted_labels[index[h]] in predicted[h]:
                        labels[h] = unlabeled_labels[h]
                        soft_labels[h] = unlabeled_soft_labels[h]
                    else:
                        count_labeled += 1
                        if clean_labels[index[h]] in predicted[h]:
                            count_true_1 += 1

            elif args.dataset == 'cifar10':
                _, predicted = torch.max(results_2, 1)
                predicted = predicted.cpu().numpy().tolist()

                count_labeled = 0
                for h in range(len(index)):
                    if predicted[h] != corrupted_labels[index[h]]:#如果出现不匹配，就把他判定为unlabeled，即用unlabeled的pesudo label
                        labels[h] = unlabeled_labels[h]
                        soft_labels[h] = unlabeled_soft_labels[h]
                    else:
                        count_labeled += 1
                        if predicted[h] == clean_labels[index[h]]:
                            count_true_1 += 1

        model.train()

        if args.DApseudolab == "False":
            images_pslab = img_pslab.to(device)

        if args.loss_term == "MixUp_ep":
            if args.dropout > 0.0 and args.drop_extra_forward == "True": # 对于image_pslab停止dropout
                if args.network == "PreactResNet18_WNdrop":
                    tempdrop = model.drop
                    model.drop = 0.0

                elif args.network == "WRN28_2_wn" or args.network == "resnet18_wndrop":
                    for m in model.modules():
                        if isinstance(m, nn.Dropout):
                            tempdrop = m.p
                            m.p = 0.0
                else:
                    tempdrop = model.drop.p
                    model.drop.p = 0.0

            if args.DApseudolab == "False":
                optimizer.zero_grad()
                output_x1 = model(images_pslab) # 这里的images_pslab是validation_transfrom的
                output_x1.detach_()
                optimizer.zero_grad()
            else:
                optimizer.zero_grad()
                output_x1 = model(images) # 这是image label是train_transform的
                output_x1.detach_()
                optimizer.zero_grad()

            if args.dropout > 0.0 and args.drop_extra_forward == "True":#这里开始恢复
                if args.network == "PreactResNet18_WNdrop":
                    model.drop = tempdrop

                elif args.network == "WRN28_2_wn" or args.network == "resnet18_wndrop":
                    for m in model.modules():
                        if isinstance(m, nn.Dropout):
                            m.p = tempdrop
                else:
                    model.drop.p = tempdrop

            images, targets_a, targets_b, lam = mixup_data(images, soft_labels, alpha, device, args.addtion)# target b是打乱了顺序后的label， images是混淆了xa和xb的

        # compute output
        outputs = model(images) #这是mixup

        if args.loss_term == "Reg_ep":
            prob, loss = loss_soft_reg_ep(outputs, labels, soft_labels, device, args)

        elif args.loss_term == "MixUp_ep":
            prob = F.softmax(output_x1, dim=1) # [100,10]，这里的ouput_x1是没有被mixup的output
            prob_mixup, loss = loss_mixup_reg_ep(outputs, labels, targets_a, targets_b, device, lam, args) # label是没变过

            outputs = output_x1

        results[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        prec1, prec5 = accuracy_v2(outputs, labels, top=[1, 1]) # 这里变成了output_x1
        train_loss.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if len(unlabeled_indexes) != 0:  # 正式训练的时候，0的时候是warmup
            if counter % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}, Accuracy: {:.0f}%, Learning rate: {:.4f}, Labeled: {}'.format(
                    epoch, counter * len(images), len(train_loader.dataset),
                           100. * counter / len(train_loader), loss.item(),
                           prec1, optimizer.param_groups[0]['lr'], count_labeled))
        else:
            if counter % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}, Accuracy: {:.0f}%, Learning rate: {:.4f}'.format(
                    epoch, counter * len(images), len(train_loader.dataset),
                           100. * counter / len(train_loader), loss.item(),
                           prec1, optimizer.param_groups[0]['lr']))
        counter = counter + 1

    print('there are %d clean data in this epoch' % (count_true_1))
    #这个epoch结束了
    if args.swa == 'True':
        if epoch > args.swa_start and epoch%args.swa_freq == 0 :
            optimizer.update_swa()

    # update soft labels
    train_loader.dataset.update_labels(results, unlabeled_indexes) # 这里才是unlableled_index用到的地方

    return train_loss.avg, top5.avg, top1.avg, batch_time.sum, count_true_1


def inference(args, model, device, test_loader, iter, log):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    log.write("Acc: %.2f%% \n" % (100. * correct / len(test_loader.dataset)))
    log.flush()

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)


def warm_up_testing(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)


def validating(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, _, target, _, _, ) in enumerate(test_loader): # 这里的数量是5000
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            loss_per_batch.append(F.nll_loss(output, target).item())
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = np.array(100. * correct / len(test_loader.dataset))

    return (loss_per_epoch, acc_val_per_epoch)