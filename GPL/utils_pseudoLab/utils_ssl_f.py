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
from sklearn.mixture import GaussianMixture
#import wandb

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
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
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

def train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, unlabeled_indexes):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    loss_per_batch = []
    acc_train_per_batch = []

    end = time.time()

    results = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32) # 用来更新unlaleled data的
    results_precison = np.zeros(len(train_loader.dataset), dtype=np.float32)

    if args.loss_term == "Reg_ep":
        alpha = None
    elif args.loss_term == "MixUp_ep":
        alpha = args.Mixup_Alpha

    count_total = 0
    counter = 1
    for i, (imgs, img_pslab, labels, soft_labels, index, unlabeled_labels, unlabeled_soft_labels) in enumerate(train_loader):

        images, labels, soft_labels, unlabeled_labels, unlabeled_soft_labels = imgs.to(device), \
                                                                               labels.to(device), soft_labels.to(device),\
                                                                               unlabeled_labels.to(device), unlabeled_soft_labels.to(device)

        if len(unlabeled_indexes) != 0:  # 正式训练的时候，0的时候是warmup
            model.eval()

            #results_2 = torch.zeros(len(index), args.num_classes).float().to(device)

            CE = nn.CrossEntropyLoss(reduction='none')
            with torch.no_grad():
                input_x = img_pslab.to(device)
                targets = labels
                outputs = model(input_x)
                loss = CE(outputs, targets)
            losses = (loss - loss.min()) / (loss.max() - loss.min())  # minibatch
            input_loss = losses.reshape(-1, 1).cpu()

            # 加入GMM
            gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm.fit(input_loss)
            prob = gmm.predict_proba(input_loss)
            prob = prob[:, gmm.means_.argmin()]
            results_precison[index.detach()] = prob

            count_labeled = 0
            for i in range(len(prob)):
                if prob[i] < args.p_threshold:
                    labels[i] = unlabeled_labels[i]
                    soft_labels[i] = unlabeled_soft_labels[i]
                else:
                    count_labeled += 1
                    count_total += 1

            if count_labeled == 0:
                print('bad luck')
                continue

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

            images, targets_a, targets_b, lam = mixup_data(images, soft_labels, alpha, device)# target b是打乱了顺序后的label， images是混淆了xa和xb的

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

    print('there are %d labeled data in this epoch' % (count_total))

    #这个epoch结束了
    if args.swa == 'True':
        if epoch > args.swa_start and epoch%args.swa_freq == 0 :
            optimizer.update_swa()

    # update soft labels
    train_loader.dataset.update_labels(results, unlabeled_indexes) # 这里才是unlableled_index用到的地方

    return train_loss.avg, top5.avg, top1.avg, batch_time.sum, count_total, results_precison


def inference(model, device, test_loader, precision, recall):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = F.log_softmax(output, dim=1)

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('Test set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    #wandb.log({"Test Accuracy": 100. * correct / len(test_loader.dataset), "Precision": precision, "Recall": recall})
