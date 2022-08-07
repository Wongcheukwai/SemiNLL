import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms

import numpy as np
import random
import sys
import argparse
import os
import time

from dataset.cifar10_f import get_dataset
import torch.nn.functional as F
sys.path.append('../utils_pseudoLab/')
from TwoSampler import *
from utils_ssl_f import *
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from ssl_networks import CNN as MT_Net
from PreResNet import PreactResNet18_WNdrop
from wideArchitectures import WRN28_2_wn
#import wandb

sys.path.append('../')
from pytorchtools import EarlyStopping

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--train_root', default='./data', help='Root for train data')
    parser.add_argument('--reg1', type=float, default=0.8, help='Hyperparam for loss')
    parser.add_argument('--reg2', type=float, default=0.4, help='Hyperparam for loss')
    parser.add_argument('--network', type=str, default='MT_Net', help='The backbone of the network')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--seed_val', type=int, default=1, help='Seed for the validation split')
    parser.add_argument('--loss_term', type=str, default='MixUp_ep', help='The loss to use: "Reg_ep" for CE, or "MixUp_ep" for M')
    parser.add_argument('--num_classes', type=int, default=10, help='Beta parameter for the EMA in the soft labels')
    parser.add_argument('--dropout', type=float, default=0.0, help='CNN dropout')
    parser.add_argument('--Mixup_Alpha', type=float, default=1, help='Alpha value for the beta dist from mixup')
    parser.add_argument('--cuda_dev', type=int, default=0, help='Set to 1 to choose the second gpu')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset name')
    parser.add_argument('--swa', type=str, default='True', help='Apply SWA')
    parser.add_argument('--swa_start', type=int, default=50, help='Start SWA')
    parser.add_argument('--swa_freq', type=float, default=5, help='Frequency')
    parser.add_argument('--swa_lr', type=float, default=0.01, help='LR')
    parser.add_argument('--DA', type=str, default='standard', help='Chose the type of DA')
    parser.add_argument('--DApseudolab', type=str, default="False", help='Apply data augmentation when computing pseudolabels')
    parser.add_argument('--drop_extra_forward', type=str, default='True', help='Do an extra forward pass to compute the labels without dropout.')

    ########## 后来加的 ########
    parser.add_argument('--noise_mode', default='sym')
    parser.add_argument('--r', default=0.8, type=float, help='noise ratio')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    ########## 后来加的 ########

    args = parser.parse_args()
    #wandb.init(project="cifar10_repeated", group="gpl", job_type="a%s" % args.r)
    #wandb.init(project="cvpr_rebuttal")
    #wandb.config.update(args)  # 用weight and bias来记录
    return args


def data_config(args, transform_train, transform_test):

    trainset, unlabeled_indexes, labeled_indexes, corrupted_labels, clean_labels = get_dataset(args, transform_train, transform_test)

    # this is for testset
    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # train and val
    print('-------> Data loading')
    return trainset, test_loader, unlabeled_indexes, labeled_indexes, corrupted_labels, clean_labels

best_prec1 = 0.0


def main(args):
    global best_prec1

    #####################
    # Initializing seeds and preparing GPU
    torch.cuda.set_device(args.cuda_dev)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)  # GPU seed
    random.seed(args.seed)  # python seed for image transformation
    np.random.seed(args.seed)
    #####################

    if args.dataset == 'cifar10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'cifar100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]

    if args.DA == "standard":
        transform_train = transforms.Compose([
            transforms.Pad(2, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif args.DA == "jitter":
        transform_train = transforms.Compose([
            transforms.Pad(2, padding_mode='reflect'),
            transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        print("Wrong value for --DA argument.")


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # data lodaer
    trainset, test_loader, unlabeled_indexes, labeled_indexes, corrupted_labels, clean_labels = data_config(args, transform_train, transform_test)

    if args.network == "MT_Net": # 3M
        print("Loading MT_Net...")
        model = MT_Net(num_classes = args.num_classes, dropRatio = args.dropout).to(device)

    elif args.network == "WRN28_2_wn": # 1M
        print("Loading WRN28_2...")
        model = WRN28_2_wn(num_classes = args.num_classes, dropout = args.dropout).to(device)

    elif args.network == "PreactResNet18_WNdrop": #11M
        print("Loading preActResNet18_WNdrop...")
        model = PreactResNet18_WNdrop(drop_val = args.dropout, num_classes = args.num_classes).to(device)

    print('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    #wandb.watch(model)

    '''
    ####################plot################
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    def visualize_tsne_points(tx, ty, labels):
        # initialize matplotlib plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        color = ['c', 'b', 'g', 'r', 'm', 'y', 'k', 'w']
        # for every class, we'll add a scatter plot separately
        for label in range(8):
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # add a scatter plot with the correponding color and label
            ax.scatter(current_tx, current_ty, c=color[i], label=label)

        # build a legend using the labels we set previously
        ax.legend(loc='best')

        # finally, show the plot
        plt.show()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    
    #results_p = np.zeros((len(train_loader.dataset), 128), dtype=np.float32)
    #results_t = np.zeros((len(train_loader.dataset)), dtype=np.int)

    
    #model.load_state_dict(torch.load('/home/zhuwang/Data/Downloads/pseudo/cifar10/checkpoint/plus_cifar10_80.pt'))
    #model.eval()
    
    #with torch.no_grad():
    #    correct = 0
    #    total = 0
    #    for images, images_pslab, labels, _, index, _, _ in train_loader:
    #        inputs, targets = images.cuda(), labels.cuda()
    #        outputs, embedding = model(inputs)
    #        results_p[index] = embedding.cpu().numpy()
    #        results_t[index] = targets.cpu().numpy()
    #        _, predicted = torch.max(outputs, 1)

    #        total += targets.size(0)
    #        correct += predicted.eq(targets).cpu().sum().item()
    #    acc = 100. * correct / total
    
    #np.save('/home/zhuwang/Data/Downloads/pseudo/cifar10/checkpoint/gpl_p.npy', results_p)
    #np.save('/home/zhuwang/Data/Downloads/pseudo/cifar10/checkpoint/gpl_t.npy', results_t)
    

    results_p = np.load('/home/zhuwang/Data/Downloads/pseudo/cifar10/checkpoint/plus_p.npy')
    results_t = np.load('/home/zhuwang/Data/Downloads/pseudo/cifar10/checkpoint/plus_t.npy')

    cls_0_idx = np.where(results_t == 3)[0]
    cls_1_idx = np.where(results_t == 9)[0]

    cls_a_idx = np.random.choice(cls_0_idx, 500, replace=False)
    cls_b_idx = np.random.choice(cls_1_idx, 500, replace=False)

    feature_a = results_p[cls_a_idx]
    feature_b = results_p[cls_b_idx]

    tsne_a = TSNE(n_components=2).fit_transform(feature_a)
    tsne_a = scale_to_01_range(tsne_a)
    tsne_b = TSNE(n_components=2).fit_transform(feature_b)
    tsne_b = scale_to_01_range(tsne_b)

    a_clean_idx = clean_labels[cls_a_idx] == results_t[cls_a_idx]
    a_noisy_idx = clean_labels[cls_a_idx] != results_t[cls_a_idx]
    b_clean_idx = clean_labels[cls_b_idx] == results_t[cls_b_idx]
    b_noisy_idx = clean_labels[cls_b_idx] != results_t[cls_b_idx]

    #plt.rcParams['figure.figsize'] = [32, 32]
    #plt.subplots_adjust(wspace=0.3, hspace=0.3)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(tsne_a[a_clean_idx][:, 0].ravel(), tsne_a[a_clean_idx][:, 1].ravel(), c='b', marker='o', s=10, label='class A: clean')
    ax.scatter(tsne_a[a_noisy_idx][:, 0].ravel(), tsne_a[a_noisy_idx][:, 1].ravel(), c='m', marker='x', s=30, label='class A: noisy')
    ax.scatter(tsne_b[b_clean_idx][:, 0].ravel(), tsne_b[b_clean_idx][:, 1].ravel(), c='r', marker='o', s=10, label='class B: clean')
    ax.scatter(tsne_b[b_noisy_idx][:, 0].ravel(), tsne_b[b_noisy_idx][:, 1].ravel(), c='c', marker='x', s=30, label='class B: noisy')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    #ax.legend_.remove()
    #ax.legend(loc='lower center', ncol=2, fontsize=20)
    plt.savefig('/home/zhuwang/Data/Downloads/pseudo/cifar10/checkpoint/mnist.pdf')
    plt.show()

    # get x and y
    # tx = tsne[:, 0]
    # ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    # tx = scale_to_01_range(tx)
    # ty = scale_to_01_range(ty)
    # visualize_tsne_points(tx, ty, results_t)
    print('done')
    
    ###################plot################
    '''

    #####################
    # 调试时候的一些设置
    if args.dataset == 'cifar10':
        warm_up = 10
    elif args.dataset == 'cifar100':
        warm_up = 30
    #####################

    if args.swa == 'True':
        # to install it:
        # pip3 install torchcontrib
        # git clone https://github.com/pytorch/contrib.git
        # cd contrib
        # sudo python3 setup.py install
        from torchcontrib.optim import SWA
        base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        optimizer = SWA(base_optimizer, swa_lr=args.swa_lr, swa_start=args.swa_start)

    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)

    loss_train_epoch = []
    loss_val_epoch = []
    acc_train_per_epoch = []
    acc_val_per_epoch = []

    cudnn.benchmark = False


    ####################################################################################################
    ###############################                 WARMUP                ##############################
    ####################################################################################################

    for warm_up_epoch in range(1, warm_up + 1):
        #start = time.time()
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
        unlabeled_indexes_warmup = []

        loss_per_epoch_train, \
        top_5_train_ac, \
        top1_train_ac, \
        train_time, _, _ = train_CrossEntropy(args, model, device, train_loader, optimizer, warm_up_epoch, unlabeled_indexes_warmup)

        loss_train_epoch += [loss_per_epoch_train]
        #end = time.time()
        #running_time = end - start
        #print('time cost : %.5f sec' % running_time)
        inference(model, device, test_loader, 0, 0)

    print('relabel after warm up') # 给所有的unlabeled label relabel以防以后用到
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    model.eval()
    results_1 = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)
    for images, images_pslab, _, _, index, _, _ in train_loader:
        images = images.to(device)
        outputs = model(images)
        prob = F.softmax(outputs, dim=1)
        results_1[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

    train_loader.dataset.update_labels(results_1, unlabeled_indexes)

    print("Start training...")
    ####################################################################################################
    ###############################               TRAINING                ##############################
    ####################################################################################################

    time_total = 0

    for iter in range(args.epoch):
        print("\n###################### doing the %d iteration ########################" % (iter + 1))

        if iter < 150:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        elif iter >= 150:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr/10

        for epoch in range(1, 2):
            #start = time.time()
            # train for one epoch
            loss_per_epoch_train, \
            top_5_train_ac, \
            top1_train_ac, \
            train_time, train_true, results_precison = train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, unlabeled_indexes)
            #end = time.time()
            #running_time = end - start
            #time_total += running_time
            #print('time cost : %.5f sec' % running_time)
            #print('time cost : %.5f sec' % time_total)
            # 在这里进行测试

            pred = (results_precison > args.p_threshold)  # 50000
            pred_idx = pred.nonzero()[0]  # 这个是labeled的个数

            clean_number = 0
            for k in pred_idx:
                if corrupted_labels[k] == clean_labels[k]:
                    clean_number += 1

            label_precision = clean_number / len(pred_idx)
            label_recall = clean_number / 39990

            model.eval()
            inference(model, device, test_loader, label_precision, label_recall)
            model_path = '/home/zhuwang/Data/Downloads/pseudo/cifar10/checkpoint/gpl_cifar10_80_true.pt'
            torch.save(model.state_dict(), model_path)

    # applying swa
    model.eval()
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=False)
    if args.swa == 'True':
        optimizer.swap_swa_sgd()
        optimizer.bn_update(train_loader, model, device)
        #inference(model, device, test_loader)


if __name__ == "__main__":
    args = parse_args()
    # train
    main(args)