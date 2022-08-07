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
from dataloader_mnist_v1 import MNIST
import torch.nn.functional as F
from torchcontrib.optim import SWA
sys.path.append('../utils_pseudoLab/')
from utils_ssl_m import *

import wandb
#wandb.init(project="cifar100_pg-and-mnist_pg")


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)
        '''
        self.fc2 = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(256, args.num_classes))
        '''
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--train_root', default='./data', help='Root for train data')
    parser.add_argument('--reg1', type=float, default=0.8, help='Hyperparam for loss')
    parser.add_argument('--reg2', type=float, default=0.4, help='Hyperparam for loss')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--seed_val', type=int, default=1, help='Seed for the validation split')
    parser.add_argument('--loss_term', type=str, default='MixUp_ep', help='The loss to use: "Reg_ep" for CE, or "MixUp_ep" for M')
    parser.add_argument('--num_classes', type=int, default=10, help='Beta parameter for the EMA in the soft labels')
    parser.add_argument('--dropout', type=float, default=0.1, help='CNN dropout')
    parser.add_argument('--Mixup_Alpha', type=float, default=1, help='Alpha value for the beta dist from mixup')
    parser.add_argument('--cuda_dev', type=int, default=0, help='Set to 1 to choose the second gpu')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset name')
    parser.add_argument('--swa', type=str, default='True', help='Apply SWA')
    parser.add_argument('--swa_start', type=int, default=50, help='Start SWA')
    parser.add_argument('--swa_freq', type=float, default=5, help='Frequency')
    parser.add_argument('--swa_lr', type=float, default=0.01, help='LR')
    #parser.add_argument('--DA', type=str, default='standard', help='Chose the type of DA')
    parser.add_argument('--DApseudolab', type=str, default="False", help='Apply data augmentation when computing pseudolabels')
    parser.add_argument('--drop_extra_forward', type=str, default='True', help='Do an extra forward pass to compute the labels without dropout.')

    ########## 后来加的 ########
    parser.add_argument('--noise_mode', default='asymmetric')
    parser.add_argument('--r', default=0.3, type=float, help='noise ratio')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    ########## 后来加的 ########

    args = parser.parse_args()
    #wandb.init(project="mnist_repeated", group="gpl", job_type="a%s" % args.r)
    wandb.init(project="loss test")
    wandb.config.update(args)  # 用weight and bias来记录


    return args


def create_model():
    model = MLPNet()
    model = model.cuda()
    return model

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

    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
        transforms.ToTensor(),
    ])

    # input dataset

    train_dataset = MNIST(root='./mnist/',
                          download=False,
                          train=True,
                          transform=transform_train,
                          noise_type=args.noise_mode,
                          noise_rate=args.r
                          )

    test_dataset = MNIST(root='./mnist/',
                         download=False,
                         train=False,
                         transform=transforms.ToTensor(),
                         noise_type=args.noise_mode,
                         noise_rate=args.r
                         )
    '''
    truei = 0
    true_labels = train_dataset.train_labels
    noise_labels = train_dataset.train_noisy_labels

    for i in range(len(noise_labels)):
        if noise_labels[i] == true_labels[i]:
            truei += 1
    percent = truei/len(noise_labels)
    '''

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=5,
                                               drop_last=True,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=5,
                                              drop_last=True,
                                              shuffle=False)

    print('| Building net')
    model = create_model()
    wandb.watch(model)

    cudnn.benchmark = False

    base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    optimizer = SWA(base_optimizer, swa_lr=args.swa_lr, swa_start=args.swa_start)

    loss_train_epoch = []

    cudnn.benchmark = False

    warm_up = 5

    ####################################################################################################
    ###############################                 WARMUP                ##############################
    ####################################################################################################

    for warm_up_epoch in range(1, warm_up + 1):
        status = 'warmup'
        loss_per_epoch_train, \
        top_5_train_ac, \
        top1_train_ac, \
        train_time, _ = train_CrossEntropy(args, model, device, train_loader, optimizer, warm_up_epoch,status)

        loss_train_epoch += [loss_per_epoch_train]

        inference(model, device, test_loader, loss_per_epoch_train)
    '''
    print('relabel after warm up') # 给所有的unlabeled label relabel以防以后用到
    model.eval()
    results_1 = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)
    for images, images_pslab, _, _, index, _, _ in train_loader:
        images = images.to(device)
        outputs = model(images)
        prob = F.softmax(outputs, dim=1)
        results_1[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

    train_loader.dataset.update_labels(results_1)
    '''

    print("Start training...")
    ####################################################################################################
    ###############################               TRAINING                ##############################
    ####################################################################################################

    for iter in range(args.epoch):
        status = 'train'
        print("\n###################### doing the %d iteration ########################" % (iter + 1))

        if iter < 150:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        elif iter >= 150:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr/10

        for epoch in range(1, 2):

            # train for one epoch
            loss_per_epoch_train, \
            top_5_train_ac, \
            top1_train_ac, \
            train_time, train_true = train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, status)

            # 在这里进行测试
            model.eval()
            inference(model, device, test_loader,loss_per_epoch_train)

    # applying swa
    model.eval()
    if args.swa == 'True':
        optimizer.swap_swa_sgd()
        optimizer.bn_update(train_loader, model, device)
        inference(model, device, test_loader, loss_per_epoch_train)

if __name__ == "__main__":
    args = parse_args()
    # train
    main(args)