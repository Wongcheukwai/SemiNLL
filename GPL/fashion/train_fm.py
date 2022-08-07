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
from dataloader_fashion_v1 import fashion
import torch.nn.functional as F
from torchcontrib.optim import SWA
sys.path.append('../utils_pseudoLab/')
from utils_ssl_fm import *
from ssl_networks import resnet18
from ssl_networks import resnet18_wndrop
from resnet import *
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Number of images in each mini-batch')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
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
    parser.add_argument('--dataset', type=str, default='fashion', help='Dataset name')
    parser.add_argument('--swa', type=str, default='True', help='Apply SWA')
    parser.add_argument('--swa_start', type=int, default=20, help='Start SWA')
    parser.add_argument('--swa_freq', type=float, default=2, help='Frequency')
    parser.add_argument('--swa_lr', type=float, default=0.01, help='LR')
    parser.add_argument('--DApseudolab', type=str, default="False", help='Apply data augmentation when computing pseudolabels')
    parser.add_argument('--drop_extra_forward', type=str, default='True', help='Do an extra forward pass to compute the labels without dropout.')
    parser.add_argument('--network', type=str, help='The backbone of the network')


    ########## 后来加的 ########
    parser.add_argument('--noise_mode', default='sym')
    parser.add_argument('--r', default=0.2, type=float, help='noise ratio')
    parser.add_argument('--epoch', default=120, type=int, help='number of epoch')
    parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
    ########## 后来加的 ########

    args = parser.parse_args()
    wandb.init(project="fashion_repeated", group="gpl", job_type="%s" % args.r)

    wandb.config.update(args)  # 用weight and bias来记录
    return args

'''
def create_model():
    
    if args.network == "resnet18":
        print("Loading Resnet18...")
        model = resnet18(num_classes=args.num_classes).cuda()

    elif args.network == "resnet18_wndrop":
        print("Loading Resnet18withdrop...")
        model = resnet18_wndrop(num_classes=args.num_classes).cuda()
    model = model.cuda()
    
    return model
'''

def create_model():
    if args.dataset == "fashion":
        model = ResNet18(args.num_classes)
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

    # input dataset

    transform_train = transforms.Compose([transforms.ColorJitter(brightness= 0.4, contrast= 0.4, saturation= 0.4, hue= 0.1),
                                          transforms.RandomCrop(28, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = fashion(root='./fashion/',
                            train=True,
                            test=False,
                            args=args,
                            transform=transform_train,
                            noise_file='./fashion/%.1f_%s.json' % (args.r, args.noise_mode),
                            )

    test_dataset = fashion(root='./fashion/',
                           train=False,
                           test=True,
                           args=args,
                           transform=transform_test,
                           )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True, num_workers=5)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False, num_workers=5)



    print('| Building net')
    model = create_model()
    wandb.watch(model)

    cudnn.benchmark = False

    base_optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    optimizer = SWA(base_optimizer, swa_lr=args.swa_lr, swa_start=args.swa_start)

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
        train_time, _ = train_CrossEntropy(args, model, device, train_loader, optimizer, warm_up_epoch, status)

        inference(model, device, test_loader)
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

        lr = args.lr
        if 0 <= iter < 40:
            lr = lr
        elif 40 <= iter < 80:
            lr /= 10
        elif iter >= 80:
            lr /= 100
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for epoch in range(1, 2):

            # train for one epoch
            loss_per_epoch_train, \
            top_5_train_ac, \
            top1_train_ac, \
            train_time, train_true = train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, status)

            # 在这里进行测试
            model.eval()
            inference(model, device, test_loader)

    # applying swa
    model.eval()
    if args.swa == 'True':
        optimizer.swap_swa_sgd()
        optimizer.bn_update(train_loader, model, device)
        inference(model, device, test_loader)

if __name__ == "__main__":
    args = parse_args()
    # train
    main(args)