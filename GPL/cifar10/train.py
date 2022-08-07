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

from dataset.cifar10 import get_dataset
import torch.nn.functional as F
sys.path.append('../utils_pseudoLab/')
from TwoSampler import *
from utils_ssl import *

from ssl_networks import CNN as MT_Net
from PreResNet import PreactResNet18_WNdrop
from wideArchitectures import WRN28_2_wn

sys.path.append('../')
from pytorchtools import EarlyStopping

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='max learning rate')
    parser.add_argument('--batch_size', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', type=int, default=150, help='Training epoches')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dataset_type', default='ssl', help='How to prepare the data: only labeled data for the warmUp ("ssl_warmUp") or unlabeled and labeled for the SSL training ("ssl")')
    parser.add_argument('--train_root', default='./data', help='Root for train data')
    parser.add_argument('--labeled_samples', type=int, default=45000, help='Number of labeled samples(whole dataset)')
    parser.add_argument('--reg1', type=float, default=0.8, help='Hyperparam for loss')
    parser.add_argument('--reg2', type=float, default=0.4, help='Hyperparam for loss')
    parser.add_argument('--network', type=str, default='MT_Net', help='The backbone of the network')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--seed_val', type=int, default=1, help='Seed for the validation split')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    #parser.add_argument('--experiment_name', type=str, default = 'Proof',help='Name of the experiment (for the output files)')
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
    parser.add_argument('--labeled_batch_size', default=16, type=int, metavar='N', help="Labeled examples per minibatch (default: no constrain)")
    parser.add_argument('--val_samples', type=int, default=5000, help='Number of samples to be kept for validation (from the training set))')
    parser.add_argument('--DA', type=str, default='standard', help='Chose the type of DA')
    parser.add_argument('--DApseudolab', type=str, default="False", help='Apply data augmentation when computing pseudolabels')
    parser.add_argument('--drop_extra_forward', type=str, default='True', help='Do an extra forward pass to compute the labels without dropout.')

    parser.add_argument('--status', default='debug', type=str, help='determine run or debug')
    parser.add_argument('--noise_mode', default='sym')
    parser.add_argument('--r', default=0.8, type=float, help='noise ratio')
    parser.add_argument('--idex', default=None, type=int, help='number of checkpoint')
    parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--alpha_1', default=0.6, type=float, help='alpha')
    parser.add_argument('--clean_method', default='self', type=str, help='determine which method for cleansing')
    parser.add_argument('--lr-rampup', default=0, type=int, metavar='EPOCHS',
                        help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=None, type=int, metavar='EPOCHS',
                        help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--initial-lr', default=0.0, type=float,
                        metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--addtion', default=False, type=bool, help='lamda addtion')
    parser.add_argument('--resume_model', default=None, type=str, help='determine resumed model')
    parser.add_argument('--resume_optimizer', default=None, type=str, help='determine resumed optimizer')
    parser.add_argument('--whole', default=False, type=bool, help='unlabeled as a whole or not')


    args = parser.parse_args()
    return args


def data_config(args, transform_train, transform_test):

    trainset, unlabeled_indexes, labeled_indexes, valset, corrupted_labels, clean_labels = get_dataset(args, transform_train, transform_test)

    # this is for valset
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # this is for testset
    testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # train and val
    print('-------> Data loading')
    return trainset, test_loader, val_loader, unlabeled_indexes, labeled_indexes, corrupted_labels, clean_labels

best_prec1 = 0.0


def main(args):
    global best_prec1

    #####################
    # Initializing seeds and preparing GPU
    if args.cuda_dev == 1:
        torch.cuda.set_device(1)
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
    trainset, test_loader, val_loader, unlabeled_indexes, labeled_indexes, corrupted_labels, clean_labels = data_config(args, transform_train, transform_test)

    if args.network == "MT_Net":
        print("Loading MT_Net...")
        model = MT_Net(num_classes = args.num_classes, dropRatio = args.dropout).to(device)

    elif args.network == "WRN28_2_wn":
        print("Loading WRN28_2...")
        model = WRN28_2_wn(num_classes = args.num_classes, dropout = args.dropout).to(device)

    elif args.network == "PreactResNet18_WNdrop":
        print("Loading preActResNet18_WNdrop...")
        model = PreactResNet18_WNdrop(drop_val = args.dropout, num_classes = args.num_classes).to(device)

    print('Total params: %2.fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    milestones = args.M

    #####################
    if args.dataset == 'cifar10':
        if args.status == 'run':
            warm_up = 10
        elif args.status == 'debug':
            warm_up = 0
    elif args.dataset == 'cifar100':
        if args.status == 'run':
            warm_up = 30
        elif args.status == 'debug':
            warm_up = 0
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
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    loss_train_epoch = []
    loss_val_epoch = []
    acc_train_per_epoch = []
    acc_val_per_epoch = []

    # optionally resume from a checkpoint
    if args.resume_model:
        assert os.path.isfile(args.resume_model), "=> no checkpoint found at '{}'".format(args.resume_model)
        print("=> loading checkpoint '{}'".format(args.resume_model))
        model.load_state_dict(torch.load(args.resume_model))
        optimizer.load_state_dict(torch.load(args.resume_optimizer))

    cudnn.benchmark = False


    ####################################################################################################
    ###############################                 WARMUP                ##############################
    ####################################################################################################

    if args.status == "run" and args.resume_model == None:
        print("start warmp up for the whole dataset")
        for warm_up_epoch in range(1, warm_up + 1):
            #scheduler.step()
            # train for one epoch

            train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            unlabeled_indexes_warmup = []

            loss_per_epoch_train, \
            top_5_train_ac, \
            top1_train_ac, \
            train_time = train_CrossEntropy(args, model, device, train_loader, optimizer, warm_up_epoch, unlabeled_indexes_warmup)

            loss_train_epoch += [loss_per_epoch_train]

            loss_per_epoch_test, acc_val_per_epoch_i = warm_up_testing(args, model, device, test_loader)

    print('relabel after warm up')

    if args.resume_model != None: # 读取checkpoint时的relabel
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model.eval()
    results_1 = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)
    for images, images_pslab, _, _, index, _, _, _, _ in train_loader:
        images = images.to(device)
        outputs = model(images)
        # prob, _ = loss_soft_reg_ep(outputs, labels, soft_labels, device, args)
        prob = F.softmax(outputs, dim=1)
        results_1[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

    train_loader.dataset.update_labels(results_1, unlabeled_indexes)

    print("Start training...")
    ####################################################################################################
    ###############################               TRAINING                ##############################
    ####################################################################################################

    if args.status == 'run':
        model_path = '%s/model_%.1f_%s_%d.pt' % (total_path, args.r, args.noise_mode, args.idex)
        optimizer_path = '%s/optimizer_%.1f_%s_%d.pt' % (total_path, args.r, args.noise_mode, args.idex)
    elif args.status == 'debug':
        model_path = '%s/model_%.1f_%s_%s.pt' % (total_path, args.r, args.noise_mode, "debug")
        optimizer_path = '%s/optimizer_%.1f_%s_%s.pt' % (total_path, args.r, args.noise_mode, "debug")
    print('model_direction', model_path)

    if args.status == 'run':
        # when run
        default_exp_dir = '{}'.format(args.idex)
        stats_log = open('./status_txt/%s_%.1f_%s_%s' % (args.dataset, args.r, args.noise_mode, default_exp_dir) + '_stats.txt','w')
    elif args.status == 'debug':
        # when debug
        stats_log = open('./status_txt/%s_%.1f_%s' % (args.dataset, args.r, args.noise_mode) + '_stats.txt', 'w')

    batch_sampler = TwoStreamBatchSampler(unlabeled_indexes, labeled_indexes, args.batch_size, args.labeled_batch_size)
    train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)

    # 定义self的prediction
    Z = torch.zeros(len(train_loader.dataset), args.num_classes).float().to(device)
    results_3 = torch.zeros(len(train_loader.dataset), args.num_classes).float().to(device)
    results_2 = torch.zeros(len(train_loader.dataset), args.num_classes).float().to(device)

    for iter in range(100):
        #early_stopping = EarlyStopping(patience=20, verbose=True, delta=0.01, model_path=model_path, optimizer_path=optimizer_path)
        print("\n###################### doing the %d iteration ########################" % (iter + 1))
        stats_log.write('Iter:%.3d \t' % (iter+1))
        stats_log.flush()

        if iter < 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
        elif iter >= 50 and iter < 75:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr/2
        elif iter >= 75:
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr/10

        if args.status == "debug":
            epoch_num = 0
        elif args.status == "run":
            epoch_num = args.epoch + 1

        for epoch in range(1, epoch_num):
            st = time.time()
            #scheduler.step()

            # train for one epoch
            if args.whole == True:
                loss_per_epoch_train, \
                top_5_train_ac, \
                top1_train_ac, \
                train_time = train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, unlabeled_indexes)
            elif args.whole == False:
                if iter == 0:
                    loss_per_epoch_train, \
                    top_5_train_ac, \
                    top1_train_ac, \
                    train_time = train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, unlabeled_indexes)
                else:
                    loss_per_epoch_train, \
                    top_5_train_ac, \
                    top1_train_ac, \
                    train_time = train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, final_unlabeled_idx)

            # validation
            loss_per_epoch_val, acc_val_per_epoch_i = validating(args, model, device, val_loader)
            if acc_val_per_epoch_i >= best_prec1:
                best_prec1 = max(acc_val_per_epoch_i, best_prec1)
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optimizer_path)
                print('save model')
            else:
                print('do not save model')

            print('bestacc', best_prec1)
            '''
            early_stopping(acc_val_per_epoch_i, model, optimizer)

            if early_stopping.early_stop:
                print("----- Early stopping after training for %d epoch -----" % (epoch))
                break
            '''
        print("loading checkpoint for the %s cleasing iteration" % (iter + 1))
        model.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optimizer_path))

        ############################### CLEANSING #############################
        model.eval()

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        #这里开始cleansing了
        if args.swa == 'True':
            optimizer.swap_swa_sgd()
            optimizer.bn_update(train_loader, model, device)
            _, acc_val_swa = validating(args, model, device, val_loader)

            if acc_val_swa > best_prec1:
                # print('yes_swa')
                # best_prec1 = acc_val_swa
                torch.save(model.state_dict(), model_path)
                torch.save(optimizer.state_dict(), optimizer_path)


        if args.clean_method == "mixmatch":
            # 这是mixmatch的区分方法
            #results_2 = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)
            results_2 = torch.zeros(len(train_loader.dataset), args.num_classes).float().to(device)
            with torch.no_grad():
                for _, _, _, _, index, _, _, img1, img2 in train_loader:
                    img1, img2 = img1.to(device), img2.to(device)
                    output_img1, output_img2 = model(img1), model(img2)
                    # compute output
                    pu = (F.softmax(output_img1, dim=1) + F.softmax(output_img2, dim=1)) / 2  # 64,10
                    ptu = pu ** (1 / args.T)  # temparature sharpening 64,10
                    targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                    targets_u = targets_u.detach()  # 64,10
                    #save
                    #results_2[index.detach().numpy().tolist()] = targets_u.cpu().detach().numpy().tolist()
                    results_2[index] = targets_u.data.clone()

            Z = args.alpha_1 * Z + (1. - args.alpha_1) * results_2
            _, predicted = torch.max(Z, 1)
            predicted = predicted.cpu().numpy().tolist()

            labeled_idx_1 = []
            unlabeled_idx_1 = []
            count_true_1 = 0

            for h in unlabeled_indexes:
                if predicted[h] == corrupted_labels[h]:
                    labeled_idx_1.append(h)
                    if predicted[h] == clean_labels[h]:
                        count_true_1 += 1
                else:
                    unlabeled_idx_1.append(h)
            print("MIXMATCH: there are %d labelled instances" % (len(labeled_idx_1)))
            print("MIXMATCH: there are %d true label after cleasing" % (count_true_1))

        # 这是self的区分方法
        elif args.clean_method == "self":
            with torch.no_grad():
                for img3, _, _, _, index, _, _, _, _ in train_loader:
                    img3 = img3.to(device)
                    output_img_3 = model(img3)
                    #save
                    #results_3[index.detach().numpy().tolist()] = output_img_3.cpu().detach().numpy().tolist()
                    results_3[index] = output_img_3.data.clone()
            Z = args.alpha_1 * Z + (1. - args.alpha_1) * results_3
            _, predicted_3 = torch.max(Z, 1)
            predicted_3 = predicted_3.cpu().numpy().tolist()

            labeled_idx_2 = []
            count_true_2 = 0
            unlabeled_idx_2 = []

            for h2 in unlabeled_indexes:
                if predicted_3[h2] == corrupted_labels[h2]:
                    labeled_idx_2.append(h2)
                    if predicted_3[h2] == clean_labels[h2]:
                        count_true_2 += 1
                else:
                    unlabeled_idx_2.append(h2)
            print("SELF: there are %d labelled instances" % (len(labeled_idx_2)))
            print("SELF: there are %d true label after cleasing" % (count_true_2))


        if args.clean_method == 'mixmatch':
            final_labeled_idx = labeled_idx_1
            final_true = count_true_1
            final_unlabeled_idx = unlabeled_idx_1
        elif args.clean_method == 'self':
            final_labeled_idx = labeled_idx_2
            final_true = count_true_2
            final_unlabeled_idx = unlabeled_idx_2

        stats_log.write("labeled: %d\t" % (len(final_labeled_idx)))
        stats_log.write("true: %d\t" % (final_true))
        stats_log.flush()

        if args.whole == True:
            batch_sampler = TwoStreamBatchSampler(unlabeled_indexes, final_labeled_idx, args.batch_size, args.labeled_batch_size)
        elif args.whole == False:
            batch_sampler = TwoStreamBatchSampler(final_unlabeled_idx, final_labeled_idx, args.batch_size, args.labeled_batch_size)

        train_loader = torch.utils.data.DataLoader(trainset, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
        #print("there are %d unlabeled data after cleasing" % (len(final_unlabeled_idx)))
        # 在这里进行测试
        loss_per_epoch_test, acc_val_per_epoch_i = testing(args, model, device, test_loader, iter, stats_log)

        #stats_log.write("stop: %d\n" % (epoch))
        #stats_log.flush()

    # applying swa
    model.eval()

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=True)
    if args.swa == 'True':
        optimizer.swap_swa_sgd()
        optimizer.bn_update(train_loader, model, device)
        loss_swa, acc_val_swa = testing(args, model, device, test_loader, iter+1, stats_log)

        if acc_val_swa > best_prec1:
            # print('yes_swa')
            # best_prec1 = acc_val_swa
            torch.save(model.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)


if __name__ == "__main__":
    args = parse_args()
    # train
    main(args)
