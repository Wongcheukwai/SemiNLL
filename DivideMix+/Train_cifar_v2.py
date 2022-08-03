from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import os
import argparse
import numpy as np
from PreResNet import *
from ssl_networks import CNN as MT_Net
from sklearn.mixture import GaussianMixture
import dataloader_cifar_v2 as dataloader
import wandb
import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.8, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--index', type=str)
parser.add_argument('--network', type=str)
parser.add_argument('--argument', type=str)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.set_num_threads(int(3))
#torch.autograd.set_detect_anomaly(True)

torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
# Training

def train(epoch, net, optimizer, train_loader, args, true_label, noise_label):
    net.train()
    num_iter = (len(train_loader.dataset) // args.batch_size) + 1
    count_true = 0
    count_label = 0
    net.eval()
    for bidex, (input_x, input_x2, labels_x, input_x3, input_x4, index) in enumerate(train_loader):
        results_2 = torch.zeros(len(index), args.num_class).float().cuda()
        if args.argument == 'no':
            with torch.no_grad():
                inputs = input_x3.cuda()
                outputs_x3 = net(inputs)
                results_2 = outputs_x3.data.clone().detach()
        elif args.argument == 'yes':
            with torch.no_grad():
                img1, img2 = input_x3.cuda(), input_x4.cuda()
                output_img1, output_img2 = net(img1), net(img2)
                # compute output
                pu = (F.softmax(output_img1, dim=1) + F.softmax(output_img2, dim=1)) / 2  # 64,10
                ptu = pu ** (1 / 0.5)  # temparature sharpening 64,10
                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()  # 64,10
                results_2 = targets_u.data.clone().detach()

        l_index = []
        u_idex = []

        if args.dataset == 'cifar10':
            _, predicted = torch.max(results_2, 1)
            predicted = predicted.cpu().numpy().tolist()

            for h in range(len(index)):
                if predicted[h] != noise_label[index[h]]:
                    u_idex.append(h)
                else:
                    l_index.append(h)
                    count_label += 1
                    if predicted[h] == true_label[index[h]]:
                        count_true += 1

        elif args.dataset == 'cifar100':
            _, predicted = results_2.topk(5, dim=1, largest=True, sorted=False)
            predicted = predicted.cpu().numpy().tolist()

            for h in range(len(index)):
                if not noise_label[index[h]] in predicted[h]:
                    u_idex.append(h)
                else:
                    l_index.append(h)
                    count_label += 1
                    if true_label[index[h]] in predicted[h]:
                        count_true += 1

        net.train()
        inputs_x = input_x[l_index]
        labels_x = labels_x[l_index]

        inputs_u = input_x[u_idex]
        inputs_u2 = input_x2[u_idex]

        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_u.size(0)
        if batch_size == 0 or batch_size_2 == 0:
            print('bad luck')
            continue
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)  # 64,10

        inputs_x, labels_x = inputs_x.cuda(), labels_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u = net(inputs_u)  # 64,10
            outputs_u2 = net(inputs_u2)  # 64,10

            pu = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2   # 64，10
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach() # 64，10

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0) # 256,3,32,32
        all_targets = torch.cat([labels_x, targets_u, targets_u], dim=0) # 256,10

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size]
        logits_u = logits[batch_size:]
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+bidex/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, bidex+1, num_iter, Lx.item(), Lu.item()))
        sys.stdout.flush()
        '''
    print('true label: %d\t  labeled: %d\t' % (count_true, count_label))
    test_log.write('labeled:%3d\t' % (count_true))
    test_log.flush()


def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, labels, path) in enumerate(dataloader):      
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        if args.noise_mode=='asym':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty      
        elif args.noise_mode=='sym':   
            L = loss
        L.backward()  
        optimizer.step() 
        '''
        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()
        '''

def inference(epoch, net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch, acc))
    #wandb.log({"Test Accuracy": acc})
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch, acc))
    test_log.flush()  


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)


class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


def create_model():
    if args.network == "MT":
        model = MT_Net(num_classes=args.num_class, dropRatio=0)
    elif args.network == "RES":
        model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model

test_log=open('./checkpoint_v2/%s_%.1f_%s_%s'%(args.dataset,args.r,args.noise_mode,args.index)+'_acc.txt','w')

if args.dataset=='cifar10':
    warm_up = 0
elif args.dataset=='cifar100':
    warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=test_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net = create_model()
#wandb.watch(net)
cudnn.benchmark = False

criterion = SemiLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

train_loader = loader.run('train')
eval_loader, noise_label, true_label = loader.run('eval_train')
test_loader = loader.run('test')
time_total = 0

for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr       

    if epoch<warm_up:
        start = time.time()
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net')
        warmup(epoch, net, optimizer, warmup_trainloader)
        end = time.time()
        running_time = end - start

        print('time cost : %.5f sec' % running_time)
    else:

        start = time.time()
        print('Train Net1')
        train(epoch, net, optimizer, train_loader, args, true_label, noise_label) # train net1

        end = time.time()
        running_time = end - start
        time_total += running_time
        print('time cost : %.5f sec' % running_time)
        print('time cost : %.5f sec' % time_total)

    inference(epoch, net)


