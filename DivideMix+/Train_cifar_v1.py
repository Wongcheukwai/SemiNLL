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
import dataloader_cifar_v1 as dataloader
from torchnet.meter import AUCMeter
import time
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='sym')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.2, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--data_path', default='./cifar-10', type=str, help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--index', type=str)
parser.add_argument('--network', type=str, default='MT')
parser.add_argument('--kl', type=bool, default=False)
parser.add_argument('--co_kl', type=float)
args = parser.parse_args()

torch.cuda.set_device(args.gpuid)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.set_num_threads(int(3))
torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode

# Training
def train(epoch, net, net2, optimizer, train_loader, args):
    net.train()
    net2.eval() #fix one network and train the other

    num_iter = (len(train_loader.dataset) // args.batch_size) + 1
    counter_for_labeled = 0
    results_1 = np.zeros(len(train_loader.dataset), dtype=np.float32)
    #holder = 0
    for bidex, (input_x, input_x2, labels_x, input_x3, index) in enumerate(train_loader):
        #start_10 = time.time()
        with torch.no_grad():
            inputs, targets = input_x3.cuda(), labels_x.cuda()
            outputs = net2(inputs)
            loss = CE(outputs, targets)
        losses = (loss - loss.min()) / (loss.max() - loss.min())  # minibatch
        input_loss = losses.reshape(-1, 1).cpu()

        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        results_1[index.detach()] = prob
        l_index = []
        u_idex = []
        for i in range(len(prob)):
            if prob[i] >= args.p_threshold:
                l_index.append(i)
                counter_for_labeled += 1
            else:
                u_idex.append(i)

        #end_10 = time.time()
        #running_time = end_10 - start_10
        #holder += running_time
        #print('time cost : %.5f sec' % running_time)

        inputs_x = input_x[l_index]
        inputs_x2 = input_x2[l_index]
        labels_x = labels_x[l_index]
        w_x = prob[l_index]
        w_x = torch.Tensor(w_x)
        inputs_u = input_x[u_idex]
        inputs_u2 = input_x2[u_idex]

        batch_size = inputs_x.size(0)
        if batch_size == 0:
            print('bad luck')
            continue
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)  # 64,10
        w_x = w_x.view(-1,1).type(torch.FloatTensor) # 64,1

        inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
        inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()

        with torch.no_grad():
            # label co-guessing of unlabeled samples
            outputs_u11 = net(inputs_u)
            outputs_u12 = net(inputs_u2)
            outputs_u21 = net2(inputs_u)
            outputs_u22 = net2(inputs_u2)            
            
            pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4   # 64，10
            ptu = pu**(1/args.T) # temparature sharpening
            
            targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
            targets_u = targets_u.detach() # 64，10

            # label refinement of labeled samples
            outputs_x = net(inputs_x)# 64，10
            outputs_x2 = net(inputs_x2)            
            
            px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
            px = w_x*labels_x + (1-w_x)*px
            ptx = px**(1/args.T) # temparature sharpening
                       
            targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize           
            targets_x = targets_x.detach()       

        # mixmatch
        l = np.random.beta(args.alpha, args.alpha)        
        l = max(l, 1-l)
                
        all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0) # 256,3,32,32
        all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0) # 256,10

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        
        mixed_input = l * input_a + (1 - l) * input_b        
        mixed_target = l * target_a + (1 - l) * target_b
                
        logits = net(mixed_input)
        logits_x = logits[:batch_size*2]
        logits_u = logits[batch_size*2:]        
           
        Lx, Lu, lamb = criterion(logits_x, mixed_target[:batch_size*2], logits_u, mixed_target[batch_size*2:], epoch+bidex/num_iter, warm_up)
        
        # regularization
        prior = torch.ones(args.num_class)/args.num_class
        prior = prior.cuda()        
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior*torch.log(prior/pred_mean))
        if args.kl:
            loss = Lx + lamb * Lu  + penalty + args.co_kl*kl
        else:
            loss = Lx + lamb * Lu + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        '''
        sys.stdout.write('\r')
        
        if args.kl:
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f  klloss: %.3f'
                    %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, bidex+1, num_iter, Lx.item(), Lu.item(), kl))
        else:
            sys.stdout.write(
                '%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                % (args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, bidex + 1, num_iter, Lx.item(),
                   Lu.item()))
        sys.stdout.flush()
        '''
    print("number of labeled: %3d\n" % (counter_for_labeled))
    #print('thisis the holder : %.5f sec' % holder)


    auc_meter = AUCMeter()
    auc_meter.reset()
    auc_meter.add(results_1, clean)
    auc, _, _ = auc_meter.value()

    pred = (results_1 > args.p_threshold)  # 50000
    pred_idx = pred.nonzero()[0]

    clean_number = 0
    for k in pred_idx:
        if noise_label[k] == train_label[k]:
            clean_number += 1

    label_precision = clean_number / len(pred_idx)
    label_recall = clean_number / 30000

    '''
    test_log.write('labeled:%3d\t' % (counter_for_labeled))
    test_log.flush()
    '''
    return auc, label_precision, label_recall


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

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()

def inference(epoch,net1,net2,auc,precision,recall):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))

    '''
    if epoch < warm_up:
        wandb.log({"Test Accuracy": acc})
    else:
        wandb.log({"Test Accuracy": acc, "AUC": auc, "Precision": precision, "Recall": recall})

    #test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    #test_log.flush()
    '''

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
    elif args.network == "Wide":
        model = InceptionResNetV2(num_classes=args.num_class)

    model = model.cuda()
    return model

test_log=open('./checkpoint_v1/%s_%.1f_%s_%s'%(args.dataset,args.r,args.noise_mode,args.index)+'_acc.txt','w')

if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30

loader = dataloader.cifar_dataloader(args.dataset,r=args.r,noise_mode=args.noise_mode,batch_size=args.batch_size,num_workers=5,\
    root_dir=args.data_path,log=test_log,noise_file='%s/%.1f_%s.json'%(args.data_path,args.r,args.noise_mode))

print('| Building net')
net1 = create_model()
#wandb.watch(net1)
net2 = create_model()
#wandb.watch(net2)
cudnn.benchmark = False

criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asym':
    conf_penalty = NegEntropy()

train_loader, clean, noisy_label, train_label = loader.run('train')

time_total = 0

for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          
    test_loader = loader.run('test')
    eval_loader, noise_label, _ = loader.run('eval_train')
    
    if epoch<warm_up:
        #start = time.time()
        warmup_trainloader = loader.run('warmup')
        print('Warmup Net1')
        warmup(epoch,net1,optimizer1,warmup_trainloader)    
        print('\nWarmup Net2')
        warmup(epoch,net2,optimizer2,warmup_trainloader)
        auc = 0
        label_precision = 0
        label_recall = 0
        #end = time.time()
        #running_time = end - start

        #print('time cost : %.5f sec' % running_time)
        #print('done')
    else:

        #start = time.time()
        print('Train Net1')
        auc, label_precision, label_recall = train(epoch,net1,net2,optimizer1, train_loader, args) # train net1

        print('\nTrain Net2')
        _, _, _ = train(epoch,net2,net1,optimizer2, train_loader, args) # train net2

        #end = time.time()
        #running_time = end - start
        #time_total += running_time
        #print('time cost : %.5f sec' % running_time)
        #print('time cost : %.5f sec' % time_total)
        #model_path = '/home/zhuwang/Data/Downloads/pseudo/cifar10/checkpoint/plus_cifar10_80.pt'
        #torch.save(net1.state_dict(), model_path)

    inference(epoch,net1,net2, auc, label_precision, label_recall)


