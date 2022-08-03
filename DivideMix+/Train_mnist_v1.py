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
from dataloader_mnist_v1 import MNIST
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch mnist Training')
parser.add_argument('--batch_size', default=64, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.02, type=float, help='initial learning rate')
parser.add_argument('--noise_mode',  default='symmetric')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--lambda_u', default=25, type=float, help='weight for unsupervised loss')
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--T', default=0.5, type=float, help='sharpening temperature')
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--r', default=0.5, type=float, help='noise ratio')
parser.add_argument('--id', default='')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--dataset', default='mnist', type=str)
parser.add_argument('--index', type=str)

args = parser.parse_args()
#wandb.init(project="mnist_repeated", group="dividemix+", job_type="a%s" % args.r)
torch.cuda.set_device(args.gpuid)

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        #print(val, n)
        if self.count == 0:
            self.avg = 0
        else:
            self.avg = self.sum / self.count


# Training
def train(epoch,net,net2,optimizer,train_loader):

    Loss_x = AverageMeter()
    Loss_u = AverageMeter()

    net.train()
    net2.eval() #fix one network and train the other
    
    num_iter = (len(train_loader.dataset)//args.batch_size)+1
    counter_for_labeled = 0
    for bidex, (input_x, input_x2, labels_x, input_x3) in enumerate(train_loader):
        with torch.no_grad():
            inputs, labels_x = input_x3.cuda(), labels_x.cuda()
            outputs = net2(inputs)
            loss = CE(outputs, labels_x)
        losses = (loss - loss.min()) / (loss.max() - loss.min())  # minibatch
        input_loss = losses.reshape(-1, 1)
        # 加入GMM
        gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        prob = prob[:, gmm.means_.argmin()]
        l_index = []
        u_idex = []
        for i in range(len(prob)):
            if prob[i] >= args.p_threshold:
                l_index.append(i)
                counter_for_labeled += 1
            else:
                u_idex.append(i)
        inputs_x = input_x[l_index]
        inputs_x2 = input_x2[l_index]
        labels_x = labels_x[l_index]
        inputs_u = input_x[u_idex]
        inputs_u2 = input_x2[u_idex]

        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_u.size(0)
        if batch_size == 0 or batch_size_2 == 0:
            print('bad luck')
            continue
        labels_x = labels_x.cpu()
        # Transform label to one-hot
        labels_x = torch.zeros(batch_size, args.num_class).scatter_(1, labels_x.view(-1,1), 1)  # 64,10

        inputs_x, inputs_x2, labels_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda()
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

            # label refinement of labeled sample
            targets_x = labels_x.detach()

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

        loss = Lx + lamb * Lu  + penalty
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        Loss_x.update(Lx, len(input_x))
        Loss_u.update(Lu, len(input_x))
        '''
        if epoch % 100 == 0:
            sys.stdout.write('\r')
            sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t Labeled loss: %.2f  Unlabeled loss: %.2f'
                    %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, bidex+1, num_iter, Lx.item(), Lu.item()))
            sys.stdout.flush()
        '''
    return Loss_x.avg, Loss_u.avg


def warmup(epoch,net,optimizer,dataloader):
    net.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (inputs, _, labels, path) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)               
        loss = CEloss(outputs, labels)      
        if args.noise_mode=='asymmetric':  # penalize confident prediction for asymmetric noise
            penalty = conf_penalty(outputs)
            L = loss + penalty
        elif args.noise_mode=='symmetric':
            L = loss
        L.backward()  
        optimizer.step() 

        sys.stdout.write('\r')
        sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f'
                %(args.dataset, args.r, args.noise_mode, epoch, args.num_epochs, batch_idx+1, num_iter, loss.item()))
        sys.stdout.flush()


def inference(epoch, net1, net2):
    net1.eval()
    net2.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, _, targets, _) in enumerate(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs1 = net1(inputs)
            outputs2 = net2(inputs)           
            outputs = outputs1+outputs2
            _, predicted = torch.max(outputs, 1)            
                       
            total += targets.size(0)
            correct += predicted.eq(targets).cpu().sum().item()                 
    acc = 100.*correct/total
    #print("\n| Test Epoch #%d\t Accuracy: %.2f%%\n" %(epoch,acc))
    #wandb.log({"Test Accuracy": acc})
    test_log.write('Epoch:%d   Accuracy:%.2f\n'%(epoch,acc))
    test_log.flush()  


def eval_train(model,all_loss):
    model.eval()
    losses = torch.zeros(len(train_dataset))
    with torch.no_grad():
        for batch_idx, (inputs, _, targets, index) in enumerate(train_loader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            outputs = model(inputs) 
            loss = CE(outputs, targets)  
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]         
    losses = (losses-losses.min())/(losses.max()-losses.min())    # 50000
    all_loss.append(losses)

    if args.r==0.9: # average loss over last 5 epochs to improve convergence stability
        history = torch.stack(all_loss)
        input_loss = history[-5:].mean(0)
        input_loss = input_loss.reshape(-1,1)
    else:
        input_loss = losses.reshape(-1,1)
    
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]   # 50000
    return prob,all_loss


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


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model():
    if args.dataset == "mnist":
        model = MLPNet()
        model = model.cuda()
    return model


test_log=open('./checkpoint_v1/%s_%.1f_%s_%s'%(args.dataset,args.r,args.noise_mode,args.index)+'_acc.txt','w')

if args.dataset=='mnist':
    warm_up = 5

# input dataset
if args.dataset == 'mnist':
    train_dataset = MNIST(root='./mnist/',
                          download=True,
                          train=True,
                          transform=transforms.ToTensor(),
                          noise_type=args.noise_mode,
                          noise_rate=args.r
                          )

    test_dataset = MNIST(root='./mnist/',
                         download=True,
                         train=False,
                         transform=transforms.ToTensor(),
                         noise_type=args.noise_mode,
                         noise_rate=args.r
                         )
truei = 0
true_labels = train_dataset.train_labels
noise_labels = train_dataset.train_noisy_labels
'''
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
net1 = create_model()
net2 = create_model()
cudnn.benchmark = False
criterion = SemiLoss()
optimizer1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

CE = nn.CrossEntropyLoss(reduction='none')
CEloss = nn.CrossEntropyLoss()
if args.noise_mode=='asymmetric':
    conf_penalty = NegEntropy()

for epoch in range(args.num_epochs+1):   
    lr=args.lr
    if epoch >= 150:
        lr /= 10      
    for param_group in optimizer1.param_groups:
        param_group['lr'] = lr       
    for param_group in optimizer2.param_groups:
        param_group['lr'] = lr          

    if epoch<warm_up:       
        print('Warmup Net1')
        warmup(epoch, net1, optimizer1, train_loader)
        print('\nWarmup Net2')
        warmup(epoch, net2, optimizer2, train_loader)
        print('done')
    else:
        print('Train Net1')

        loss_x, loss_u = train(epoch, net1, net2, optimizer1, train_loader) # train net1

        print('\nTrain Net2')

        _, _ = train(epoch, net2, net1, optimizer2, train_loader) # train net2

    inference(epoch, net1, net2)

