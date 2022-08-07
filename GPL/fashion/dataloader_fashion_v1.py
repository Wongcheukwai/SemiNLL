import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import codecs
import torchvision.transforms as transforms
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import argparse
import json
import random

class fashion(data.Dataset):
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train, test, args, transform=None, target_transform=None, download=True, noise_file='', index=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.test = test
        self.transition = {0: 0, 1: 1, 2: 6, 3: 4, 4: 3, 5: 5, 6: 6, 7: 5, 8: 8, 9: 7}  # class transition for asymmetric noise
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        train_data, train_labels = torch.load(
                os.path.join(root, self.processed_folder, self.training_file))

        test_data, test_labels = torch.load(os.path.join(root, self.processed_folder, self.test_file))

        self.train_data = train_data[0:60000] # 60000, 28, 28
        self.test_data = test_data # 10000, 28, 28
        self.train_label = train_labels[0:60000] # 60000 乱序
        self.test_label = test_labels #10000 乱序

        if self.train:
            if os.path.exists(noise_file):
                noise_label = json.load(open(noise_file, "r"))
                print('load data')
            else:  # inject noise
                noise_label = []
                idx = list(range(len(self.train_label)))
                random.shuffle(idx) # idx 乱序
                num_noise = int(args.r * len(self.train_label))
                noise_idx = idx[:num_noise]
                for i in range(len(self.train_label)):
                    if i in noise_idx:
                        if args.noise_mode == 'sym':
                            noiselabel = random.randint(0, 9)
                            noise_label.append(noiselabel)
                        elif args.noise_mode == 'asym':
                            noiselabel = self.transition[int(self.train_label[i])]
                            noise_label.append(noiselabel)
                    else:
                        noise_label.append(int(self.train_label[i]))
                print("save noisy labels to %s ..." % noise_file)
                json.dump(noise_label, open(noise_file, "w"))
            clean = 0
            for k in range(len(noise_label)):
                if noise_label[k] == self.train_label[k]:
                    clean += 1
            per1 = clean/len(noise_label)
            self.noise_label = noise_label

            # 针对pesudolabelling
            self.soft_labels = np.zeros((len(self.train_label), 10), dtype=np.float32)
            for i in range(len(self.train_label)):
                self.soft_labels[i][self.noise_label[i]] = 1

            self.unlabeled_labels = np.zeros(len(self.train_label), dtype=np.int64) - 1
            self.unlabeled_soft_labels = np.zeros((len(self.train_label), 10), dtype=np.float32)


    def __getitem__(self, index):
        if self.train:
            #img, target = self.train_data[index], self.noise_label[index]
            img, target, soft_labels, unlabeled_soft_labels, unlabeled_labels = self.train_data[index], \
                                                                                self.noise_label[index], \
                                                                                self.soft_labels[index], \
                                                                                self.unlabeled_soft_labels[index], \
                                                                                self.unlabeled_labels[index]
        else:
            img, target = self.test_data[index], self.test_label[index]

        img = Image.fromarray(img.numpy(), mode='L')
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))])

        if self.transform is not None:
            img1 = self.transform(img)
            img2 = transform_test(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img1, img2, target, soft_labels, index, unlabeled_labels, unlabeled_soft_labels
        else:
            return img1, target

    def update_labels(self, result):
        self.unlabeled_soft_labels = result
        self.unlabeled_labels = self.unlabeled_soft_labels.argmax(axis = 1).astype(np.int64)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        if self._check_exists():
            return
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)

