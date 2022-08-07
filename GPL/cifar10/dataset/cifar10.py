import torchvision as tv
import numpy as np
from PIL import Image
import random
import time
import os
import json

def get_dataset(args, transform_train, transform_val):
    # prepare datasets
    cifar10_train_val = tv.datasets.CIFAR10(args.train_root, train=True, download=False)

    # get train/val dataset
    train_indexes, val_indexes = train_val_split(args, cifar10_train_val.train_labels)
    train = Cifar10Train(args, train_indexes, train=True, transform=transform_train, pslab_transform = transform_val)
    validation = Cifar10Train(args, val_indexes, train=True, transform=transform_val, pslab_transform = transform_val)

    unlabeled_indexes, labeled_indexes, corrupted_labels, clean_labels = train.prepare_data_ssl_warmUp()
    _, _, _, _ = validation.prepare_data_ssl_warmUp()

    return train, unlabeled_indexes, labeled_indexes, validation, corrupted_labels, clean_labels


def train_val_split(args, train_val):
    np.random.seed(args.seed_val)
    train_val = np.array(train_val)
    train_indexes = []
    val_indexes = []
    val_num = int(args.val_samples / args.num_classes)

    for id in range(args.num_classes):
        indexes = np.where(train_val == id)[0]
        np.random.shuffle(indexes)
        val_indexes.extend(indexes[:val_num])
        train_indexes.extend(indexes[val_num:])
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)

    return train_indexes, val_indexes


class Cifar10Train(tv.datasets.CIFAR10):
    def __init__(self, args, train_indexes=None, train=True, transform=None, target_transform=None, pslab_transform=None, download=False):
        super(Cifar10Train, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.args = args
        if train_indexes is not None:
            self.train_data = self.train_data[train_indexes]
            self.train_labels = np.array(self.train_labels)[train_indexes]
        self.soft_labels = np.zeros((len(self.train_labels), 10), dtype=np.float32)

        if len(self.train_labels) == 5000:
            self._num = 0
        else:
            self._num = int(len(self.train_labels) - int(args.labeled_samples)) # 45000-4000
            self.unlabeled_labels = np.zeros(len(self.train_labels), dtype=np.int64) - 1
            self.unlabeled_soft_labels = np.zeros((len(self.train_labels), 10), dtype=np.float32)

        self.original_labels = np.copy(self.train_labels) # 45000
        self.pslab_transform = pslab_transform
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}  # class transition for asymmetric noise

    def prepare_data_ssl_warmUp(self):
        np.random.seed(self.args.seed)

        original_labels = np.copy(self.train_labels)
        unlabeled_indexes = [] # initialize the vector
        train_indexes = []

        num_unlab_samples = self._num
        num_labeled_samples = len(self.train_labels) - num_unlab_samples

        labeled_per_class = int(num_labeled_samples / self.args.num_classes)
        unlab_per_class = int(num_unlab_samples / self.args.num_classes)

        for id in range(self.args.num_classes):
            indexes = np.where(original_labels == id)[0]
            np.random.shuffle(indexes)

            unlabeled_indexes.extend(indexes)
            train_indexes.extend(indexes[unlab_per_class:])

        np.asarray(train_indexes)

        self.train_data = self.train_data[train_indexes]
        self.train_labels = np.array(self.train_labels)[train_indexes]

        ###########inject noise###########

        noise_file = '%s/%.1f_%s.json' % (noise_path, self.args.r, self.args.noise_mode)

        noise_label = np.copy(self.train_labels)
        noise_label = noise_label.tolist()

        if os.path.exists(noise_file) and len(self.train_labels) == 45000:
            noise_label = json.load(open(noise_file, "r"))
            print("using existed noisy labels %s ..." % noise_file)
        else:  # inject noise
            for id in range(self.args.num_classes):
                indexes = np.where(self.train_labels == id)[0]
                num_noise = int(self.args.r * len(self.train_data) / self.args.num_classes)
                noise_idx = indexes[:num_noise]
                for i in indexes:
                    if i in noise_idx:
                        if self.args.noise_mode == 'sym':
                            noiselabel = int(np.int64(random.randint(0, 9)))
                            noise_label[i] = noiselabel
                        elif self.args.noise_mode == 'asym':
                            noiselabel = self.transition[self.train_labels[i]]
                            noise_label[i] = noiselabel
                    else:
                        noise_label[i] = int(self.train_labels[i])

            if len(self.train_labels) == 45000:
                print("save noisy labels to %s ..." % noise_file)
                json.dump(noise_label, open(noise_file, "w"))

        ###########inject noise###########

        test_as = np.copy(self.train_labels)

        for j in range(len(noise_label)):
            self.train_labels[j] = noise_label[j]


        count = 0
        for k in range(len(self.train_labels)):
            if test_as[k] == self.train_labels[k]:
                count += 1
        percent1 = count / len(self.train_labels)

        self.soft_labels = np.zeros((len(self.train_labels), self.args.num_classes), dtype=np.float32)

        for i in range(len(self.train_data)):
            self.soft_labels[i][self.train_labels[i]] = 1

        unlabeled_indexes = np.asarray(unlabeled_indexes)

        return np.asarray(unlabeled_indexes), np.asarray(train_indexes), noise_label, test_as

    def update_labels(self, result, unlabeled_indexes):
        relabel_indexes = list(unlabeled_indexes)

        self.unlabeled_soft_labels[relabel_indexes] = result[relabel_indexes]
        self.unlabeled_labels[relabel_indexes] = self.unlabeled_soft_labels[relabel_indexes].argmax(axis = 1).astype(np.int64)

    def __getitem__(self, index):
        if len(self.train_labels) == 5000:
            img, labels, soft_labels = self.train_data[index], self.train_labels[index], self.soft_labels[index]
        else:
            img, labels, soft_labels, unlabeled_labels, unlabeled__soft_labels = self.train_data[index], \
                                                                                 self.train_labels[index], \
                                                                                 self.soft_labels[index], \
                                                                                 self.unlabeled_labels[index], \
                                                                                 self.unlabeled_soft_labels[index]
        img_base = Image.fromarray(img)

        if self.args.DApseudolab == "False":
            img_pseudolabels = self.pslab_transform(img) # test transform
        else:
            img_pseudolabels = 0

        if self.transform is not None:
            img = self.transform(img_base)
            img1 = self.transform(img_base)
            img2 = self.transform(img_base)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        # 这里是输出
        if len(self.train_labels) == 5000: # validation
            return img, img_pseudolabels, labels, soft_labels, index
        else:
            return img, img_pseudolabels, labels, soft_labels, index, unlabeled_labels, unlabeled__soft_labels, img1, img2
