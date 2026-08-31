import numpy as np
import matplotlib.pyplot as plt
from dataset.transform import *
import math
import os
from PIL import Image
#import nibabel as nib
import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from util.utils import *
import torch
from torch.utils.data import DataLoader
from model.tools import *


class VocDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 21  # voc

        self.img_path = root + '/JPEGImages/'
        self.true_mask_path = root + '/SegmentationClass/'
        self.edge_path = root + '/CannyEdge/'

        if mode == 'val':
            self.label_path = self.true_mask_path
            id_path = 'dataset/splits/%s/val.txt' % name
            #id_path = 'dataset/splits/%s/plain_BG2.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()


        else:
            id_path = 'dataset/splits/%s/train.txt' % name
            #id_path = 'dataset/splits/%s/plain_BG2.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

            if mode == 'full':
                print('loading full masks')
                self.label_path = self.true_mask_path
            elif mode == 'point':
                self.label_path = root + '/SegmentationPoint'
            elif mode == 'scribble':
                self.label_path = root + '/SegmentationScribble'

            else:
                self.label_path = root + '/SegmentationScribble'

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))
        edge = Image.open(os.path.join(self.edge_path, id.split(' ')[1]))

        if len(np.array(mask).shape) == 3:
            mask, _, _ = mask.split()

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img, mask, edge = resize([img, mask, edge], (0.5, 2.0))
        img, mask, edge = crop([img, mask, edge], self.size)
        img, mask, edge = hflip([img, mask, edge], p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img, mask, edge = normalize(img, mask, edge) # to tensor and normalize
        #img, mask, edge = to_tensor(img, mask, edge)
        cls_label = self.get_cls_label(cls_label)
        return img, mask, edge, cls_label, id

    def __len__(self):
        return len(self.ids)


class ACDCDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=False):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root #../../Datasets/ACDC
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 4

        if not self.mode == 'val':
            self.img_path = root + '/training/JPEGImage/'
            self.label_path = root + '/training/Mask/'
            self.edge_path = root + '/training/CannyEdge/'
            self.ids = os.listdir(self.img_path)
        else:
            self.img_path = root + '/testing/JPEGImage/'
            self.label_path = root + '/testing/Mask/'
            self.ids = os.listdir(self.img_path)



    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label



    def __getitem__(self, item):
        img = Image.open(os.path.join(self.img_path, self.ids[item]))
        mask = Image.open(os.path.join(self.label_path, self.ids[item][:-4]+'.png'))


        if len(np.array(mask).shape) == 3:
            mask, _, _ = mask.split()

        if self.mode == 'val':
            img, mask = to_tensor(img, mask)
            return img, mask, self.ids[item][:-4]

        cls_label = np.unique(np.asarray(mask))
        edge = Image.open(os.path.join(self.edge_path, self.ids[item][:-4] + '.png'))

        # basic augmentation on all training images
        img, mask, edge = resize([img, mask, edge], (0.5, 2.0))
        img, mask, edge = crop([img, mask, edge], self.size)

        # img, mask, edge = hflip([img, mask, edge], p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img, mask, edge = to_tensor(img, mask, edge) #to tensor and normalize
        cls_label = self.get_cls_label(cls_label)
        return img, mask, edge, cls_label, self.ids[item][:-4]

    def __len__(self):
        return len(self.ids)


class ACDCRkhsDataset(Dataset):
    def __init__(self, name, root, mode, size, n_class, aug=True):
        """
        :param name: dataset name
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 4


        if self.mode == 'val':
            # self.img_path = root + '/testing/JPEGImage/'
            # self.label_path = root + '/testing/Mask/'
            self.img_path = root + '/training/JPEG2/'
            self.label_path = root + '/training/FullMask/'
            self.ids = sorted(os.listdir(self.img_path))
        else:
            self.img_path = root + '/training/JPEG2/'
            self.label_path = root + '/training/FullMask/'
            self.prob_path = root + '/trainingCropScrib/L2RK_prob/'
            self.ids = sorted(os.listdir(self.img_path))
            with open('dataset/classes/%s.txt' % name, 'r') as f:
                self.class_indexes = f.read().splitlines()
            self.n_class = n_class

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def transform(self, image, mask, prob, item):

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()

        # Reconstruct the prob to full version (from few class to all class)
        _, h, w = image.size()
        prob_full = torch.zeros(self.n_class, h, w)
        class_indexes = self.class_indexes[item].split(' ')
        for pk, k in enumerate(class_indexes):
            prob_full[int(k)] = prob[pk]
        prob = prob_full
        del prob_full

        # Remove boundary pixel
        # br = 5
        # image = image[:, br:-br, br:-br]
        # mask = mask[br:-br, br:-br]
        # edge = edge[:, br:-br, br:-br]
        # prob = prob[br:-br, br:-br]

        # Resize
        image = TF.resize(image, self.size)
        mask = TF.resize(mask.unsqueeze(0), self.size, transforms.InterpolationMode.NEAREST).squeeze()
        prob = TF.resize(prob, self.size, transforms.InterpolationMode.NEAREST)

        # Random flip and rotate
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            prob = TF.hflip(prob)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            prob = TF.vflip(prob)

        if random.random() > 0.5:
            image = image.transpose(1, 2)
            mask = mask.transpose(0, 1)
            prob = prob.transpose(1, 2)

        if self.aug and random.random() > 0.5:
            # Resize
            # image = TF.resize(image, [2 * h, 2 * w])
            # mask = TF.resize(mask.unsqueeze(0), [2 * h, 2 * w])
            # prob = TF.resize(prob, [2 * h, 2 * w])
            mask = mask.unsqueeze(0)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, self.size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w).squeeze()
            prob = TF.crop(prob, i, j, h, w)

        # Normalize
        #image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        image = transforms.Normalize(torch.mean(image, dim=[1, 2]), torch.std(image, dim=[1, 2]))(image)

        return image, mask, prob

    def __getitem__(self, item):

        img = Image.open(os.path.join(self.img_path, self.ids[item]))
        mask = Image.open(os.path.join(self.label_path, self.ids[item][:-4]+'.png'))
        if len(np.array(mask).shape) == 3:
            mask, _, _ = mask.split()

        if self.mode == 'val':
            img = TF.to_tensor(img)
            mask = torch.from_numpy(np.array(mask)).long()
            img = TF.resize(img, self.size)
            mask = TF.resize(mask.unsqueeze(0), self.size, transforms.InterpolationMode.NEAREST).squeeze()
            img = transforms.Normalize(torch.mean(img, dim=[1, 2]), torch.std(img, dim=[1, 2]))(img)
            return img, mask, 0, self.ids[item][:-4]

        cls_label = np.unique(np.asarray(mask))
        prob = torch.load(os.path.join(self.prob_path, self.ids[item][:-4] + '.pt'), map_location='cpu').float()

        # basic augmentation on all training images
        img, mask, prob = self.transform(img, mask, prob, item)
        #if len(cls_label) == 2 and max(mask) == 255: mask = (mask / 255).long()
        cls_label = self.get_cls_label(cls_label)
        return img, mask, 1, prob, cls_label, self.ids[item][:-4]

    def __len__(self):
        return len(self.ids)


class VocRkhsDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 21  # voc

        self.img_path = root + '/JPEGImages/'
        #self.true_mask_path = root + '/SegmentationClass/'
        self.true_mask_path = root + '/L2RK/'
        self.prob_path = root + '/L2RK_prob/'
        self.edge_path = root + '/CannyEdge/'

        if mode == 'val':
            self.label_path = self.true_mask_path
            id_path = 'dataset/splits/%s/sal20.txt' % name
            #id_path = 'dataset/splits/%s/plain_BG2.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

        elif mode == 'run':
            id_path = 'dataset/splits/%s/sal5.txt' % name
            #id_path = 'dataset/splits/%s/plain_BG2.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

        else:
            id_path = 'dataset/splits/%s/sal900.txt' % name
            #id_path = 'dataset/splits/%s/plain_BG2.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

            if mode == 'full':
                print('loading full masks')
                self.label_path = self.true_mask_path
            elif mode == 'point':
                self.label_path = root + '/SegmentationPoint'
            elif mode == 'scribble':
                self.label_path = root + '/SegmentationScribble'
            elif mode == 'thre':
                self.label_path = root + '/L2RK/'
            else:
                self.label_path = root + '/SegmentationScribble'

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def transform(self, image, mask, edge, prob):

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        edge = TF.to_tensor(edge)

        # Remove boundary pixel
        # br = 5
        # image = image[:, br:-br, br:-br]
        # mask = mask[br:-br, br:-br]
        # edge = edge[:, br:-br, br:-br]
        # prob = prob[br:-br, br:-br]


        # Random flip and rotate
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            edge = TF.hflip(edge)
            prob = TF.hflip(prob)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            edge = TF.vflip(edge)
            prob = TF.vflip(prob)

        if random.random() > 0.5:
            image = image.transpose(1, 2)
            mask = mask.transpose(0, 1)
            edge = edge.transpose(1, 2)
            prob = prob.transpose(0, 1)

        if self.aug and random.random() > 0.5:
            # Resize
            # _, h, w = image.size()
            # image = TF.resize(image, [2 * h, 2 * w])
            # mask = TF.resize(mask.unsqueeze(0), [2 * h, 2 * w])
            # edge = TF.resize(edge, [2 * h, 2 * w])
            # prob = TF.resize(prob.unsqueeze(0), [2 * h, 2 * w])
            mask = mask.unsqueeze(0)
            prob = prob.unsqueeze(0)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, self.size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w).squeeze()
            edge = TF.crop(edge, i, j, h, w)
            prob = TF.crop(prob, i, j, h, w).squeeze()

        # Normalize
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        # edge = (edge-torch.mean(edge))/torch.std(edge)
        # edge = (edge-torch.min(edge))/(torch.max(edge)-torch.min(edge))

        return image, mask, edge, prob

    def __getitem__(self, item):
        id = self.ids[item]

        # if self.mode == 'run':
        #     img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        #     return normalize(img), 0, id
        if self.mode == 'run':
            img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
            prob = torch.load(os.path.join(self.prob_path, id.split(' ')[2]), map_location='cpu').float()
            return normalize(img), 0, prob, id

        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        mask = Image.open(os.path.join(self.label_path, id.split(' ')[1][:-4]+'_thre.png'))
        edge = Image.open(os.path.join(self.edge_path, id.split(' ')[1]))
        prob = torch.load(os.path.join(self.prob_path, id.split(' ')[2]), map_location='cpu').float()

        if len(np.array(mask).shape) == 3:
            mask, _, _ = mask.split()

        cls_label = np.unique(np.asarray(mask))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            #img, mask, _, _ = self.transform(img, mask, edge, prob)
            return img, mask, prob, id



        # basic augmentation on all training images
        img, mask, edge, prob = self.transform(img, mask, edge, prob)
        #if len(cls_label) == 2 and max(mask) == 255: mask = (mask / 255).long()
        cls_label = self.get_cls_label(cls_label)
        return img, mask, edge, prob, cls_label, id

    def __len__(self):
        return len(self.ids)

class HAMDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 21  # voc

        self.img_path = root + '/images/'
        self.true_mask_path = root + '/masks/'
        self.edge_path = root + '/CannyEdge/'

        if mode == 'val':
            self.label_path = self.true_mask_path
            id_path = 'dataset/splits/%s/val.txt' % name
            #id_path = 'dataset/splits/%s/train_s.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

        else:
            id_path = 'dataset/splits/%s/train.txt' % name
            #id_path = 'dataset/splits/%s/train_s.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

            if mode == 'full':
                print('loading full masks')
                self.label_path = self.true_mask_path
            elif mode == 'point':
                self.label_path = root + '/points'
            elif mode == 'scribble':
                self.label_path = root + '/SegmentationScribble'

            else:
                self.label_path = root + '/SegmentationScribble'

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))
        edge = Image.open(os.path.join(self.edge_path, id.split(' ')[2]))

        if len(np.array(mask).shape) == 3:
            mask, _, _ = mask.split()

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img, mask, edge = resize([img, mask, edge], (0.5, 2.0))
        img, mask, edge = crop([img, mask, edge], self.size)
        img, mask, edge = hflip([img, mask, edge], p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img, mask, edge = normalize(img, mask, edge) # to tensor and normalize
        cls_label = self.get_cls_label(cls_label)
        return img, mask, edge, cls_label, id

    def __len__(self):
        return len(self.ids)

class ECSSDRkhsDataset(Dataset):
    def __init__(self, name, root, mode, size, n_class, aug=True):
        """
        :param name: dataset name
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug


        if self.mode == 'val':
            self.img_path = root + '/images'
            self.label_path = root + '/ground_truth_mask'
            self.prob_path = root + '/prob'
            self.ids = sorted(os.listdir(self.img_path))[:200]
            self.n_class = n_class
        else:
            self.img_path = root + '/images'
            self.label_path = root + '/ground_truth_mask' #'/scribbles' #if gt is used, need to divide by 255 before feed
            self.prob_path = root + '/prob'
            self.ids = sorted(os.listdir(self.img_path))[200:]
            with open('dataset/classes/%s.txt' % name, 'r') as f:
                self.class_indexes = f.read().splitlines()
            self.n_class = n_class


    def transform(self, image, mask, prob, item):

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)/255).long() #gt used, /255

        _, h, w = image.size()

        # Reconstruct the prob to full version (from few class to all class)
        if self.n_class > 2:
            prob_full = torch.zeros(self.n_class, h, w)
            class_indexes = self.class_indexes[item].split(' ')
            for pk, k in enumerate(class_indexes):
                prob_full[int(k)] = prob[pk]
            prob = prob_full
            del prob_full

        # (Normalize prob)
        # prob_p = prob[prob >= 0.5]
        # prob[prob >= 0.5] = (1+(prob_p-torch.min(prob_p))/(torch.max(prob_p)-torch.min(prob_p)))*0.5
        # prob_p = 0.5 - prob[prob < 0.5]
        # prob[prob < 0.5] = (1-(prob_p-torch.min(prob_p))/(torch.max(prob_p)-torch.min(prob_p)))*0.5

        # Remove boundary pixel
        # br = 5
        # image = image[:, br:-br, br:-br]
        # mask = mask[br:-br, br:-br]
        # edge = edge[:, br:-br, br:-br]
        # prob = prob[br:-br, br:-br]

        # Resize
        # image = TF.resize(image, self.size)
        # mask = TF.resize(mask.unsqueeze(0), self.size, transforms.InterpolationMode.NEAREST).squeeze()
        # prob = TF.resize(prob, self.size, transforms.InterpolationMode.NEAREST)

        # Random flip and rotate
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            prob = TF.hflip(prob)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            prob = TF.vflip(prob)

        if random.random() > 0.5:
            image = image.transpose(1, 2)
            mask = mask.transpose(0, 1)
            prob = prob.transpose(1, 2) if self.n_class > 2 else prob.transpose(0, 1)

        if self.aug:
            # Resize
            # image = TF.resize(image, [2 * h, 2 * w])
            # mask = TF.resize(mask.unsqueeze(0), [2 * h, 2 * w])
            # prob = TF.resize(prob, [2 * h, 2 * w]) if self.n_class > 2 else TF.resize(prob.unsqueeze(0), [2 * h, 2 * w])
            mask = mask.unsqueeze(0)
            if self.n_class == 2: prob = prob.unsqueeze(0)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, self.size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w).squeeze()
            prob = TF.crop(prob, i, j, h, w)
            if self.n_class == 2: prob = prob.squeeze()

            # randomly color compress, turns to gray, blur and add noise
            strong_aug_prob = 0.5
            if random.random() < strong_aug_prob:
                image = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image)
            image = transforms.RandomGrayscale(p=0.1)(image)
            if random.random() < strong_aug_prob:
                image = GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))(image)
            image = random_bright2(image, p=strong_aug_prob)

        # Normalize
        #image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        #image = transforms.Normalize(torch.mean(image, dim=[1, 2]), torch.std(image, dim=[1, 2]))(image)

        return image, mask, prob

    def __getitem__(self, item):

        img = Image.open(os.path.join(self.img_path, self.ids[item])).convert('RGB')
        mask = Image.open(os.path.join(self.label_path, self.ids[item][:-4]+'.png'))
        if len(np.array(mask).shape) == 3:
            mask, _, _ = mask.split()

        if self.mode == 'val':
            img = TF.to_tensor(img)
            mask = torch.from_numpy(np.array(mask)/255).long()
            #img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            #img = transforms.Normalize(torch.mean(img, dim=[1, 2]), torch.std(img, dim=[1, 2]))(img)
            #prob = torch.load(os.path.join(self.prob_path, self.ids[item][:-4] + '.pt'), map_location='cpu').float()
            return img, mask, 0, self.ids[item][:-4]


        prob = torch.load(os.path.join(self.prob_path, self.ids[item][:-4] + '.pt'), map_location='cpu').float()

        # basic augmentation on all training images
        img, mask, prob = self.transform(img, mask, prob, item)

        return img, mask, 1, prob, 1, self.ids[item][:-4] #img, mask, edge, prob, cls_label, ids

    def __len__(self):
        return len(self.ids)

# class ACDCinteractDataset(Dataset):
#     def __init__(self, name, root, mode, size, aug=False):
#         """
#         :param name: dataset name, pascal or cityscapes
#         :param root: root path of the dataset.
#         :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
#                      label: pseudo labeling the remaining unlabeled images.
#                      semi_train: semi-supervised learning with both labeled and unlabeled images.
#                      val: validation.
#
#         :param size: crop size of training images.
#         """
#         self.name = name
#         self.root = root #../../Datasets/ACDC
#         self.mode = mode
#         self.size = size
#         self.aug = aug
#         self.ignore_class = 4  # voc
#
#         if not self.mode == 'val':
#             self.img_path = root + '/tz/interact/JPEGImageAllslice/'
#             self.label_path = root + '/tz/interact/Mask/'
#             self.edge_path = root + '/tz/interact/CannyEdge/'
#             self.ids = os.listdir(self.img_path)
#         else:
#             self.img_path = root + '/tz/interact/JPEGImageAllslice/'
#             self.label_path = root + '/tz/interact/FullMask/'
#             self.ids = os.listdir(self.img_path)
#
#         self.make_data()
#
#     def make_data(self):
#         for img_file in self.ids:
#             mask_file = img_file[:-4] + '.png'
#             if mask_file not in os.listdir(self.label_path):
#                 im_size = Image.open(os.path.join(self.img_path, img_file)).size
#                 mask = Image.new("RGB", im_size, (4, 4, 4))
#                 mask.save(os.path.join(self.label_path, mask_file), "PNG")
#
#
#     def get_cls_label(self, cls_label):
#         cls_label_set = list(cls_label)
#
#         if self.ignore_class in cls_label_set:
#             cls_label_set.remove(self.ignore_class)
#         if 255 in cls_label_set:
#             cls_label_set.remove(255)
#
#         cls_label = np.zeros(self.ignore_class)
#         for i in cls_label_set:
#             cls_label[i] += 1
#         cls_label = torch.from_numpy(cls_label).float()
#         return cls_label
#
#
#
#     def __getitem__(self, item):
#         img = Image.open(os.path.join(self.img_path, self.ids[item]))
#         mask = Image.open(os.path.join(self.label_path, self.ids[item][:-4]+'.png'))
#
#
#         if len(np.array(mask).shape) == 3:
#             mask, _, _ = mask.split()
#
#         if self.mode == 'val':
#             img, mask = to_tensor(img, mask)
#             return img, mask, self.ids[item][:-4]
#
#         cls_label = np.unique(np.asarray(mask))
#         edge = Image.open(os.path.join(self.edge_path, self.ids[item][:-4] + '.png'))
#
#         img, mask, edge = to_tensor(img, mask, edge) #to tensor and normalize
#         cls_label = self.get_cls_label(cls_label)
#
#         return img, mask, edge, cls_label, self.ids[item][:-4]
#
#     def __len__(self):
#         return len(self.ids)

class VocCompareDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 21  # voc

        self.img_path = root + '/JPEGImages/'
        #self.true_mask_path = root + '/SegmentationClass/'
        self.true_mask_path = root + '/L2RK/'
        self.prob_path = root + '/L2RK_prob/'
        self.edge_path = root + '/CannyEdge/'

        if mode == 'val':
            self.label_path = self.true_mask_path
            id_path = 'dataset/splits/%s/sal20.txt' % name
            #id_path = 'dataset/splits/%s/plain_BG2.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

        elif mode == 'run':
            id_path = 'dataset/splits/%s/sal500.txt' % name
            #id_path = 'dataset/splits/%s/plain_BG2.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

        else:
            id_path = 'dataset/splits/%s/sal500.txt' % name
            #id_path = 'dataset/splits/%s/plain_BG2.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

            if mode == 'full':
                print('loading full masks')
                self.label_path = self.true_mask_path
            elif mode == 'point':
                self.label_path = root + '/SegmentationPoint'
            elif mode == 'scribble':
                self.label_path = root + '/SegmentationScribble'
            elif mode == 'thre':
                self.label_path = root + '/L2RK/'
            else:
                self.label_path = root + '/SegmentationScribble'

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def transform(self, image, mask, edge, prob):

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        edge = TF.to_tensor(edge)

        # Remove boundary pixel
        # br = 5
        # image = image[:, br:-br, br:-br]
        # mask = mask[br:-br, br:-br]
        # edge = edge[:, br:-br, br:-br]
        # prob = prob[br:-br, br:-br]


        # Random flip and rotate
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            edge = TF.hflip(edge)
            prob = TF.hflip(prob)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            edge = TF.vflip(edge)
            prob = TF.vflip(prob)

        if random.random() > 0.5:
            image = image.transpose(1, 2)
            mask = mask.transpose(0, 1)
            edge = edge.transpose(1, 2)
            prob = prob.transpose(0, 1)

        if self.aug and random.random() > 0.5:
            # Resize
            # _, h, w = image.size()
            # image = TF.resize(image, [2 * h, 2 * w])
            # mask = TF.resize(mask.unsqueeze(0), [2 * h, 2 * w])
            # edge = TF.resize(edge, [2 * h, 2 * w])
            # prob = TF.resize(prob.unsqueeze(0), [2 * h, 2 * w])
            mask = mask.unsqueeze(0)
            prob = prob.unsqueeze(0)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, self.size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w).squeeze()
            edge = TF.crop(edge, i, j, h, w)
            prob = TF.crop(prob, i, j, h, w).squeeze()

        # Normalize
        image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
        # edge = (edge-torch.mean(edge))/torch.std(edge)
        # edge = (edge-torch.min(edge))/(torch.max(edge)-torch.min(edge))

        return image, mask, edge, prob

    def __getitem__(self, item):
        id = self.ids[item]

        # if self.mode == 'run':
        #     img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        #     return normalize(img), 0, id
        if self.mode == 'run':
            img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
            prob = torch.load(os.path.join(self.prob_path, id.split(' ')[2]), map_location='cpu').float()
            return normalize(img), 0, prob, id

        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        if self.mode == 'val' or self.mode == 'thre':
            mask = Image.open(os.path.join(self.label_path, id.split(' ')[1][:-4] + '_thre.png'))
        else:
            mask = Image.open(os.path.join(self.label_path, id.split(' ')[1][:-4]+'.png'))
        edge = Image.open(os.path.join(self.edge_path, id.split(' ')[1]))
        prob = torch.load(os.path.join(self.prob_path, id.split(' ')[2]), map_location='cpu').float()

        if len(np.array(mask).shape) == 3:
            mask, _, _ = mask.split()

        cls_label = np.unique(np.asarray(mask))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            #img, mask, _, _ = self.transform(img, mask, edge, prob)
            return img, mask, prob, id



        # basic augmentation on all training images
        img, mask, edge, prob = self.transform(img, mask, edge, prob)
        mask[mask == cls_label[1]] = 1
        #if len(cls_label) == 2 and max(mask) == 255: mask = (mask / 255).long()
        cls_label = self.get_cls_label(cls_label)
        # edge = (edge-torch.mean(edge))/torch.std(edge)
        # edge = (edge-torch.min(edge))/(torch.max(edge)-torch.min(edge))
        return img, mask, edge, prob, cls_label, id

    def __len__(self):
        return len(self.ids)

