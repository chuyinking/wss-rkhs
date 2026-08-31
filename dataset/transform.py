import numpy as np
from PIL import Image, ImageOps, ImageFilter
import random
import torch
from torchvision import transforms
from util.utils import *


def resize(inputs, ratio_range):
    img = inputs[0]
    outputs = []
    w, h = img.size

    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    for i, input in enumerate(inputs):
        if i == 0 or i == 2:
            input = input.resize((ow, oh), Image.BILINEAR)
        else:
            input = input.resize((ow, oh), Image.NEAREST)
        outputs.append(input)
    return outputs


def crop(inputs, size):
    # padding height or width if smaller than cropping size
    img = inputs[0]
    outputs = []

    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0

    for i, input in enumerate(inputs):
        if i == 0 or i == 2 :
            input = ImageOps.expand(input, border=(0, 0, padw, padh), fill=0)
        else:
            # mask
            input = ImageOps.expand(input, border=(0, 0, padw, padh), fill=255)
        outputs.append(input)

    # cropping
    w, h = outputs[0].size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    new_outputs = []
    for output in outputs:
        new_output = output.crop((x, y, x + size, y + size))
        new_outputs.append(new_output)

    return new_outputs


def hflip(inputs, p=0.5):
    return [stuff.transpose(Image.FLIP_LEFT_RIGHT) for stuff in inputs] if random.random() < p else inputs


def vflip(inputs, p=0.5):
    return [stuff.transpose(Image.FLIP_TOP_BOTTOM) for stuff in inputs] if random.random() < p else inputs


def rotate(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.ROTATE_90)
        mask = mask.transpose(Image.ROTATE_90)
    return img, mask


def to_tensor(img, mask=None, edge=None, crop_size=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor()
    ])(img)
    if crop_size:
        img = transforms.Resize((crop_size//2, crop_size))(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        if crop_size:
            mask = transforms.Resize((crop_size//2, crop_size))(mask.unsqueeze(0)).squeeze()
        if edge is not None:
            edge = transforms.ToTensor()(edge)
            if crop_size:
                edge = transforms.Resize((crop_size // 2, crop_size))(edge)
            return img, mask, edge
        return img, mask
    return img

def normalize(img, mask=None, edge=None, prob=None):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        if edge is not None:
            edge = transforms.ToTensor()(edge)
            return img, mask, edge
        return img, mask
    return img


def normalize_back(img, mask=None, dataset='pascal'):
    means = np.asarray([0.485, 0.456, 0.406])
    stds = np.asarray([0.229, 0.224, 0.225])
    img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = img*stds + means
    img = img*255

    if mask is not None:
        cmap = color_map(dataset)
        mask = mask.squeeze(0).data.cpu().numpy()
        if dataset == 'ade':
            mask[mask > 149] = -1
            mask = mask+1
        mask = Image.fromarray(mask.astype(np.uint8))
        mask.putpalette(cmap.astype(np.uint8))
        return img, mask

    return img


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def random_bright(img, p=0.5):
    if random.random() < p:
        shift_value = 10
        shift = np.random.uniform(-shift_value, shift_value, size=1)
        img = np.array(img).astype(np.float32)
        img[:, :, :] += shift
        img = np.around(img)
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        img = Image.fromarray(img)
    return img


def random_bright2(img, p=0.5, level=0.1):
    '''For tensor input'''
    if random.random() < p:
        img = img + torch.max(img)*(torch.rand(1)-0.5)*level*2
    return torch.clamp(img, 0, 1)


