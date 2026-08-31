#import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from kornia.filters import bilateral_blur
import numpy as np
from util.utils import *

import matplotlib.pyplot as plt
from PIL import Image


def build_cur_cls_label(mask, nclass):
    """some point annotations are cropped out, thus the prototypes are partial"""
    b = mask.size()[0]
    mask_one_hot = one_hot(mask, nclass)
    cur_cls_label = mask_one_hot.view(b, nclass, -1).max(-1)[0]
    return cur_cls_label.view(b, nclass, 1, 1)


def clean_mask(mask, cls_label, softmax=True):
    if softmax:
        mask = F.softmax(mask, dim=1)
    n, c = cls_label.size()
    """Remove any masks of labels that are not present"""
    return mask * cls_label.view(n, c, 1, 1)

def CEloss(preds, label, ignore_indexes, reduction='mean'):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    n_class = label.max()
    criterion = nn.BCELoss(reduction=reduction)
    ce_loss = torch.tensor([0.0], device=label.device)

    preds = F.softmax(preds, 1)

    count = 0
    for k in range(n_class):
        if (label==k).sum() and k not in ignore_indexes:
            count += 1
            tmp_ind = 0 if k==0 else 1
            ce_loss += criterion(preds[:,tmp_ind,:,:][label==k], (label[label==k]+1)/(k+1)) #criterion(preds[:,k,:,:][label==k], (label[label==k]+1)/(k+1))
    loss = ce_loss/count if bool(count) else ce_loss*count

    return loss

def fidelityloss(preds, prob, fk_form='1-2p', n_class=2):
    """
    the unsqueezed dimension is for number of class
    """
    err = 1e-6
    if n_class < 3:
        preds = torch.sigmoid(preds)
        f = torch.unsqueeze(1-2*prob, 1) if fk_form == '1-2p' else torch.unsqueeze(-torch.log(err+prob/(1-prob+err)), 1)
    else:
        preds = torch.softmax(preds, 1)
        f = 1-2*prob if fk_form == '1-2p' else -torch.log(err+prob/(1-prob+err))

    return torch.mean(preds*f)

def invariantCircle(tmp, filter_mode='Gaussian', kernel_size=3, invar=True):
    tmp = F.softmax(tmp, dim=1)[:,1:,:,:]
    predict = tmp.clone()
    predict[:,1,:,:] = tmp[:,0,:,:]+tmp[:,1,:,:]
    if filter_mode == 'Bilateral':
        filt = bilateral_blur((1 - predict), kernel_size=kernel_size, sigma_color=0.04,
                              sigma_space=(.6, .6))  # Default: sigma_color=12., sigma_space=(60., 60.)
        kcut = filt * predict
    else:
        SmoothFilter = GaussianBlur(kernel_size=kernel_size, sigma=(0.01, 0.05))
        kcut = SmoothFilter(1 - predict) * predict
    length = kcut.sum(dim=[2, 3])
    area = predict.sum(dim=[2, 3])
    Ncut = length ** 2 / area / 1000 if invar else length / area
    return Ncut.mean()

def invariantNcut(predict, filter_mode='Gaussian', kernel_size=3, invar=True):
    predict = F.softmax(predict, dim=1)#[:,1:,:,:]
    if filter_mode == 'Bilateral':
        filt = bilateral_blur((1-predict), kernel_size=kernel_size, sigma_color=0.1
                              , sigma_space=(.2, .2)) #Default: sigma_color=12., sigma_space=(60., 60.)
        kcut = filt * predict
    else:
        SmoothFilter = GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 0.1))
        kcut = SmoothFilter(1-predict)*predict

    length = kcut.sum(dim=[2, 3])
    area = predict.sum(dim=[2, 3])
    Ncut = length**2/area/1000 if invar else length/area


    return Ncut.mean()

def Ncut(predict, filter_mode='Gaussian', kernel_size=3):
    predict = torch.sigmoid(predict)
    if filter_mode == 'Bilateral':
        filt_inv = bilateral_blur((1-predict), kernel_size=kernel_size, sigma_color=0.1
                              , sigma_space=(.2, .2)) #sigma_color=0.1, sigma_space=(0.2, 0.2)
        kcut = filt_inv * predict
        filt_predict = bilateral_blur(predict, kernel_size=kernel_size, sigma_color=0.1
                              , sigma_space=(.2, .2))

    else:
        SmoothFilter = GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 0.1))
        kcut = SmoothFilter(1-predict)*predict
        filt_predict = SmoothFilter(predict)

    length = kcut.sum(dim=[2, 3])
    area = filt_predict.sum(dim=[2, 3])
    Ncut = length/area


    return Ncut.mean()

def weightEdge(preds, edge, beta, kernel_size=3, filter_mode='Gaussian', weight_mode='classic'):
    #predict = F.softmax(preds, dim=1)#[:, 1:, :, :]
    predict = torch.sigmoid(preds)

    if filter_mode == 'Bilateral':
        filt_inv = bilateral_blur((1-predict), kernel_size==kernel_size, sigma_color=0.1
                              , sigma_space=(.2, .2))
        kcut = filt_inv * predict
    else:
        SmoothFilter = GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 0.1))
        kcut = SmoothFilter(1 - predict) * predict


    if weight_mode == 'sigmoid':
        tur = 0.3
        weight_length = (kcut*torch.sigmoid(-beta * (edge - tur))).sum(dim=[2, 3])
    elif weight_mode == 'classic':
        weight_length = (kcut / (1 + beta * edge)).sum(dim=[2, 3])
    else:
        weight_length = kcut.sum(dim=[2, 3])

    # area = predict.sum(dim=[2, 3])
    # penalty_value = weight_length**2/area
    penalty_value = weight_length

    return penalty_value.mean()

def MS(img, pred):
    loss = 0.0
    preds = torch.sigmoid(pred)
    for predict in [preds, 1-preds]:
        area = predict.sum(dim=[2, 3])
        area[area == 0] = 1
        c = (img * predict).sum(dim=[2, 3]) / area  # c:(bs, img_channel)
        c = torch.unsqueeze(torch.unsqueeze(c, 2), 3)
        loss += (predict*(img-c)**2).sum(dim=1).mean()
    return loss

def TV(pred):
    predict = torch.sigmoid(pred)
    h_diff = F.l1_loss(predict[:, :, 1:, :], predict[:, :, :-1, :])
    w_diff = F.l1_loss(predict[:, :, :, 1:], predict[:, :, :, :-1])
    return (h_diff+w_diff)/2

def PierceWise(img, pred):
    loss = 0.0
    for k in [2]: #range(pred.size()[1]):
        predict = F.softmax(pred, dim=1)[:, (0+k):(1+k), :, :]
        area = predict.sum(dim=[2, 3])
        area[area == 0] = 1
        c = (img * predict).sum(dim=[2, 3]) / area  # c:(bs, img_channel)
        c = torch.unsqueeze(torch.unsqueeze(c, 2), 3)
        loss += (predict*(img-c)**2).sum(dim=[2,3]).mean()
    return loss / 100

def compactness(predict, kernel_size=3, invar=True):
    def cal_compactness(pred, kernel_size, invar):
        SmoothFilter = GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 0.1))
        kcut = SmoothFilter(1 - pred) * pred
        length = kcut.sum(dim=[2, 3])
        area = pred.sum(dim=[2, 3])
        return length ** 2 / area / 100 if invar else length / area

    predict = F.softmax(predict, dim=1)
    lv_predict = predict[:, 3:4, :, :]
    lvmyo_predict = predict[:, 2:3, :, :] + lv_predict

    lv_Ncut = cal_compactness(lv_predict, kernel_size, invar)
    lvmyo_Ncut = cal_compactness(lvmyo_predict, kernel_size, invar)

    return 0.5*(lv_Ncut+lvmyo_Ncut).mean()

def Gaufunc(xi, xj, sigma):
    return torch.exp(-((xi-xj)**2)/(2*sigma**2))

def gauMatrix(X, sigma): #X:tensor(h*w)
    GauMatrix = torch.zeros([len(X), len(X)])
    for i in range(len(X)):
        for j in range(i, len(X)):
            #if abs(X[i] - X[j]) < sigma * 2:
            GauMatrix[i, j] = Gaufunc(X[i], X[j], sigma)
            GauMatrix[j, i] = GauMatrix[i, j]
    return GauMatrix/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))

def gauMatrixBlur(X, sigma):
    Blured = torch.zeros([len(X)]).to(device=X.device)
    for i in range(len(X)):
        Blured[i] = torch.matmul(torch.exp((-(X-X[i])**2))/(2*sigma**2), X)
    return Blured/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))



def phaseGau(predict, edge, beta, weight_mode='classic'):
    predict = F.softmax(predict, dim=1)[:, 1:, :, :]
    inv_predict = 1-predict
    N, K, H, W = inv_predict.size()
    inv_predict = inv_predict.reshape(-1)
    filt = inv_predict.clone()
    for n in range(N):
        for k in range(K):
            filt[n, k, :, :] = gauMatrixBlur(inv_predict, sigma=0.1).reshape(H, W)
            #filt[n, k, :, :] = torch.matmul(gauMatrix(inv_predict, sigma=0.1), inv_predict).reshape(H, W)
    kcut = filt * predict

    if weight_mode == 'sigmoid':
        tur = 80
        weight_length = (kcut*torch.sigmoid(-beta * (edge - tur))).sum(dim=[2, 3])
    else:
        weight_length = (kcut / (1 + beta * edge)).sum(dim=[2, 3])

    return weight_length.mean() / 1000

def compute_edge_mask(image, sigma, color=True):

    if color:
        left_0 = image[:, 0, :-1, :]
        right_0 = image[:, 0, 1:, :]
        top_0 = image[:, 0, :, :-1]
        bottom_0 = image[:, 0, :, 1:]

        left_1 = image[:, 1, :-1, :]
        right_1 = image[:, 1, 1:, :]
        top_1 = image[:, 1, :, :-1]
        bottom_1 = image[:, 1, :, 1:]

        left_2 = image[:, 2, :-1, :]
        right_2 = image[:, 2, 1:, :]
        top_2 = image[:, 2, :, :-1]
        bottom_2 = image[:, 2, :, 1:]

        mask_h = torch.exp(-1 * ( (left_0 - right_0) ** 2 +(left_1 - right_1) ** 2+(left_2 - right_2) ** 2 )/ (2 * sigma ** 2))
        mask_v = torch.exp(-1 * ( (top_0 - bottom_0) ** 2 +  (top_1 - bottom_1) ** 2 + (top_2 - bottom_2) ** 2) / (2 * sigma ** 2))

    else:
        image_dims = list(image.size())

        if image_dims[1] > 1:
            image = torch.sum(image,dim=1)

        left_ = image[:, :-1, :]
        right_ = image[:, 1:, :]
        top_ = image[:, :, :-1]
        bottom_ = image[:, :, 1:]

        mask_h = torch.exp(-1 * (left_ - right_) ** 2 / (2 * sigma ** 2))
        mask_v = torch.exp(-1 * (top_ - bottom_) ** 2 / (2 * sigma ** 2))

    return mask_h, mask_v #size:(bs, h, w-1), (bs, h-1, w)

def regularized_loss_per_channel(mask_h, mask_v, cl, prediction):

    left = prediction[:, cl ,:-1, :]
    right = prediction[:, cl , 1:, :]
    top = prediction[:,cl ,:,:-1]
    bottom = prediction[:,cl ,:,1:]

    h = torch.mean(abs(left - right) * mask_h)
    v = torch.mean(abs(top - bottom) * mask_v)

    return (h + v)/2.0

def SparseCRFLoss(predict, image, epoch):
    num_classes = predict.size()[1]

    steps = 15
    # weight_start = 1.0
    # weight_end = 1.0
    # weight = weight_start+(weight_end-weight_start)*epoch/(steps-1) if epoch < 15 else 1

    sigma_start = 0.05
    sigma_end = 0.15
    sigma_delta = (sigma_end - sigma_start) / (steps - 1)
    sigma = sigma_start+sigma_delta*epoch
    subtract_eps = 0.0

    mask_h, mask_v = compute_edge_mask(image, sigma)

    mask_h = mask_h - subtract_eps
    mask_v = mask_v - subtract_eps

    loss = 0.
    # regularized loss is not applied to the background class
    for ch in range(1, num_classes):
        loss = loss + regularized_loss_per_channel(mask_h, mask_v, ch, predict)

    return loss #* weight



def get_cls_loss(predict, cls_label, mask):
    """cls_label: (b, k)"""
    """ predict: (b, k, h, w)"""
    """ mask: (b, h, w) """
    b, k, h, w = predict.size()
    predict = torch.softmax(predict, dim=1).view(b, k, -1)
    mask = mask.view(b, -1)

    # if a patch does not contain label k,
    # then none of the pixels in this patch can be assigned to label k
    loss = - (1 - cls_label.view(b, k, 1)) * torch.log(1 - predict + 1e-6)
    loss = torch.sum(loss, dim=1)
    loss = loss[mask != 255].mean()
    return loss


def one_hot(label, nclass):
    b, h, w = label.size()
    label_cp = label.clone()

    label_cp[label > nclass] = nclass
    label_cp = label_cp.view(b, 1, h*w)

    mask = torch.zeros(b, nclass+1, h*w).to(label.device)
    mask = mask.scatter_(1, label_cp.long(), 1).view(b, nclass+1, h, w).float()
    return mask[:, :-1, :, :]


def one_hot_2d(label, nclass):
    h, w = label.size()
    label_cp = label.clone()

    label_cp[label > nclass] = nclass
    label_cp = label_cp.view(1, h*w)

    mask = torch.zeros(nclass+1, h*w).to(label.device)
    mask = mask.scatter_(0, label_cp.long(), 1).view(nclass+1, h, w).float()
    return mask[:-1, :, :]


def cal_protypes(feat, mask, nclass):
    feat = F.interpolate(feat, size=mask.size()[-2:], mode='bilinear')
    b, c, h, w = feat.size()
    prototypes = torch.zeros((b, nclass, c),
                           dtype=feat.dtype,
                           device=feat.device)
    for i in range(b):
        cur_mask = mask[i]
        cur_mask_onehot = one_hot_2d(cur_mask, nclass)

        cur_feat = feat[i]
        cur_prototype = torch.zeros((nclass, c),
                           dtype=feat.dtype,
                           device=feat.device)

        cur_set = list(torch.unique(cur_mask))
        if nclass in cur_set:
            cur_set.remove(nclass)
        if 255 in cur_set:
            cur_set.remove(255)

        for cls in cur_set:
            m = cur_mask_onehot[cls].view(1, h, w)
            sum = m.sum()
            m = m.expand(c, h, w).view(c, -1)
            cls_feat = (cur_feat.view(c, -1)[m == 1]).view(c, -1).sum(-1)/(sum + 1e-6)
            cur_prototype[cls, :] = cls_feat

        prototypes[i] += cur_prototype

    cur_cls_label = build_cur_cls_label(mask, nclass).view(b, nclass, 1)
    mean_vecs = (prototypes.sum(0)*cur_cls_label.sum(0))/(cur_cls_label.sum(0)+1e-6)

    loss = proto_loss(prototypes, mean_vecs, cur_cls_label)

    return prototypes.view(b, nclass, c), loss


def proto_loss(prototypes, vecs, cur_cls_label):
    b, nclass, c = prototypes.size()

    # abs = torch.abs(prototypes - vecs).mean(2)
    # positive = torch.exp(-(abs * abs))
    # positive = (positive*cur_cls_label.view(b, nclass)).sum()/(cur_cls_label.sum()+1e-6)
    # positive_loss = 1 - positive

    vecs = vecs.view(nclass, c)
    total_cls_label = (cur_cls_label.sum(0) > 0).long()
    negative = torch.zeros(1,
                           dtype=prototypes.dtype,
                           device=prototypes.device)

    num = 0
    for i in range(nclass):
        if total_cls_label[i] == 1:
            for j in range(i+1, nclass):
                if total_cls_label[j] == 1:
                    if i != j:
                        num += 1
                        x, y = vecs[i].view(1, c), vecs[j].view(1, c)
                        abs = torch.abs(x - y).mean(1)
                        negative += torch.exp(-(abs * abs))
                        # print(negative)

    negative = negative/(num+1e-6)
    negative_loss = negative

    return negative_loss


def GMM(feat, vecs, pred, true_mask, cls_label):
    b, k, oh, ow = pred.size()

    preserve = (true_mask < 255).long().view(b, 1, oh, ow)
    preserve = F.interpolate(preserve.float(), size=feat.size()[-2:], mode='bilinear')
    pred = F.interpolate(pred, size=feat.size()[-2:], mode='bilinear')
    _, _, h, w = pred.size()

    vecs = vecs.view(b, k, -1, 1, 1)
    feat = feat.view(b, 1, -1, h, w)

    """ 255 caused by cropping, using preserve mask """
    abs = torch.abs(feat - vecs).mean(2)
    abs = abs * cls_label.view(b, k, 1, 1) * preserve.view(b, 1, h, w)
    abs = abs.view(b, k, h*w)

    # """ calculate std """
    # pred = pred * preserve
    # num = pred.view(b, k, -1).sum(-1)
    # std = ((pred.view(b, k, -1)*(abs ** 2)).sum(-1)/(num + 1e-6)) ** 0.5
    # std = std.view(b, k, 1, 1).detach()

    # std = ((abs ** 2).sum(-1)/(preserve.view(b, 1, -1).sum(-1)) + 1e-6) ** 0.5
    # std = std.view(b, k, 1, 1).detach()

    abs = abs.view(b, k, h, w)
    res = torch.exp(-(abs * abs))
    # res = torch.exp(-(abs*abs)/(2*std*std + 1e-6))
    res = F.interpolate(res, size=(oh, ow), mode='bilinear')
    res = res * cls_label.view(b, k, 1, 1)

    return res


def loss_calc(preds, label, ignore_index, reduction='mean', multi=False, class_weight=False,
              ohem=False):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    label_cp = label.clone()
    # label_cp[label == ignore_index] = 1

    # if ohem:
    #     ce = OhemCrossEntropy(use_weight=True)
    # else:
    #     if class_weight:
    #         weight = torch.FloatTensor(
    #             [0.3, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
    #              1.4286, 0.5, 3.3333, 5.0, 10.0, 2.5, 0.8333]).cuda()
    #         ce = torch.nn.CrossEntropyLoss(
    #             ignore_index=255, reduction=reduction, weight=weight)
    #     else:
    #         ce = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)
    #
    # if multi:
    #     aux_pred, pred = preds
    #     loss = ce(aux_pred, label_cp.long())*0.4 + ce(pred, label_cp.long())*0.6
    #
    # else:
    #     loss = ce(preds, label_cp.long())

    ce = nn.BCEWithLogitsLoss()
    loss = ce(preds, label_cp.float())

    return loss

def pCE(preds, label, ignore_index, reduction='mean', multi=False, class_weight=False,
              ohem=False):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    label_cp = label.clone()
    index = label_cp != ignore_index

    ce = nn.BCEWithLogitsLoss()
    loss = ce(preds[index], label_cp.float()[index])

    return loss

def loss_calc2(preds, label, ignore_index, reduction='mean', multi=False, class_weight=False,
              ohem=False):
    """
        This function returns cross entropy loss for semantic segmentation
    """
    label_cp = label.clone()
    return CEloss(preds, label_cp.long(), [ignore_index])


def cal_gmm_loss(pred, res, cls_label, true_mask):
    n, k, h, w = pred.size()
    loss1 = - res * torch.log(pred + 1e-6) - (1 - res) * torch.log(1 - pred + 1e-6)
    loss1 = loss1/2
    loss1 = (loss1*cls_label).sum(1)/(cls_label.sum(1)+1e-6)
    loss1 = loss1[true_mask != 255].mean()

    true_mask_one_hot = one_hot(true_mask, k)
    loss2 = - true_mask_one_hot * torch.log(res + 1e-6) \
            - (1 - true_mask_one_hot) * torch.log(1 - res + 1e-6)
    loss2 = loss2/2
    loss2 = (loss2 * cls_label).sum(1) / (cls_label.sum(1) + 1e-6)
    loss2 = loss2[true_mask < k].mean()
    return loss1+loss2


class OhemCrossEntropy(nn.Module):
    """
    Ohem Cross Entropy Tensor Version
    """

    def __init__(
        self, ignore_index=255, thresh=0.7, min_kept=1e6, use_weight=False, reduce=False
    ):
        super(OhemCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            # weight = torch.FloatTensor(
            #     [
            #         0.8373,
            #         0.918,
            #         0.866,
            #         1.0345,
            #         1.0166,
            #         0.9969,
            #         0.9754,
            #         1.0489,
            #         0.8786,
            #         1.0023,
            #         0.9539,
            #         0.9843,
            #         1.1116,
            #         0.9037,
            #         1.0865,
            #         1.0955,
            #         1.0865,
            #         1.1529,
            #         1.0507,
            #     ]
            # ).cuda()
            # weight = torch.FloatTensor(
            #    [0.4762, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
            #    1.4286, 0.5, 3.3333,5.0, 10.0, 2.5, 0.8333]).to(label.device)
            weight = torch.FloatTensor(
                [0.3, 0.5, 0.4762, 1.4286, 1.1111, 0.4762, 0.8333, 0.5, 0.5, 0.8333, 0.5263, 0.5882,
                 1.4286, 0.5, 3.3333, 5.0, 10.0, 2.5, 0.8333]).cuda()
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", weight=weight, ignore_index=ignore_index
            )
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="none", ignore_index=ignore_index
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="mean", ignore_index=ignore_index
            )

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
            # print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


if __name__ == '__main__':
    # proto = torch.rand(8, 21, 256)
    # vecs = torch.rand(1, 21, 256)
    # cls_label = torch.rand(8, 21, 1)
    # proto_loss(proto, vecs, cls_label)

    def plot_Ncut_comp(abc):
        #SmoothFilter = GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        #kcut = SmoothFilter(abc) * (1 - abc)
        filt = bilateral_blur((1 - abc), kernel_size=3, sigma_color=float('inf'), sigma_space=(float('inf'),float('inf')))
        filt2 = bilateral_blur((1 - abc), kernel_size=5, sigma_color=12., sigma_space=(60., 60.))
        kcut = filt * abc
        area = abc.sum(dim=[2, 3])
        length = kcut.sum(dim=[2, 3])
        Ncut = length / area  # if area>0 else 0
        loss = length ** 2 / area  # if area>0 else 0
        # kcut2 = filt2 * abc
        # length2 = kcut2.sum(dim=[2, 3])
        # loss = length*length2/area

        print('Boundary length:', length)
        print('% change in length:', (length[1][0] - length[0][0]) * 100 / length[0][0])
        print('Mask area:', area)
        print('% change in area:', (area[1][0] - area[0][0]) * 100 / area[0][0])
        print('Normalized Cut:', Ncut)
        print('Ncut % diff:', (Ncut[1][0] - Ncut[0][0]) * 100 / Ncut[0][0])
        print('Proposed:', loss)
        print('Proposed % diff:', (loss[1][0] - loss[0][0]) * 100 / loss[0][0])
        print('')

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(abc[0, 0, :, :])
        ax[0, 0].set(title='Bigger pattern', ylabel='Raw')
        ax[0, 1].imshow(abc[1, 0, :, :])
        ax[0, 1].set(title='Smaller pattern')
        ax[1, 0].imshow(kcut[0, 0, :, :])
        ax[1, 0].set(ylabel='Detected boundary')
        ax[1, 1].imshow(kcut[1, 0, :, :])
        plt.show()

    def plot1(abc):
        n, _, _, _ = abc.size()
        fig, ax = plt.subplots(1, n)
        titles = ['Image', 'GT Mask', 'Given Label']
        for i in range(n):
            ax[i].imshow(abc[i, 0, :, :])
            ax[i].set_title(titles[i])
        plt.show()

    # scale_diff = 120
    # x1 = torch.zeros((2, 1, 500, 500))
    # _, _, h, w = x1.size()
    # for i in range(h):
    #     for j in range(w):
    #         if ((i - h//2) ** 2 + (j - w//2) ** 2) < (min(h, w)//2-10)**2:
    #             x1[0, :, i, j] = 1
    #         if ((i - h // 2) ** 2 + (j - w // 2) ** 2) < (min(h, w) // 2 - 10 - scale_diff) ** 2:
    #             x1[1, :, i, j] = 1
    #
    # plot_Ncut_comp(x1)
    #
    # x2 = torch.zeros((2, 1, 500, 500))
    # _, _, h, w = x2.size()
    # for i in range(h):
    #     for j in range(w):
    #         if ((i - h // 2) ** 2 + (j - w // 2) ** 2) < (min(h, w) // 2 - 10) ** 2:
    #             x2[0, :, i, j] = 1
    #             if ((i - h // 2) ** 2 + (j - w // 2) ** 2) < ((min(h, w) // 2 - 10) ** 2)//2:
    #                 x2[0, :, i, j] = 0
    #         if ((i - h // 2) ** 2 + (j - w // 2) ** 2) < (min(h, w) // 2 - 10 - scale_diff) ** 2:
    #             x2[1, :, i, j] = 1
    #             if ((i - h // 2) ** 2 + (j - w // 2) ** 2) < ((min(h, w) // 2 - 10 - scale_diff) ** 2)//2:
    #                 x2[1, :, i, j] = 0
    #
    # plot_Ncut_comp(x2)
    #
    # x3 = torch.zeros((2, 1, 500, 500))
    # _, _, h, w = x3.size()
    # for i in range(h):
    #     for j in range(w):
    #         if abs(i - h // 2) < (min(h, w) // 2 - 50) and abs(j - w // 2) < (min(h, w) // 2 - 50):
    #             x3[0, :, i, j] = 1
    #         if abs(i - h // 2) < (min(h, w) // 2 - 50 - scale_diff) and abs(j - w // 2) < (min(h, w) // 2 - 50 - scale_diff):
    #             x3[1, :, i, j] = 1
    #
    # plot_Ncut_comp(x3)


    def save_img(im, input_type, id):
        suffle = '.jpg' if input_type == 'image' else '.png'
        base_path = '../../../Datasets/ACDC/tz/small'
        if input_type == 'image':
            path = '/JPEGImage'
        elif input_type == 'fullmask':
            path = '/FullMask'
        else:
            path = '/TestMask'
        path = base_path + path
        im = Image.fromarray(np.uint8(im))
        if not os.path.exists(path):
            os.makedirs(path)
        im.save(os.path.join(path, 'plain' + str(id) + suffle))

    # x5 = torch.zeros(8, 3, 20, 20)
    # x5[:, :, 6:15, 6: 15] = 1
    # predict5 = torch.zeros(8, 2, 20, 20)
    # predict5[:, 1, :,:] = 1
    # predict5[:, 0, :, :] = 1-predict5[:, 1, :, :]
    #
    # predict = predict5[:,0:1,:,:]
    # img = x5
    #
    # c = (img * predict).sum(dim=[2, 3]) # c:(bs, img_channel)
    # area = predict.sum(dim=[2, 3])
    # area[area==0] = 1
    # c = c / area
    # c = torch.unsqueeze(torch.unsqueeze(c, 2), 3)
    # value = (predict * (img - c) ** 2).sum(dim=[2, 3]).mean()


    # path = '../records/generated_images/plain'
    # for file in os.listdir(path):
    #     im = Image.open(os.path.join(path, file))
    #     a = np.uint8(im)
    #     print(file, np.sum(a==0))
    #     plt.imshow(a)
    #     plt.show()
    #     print(' ')
    x4 = torch.zeros((3, 1, 50, 50))
    x4[2, :, :, :] = 4
    _, _, h, w = x4.size()

    for i in range(h):
        for j in range(w):
            #x4[0, :, i, j] += 0.3 * torch.rand(1).item()

            if ((i - h // 4) ** 2 + (j - w // 4) ** 2) < (min(h, w) // 4 ) ** 2:
                x4[0, :, i, j] = 0.2
                x4[1, :, i, j] = 2

                if ((i - h // 4) ** 2 + (j - w // 4) ** 2) < (min(h, w) // 6) ** 2:
                    x4[0, :, i, j] = 0.7
                    x4[1, :, i, j] = 1

            if ((i - 3*h // 4) ** 2 + (j - 3*w // 4) ** 2) < (min(h, w) // 4) ** 2:
                x4[0, :, i, j] = 0.2
                x4[1, :, i, j] = 2

                if ((i - 3*h // 4) ** 2 + (j - 3*w // 4) ** 2) < (min(h, w) // 6) ** 2:
                    x4[0, :, i, j] = 0.7
                    x4[1, :, i, j] = 1

                # if ((i - 3 * h // 4) ** 2 + (j - 3 * w // 4) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 2:
                #     x4[0, :, i, j] = 1
                #     x4[1, :, i, j] = 3

            # if ((i - h // 4) ** 2 + (j - 3* w // 4) ** 2) < (min(h, w) // 4 - 10) ** 2:
            #     x4[0, :, i, j] = 0.2
            #     x4[1, :, i, j] = 2
            #
            #     if ((i - h // 4) ** 2 + (j - 3* w // 4) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 2:
            #         x4[0, :, i, j] = 1
            #         x4[1, :, i, j] = 3

                    # if i < 131 and i > 126 and j > 120 and j < 140:
                    #     x4[0, :, i, j] = 0.2
                    # if ((i - 157) ** 2 + (j - 165) ** 2) < 10:
                    #     x4[0, :, i, j] = 0.2
                    # if ((i - 160) ** 2 + (j - 162) ** 2) < 12:
                    #     x4[0, :, i, j] = 0.2
                    # if ((i - 137) ** 2 + (j - 125) ** 2) < 50:
                    #     x4[0, :, i, j] = 0.2
                    # if ((i - 140) ** 2 + (j - 124) ** 2) < 30:
                    #     x4[0, :, i, j] = 0.2
                    # if ((i - 146) ** 2 + (j - 120) ** 2) < 36:
                    #     x4[0, :, i, j] = 0.2


            #Circular scribbles
            # if ((i - h // 4) ** 2 + (j - w // 4) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 0.3:
            #     x4[2, :, i, j] = 0
            #     if ((i - h // 4) ** 2 + (j - w // 4) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 0.305:
            #         x4[2, :, i, j] = 4
            # if ((i - h // 4) ** 2 + (j - w // 4) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 1.45:
            #     # if (i>140 and i<161) or (j>140 and j<161):
            #     x4[2, :, i, j] = 1 #2
            #     if ((i - h // 4) ** 2 + (j - w // 4) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 1.5:
            #         x4[2, :, i, j] = 4
            # if ((i - 3 * h // 4) ** 2 + (j - 3 * w // 4) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 3:
            #     x4[2, :, i, j] = 3
            #     if ((i - 3 * h // 4) ** 2 + (j - 3 * w // 4) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 3.05:
            #         x4[2, :, i, j] = 4
            # if i > 138 and i < 162 and j >138 and j< 162:
            #     if i + j == 300 and i==j:
            #         x4[2, :, i, j] = 3
                # if i==j:
                #     x4[2, :, i, j] = 3

            # if ((i - h // 2) ** 2 + (j - 0) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 0.3:
            #     if j>30 and j<91:
            #         x4[2, :, i, j] = 0
            #         if ((i - h // 2) ** 2 + (j - 0) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 0.305:
            #             x4[2, :, i, j] = 4

            # if i>99 and i<101 and j>98:
            #     x4[2, :, i, j] = 0


            #Partial Mask
            if ((i - h // 4) ** 2 + (j - w // 4) ** 2) < (min(h, w) // 4 +5) ** 2:
                x4[2, :, i, j] = 0
                if ((i - h // 4) ** 2 + (j - w // 4) ** 2) < (min(h, w) // 4 + 4) ** 2:
                    x4[2, :, i, j] = 4
            if ((i - h // 4) ** 2 + (j - w // 4) ** 2) < (min(h, w) // 4 -1) ** 2:
                x4[2, :, i, j] = 2
                if ((i - h // 4) ** 2 + (j - w // 4) ** 2) < (min(h, w) // 4 -2) ** 2:
                    x4[2, :, i, j] = 4

            if ((i - h // 4) ** 2 + (j - w // 4) ** 2) < (min(h, w) // 6-1) ** 2:
                x4[2, :, i, j] = 1
                if ((i - h // 4) ** 2 + (j - w // 4) ** 2) < (min(h, w) // 6 - 2) ** 2:
                    x4[2, :, i, j] = 4

            if i <= h//4 + 1 and i >= h//4-1:
                x4[2, :, i, j] = 4
            if j <= w//4 + 1 and j >= w//4-1:
                x4[2, :, i, j] = 4

            # if ((i - 3 * h // 4) ** 2 + (j - 3 * w // 4) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 0.3:
            #     x4[2, :, i, j] = 0
            #     if ((i - 3 * h // 4) ** 2 + (j - 3 * w // 4) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 0.305:
            #         x4[2, :, i, j] = 4
            #
            # if ((i - 3*h // 4) ** 2 + (j - 3*w // 4) ** 2) < (min(h, w) // 4 - 10) ** 2:
            #     x4[2, :, i, j] = 2
            #     if ((i - 3 * h // 4) ** 2 + (j - 3 * w // 4) ** 2) < ((min(h, w) // 4 - 10) ** 2) // 2:
            #         x4[2, :, i, j] = 3
            #
            # if i > (3 * h // 4) and j>100:
            #     x4[2, :, i, j] = 4
            # if j > (3 * w // 4) and i>100:
            #     x4[2, :, i, j] = 4

            # if i == j :
            #     x4[2, :, i, j] = 0
            # if i + j == 200 and (i>=100):
            #     x4[2, :, i, j] = 0

    # point Mask
    # x4[2, :, h//4, 3*w//4] = 1
    # x4[2, :, (h-1):(h+1), 0:2] = 0
    # x4[2, :, 0:2, 0:2] = 0
    # x4[2, :, 0:2, (w-1):(w+1)] = 0
    # x4[2, :, (h-1):(h+1), (w-1):(w+1)] = 0
    #
    # divided = 10
    # d = int(x4.shape[2]//divided)
    # for i in range(divided):
    #     x4[0, :, d*i:d*i+d, :] = 0.25 + 0.5 * (i%2)
    #     for j in range(x4.shape[3]):
    #         x4[0, :, d * i:d * i + d, j] += 0.1 * torch.rand(5)
    #
    # x4[1, :, :25, :] = 1
    # x4[1, :, 25:, :] = 0
    # x4[2, :, 7, :] = 1
    # x4[2, :, 43, :] = 0



    plot1(x4)
    im = x4[0,0,:,:].cpu().numpy()*255
    fullmask = x4[1,0,:,:].cpu().numpy()
    scribble = x4[2,0,:,:].cpu().numpy()
    for i in range(8, 9):
        save_img(im, 'image', i)
        save_img(fullmask, 'fullmask', i)
        save_img(scribble, 'mask', i)


