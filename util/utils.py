import numpy as np
import logging
import os
import torch
import torch.nn.functional as F
from math import *
from PIL import Image
import matplotlib.pyplot as plt

from kornia.core import Module, Tensor, pad
from kornia.core.check import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE
from kornia.filters.kernels import _unpack_2d_ks, get_gaussian_kernel2d
from kornia.filters.median import _compute_zero_padding

def _bilateral_blur(
    input: Tensor,
    guidance,
    kernel_size,
    sigma_color,
    sigma_space,
    border_type: str = 'reflect',
    color_distance_type: str = 'l1',
    visual: bool = False
) -> Tensor:
    "Single implementation for both Bilateral Filter and Joint Bilateral Filter"

    KORNIA_CHECK_IS_TENSOR(input)
    KORNIA_CHECK_SHAPE(input, ['B', 'C', 'H', 'W'])
    if guidance is not None:
        # NOTE: allow guidance and input having different number of channels
        KORNIA_CHECK_IS_TENSOR(guidance)
        KORNIA_CHECK_SHAPE(guidance, ['B', 'C', 'H', 'W'])
        KORNIA_CHECK(
            (guidance.shape[0] == input.shape[0]) and (guidance.shape[-2:] == input.shape[-2:]),
            "guidance and input should have the same batch size and spatial dimensions",
        )

    if isinstance(sigma_color, Tensor):
        KORNIA_CHECK_SHAPE(sigma_color, ['B'])
        sigma_color = sigma_color.to(device=input.device, dtype=input.dtype).view(-1, 1, 1, 1, 1)

    ky, kx = _unpack_2d_ks(kernel_size)
    pad_y, pad_x = _compute_zero_padding(kernel_size)

    padded_input = pad(input, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
    unfolded_input = padded_input.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    if guidance is None:
        guidance = input
        unfolded_guidance = unfolded_input
    else:
        padded_guidance = pad(guidance, (pad_x, pad_x, pad_y, pad_y), mode=border_type)
        unfolded_guidance = padded_guidance.unfold(2, ky, 1).unfold(3, kx, 1).flatten(-2)  # (B, C, H, W, Ky x Kx)

    diff = unfolded_guidance - guidance.unsqueeze(-1)
    if color_distance_type == "l1":
        color_distance_sq = diff.abs().sum(1, keepdim=True).square()
    elif color_distance_type == "l2":
        color_distance_sq = diff.square().sum(1, keepdim=True)
    else:
        raise ValueError("color_distance_type only acceps l1 or l2")
    color_kernel = (-0.5 / sigma_color**2 * color_distance_sq).exp()  # (B, 1, H, W, Ky x Kx)

    space_kernel = get_gaussian_kernel2d(kernel_size, sigma_space, device=input.device, dtype=input.dtype)
    space_kernel = space_kernel.view(-1, 1, 1, 1, kx * ky)

    kernel = space_kernel * color_kernel
    out = (unfolded_input * kernel).sum(-1) / kernel.sum(-1)
    if visual:
        _, _, H, W, _ = unfolded_input.size()
        fig, ax = plt.subplots(1,3)
        names = ['color_kernel', 'space_kernel', 'bilateral_kernel']

        for a, ke in enumerate([color_kernel, space_kernel, kernel]):
            kw = torch.ones(unfolded_input.size()) * ke
            kw = kw[0, 0, :, :, :]
            vk = torch.zeros(H * W, H * W)
            for i in range(H):
                for j in range(W):
                    n = i * W + j
                    if i == 0 and j == 0:
                        vk[n, n:n + 2] = kw[i, j, 4:6]
                        vk[n, (i + W):(i + W + 2)] = kw[i, j, 7:9]
                    elif i == 0 and j == W - 1:
                        vk[n, (n - 1):(n + 1)] = kw[i, j, 3:5]
                        vk[n, (n + W - 1):(n + W + 1)] = kw[i, j, 6:8]
                    elif i == H - 1 and j == 0:
                        vk[n, n:n + 2] = kw[i, j, 4:6]
                        vk[n, (n - W):(n - W + 2)] = kw[i, j, 2:4]
                    elif i == H - 1 and j == W - 1:
                        vk[n, (n - 1):(n + 1)] = kw[i, j, 3:5]
                        vk[n, (n - W - 1):(n - W + 1)] = kw[i, j, 0:2]
                    elif i == 0:
                        vk[n, n - 1:n + 2] = kw[i, j, 3:6]
                        vk[n, n + W - 1:n + W + 2] = kw[i, j, 6:9]
                    elif i == H - 1:
                        vk[n, n - 1:n + 2] = kw[i, j, 3:6]
                        vk[n, n - W - 1:n - W + 2] = kw[i, j, 0:3]
                    elif j == 0:
                        vk[n, n - W:n - W + 2] = kw[i, j, 1:3]
                        vk[n, n:n + 2] = kw[i, j, 4:6]
                        vk[n, n + W:n + W + 2] = kw[i, j, 7:9]
                    elif j == W - 1:
                        vk[n, n - W - 1:n - W + 1] = kw[i, j, 0:2]
                        vk[n, n - 1:n + 1] = kw[i, j, 3:5]
                        vk[n, n + W - 1:n + W + 1] = kw[i, j, 6:8]
                    else:
                        vk[n, n - W - 1:n - W + 2] = kw[i, j, 0:3]
                        vk[n, n - 1:n + 2] = kw[i, j, 3:6]
                        vk[n, (n + W - 1):(n + W + 2)] = kw[i, j, 6:9]

            ax[a].imshow(vk.cpu().numpy())
            ax[a].set_title(names[a])
        plt.show()

    return out



def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6

def get_model_name(mode):
    if mode == 'full':
        return 'f_'
    elif mode == 'point':
        return 'p_'
    else:
        return 's_'


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

        cmap[19] = np.array([0, 0, 0])
        cmap[255] = np.array([0, 0, 0])

    elif dataset == 'acdc':
        # For test set, change cmap[4] to cmap[0]; For scribble version, keep 1st line and [0] to [4]
        #cmap[0] = np.array([153, 153, 153])
        cmap[1] = np.array([250, 170, 30])
        cmap[2] = np.array([152, 251, 152])
        cmap[3] = np.array([0,  0, 230])
        cmap[0] = np.array([0, 0, 0])

    elif dataset == 'ham' or dataset == 'plain':
        cmap[0] = np.array([0, 0, 0])
        cmap[1] = np.array([255, 255, 255])

    return cmap


def cal_acc(hist, num_class):
    accuracy = np.zeros(num_class)
    for i in range(num_class):
        accuracy[i] = hist[i, i] / np.sum(hist[i, :])
    return accuracy


def eval_Area(predictions, gts, num_classes):
    for lp, lt in zip(predictions, gts):
        label_pred, label_true = lp.flatten(), lt.flatten()
        mask_pt = (label_true >= 0) & (label_true < num_classes)
        n_c = num_classes if num_classes > 1 else 2
        hist = np.bincount(n_c * label_true[mask_pt].astype(int) + label_pred[mask_pt],
                           minlength=n_c ** 2).reshape(n_c, n_c)

        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        dice = 2*np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0))
        acc = cal_acc(hist, num_classes)
        return np.insert(iu, 0, np.nanmean(iu)), np.insert(dice, 0, np.nanmean(dice)), np.insert(acc, 0, np.nanmean(acc))


def cal_tv(x): #x: [batch_size, h, w]
    x = np.float32(x)
    h_diffs = np.zeros_like(x, dtype=np.float32)
    w_diffs = np.zeros_like(x, dtype=np.float32)
    h_diffs[:, 0:-1, :] = np.power(x[:, 1:, :] - x[:, 0:-1, :], 2)
    w_diffs[:, :, 0:-1] = np.power(x[:, :, 1:] - x[:, :, 0:-1], 2)
    TV = np.sqrt(h_diffs + w_diffs)
    return np.mean(TV)/2


def eval_boundary(predictions, gts, num_classes):
    pred_length = np.zeros(num_classes)
    gt_length = np.zeros(num_classes)
    pred_inv = np.zeros(num_classes)
    gt_inv = np.zeros(num_classes)

    for i in range(1, num_classes):
        predict = predictions == i
        gt = gts == i

        #kcut_pred = (gaussian_filter(1-predict, sigma=0.4, truncate=2.5) * predict).sum()
        #kcut_gt = (gaussian_filter(1-gt, sigma=0.4, truncate=2.5) * gt).sum()
        kcut_pred = cal_tv(predict)
        kcut_gt = cal_tv(gt)

        pred_length[i] = kcut_pred
        gt_length[i] = kcut_gt
        pred_inv[i] = kcut_pred**2/predict.sum()
        gt_inv[i] = kcut_gt**2/gt.sum()

    pred_length[0] = np.mean(pred_length[pred_length != 0])
    gt_length[0] = np.mean(pred_length[pred_length != 0])
    pred_inv[0] = np.nanmean(pred_inv[1:])
    gt_inv[0] = np.nanmean(gt_inv[1:])

    return pred_length, gt_length, pred_inv, gt_inv



class meanIOU:
    def __init__(self, num_classes):
        self.num_classes = num_classes if num_classes>1 else 2
        self.hist = np.zeros((self.num_classes, self.num_classes))

    def _fast_hist(self, prediction, gt):
        pred_vec, gt_vec = prediction.copy(), gt.copy()
        mask_pt = (gt >= 0) & (gt < self.num_classes)
        hist = np.bincount(self.num_classes * gt_vec[mask_pt].astype(int) + pred_vec[mask_pt],
                           minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for prediction, gt in zip(predictions, gts):
            self.hist += self._fast_hist(prediction.flatten(), gt.flatten())

    def evaluate(self):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        dice = 2 * np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0))
        acc = cal_acc(self.hist, self.num_classes)
        return iu, dice, acc, np.nanmean(iu), np.nanmean(dice), np.nanmean(acc)


class RecordLoss(object):
    def __init__(self):
        self.total_loss = []
        self.data_loss = []
        self.edge_loss = []
        self.mIoU = []
        self.mDice = []
        self.mAcc = []

    def add_batch(self, total_loss, data_loss, edge_loss, mIoU, mDice, mAcc):
        self.total_loss.append(total_loss)
        self.data_loss.append(data_loss)
        self.edge_loss.append(edge_loss)
        self.mIoU.append(mIoU)
        self.mDice.append(mDice)
        self.mAcc.append(mAcc)

    def save_graph(self, model_name):
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('losses')
        ax1.plot(self.total_loss, 'b', label='total_loss')
        ax1.plot(self.data_loss, 'r', label='data_loss')
        #ax1.plot(self.edge_loss, 'g', label='edge_loss')
        ax1.tick_params(axis='y')
        ax1.legend()

        ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        ax2.set_ylabel('metrics')  # we already handled the x-label with ax1
        ax2.plot(self.mIoU, 'c', label='mIoU')
        ax2.plot(self.mDice, 'm', label='mDice')
        ax2.plot(self.mAcc, 'y', label='mAcc')
        ax2.tick_params(axis='y')
        ax2.legend()

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(model_name + '.png')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = F.pad(img, (0, 0, rows_missing, cols_missing), 'constant', 0)
    return padded_img


def pre_slide(model, image, num_classes=21, tile_size=(321, 321), tta=False):
    image_size = image.shape  # bigger than (1, 3, 512, 512), i.e. (1,3,1024,1024)
    overlap = 2 / 3  # 每次滑动的重合率为1/2

    stride = ceil(tile_size[0] * (1 - overlap))  # 滑动步长:769*(1-1/3) = 513
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # 行滑动步数:(1024-769)/513 + 1 = 2
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)  # 列滑动步数:(2048-769)/513 + 1 = 4

    full_probs = torch.zeros((1, num_classes, image_size[2], image_size[3])).cuda()  # 初始化全概率矩阵 shape(1024,2048,19)

    count_predictions = torch.zeros((1, 1, image_size[2], image_size[3])).cuda()  # 初始化计数矩阵 shape(1024,2048,19)
    tile_counter = 0  # 滑动计数0

    for row in range(tile_rows):  # row = 0,1
        for col in range(tile_cols):  # col = 0,1,2,3
            x1 = int(col * stride)  # 起始位置x1 = 0 * 513 = 0
            y1 = int(row * stride)  # y1 = 0 * 513 = 0
            x2 = min(x1 + tile_size[1], image_size[3])  # 末位置x2 = min(0+769, 2048)
            y2 = min(y1 + tile_size[0], image_size[2])  # y2 = min(0+769, 1024)
            x1 = max(int(x2 - tile_size[1]), 0)  # 重新校准起始位置x1 = max(769-769, 0)
            y1 = max(int(y2 - tile_size[0]), 0)  # y1 = max(769-769, 0)

            img = image[:, :, y1:y2, x1:x2]  # 滑动窗口对应的图像 imge[:, :, 0:769, 0:769]
            padded_img = pad_image(img, tile_size)  # padding 确保扣下来的图像为769*769

            tile_counter += 1  # 计数加1
            # print("Predicting tile %i" % tile_counter)

            # 将扣下来的部分传入网络，网络输出概率图。
            # use softmax
            if tta:
                padded = model(padded_img, True)
            else:
                padded = model(padded_img)[0] if isinstance(model(img), tuple) else model(padded_img)
                padded = F.softmax(padded, dim=1)

            pre = padded[:, :, 0:img.shape[2], 0:img.shape[3]]  # 扣下相应面积 shape(769,769,19)
            count_predictions[:, :, y1:y2, x1:x2] += 1  # 窗口区域内的计数矩阵加1
            full_probs[:, :, y1:y2, x1:x2] += pre  # 窗口区域内的全概率矩阵叠加预测结果

    # average the predictions in the overlapping regions
    full_probs /= count_predictions  # 全概率矩阵 除以 计数矩阵 即得 平均概率

    return full_probs   # 返回整张图的平均概率 shape(1, 1, 1024,2048)


def ms_test(model, img):
    n, c, h, w = img.size()
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    final_result = None

    for scale in scales:
        cur_h, cur_w = int(h * scale), int(w * scale)
        cur_x = F.interpolate(img, size=(cur_h, cur_w), mode='bilinear', align_corners=True)

        out = model(cur_x)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        final_result = out if final_result is None else (final_result + out)

        out = model(cur_x.flip(3)).flip(3)
        out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
        final_result += out

    return final_result / 14


def evaluate(model, loader, mode, cfg):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    metric = meanIOU(num_classes=cfg['nclass'])
    scores = []

    with torch.no_grad():
        for img, mask, prob, id in loader:
            if len(img.size()) > 4:
                img = torch.squeeze(img, 0).to(device=device)
                mask = torch.squeeze(mask, 0).to(device=device)
            else:
                img = img.to(device=device)
                mask = mask.to(device=device)

            if mode == 'sliding_window':
                final = pre_slide(model, img, num_classes=cfg['nclass'],
                                 tile_size=(cfg['crop_size'], cfg['crop_size']), tta=False)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                if cfg['nclass'] > 2:
                    pred = model(img).argmax(1)
                else:
                    pred = torch.sigmoid(model(img).squeeze(1))
                    pred[pred >= 0.5] = 1
                    pred[pred < 0.5] = 0
                    pred = pred.long()

            metric.add_batch(pred.cpu().numpy(), mask.cpu().numpy())
            #iu, dice, acc = eval_Area(pred.cpu().numpy(), mask.cpu().numpy(), cfg['nclass'])
            #scores.append([iu, dice, acc])

    iou_class, dice_class, pa_class, mIOU, mDICE, mPA = metric.evaluate()
    #scores = np.nanmean(scores, axis=0) * 100
    scores = None

    return mIOU * 100.0, mDICE * 100.0, mPA * 100.0, iou_class, dice_class, pa_class, scores

def evaluate_res(model, loader, mode, cfg):
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    metric = meanIOU(num_classes=cfg['nclass'])
    scores = []

    with torch.no_grad():
        for img, mask, prob, id in loader:
            if len(img.size()) > 4:
                img = torch.squeeze(img, 0).to(device=device)
                mask = torch.squeeze(mask, 0).to(device=device)
                prob = torch.squeeze(prob, 0).to(device=device)
            else:
                img = img.to(device=device)
                prob = prob.to(device=device)

            #img = torch.cat((img, prob.unsqueeze(1)), dim=1)

            if mode == 'sliding_window':
                final = pre_slide(model, img, num_classes=cfg['nclass'],
                                 tile_size=(cfg['crop_size'], cfg['crop_size']), tta=False)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                #pred = model(img).argmax(1)
                pred = torch.sigmoid(model(img).squeeze(1))
                pred = pred+prob
                pred[pred >= 0.5] = 1
                pred[pred < 0.5] = 0
                pred = pred.long()

            metric.add_batch(pred.cpu().numpy(), mask.cpu().numpy())
            #iu, dice, acc = eval_Area(pred.cpu().numpy(), mask.cpu().numpy(), cfg['nclass'])
            #scores.append([iu, dice, acc])

    iou_class, dice_class, pa_class, mIOU, mDICE, mPA = metric.evaluate()
    #scores = np.nanmean(scores, axis=0) * 100
    scores = None

    return mIOU * 100.0, mDICE * 100.0, mPA * 100.0, iou_class, dice_class, pa_class, scores


if __name__ == '__main__':
    n_classes = 4
    a = torch.ones((4,n_classes,6,6))-(torch.rand((1,n_classes,6,6)))*1
    a = a/a.sum(dim=1).unsqueeze(dim=1)
    _bilateral_blur(a, None, 3, 0.2, (10,10), 'reflect', 'l1', True)
    print('')
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0


    r = g = b = 0
    c = 15

    for i in range(256):
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
    print(r, g, b)
