import argparse
from itertools import cycle
import logging
import os
import pprint
import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.utils.data import DataLoader
import yaml

from dataset.sass import *
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, get_model_name, AverageMeter, RecordLoss, intersectionAndUnion, init_log, evaluate
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Sparsely-annotated Semantic Segmentation')
parser.add_argument('--config', default='./configs/ecssd_ms.yaml', type=str) #default='./configs/plain_rkhs.yaml'
parser.add_argument('--mode', default=None, type=str)
parser.add_argument('--save_path', default='./checkpoints/ecssdplus_ms', type=str) #default='./checkpoints/rkhs'
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')
    if args.mode is not None:
        cfg['mode'] = args.mode
        cfg['model_name'] = get_model_name(args.mode) + cfg['model_name']
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    logger.info('{}\n'.format(pprint.pformat(cfg)))

    cudnn.enabled = True
    cudnn.benchmark = True

    # Load network
    os.makedirs(args.save_path, exist_ok=True)
    if cfg['network'] == 'DeepLabV3Plus':
        from model.semseg.deeplabv3plus import DeepLabV3Plus
        model = DeepLabV3Plus(cfg, aux=cfg['aux']).to(device=device)
    else:
        from model.semseg.unet_model import UNet
        model = UNet(cfg).to(device=device)
    if cfg['load']:
        state_dict = torch.load(cfg['load'], map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {cfg["load"]}')
    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    previous_best_loss_mIoU = float(cfg['load'][-9:-4]) if cfg['load'] else 0.0
    previous_best_loss = float(cfg['load'][-15:-10]) + 4 if cfg['load'] else 10.0
    previous_best_mIoU = previous_best_loss_mIoU
    previous_best_mIoU_loss = previous_best_loss
    records = RecordLoss()


    # Optimizer
    optimizer = Adam([param for name, param in model.named_parameters()], lr=cfg['lr'])
    scheduler = StepLR(optimizer, step_size=cfg['lr_step'], gamma=cfg['lr_multi'])
    # scheduler2 = ExponentialLR(optimizer, gamma=0.99)

    # Load data
    if cfg['dataset'] == 'pascal':
        trainset = VocDataset(cfg['dataset'], cfg['data_root'], cfg['mode'],
                              cfg['crop_size'], False)
        valset = VocDataset(cfg['dataset'], cfg['data_root'], 'val', None)
    # elif cfg['dataset'] == 'acdc':
    #     trainset = ACDCDataset(cfg['dataset'], cfg['data_root'], cfg['mode'],
    #                            cfg['crop_size'], False)
    #     valset = ACDCDataset(cfg['dataset'], cfg['data_root'], 'val', None)
    elif cfg['dataset'] == 'acdc':
        trainset = ACDCRkhsDataset(cfg['dataset'], cfg['data_root'], cfg['mode'], cfg['crop_size'], cfg['nclass'], cfg['aug'])
        valset = ACDCRkhsDataset(cfg['dataset'], cfg['data_root'], 'val', cfg['crop_size'], cfg['nclass'])
    elif cfg['dataset'] == 'ecssd':
        trainset = ECSSDRkhsDataset(cfg['dataset'], cfg['data_root'], cfg['mode'], cfg['crop_size'], cfg['nclass'], cfg['aug'])
        valset = ECSSDRkhsDataset(cfg['dataset'], cfg['data_root'], 'val', cfg['crop_size'], cfg['nclass'])
    elif cfg['dataset'] == 'ham':
        trainset = HAMDataset(cfg['dataset'], cfg['data_root'], cfg['mode'],
                              cfg['crop_size'], cfg['aug'])
        valset = HAMDataset(cfg['dataset'], cfg['data_root'], 'val', None)
    else:
        trainset = VocRkhsDataset(cfg['dataset'], cfg['data_root'], cfg['mode'], cfg['crop_size'], cfg['aug'])
        valset = VocRkhsDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=False, num_workers=0, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=False, num_workers=0,
                           drop_last=False)

    iters = 0
    # total_iters = len(trainloader) * cfg['epochs']

    #for epoch in range(cfg['epochs']):
    epoch = -1
    stop_training = False
    error_bound = 0.001
    while not stop_training:
        epoch += 1
        logger.info('===========> Epoch: {:}, LR: {:.6f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best_loss_mIoU))

        model.train()
        loss_m = AverageMeter()
        data_m = AverageMeter()
        ms_m = AverageMeter()
        tv_m = AverageMeter()

        for i, (img, mask, edge, prob, cls_label, id) in enumerate(trainloader):
            img, mask, edge, prob, cls_label = img.to(device), mask.to(device), edge.to(device), prob.to(device), cls_label.to(device)
            # plt.imshow(torch.permute(edge[0], (1,2,0)))
            # plt.show()
            if cfg['network'] == 'DeepLabV3Plus':
                feat, pred = model(img)
            else:
                pred = model(img)

            # plt.imshow(torch.sigmoid(pred.squeeze(1)).detach().numpy()[0])
            # plt.show()

            data_loss = pCE(pred.squeeze(1), mask,
                            ignore_index=255, multi=False,
                            class_weight=False, ohem=False)
            # data_loss = fidelityloss(pred, prob, fk_form='1-2p', n_class=1)

            ms_loss = MS(img, pred)
            tv_loss = TV(pred)

            # total loss
            loss = data_loss + cfg['ms_alpha'] * (ms_loss + cfg['w_e_alpha'] * tv_loss)

            # Update Optimizer
            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clips
            optimizer.step()

            loss_m.update(loss.item(), img.size()[0])
            data_m.update(data_loss.item(), img.size()[0])
            ms_m.update(ms_loss.item(), img.size()[0])
            tv_m.update(tv_loss.item(), img.size()[0])

            iters += 1
            # lr = cfg['lr'] * (1 - iters / (total_iters*1.25)) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr
            # if cfg['network'] == 'DeepLabV3Plus':
            #     optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            # optimizer.param_groups[0]["lr"] = cfg['lr'] * cfg['lr_multi']

            if epoch != 0 and (i%100)==0:
                logger.info('Iter:{:}, avg loss (total): {:.6f}'.format(i, loss_m.avg)
                            + ' | (scribble term): {:.6f} | (ms term): {:.6f} | (tv term): {:.6f}'.format(data_m.avg, ms_m.avg, tv_m.avg)
                            + ' | ratio: {:.6f}'.format(tv_m.avg / ms_m.avg))
            if epoch == 0 and (i%50)==0:
                logger.info('Iter:{:}, image loss (total): {:.6f}'.format(i, loss)
                            + ' | (scribble term): {:.6f} | (ms term): {:.6f} | (tv term): {:.6f}'.format(data_loss, ms_m.avg, tv_loss)
                            + ' | ratio: {:.6f}'.format(tv_loss/ms_loss))

        mIOU, mDICE, mPA = evaluate(model, valloader, 'original', cfg)[0:3]
        #scores = evaluate(model, valloader, eval_mode, cfg)[-1]

        logger.info(
            '***** Evaluation {} ***** >>>> mIOU: {:.2f}, mDICE:{:.2f}, mPA:{:.2f}'.format('original', mIOU, mDICE, mPA))
        loss_logging = 'Epoch {} | loss:{:.3f}, scribble_loss:{:.3f}, ms_loss:{:.3f}, tv_loss:{:.3f} ' \
                       '| mIOU:{:.2f}\n'.format(epoch, loss_m.avg,
                                                data_m.avg, ms_m.avg,
                                                tv_m.avg, mIOU)
        records.add_batch(loss_m.avg, data_m.avg, tv_m.avg, mIOU, mDICE, mPA)

        scheduler.step()

        #previous_best = 100.0 if epoch==0 else previous_best
        if loss_m.avg < previous_best_loss:
            if previous_best_loss_mIoU != 0:
                tmp_loss_path = os.path.join(args.save_path, 'loss_%s_%.3f_%.2f.pth' % (cfg['model_name'], previous_best_loss, previous_best_loss_mIoU))
                if os.path.exists(tmp_loss_path):
                    os.remove(tmp_loss_path)
            previous_best_loss_mIoU = mIOU
            previous_best_loss = loss_m.avg
            torch.save(model.state_dict(),
                       os.path.join(args.save_path, 'loss_%s_%.3f_%.2f.pth' % (cfg['model_name'], loss_m.avg, mIOU)))

        if mIOU > previous_best_mIoU:
            if previous_best_mIoU != 0:
                tmp_loss_path = os.path.join(args.save_path, 'mIoU_%s_%.3f_%.2f.pth' % (
                    cfg['model_name'], previous_best_mIoU_loss, previous_best_mIoU))
                if os.path.exists(tmp_loss_path):
                    os.remove(tmp_loss_path)
            previous_best_mIoU = mIOU
            previous_best_mIoU_loss = loss_m.avg
            torch.save(model.state_dict(),
                       os.path.join(args.save_path, 'mIoU_%s_%.3f_%.2f.pth' % (cfg['model_name'], loss_m.avg, mIOU)))

        # logger.info(loss_logging)
        # with open('{}.txt'.format(cfg['model_name']), 'a') as f:
        #     f.write(loss_logging)
        if epoch % 1000 == 0:
            records.save_graph(cfg['model_name'])

        # Stop iteration
        # if seg_m.avg<error_bound and inv_edge_m.avg<error_bound and weight_edge_m.avg<error_bound:
        #     stop_training = True
        if epoch == cfg['epochs']:
            stop_training = True

    torch.save(model.state_dict(),  # model.module.state_dict(),  # if Distribute is used
               os.path.join(args.save_path, '%s_%.3f_%.2f.pth' % (cfg['model_name'], loss_m.avg, mIOU)))
    records.save_graph(cfg['model_name'])



if __name__ == '__main__':
    main()
