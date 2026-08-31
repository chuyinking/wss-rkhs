from dataset.sass import *
from dataset.transform import normalize_back
from util.utils import *
import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

#os.environ['CUDA_VISIBLE_DEVICES'] = "0"

'''
This document is a copy of eval.py for evaluating the RKHS (no training) against the GT over the training sets
'''

MODE = None
model_name = 'edge5e-5t_sal1400_bs32_max'

save_img = None
save_record = False

def parse_args():
    name = 'ecssd'

    parser = argparse.ArgumentParser(description='SASS Framework')
    parser.add_argument('--resume_model', type=str,
                        default='./checkpoints/ecssdplus_rkhs/mIoU_edge5e-5_fk_lr1e-4_B32t_-0.301_75.49.pth')
    parser.add_argument('--config', type=str, default='./configs/%s'%(name+'_rkhs.yaml'))
    parser.add_argument('--save-mask-path', type=str, default='records/generated_images/%s' % (name)) #(name+'_'+mode+'_ablation'))
    parser.add_argument('--mode', type=str, default='val', help='val: evalutaion; run: generate result only')
    args = parser.parse_args()
    return args


def get_dataset(cfg, args):
    if cfg['dataset'] == 'pascal':
        valset = VocDataset(cfg['dataset'], cfg['data_root'], args.mode, None)

    # elif cfg['dataset'] == 'cityscapes':
    #     valset = CityDataset(cfg['dataset'], cfg['data_root'], 'val', None)

    elif cfg['dataset'] == 'ham':
        valset = HAMDataset(cfg['dataset'], cfg['data_root'], args.mode, None)

    elif cfg['dataset'] == 'acdc':
        valset = ACDCRkhsDataset(cfg['dataset'], cfg['data_root'], 'val', cfg['crop_size'], cfg['nclass'])

    elif cfg['dataset'] == 'ecssd':
        valset = ECSSDRkhsDataset(cfg['dataset'], cfg['data_root'], 'val', None, cfg['nclass'])

    elif cfg['dataset'] == 'plain':
        #valset = ACDCDataset('acdc', cfg['data_root'], 'val', None)
        valset = VocRkhsDataset(cfg['dataset'], cfg['data_root'], args.mode, None)

    else:
        valset = None

    return valset


def main(args):


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    if cfg['network'] == 'DeepLabV3Plus':
        from model.semseg.deeplabv3plus import DeepLabV3Plus
        model = DeepLabV3Plus(cfg, aux=cfg['aux']).to(device=device)
    else:
        from model.semseg.unet_model import UNet
        model = UNet(cfg).to(device=device)
    checkpoint = torch.load(args.resume_model, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    print('\nParams: %.1fM' % count_params(model))
    model = model.to(device=device)
    model.eval()

    if not os.path.exists(args.save_mask_path):
        os.makedirs(args.save_mask_path)

    dataset = get_dataset(cfg, args)
    valloader = DataLoader(dataset, batch_size=1,
                           shuffle=False, pin_memory=True, num_workers=8, drop_last=False)
    tbar = tqdm(valloader)

    metric = meanIOU(num_classes=cfg['nclass'])
    scores = []

    with torch.no_grad():
        for img, mask, prob, id in tbar:
            img = img.to(device=device)
            mask = mask.to(device=device)

            # seg = model(img)
            #
            # if cfg['nclass'] > 2:
            #     pred = seg.argmax(1)
            # else:
            #     pred = torch.sigmoid(seg.squeeze(1))
            #     pred[pred >= 0.5] = 1
            #     pred[pred < 0.5] = 0
            #     pred = pred.long()

            # pred = torch.zeros_like(prob)
            # pred[prob >= 0.5] = 1
            # pred[prob < 0.5] = 0
            # pred = pred.long()
            seg = None
            label_path = cfg['data_root']+'/prob_visual'
            pred = Image.open(os.path.join(label_path, id[0] + '_thre.png'))
            pred = torch.from_numpy(np.array(pred)).long().unsqueeze(0)



            if args.mode == 'val':
                metric.add_batch(pred.cpu().numpy(), mask.cpu().numpy())
                IOU, DICE, PA, mIOU, mDICE, mPA = metric.evaluate()
                # iu, dice, acc = eval_Area(pred.cpu().numpy(), mask.cpu().numpy(), cfg['nclass'])
                # pred_RE, gt_RE, pred_compact, gt_compact = eval_boundary(pred.cpu().numpy(), mask.cpu().numpy(),
                #                                                          cfg['nclass'])
                #
                # scores.append([IOU, DICE, PA])
                #
                # if save_record:
                #     print(iu, dice, acc)
                #     print('\n================\n')
                #     print(pred_RE, gt_RE)
                #     print(pred_compact, gt_compact)
                #     print(' ')
                # else:
                #     tbar.set_description('mIOU: %.2f, %.2f | Re: %.4f, %.4f' % (
                #         np.nanmean(scores, axis=0)[0][0] * 100.0,
                #         mIOU * 100.0,
                #         abs(pred_RE[0] - gt_RE[0]),
                #         abs(pred_compact[0] - gt_compact[0])))


            if save_img == 'image' and (args.mode == 'run' or args.mode == 'val'):
                pred = pred.squeeze().cpu().numpy().astype(np.uint8)
                pred = Image.fromarray(pred, mode='P')
                if cfg['dataset'] == 'acdc': #or cfg['dataset'] == 'plain':
                    pred.putpalette(color_map('acdc'))
                    pred.save('%s/%s_%s.png' % (args.save_mask_path, id[0], model_name))
                elif cfg['dataset'] == 'pascal':
                    pred.putpalette(color_map('pascal'))
                    pred.save('%s/%s' % (args.save_mask_path, os.path.basename(id[0].split(' ')[1])))
                elif cfg['dataset'] == 'ecssd':
                    pred.putpalette(color_map('plain'))
                    pred.save('%s/%s_%s.png' % (args.save_mask_path, id[0], model_name))
                else:
                    pred.putpalette(color_map('plain'))
                    pred.save('%s/%s_%s.png' % (args.save_mask_path, os.path.basename(id[0].split(' ')[1]).split('.')[0], model_name))
            elif save_img == 'prob' and args.mode == 'run':
                cm = plt.get_cmap('inferno')
                p = cm(seg) * 255
                im = Image.fromarray(p.astype(np.uint8))
                if cfg['dataset'] == 'pascal':
                    im.save('%s/%s_prob' % (args.save_mask_path, os.path.basename(id[0].split(' ')[1])))
                elif cfg['dataset'] == 'ham':
                    im.save('%s/%s_%s_prob.png' % (
                    args.save_mask_path, os.path.basename(id[0].split(' ')[1]).split('.')[0], model_name))
                else:  # or cfg['dataset'] == 'plain':
                    im.save('%s/%s_%s_prob.png' % (args.save_mask_path, id[0], model_name))

        if args.mode == 'val':
            print('IoU (class):', IOU, ' and mIOU:', mIOU)
            print('Dice (class):', DICE, ' and mDice:', mDICE)
            print('PA (class):', PA, ' and mPA:', mPA)


if __name__ == '__main__':
    args = parse_args()

    print()
    print(args)

    main(args)
