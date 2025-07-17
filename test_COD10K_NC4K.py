import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import cv2
from MFRNet import MFRNet
from data_val import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--pth_path', type=str, default='./MFRNet_train_pth/Net_epoch_best.pth')
opt = parser.parse_args()

for _data_name in ['CAMO', 'COD10K', 'CHAMELEON' , 'NC4K']:
    data_path = './Dataset/TestDataset/{}'.format(_data_name)
    save_path = './MFRNet_train_pth/{}/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
    model = MFRNet().cuda()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res= model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> {} - {}'.format(_data_name, name))
        # misc.imsave(save_path+name, res)
        # If `mics` not works in your environment, please comment it and then use CV2
        cv2.imwrite(save_path+name,res*255)
