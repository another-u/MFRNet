import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pdb, os, argparse
from MFRNet import MFRNet
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=True, help='VGG or ResNet backbone')
opt = parser.parse_args()

#-------------------dataset root--------------------------------
image_root1 = './CoCOD8K/test/image/'
gt_root1 = './CoCOD8K/test/groundtruth/'

#-------------------model & path--------------------------------
model = MFRNet()
model.load_state_dict(torch.load('./MFRNet_Baseline_train_pth/Net_epoch_best.pth'))
model.cuda()
model.eval()

test_datasets=os.listdir(image_root1)

for dataset in test_datasets:
    print(dataset)

    image_root = image_root1+dataset+'/'
    gt_root = gt_root1+dataset+'/'

    save_path = './MFRNet_Baseline_train_pth/pred/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    sub_test_datasets = os.listdir(image_root)
    for sub_dataset in sub_test_datasets:
        # sub_gt_root = os.listdir(gt_root)
        image_root2 = image_root + sub_dataset + '/'
        gt_root2 = gt_root + sub_dataset + '/'
        test_loader = test_dataset(image_root2, gt_root2, opt.testsize)

        save_path1 = './MFRNet_Baseline_train_pth/pred/' + dataset + '/' + sub_dataset + '/'
        save_path_all = 'MFRNet_Baseline_train_pth/pred/all/all/'
        if not os.path.exists(save_path1):
            os.makedirs(save_path1)
        if not os.path.exists(save_path_all):
            os.makedirs(save_path_all)

        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res = model(image)
            res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    
            cv2.imwrite(save_path1 + name, res*255)
            cv2.imwrite(save_path_all + name, res * 255)

    
    




