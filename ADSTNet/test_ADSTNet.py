import torch
import torch.nn.functional as F
import imageio
import numpy as np
import os, argparse
import time
from skimage import img_as_ubyte
from model.ADSTNet import ADSTNet
from data import test_dataset

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
opt = parser.parse_args()

dataset_path = '/Datasets/SOD/test_dataset/'

model = ADSTNet(1)
model.load_state_dict(torch.load('./models/ADSTNet.pth'))

model.cuda()
model.eval()

test_datasets = ['ORSSD']
# test_datasets = ['EORSSD']

for dataset in test_datasets:
    save_path = './results/Your_Files/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/Images/'
    print(dataset)
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()

        res, s1, s2, s3, s5, s0_sig, s1_sig, s2_sig, s3_sig, s5_sig, eg1 = model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave(save_path+name, img_as_ubyte(res))
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size/time_sum))