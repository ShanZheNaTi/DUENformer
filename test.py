import math

import numpy as np
from torch.autograd import Variable
import cv2
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import yaml
import os
import torch
import torchvision.transforms as transforms
from model1 import Generator as net
import cv2
import torch.nn
import time
import os

torch.nn.Module.dump_patches = True
nets = net()
nets.to('cuda')
check = torch.load(r'./checkpoint/checkpoint_0_epoch.pkl')
nets.load_state_dict(check['model'])
nets.eval()
img_dir = r'./test/input'

dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_tensor_type(torch.FloatTensor)


def compute_psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def compute_mse(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    return mse


# image test
def testimage():
    with torch.no_grad():
        for i in os.listdir(img_dir):
            img = cv2.imread(img_dir + '/' + i)
            img = cv2.resize(img, (256, 256))
            transform = transforms.ToTensor()
            imgs = transform(img).float()
            imgs = torch.unsqueeze(imgs, dim=0)
            imgs = torch.tensor(imgs, requires_grad=False, device='cuda')
            outimg = nets(imgs)
            outimg = outimg.clone().detach().requires_grad_(False)
            outimg = torch.squeeze(outimg)
            out = torch.transpose(outimg, 0, 1)
            out = torch.transpose(out, 1, 2).cpu().numpy() * 255
            # save_image(out, "./test/output/" + i, nrow=5, normalize=True)
            cv2.imwrite(r'./test/output/' + i.split('.')[0] + '.' + i.split('.')[-1], out)
            print(i)


def calculate():
    path1 = './test/GT/'  # 要改
    path2 = './test/output/'  # 要改
    path_list1 = os.listdir(path1)
    path_list1.sort(key=lambda x: int(x.split('.')[0]))
    PSNR = []

    for item in path_list1:
        impath1 = path1 + item
        impath2 = path2 + item
        imgx = cv2.imread(impath1)
        imgx = cv2.resize(imgx, (256, 256))
        imgy = cv2.imread(impath2)
        imgy = cv2.resize(imgy, (256, 256))
        # print(imgx.shape)
        psnr1 = compute_psnr(imgx[:, :, 0], imgy[:, :, 0])
        psnr2 = compute_psnr(imgx[:, :, 1], imgy[:, :, 1])
        psnr3 = compute_psnr(imgx[:, :, 2], imgy[:, :, 2])

        psnr = (psnr1 + psnr2 + psnr3) / 3.0

        PSNR.append(psnr)

    PSNR = np.array(PSNR)
    print(PSNR.mean())


if __name__ == '__main__':
    testimage()
    calculate()
