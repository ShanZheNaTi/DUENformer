import argparse
from my_dataset import MyDataSet as data
from model1 import Generator as net
import cv2
import os
import random
import numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument('-b', '--batch', help='the number of batch', type=int, default='1')
ap.add_argument('-e', '--epoch', help='the number of training', type=int, default='500')
ap.add_argument('-r', '--resume', help='the choice of resume', type=bool, default=False)

args = vars(ap.parse_args())


def log_images(writer, img, out, ll256, gt, it):
    images_array = vutils.make_grid(img).to('cpu')
    # 该函数用于将一批图像拼接成一个网格，以便于可视化。
    # vutils.make_grid(img)调用了 make_grid 函数，将输入的图像张量拼接成一个网格。返回的结果是一个包含所有图像的单一张图像
    out_array = vutils.make_grid(out * 255).to('cpu').detach()
    # .detach() 用于创建一个新的张量，其值与原始张量相同，但是与计算图不再有关系。
    # 这样做的目的通常是为了防止梯度传播到这个张量，因为在可视化或其他后续处理步骤中，我们可能不需要梯度信息。
    ll256_array = vutils.make_grid(ll256 * 255).to('cpu').detach()
    gt = vutils.make_grid(gt).to('cpu')

    writer.add_image('input', images_array, it)
    # writer 是一个 SummaryWriter 对象，它是 PyTorch 中 TensorBoard 的接口，用于记录训练过程中的各种信息，例如损失、参数直方图、图像等。
    # SummaryWriter 允许你将这些信息写入 TensorBoard 日志目录中，以便在 TensorBoard 中进行可视化。
    writer.add_image('out', out_array, it)
    writer.add_image('ll256', ll256_array, it)
    writer.add_image('gt', gt, it)
    # it 是一个表示迭代或批次数的整数，用于标识当前的训练步骤。
    # 这个参数可以帮助你在 TensorBoard 中查看图像的变化随着训练的进行而发生的情况。


net = net()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args['resume']:
    files = './checkpoint/checkpoint_0_epoch.pkl'
    assert os.path.isfile(files), "{} is not a file.".format(args['resume'])
    state = torch.load(files)
    net.load_state_dict(state['model'])
    iteration = state['epoch'] + 1
    optimizer = state['optimizer']
    print("Checkpoint is loaded at {} | epochs: {}".format(args['resume'], iteration))
else:
    iteration = 0

# da = data('/root/ThirtyoneDemo/data')
da = data('./data')
dataloder = DataLoader(da, batch_size=args['batch'], shuffle=True)

optimizer = torch.optim.Adam(lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, params=net.parameters())
scheduler = StepLR(optimizer, step_size=150, gamma=0.5)


def proimage(im):
    images = im[image_idx, :, :, :].clone().detach().requires_grad_(False)
    # 该行代码使用索引image_idx从输入的im中选择一个特定的图像。
    # clone()方法创建了图像的副本
    # detach()方法用于分离图像的计算图
    # requires_grad_(False)用于确保图像不需要梯度计算
    # 这一系列的操作表明该代码可能是为了处理单个图像，而不是整个批次。
    image = torch.transpose(images, 0, 1)
    # 这一行代码使用PyTorch的torch.transpose函数交换了张量images的维度0和1。
    image = torch.transpose(image, 1, 2).cpu().numpy() * 255
    # 该行代码首先将PyTorch张量转换为NumPy数组，然后将数值范围从0到1映射到0到255。
    # 这通常是因为图像处理中，像素值常常以0到255的整数形式表示。
    return image


writer = SummaryWriter(comment='test_comment', filename_suffix="test_suffix")
j = 0
h = 0
for iter in range(iteration, args['epoch']):
    print(iter)
    prob = tqdm(enumerate(dataloder), total=len(dataloder))
    if iter < 1000:
        L1 = nn.MSELoss()
    else:
        L1 = nn.L1Loss()
    for i, data in prob:
        gt = torch.tensor(data[0].numpy(), requires_grad=True, device='cuda')
        raw = torch.tensor(data[1].numpy(), requires_grad=True, device='cuda')
        net.to('cuda')
        a = net(raw)
        L1loss = L1(a, gt)
        loss = L1loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prob.set_postfix(Loss1=L1loss)
        # set_postfix: 这是一个用于设置额外信息的方法，通常用于在训练循环中显示额外的信息，例如损失值、准确度等。
        # 综合来说，这段代码的目的是在训练过程中以 Loss1 为标签，将 L1loss 的值添加到训练过程中的输出信息中。
        # 这对于实时监控模型的训练进度和性能是很有帮助的。
        h = h + 1
        writer.add_scalar('loss', loss, h)

        c = a
        if i % 100 == 0:
            j += 100
            image_idx = random.randint(0, 0)
            predi = c[image_idx, :, :, :].clone().detach().requires_grad_(False)
            predi = torch.transpose(predi, 0, 1)
            predi = torch.transpose(predi, 1, 2).cpu().numpy() * 255
            gti = proimage(gt)
            rawi = proimage(raw)
            image = np.concatenate((rawi, predi, gti), axis=1)
            image_name = 'sample12' + "/out" + str(iter) + '_' + str(i) + ".png"
            cv2.imwrite(image_name, image)
            # 使用 OpenCV 的 imwrite 函数将图像数据 image 保存到文件系统中，文件名为 image_name。

    if (iter + 1) % 1 == 0:
        checkpoint = {"model": net.state_dict(),
                      "optimizer": optimizer,
                      "epoch": iter}

        if not os.path.exists('checkpoint'):
            # 检查当前工作目录下是否存在一个名为 'checkpoint' 的目录。
            os.mkdir('checkpoint')
            # os.mkdir('checkpoint') 创建一个名为 'checkpoint' 的目录。这个目录用于存储检查点文件。
        path_checkpoint = "checkpoint/checkpoint_{}_epoch.pkl".format(iter)
        # 检查点文件的路径
        torch.save(checkpoint, path_checkpoint)
        # 将检查点数据 checkpoint 保存到文件 path_checkpoint 中。
    # 综合起来，这段代码的作用是检查是否存在名为 'checkpoint' 的目录，如果不存在则创建它，
    # 然后保存当前训练状态的检查点文件到该目录中。每个检查点文件的名称包含了当前迭代的 epoch 数。

    scheduler.step()
    print(optimizer.param_groups[0]['lr'])
