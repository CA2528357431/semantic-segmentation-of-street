import torch
from torch.utils.data import DataLoader
import os
from torch import nn
import numpy as np
from DataLoader import DataSet
import h5py
from DataLoader import color_show
import matplotlib.pyplot as plt
from PIL import Image
# from styler import Unet
from uncertain import ep
from uncertain import al
from hot import hot


import torch
import torch.nn as nn
import torchvision
import numpy as np


class UpBlock(nn.Module):
    def __init__(self, in_channel=128, out_channel=64):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channel + in_channel, out_channel,
                      3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel,
                      3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.skip = nn.Conv2d(out_channel + in_channel, out_channel, 1, 1, 0)

    def forward(self, input, FB_in):
        out_temp = self.upsample(input)
        out_temp = torch.cat([out_temp, FB_in], dim=1)
        out = self.conv(out_temp) + self.skip(out_temp)

        return out


class DownBlock(nn.Module):
    def __init__(self, in_channel=3, out_channel=64):
        super().__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU(),
                                  nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                                  nn.BatchNorm2d(out_channel),
                                  nn.ReLU()
                                  )
        self.skip = nn.Conv2d(in_channel, out_channel, 1, 1, 0)
        # self.downsample = nn.MaxPool2d(2,2)
        self.downsample = nn.Conv2d(out_channel, out_channel, 4, 2, 1)

    def forward(self, input):
        out_temp = self.conv(input) + self.skip(input)
        out = self.downsample(out_temp)
        return out, out_temp


class EncodingBlock(nn.Module):
    def __init__(self, in_channel=256, out_channel=512):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      3, 1, 1),
            # nn.BatchNorm2d(out_channel),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel,
                      3, 1, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        self.skip = nn.Conv2d(in_channel, out_channel, 1, 1, 0)

    def forward(self, input):
        out = self.conv(input) + self.skip(input)
        return out


class Unet(nn.Module):
    def __init__(self, ngf=16, input_channel=3, output_channel=34):
        super().__init__()
        self.conv_init = nn.Conv2d(input_channel, ngf, 1, 1, 0)
        self.init = EncodingBlock(ngf, ngf)
        self.down1 = DownBlock(ngf, ngf)
        self.down2 = DownBlock(ngf, 2 * ngf)
        self.down3 = DownBlock(2 * ngf, 4 * ngf)

        self.encoding = EncodingBlock(4 * ngf, 8 * ngf)
        self.up3 = UpBlock(8 * ngf, 4 * ngf)
        self.up2 = UpBlock(4 * ngf, 2 * ngf)
        self.up1 = UpBlock(2 * ngf, ngf)
        self.out = EncodingBlock(2 * ngf, ngf)
        self.conv_fin = nn.Conv2d(ngf, output_channel, 1, 1, 0)
        self.tan = nn.Tanh()

        self.drop0 = nn.Dropout(0.3)
        self.drop1 = nn.Dropout(0.3)
        self.drop2 = nn.Dropout(0.3)
        self.drop3 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv_init(x)
        x = self.init(x)
        d1, d1_f = self.down1(x)
        d2, d2_f = self.down2(d1)
        d3, d3_f = self.down3(d2)

        h = self.encoding(d3)
        hu3 = self.up3(self.drop3(h), d3_f)
        hu2 = self.up2(self.drop2(hu3), d2_f)
        hu1 = self.up1(self.drop1(hu2), d1_f)

        h = self.out(torch.cat([self.drop0(hu1), x], dim=1))
        h = self.conv_fin(h)
        h = self.tan(h)
        return h


BATCH_SIZE = 1
EPOCH = 200
LR = 0.0001

current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = current_dir + '/data/driving_test_data.h5'
test_dataset = DataSet(path=test_dir,
                       split='test',
                       overwrite=False,
                       transform=False
                       )

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)

cuda = torch.device("cuda")
cpu = torch.device('cpu')

model = torch.load('best.pth')

# loss_func = nn.CrossEntropyLoss()  # BATCH_SIZE个loss的平均
#
# total_test_step = 0
# total_test_loss = 0
# test_correct_pred = 0
# test_total_pred = 0
# total_LE = 0
# total_LC = 0
# test_loss_each = torch.zeros(3)  # 某类的loss
# test_correct_each = torch.zeros(3)  # 某类的正确预测次数
# test_total_each = torch.zeros(3)  # 某类的总预测次数
#
# with torch.no_grad():
#     for data in test_dataloader:
#         inputs, targets = data
#         loss_func = loss_func.to(cuda)
#         inputs = inputs.to(cuda)
#         targets = targets.to(cuda)
#         outputs = model(inputs)
#         loss = loss_func(outputs, targets)
#         total_test_loss = total_test_loss + loss
#         total_test_step = total_test_step + 1
#
# print("整体测试集上的平均loss:{}".format(total_test_loss / total_test_step))


torch.cuda.empty_cache()

h5f = h5py.File(test_dir, 'r')
color_codes = np.array(h5f['color_codes'])
save_dir = current_dir + '/results'
model.to(cpu)
i = 0
with torch.no_grad():
    for data in test_dataloader:
        i += 1
        inputs, _ = data
        outputs = model(inputs)
        epv = ep(model, inputs.view(1, 3, 128, 256))
        hot(epv, save_dir + '/' + str(i) + 'ep.jpg')
        alv = al(model, inputs.view(1, 3, 128, 256))
        hot(alv, save_dir + '/' + str(i) + 'al.jpg')

        rgb_outputs = color_show(outputs, color_codes)
        inputs = np.array(inputs.view(128, 256, 3)).astype(np.uint8)

        # print(epv.shape)
        # print(rgb_outputs.shape)
        inputs = Image.fromarray(np.array(inputs))
        outputs = Image.fromarray(np.array(rgb_outputs))

        plt.figure()
        plt.imshow(inputs)
        plt.show()
        plt.imsave(save_dir + '/' + str(i) + 'inputs.jpg', np.array(inputs))

        plt.figure()
        plt.imshow(outputs)
        plt.show()
        plt.imsave(save_dir + '/' + str(i) + 'outputs.jpg', np.array(outputs))
        #
        if i == 5:
            break
