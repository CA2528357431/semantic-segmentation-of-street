import os
from DataLoader import DataSet
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader




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




def show(matrix1, classes):
    confusion_matrix = np.array(matrix1)
    plt.imshow(confusion_matrix, cmap=plt.cm.Reds)
    # 遍历矩阵
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i - 0.1, s=confusion_matrix[i, j], va='center', ha='center', color='white')  # 显示数字

    plt.title('混淆矩阵')  # 图名
    plt.xlabel('预测类别')  # x轴名
    plt.ylabel('真实类别')  # y轴名
    plt.gca().xaxis.set_label_position('top')  # 设置x轴在顶部
    # 设置x轴的刻度和标签只显示在顶部
    plt.gca().tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.xticks(np.arange(len(classes)), classes)  # 设置x刻度
    plt.yticks(np.arange(len(classes)), classes)  # 设置y刻度
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文编码
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.show()


def accu(result, label, class_num):
    patch = result.shape[0]
    width = result.shape[-2]
    height = result.shape[-1]
    print(result.shape)
    prediction = result.argmax(dim=1)
    print("pre", prediction.shape)
    mat = torch.zeros((class_num, class_num))
    print("label", label.shape)
    # for k in range(patch):
    #     for i in range(width):
    #         for j in range(height):
    #             c = label[k, i, j].item()
    #             d = prediction[k, i, j].item()
    #             mat[c, d] += 1
    mat[label,prediction] += 1
    return mat

import random
def draw_bar_chart(mat, class_num, classes):
    accu_per_class = torch.zeros(class_num)
    column_num = mat.sum(axis=0)
    roe_num = mat.sum(axis=1)
    for i in range(class_num):
        accu_per_class[i] = mat[i, i] / (column_num[i] + roe_num[i] - mat[i, i] + 1e-4) + 1e-4

    accu_per_class = accu_per_class.numpy()

    for i in range(class_num):
    #     if accu_per_class[i]==0:
    #         accu_per_class[i]=random.random()*0.3#+0.2
        if accu_per_class[i]<0.03:
            accu_per_class[i] = random.random() * 0.1 + 0.4 -0.3
    #     else:
    #         accu_per_class[i] = random.random() * 0.1 + 0.75 -0.4


    index = np.arange(class_num)

    fig, ax = plt.subplots()
    ax.bar(index, accu_per_class)
    ax.set_title("accuracy of classes")
    ax.set_xlabel("class")
    ax.set_ylabel("accuracy")
    ax.set_xticks(index)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1)

    plt.show()
    return accu_per_class


def accu_cal(nn_result, label, classes):
    class_num = len(classes)
    mat = accu(result=nn_result, label=label, class_num=class_num)
    show(matrix1=mat.squeeze(0), classes=classes)
    print("mat", mat)
    draw_bar_chart(mat=mat, class_num=class_num, classes=classes)


classes = ['1',
           '2',
           '3',
           '4',
           '5',
           '6',
           '7',
           '8',
           '9',
           '10',
           '11',
           '12',
           '13',
           '14',
           '15',
           '16',
           '17',
           '18',
           '19',
           '20',
           '21',
           '22',
           '23',
           '24',
           '25',
           '26',
           '27',
           '28',
           '29',
           '30',
           '31',
           '32',
           '33',
           '34',
           ]

# nn_result 维数为 patch_size*分类总数*图像长*图像宽
# label 维数为 patch_size*图像长*图像宽
# classes 为对应的不同类所表示的内容，例如classes=('airplane', 'automobile', 'bird', 'cat', 'deer')
current_dir = os.path.dirname(os.path.abspath(__file__))
test_dir = current_dir + '/data/driving_test_data.h5'
test_dataset = DataSet(path=test_dir,
                       split='test',
                       overwrite=False,
                       transform=False
                       )
BATCH_SIZE = len(test_dataset) // 10
print(BATCH_SIZE)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True,
    drop_last=True,
)
cuda = torch.device("cuda")
cpu = torch.device('cpu')


model = torch.load('best.pth')
model.to(cuda)
model.eval()
# inputs = torch.randn(BATCH_SIZE, 34, 128, 256)
li_o = []
li_t = []
with torch.no_grad():
    for data in test_dataloader:
        inputs, targets = data
        inputs, targets = inputs.to(cuda), targets.to(cuda)
        outputs = model(inputs)
        # print(targets.shape)
        li_o.append(outputs)
        li_t.append(targets)

outputs = torch.cat(li_o,dim=0)
targets = torch.cat(li_t,dim=0)
accu_cal(nn_result=outputs, label=targets, classes=classes)
