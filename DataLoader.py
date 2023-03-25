from torchvision import transforms as tfs
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import random
import scipy.io as scio


def color_decode(seg):
    rgb = np.zeros([len(seg), len(seg[0]), 3])
    for i in range(len(seg)):
        for j in range(len(seg[0])):
            rgb[i, j] = seg[i, j], seg[i, j], seg[i, j]
    rgb = rgb.astype(np.uint8)
    return rgb


def color_show(seg, color_codes):
    seg = seg.argmax(dim=1)

    rgb = np.zeros([seg.shape[1], seg.shape[2], 3])
    for i in range(seg.shape[1]):
        for j in range(seg.shape[2]):
            # print(seg[:, i, j])
            rgb[i, j] = color_codes[seg[:, i, j]]
    rgb = rgb.astype(np.uint8)
    return rgb


def color_encode(rgb):
    rgb = np.array(rgb)
    seg = np.zeros([1, rgb.shape[1], rgb.shape[2]])
    for i in range(rgb.shape[1]):
        for j in range(rgb.shape[2]):
            seg[:, i, j] = rgb[0, i, j]
    seg = seg.astype(np.uint8)
    return seg


class DataSet(Dataset):
    def __init__(self, path, split, overwrite, transform):
        h5f = h5py.File(path, 'r')
        color_codes = np.array(h5f['color_codes'])
        # print(color_codes.shape)
        rgb = np.array(h5f['rgb'])
        # plt.figure()
        # plt.imshow(Image.fromarray(rgb[0]))
        # print(rgb[0].shape)
        # plt.show()
        seg = np.array(h5f['seg'])
        # print(seg.max())
        self.len = len(rgb)
        self.transform = transform
        self.trans = tfs.Compose([
            tfs.RandomHorizontalFlip(p=2.0),
            tfs.RandomVerticalFlip(p=2.0),
            tfs.ToTensor(),
        ])
        self.split = split
        if self.split != 'test' and transform:
            self.rgb_container = [None] * self.len * 2
            self.seg_container = [None] * self.len * 2
            self.feature_path = [''] * self.len * 2
            self.label_path = [''] * self.len * 2
            for i in range(self.len):
                self.feature_path[2 * i] = 'feature/' + self.split + '/' + str(2 * i) + '.mat'
                self.feature_path[2 * i + 1] = 'feature/' + self.split + '/' + str(2 * i + 1) + '.mat'
                # self.feature_path[4 * i + 2] = 'feature/' + self.split + '/' + str(4 * i + 2) + '.mat'
                # self.feature_path[4 * i + 3] = 'feature/' + self.split + '/' + str(4 * i + 3) + '.mat'
                self.label_path[2 * i] = 'label/' + self.split + '/' + str(2 * i) + '.mat'
                self.label_path[2 * i + 1] = 'label/' + self.split + '/' + str(2 * i + 1) + '.mat'
                # self.label_path[4 * i + 2] = 'label/' + self.split + '/' + str(4 * i + 2) + '.mat'
                # self.label_path[4 * i + 3] = 'label/' + self.split + '/' + str(4 * i + 3) + '.mat'
        else:
            self.rgb_container = [None] * self.len
            self.seg_container = [None] * self.len
            self.feature_path = [''] * self.len
            self.label_path = [''] * self.len
            for i in range(self.len):
                self.feature_path[i] = 'feature/' + self.split + '/' + str(i) + '.mat'
                self.label_path[i] = 'label/' + self.split + '/' + str(i) + '.mat'

        if overwrite:
            print(self.split)
            print("重新提取特征")
            if self.split != 'test' and transform:
                for i in range(self.len):
                    self.rgb_container[2 * i] = np.array(torch.from_numpy(rgb[i]).view(3, rgb[i].shape[0], rgb[i].shape[1]))
                    self.seg_container[2 * i] = np.array(torch.from_numpy(seg[i]).view(1, seg[i].shape[0], seg[i].shape[1]))

                    # seed = np.random.randint(0, 2 ** 16)
                    # random.seed(seed)
                    # torch.manual_seed(seed)
                    self.rgb_container[2 * i + 1] = np.array(self.trans(Image.fromarray(rgb[i])).view(3, rgb[i].shape[0], rgb[i].shape[1]))
                    # random.seed(seed)
                    # torch.manual_seed(seed)
                    decode_seg = color_decode(seg[i])
                    # print(decode_seg.shape)
                    self.seg_container[2 * i + 1] = color_encode(np.array(self.trans(Image.fromarray(decode_seg)).view(3, seg[i].shape[0], seg[i].shape[1])))

                    scio.savemat(self.feature_path[2 * i], {'feature': self.rgb_container[2 * i]})
                    scio.savemat(self.label_path[2 * i], {'label': self.seg_container[2 * i]})

                    scio.savemat(self.feature_path[2 * i + 1], {'feature': self.rgb_container[2 * i + 1]})
                    scio.savemat(self.label_path[2 * i + 1], {'label': self.seg_container[2 * i + 1]})

                    if i % 500 == 0:
                        print("已提取个数：{}".format(i * 2))
                    # print(self.seg_container[i].shape)
                    # print(self.seg_container[2 * i].shape)
                    # print(self.seg_container[2 * i + 1].shape)
                    # plt.figure()
                    # plt.imshow(Image.fromarray(self.seg_container[i]))
                    # plt.show()
            else:
                # print('ok')
                for i in range(self.len):
                    self.rgb_container[i] = np.array(torch.from_numpy(rgb[i]).view(3, rgb[i].shape[0], rgb[i].shape[1]))#np.reshape(rgb[i], [3, rgb[i].shape[0], rgb[i].shape[1]])
                    self.seg_container[i] = np.array(torch.from_numpy(seg[i]).view(1, seg[i].shape[0], seg[i].shape[1]))#np.reshape(seg[i], [1, seg[i].shape[0], seg[i].shape[1]])
                    # print(self.seg_container[i].shape)
                    scio.savemat(self.feature_path[i], {'feature': self.rgb_container[i]})
                    scio.savemat(self.label_path[i], {'label': self.seg_container[i]})

        else:
            print(self.split)
            print("不重新提取特征")
            if self.split != 'test' and transform:
                length = self.len * 2
            else:
                length = self.len
            for i in range(length):
                datar = scio.loadmat(self.feature_path[i])
                datas = scio.loadmat(self.label_path[i])
                r = datar['feature']
                s = datas['label']
                r, s = r.astype(np.float32), s.astype(np.int64)

                # print(s.shape)

                r = torch.from_numpy(r)
                s = torch.from_numpy(s)
                # r, s = r.T, s.T
                s = torch.LongTensor(s.view(s.shape[1], s.shape[2]))
                # print(r.shape)
                # print(s.shape)
                # print(s.min())
                # print(s.max())
                # print(s)
                self.rgb_container[i] = r
                self.seg_container[i] = s
                if i % 1000 == 0:
                    print("已读取个数：{}".format(i))

    def __getitem__(self, index):
        return (
            self.rgb_container[index],
            self.seg_container[index]
        )

    def __len__(self):
        if self.split != 'test' and self.transform:
            length = self.len * 2
        else:
            length = self.len
        return length
