import torch
import torch.nn as nn
import torch.utils as utils

from styler import Unet


def ep(net, input):
    li = []
    for i in range(5):
        out = net(input)
        li.append(out)
    neo = torch.cat(li, dim=0)
    avg = torch.sum(neo, keepdim=True, dim=0) / 5
    res = torch.sum((neo - avg) ** 2, keepdim=True, dim=0) / 5
    res = torch.sum(res, keepdim=True, dim=1)
    return res


def al(net, input):
    li = []
    for i in range(5):
        out = net(input)
        li.append(out)
    neo = torch.cat(li, dim=0)
    avg = torch.sum(neo, keepdim=True, dim=0) / 5
    res_ep = torch.sum((neo - avg) ** 2, keepdim=True, dim=0) / 5

    res = res_ep + torch.sum(neo ** 2, keepdim=True, dim=0) / 5 - avg ** 2
    res = torch.sum(res, keepdim=True, dim=1)
    return res


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Unet(device)

    x = torch.randn(1, 3, 128, 256).to(device)
    ep_v = ep(net, x)
    al_v = al(net, x)

    print(ep_v.shape)
    print(al_v.shape)

    target = torch.randn(1, 128, 256).to(device).long()
    out = net(x)

    loss_func = nn.CrossEntropyLoss()
    loss = loss_func(out, target)
    print(loss.item())
