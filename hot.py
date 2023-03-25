import numpy
import torch
import matplotlib.pyplot as plt

def hot(pic, path):

    pic = pic.cpu().numpy()[0,0]

    fig,_ = plt.subplots()
    plt.imshow(pic)
    plt.tight_layout()
    plt.show()
    fig.savefig(path,dpi=100)

if __name__ == '__main__':

    pic = torch.randn(1,1,256,256)
    # print(pic)

    path = "./test.jpg"

    hot(pic, path)