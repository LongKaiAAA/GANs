import numpy as np
import torch.nn as nn
import torch


"""生成器模型"""
class Generator(nn.Module):
    def __init__(self,input_dim,img_shape):
        """
        :param input_dim: 干扰数据的长度
        :param img_shape: 想要生成的图片的尺寸(与判别器输入的图像尺寸保持一致)
        """
        self.input_dim = input_dim
        self.img_shape = img_shape
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self,input):
        img = self.model(input)
        img = img.view(img.size(0), *(self.img_shape))
        return img


# 判别器模型
class Discriminator(nn.Module):
    def __init__(self,img_shape):
        self.img_shape = img_shape
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            )

    def forward(self, img):
        img_flat = img.view(-1, int(np.prod(self.img_shape)))
        validity = self.model(img_flat)

        return validity


if __name__ == "__main__":
    # 定义一些变量
    input_dim = 100
    img_size = [3, 256, 256]
    # 创建判别器
    d_model = Discriminator(img_shape= img_size)

    # 创建生成器
    g_model = Generator(input_dim=input_dim, img_shape=img_size)
